"""Evaluate a bounce policy in the training env (cameras + filter, randomized servos, cameras and ball physics).

Runs every world for `--seconds` (consecutive 20 s episodes) and reports:

- per finished episode: bounces (only those sending the ball >= 5 cm up), the share that lasted the full 20 s and
  why the others ended, also split by the ball-blade restitution drawn for the episode;
- per bounce: outgoing apex vs the target, predicted landing error, contact offset from the blade centre and (with
  the scripted bouncer running) impact time relative to the bouncer's plan;
- per control step: the goal's departure from the base (scripted goal, or ready for the direct task), action rate,
  goals clipped at a joint limit, encoder speeds vs the fitted firmware speed limit;
- ball tracking.

`--agent zero` on the residual task is the scripted bouncer alone (the policy's starting point).

Run (from sim/):
  uv run scripts/eval_task.py --agent zero                     # scripted bouncer baseline
  uv run scripts/eval_task.py --agent random
  uv run scripts/eval_task.py --checkpoint logs/rsl_rl/pingpong_residual/<run>/model_1000.pt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

import pingpong_sim.task  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pingpong_sim.robot import ARM_JOINTS
from pingpong_sim.servo_actuator import FIT_PARAMS, encoder_vel
from pingpong_sim.task.state import get

# Ball-blade damping ratio bins (randomize_ball_contact) and their restitution at 1 ms contact (contact_sweep.py).
DAMP_BINS = ((0.155, 0.182, "e 0.65-0.61"), (0.182, 0.208, "e 0.61-0.56"), (0.208, 0.235, "e 0.56-0.51"))


def pct(x: np.ndarray, scale: float = 1.0, fmt: str = ".0f") -> str:
    """p50 / p90 / p99 of x * scale."""
    return " / ".join(f"{scale * np.percentile(x, q):{fmt}}" for q in (50, 90, 99))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PingPong-Bounce-Residual")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--agent", choices=["zero", "random"], default="zero")
    ap.add_argument("--checkpoint", help="rsl_rl checkpoint (overrides --agent)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.envs
    env_cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    venv = RslRlVecEnvWrapper(env)
    n, A = args.envs, env.action_space.shape[-1]
    if args.checkpoint:
        runner = MjlabOnPolicyRunner(venv, asdict(load_rl_cfg(args.task)), device=device)
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        label = args.checkpoint
    else:
        policy = (lambda obs: torch.zeros(n, A, device=device)) if args.agent == "zero" else \
            (lambda obs: 2 * torch.rand(n, A, device=device) - 1)  # noqa: E731
        label = f"{args.agent} action"

    st = get(env)
    arm = env.action_manager.get_term("arm")
    pid = env.sim.mj_model.pair("ball_blade").id
    term_names = [t for t in env.termination_manager.active_terms if t != "time_out"]
    count = torch.zeros(n, device=device)
    episodes = []  # (bounces, timed_out, cause index or -1, ball-blade damping ratio)
    err, trk = [], []
    hits = []  # per good bounce: apex, landing error, lateral offset, impact time - planned impact time
    dev_goal, rate, clipped, speed = [], [], [], []  # per control step, released worlds
    prev_a = torch.zeros(n, A, device=device)
    obs = venv.get_observations()
    damp = env.sim.model.pair_solref[:, pid, 1].clone()
    with torch.inference_mode():
        for _ in range(int(args.seconds / env.step_dt)):
            a = policy(obs)
            # Read before the step: during it the bouncer may replan for the next impact and moves its goal on.
            t_plan = st.ctrl.bouncer.t_hit.clone()
            base = st.ctrl.goal_base().clone()  # what this step's action is added to
            obs, _, dones, _ = venv.step(a)
            s = st.sum
            count += s["good"].float()
            err.append(st.track_err[st.track_err > 0])
            trk.append(st.ctrl.tracked.float().mean())
            g = s["good"]
            if g.any():
                late = (st.now - t_plan) if st.cfg.use_bouncer else torch.full_like(st.track_err, torch.nan)
                hits.append(torch.stack([s["apex"][g], s["landing_err"][g], s["lateral"][g], late[g]], 1))
            done = dones.bool()
            live = s["released"] & ~done
            dev_goal.append((arm._goal - base)[live])
            rate.append((a - prev_a)[live].norm(dim=-1))
            clipped.append(((arm._goal <= st.kin.lo) | (arm._goal >= st.kin.hi))[live].float())
            speed.append(encoder_vel(st.robot)[live].abs())
            prev_a = torch.where(done[:, None], torch.zeros_like(a), a)
            if done.any():
                idx = done.nonzero(as_tuple=True)[0]
                tout = env.termination_manager.time_outs[idx]
                causes = torch.stack([env.termination_manager.get_term(t)[idx] for t in term_names], 1)
                cause = torch.where(causes.any(1), causes.float().argmax(1), torch.full_like(idx, -1))
                # The reset already drew the next episode's damping: the ended one's is gone. Use the step before.
                episodes.append(torch.stack([count[idx], tout.float(), cause.float(), damp[idx]], 1))
                count[idx] = 0
            damp = env.sim.model.pair_solref[:, pid, 1].clone()

    ep = torch.cat(episodes).cpu().numpy() if episodes else np.zeros((0, 4))
    e = torch.cat(err).cpu().numpy()
    print(f"{args.task}, {label}: {n} worlds x {args.seconds:.0f} s, {len(ep)} finished episodes, "
          f"control step {1e3 * env.step_dt:.0f} ms")
    if len(ep):
        b = ep[:, 0]
        print(f"  bounces per episode p10/p50/p90: {np.percentile(b, 10):.0f} / {np.percentile(b, 50):.0f} / "
              f"{np.percentile(b, 90):.0f}  (mean {b.mean():.1f}); full 20 s: {100 * ep[:, 1].mean():.1f} %")
        ended = ep[ep[:, 2] >= 0]
        if len(ended):
            print("  ended by: " + ", ".join(f"{t} {100 * np.mean(ended[:, 2] == i):.0f} %"
                                             for i, t in enumerate(term_names)))
        parts = []
        for lo, hi, name in DAMP_BINS:
            m = (ep[:, 3] >= lo) & (ep[:, 3] < hi)
            if m.any():
                parts.append(f"{name}: {100 * ep[m, 1].mean():.0f} % ({m.sum()} ep)")
        print("  full 20 s by restitution: " + ", ".join(parts))
    if hits:
        h = torch.cat(hits).cpu().numpy()
        print(f"  per bounce ({len(h)}): apex {1e3 * h[:, 0].mean():.0f} +- {1e3 * h[:, 0].std():.0f} mm "
              f"(target {1e3 * st.cfg.apex:.0f}); landing error p50/p90/p99 {pct(h[:, 1], 1e3)} mm; "
              f"contact off centre p50/p90/p99 {pct(h[:, 2], 1e3)} mm")
        if st.cfg.use_bouncer:
            late = h[:, 3][np.abs(h[:, 3]) < 0.15]  # else no live plan for this impact
            print(f"  impact time - scripted plan: {1e3 * late.mean():.0f} +- {1e3 * late.std():.0f} ms over "
                  f"{100 * len(late) / len(h):.0f} % of bounces (rest unplanned; detected on the first control "
                  f"step after contact)")
    d = torch.cat(dev_goal).cpu().numpy()
    what = "scripted goal" if st.cfg.use_bouncer else "ready pose"
    print(f"  goal - {what}, |deg| p50/p90/p99 per joint:")
    for j, name in enumerate(ARM_JOINTS):
        print(f"    {name:14s} {pct(np.abs(d[:, j]), 180 / np.pi, '.1f')}")
    r = torch.cat(rate).cpu().numpy()
    c = torch.cat(clipped).cpu().numpy()
    v = torch.cat(speed).cpu().numpy()
    vmax = json.loads(FIT_PARAMS.read_text())["max_velocity"]
    print(f"  action change per step |da| p50/p90/p99: {pct(r, 1, '.2f')}; goal at a joint limit: "
          + ", ".join(f"{name} {100 * c[:, j].mean():.1f} %" for j, name in enumerate(ARM_JOINTS)))
    print(f"  encoder speed p99 (rad/s, fitted firmware limit {vmax:.1f}): "
          + ", ".join(f"{name} {np.percentile(v[:, j], 99):.1f}" for j, name in enumerate(ARM_JOINTS)))
    print(f"  ball tracked {100 * torch.stack(trk).mean():.1f} % of steps; tracking error p50/p90 "
          f"{1e3 * np.percentile(e, 50):.1f} / {1e3 * np.percentile(e, 90):.1f} mm")


if __name__ == "__main__":
    main()
