"""Evaluate a bounce policy in the training env (cameras + filter, randomized servos and cameras).

Runs every world for `--seconds` (consecutive 20 s episodes) and reports per finished episode: bounces (only those
sending the ball >= 5 cm up), the share that lasted the full 20 s, why the others ended, and the ball tracking
error. `--agent zero` on the residual task is the scripted bouncer alone (the policy's starting point).

Run (from sim/):
  uv run scripts/eval_task.py --agent zero                     # scripted bouncer baseline
  uv run scripts/eval_task.py --agent random
  uv run scripts/eval_task.py --checkpoint logs/rsl_rl/pingpong_residual/<run>/model_1000.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import torch

import pingpong_sim.task  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pingpong_sim.task.state import get


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
    term_names = [t for t in env.termination_manager.active_terms if t != "time_out"]
    count = torch.zeros(n, device=device)
    episodes = []  # (bounces, timed_out, cause index or -1)
    err, trk = [], []
    obs = venv.get_observations()
    with torch.inference_mode():
        for _ in range(int(args.seconds / env.step_dt)):
            obs, _, dones, _ = venv.step(policy(obs))
            count += st.sum["good"].float()
            err.append(st.track_err[st.track_err > 0])
            trk.append(st.tracked.float().mean())
            done = dones.bool()
            if done.any():
                idx = done.nonzero(as_tuple=True)[0]
                tout = env.termination_manager.time_outs[idx]
                causes = torch.stack([env.termination_manager.get_term(t)[idx] for t in term_names], 1)
                cause = torch.where(causes.any(1), causes.float().argmax(1), torch.full_like(idx, -1))
                episodes.append(torch.stack([count[idx], tout.float(), cause.float()], 1))
                count[idx] = 0
    ep = torch.cat(episodes).cpu().numpy() if episodes else np.zeros((0, 3))
    e = torch.cat(err).cpu().numpy()
    print(f"{args.task}, {label}: {n} worlds x {args.seconds:.0f} s, {len(ep)} finished episodes")
    if len(ep):
        b = ep[:, 0]
        print(f"  bounces per episode p10/p50/p90: {np.percentile(b, 10):.0f} / {np.percentile(b, 50):.0f} / "
              f"{np.percentile(b, 90):.0f}  (mean {b.mean():.1f}); full 20 s: {100 * ep[:, 1].mean():.1f} %")
        ended = ep[ep[:, 2] >= 0]
        if len(ended):
            print("  ended by: " + ", ".join(f"{t} {100 * np.mean(ended[:, 2] == i):.0f} %"
                                             for i, t in enumerate(term_names)))
    print(f"  ball tracked {100 * torch.stack(trk).mean():.1f} % of steps; tracking error p50/p90 "
          f"{1e3 * np.percentile(e, 50):.1f} / {1e3 * np.percentile(e, 90):.1f} mm")


if __name__ == "__main__":
    main()
