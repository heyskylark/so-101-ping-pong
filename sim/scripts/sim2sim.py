"""Sim-to-sim check of an exported policy: plain CPU MuJoCo, driven through the deployment code path.

Training and eval_task.py: MuJoCo Warp via mjlab, observations assembled by mjlab's managers, PyTorch actor.
Here, one world per process (`real.sim_backend.CpuWorld`):
- physics: MuJoCo C (`mujoco.mj_step`) on the same compiled scene and solver options;
- servos: the same `ServoDrive` (servo model, command delay, torque disturbance), parameters drawn per episode as
  in training;
- sensing and control: the simulated cameras feeding `BounceController`, and `OnnxPolicy` (policy.json layout,
  onnxruntime) - the code the robot runs;
- episodes: the same ball launch, restitution draw, ball pushes, terminations and bounce counting. Masses stay
  nominal (training draws ball +-4.5 % and paddle +-2.5 % per world).

A large gap to eval_task.py on the same policy means it leans on something MuJoCo Warp or the mjlab glue does
differently (contacts, integration, observation assembly).

Run (from sim/):
  uv run scripts/sim2sim.py policies/residual_v1 --episodes 96 --workers 12
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from pingpong_sim.controller import BounceController, OnnxPolicy, controller_cfg_from_policy, paddle_from_encoders
from pingpong_sim.real.sim_backend import CpuWorld
from pingpong_sim.ball import G
from pingpong_sim.task.state import bounce_events


class Runner:
    def __init__(self, policy_dir: Path, seed: int, belief_gravity: float = 1.0):
        self.belief_gravity = belief_gravity
        torch.manual_seed(seed)
        spec = json.loads((policy_dir / "policy.json").read_text())
        self.w = w = CpuWorld(spec["task_id"])
        self.ctrl = BounceController(controller_cfg_from_policy(spec, w.task.controller_cfg()), w.kin, w.q_ready, 1,
                                     w.dt, "cpu")
        self.policy = OnnxPolicy(policy_dir, self.ctrl)

    def sense(self) -> None:
        """What the robot does each tick: encoders, cameras -> controller."""
        w = self.w
        q, dq = w.encoders()
        self.ctrl.sense(w.now, q, dq)
        self.ctrl.update(w.rig.detect(w.now))

    def episode(self) -> dict:
        w, ctrl, task = self.w, self.ctrl, self.w.task
        w.reset()
        ctrl.reset(w.ids, w.now)
        ctrl.set_gravity(w.ids, torch.full((1,), G * self.belief_gravity))
        self.policy.reset()
        self.sense()
        prev_vn, armed, t0 = torch.zeros(1), torch.zeros(1, dtype=torch.bool), w.now
        bounces, hits, errs = 0, [], []
        cause = "time_out"
        while w.now - t0 < w.episode_s - 1e-9:
            w.physics(self.policy.act())
            released = torch.tensor([w.released()])
            p, v, face, normal = w.truth()
            q, dq = w.encoders()
            _, _, face_vel = paddle_from_encoders(w.kin, q, dq)
            ev = bounce_events(task, w.target, p, v, face, normal, face_vel, prev_vn, armed, released)
            prev_vn, armed = ev["vn"], ev["armed"]
            if ev["good"][0]:
                bounces += 1
                hits.append((float(ev["apex"][0]), float(ev["landing_err"][0]), float(ev["lateral"][0])))
            # Terminations (env_cfg order).
            if released[0] and p[0, 2] < w.target[2] - w.term["ball_dropped"]["depth"]:
                cause = "ball_dropped"
            elif released[0] and (p[0, :2] - w.target[:2]).norm() > w.term["ball_away"]["radius"]:
                cause = "ball_away"
            elif w.ball_touches_arm():
                cause = "ball_hit_robot"
            elif normal[0, 2] < np.cos(np.radians(w.term["paddle_tilted"]["max_deg"])):
                cause = "paddle_tilted"
            if cause != "time_out":
                break
            w.events()
            w.record()
            self.sense()
            if ctrl.tracked[0] and released[0]:
                errs.append(float((ctrl.est_p - p).norm()))
        return {"bounces": bounces, "cause": cause, "hits": hits, "errs": errs}


def run(job: tuple[str, int, int, float]) -> list[dict]:
    policy_dir, seed, n, belief = job
    torch.set_num_threads(1)
    np.random.seed(seed)
    r = Runner(Path(policy_dir), seed, belief)
    return [r.episode() for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="exported policy directory (policy.onnx + policy.json)")
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--belief-gravity", type=float, default=1.0,
                    help="gravity the ball filter and bouncer assume, x the true one (the rig acted like ~0.89)")
    args = ap.parse_args()
    per = -(-args.episodes // args.workers)
    jobs = [(args.policy, args.seed + i, per, args.belief_gravity) for i in range(args.workers)]
    t = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        eps = [e for r in pool.map(run, jobs) for e in r]
    b = np.array([e["bounces"] for e in eps])
    causes = [e["cause"] for e in eps]
    print(f"sim2sim (CPU MuJoCo, onnxruntime), {args.policy}, belief gravity x{args.belief_gravity:.2f}: {len(eps)} "
          f"episodes in {time.time() - t:.0f} s")
    print(f"  bounces per episode p10/p50/p90: {np.percentile(b, 10):.0f} / {np.percentile(b, 50):.0f} / "
          f"{np.percentile(b, 90):.0f}  (mean {b.mean():.1f}); full 20 s: {100 * np.mean([c == 'time_out' for c in causes]):.1f} %")
    ended = [c for c in causes if c != "time_out"]
    if ended:
        names = ("ball_dropped", "ball_away", "ball_hit_robot", "paddle_tilted")
        print("  ended by: " + ", ".join(f"{k} {100 * ended.count(k) / len(ended):.0f} %" for k in names))
    h = np.array([x for e in eps for x in e["hits"]])
    if len(h):
        print(f"  per bounce ({len(h)}): apex {1e3 * h[:, 0].mean():.0f} +- {1e3 * h[:, 0].std():.0f} mm; landing error "
              f"p50/p90 {1e3 * np.percentile(h[:, 1], 50):.0f} / {1e3 * np.percentile(h[:, 1], 90):.0f} mm; contact off "
              f"centre p50/p90 {1e3 * np.percentile(h[:, 2], 50):.0f} / {1e3 * np.percentile(h[:, 2], 90):.0f} mm")
    e = np.array([x for ep in eps for x in ep["errs"]])
    if len(e):
        print(f"  tracking error p50/p90 {1e3 * np.percentile(e, 50):.1f} / {1e3 * np.percentile(e, 90):.1f} mm")


if __name__ == "__main__":
    main()
