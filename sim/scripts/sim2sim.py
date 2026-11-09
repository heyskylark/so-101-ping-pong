"""Sim-to-sim check of an exported policy: plain CPU MuJoCo, driven through the deployment code path.

Training and eval_task.py: MuJoCo Warp via mjlab, observations assembled by mjlab's managers, PyTorch actor.
Here, one world per process:
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
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

import pingpong_sim.task  # noqa: F401  (registers the tasks)
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from pingpong_sim.ball import SIM_DT
from pingpong_sim.camera_sim import CameraRig
from pingpong_sim.controller import BounceController, OnnxPolicy, paddle_from_encoders
from pingpong_sim.kinematics import ArmKinematics
from pingpong_sim.robot import ARM_JOINTS, READY_JOINT_POS
from pingpong_sim.servo_actuator import ServoDrive
from pingpong_sim.servo_model import ServoModel
from pingpong_sim.task.state import TaskCfg, bounce_events, launch_ball

DATA_FIELDS = ("qpos", "qvel", "qfrc_actuator", "qfrc_bias")


class Mirror:
    """float32 torch copies (1, ...) of MuJoCo model/data arrays, the layout the torch servo model indexes."""

    def __init__(self, src, names):
        self.src, self.names = src, names
        for k in names:
            setattr(self, k, torch.tensor(np.asarray(getattr(src, k)), dtype=torch.float32)[None].clone())

    def pull(self, names=None) -> None:
        for k in names or self.names:
            getattr(self, k)[0] = torch.from_numpy(np.asarray(getattr(self.src, k)))

    def push(self, names=None) -> None:
        for k in names or self.names:
            getattr(self.src, k)[...] = getattr(self, k)[0].numpy()


class World:
    def __init__(self, policy_dir: Path, seed: int):
        torch.manual_seed(seed)
        self.spec = json.loads((policy_dir / "policy.json").read_text())
        env_cfg = load_env_cfg(self.spec["task_id"])
        self.task: TaskCfg = env_cfg.events["reset_episode"].params["task"]
        self.damp = env_cfg.events["ball_contact"].params["dampratio"]
        self.push_dv = env_cfg.events["push_ball"].params["dv"]
        self.push_every = env_cfg.events["push_ball"].interval_range_s
        self.term = {k: env_cfg.terminations[k].params for k in ("ball_dropped", "ball_away", "paddle_tilted")}
        self.episode_s = env_cfg.episode_length_s
        self.decimation = env_cfg.decimation
        self.dt = SIM_DT * self.decimation
        scene = Scene(replace(env_cfg.scene, num_envs=1), "cpu")
        m = self.m = scene.compile()
        env_cfg.sim.mujoco.apply(m)
        self.d = mujoco.MjData(m)
        # Arm joints, their motors, the ball, the blade face, the wrist camera, the contact pair.
        names = [f"robot/{j}" for j in ARM_JOINTS]
        jids = [m.joint(n).id for n in names]
        self.act = [next(i for i in range(m.nu) if m.actuator_trnid[i, 0] == j) for j in jids]
        jb = m.joint("ball/ball_joint").id
        self.qa, self.va = int(m.jnt_qposadr[jb]), int(m.jnt_dofadr[jb])
        self.ball_body = int(m.jnt_bodyid[jb])
        paddle = m.body("robot/paddle").id
        self.robot_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith("robot/") and b != paddle}
        self.site = m.site("robot/blade_face_cap").id
        self.wcam = m.camera("robot/wrist_cam").id
        self.pair = m.pair("ball_blade").id
        self.flip = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        # Servos.
        self.mm, self.dm = Mirror(m, ServoModel.EXPANDED_FIELDS), Mirror(self.d, DATA_FIELDS)
        servo = ServoModel(self.mm, self.dm, m, names, self.spec["firmware_p"], 1, "cpu")
        act_cfg = env_cfg.scene.entities["robot"].articulation.actuators[0]
        self.drive = ServoDrive(act_cfg, servo, list(ARM_JOINTS), SIM_DT, "cpu")
        # Deployment pipeline.
        self.kin = ArmKinematics(m, device="cpu")
        self.q_ready = torch.tensor([READY_JOINT_POS[j] for j in ARM_JOINTS])
        self.ctrl = BounceController(self.task.controller_cfg(), self.kin, self.q_ready, 1, self.dt, "cpu")
        self.rig = CameraRig(list(self.task.cameras), 1, self.dt, self.ctrl.arm.hist, "cpu")
        self.policy = OnnxPolicy(policy_dir, self.ctrl)
        self.ids = torch.zeros(1, dtype=torch.long)
        self.now = 0.0

    def truth(self):
        d = self.d
        p = torch.tensor(d.qpos[self.qa:self.qa + 3], dtype=torch.float32)[None]
        v = torch.tensor(d.qvel[self.va:self.va + 3], dtype=torch.float32)[None]
        face = torch.tensor(d.site_xpos[self.site], dtype=torch.float32)[None]
        normal = torch.tensor(d.site_xmat[self.site].reshape(3, 3)[:, 2], dtype=torch.float32)[None]
        return p, v, face, normal

    def sense(self, restart: bool) -> None:
        """What the robot does each tick: encoders, cameras -> controller."""
        d = self.d
        self.dm.pull()
        servo = self.drive.servo
        self.ctrl.sense(self.now, servo.encoder().clone(), servo.encoder_vel().clone())
        p, v, face, normal = self.truth()
        wR = torch.tensor(d.cam_xmat[self.wcam].reshape(3, 3), dtype=torch.float32)[None] @ self.flip
        self.rig.record(self.now, p, v, face, normal, wR, torch.tensor(d.cam_xpos[self.wcam], dtype=torch.float32)[None])
        if restart:
            self.rig.restart(self.ids)
        self.ctrl.update(self.rig.detect(self.now))

    def set_ball(self, p: torch.Tensor, v: torch.Tensor) -> None:
        self.d.qpos[self.qa:self.qa + 3] = p[0].numpy()
        self.d.qpos[self.qa + 3:self.qa + 7] = (1.0, 0.0, 0.0, 0.0)
        self.d.qvel[self.va:self.va + 6] = 0.0
        self.d.qvel[self.va:self.va + 3] = v[0].numpy()

    def episode(self) -> dict:
        m, d, task = self.m, self.d, self.task
        # Reset: servo draw, restitution, arm at ready + noise, ball launch, pipeline.
        self.drive.request_resample(self.ids)
        self.drive.mark_pending(self.ids)
        m.pair_solref[self.pair, 1] = self.damp[0] + (self.damp[1] - self.damp[0]) * np.random.rand()
        mujoco.mj_resetData(m, d)
        self.dm.pull()
        noise = (2 * torch.rand(1, 5) - 1) * task.joint_noise
        self.drive.servo.teleport(self.q_ready[None] + noise, torch.zeros(1, 5))
        self.dm.push(("qpos", "qvel"))
        hold_p, v0, release = launch_ball(task, self.ctrl.target, 1, self.now, "cpu")
        self.set_ball(hold_p, v0)
        mujoco.mj_forward(m, d)
        self.rig.reset(self.ids, self.now)
        self.ctrl.reset(self.ids, self.now)
        self.policy.reset()
        self.sense(restart=True)
        prev_vn, t0 = torch.zeros(1), self.now
        next_push = self.now + np.random.uniform(*self.push_every)
        bounces, hits, errs = 0, [], []
        cause = "time_out"
        while self.now - t0 < self.episode_s - 1e-9:
            goal = self.policy.act()
            for _ in range(self.decimation):
                self.dm.pull()
                tau = self.drive.step(goal)
                self.mm.push()
                d.ctrl[self.act] = tau[0].numpy()
                mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            self.now += self.dt
            released = torch.tensor([self.now >= float(release[0])])
            p, v, face, normal = self.truth()
            self.dm.pull()
            servo = self.drive.servo
            _, _, face_vel = paddle_from_encoders(self.kin, servo.encoder(), servo.encoder_vel())
            ev = bounce_events(task, self.ctrl.target, p, v, face, normal, face_vel, prev_vn, released)
            prev_vn = ev["vn"]
            if ev["good"][0]:
                bounces += 1
                hits.append((float(ev["apex"][0]), float(ev["landing_err"][0]), float(ev["lateral"][0])))
            # Terminations (env_cfg order).
            if released[0] and p[0, 2] < self.ctrl.target[2] - self.term["ball_dropped"]["depth"]:
                cause = "ball_dropped"
            elif released[0] and (p[0, :2] - self.ctrl.target[:2]).norm() > self.term["ball_away"]["radius"]:
                cause = "ball_away"
            elif any({int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])} - {self.ball_body} <= self.robot_bodies
                     and self.ball_body in (int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2]))
                     for c in d.contact[:d.ncon]):
                cause = "ball_hit_robot"
            elif normal[0, 2] < np.cos(np.radians(self.term["paddle_tilted"]["max_deg"])):
                cause = "paddle_tilted"
            if cause != "time_out":
                break
            # Step events: hold the ball until release; random pushes once released.
            if not released[0]:
                self.set_ball(hold_p, torch.zeros(1, 3))
                mujoco.mj_forward(m, d)
            elif self.now >= next_push:
                d.qvel[self.va:self.va + 3] += np.random.uniform(-self.push_dv, self.push_dv, 3)
                next_push = self.now + np.random.uniform(*self.push_every)
            self.sense(restart=False)
            if self.ctrl.tracked[0] and released[0]:
                errs.append(float((self.ctrl.est_p - p).norm()))
        return {"bounces": bounces, "cause": cause, "hits": hits, "errs": errs,
                "restitution_damp": float(m.pair_solref[self.pair, 1])}


def run(job: tuple[str, int, int]) -> list[dict]:
    policy_dir, seed, n = job
    torch.set_num_threads(1)
    np.random.seed(seed)
    w = World(Path(policy_dir), seed)
    return [w.episode() for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="exported policy directory (policy.onnx + policy.json)")
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    per = -(-args.episodes // args.workers)
    jobs = [(args.policy, args.seed + i, per) for i in range(args.workers)]
    t = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        eps = [e for r in pool.map(run, jobs) for e in r]
    b = np.array([e["bounces"] for e in eps])
    causes = [e["cause"] for e in eps]
    print(f"sim2sim (CPU MuJoCo, onnxruntime), {args.policy}: {len(eps)} episodes in {time.time() - t:.0f} s")
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
