"""Sensors -> servo goals, shared by the training env (`task/state.py`) and deployment.

Per control tick (every `dt`), for B worlds (B = 1 on the real arm):

1. `sense(now, q_enc, dq_enc)`: encoder angles and velocities -> `ArmHistory` (paddle face centre, normal,
   velocity and wrist camera pose by forward kinematics). The ball filter predicts bounces off this paddle, and the
   wrist camera's pose at a detection's capture time comes from it.
2. `update(now, detections)`: camera detections (stacked per camera, sorted by capture time) -> `BallFilter` ->
   estimate; on a converged track the scripted bouncer predicts the next impact and (residual policy) plans the
   base servo goal.
3. Actor observation terms (`obs_*`, in `OBS_TERMS` by the names the task config uses) for the policy.

`OnnxPolicy` runs an exported actor (`scripts/export_policy.py`) on one world: it stacks the terms as the training
env did (order, histories), and maps the action to servo goals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from pingpong_sim.ball import BALL_RADIUS
from pingpong_sim.ball_filter import BallFilter, BallFilterCfg, TickHistory
from pingpong_sim.bouncer import BouncerCfg, ScriptedBouncer
from pingpong_sim.kinematics import ArmKinematics


def paddle_from_encoders(kin: ArmKinematics, q_enc: torch.Tensor, dq_enc: torch.Tensor):
    """Blade face centre, normal and velocity (B, 3 each) from encoder angles and velocities."""
    face, normal, o, a = kin.fk(q_enc)
    Jp, _ = kin.jacobian(face, normal, o, a)
    return face, normal, (Jp @ dq_enc[:, :, None])[:, :, 0]


class ArmHistory:
    """What the encoders tell about the arm, every tick: paddle face centre, normal and finite-difference velocity,
    and the wrist camera pose (OpenCV axes), in `hist` (a TickHistory)."""

    def __init__(self, kin: ArmKinematics, num_envs: int, dt: float, device, ticks: int = 32):
        self.kin, self.dt = kin, dt
        self.hist = TickHistory(num_envs, ticks, dt, {"center": (3,), "normal": (3,), "vel": (3,), "wR": (3, 3),
                                                      "wp": (3,)}, device)
        self._prev_face: torch.Tensor | None = None

    def push(self, now: float, q_enc: torch.Tensor) -> None:
        f, nrm, _, _ = self.kin.fk(q_enc)
        wR, wp = self.kin.wrist_camera(q_enc)
        vel = torch.zeros_like(f) if self._prev_face is None else (f - self._prev_face) / self.dt
        self._prev_face = f
        self.hist.push(now, center=f, normal=nrm, vel=vel, wR=wR, wp=wp)

    def restart(self, env_ids: torch.Tensor) -> None:
        """After `push`: the latest sample, at rest, becomes the whole history of env_ids (just reset)."""
        last = (self.hist.count - 1) % self.hist.L
        self.hist.buf["vel"][env_ids, last] = 0.0
        self.hist.hold_last(env_ids)


@dataclass
class ControllerCfg:
    target_xy: tuple[float, float] = (0.40, 0.0)  # m, bounce point (blade centre at the ready pose)
    hit_z: float = 0.25  # m, blade face height at impact
    apex: float = 0.25  # m, ball rise above the hitting plane
    filter: BallFilterCfg = field(default_factory=BallFilterCfg)
    bouncer: BouncerCfg = field(default_factory=BouncerCfg)
    use_bouncer: bool = True  # run the scripted bouncer (residual policy base; off for the direct policy)
    trust_vel_std: float = 0.5  # m/s: the bouncer plans only on a converged track
    # History observation: a frame every `history_stride` ticks, the last `history_len` frames (16 x 40 ms = 0.64 s,
    # at least one bounce cycle), so the policy can compare what was predicted with what happened and adapt.
    history_len: int = 16
    history_stride: int = 4


HISTORY_FRAME = 10  # ball position vs target (3), velocity (3), tracked, face height vs target, face vz, time to impact


def controller_cfg_from_policy(spec: dict, base: ControllerCfg) -> ControllerCfg:
    """The controller settings an exported policy was trained with (its policy.json "task", "bouncer" and
    "controller" entries) over `base` (the current task's): a policy keeps its bounce height, stroke and history
    when the task defaults move on."""
    from dataclasses import fields

    t, c = spec.get("task", {}), spec.get("controller", {})
    known = {f.name for f in fields(BouncerCfg)}
    b = {k: tuple(v) if isinstance(v, list) else v for k, v in spec.get("bouncer", {}).items() if k in known}
    return replace(base, target_xy=tuple(t.get("target_xy", base.target_xy)), hit_z=t.get("hit_z", base.hit_z),
                   apex=t.get("apex", base.apex), trust_vel_std=t.get("trust_vel_std", base.trust_vel_std),
                   use_bouncer=t.get("use_bouncer", base.use_bouncer), bouncer=replace(base.bouncer, **b),
                   history_len=c.get("history_len", base.history_len),
                   history_stride=c.get("history_stride", base.history_stride))


class BounceController:
    def __init__(self, cfg: ControllerCfg, kin: ArmKinematics, q_ready: torch.Tensor, num_envs: int, dt: float,
                 device):
        self.cfg, self.kin, self.n, self.dev = cfg, kin, num_envs, device
        self.q_ready = q_ready
        self.target = torch.tensor([*cfg.target_xy, cfg.hit_z + BALL_RADIUS], device=device)  # ball centre at impact
        self.face_target = torch.tensor([*cfg.target_xy, cfg.hit_z], device=device)
        self.arm = ArmHistory(kin, num_envs, dt, device)
        self.filter = BallFilter(cfg.filter, num_envs, device)
        self.bouncer = ScriptedBouncer(replace(cfg.bouncer, target_xy=cfg.target_xy, hit_z=cfg.hit_z, apex=cfg.apex),
                                       kin, q_ready, num_envs)
        z3 = torch.zeros(num_envs, 3, device=device)
        z1 = torch.zeros(num_envs, device=device)
        self.now = 0.0
        self.q_enc, self.dq_enc = torch.zeros(num_envs, 5, device=device), torch.zeros(num_envs, 5, device=device)
        self.est_p, self.est_v = z3.clone(), z3.clone()
        self.p_std, self.v_std, self.age = z1.clone(), z1.clone(), z1.clone()
        self.tracked = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.trusted = self.tracked.clone()
        self.pred_t, self.pred_p, self.pred_v = torch.full((num_envs,), torch.inf, device=device), z3.clone(), z3.clone()
        self.base = q_ready.expand(num_envs, 5).clone()  # bouncer goal (ready without a plan)
        self._restart = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.history = torch.zeros(num_envs, cfg.history_len, HISTORY_FRAME, device=device)
        self._history_fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._tick = 0

    def reset(self, env_ids: torch.Tensor, now: float) -> None:
        """New episode for env_ids (arm teleported, ball relaunched): forget the track and the plan; the arm history
        restarts at the next `sense`."""
        self.filter.reset(env_ids)
        self.bouncer.reset(env_ids, now)
        self._restart[env_ids] = True
        self._history_fresh[env_ids] = True
        self.history[env_ids] = 0.0  # the first observation after a reset comes before the next update

    def set_gravity(self, env_ids: torch.Tensor, g: torch.Tensor) -> None:
        """The gravity the ball filter and the bouncer assume for env_ids (B_ids,) (training: a mismatch with the
        simulated ball's; deployed: the true value, the default)."""
        self.filter.g[env_ids] = g
        self.bouncer.set_gravity(env_ids, g)

    def sense(self, now: float, q_enc: torch.Tensor, dq_enc: torch.Tensor) -> None:
        self.now = now
        self.q_enc, self.dq_enc = q_enc, dq_enc
        self.arm.push(now, q_enc)
        if self._restart.any():
            ids = self._restart.nonzero(as_tuple=True)[0]
            self.arm.restart(ids)
            self._restart[ids] = False

    def update(self, det: dict[str, torch.Tensor]) -> None:
        """Fuse this tick's detections (keys t, R, p, z, sigma, valid; (C, B, ...), per world sorted by t), then
        estimate, predict the impact and plan, at the time of the last `sense`."""
        now = self.now
        for c in range(det["t"].shape[0]):
            self.filter.update(det["t"][c], det["R"][c], det["p"][c], det["z"][c], det["sigma"][c], det["valid"][c],
                               self.arm.hist)
        self.est_p, self.est_v, self.p_std, self.v_std, self.age, self.tracked = self.filter.estimate(now, self.arm.hist)
        self.trusted = self.tracked & (self.v_std < self.cfg.trust_vel_std)
        plan_p = torch.where(self.trusted[:, None], self.est_p, torch.full_like(self.est_p, -1.0))  # -1: nothing to hit
        self.pred_t, self.pred_p, self.pred_v = self.bouncer.predict(plan_p, self.est_v)
        if self.cfg.use_bouncer:
            self.base = self.bouncer.step(now, plan_p, self.est_v, prediction=(self.pred_t, self.pred_p, self.pred_v))
        self._push_history()

    def _push_history(self) -> None:
        """A history frame every `history_stride` ticks; worlds just reset fill the whole history with it."""
        fresh = self._history_fresh
        if self._tick % self.cfg.history_stride == 0 or bool(fresh.any()):
            trk = self.tracked[:, None]
            face, _, vel = paddle_from_encoders(self.kin, self.q_enc, self.dq_enc)
            ok = torch.isfinite(self.pred_t)
            t = torch.where(ok, self.pred_t, torch.full_like(self.pred_t, 1.2)).clamp(0, 1.2)
            frame = torch.cat([torch.where(trk, self.est_p - self.target, torch.zeros_like(self.est_p)),
                               torch.where(trk, self.est_v, torch.zeros_like(self.est_v)), trk.float(),
                               (face[:, 2] - self.face_target[2])[:, None], vel[:, 2:3], t[:, None]], dim=-1)
            if self._tick % self.cfg.history_stride == 0:
                self.history = torch.cat([self.history[:, 1:], frame[:, None]], dim=1)
            self.history = torch.where(fresh[:, None, None], frame[:, None].expand_as(self.history), self.history)
            self._history_fresh[:] = False
        self._tick += 1

    def goal_base(self) -> torch.Tensor:
        """What the action is added to: the bouncer goal (residual) or the ready pose (direct)."""
        return self.base if self.cfg.use_bouncer else self.q_ready.expand(self.n, 5)

    # -- actor observation terms -------------------------------------------------------------------------------------

    def obs_joint_pos(self) -> torch.Tensor:
        """Encoder angles relative to the ready pose (5)."""
        return self.q_enc - self.q_ready

    def obs_joint_vel(self) -> torch.Tensor:
        """Encoder velocities (5)."""
        return self.dq_enc

    def obs_paddle(self) -> torch.Tensor:
        """Blade face centre relative to its target, face normal, face velocity, from the encoders (9)."""
        face, normal, vel = paddle_from_encoders(self.kin, self.q_enc, self.dq_enc)
        return torch.cat([face - self.face_target, normal, vel], dim=-1)

    def obs_ball(self) -> torch.Tensor:
        """Ball filter estimate: position relative to the impact target, velocity, position / velocity std, time since
        the last detection (capped at 0.5 s), tracked flag (10)."""
        trk = self.tracked[:, None]
        p = torch.where(trk, self.est_p - self.target, torch.zeros_like(self.est_p))
        return torch.cat([p, self.est_v, self.p_std[:, None], self.v_std[:, None], self.age.clamp(0, 0.5)[:, None],
                          trk.float()], dim=-1)

    def obs_impact(self) -> torch.Tensor:
        """Predicted crossing of the hitting plane on a converged track: time to it (capped at 1.2 s), position
        relative to the target, incoming velocity, valid flag (7)."""
        ok = torch.isfinite(self.pred_t)
        okf = ok[:, None].float()
        t = torch.where(ok, self.pred_t, torch.full_like(self.pred_t, 1.2)).clamp(0, 1.2)
        return torch.cat([t[:, None], (self.pred_p[:, :2] - self.target[:2]) * okf, self.pred_v * okf, okf], dim=-1)

    def obs_bouncer(self) -> torch.Tensor:
        """Scripted bouncer goal relative to ready, and time to its planned impact (capped to [-0.2, 1.2] s) (6)."""
        t = (self.bouncer.t_hit - self.now).clamp(-0.2, 1.2)
        return torch.cat([self.base - self.q_ready, t[:, None]], dim=-1)

    def obs_history(self) -> torch.Tensor:
        """The last `history_len` frames, oldest first, one every `history_stride` ticks: ball estimate vs the impact
        target and its velocity, tracked flag, blade face height vs target and vertical speed, predicted time to the
        hitting plane (capped at 1.2 s) (history_len x 10)."""
        return self.history.reshape(self.n, -1)


# Actor observation term name (task config, policy.json) -> BounceController method. "actions" (the previous raw
# actions) is kept by the policy runner.
OBS_TERMS = {"joint_pos": "obs_joint_pos", "joint_vel": "obs_joint_vel", "paddle": "obs_paddle", "ball": "obs_ball",
             "impact": "obs_impact", "bouncer": "obs_bouncer", "history": "obs_history"}


class OnnxPolicy:
    """An exported actor (`sim/policies/<name>/`: policy.onnx + policy.json) driving one world of a BounceController.
    Call `reset()` at episode start, then once per tick after `ctrl.update`: `goal = policy.act()`."""

    def __init__(self, path: str | Path, ctrl: BounceController):
        import onnxruntime as ort

        path = Path(path)
        self.spec = json.loads((path / "policy.json").read_text())
        self.ctrl = ctrl
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1  # a small MLP: threads only add latency
        self.session = ort.InferenceSession(str(path / "policy.onnx"), opts, providers=["CPUExecutionProvider"])
        self.terms = self.spec["observation"]["terms"]
        for t in self.terms:
            if t["name"] != "actions" and t["name"] not in OBS_TERMS:
                raise ValueError(f"policy observation term {t['name']!r} has no BounceController method")
        act = self.spec["action"]
        if act["residual"] != ctrl.cfg.use_bouncer:
            raise ValueError("policy and controller disagree on the residual (scripted bouncer) mode")
        self.scale = act["scale"]
        self.lo = torch.tensor(self.spec["joint_lower"], device=ctrl.dev)
        self.hi = torch.tensor(self.spec["joint_upper"], device=ctrl.dev)
        self.hist = {t["name"]: np.zeros((t["history"], t["dim"]), dtype=np.float32) for t in self.terms}
        self.last_action = np.zeros(5, dtype=np.float32)
        self._fresh = True

    def reset(self) -> None:
        self.last_action[:] = 0.0
        self._fresh = True  # histories fill with the first frame, as mjlab's history buffers do after a reset

    def observation(self) -> np.ndarray:
        parts = []
        for t in self.terms:
            name = t["name"]
            v = self.last_action if name == "actions" else \
                getattr(self.ctrl, OBS_TERMS[name])()[0].detach().cpu().numpy().astype(np.float32)
            h = self.hist[name]
            if self._fresh:
                h[:] = v
            else:
                h[:-1] = h[1:]
                h[-1] = v
            parts.append(h.reshape(-1))
        self._fresh = False
        return np.concatenate(parts)[None]

    def act(self) -> torch.Tensor:
        """Servo goals (1, 5) for this tick."""
        a = self.session.run(["actions"], {"obs": self.observation()})[0][0]
        self.last_action = a.astype(np.float32)
        base = self.ctrl.goal_base()
        return torch.clamp(base + self.scale * torch.as_tensor(a, device=self.ctrl.dev), self.lo, self.hi)
