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
from pingpong_sim.robot import ARM_JOINTS, FIRMWARE_P, READY_X, READY_Z


def paddle_from_encoders(kin: ArmKinematics, q_enc: torch.Tensor, dq_enc: torch.Tensor):
    """Blade face centre, normal and velocity (B, 3 each) from encoder angles and velocities."""
    face, normal, o, a = kin.fk(q_enc)
    Jp, _ = kin.jacobian(face, normal, o, a)
    return face, normal, (Jp @ dq_enc[:, :, None])[:, :, 0]


class ArmHistory:
    """What the encoders tell about the arm, every tick: paddle face centre, normal and finite-difference velocity,
    and the wrist camera pose (OpenCV axes), in `hist` (a TickHistory). `lag`: how long after the encoders the
    paddle (and the camera on it) gets there."""

    def __init__(self, kin: ArmKinematics, num_envs: int, dt: float, device, ticks: int = 32, lag: float = 0.0):
        self.kin, self.dt, self.lag = kin, dt, lag
        self.hist = TickHistory(num_envs, ticks, dt, {"center": (3,), "normal": (3,), "vel": (3,), "wR": (3, 3),
                                                      "wp": (3,)}, device)
        self._prev_face: torch.Tensor | None = None

    def push(self, now: float, q_enc: torch.Tensor) -> None:
        """Encoder angles read at `now`: the paddle's state at now + lag."""
        f, nrm, _, _ = self.kin.fk(q_enc)
        wR, wp = self.kin.wrist_camera(q_enc)
        vel = torch.zeros_like(f) if self._prev_face is None else (f - self._prev_face) / self.dt
        self._prev_face = f
        self.hist.push(now + self.lag, center=f, normal=nrm, vel=vel, wR=wR, wp=wp)

    def restart(self, env_ids: torch.Tensor) -> None:
        """After `push`: the latest sample, at rest, becomes the whole history of env_ids (just reset)."""
        last = (self.hist.count - 1) % self.hist.L
        self.hist.buf["vel"][env_ids, last] = 0.0
        self.hist.hold_last(env_ids)


@dataclass
class ControllerCfg:
    target_xy: tuple[float, float] = (READY_X, 0.0)  # m, bounce point (blade centre at the ready pose)
    hit_z: float = READY_Z  # m, blade face height at impact
    apex: float = 0.25  # m, ball rise above the hitting plane
    filter: BallFilterCfg = field(default_factory=BallFilterCfg)
    bouncer: BouncerCfg = field(default_factory=BouncerCfg)
    use_bouncer: bool = True  # run the scripted bouncer (residual policy base; off for the direct policy)
    trust_vel_std: float = 0.5  # m/s: the bouncer plans only on a converged track
    # s: the paddle and the camera on it are where the encoders were this long ago (the real arm: holder and gear
    # compliance between the wrist servos and the paddle, measured from runs; ArmCalib.paddle_lag). 0 in the sim.
    paddle_lag: float = 0.0
    # History observation: a frame every `history_stride` ticks, the last `history_len` frames (16 x 40 ms = 0.64 s,
    # at least one bounce cycle), so the policy can compare what was predicted with what happened and adapt.
    history_len: int = 16
    history_stride: int = 4
    # Bounce memory observation (BounceMemory): this rally's last `memory_slots` bounces, each scored
    # `memory_delay` after it (3-4 frames of the ball leaving at 30 fps, after the camera's latency).
    memory_slots: int = 6
    memory_delay: float = 0.16


HISTORY_FRAME = 10  # ball position vs target (3), velocity (3), tracked, face height vs target, face vz, time to impact
MEMORY_SLOT = 14  # one bounce in BounceMemory


def controller_cfg_from_policy(spec: dict, base: ControllerCfg) -> ControllerCfg:
    """The controller settings an exported policy was trained with (its policy.json "task", "bouncer" and
    "controller" entries) over `base` (the current task's): a policy keeps its bounce height, stroke and history
    when the task defaults move on."""
    from dataclasses import fields

    t, c, flt = spec.get("task", {}), spec.get("controller", {}), spec.get("filter", {})
    known = {f.name for f in fields(BouncerCfg)}
    b = {k: tuple(v) if isinstance(v, list) else v for k, v in spec.get("bouncer", {}).items() if k in known}
    b.setdefault("reach_x", 0.0)  # policies exported before the bouncer carried the blade along x
    return replace(base, target_xy=tuple(t.get("target_xy", base.target_xy)), hit_z=t.get("hit_z", base.hit_z),
                   apex=t.get("apex", base.apex), trust_vel_std=t.get("trust_vel_std", base.trust_vel_std),
                   use_bouncer=t.get("use_bouncer", base.use_bouncer), bouncer=replace(base.bouncer, **b),
                   filter=replace(base.filter, restitution=flt.get("restitution", base.filter.restitution)),
                   history_len=c.get("history_len", base.history_len),
                   history_stride=c.get("history_stride", base.history_stride),
                   memory_slots=c.get("memory_slots", base.memory_slots),
                   memory_delay=c.get("memory_delay", base.memory_delay))


class BounceMemory:
    """What this rally's bounces did, from the ball estimate and the encoders alone (the same on the arm and in
    training), for a policy to learn what its arm is like while it plays: a blade that leans when the encoders say
    level sends every bounce off the same way. Cleared with every new ball.

    A bounce is the estimate's vertical speed turning from falling to rising at the blade (the filter predicts
    through the bounce with the encoders' blade and its nominal restitution, before any camera frame shows it).
    `delay` later, with frames of the ball leaving fused, the bounce is scored: the estimate then against that
    prediction. A bounce followed by another one or by a lost track before that is kept unscored.

    One record, MEMORY_SLOT numbers: state (1 scored, 0.5 unscored, 0 empty), age [s, capped at 5], the encoders'
    face normal lean at the bounce (n_x / n_z, n_y / n_z), blade speed along the normal [m/s], where the ball
    landed from the face centre (x, y) [m], its downward speed coming in [m/s], its velocity going out (3) and the
    surprise (3): that velocity minus the filter's prediction, both taken back to the bounce [m/s]; zero when
    unscored."""

    V_IN, V_OUT = 0.3, 0.2  # m/s: falling at least this fast before, rising at least this fast after
    NEAR_XY, NEAR_Z = 0.12, 0.08  # m: the estimate's distance from the face centre at the turn
    MAX_AGE = 0.1  # s: scored only with a detection this recent (30 fps + the camera's latency: 0.03-0.08 s)

    def __init__(self, slots: int, delay: float, num_envs: int, device):
        self.delay, self.n = delay, num_envs
        self.rec = torch.zeros(num_envs, slots, MEMORY_SLOT, device=device)  # newest first; the age is filled in obs
        self.t = torch.zeros(num_envs, slots, device=device)  # when each bounce was
        # The bounce waiting to be scored: lean (2), blade speed, landing point (2), speed in, predicted velocity
        # out (3); at pend_t (inf: none).
        self.pend = torch.zeros(num_envs, 9, device=device)
        self.pend_t = torch.full((num_envs,), torch.inf, device=device)
        self.prev_v = torch.zeros(num_envs, 3, device=device)
        self.prev_ok = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.sum = torch.zeros(num_envs, 2, device=device)  # of the scored bounces' sideways surprise
        self.count = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.rec[env_ids] = 0.0
        self.t[env_ids] = 0.0
        self.pend_t[env_ids] = torch.inf
        self.prev_ok[env_ids] = False
        self.sum[env_ids] = 0.0
        self.count[env_ids] = 0.0

    def update(self, now: float, tracked: torch.Tensor, age: torch.Tensor, p: torch.Tensor, v: torch.Tensor,
               g: torch.Tensor, paddle) -> None:
        """One tick: the ball estimate (tracked (B,), time since the last detection, position, velocity), the
        gravity the filter believes (B,), and `paddle()` -> face centre, normal, velocity from the encoders (called
        only on a tick with a bounce)."""
        turn = self.prev_ok & tracked & (self.prev_v[:, 2] < -self.V_IN) & (v[:, 2] > self.V_OUT)
        if bool(turn.any()):
            face, normal, face_vel = paddle()
            rel = p - face
            turn = turn & (rel[:, :2].norm(dim=-1) < self.NEAR_XY) & (rel[:, 2].abs() < self.NEAR_Z)
        waiting = torch.isfinite(self.pend_t)
        if bool(waiting.any()):
            due = waiting & (now - self.pend_t >= self.delay)
            close = due | (waiting & turn)
            if bool(close.any()):
                ids = close.nonzero(as_tuple=True)[0]
                scored = due[ids] & tracked[ids] & (age[ids] < self.MAX_AGE)
                v_out = v[ids].clone()
                v_out[:, 2] += g[ids] * (now - self.pend_t[ids])  # back to the bounce
                keep = scored[:, None].float()
                v_out, surprise = keep * v_out, keep * (v_out - self.pend[ids, 6:9])
                rec = torch.cat([0.5 + 0.5 * keep, torch.zeros_like(keep), self.pend[ids, :6], v_out, surprise], dim=1)
                self.rec[ids] = torch.cat([rec[:, None], self.rec[ids, :-1]], dim=1)
                self.t[ids] = torch.cat([self.pend_t[ids, None], self.t[ids, :-1]], dim=1)
                self.sum[ids] += surprise[:, :2]
                self.count[ids] += keep[:, 0]
                self.pend_t[ids] = torch.inf
        if bool(turn.any()):
            ids = turn.nonzero(as_tuple=True)[0]
            nrm = normal[ids]
            self.pend[ids] = torch.cat([nrm[:, :2] / nrm[:, 2:3], (face_vel[ids] * nrm).sum(-1, keepdim=True),
                                        rel[ids, :2], -self.prev_v[ids, 2:3], v[ids]], dim=1)
            self.pend_t[ids] = now
        self.prev_v, self.prev_ok = v, tracked

    def obs(self, now: float) -> torch.Tensor:
        rec = self.rec.clone()
        rec[:, :, 1] = torch.where(rec[:, :, 0] > 0, (now - self.t).clamp(0.0, 5.0), torch.zeros_like(self.t))
        mean = self.sum / self.count.clamp_min(1.0)[:, None]
        return torch.cat([rec.reshape(self.n, -1), mean, (self.count / 10.0).clamp(max=1.0)[:, None]], dim=-1)


class BounceController:
    def __init__(self, cfg: ControllerCfg, kin: ArmKinematics, q_ready: torch.Tensor, num_envs: int, dt: float,
                 device):
        self.cfg, self.kin, self.n, self.dev = cfg, kin, num_envs, device
        self.q_ready = q_ready
        self.target = torch.tensor([*cfg.target_xy, cfg.hit_z + BALL_RADIUS], device=device)  # ball centre at impact
        self.face_target = torch.tensor([*cfg.target_xy, cfg.hit_z], device=device)
        self.arm = ArmHistory(kin, num_envs, dt, device, lag=cfg.paddle_lag)
        self.filter = BallFilter(cfg.filter, num_envs, device)
        # The stroke is planned for the encoders: the paddle arrives `paddle_lag` later, so plan that much earlier.
        self.bouncer = ScriptedBouncer(replace(cfg.bouncer, target_xy=cfg.target_xy, hit_z=cfg.hit_z, apex=cfg.apex,
                                               impact_bias=cfg.bouncer.impact_bias - cfg.paddle_lag),
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
        self.memory = BounceMemory(cfg.memory_slots, cfg.memory_delay, num_envs, device)
        self._history_fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._tick = 0

    def reset(self, env_ids: torch.Tensor, now: float, keep_trim: bool = False) -> None:
        """New episode for env_ids (arm teleported, ball relaunched): forget the track and the plan; the arm history
        restarts at the next `sense`. `keep_trim`: see ScriptedBouncer.reset."""
        self.filter.reset(env_ids)
        self.bouncer.reset(env_ids, now, keep_trim)
        self._restart[env_ids] = True
        self._history_fresh[env_ids] = True
        self.history[env_ids] = 0.0  # the first observation after a reset comes before the next update
        self.memory.reset(env_ids)

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
        self.memory.update(now, self.tracked, self.age, self.est_p, self.est_v, self.filter.g,
                           lambda: paddle_from_encoders(self.kin, self.q_enc, self.dq_enc))
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

    def obs_memory(self) -> torch.Tensor:
        """This rally's last `memory_slots` bounces, newest first, 14 numbers each (BounceMemory), then the mean
        sideways surprise of the scored ones (2) and their count / 10 (capped at 1) (memory_slots x 14 + 3)."""
        return self.memory.obs(self.now)


# Actor observation term name (task config, policy.json) -> BounceController method. "actions" (the previous raw
# actions) is kept by the policy runner.
OBS_TERMS = {"joint_pos": "obs_joint_pos", "joint_vel": "obs_joint_vel", "paddle": "obs_paddle", "ball": "obs_ball",
             "impact": "obs_impact", "bouncer": "obs_bouncer", "history": "obs_history", "memory": "obs_memory"}


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
        q_trained = torch.tensor(self.spec["q_ready"], device=ctrl.dev)
        if (q_trained - ctrl.q_ready).abs().max() > 1e-3:
            raise ValueError(f"{path.name} was trained at another ready pose ({np.degrees(q_trained.cpu().numpy()).round(1)} "
                             "deg): its observations and corrections are relative to that pose")
        gains = [FIRMWARE_P[j] for j in ARM_JOINTS]
        if list(self.spec.get("firmware_p", gains)) != gains:
            raise ValueError(f"{path.name} was trained with firmware gains {self.spec['firmware_p']}; the arm now "
                             f"runs {gains} (robot.FIRMWARE_P)")
        act = self.spec["action"]
        if act["residual"] != ctrl.cfg.use_bouncer:
            raise ValueError("policy and controller disagree on the residual (scripted bouncer) mode")
        self.scale = torch.as_tensor(act["scale"], dtype=torch.float32, device=ctrl.dev)  # one value or per joint
        self.clip = act.get("clip")
        rate = act.get("max_rate")  # rad/s per joint: the goal moves at most this fast (None: unlimited)
        self.step = None if rate is None else \
            torch.as_tensor(rate, dtype=torch.float32, device=ctrl.dev) * float(self.spec["control_dt"])
        self.last_goal = None
        self.lo = torch.tensor(self.spec["joint_lower"], device=ctrl.dev)
        self.hi = torch.tensor(self.spec["joint_upper"], device=ctrl.dev)
        self.hist = {t["name"]: np.zeros((t["history"], t["dim"]), dtype=np.float32) for t in self.terms}
        self.last_action = np.zeros(5, dtype=np.float32)
        self._fresh = True

    def reset(self) -> None:
        self.last_action[:] = 0.0
        self.last_goal = None
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
        a = torch.as_tensor(a, device=self.ctrl.dev)
        if self.clip is not None:
            a = a.clamp(-self.clip, self.clip)
        goal = torch.clamp(base + self.scale * a, self.lo, self.hi)
        if self.step is not None:  # as the training env's action term: from the encoders after a reset
            last = self.ctrl.q_enc if self.last_goal is None else self.last_goal
            goal = torch.minimum(torch.maximum(goal, last - self.step), last + self.step)
            self.last_goal = goal
        return goal
