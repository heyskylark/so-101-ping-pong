"""Per-env state of the bounce task, shared by its MDP terms (`mdp.py`).

Created on first use (`get(env)`) from the `reset_episode` event's `task` parameter and stored on the env:
- Sensors -> servo goals: `controller.BounceController`, the code the deployment runs. Simulated overhead + wrist
  cameras feed its ball filter once per control step after physics (`update`); its estimate is what the policy
  gets, the true ball state only reaches the critic, rewards, terminations and metrics. Its scripted bouncer gives
  the residual policy's base command and the impact prediction (an observation for both policies).
- Launch phase: the ball is held still "in a hand" until a per-episode release time (some episodes start
  mid-flight instead).
- Bounce bookkeeping from the true state (`summary`), once per control step.

Clock: env.common_step_counter * step_dt, shared by all worlds and running through episode resets.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from pingpong_sim.ball import BALL_RADIUS, DRAG_K, G
from pingpong_sim.ball_filter import BallFilterCfg
from pingpong_sim.bouncer import BouncerCfg
from pingpong_sim.camera_sim import OVERHEAD, WRIST, CameraRig, CameraSimCfg
from pingpong_sim.controller import BounceController, ControllerCfg, paddle_from_encoders
from pingpong_sim.kinematics import ArmKinematics
from pingpong_sim.robot import ARM_JOINTS, READY_JOINT_POS
from pingpong_sim.servo_actuator import encoder_pos, encoder_vel


@dataclass
class TaskCfg:
    target_xy: tuple[float, float] = (0.40, 0.0)  # m, bounce point (blade centre at the ready pose)
    hit_z: float = 0.25  # m, blade face height at impact
    apex: float = 0.25  # m, ball rise above the hitting plane
    # The wrist camera is optional: present in 70 % of episodes.
    cameras: tuple[CameraSimCfg, ...] = (OVERHEAD, replace(WRIST, enable_prob=0.7))
    filter: BallFilterCfg = field(default_factory=BallFilterCfg)
    bouncer: BouncerCfg = field(default_factory=BouncerCfg)
    use_bouncer: bool = True  # run the scripted bouncer (residual policy base; off for the direct policy)
    trust_vel_std: float = 0.5  # m/s: the bouncer plans only on a converged track
    # Launch phase.
    drop_height: tuple[float, float] = (0.20, 0.50)  # m, ball centre above the hitting plane
    drop_offset: float = 0.03  # m, uniform per horizontal axis around the target
    hold_time: tuple[float, float] = (0.0, 1.0)  # s, held still before release
    midflight_prob: float = 0.25  # episodes that start with the ball already flying
    midflight_vel: tuple[tuple[float, float], ...] = ((-0.2, 0.2), (-0.2, 0.2), (-1.5, 0.5))  # m/s
    joint_noise: float = 0.03  # rad, start pose around ready
    # Bounce detection (true state): relative normal velocity flips from approach to departure near the face, and
    # the ball has fallen since the last counted bounce (one bounce per fall: a paddle chasing or double-tapping a
    # rising ball also flips the relative velocity, and the direct policy learned to farm that).
    bounce_vn: float = 0.05  # m/s
    min_fall: float = 0.5  # m/s, downward ball speed that arms the next bounce (5 cm dribble: ~1 m/s at impact)
    min_apex: float = 0.05  # m, outgoing rise for a bounce to count

    def controller_cfg(self) -> ControllerCfg:
        return ControllerCfg(target_xy=self.target_xy, hit_z=self.hit_z, apex=self.apex, filter=self.filter,
                             bouncer=self.bouncer, use_bouncer=self.use_bouncer, trust_vel_std=self.trust_vel_std)


def launch_ball(cfg: TaskCfg, target: torch.Tensor, k: int, now: float, device):
    """Episode start of k balls: held still above the target until a random release time, or already flying
    (released now). Returns position, velocity (k, 3) and release time (k,)."""
    u = lambda lo, hi, *shape: lo + (hi - lo) * torch.rand(k, *shape, device=device)  # noqa: E731
    p = target.expand(k, 3).clone()
    p[:, :2] += u(-cfg.drop_offset, cfg.drop_offset, 2)
    p[:, 2] += u(*cfg.drop_height)
    mid = torch.rand(k, device=device) < cfg.midflight_prob
    v = torch.stack([u(*r) for r in cfg.midflight_vel], dim=1) * mid[:, None]
    release = torch.where(mid, torch.full_like(p[:, 0], now), now + u(*cfg.hold_time))
    return p, v, release


def bounce_events(cfg: TaskCfg, target: torch.Tensor, p, v, face, normal, face_vel, prev_vn, armed, released):
    """True-state bounce detection for one control step: the ball's velocity relative to the paddle along the face
    normal (`vn`) flips from approach to departure near the face, and the ball is `armed` (it fell faster than
    `min_fall` since the last counted bounce; the falling sample can't be the step before the bounce, which may
    already be mid-contact). `target`: ball centre at impact. Returns a dict: p, v, normal, released, vn and armed
    (pass back next step), bounced, good (sends the ball >= min_apex up), apex (outgoing rise), landing_err
    (drag-free return to the hitting plane vs target), lateral (contact offset)."""
    rel = p - face
    vn = ((v - face_vel) * normal).sum(-1)
    gap = (rel * normal).sum(-1) - BALL_RADIUS
    lateral = (rel - (rel * normal).sum(-1, keepdim=True) * normal).norm(dim=-1)
    bounced = (released & armed & (prev_vn < -cfg.bounce_vn) & (vn > cfg.bounce_vn) & (gap < 0.04)
               & (lateral < 0.09))
    armed = (armed & ~bounced) | (v[:, 2] < -cfg.min_fall)
    vz = v[:, 2].clamp_min(0.0)
    apex = (p[:, 2] - target[2]) + torch.log1p(DRAG_K * vz * vz / G) / (2 * DRAG_K)
    landing = p[:, :2] + v[:, :2] * (2 * vz / G)[:, None]
    good = bounced & (apex >= cfg.min_apex)
    return {"p": p, "v": v, "normal": normal, "released": released, "vn": vn, "armed": armed, "bounced": bounced,
            "good": good, "apex": apex, "landing_err": (landing - target[:2]).norm(dim=-1), "lateral": lateral}


def get(env) -> PingPongState:
    """The env's task state, created on first use."""
    st = getattr(env, "_pingpong", None)
    if st is None:
        st = env._pingpong = PingPongState(env, env.cfg.events["reset_episode"].params["task"])
    return st


class PingPongState:
    def __init__(self, env, cfg: TaskCfg):
        self.env, self.cfg = env, cfg
        n, dev = env.num_envs, env.device
        self.n, self.dev = n, dev
        m = env.sim.mj_model
        self.robot, self.ball = env.scene["robot"], env.scene["ball"]
        self.kin = ArmKinematics(m, device=dev)
        ids, _ = self.robot.find_joints(ARM_JOINTS, preserve_order=True)
        self.joint_ids = torch.tensor(ids, device=dev)
        self.q_ready = torch.tensor([READY_JOINT_POS[j] for j in ARM_JOINTS], device=dev)
        self.z_ball = cfg.hit_z + BALL_RADIUS  # ball centre height at impact
        self.target = torch.tensor([*cfg.target_xy, self.z_ball], device=dev)  # ball centre at impact
        self.face_target = torch.tensor([*cfg.target_xy, cfg.hit_z], device=dev)
        self.site = m.site("robot/blade_face_cap").id
        self.wcam = m.camera("robot/wrist_cam").id
        jb = m.joint("ball/ball_joint").id
        self.qa, self.va = int(m.jnt_qposadr[jb]), int(m.jnt_dofadr[jb])
        self.flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=dev))  # MuJoCo camera axes -> OpenCV
        dt = env.step_dt
        self.ctrl = BounceController(cfg.controller_cfg(), self.kin, self.q_ready, n, dt, dev)
        self.rig = CameraRig(list(cfg.cameras), n, dt, self.ctrl.arm.hist, dev)
        z3 = torch.zeros(n, 3, device=dev)
        z1 = torch.zeros(n, device=dev)
        self.release = z1.clone()  # time the ball is let go
        self.hold_p = z3.clone()
        self._restart = torch.zeros(n, dtype=torch.bool, device=dev)
        self._updated = -1
        self.track_err = z1.clone()  # |estimate - truth| at the last update (tracked, released), for metrics
        # Nothing is tracked before the first episode reset (mjlab probes the observation shapes before it).
        self._ready = False
        # True-state bookkeeping (`summary`).
        self.prev_vn = z1.clone()
        self.armed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.bounces = torch.zeros(n, dtype=torch.long, device=dev)
        self._summary_at = -1
        self.sum: dict[str, torch.Tensor] = {}

    @property
    def now(self) -> float:
        return self.env.common_step_counter * self.env.step_dt

    # -- episode start -----------------------------------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> None:
        cfg, k, dev, now = self.cfg, len(env_ids), self.dev, self.now
        u = lambda lo, hi, *shape: lo + (hi - lo) * torch.rand(k, *shape, device=dev)  # noqa: E731
        # Arm: ready pose plus noise, at rest (gear joints relaxed).
        q = self.robot.data.default_joint_pos[env_ids].clone()
        q[:, self.joint_ids] += u(-cfg.joint_noise, cfg.joint_noise, 5)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        # Ball: held above the target, or already flying.
        p, v, release = launch_ball(cfg, self.target, k, now, dev)
        root = torch.zeros(k, 13, device=dev)
        root[:, :3], root[:, 3], root[:, 7:10] = p, 1.0, v
        self.ball.write_root_state_to_sim(root, env_ids=env_ids)
        self.hold_p[env_ids] = p
        self.release[env_ids] = release
        # Pipeline and bookkeeping.
        self.rig.reset(env_ids, now)
        self.ctrl.reset(env_ids, now)
        self._restart[env_ids] = True
        self.prev_vn[env_ids] = 0.0
        self.armed[env_ids] = False
        self.bounces[env_ids] = 0
        self._ready = True

    def hold(self) -> None:
        """Keep held balls still (step event, before the observations)."""
        held = (self.now < self.release).nonzero(as_tuple=True)[0]
        if len(held):
            root = torch.zeros(len(held), 13, device=self.dev)
            root[:, :3], root[:, 3] = self.hold_p[held], 1.0
            self.ball.write_root_state_to_sim(root, env_ids=held)

    # -- per control step --------------------------------------------------------------------------------------------

    def ball_true(self):
        d = self.env.sim.data
        return d.qpos[:, self.qa:self.qa + 3], d.qvel[:, self.va:self.va + 3]

    def face_true(self):
        d = self.env.sim.data
        return d.site_xpos[:, self.site], d.site_xmat[:, self.site].reshape(self.n, 3, 3)[:, :, 2]

    def update(self) -> None:
        """Encoders + cameras -> controller (filter, bouncer), once per control step (lazy: the first caller after
        physics runs it)."""
        if not self._ready or self.env.common_step_counter == self._updated:
            return
        self._updated = self.env.common_step_counter
        now, n = self.now, self.n
        d = self.env.sim.data
        p, v = self.ball_true()
        face, normal = self.face_true()
        self.ctrl.sense(now, encoder_pos(self.robot), encoder_vel(self.robot))
        wR = d.cam_xmat[:, self.wcam].reshape(n, 3, 3) @ self.flip
        self.rig.record(now, p, v, face, normal, wR, d.cam_xpos[:, self.wcam])
        if self._restart.any():
            ids = self._restart.nonzero(as_tuple=True)[0]
            self.rig.restart(ids)
            self._restart[ids] = False
        self.ctrl.update(self.rig.detect(now))
        ok = self.ctrl.tracked & (now >= self.release)
        self.track_err = torch.where(ok, (self.ctrl.est_p - p).norm(dim=-1), torch.zeros_like(self.ctrl.age))

    def paddle_from_encoders(self):
        """Face centre, normal and velocity from the encoders right now (the controller's copy is from its last
        `sense`, which rewards and terminations run before)."""
        return paddle_from_encoders(self.kin, encoder_pos(self.robot), encoder_vel(self.robot))

    def summary(self) -> dict[str, torch.Tensor]:
        """True-state bounce events of this control step (once per step; rewards, terminations, metrics)."""
        if self.env.common_step_counter == self._summary_at:
            return self.sum
        self._summary_at = self.env.common_step_counter
        p, v = self.ball_true()
        face, normal = self.face_true()
        _, _, face_vel = self.paddle_from_encoders()
        self.sum = bounce_events(self.cfg, self.target, p, v, face, normal, face_vel, self.prev_vn, self.armed,
                                 self.now >= self.release)
        self.prev_vn, self.armed = self.sum["vn"], self.sum["armed"]
        self.bounces += self.sum["good"].long()
        return self.sum
