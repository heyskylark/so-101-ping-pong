"""MDP terms of the bounce task: events, the action term, observations, rewards, terminations, metrics.

All of them read the shared per-env state (`state.py`). Observations for the actor use only what the deployed
policy will have: encoder angles/velocities (+ forward kinematics), the ball filter's estimate, the scripted
bouncer's command on that estimate, and its own past actions. Rewards, terminations, metrics and the critic use the
true state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pingpong_sim.ball import G
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import requires_model_fields

from pingpong_sim.controller import BounceController
from pingpong_sim.robot import ARM_JOINTS, joint_inertia
from pingpong_sim.servo_actuator import encoder_vel, servo_model
from pingpong_sim.task.state import PingPongState, TaskCfg, get

__all__ = ["TaskCfg"]

# -- events --------------------------------------------------------------------------------------------------------


def reset_episode(env, env_ids, task: TaskCfg) -> None:
    """Reset: arm at ready (+ noise), ball held or flying, fresh camera draw, filter, bouncer plan. `task` configures
    the shared task state (read once, when the state is created)."""
    del task
    get(env).reset(_ids(env, env_ids))


def hold_ball(env, env_ids) -> None:
    """Step: keep the held balls still until their release time."""
    del env_ids
    get(env).hold()


def _ids(env, env_ids) -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids


@requires_model_fields("pair_solref")
def randomize_ball_contact(env, env_ids, dampratio: tuple[float, float]) -> None:
    """Reset: ball-blade contact damping ratio, which sets the restitution (0.105-0.17 ~ e 0.76-0.61 at 1 ms,
    scripts/contact_sweep.py; nominal 0.135 ~ 0.69). The drag coefficient can't vary per world: MuJoCo Warp 3.7 keeps
    `geom_fluid` shared."""
    env_ids = _ids(env, env_ids)
    pid = env.sim.mj_model.pair("ball_blade").id
    lo, hi = dampratio
    env.sim.model.pair_solref[env_ids, pid, 1] = lo + (hi - lo) * torch.rand(len(env_ids), device=env.device)


def randomize_arm_flex(env, env_ids, stiffness: dict[str, tuple[float, float]],
                       damping_ratio: tuple[float, float], play_deg: dict[str, tuple[float, float]] | None = None,
                       zero_error_deg: dict[str, float] | None = None) -> None:
    """Reset: stiffness [N m/rad] of the structure after the named joints' encoders, log-uniform per joint (the
    other joints stay rigid), and its damping ratio (ServoModel.set_link_stiffness). The blade then droops under
    its own weight and rings after a stroke by more than the encoders report. `play_deg`: free play in the named
    joints' gears, total travel, uniform (ServoModel.set_play). `zero_error_deg`: calibration error of the named
    joints' zeros, uniform within +- the value (ServoModel.set_zero_error)."""
    env_ids = _ids(env, env_ids)
    inertia = getattr(env, "_joint_inertia", None)
    if inertia is None:
        inertia = env._joint_inertia = joint_inertia(env.sim.mj_model)
    k, c = draw_link_flex(stiffness, damping_ratio, inertia, len(env_ids), env.device)
    servo = servo_model(env.scene["robot"])
    servo.set_link_stiffness(k, c, env_ids)
    if play_deg:
        servo.set_play(draw_play(play_deg, len(env_ids), env.device), env_ids)
    if zero_error_deg:
        servo.set_zero_error(draw_zero_error(zero_error_deg, len(env_ids), env.device), env_ids)


def draw_zero_error(zero_error_deg: dict[str, float], k: int, device="cpu") -> torch.Tensor:
    """Joint zero errors [rad], (k, 5), uniform within +- the given degrees: the draw of `randomize_arm_flex`."""
    err = torch.zeros(k, len(ARM_JOINTS), device=device)
    for name, a in zero_error_deg.items():
        err[:, ARM_JOINTS.index(name)] = torch.deg2rad(a * (2 * torch.rand(k, device=device) - 1))
    return err


def draw_play(play_deg: dict[str, tuple[float, float]], k: int, device="cpu") -> torch.Tensor:
    """Gear play [rad], (k, 5): the draw of `randomize_arm_flex`."""
    play = torch.zeros(k, len(ARM_JOINTS), device=device)
    for name, (lo, hi) in play_deg.items():
        play[:, ARM_JOINTS.index(name)] = torch.deg2rad(lo + (hi - lo) * torch.rand(k, device=device))
    return play


def draw_link_flex(stiffness: dict[str, tuple[float, float]], damping_ratio: tuple[float, float],
                   inertia: list[float], k: int, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Link stiffness and damping, each (k, 5): the draw of `randomize_arm_flex`."""
    ks = torch.full((k, len(ARM_JOINTS)), torch.inf, device=device)
    cs = torch.zeros(k, len(ARM_JOINTS), device=device)
    for name, (lo, hi) in stiffness.items():
        j = ARM_JOINTS.index(name)
        ks[:, j] = lo * (hi / lo) ** torch.rand(k, device=device)
        zeta = damping_ratio[0] + (damping_ratio[1] - damping_ratio[0]) * torch.rand(k, device=device)
        cs[:, j] = 2 * zeta * torch.sqrt(ks[:, j] * inertia[j])
    return ks, cs


def push_ball(env, env_ids, dv: float) -> None:
    """Interval: add a uniform +-dv velocity kick (per axis) to released balls (air drafts, spin effects)."""
    st = get(env)
    env_ids = _ids(env, env_ids)
    env_ids = env_ids[st.now >= st.release[env_ids]]
    if len(env_ids) == 0:
        return
    ball = st.ball
    vel = torch.cat([ball.data.root_link_lin_vel_w[env_ids], ball.data.root_link_ang_vel_w[env_ids]], dim=-1)
    vel[:, :3] += dv * (2 * torch.rand(len(env_ids), 3, device=env.device) - 1)
    ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


# -- actions -------------------------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class BounceActionCfg(ActionTermCfg):
    """Servo goals for the 5 arm joints: `base + scale * clip(action)`, base = the scripted bouncer's goal (residual)
    or the ready pose (direct). Goals are clipped to the joint limits."""

    entity_name: str = "robot"
    scale: float | tuple[float, ...] = 0.1  # rad per unit action (one value, or one per joint)
    clip: float | None = None  # raw actions are clipped to +-clip before scaling (bounds what the policy can add)
    residual: bool = True

    def build(self, env) -> BounceAction:
        return BounceAction(self, env)


class BounceAction(ActionTerm):
    cfg: BounceActionCfg

    def __init__(self, cfg: BounceActionCfg, env):
        super().__init__(cfg, env)
        ids, _ = self._entity.find_joints(ARM_JOINTS, preserve_order=True)
        self._ids = torch.tensor(ids, device=self.device)
        self._raw = torch.zeros(self.num_envs, 5, device=self.device)
        self._goal = torch.zeros(self.num_envs, 5, device=self.device)
        self._scale = torch.as_tensor(cfg.scale, dtype=torch.float32, device=self.device).expand(5)

    @property
    def action_dim(self) -> int:
        return 5

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        st = get(self._env)
        st.update()  # normally done by the last observation already
        self._raw[:] = actions
        a = actions if self.cfg.clip is None else actions.clamp(-self.cfg.clip, self.cfg.clip)
        self._goal = torch.clamp(st.ctrl.goal_base() + self._scale * a, st.kin.lo, st.kin.hi)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._goal, joint_ids=self._ids)

    def reset(self, env_ids=None) -> None:
        self._raw[env_ids] = 0.0


# -- observations (actor) ------------------------------------------------------------------------------------------


def _st(env) -> PingPongState:
    st = get(env)
    st.update()
    return st


def _ctrl(env) -> BounceController:
    """The deployment builds the same terms from the same controller (controller.OnnxPolicy)."""
    return _st(env).ctrl


def joint_pos(env) -> torch.Tensor:
    """Encoder angles relative to the ready pose (5)."""
    return _ctrl(env).obs_joint_pos()


def joint_vel(env) -> torch.Tensor:
    """Encoder velocities (5)."""
    return _ctrl(env).obs_joint_vel()


def paddle(env) -> torch.Tensor:
    """Blade face centre relative to its target, face normal, face velocity, from the encoders (9)."""
    return _ctrl(env).obs_paddle()


def ball_estimate(env) -> torch.Tensor:
    """Ball filter estimate: position relative to the impact target, velocity, position / velocity std, time since
    the last detection (capped at 0.5 s), tracked flag (10)."""
    return _ctrl(env).obs_ball()


def impact_prediction(env) -> torch.Tensor:
    """Predicted crossing of the hitting plane on a converged track: time to it (capped at 1.2 s), position
    relative to the target, incoming velocity, valid flag (7)."""
    return _ctrl(env).obs_impact()


def bouncer_command(env) -> torch.Tensor:
    """Scripted bouncer goal relative to ready, and time to its planned impact (capped to [-0.2, 1.2] s) (6)."""
    return _ctrl(env).obs_bouncer()


def history(env) -> torch.Tensor:
    """0.64 s of frames (one per 40 ms): ball estimate and velocity, tracked flag, blade face height and vertical
    speed, predicted time to the hitting plane (160)."""
    return _ctrl(env).obs_history()


def memory(env) -> torch.Tensor:
    """This rally's last 6 bounces, newest first, 14 numbers each: state (1 scored, 0.5 unscored, 0 empty), age,
    face lean at the bounce by the encoders (2), blade speed along its normal, landing point from the face centre
    (2), speed in, velocity out (3), surprise = velocity out minus the filter's prediction (3); then the mean
    sideways surprise of the scored ones (2) and their count / 10 (87)."""
    return _ctrl(env).obs_memory()


# -- observations (critic, privileged) -----------------------------------------------------------------------------


def ball_true(env) -> torch.Tensor:
    """True ball position relative to the impact target and velocity (6)."""
    st = _st(env)
    p, v = st.ball_true()
    return torch.cat([p - st.target, v], dim=-1)


def paddle_true(env) -> torch.Tensor:
    """True blade face centre relative to its target and normal, including gear flex (6)."""
    st = _st(env)
    face, normal = st.face_true()
    return torch.cat([face - st.face_target, normal], dim=-1)


def launch_state(env) -> torch.Tensor:
    """Released flag and time to release (capped at 1 s) (2)."""
    st = _st(env)
    left = (st.release - st.now).clamp(0, 1.0)
    return torch.stack([(left <= 0).float(), left], dim=-1)


def mismatch(env) -> torch.Tensor:
    """Gravity the filter and bouncer believe, relative to the true one, minus 1 (1)."""
    return (_ctrl(env).filter.g / G - 1.0)[:, None]


def zero_error(env) -> torch.Tensor:
    """Calibration error of the joint zeros [rad] (5)."""
    return servo_model(env.scene["robot"]).zero_error.clone()


# -- rewards -------------------------------------------------------------------------------------------------------


def _apex_weight(env, std: float) -> torch.Tensor:
    """Per good bounce: exp(-((apex - target) / std)^2), else 0. Scales the per-bounce rewards so that more, lower
    bounces don't pay: unweighted, the direct policy learned to dribble ~5 bounces/s just above `min_apex`."""
    st = get(env)
    s = st.summary()
    return s["good"].float() * torch.exp(-(((s["apex"] - st.cfg.apex) / std) ** 2))


def bounce(env, apex_std: float) -> torch.Tensor:
    """Per bounce that sends the ball at least `min_apex` up, weighted by how close its apex is to the target."""
    return _apex_weight(env, apex_std)


def bounce_apex(env, std: float) -> torch.Tensor:
    """At each bounce: how close the outgoing apex (from the outgoing velocity) is to the target height (sharper)."""
    return _apex_weight(env, std)


def bounce_landing(env, std: float, apex_std: float) -> torch.Tensor:
    """At each bounce: how close the predicted next landing point is to the target, weighted by the apex."""
    s = get(env).summary()
    return _apex_weight(env, apex_std) * torch.exp(-((s["landing_err"] / std) ** 2))


def bounce_centered(env, std: float, apex_std: float) -> torch.Tensor:
    """At each bounce: how close the contact is to the blade centre (where the restitution was measured), weighted
    by the apex."""
    s = get(env).summary()
    return _apex_weight(env, apex_std) * torch.exp(-((s["lateral"] / std) ** 2))


def ball_energy(env, radius: float) -> torch.Tensor:
    """Every step: the ball's vertical energy above the hitting plane (height + speed) as a share of the target
    bounce's, capped at 1, while it is over the blade (within `radius` of the true face centre) and released. A
    ball at rest on the blade earns nothing; one bounced at the target height earns 1 all the way."""
    st = get(env)
    s = st.summary()
    p, v = s["p"], s["v"]
    face, _ = st.face_true()
    over = s["released"] & ((p[:, :2] - face[:, :2]).norm(dim=-1) < radius) & (p[:, 2] > face[:, 2])
    energy = G * (p[:, 2] - st.z_ball).clamp_min(0.0) + 0.5 * v[:, 2] ** 2
    return over.float() * (energy / (G * st.cfg.apex)).clamp(max=1.0)


def action_l2(env) -> torch.Tensor:
    """Squared action (residual: departure from the scripted bouncer)."""
    return (env.action_manager.action ** 2).sum(-1)


def joint_vel_l2(env) -> torch.Tensor:
    return (encoder_vel(get(env).robot) ** 2).sum(-1)


# -- terminations --------------------------------------------------------------------------------------------------


def ball_dropped(env, depth: float) -> torch.Tensor:
    """Ball centre more than `depth` below the hitting plane."""
    st = get(env)
    s = st.summary()
    return s["released"] & (s["p"][:, 2] < st.z_ball - depth)


def ball_away(env, radius: float) -> torch.Tensor:
    """Ball more than `radius` from the target horizontally."""
    st = get(env)
    s = st.summary()
    return s["released"] & ((s["p"][:, :2] - st.target[:2]).norm(dim=-1) > radius)


def ball_hit_robot(env, sensor_name: str) -> torch.Tensor:
    """Ball touching the arm (anything but the blade)."""
    return torch.any(env.scene[sensor_name].data.found > 0, dim=-1)


def paddle_tilted(env, max_deg: float) -> torch.Tensor:
    """Blade face tilted more than `max_deg` from level."""
    _, normal = get(env).face_true()
    return normal[:, 2] < torch.cos(torch.deg2rad(torch.tensor(max_deg)))


# -- metrics -------------------------------------------------------------------------------------------------------


def bounces(env) -> torch.Tensor:
    """Good bounces so far this episode (report with reduce='last')."""
    st = get(env)
    st.summary()
    return st.bounces.float()


def tracking_error(env) -> torch.Tensor:
    """Ball estimate position error (m) at the last update, while tracked and released; 0 otherwise."""
    return get(env).track_err


def tracked(env) -> torch.Tensor:
    return get(env).ctrl.tracked.float()
