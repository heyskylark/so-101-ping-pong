"""MDP terms of the bounce task: events, the action term, observations, rewards, terminations, metrics.

All of them read the shared per-env state (`state.py`). Observations for the actor use only what the deployed
policy will have: encoder angles/velocities (+ forward kinematics), the ball filter's estimate, the scripted
bouncer's command on that estimate, and its own past actions. Rewards, terminations, metrics and the critic use the
true state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import requires_model_fields

from pingpong_sim.robot import ARM_JOINTS
from pingpong_sim.servo_actuator import encoder_pos, encoder_vel
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
    """Reset: ball-blade contact damping ratio, which sets the restitution (0.155-0.235 ~ e 0.65-0.51 at 1 ms,
    scripts/contact_sweep.py; nominal 0.19 ~ 0.58). The drag coefficient can't vary per world: MuJoCo Warp 3.7 keeps
    `geom_fluid` shared."""
    env_ids = _ids(env, env_ids)
    pid = env.sim.mj_model.pair("ball_blade").id
    lo, hi = dampratio
    env.sim.model.pair_solref[env_ids, pid, 1] = lo + (hi - lo) * torch.rand(len(env_ids), device=env.device)


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
    """Servo goals for the 5 arm joints: `base + scale * action`, base = the scripted bouncer's goal (residual) or
    the ready pose (direct). Goals are clipped to the joint limits."""

    entity_name: str = "robot"
    scale: float = 0.1  # rad per unit action
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
        base = st.base if self.cfg.residual else st.q_ready
        self._goal = torch.clamp(base + self.cfg.scale * actions, st.kin.lo, st.kin.hi)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._goal, joint_ids=self._ids)

    def reset(self, env_ids=None) -> None:
        self._raw[env_ids] = 0.0


# -- observations (actor) ------------------------------------------------------------------------------------------


def _st(env) -> PingPongState:
    st = get(env)
    st.update()
    return st


def joint_pos(env) -> torch.Tensor:
    """Encoder angles relative to the ready pose (5)."""
    st = _st(env)
    return encoder_pos(st.robot) - st.q_ready


def joint_vel(env) -> torch.Tensor:
    """Encoder velocities (5)."""
    return encoder_vel(_st(env).robot)


def paddle(env) -> torch.Tensor:
    """Blade face centre relative to its target, face normal, face velocity, from the encoders (9)."""
    st = _st(env)
    face, normal, vel = st.paddle_from_encoders()
    return torch.cat([face - st.face_target, normal, vel], dim=-1)


def ball_estimate(env) -> torch.Tensor:
    """Ball filter estimate: position relative to the impact target, velocity, position / velocity std, time since
    the last detection (capped at 0.5 s), tracked flag (10)."""
    st = _st(env)
    trk = st.tracked[:, None]
    p = torch.where(trk, st.est_p - st.target, torch.zeros_like(st.est_p))
    return torch.cat([p, st.est_v, st.p_std[:, None], st.v_std[:, None], st.age.clamp(0, 0.5)[:, None],
                      trk.float()], dim=-1)


def impact_prediction(env) -> torch.Tensor:
    """Predicted crossing of the hitting plane on a converged track: time to it (capped at 1.2 s), position
    relative to the target, incoming velocity, valid flag (7)."""
    st = _st(env)
    ok = torch.isfinite(st.pred_t)
    okf = ok[:, None].float()
    t = torch.where(ok, st.pred_t, torch.full_like(st.pred_t, 1.2)).clamp(0, 1.2)
    return torch.cat([t[:, None], (st.pred_p[:, :2] - st.target[:2]) * okf, st.pred_v * okf, okf], dim=-1)


def bouncer_command(env) -> torch.Tensor:
    """Scripted bouncer goal relative to ready, and time to its planned impact (capped to [-0.2, 1.2] s) (6)."""
    st = _st(env)
    t = (st.bouncer.t_hit - st.now).clamp(-0.2, 1.2)
    return torch.cat([st.base - st.q_ready, t[:, None]], dim=-1)


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


# -- rewards -------------------------------------------------------------------------------------------------------


def bounce(env) -> torch.Tensor:
    """1 per bounce that sends the ball at least `min_apex` up."""
    return get(env).summary()["good"].float()


def bounce_apex(env, std: float) -> torch.Tensor:
    """At each bounce: how close the outgoing apex (from the outgoing velocity) is to the target height."""
    st = get(env)
    s = st.summary()
    return s["good"].float() * torch.exp(-(((s["apex"] - st.cfg.apex) / std) ** 2))


def bounce_landing(env, std: float) -> torch.Tensor:
    """At each bounce: how close the predicted next landing point is to the target."""
    s = get(env).summary()
    return s["good"].float() * torch.exp(-((s["landing_err"] / std) ** 2))


def bounce_centered(env, std: float) -> torch.Tensor:
    """At each bounce: how close the contact is to the blade centre (where the restitution was measured)."""
    s = get(env).summary()
    return s["good"].float() * torch.exp(-((s["lateral"] / std) ** 2))


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
    return get(env).tracked.float()
