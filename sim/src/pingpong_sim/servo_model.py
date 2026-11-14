"""STS3215 servo model on MuJoCo Warp with per-world parameters and gear compliance.

Physics is BAM's STS3215 m6 model (Duclusaud et al., ICRA 2025; bam/feetech/actuator.py, bam/mjlab.py):

- firmware: internal target moves toward the goal at most `max_velocity`; duty = clip((target - q_enc) * kp *
  0.166 * error_gain_ratio, +-0.97), also clipped so the motor current stays under `max_current` (BAM's
  VoltageControlledActuator limiter); voltage = vin * duty
- DC motor: tau = kt / R * voltage - kt^2 / R * dq_motor
- friction (m6): Coulomb + Stribeck + directional load-dependent terms, written into MuJoCo's dof_frictionloss
  (static friction solved by the constraint solver) and dof_damping (viscous)

Additions for this arm:

- **Gear compliance.** Each servo joint gets a passive hinge in series on the same axis (`<joint>_gear`): a
  spring-damper (stiffness per joint, damping shared) with a +-5 deg hard stop. The motor torque, armature,
  friction and back-EMF act on the servo (motor-side) joint; the STS3215 encoder is on the output shaft, so the
  controller sees q_enc = q_servo + q_gear, the angle that is logged. Fitted to the 2026-11-08 recording: the
  holding joints are pushed 2-3x further by a neighbour's step than with rigid gears, and free play (backlash)
  fitted to zero, so compliance replaced it.
- **Gear play.** Optionally (`set_play`), free travel in a joint's gears: a dead zone of the gear joint's spring
  (its torque inside +-play/2 is cancelled with an applied one), read in full by the output encoder. On the rig
  (2026-11-12, replacement shoulder servo) the lift jumps ~1.5 deg between two positions whenever a wrist stroke's
  reaction reverses, at any firmware gain: the ready pose puts little gravity load on it to keep the gears closed.
- **Link flex.** Optionally (`set_link_stiffness`), the structure after an encoder bends too (printed links, horn
  screws, the paddle holder): a second spring in series on the same axis. It has no inertia between it and the gear
  spring, so the gear joint carries both (series stiffness) and the encoder reads only the gear's share of the
  deflection: the blade droops and rings by more than the encoders report.
- **Per-world parameters.** Every parameter is a (num_envs, 1) or (num_envs, num_joints) tensor, so a batch of
  worlds can run a batch of candidate parameter sets (system identification) or randomized servos (training).
- **Feedback delay.** The firmware acts on the encoder angle `feedback_delay` ago (sensor/loop/PWM latency inside
  the servo). On the real arm, holding joints keep hunting at 3-4 Hz once a neighbour has moved them, which an
  instantaneous P loop doesn't do.
- **Command delay** (host -> servo) is applied by the caller (it owns the goal stream).
- **External load** for the friction terms is -qfrc_bias at the servo joint (gravity/Coriolis of everything it
  carries); BAM's mjlab backend also adds constraint forces, which here are only the friction itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

ERROR_GAIN = 0.166  # BAM STS3215Actuator: duty per (rad * firmware P), measured with an oscilloscope
MAX_PWM = 0.97
GEAR_SUFFIX = "_gear"
GEAR_ARMATURE = 1e-5  # kg m^2, numerical only
GEAR_RANGE = math.radians(5.0)  # hard stop of the compliant gear joint
GEAR_SOLREF_LIMIT = (0.004, 1.0)
RIGID = math.inf  # gear_stiffness value meaning "no compliance" (gear joint locked by a ~0 range)
FEEDBACK_HISTORY = 32  # steps of encoder history kept for the feedback delay (>= its upper bound / dt + 1)
# BAM: stiff joint-friction constraint (MuJoCo Warp has no noslip solver), direct form.
FRICTION_SOLREF = (-5.0e4, -2.0e2)
FRICTION_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)


@dataclass(frozen=True)
class Param:
    name: str
    lo: float
    hi: float
    log: bool
    per_joint: bool = False


# Search space for system identification. Shared parameters are properties of the STS3215 12 V servo; per-joint
# ones absorb assembly differences (gear wear, horn fit, cable drag).
PARAMS = (
    Param("kt", 1.0, 5.0, True),
    Param("R", 2.0, 14.0, True),
    Param("error_gain_ratio", 0.5, 3.0, True),
    Param("max_velocity", 3.0, 12.0, False),
    Param("armature", 0.005, 0.06, True),
    Param("friction_base", 0.005, 0.3, True),
    Param("friction_viscous", 0.001, 0.2, True),
    Param("load_friction_external", 0.02, 1.0, True),
    Param("load_friction_motor", 0.005, 2.0, True),
    Param("stribeck_scale", 0.01, 10.0, True),
    Param("dtheta_stribeck", 0.05, 2.0, True),
    Param("delay", 0.0, 0.05, False),  # s, host command -> servo goal
    Param("feedback_delay", 0.0, 0.02, False),  # s, encoder -> PWM inside the servo
    Param("max_current", 0.5, 8.0, True),  # A, firmware current limit (register Protection_Current: 2.0 A)
    # Gear spring [N m/rad]. Upper bound kept where explicit integration at 1 ms stays stable on the wrist roll
    # (paddle inertia about the roll axis ~2.5e-4 kg m^2).
    Param("gear_stiffness", 20.0, 600.0, True, per_joint=True),
    Param("gear_damping", 0.002, 1.0, True),  # N m s/rad
    Param("friction_scale", 0.3, 3.0, True, per_joint=True),
)
# Fixed m6 terms (scaled together by stribeck_scale): from BAM's 7.4 V STS3215 fit.
M6_FIXED = dict(
    friction_stribeck=4.257e-05,
    load_friction_motor_stribeck=1.245e-04,
    load_friction_external_stribeck=0.1423,
    load_friction_motor_quad=7.595e-03,
    load_friction_external_quad=3.404e-03,
    alpha=9.931,
)


def param_dim(num_joints: int) -> int:
    return sum(num_joints if p.per_joint else 1 for p in PARAMS)


def decode(u: np.ndarray, num_joints: int) -> dict[str, np.ndarray]:
    """Unit-cube coordinates (..., D) -> parameter values; per-joint params get a trailing joint axis."""
    out, k = {}, 0
    for p in PARAMS:
        n = num_joints if p.per_joint else 1
        x = np.clip(u[..., k:k + n], 0.0, 1.0)
        v = p.lo * (p.hi / p.lo) ** x if p.log else p.lo + (p.hi - p.lo) * x
        out[p.name] = v if p.per_joint else v[..., 0]
        k += n
    return out


def encode(values: dict[str, np.ndarray | float], num_joints: int) -> np.ndarray:
    """Inverse of decode (values outside the bounds are clipped)."""
    parts = []
    for p in PARAMS:
        v = np.broadcast_to(np.asarray(values[p.name], float), (num_joints,) if p.per_joint else ())
        v = np.clip(v, p.lo, p.hi)
        x = np.log(v / p.lo) / np.log(p.hi / p.lo) if p.log else (v - p.lo) / (p.hi - p.lo)
        parts.append(np.atleast_1d(x))
    return np.concatenate(parts)


def add_gear_joints(spec: mujoco.MjSpec, joints: tuple[str, ...]) -> None:
    """Insert a passive compliant hinge after each servo joint, on the same axis, in the same body."""
    for name in joints:
        j = spec.joint(name)
        g = j.parent.add_joint(
            name=name + GEAR_SUFFIX,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=j.axis,
            pos=j.pos,
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            range=(-GEAR_RANGE, GEAR_RANGE),
            armature=GEAR_ARMATURE,
            damping=[0.0],
            frictionloss=0.0,
        )
        g.solref_limit = GEAR_SOLREF_LIMIT


def motorize(spec: mujoco.MjSpec, joints: tuple[str, ...] | list[str]) -> list[mujoco.MjsActuator]:
    """The position actuators of `joints` -> unlimited torque motors (gear 1); the joints' friction, damping and
    armature are set per world at run time. Returns the actuators in `joints` order."""
    by_target = {a.target: a for a in spec.actuators}
    acts = []
    for name in joints:
        if name not in by_target:
            raise ValueError(f"no actuator on joint {name!r} to convert")
        act = by_target[name]
        act.set_to_motor()
        act.gear = [1.0, 0, 0, 0, 0, 0]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        j = spec.joint(name)
        j.damping = np.zeros((3, 1))
        j.frictionloss = 0.0
        j.solref_friction = FRICTION_SOLREF
        j.solimp_friction = FRICTION_SOLIMP
        acts.append(act)
    return acts


class ServoModel:
    """Vectorized STS3215 servos for `num_envs` worlds x len(joints) joints of one MuJoCo Warp simulation.

    `model` / `data` are mjlab's torch views of the MuJoCo Warp model and data (Simulation.model / .data, or what
    an mjlab actuator receives in initialize); the fields in EXPANDED_FIELDS must be expanded per world.
    `joints` are compiled-model joint names (with any entity prefix); the gear joints are `<joint>_gear`.
    """

    EXPANDED_FIELDS = ("dof_frictionloss", "dof_damping", "dof_armature", "jnt_range", "jnt_stiffness")

    def __init__(self, model, data, mj_model: mujoco.MjModel, joints: list[str], firmware_p: list[float],
                 num_envs: int, device: str):
        self.model, self.data, self.joints = model, data, joints
        dev = device
        jid = [mj_model.joint(j).id for j in joints]
        gid = [mj_model.joint(j + GEAR_SUFFIX).id for j in joints]
        self.servo_q = torch.tensor(mj_model.jnt_qposadr[jid], device=dev)
        self.servo_v = torch.tensor(mj_model.jnt_dofadr[jid], device=dev)
        self.gear_q = torch.tensor(mj_model.jnt_qposadr[gid], device=dev)
        self.gear_v = torch.tensor(mj_model.jnt_dofadr[gid], device=dev)
        self.gear_jnt = torch.tensor(gid, device=dev)
        self.kp = torch.tensor(firmware_p, dtype=torch.float32, device=dev)[None]
        self.n, self.J, self.device = num_envs, len(joints), dev
        self.p = {p.name: torch.zeros(self.n, self.J if p.per_joint else 1, device=dev) for p in PARAMS}
        self.target = torch.zeros(self.n, self.J, device=dev)
        self.vin = torch.full((self.n, 1), 12.0, device=dev)
        # Encoder history for the feedback delay (ring buffer, newest at self.ptr).
        self.hist = torch.zeros(FEEDBACK_HISTORY, self.n, self.J, device=dev)
        # Structure flex after the encoder (set_link_stiffness): rigid by default, as the model was fitted.
        self.link_stiffness = torch.full((self.n, self.J), math.inf, device=dev)
        self.link_damping = torch.zeros(self.n, self.J, device=dev)
        self.seen = torch.ones(self.n, self.J, device=dev)  # share of the gear joint's deflection the encoder reads
        # Free play in the gears (set_play): none by default.
        self.play = torch.zeros(self.n, self.J, device=dev)
        self.k_series = torch.zeros(self.n, self.J, device=dev)  # the gear joint's spring (gear and link in series)
        self.has_play = False
        self.zero_error = torch.zeros(self.n, self.J, device=dev)  # set_zero_error
        self.ptr = 0
        self.rows = torch.arange(self.n, device=dev)

    def set_params(self, values: dict, env_ids: torch.Tensor | slice = slice(None)) -> None:
        """Parameter values for the worlds `env_ids`: shared params (n,) or (n, 1), per-joint (n, J); numpy or torch.
        Writes armature and the gear joints (gear_stiffness = RIGID and a rigid link lock a gear joint)."""
        for p in PARAMS:
            v = torch.as_tensor(values[p.name], dtype=torch.float32, device=self.device)
            self.p[p.name][env_ids] = v.reshape(-1, self.J if p.per_joint else 1)
        rows = env_ids[:, None] if isinstance(env_ids, torch.Tensor) else env_ids
        m = self.model
        m.dof_armature[rows, self.servo_v] = self.p["armature"][env_ids].expand(-1, self.J)
        self._write_gears(env_ids)

    def set_link_stiffness(self, k_link: torch.Tensor, damping: torch.Tensor | float = 0.0,
                           env_ids: torch.Tensor | slice = slice(None)) -> None:
        """Stiffness [N m/rad] of the structure after each encoder, (n, J); inf = rigid (the default). `damping`
        [N m s/rad], (n, J): the structure's own, added to the gear joint's (which the servo loop no longer
        damps much once the encoder sees only part of the deflection)."""
        self.link_stiffness[env_ids] = torch.as_tensor(k_link, dtype=torch.float32, device=self.device)
        self.link_damping[env_ids] = torch.as_tensor(damping, dtype=torch.float32, device=self.device)
        self._write_gears(env_ids)

    def _write_gears(self, env_ids: torch.Tensor | slice) -> None:
        """The gear joint carries the gear and the link compliance in series; the encoder sees the gear's share."""
        rows = env_ids[:, None] if isinstance(env_ids, torch.Tensor) else env_ids
        m = self.model
        k_gear = self.p["gear_stiffness"][env_ids]
        c_gear = torch.where(k_gear > 0, 1.0 / k_gear, torch.zeros_like(k_gear))  # inf (rigid) or not yet set: 0
        c_link = 1.0 / self.link_stiffness[env_ids]
        c = c_gear + c_link
        rigid = c <= 0
        self.seen[env_ids] = torch.where(rigid, torch.ones_like(c), c_gear / c.clamp_min(1e-12))
        half = torch.where(rigid, torch.full_like(c, 1e-6), torch.full_like(c, GEAR_RANGE))
        m.jnt_range[rows, self.gear_jnt, 0] = -half
        m.jnt_range[rows, self.gear_jnt, 1] = half
        self.k_series[env_ids] = torch.where(rigid, torch.zeros_like(c), 1.0 / c.clamp_min(1e-12))
        m.jnt_stiffness[rows, self.gear_jnt] = self.k_series[env_ids]
        m.dof_damping[rows, self.gear_v] = self.p["gear_damping"][env_ids].expand(-1, self.J) + self.link_damping[env_ids]

    def set_play(self, play: torch.Tensor, env_ids: torch.Tensor | slice = slice(None)) -> None:
        """Free play [rad, total travel] in each joint's gears, (n, J); 0 = none (the default, as the model was
        fitted). Only for joints whose gear joint is compliant (not RIGID): the play is a dead zone of its spring."""
        self.play[env_ids] = torch.as_tensor(play, dtype=torch.float32, device=self.device)
        self.has_play = bool((self.play > 0).any())
        if not self.has_play:
            self.data.qfrc_applied[:, self.gear_v] = 0.0

    def _gear_seen(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gear joint angles (n, J) -> the part inside the free play (which the output encoder reads in full) and
        the elastic rest (of which it reads the gear's share)."""
        half = 0.5 * self.play
        inner = torch.clamp(g, -half, half)
        return inner, g - inner

    def set_zero_error(self, err: torch.Tensor, env_ids: torch.Tensor | slice = slice(None)) -> None:
        """Calibration error of each joint's zero [rad], (n, J): every angle that leaves or enters the servo (the
        encoder reading, the goal) is the true one plus this, as with a wrong offset in the joint calibration. The
        arm then is not where its kinematics say: a blade thought level leans."""
        self.zero_error[env_ids] = torch.as_tensor(err, dtype=torch.float32, device=self.device)

    def encoder(self) -> torch.Tensor:
        """What the servo encoders read: motor-side angle + the gears' free play + the gear's share of the elastic
        deflection (+ the zero's calibration error), (num_envs, J)."""
        q = self.data.qpos
        inner, rest = self._gear_seen(q[:, self.gear_q])
        return q[:, self.servo_q] + inner + self.seen * rest + self.zero_error

    def encoder_vel(self) -> torch.Tensor:
        v = self.data.qvel
        g = self.data.qpos[:, self.gear_q]
        share = torch.where(g.abs() < 0.5 * self.play, torch.ones_like(self.seen), self.seen)
        return v[:, self.servo_v] + share * v[:, self.gear_v]

    def teleport(self, q_enc: torch.Tensor, dq: torch.Tensor, env_ids: torch.Tensor | slice = slice(None)) -> None:
        """Set encoder angles/velocities with the gear springs relaxed, and sync the firmware state."""
        d = self.data
        rows = env_ids[:, None] if isinstance(env_ids, torch.Tensor) else env_ids
        d.qpos[rows, self.servo_q] = q_enc - self.zero_error[env_ids]
        d.qpos[rows, self.gear_q] = 0.0
        d.qvel[rows, self.servo_v] = dq
        d.qvel[rows, self.gear_v] = 0.0
        self.sync(q_enc, env_ids)

    def sync(self, q_enc: torch.Tensor, env_ids: torch.Tensor | slice = slice(None)) -> None:
        """Firmware state after a reset: target and encoder history = the current encoder angle."""
        self.target[env_ids] = q_enc
        self.hist[:, env_ids] = q_enc

    def set_target(self, q_enc: torch.Tensor, env_ids: torch.Tensor | slice = slice(None)) -> None:
        self.target[env_ids] = q_enc

    def apply(self, goal: torch.Tensor, dt: float) -> torch.Tensor:
        """One control step: goal (num_envs, J) encoder angles, already delayed. Writes the friction fields and
        returns the motor torques for the servo joints' actuators."""
        p, d, m = self.p, self.data, self.model
        self.ptr = (self.ptr + 1) % FEEDBACK_HISTORY
        self.hist[self.ptr] = self.encoder()
        lag = torch.round(p["feedback_delay"][:, 0] / dt).long()
        q_fb = self.hist[(self.ptr - lag) % FEEDBACK_HISTORY, self.rows]  # what the firmware sees
        dq = d.qvel[:, self.servo_v]
        step = p["max_velocity"] * dt
        self.target = torch.clamp(goal, self.target - step, self.target + step)
        duty = (self.target - q_fb) * self.kp * ERROR_GAIN * p["error_gain_ratio"]
        # Firmware current limit (Protection_Current), as in BAM's VoltageControlledActuator: keep
        # |I| = |duty * vin - kt * dq| / R <= max_current, then the physical PWM limit.
        back_emf = p["kt"] * dq
        span = p["R"] * p["max_current"]
        duty = torch.clamp(duty, (back_emf - span) / self.vin, (back_emf + span) / self.vin)
        duty = torch.clamp(duty, -MAX_PWM, MAX_PWM)
        kt_r = p["kt"] / p["R"]
        tau = kt_r * (self.vin * duty) - kt_r * back_emf
        if self.has_play:
            # Free play: cancel the gear joint's spring inside +-play/2 (it then acts on the excess only).
            half = 0.5 * self.play
            d.qfrc_applied[:, self.gear_v] = self.k_series * torch.clamp(d.qpos[:, self.gear_q], -half, half)

        # Friction budget (BAM m6), from the previous step's motor torque and the current external load.
        prev_tau = d.qfrc_actuator[:, self.servo_v]
        ext = -d.qfrc_bias[:, self.servo_v]
        s = p["stribeck_scale"]
        stribeck = torch.exp(-torch.pow(torch.abs(dq) / p["dtheta_stribeck"], M6_FIXED["alpha"]))
        gear = torch.abs(ext * p["load_friction_external"] - prev_tau * p["load_friction_motor"])
        gear_s = torch.abs(ext * (s * M6_FIXED["load_friction_external_stribeck"])
                           - prev_tau * (s * M6_FIXED["load_friction_motor_stribeck"]))
        abs_ext, abs_mot = torch.abs(ext), torch.abs(prev_tau)
        drive = (abs_mot > abs_ext).float()
        quad = s * (drive * M6_FIXED["load_friction_external_quad"] * abs_ext**2
                    + (1 - drive) * M6_FIXED["load_friction_motor_quad"] * abs_mot**2)
        fl = p["friction_base"] + stribeck * (s * M6_FIXED["friction_stribeck"] + gear_s + quad) + gear
        m.dof_frictionloss[:, self.servo_v] = fl * p["friction_scale"]
        m.dof_damping[:, self.servo_v] = (p["friction_viscous"] * p["friction_scale"])
        return tau


def bam_baseline(params_json: dict, num_joints: int) -> dict[str, np.ndarray]:
    """The pre-fit model: BAM params scaled to 12 V, rigid gears, no delays, no current limit, unit scales."""
    v = {k: params_json[k] for k in ("kt", "R", "error_gain_ratio", "max_velocity", "armature", "friction_base",
                                     "friction_viscous", "load_friction_external", "load_friction_motor",
                                     "dtheta_stribeck")}
    v.update(stribeck_scale=1.0, delay=0.0, gear_stiffness=np.full(num_joints, RIGID), gear_damping=0.05,
             friction_scale=np.ones(num_joints), feedback_delay=0.0, max_current=1e3)
    return v


PARAMS_BY_NAME = {p.name: p for p in PARAMS}
