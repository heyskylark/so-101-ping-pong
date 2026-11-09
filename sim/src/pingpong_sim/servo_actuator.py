"""mjlab actuator for the SO-101's STS3215 servos: the fitted servo model (servo_model.py) with domain randomization.

Fit: measurements/servo/fit_2026-11-08_221524 (assets/actuators/sts3215_12v_fit.json). With `randomize`, every
world draws its own servo at each episode reset, around the fit:

- `scale` params: fit x exp(U(-ln f, ln f)), f wider for parameters the recording identified poorly
  (sensitivity in the fit report) and for those the policy must not depend on;
- `range` params: U(lo, hi), `log_range`: log-uniform. Gear stiffness spans 80-600 N m/rad, from about the
  fit's stiffness down to where a neighbour's step moves a holding joint ~2x as much as on the real arm (the real
  arm hunts under hard commands, likely PLA flex the model doesn't have);
- command delay (host -> servo goal, fitted 8.7 ms) 4-20 ms: the deployed loop adds its own latency;
- supply voltage 11.5-12.3 V (logged range under load);
- a disturbance torque per joint: Ornstein-Uhlenbeck noise, std drawn per world, for the flex-driven hunting.

Joint targets are encoder angles (what the servos read and LeRobot reports). Policies must observe
`encoder_pos(robot)`, not the entity's joint_pos: the servo joints there are the motor side of the gear springs.

Per-world model fields: add the `servo_init` startup event to the env cfg (mjlab then expands
ServoModel.EXPANDED_FIELDS), or call sim.expand_model_fields(ServoModel.EXPANDED_FIELDS) without an env.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd, CommandField
from mjlab.managers.event_manager import requires_model_fields

from pingpong_sim.servo_model import GEAR_SUFFIX, PARAMS, ServoModel, motorize

if TYPE_CHECKING:
    from mjlab.entity import Entity

FIT_PARAMS = Path(__file__).resolve().parent / "assets" / "actuators" / "sts3215_12v_fit.json"

# name -> ("scale", f) | ("range", lo, hi) | ("log_range", lo, hi)
DEFAULT_RANDOMIZATION: dict[str, tuple] = {
    "kt": ("scale", 1.10),
    "R": ("scale", 1.15),
    "error_gain_ratio": ("scale", 1.10),
    "max_velocity": ("scale", 1.10),
    "armature": ("scale", 1.30),
    "friction_base": ("scale", 1.50),
    "friction_viscous": ("scale", 1.50),
    "load_friction_external": ("scale", 1.50),
    "load_friction_motor": ("scale", 1.20),  # sharp: the fit's cost rose 155 % at 10 % of its range
    "stribeck_scale": ("scale", 2.00),
    "dtheta_stribeck": ("scale", 1.50),
    "delay": ("range", 0.004, 0.020),
    "feedback_delay": ("range", 0.0, 0.006),
    "max_current": ("scale", 1.25),
    "gear_stiffness": ("log_range", 80.0, 600.0),
    "gear_damping": ("scale", 2.00),
    "friction_scale": ("scale", 1.30),
}


@dataclass(kw_only=True)
class ServoActuatorCfg(ActuatorCfg):
    """STS3215 servos on `target_names_expr` joints. The base class's delay_* fields stay 0: the command delay is
    a servo parameter here."""

    firmware_p: dict[str, float]
    params_path: Path = FIT_PARAMS
    randomize: bool = False
    randomization: dict[str, tuple] = field(default_factory=lambda: dict(DEFAULT_RANDOMIZATION))
    vin: float = 12.0
    vin_range: tuple[float, float] = (11.5, 12.3)
    disturbance_std_range: tuple[float, float] = (0.0, 0.08)  # N m
    disturbance_tau: float = 0.05  # s, correlation time (~3 Hz)

    def build(self, entity: Entity, target_ids: list[int], target_names: list[str]) -> ServoActuator:
        return ServoActuator(self, entity, target_ids, target_names)


@requires_model_fields(*ServoModel.EXPANDED_FIELDS)
def servo_init(env, env_ids=None) -> None:
    """Startup event: makes mjlab expand the servo model's per-world fields. The body is intentionally empty."""
    del env, env_ids


class ServoActuator(Actuator[ServoActuatorCfg]):
    def __init__(self, cfg: ServoActuatorCfg, entity: Entity, target_ids: list[int], target_names: list[str]):
        super().__init__(cfg, entity, target_ids, target_names)
        fit = json.loads(Path(cfg.params_path).read_text())
        order = fit["joints"]
        self._nominal = {}
        for p in PARAMS:
            v = fit[p.name]
            self._nominal[p.name] = [v[order.index(n)] for n in target_names] if p.per_joint else v
        self.servo: ServoModel | None = None

    @property
    def command_field(self) -> CommandField:
        return "position"

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        # The gear joints are part of the robot spec (robot.get_spec), so mjlab indexes them as entity joints.
        for name in target_names:
            if spec.joint(name + GEAR_SUFFIX) is None:
                raise ValueError(f"{name}{GEAR_SUFFIX} missing: add servo_model.add_gear_joints to the spec_fn")
        self._mjs_actuators = motorize(spec, target_names)

    def initialize(self, mj_model: mujoco.MjModel, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        n = data.nworld
        joint_ids = self.entity.indexing.joint_ids[self.target_ids].tolist()
        names = [mj_model.joint(i).name for i in joint_ids]
        self.servo = ServoModel(model, data, mj_model, names, [self.cfg.firmware_p[t] for t in self.target_names],
                                n, device)
        self._dt = float(mj_model.opt.timestep)
        self._device, self._n, J = device, n, len(names)
        dmax = self.cfg.randomization["delay"][2] if self.cfg.randomize else self._nominal["delay"]
        self._H = int(math.ceil(dmax / self._dt)) + 2
        self._goals = torch.zeros(self._H, n, J, device=device)
        self._ptr = 0
        self._lag = torch.zeros(n, dtype=torch.long, device=device)
        self._rows = torch.arange(n, device=device)
        self._dist = torch.zeros(n, J, device=device)
        self._dist_std = torch.zeros(n, 1, device=device)
        # Model fields are only written from compute(): initialize() runs before mjlab expands them per world.
        self._resample = torch.ones(n, dtype=torch.bool, device=device)
        self._pending = torch.ones(n, dtype=torch.bool, device=device)
        self._checked = False

    def _sample(self, env_ids: torch.Tensor) -> None:
        """Draw servo parameters (fit values when not randomizing) for env_ids."""
        k, J, dev = len(env_ids), self.servo.J, self._device
        values = {}
        for p in PARAMS:
            nom = torch.as_tensor(self._nominal[p.name], dtype=torch.float32, device=dev)
            shape = (k, J) if p.per_joint else (k, 1)
            nom = nom.expand(shape)
            rule = self.cfg.randomization.get(p.name) if self.cfg.randomize else None
            u = torch.rand(shape, device=dev)
            if rule is None:
                v = nom.clone()
            elif rule[0] == "scale":
                v = nom * torch.exp((2 * u - 1) * math.log(rule[1]))
            elif rule[0] == "range":
                v = rule[1] + (rule[2] - rule[1]) * u
            elif rule[0] == "log_range":
                v = rule[1] * (rule[2] / rule[1]) ** u
            else:
                raise ValueError(f"unknown randomization rule {rule} for {p.name}")
            values[p.name] = v
        self.servo.set_params(values, env_ids)
        self._lag[env_ids] = torch.round(values["delay"][:, 0] / self._dt).long().clamp(0, self._H - 1)
        if self.cfg.randomize:
            lo, hi = self.cfg.vin_range
            self.servo.vin[env_ids] = lo + (hi - lo) * torch.rand(k, 1, device=dev)
            lo, hi = self.cfg.disturbance_std_range
            self._dist_std[env_ids] = lo + (hi - lo) * torch.rand(k, 1, device=dev)
        else:
            self.servo.vin[env_ids] = self.cfg.vin
            self._dist_std[env_ids] = 0.0
        self._dist[env_ids] = 0.0

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if self.servo is None:
            return
        ids = self._rows[env_ids] if isinstance(env_ids, slice) else (self._rows if env_ids is None else env_ids)
        if self.cfg.randomize:
            self._resample[ids] = True
        # The reset joint state is written after this call: sync the firmware state at the next compute().
        self._pending[ids] = True

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        servo = self.servo
        if not self._checked:
            if self._n > 1 and any(getattr(servo.model, f).stride(0) == 0 for f in ServoModel.EXPANDED_FIELDS):
                raise RuntimeError("ServoActuator needs per-world model fields: add the servo_init startup event "
                                   "(pingpong_sim.servo_actuator.servo_init) or call "
                                   "sim.expand_model_fields(ServoModel.EXPANDED_FIELDS)")
            self._checked = True
        if self._resample.any():
            self._sample(self._resample.nonzero(as_tuple=True)[0])
            self._resample[:] = False
        if self._pending.any():
            ids = self._pending.nonzero(as_tuple=True)[0]
            servo.sync(servo.encoder()[ids], ids)
            self._goals[:, ids] = cmd.position_target[ids]
            self._pending[ids] = False
        self._ptr = (self._ptr + 1) % self._H
        self._goals[self._ptr] = cmd.position_target
        goal = self._goals[(self._ptr - self._lag) % self._H, self._rows]
        tau = servo.apply(goal, self._dt)
        if self.cfg.randomize:
            a = self._dt / self.cfg.disturbance_tau
            self._dist = (1 - a) * self._dist + self._dist_std * math.sqrt(2 * a) * torch.randn_like(self._dist)
            tau = tau + self._dist
        return tau


def _servo_actuator(entity: Entity) -> ServoActuator:
    acts = [a for a in entity.actuators if isinstance(a, ServoActuator)]
    if len(acts) != 1:
        raise ValueError(f"expected one ServoActuator on the entity, found {len(acts)}")
    return acts[0]


def encoder_pos(entity: Entity) -> torch.Tensor:
    """Encoder angles of the entity's servo joints (num_envs, J), in the actuator's target order."""
    return _servo_actuator(entity).servo.encoder()


def encoder_vel(entity: Entity) -> torch.Tensor:
    """Encoder velocities of the entity's servo joints (num_envs, J), in the actuator's target order."""
    return _servo_actuator(entity).servo.encoder_vel()
