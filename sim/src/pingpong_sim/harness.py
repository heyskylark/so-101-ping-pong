"""Scene + MuJoCo Warp simulation without an RL env, for model checks and system identification scripts."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from pingpong_sim.ball import SIM_DT, get_ball_cfg, scene_spec_fn
from pingpong_sim.robot import get_robot_cfg
from pingpong_sim.servo_model import ServoModel


@dataclass
class Harness:
    scene: Scene
    sim: Simulation
    robot: Entity
    ball: Entity
    mj_model: mujoco.MjModel
    dt: float


def make_harness(num_envs: int = 1, dt: float = SIM_DT, device: str | None = None, randomize: bool = False) -> Harness:
    """`randomize`: each world draws its own servos (servo_actuator.DEFAULT_RANDOMIZATION) at every reset."""
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    with warnings.catch_warnings():
        # Menagerie's <option> isn't propagated by MjSpec.attach; the options come from MujocoCfg below.
        warnings.filterwarnings("ignore", message=".*non-default <option> fields.*")
        scene = Scene(
            SceneCfg(num_envs=num_envs, env_spacing=1.0,
                     entities={"robot": get_robot_cfg(randomize), "ball": get_ball_cfg()},
                     spec_fn=scene_spec_fn),
            device,
        )
    mj_model = scene.compile()
    sim = Simulation(
        num_envs=num_envs,
        # Euler: MuJoCo Warp 3.7 has no fluid model with implicit integrators.
        cfg=SimulationCfg(mujoco=MujocoCfg(timestep=dt, integrator="euler", cone="elliptic", impratio=10.0)),
        model=mj_model,
        device=device,
    )
    scene.initialize(sim.mj_model, sim.model, sim.data)
    # The servo model writes per-world parameters and friction into these fields.
    sim.expand_model_fields(ServoModel.EXPANDED_FIELDS)
    sim.reset()
    scene.reset()
    return Harness(scene, sim, scene["robot"], scene["ball"], mj_model, dt)


def step(h: Harness, n: int = 1) -> None:
    for _ in range(n):
        h.scene.write_data_to_sim()
        h.sim.step()
        h.scene.update(dt=h.dt)
