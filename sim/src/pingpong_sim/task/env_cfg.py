"""Env and PPO configs of the bounce task (design: TODO.md §3 "Task cfg").

100 Hz policy (10 x 1 ms physics), 20 s episodes. Rewards are per event (bounce) or per step, not scaled by dt.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mj_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from pingpong_sim.ball import SIM_DT, get_ball_cfg, scene_spec_fn
from pingpong_sim.robot import ARM_JOINTS, get_robot_cfg
from pingpong_sim.servo_actuator import servo_init
from pingpong_sim.task import mdp
from pingpong_sim.task.state import TaskCfg

CONTROL_DT = 0.01
DECIMATION = round(CONTROL_DT / SIM_DT)


def make_env_cfg(residual: bool = True, play: bool = False) -> ManagerBasedRlEnvCfg:
    """`residual`: actions correct the scripted bouncer's goal (else: offsets from the ready pose)."""
    task = TaskCfg(use_bouncer=residual)

    def actor_terms(noisy: bool) -> dict[str, ObservationTermCfg]:
        noise = (lambda n: Unoise(n_min=-n, n_max=n)) if noisy else (lambda n: None)  # noqa: E731
        terms = {
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos, noise=noise(0.002), history_length=3),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel, noise=noise(0.05), history_length=3),
            "paddle": ObservationTermCfg(func=mdp.paddle),
            "ball": ObservationTermCfg(func=mdp.ball_estimate),
            "impact": ObservationTermCfg(func=mdp.impact_prediction),
            "actions": ObservationTermCfg(func=mj_mdp.last_action, history_length=5),
            # 0.64 s of ball / paddle / predicted-impact frames: how the last bounce went vs what was predicted.
            "history": ObservationTermCfg(func=mdp.history),
        }
        if residual:
            terms["bouncer"] = ObservationTermCfg(func=mdp.bouncer_command)
        return terms

    critic_terms = {
        **actor_terms(noisy=False),
        "ball_true": ObservationTermCfg(func=mdp.ball_true),
        "paddle_true": ObservationTermCfg(func=mdp.paddle_true),
        "mismatch": ObservationTermCfg(func=mdp.mismatch),
        "launch": ObservationTermCfg(func=mdp.launch_state),
    }
    observations = {
        "actor": ObservationGroupCfg(actor_terms(noisy=True), enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    actions = {"arm": mdp.BounceActionCfg(scale=0.1 if residual else 0.5, residual=residual)}

    events = {
        "servo_init": EventTermCfg(func=servo_init, mode="startup"),
        # Masses once per world (set_const recompute; 4096 worlds cover the range): ball 2.5-3.0 g, paddle +-5 %.
        "ball_mass": EventTermCfg(func=dr.pseudo_inertia, mode="startup",
                                  params={"alpha_range": (-0.045, 0.045),
                                          "asset_cfg": SceneEntityCfg("ball", body_names=("ball",))}),
        "paddle_mass": EventTermCfg(func=dr.pseudo_inertia, mode="startup",
                                    params={"alpha_range": (-0.025, 0.025),
                                            "asset_cfg": SceneEntityCfg("robot", body_names=("paddle",))}),
        "reset_episode": EventTermCfg(func=mdp.reset_episode, mode="reset", params={"task": task}),
        # Restitution ~0.50-0.65 per episode.
        "ball_contact": EventTermCfg(func=mdp.randomize_ball_contact, mode="reset",
                                     params={"dampratio": (0.155, 0.235)}),
        "hold_ball": EventTermCfg(func=mdp.hold_ball, mode="step"),
        "push_ball": EventTermCfg(func=mdp.push_ball, mode="interval", interval_range_s=(1.0, 4.0),
                                  params={"dv": 0.1}),
    }

    # Per-bounce rewards are weighted by the apex (std 0.07 m around the 0.15 m target): a 5 cm dribble earns ~0.13
    # of a target bounce, so ~5 dribbles/s pay ~1/4 of ~2.9 target bounces/s.
    rewards = {
        "bounce": RewardTermCfg(func=mdp.bounce, weight=1.0, params={"apex_std": 0.07}),
        "bounce_apex": RewardTermCfg(func=mdp.bounce_apex, weight=1.0, params={"std": 0.035}),
        "bounce_landing": RewardTermCfg(func=mdp.bounce_landing, weight=1.0, params={"std": 0.05, "apex_std": 0.07}),
        "bounce_centered": RewardTermCfg(func=mdp.bounce_centered, weight=0.5, params={"std": 0.04, "apex_std": 0.07}),
        "action_rate": RewardTermCfg(func=mj_mdp.action_rate_l2, weight=-0.01),
        "action": RewardTermCfg(func=mdp.action_l2, weight=-0.005 if residual else 0.0),
        "joint_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-4),
        "joint_limits": RewardTermCfg(func=mj_mdp.joint_pos_limits, weight=-1.0,
                                      params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)}),
        "failure": RewardTermCfg(func=mj_mdp.is_terminated, weight=-5.0),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mj_mdp.time_out, time_out=True),
        "ball_dropped": TerminationTermCfg(func=mdp.ball_dropped, params={"depth": 0.10}),
        "ball_away": TerminationTermCfg(func=mdp.ball_away, params={"radius": 0.20}),
        "ball_hit_robot": TerminationTermCfg(func=mdp.ball_hit_robot, params={"sensor_name": "ball_robot"}),
        "paddle_tilted": TerminationTermCfg(func=mdp.paddle_tilted, params={"max_deg": 35.0}),
    }

    metrics = {
        "bounces": MetricsTermCfg(func=mdp.bounces, reduce="last"),
        "tracking_error": MetricsTermCfg(func=mdp.tracking_error),
        "tracked": MetricsTermCfg(func=mdp.tracked),
    }

    # Ball touching any arm body except the paddle (the blade contact is the explicit pair): one sensor per arm body.
    ball_robot = ContactSensorCfg(
        name="ball_robot",
        primary=ContactMatch(mode="body", pattern=".*", entity="robot", exclude=("paddle",)),
        secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
        fields=("found",),
        reduce="none",
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1,
            env_spacing=1.0,
            entities={"robot": get_robot_cfg(randomize=True), "ball": get_ball_cfg()},
            sensors=(ball_robot,),
            spec_fn=scene_spec_fn,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.WORLD, lookat=(0.35, 0.0, 0.32), distance=1.1,
                            elevation=-15.0, azimuth=140.0),
        # Euler: MuJoCo Warp 3.7 has no fluid model with implicit integrators.
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=SIM_DT, integrator="euler", cone="elliptic", impratio=10.0)),
        decimation=DECIMATION,
        episode_length_s=20.0,
        scale_rewards_by_dt=False,
    )
    if play:
        cfg.scene = replace(cfg.scene, num_envs=16)
    return cfg


def make_ppo_cfg(residual: bool = True) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            # Residual: start close to the scripted bouncer (0.3 x 0.1 rad exploration noise).
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3 if residual else 0.5,
                              "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.002,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="pingpong_residual" if residual else "pingpong_direct",
        logger="tensorboard",
        save_interval=100,
        num_steps_per_env=32,
        max_iterations=3000,
    )
