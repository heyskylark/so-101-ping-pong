"""Env and PPO configs of the bounce task (design: TODO.md §3 "Task cfg").

100 Hz policy (10 x 1 ms physics), 20 s episodes. Rewards are per event (bounce) or per step, not scaled by dt.
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mj_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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
from pingpong_sim.camera_sim import WRIST
from pingpong_sim.robot import ARM_JOINTS, get_robot_cfg
from pingpong_sim.servo_actuator import servo_init
from pingpong_sim.task import mdp
from pingpong_sim.task.state import TaskCfg

CONTROL_DT = 0.01
DECIMATION = round(CONTROL_DT / SIM_DT)
# deg/s per joint: how fast run_robot.py lets a servo goal move on the rig (--goal-rate), and the solo task's goals.
SOLO_GOAL_RATE = (400.0, 200.0, 300.0, 860.0, 600.0)


def make_env_cfg(residual: bool = True, play: bool = False, stroke_joint: str = "elbow_flex",
                 gentle: bool = False, memory: bool = False, carry: bool = False,
                 lead: bool = False, rough: bool = False, solo: bool = False) -> ManagerBasedRlEnvCfg:
    """`residual`: actions correct the scripted bouncer's goal (else: offsets from the ready pose). `stroke_joint`:
    the one joint the scripted bouncer strokes with (the elbow, or the wrist: a third of the inertia to swing).
    `gentle` (residual): the bouncer aims and strokes with the pan and the wrist only while lift and elbow hold
    (BouncerCfg.wrist_only), the wrist camera alone tracks the ball, the policy's corrections are small and capped,
    and the arm's structure flexes unseen by the encoders. For the real arm, whose printed links can't take the
    full-arm bouncer's 40-70 deg swings between bounces. `memory`: the actor also sees the bounce memory, and the
    critic the joint zeros' true errors. `carry` (gentle): the bouncer also shifts its hold pose along x under the
    ball (BouncerCfg.reach_x). `lead` (gentle): the policy may move twice as far from the bouncer's goal and is
    paid for the ball's energy at every step, to hit where the bouncer plans nothing. `rough` (gentle): every bounce
    has its own restitution. `solo`: no bouncer at all: the policy's actions are the servo goals (offsets from the
    ready pose, rate-limited as on the rig), in the gentle task's arm and cameras with the rough bounces, paid for
    the ball's energy and charged for jerky goals and fast joints."""
    if solo:
        # No bouncer; the real arm's sim with everything learned about it; a new restitution per bounce. Not the
        # bounce memory: trained from scratch its inputs are zero until rallies happen and the observation
        # normalizer then feeds the first records in at 100 sigma (the 2026-11-13 memory run); it can be added to
        # a trained policy with scripts/warm_start.py.
        residual, gentle, rough = False, True, True
    task = TaskCfg(use_bouncer=residual)
    task.bouncer = replace(task.bouncer, stroke_joint=ARM_JOINTS.index(stroke_joint))
    if solo:
        task.drop_offset = 0.04  # hand drops on the rig land -27..+34 mm along the blade, -45..0 mm across
    if gentle:
        # No tilt trim: on the rig it chased mistimed hits; the policy corrects instead. The bouncer and the filter
        # believe the restitution measured on the arm.
        task.bouncer = replace(task.bouncer, wrist_only=True, stroke_joint=ARM_JOINTS.index("wrist_flex"), reach=0.02,
                               trim_gain=0.0, restitution=0.62)
        task.filter = replace(task.filter, restitution=0.62)
        # One camera: in simulation either camera alone tracks better than the two together.
        task.cameras = (replace(WRIST, stamp_bias_ms=(-15.0, 15.0)),)
        # The wrist camera sees ~0.2 m above the blade; the rig's drops arrive at 2.2-2.9 m/s (0.25-0.43 m).
        task.drop_height = (0.10, 0.45)
        if carry:
            # The wrist's stroke turns the face at 60-100 deg/s, so 10-15 ms of timing error is 1 deg of lean and
            # every bounce leaves a few cm forward or back of where it was sent; rig rallies end off the blade's
            # front or back (32 of 39, 2026-11-13). Sideways the pan follows the ball; this is the same along x.
            task.bouncer = replace(task.bouncer, reach_x=0.04)

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
        if memory:
            # What this rally's bounces did against what the filter predicted (BounceMemory): the arm's own
            # calibration errors show there, bounce after bounce.
            terms["memory"] = ObservationTermCfg(func=mdp.memory)
        return terms

    critic_terms = {
        **actor_terms(noisy=False),
        "ball_true": ObservationTermCfg(func=mdp.ball_true),
        "paddle_true": ObservationTermCfg(func=mdp.paddle_true),
        "mismatch": ObservationTermCfg(func=mdp.mismatch),
        "launch": ObservationTermCfg(func=mdp.launch_state),
    }
    if memory or solo:
        critic_terms["zero_error"] = ObservationTermCfg(func=mdp.zero_error)
    observations = {
        "actor": ObservationGroupCfg(actor_terms(noisy=True), enable_corruption=True),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    if solo:
        # The whole goal is the policy's: offsets from the ready pose up to +-26 deg on the pan, +-12 deg on the
        # lift, +-15 deg on the elbow, +-34 deg on the wrist and +-17 deg on the roll (the lift's range moves the
        # level blade +-4 cm; the full-arm bouncer's 40-70 deg swings wore out two servos), and no faster than the
        # rig's --goal-rate lets a goal move.
        actions = {"arm": mdp.BounceActionCfg(scale=(0.15, 0.07, 0.09, 0.2, 0.1), clip=3.0, residual=False,
                                              max_rate=tuple(math.radians(r) for r in SOLO_GOAL_RATE))}
    elif gentle:
        # Small, capped corrections: at most +-5.7 deg on the pan, +-3.4 deg on lift and elbow, +-11.5 deg on the
        # wrist joints, so the policy can't bring the big swings back. `lead`: twice that (+-23 deg on the wrist:
        # +-10 cm at the blade centre, a stroke of its own); the raw action keeps its meaning, so a policy trained
        # with the smaller cap can be continued.
        actions = {"arm": mdp.BounceActionCfg(scale=(0.05, 0.03, 0.03, 0.1, 0.1), clip=4.0 if lead else 2.0,
                                              residual=True)}
    else:
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
        # Restitution ~0.61-0.76 per episode (black rubber, nominal 0.69).
        "ball_contact": EventTermCfg(func=mdp.randomize_ball_contact, mode="reset",
                                     params={"dampratio": (0.105, 0.17)}),
        "hold_ball": EventTermCfg(func=mdp.hold_ball, mode="step"),
        "push_ball": EventTermCfg(func=mdp.push_ball, mode="interval", interval_range_s=(1.0, 4.0),
                                  params={"dv": 0.1}),
    }
    if gentle:
        # Bending after the encoders [N m/rad] (printed links, horn screws, paddle holder), on the three joints that
        # carry the paddle's weight. Wrist, measured on the rig 2026-11-12 with the holder tight: the blade hangs
        # 1.6 deg further forward than the encoders say (ruler) and follows them ~14 ms late in strokes (replayed
        # runs); the sim gives 1.7 deg / 6 ms at 15 N m/rad and 3.7 deg / 16 ms at 6.5 (run_robot.py --sim
        # --sim-flex wrist_flex=K), where the scripted bouncer also drops to the rig's 2-3 bounces per drop. The
        # range spans both readings. Lift and elbow: not measured; the damping ratio is a guess for bolted PLA.
        # Free play in the gears [deg, total travel]: on the rig the lift jumps ~1.5 deg and the elbow ~1.1 deg
        # under each wrist stroke's reaction (tune_hold.py, P 32); the sim does that with 0.9 deg of play on the
        # lift and 2.0 on the elbow, gears of ~100-200 N m/rad and 7 ms of feedback delay (fit_hold.py), and 0.4 /
        # 0.2 deg without play. The other joints' play is the user's hand test ("similar"), not measured.
        # Sideways give (pan axis), from rocking the blade by hand on the held arm (arm_pose.py --play,
        # 2026-11-13): the tip moved ~64 mm by eye for 22 mm by the encoders, under a push the pan servo's own
        # give puts at ~1.5 N: ~10-20 N m/rad unseen about the pan axis (rough: a tape measure held in the air).
        # Up and down the same test gives ~30 N m/rad at the wrist, inside the range above.
        # Calibration errors of the joint zeros [deg, +-]: on the rig the tag fit left the blade leaning 1.5-2.3
        # deg sideways (ruler, first bounces; 2026-11-13) and its roll zero has read +1.7, +10.1 and +12.6 deg in
        # three fits. A policy trained without them (gentle_v2) falls from 12 bounces per drop to 2-3 at 2-3 deg
        # of roll or wrist error in the sim, which is what it did on the rig.
        events["arm_flex"] = EventTermCfg(
            func=mdp.randomize_arm_flex, mode="reset",
            params={"stiffness": {"shoulder_pan": (6.0, 40.0), "shoulder_lift": (300.0, 3000.0),
                                  "elbow_flex": (200.0, 2000.0), "wrist_flex": (5.0, 40.0)},
                    "damping_ratio": (0.03, 0.10),
                    "play_deg": {"shoulder_pan": (0.0, 1.2), "shoulder_lift": (0.6, 2.2), "elbow_flex": (1.5, 4.0),
                                 "wrist_flex": (0.0, 1.0), "wrist_roll": (0.0, 1.0)},
                    "zero_error_deg": {"shoulder_pan": 3.0, "shoulder_lift": 1.5, "elbow_flex": 1.5,
                                       "wrist_flex": 2.5, "wrist_roll": 3.0}})
        # On the arm-held paddle a bounce returns 0.57-0.64 of the speed (flight times, 2026-11-12), part of it the
        # soft wrist's doing: contact restitution ~0.53-0.72.
        events["ball_contact"].params["dampratio"] = (0.12, 0.22)
        if rough:
            # A new restitution every 0.15-0.3 s, so each bounce has its own (~0.47-0.78): on the rig what a bounce
            # returns is not what the filter's fixed restitution expects, the predicted next impact is off by
            # -41..+38 ms when the stroke is committed (phone video, 2026-11-13; sim -17..+21 ms), and the hits
            # off by more than 30 ms end the rallies. With the draw (0.08, 0.30) the sim's committed hit time is off
            # by -26..+43 ms and `gentle_memory_v1` falls from 172 bounces per drop to 3.9 (the rig: 2.5); with
            # (0.10, 0.25): -28..+26 ms, 4.2.
            events["bounce_contact"] = EventTermCfg(func=mdp.randomize_ball_contact, mode="interval",
                                                    interval_range_s=(0.15, 0.3), params={"dampratio": (0.09, 0.28)})

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
    if lead:
        # The model is to keep the ball going where the planner has no stroke (a hop shorter than its ~0.25 s; a
        # ball that died on the blade: 6 of 12 rig rallies ended so, 2026-11-13): every step pays for the ball's
        # vertical energy over the blade, up to the target bounce's; ~1/5 of what target bounces pay. Leaving the
        # planner's goal costs a fifth of what it did.
        rewards["ball_energy"] = RewardTermCfg(func=mdp.ball_energy, weight=0.02, params={"radius": 0.075})
        rewards["action"] = RewardTermCfg(func=mdp.action_l2, weight=-0.001)
    curriculum = {}
    if solo:
        # Without a scripted stroke the ball's energy is the only reward a first passive bounce can lead on from.
        # Smooth: a jerking goal and fast joints cost, the lift and the elbow (whose gears wear first, and which
        # carry the arm) four times the wrist. The cost comes in three stages (iterations 0, 1200, 2400 at 32 steps
        # each): charged in full from the start, a policy that can't juggle yet learns to hold still (random
        # actions cost 2.7 per second at the last stage's weight, the ball's energy pays at most 2). At the last
        # stage a wrist stroke per bounce costs ~3 % of what steady target bounces pay, a lift moving at 30 deg/s
        # as much again.
        rewards["ball_energy"] = RewardTermCfg(func=mdp.ball_energy, weight=0.02, params={"radius": 0.075})
        rewards["action_accel"] = RewardTermCfg(func=mdp.action_accel_l2, weight=-0.0005)
        rewards["joint_vel"] = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.0002,
                                             params={"weights": (1.0, 4.0, 4.0, 1.0, 1.0)})
        stage = lambda *w: [{"step": 32 * it, "weight": x} for it, x in zip((0, 1200, 2400), w)]  # noqa: E731
        curriculum = {
            "smooth_goals": CurriculumTermCfg(func=mj_mdp.reward_curriculum, params={
                "reward_name": "action_accel", "stages": stage(-0.0005, -0.002, -0.005)}),
            "slow_joints": CurriculumTermCfg(func=mj_mdp.reward_curriculum, params={
                "reward_name": "joint_vel", "stages": stage(-0.0002, -0.001, -0.003)}),
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
    if solo:
        cfg.curriculum = curriculum
    if play:
        cfg.scene = replace(cfg.scene, num_envs=16)
    return cfg


def make_ppo_cfg(residual: bool = True, experiment_name: str | None = None,
                 init_std: float | None = None) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            # Residual: start close to the scripted bouncer (0.3 x 0.1 rad exploration noise).
            distribution_cfg={"class_name": "GaussianDistribution",
                              "init_std": init_std if init_std is not None else (0.3 if residual else 0.5),
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
        experiment_name=experiment_name or ("pingpong_residual" if residual else "pingpong_direct"),
        logger="tensorboard",
        save_interval=100,
        num_steps_per_env=32,
        max_iterations=3000,
    )
