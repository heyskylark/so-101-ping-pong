"""Bounce task for mjlab. Importing this package registers:

- `PingPong-Bounce-Residual`: actions correct the scripted bouncer's servo goals (elbow stroke).
- `PingPong-Bounce-Residual-Wrist`: the same with the scripted bouncer stroking with the wrist (wrist_flex).
- `PingPong-Bounce-Gentle`: residual on the pan-and-wrist-only bouncer, wrist camera only, small capped corrections,
  arm flex the encoders can't see (the real arm's task).
- `PingPong-Bounce-Gentle-Memory`: the same, and the policy sees what this rally's bounces did against what was
  predicted (`controller.BounceMemory`), to adapt to its arm's calibration errors while it plays.
- `PingPong-Bounce-Gentle-Carry`: the gentle task with a bouncer that also carries the blade under the ball forward
  and back (lift, elbow and wrist shift the hold pose by up to 4 cm between bounces; the stroke stays the wrist's).
- `PingPong-Bounce-Gentle-Lead`: the memory task with the policy allowed twice as far from the bouncer's goal and
  paid for the ball's energy at every step: it is to hit where the bouncer plans nothing (short hops, a dead ball).
- `PingPong-Bounce-Gentle-Rough`: the lead task with a new restitution for every bounce, so the next impact can't be
  timed from the last one (on the rig the committed hit time is off by -41..+38 ms).
- `PingPong-Bounce-Solo`: no scripted bouncer: the policy's actions are the servo goals, in the real arm's sim (gear
  play, link flex, zero errors, wrist camera, a new restitution every bounce), rate-limited and charged for jerk.
- `PingPong-Bounce-Direct`: actions are servo goals relative to the ready pose (comparison).
"""

from mjlab.tasks.registry import register_mjlab_task

from pingpong_sim.task.env_cfg import make_env_cfg, make_ppo_cfg

register_mjlab_task(
    task_id="PingPong-Bounce-Residual",
    env_cfg=make_env_cfg(residual=True),
    play_env_cfg=make_env_cfg(residual=True, play=True),
    rl_cfg=make_ppo_cfg(residual=True),
)
register_mjlab_task(
    task_id="PingPong-Bounce-Residual-Wrist",
    env_cfg=make_env_cfg(residual=True, stroke_joint="wrist_flex"),
    play_env_cfg=make_env_cfg(residual=True, play=True, stroke_joint="wrist_flex"),
    rl_cfg=make_ppo_cfg(residual=True, experiment_name="pingpong_residual_wrist"),
)
register_mjlab_task(
    task_id="PingPong-Bounce-Gentle",
    env_cfg=make_env_cfg(gentle=True),
    play_env_cfg=make_env_cfg(gentle=True, play=True),
    rl_cfg=make_ppo_cfg(residual=True, experiment_name="pingpong_gentle"),
)
register_mjlab_task(
    task_id="PingPong-Bounce-Gentle-Memory",
    env_cfg=make_env_cfg(gentle=True, memory=True),
    play_env_cfg=make_env_cfg(gentle=True, memory=True, play=True),
    rl_cfg=make_ppo_cfg(residual=True, experiment_name="pingpong_gentle_memory"),
)
register_mjlab_task(
    task_id="PingPong-Bounce-Gentle-Carry",
    env_cfg=make_env_cfg(gentle=True, carry=True),
    play_env_cfg=make_env_cfg(gentle=True, carry=True, play=True),
    rl_cfg=make_ppo_cfg(residual=True, experiment_name="pingpong_gentle_carry"),
)
register_mjlab_task(
    task_id="PingPong-Bounce-Gentle-Lead",
    env_cfg=make_env_cfg(gentle=True, memory=True, lead=True),
    play_env_cfg=make_env_cfg(gentle=True, memory=True, lead=True, play=True),
    rl_cfg=make_ppo_cfg(residual=True, experiment_name="pingpong_gentle_lead"),
)
register_mjlab_task(
    task_id="PingPong-Bounce-Gentle-Rough",
    env_cfg=make_env_cfg(gentle=True, memory=True, lead=True, rough=True),
    play_env_cfg=make_env_cfg(gentle=True, memory=True, lead=True, rough=True, play=True),
    rl_cfg=make_ppo_cfg(residual=True, experiment_name="pingpong_gentle_rough"),
)
register_mjlab_task(
    task_id="PingPong-Bounce-Solo",
    env_cfg=make_env_cfg(solo=True),
    play_env_cfg=make_env_cfg(solo=True, play=True),
    rl_cfg=make_ppo_cfg(residual=False, experiment_name="pingpong_solo", init_std=0.3),
)
register_mjlab_task(
    task_id="PingPong-Bounce-Direct",
    env_cfg=make_env_cfg(residual=False),
    play_env_cfg=make_env_cfg(residual=False, play=True),
    rl_cfg=make_ppo_cfg(residual=False),
)
