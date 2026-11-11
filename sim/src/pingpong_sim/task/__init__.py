"""Bounce task for mjlab. Importing this package registers:

- `PingPong-Bounce-Residual`: actions correct the scripted bouncer's servo goals (elbow stroke).
- `PingPong-Bounce-Residual-Wrist`: the same with the scripted bouncer stroking with the wrist (wrist_flex).
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
    task_id="PingPong-Bounce-Direct",
    env_cfg=make_env_cfg(residual=False),
    play_env_cfg=make_env_cfg(residual=False, play=True),
    rl_cfg=make_ppo_cfg(residual=False),
)
