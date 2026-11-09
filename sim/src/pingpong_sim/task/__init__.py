"""Bounce task for mjlab. Importing this package registers:

- `PingPong-Bounce-Residual`: actions correct the scripted bouncer's servo goals.
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
    task_id="PingPong-Bounce-Direct",
    env_cfg=make_env_cfg(residual=False),
    play_env_cfg=make_env_cfg(residual=False, play=True),
    rl_cfg=make_ppo_cfg(residual=False),
)
