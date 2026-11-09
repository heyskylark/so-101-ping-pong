"""Train a bounce policy: mjlab's train CLI with this repo's tasks registered.

Run (from sim/):
  uv run scripts/train.py PingPong-Bounce-Residual --env.scene.num-envs 4096
  uv run scripts/train.py PingPong-Bounce-Direct --env.scene.num-envs 4096
Logs and checkpoints: logs/rsl_rl/<experiment>/<time>/  (uv run tensorboard --logdir logs/rsl_rl)
"""

import pingpong_sim.task  # noqa: F401  (registers the tasks)
from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
