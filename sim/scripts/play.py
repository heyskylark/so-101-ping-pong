"""Watch a bounce policy: mjlab's play CLI (interactive viewer) with this repo's tasks registered.

Run (from sim/):
  uv run scripts/play.py PingPong-Bounce-Residual --checkpoint-file logs/rsl_rl/pingpong_residual/<run>/model_1000.pt
  uv run scripts/play.py PingPong-Bounce-Residual --agent zero     # the scripted bouncer alone
"""

import pingpong_sim.task  # noqa: F401  (registers the tasks)
from mjlab.scripts.play import main

if __name__ == "__main__":
    main()
