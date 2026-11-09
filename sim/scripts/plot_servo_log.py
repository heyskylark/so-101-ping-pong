"""Plot a servo recording (record_servos.py output): goal vs measured position per joint, one figure per pose.

Run (from sim/): uv run scripts/plot_servo_log.py <recording dir>   -> <dir>/plot_<pose>.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    d = Path(sys.argv[1])
    log = np.load(d / "log.npz")
    meta = json.loads((d / "meta.json").read_text())
    joints, segs = meta["joints"], meta["segments"]
    t, seg = log["t_read"], log["seg"]
    for pose in dict.fromkeys(s.get("pose") for s in segs if s.get("pose")):
        if not np.isin(seg, [k for k, s in enumerate(segs) if s.get("pose") == pose and s.get("joint")]).any():
            continue
        fig, axes = plt.subplots(len(joints), 1, figsize=(14, 2.4 * len(joints)), sharex=False)
        for i, (ax, j) in enumerate(zip(axes, joints)):
            idx = [k for k, s in enumerate(segs) if s.get("pose") == pose and s.get("joint") == j]
            sel = np.isin(seg, idx)
            if not sel.any():
                continue
            tt = t[sel]
            ax.plot(tt, np.degrees(log["goal"][sel, i]), lw=0.8, label="goal")
            ax.plot(tt, np.degrees(log["pos"][sel, i]), lw=0.8, label="measured")
            ax2 = ax.twinx()
            ax2.plot(tt, log["load"][sel, i] / 10, lw=0.5, color="0.6")
            ax2.set_ylabel("load %", color="0.5")
            ax.set_ylabel(f"{j} [deg]")
            ax.legend(loc="upper left", fontsize=7)
        axes[0].set_title(f"{d.name} ({meta['backend']}): pose {pose}")
        axes[-1].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(d / f"plot_{pose}.png", dpi=90)
        plt.close(fig)
        print("wrote", d / f"plot_{pose}.png")


if __name__ == "__main__":
    main()
