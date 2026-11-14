"""How the real arm executed the bouncer's dry strokes (`record_servos.py strokes`): the blade's motion from the
encoders against the plan, around each planned impact.

  uv run scripts/analyze_strokes.py 2026-11-12_114800_strokes [more recordings under measurements/servo]

Per recording, medians over the strokes planned at >= 0.3 m/s, at -20 .. +20 ms from the planned impact (the ball
arrives within about +-20 ms of it): the blade's speed along the planned face normal / the planned impact speed,
its pitch and its height relative to the planned impact pose. The plan's speed is a triangle peaking at the impact
(constant acceleration, then the follow-through), so the speed at the ball's contact depends on the timing.

2026-11-12, screws locked, the sim's tracking model: 0.96 / 1.07 / 1.13 / 0.87 / 0.31 of the planned speed at
-20 .. +20 ms, level and on height at the impact (+-1.3 deg, +-9 mm at +-20 ms). Less feedforward lead (0.7) made
the blade 10 ms late (0.8 deg tipped forward, 5 mm low); sending the goals 15 ms ahead as well put the speed peak
20 ms early (0.5 of the planned speed at the impact). Neither was kept.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pingpong_sim.real.arm import RATE, ArmModel

REPO = Path(__file__).resolve().parents[2]
OFFSETS = np.arange(-2, 3)  # samples (10 ms) around the planned impact


def profile(run: str, model: ArmModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    """Medians over the strokes at OFFSETS: speed / planned impact speed, pitch error [deg], height error [m]."""
    d = np.load(REPO / "measurements" / "servo" / run / "log.npz", allow_pickle=True)
    meta = json.loads((REPO / "measurements" / "servo" / run / "meta.json").read_text())
    t = d["t_read"]

    def face(Q):
        F, N, P = [], [], []
        for x in Q:
            f, p = model.face_pose(x)
            F.append(f)
            P.append(p)
            N.append(model.d.site_xmat[model.face].reshape(3, 3)[:, 2].copy())
        return np.array(F), np.array(N), np.array(P)

    Fp, Np, Pp = face(d["plan_q"])
    Fa, _, Pa = face(d["pos"])
    vp = (np.gradient(Fp, t, axis=0) * Np).sum(1)
    va = (np.gradient(Fa, t, axis=0) * Np).sum(1)
    speed, pitch, height = [], [], []
    for ti in d["impacts"]:
        i = int(round(ti * RATE))  # log rows are the plan's samples
        if i < 20 or i > len(t) - 20:
            continue
        s = vp[i - 1]  # the stroke's last sample: the planned impact speed (the follow-through starts at i)
        if s < 0.3:
            continue
        speed.append(va[i + OFFSETS] / s)
        pitch.append(Pa[i + OFFSETS] - Pp[i])
        height.append(((Fa[i + OFFSETS] - Fp[i]) * Np[i]).sum(1))
    return np.median(speed, 0), np.median(pitch, 0), np.median(height, 0), len(speed), meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="directory names under measurements/servo")
    args = ap.parse_args()
    model = ArmModel()
    for run in args.runs:
        speed, pitch, height, n, meta = profile(run, model)
        print(f"{run}: {n} strokes planned at >= 0.3 m/s, {meta.get('timing')}")
        print("  time from the planned impact [ms]  " + "".join(f"{10 * o:+7d}" for o in OFFSETS))
        print("  blade speed / planned impact speed " + "".join(f"{x:7.2f}" for x in speed))
        print("  pitch, + = tip up [deg]            " + "".join(f"{x:+7.1f}" for x in pitch))
        print("  height [mm]                        " + "".join(f"{1e3 * x:+7.1f}" for x in height))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
