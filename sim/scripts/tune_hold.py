"""How steadily the shoulder lift and elbow hold with other firmware gains (the real arm).

  uv run scripts/tune_hold.py --port /dev/ttyACM0                       # ~15 s per setting, you watching
  uv run scripts/tune_hold.py --port /dev/ttyACM0 --settings 64:0 32:16 16:32

With P only (this project's gains: 64 on both joints, D = 0, what the simulator models) the two joints that carry
the arm hunt at 3-4 Hz around their goal once something has moved them: a P loop across the gears' free play. The
firmware's D term (LeRobot's stock gains are P 16, D 32) or a lower P damps it, at the price of a softer hold.

Per setting `P:D`, written to `--joints` (the others keep the project's gains), the arm goes from where it rests to
the ready pose and back. In between, with every goal and encoder reading logged at 100 Hz:

1. arrives and holds 2.5 s (the settling you see when a run starts),
2. the lift's goal steps 3 deg up and back (a clean disturbance),
3. four wrist strokes as in play (9 deg up in 0.1 s): what the reaction does to the two holding joints,
4. holds 1 s.

Printed per setting: the lift's and elbow's wobble (encoder, peak to peak, slow drift removed) after arriving, after
the step and around the strokes, what the two joints' motion alone does to the blade during the strokes, and how
far below its goal each joint hangs. Logs: measurements/servo/<time>_tune_hold/<P>_<D>.npz. The gains found here
have to go into the simulator too (robot.FIRMWARE_P; a D term does not exist there yet).
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from pingpong_sim.real.arm import RATE, ArmSession, PlanBuilder, gain_profile, open_bus
from pingpong_sim.real.camera_model import ArmCalib
from pingpong_sim.robot import READY_JOINT_POS
from pingpong_sim.servo_bus import JOINTS

REPO = Path(__file__).resolve().parents[2]
READY = np.array([READY_JOINT_POS[j] for j in JOINTS])
LIFT, ELBOW, FLEX = JOINTS.index("shoulder_lift"), JOINTS.index("elbow_flex"), JOINTS.index("wrist_flex")
STEP = np.radians(3.0)
COCK, SWING = np.radians(5.0), np.radians(9.0)


def smooth(n: int) -> np.ndarray:
    s = np.linspace(0, 1, n + 1)[1:]
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def plan_test() -> tuple[np.ndarray, dict[str, tuple[int, int]], list[int]]:
    """Goal samples from ready (true angles), the sample ranges of the phases, and the samples of the strokes'
    fastest moment."""
    rows, marks, peaks = [], {}, []

    def add(name: str, q: np.ndarray) -> None:
        marks[name] = (sum(len(r) for r in rows), sum(len(r) for r in rows) + len(q))
        rows.append(q)

    hold = lambda s: np.repeat(READY[None], int(s * RATE), axis=0)  # noqa: E731
    add("arrive", hold(2.5))
    up = READY[None] + np.outer(smooth(30), np.eye(5)[LIFT] * STEP)
    add("step", np.concatenate([up, np.repeat(up[-1:], 150, axis=0), up[::-1], hold(1.5)]))
    stroke = np.concatenate([-COCK * smooth(15), -COCK + SWING * smooth(10), (SWING - COCK) * (1 - smooth(15)), np.zeros(40)])
    start = sum(len(r) for r in rows)
    add("strokes", np.concatenate([READY[None] + np.outer(stroke, np.eye(5)[FLEX]) for _ in range(4)]))
    peaks = [start + k * len(stroke) + 20 for k in range(4)]
    add("rest", hold(1.0))
    return np.concatenate(rows), marks, peaks


def wobble(x: np.ndarray) -> float:
    """Peak to peak [deg] of a joint angle with its slow drift (0.5 s running mean) removed."""
    if len(x) < 60:
        return float("nan")
    d = np.degrees(x - np.convolve(x, np.ones(51) / 51, "same"))[26:-26]
    return float(np.percentile(d, 99) - np.percentile(d, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--settings", nargs="+", default=["64:0", "32:0", "64:16", "32:16", "32:32", "16:32"], metavar="P:D")
    ap.add_argument("--joints", nargs="+", default=["shoulder_lift", "elbow_flex"], choices=JOINTS)
    args = ap.parse_args()
    settings = [tuple(int(v) for v in s.split(":")) for s in args.settings]
    offsets = ArmCalib.load().joint_offsets
    q_test, marks, peaks = plan_test()
    out = REPO / "measurements" / "servo" / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}_tune_hold"
    out.mkdir(parents=True)
    print(f"{len(settings)} settings on {', '.join(args.joints)}; the arm goes to ready and back for each one")
    results = []
    for n, (p, d) in enumerate(settings):
        gains = gain_profile()
        for j in args.joints:
            gains[j].update(P_Coefficient=p, D_Coefficient=d)
        log: list[tuple] = []
        with ArmSession(open_bus(args.port), joint_offsets=offsets, confirm=n == 0, gains=gains) as s:
            b = PlanBuilder(s.start)
            b.move(READY, name="to_ready")
            n_move = len(b.build().q)
            b.add(q_test, name="test")
            plan = b.build()
            problems = s.check(plan)
            if problems:
                raise SystemExit("plan check failed:\n  " + "\n  ".join(problems))
            print(f"P {p}, D {d} ...", flush=True)
            s.follow(plan.q, on_tick=lambda k, g, r: log.append((k, r.t, *g, *r.q, *r.load)))
            s.move_to(s.start)
        if s.status != "complete":
            print(s.status)
            return 2
        a = np.array(log)
        k, goal, q = a[:, 0].astype(int) - n_move, a[:, 2:7], a[:, 7:12]
        np.savez(out / f"{p}_{d}.npz", k=k, t=a[:, 1], goal=goal, q=q, load=a[:, 12:17], marks=np.array(list(marks.items()), dtype=object),
                 joints=np.array(args.joints))
        seg = lambda name, lo=0, hi=None: (k >= marks[name][0] + lo) & (k < (marks[name][1] if hi is None else marks[name][0] + hi))  # noqa: E731
        row = {"P": p, "D": d}
        for j, name in ((LIFT, "lift"), (ELBOW, "elbow")):
            row[f"{name}_arrive"] = wobble(q[seg("arrive"), j])
            row[f"{name}_step"] = wobble(q[seg("step", 30, 180), j])
            row[f"{name}_strokes"] = float(np.median([np.degrees(np.ptp(q[(k >= pk - 12) & (k < pk + 12), j])) for pk in peaks]))
            row[f"{name}_sag"] = float(np.degrees(np.median((goal - q)[seg("rest"), j])))
        # What lift and elbow alone do to the blade around the strokes (wrist joints frozen at ready).
        dz, tilt = [], []
        for pk in peaks:
            m = (k >= pk - 12) & (k < pk + 12)
            qq = q[m].copy()
            qq[:, [0, FLEX, 4]] = READY[[0, FLEX, 4]]
            poses = [s.model.face_pose(x) for x in qq[::2]]
            dz.append(1e3 * np.ptp([po[0][2] for po in poses]))
            tilt.append(np.ptp([po[1] for po in poses]))
        row["blade_mm"], row["blade_deg"] = float(np.median(dz)), float(np.median(tilt))
        results.append(row)
    print("\nwobble = encoder peak to peak [deg], slow drift removed; strokes = motion within 120 ms of a stroke")
    print("  P   D   lift: arriving  after step  strokes  hangs   elbow: arriving  after step  strokes  hangs   blade in strokes")
    for r in results:
        print(f" {r['P']:3d} {r['D']:3d}        {r['lift_arrive']:5.2f}      {r['lift_step']:5.2f}    {r['lift_strokes']:5.2f}  {r['lift_sag']:+5.2f}"
              f"           {r['elbow_arrive']:5.2f}      {r['elbow_step']:5.2f}    {r['elbow_strokes']:5.2f}  {r['elbow_sag']:+5.2f}"
              f"     {r['blade_mm']:4.1f} mm, {r['blade_deg']:.1f} deg")
    print("logs:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
