"""Move the arm to a named pose, hold it, optionally take camera snapshots there, and return to where it started.

  uv run scripts/arm_pose.py ready --port /dev/ttyACM0                         # hold until Enter
  uv run scripts/arm_pose.py ready --port /dev/ttyACM0 --snapshot overhead wrist    # pictures after Enter
  uv run scripts/arm_pose.py ready --port /dev/ttyACM0 --hold 5 --snapshot overhead wrist
  uv run scripts/arm_pose.py ready --port /dev/ttyACM0 --readings                # load test: Enter prints the encoders' face height

Used to aim the cameras at the true bounce point (ready pose: blade centre 0.28 m in front of the base axis, 0.31
m up, level) and for the ruler measurement of `calibrate.py lift`. The move is checked in the sim and runs at 0.5
rad/s; the calibrated joint offsets are applied if measurements/calibration/arm.json exists. The servos hold a
goal with ~1 deg of sag per joint (no integral gain), so the hold reports where the encoders put the face and
saves their mean angles to measurements/calibration/raw/<time>_<pose>.npz. Snapshots go to
sim/build/snapshots/<camera>_<pose>.jpg with the paddle face's pixel position (red rubber) printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from pingpong_sim.real.arm import RATE, TABLE_Z, ArmSession, open_bus
from pingpong_sim.real.camera_model import CALIB_DIR, ArmCalib
from pingpong_sim.robot import READY_JOINT_POS
from pingpong_sim.servo_bus import JOINTS

POSES = {"ready": np.array([READY_JOINT_POS[j] for j in JOINTS])}
OUT = Path(__file__).resolve().parents[1] / "build" / "snapshots"


def red_face(frame: np.ndarray) -> str:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = (((hsv[..., 0] >= 170) | (hsv[..., 0] <= 3)) & (hsv[..., 1] > 140) & (hsv[..., 2] > 35)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return "paddle face not found"
    k = 1 + int(np.argmax(st[1:, 4]))
    return f"paddle face centre ({cen[k][0]:.0f}, {cen[k][1]:.0f}) px, {st[k, 2]} x {st[k, 3]} px"


def snapshot(names: list[str], pose: str) -> None:
    from pingpong_sim.real.cameras import RigConfig, UvcCamera

    rig = RigConfig.load()
    cams = [UvcCamera(n, rig.cameras[n], None) for n in names]
    for c in cams:
        c.start()
    time.sleep(2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for c in cams:
        if c.last_frame is None:
            print(f"  {c.name_}: no frame ({c.error})")
            continue
        path = OUT / f"{c.name_}_{pose}.jpg"
        cv2.imwrite(str(path), c.last_frame)
        print(f"  {c.name_}: {red_face(c.last_frame)} -> {path}")
    for c in cams:
        c.stop()
        c.join(2.0)


def report(s: ArmSession, pose: str, q: np.ndarray) -> None:
    """What the encoders (true angles (N, 5)) read while the goal was held, less the settling after the move
    (2 s, or the first half of a short hold): printed, and saved as raw angles for `calibrate.py lift`."""
    q = q[min(int(2.0 * RATE), len(q) // 2):].mean(0)
    p, tilt = s.model.face_pose(q)
    print("encoders - goal while holding [deg]: "
          + ", ".join(f"{j} {e:+.1f}" for j, e in zip(JOINTS, np.degrees(q - POSES[pose]))))
    print(f"the encoders put the face centre {100 * (p[2] - TABLE_Z):.1f} cm above the table, tipped {tilt:+.1f} deg "
          "(+ = tip up)")
    path = CALIB_DIR / "raw" / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}_{pose}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, q_raw=q - s.offsets, offsets=s.offsets, goal=POSES[pose])
    print("saved", path)


def run(s: ArmSession, args) -> None:
    """Move to the pose, hold its goal (for --hold seconds or until Enter, then while the snapshots are taken: so
    a ball can be set on the blade first), report the hold and return to the start pose."""
    goal = POSES[args.pose]
    print("start pose [deg]: " + ", ".join(f"{j} {np.degrees(v):+.1f}" for j, v in zip(JOINTS, s.start)))
    s.move_to(goal)
    rows: list[np.ndarray] = []

    def hold(seconds: float) -> None:  # keeps commanding the goal (and the watchdog)
        s.follow(np.repeat(goal[None], max(1, int(seconds * RATE)), axis=0), on_tick=lambda k, g, r: rows.append(r.q))

    def hold_while(target, *a) -> None:
        t = threading.Thread(target=target, args=a, daemon=True)
        t.start()
        while t.is_alive():
            hold(0.2)

    if args.play is not None:
        motions = (("up and down", "push the blade's tip gently up, then down"),
                   ("side to side", "push the blade's tip gently left, then right"),
                   ("twist", "turn the blade about its handle like a door knob, both ways"))
        print(f"three motions of {args.play:.0f} s each, by hand, while the arm holds. Go only as far as the blade moves "
              "freely: stop where it starts to push back.")
        log = {}
        for name, how in motions:
            hold_while(input, f"next: {name} ({how}). Enter to start ")
            n0 = len(rows)
            print(f"  {name}: now, for {args.play:.0f} s ...", flush=True)
            hold(args.play)
            qs = log[name] = np.array(rows[n0:])
            rows[n0:] = []  # the rocking is not the hold `report` saves
            lo, hi = np.percentile(qs, 1, axis=0), np.percentile(qs, 99, axis=0)
            mid = np.median(qs, axis=0)
            print("    joint          encoder travel [deg]   moves the blade centre [mm] (x, y, z)")
            for k, j in enumerate(JOINTS):
                a, b = mid.copy(), mid.copy()
                a[k], b[k] = lo[k], hi[k]
                d = 1e3 * (s.model.face_pose(b)[0] - s.model.face_pose(a)[0])
                print(f"    {j:14s} {np.degrees(hi[k] - lo[k]):6.2f}                 {d[0]:+5.1f} {d[1]:+5.1f} {d[2]:+5.1f}")
            p = np.array([s.model.face_pose(x)[0] for x in qs[::5]])
            d = 1e3 * (np.percentile(p, 99, axis=0) - np.percentile(p, 1, axis=0))
            print(f"    blade centre by the encoders: {d[0]:.1f} / {d[1]:.1f} / {d[2]:.1f} mm of travel (x / y / z). If it "
                  "visibly moved more, the rest is play or bending after the encoders.")
        path = CALIB_DIR.parent / "servo" / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}_play.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, joints=np.array(JOINTS), **{k.replace(" ", "_"): v for k, v in log.items()})
        print("saved", path)
        hold(1.0)
    elif args.hold is not None:
        hold(args.hold)
    elif args.readings:
        typed: list[str] = []
        while True:
            hold_while(lambda: typed.append(input("holding; Enter for a reading, q + Enter to return ")))
            if typed[-1].strip().lower() == "q":
                break
            p, tilt = s.model.face_pose(np.mean(rows[-RATE:], axis=0))  # the last second
            print(f"  encoders: face centre {1e3 * (p[2] - TABLE_Z):.1f} mm above the table, tipped {tilt:+.2f} deg")
    else:
        hold_while(input, "holding; Enter to " + ("take the pictures and return " if args.snapshot else "return "))
    if args.snapshot:
        hold_while(snapshot, args.snapshot, args.pose)
    report(s, args.pose, np.array(rows))
    s.move_to(s.start)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pose", choices=sorted(POSES))
    ap.add_argument("--port", required=True)
    ap.add_argument("--hold", type=float, help="seconds to hold (default: until Enter)")
    ap.add_argument("--readings", action="store_true",
                    help="load test: each Enter prints where the encoders put the face (mean of the last second); "
                         "with a ruler at the blade, the difference is what bends unseen by the encoders")
    ap.add_argument("--play", type=float, metavar="SECONDS",
                    help="backlash test: the arm holds while you move the blade gently by hand to both ends of its "
                         "free play, three motions of this long each (up-down, side to side, twist; Enter starts "
                         "each); prints how far each joint's encoder travelled and what that moves the blade (the "
                         "play the encoders see: in the gears) and saves the traces to measurements/servo/")
    ap.add_argument("--snapshot", nargs="*", default=[], help="cameras to photograph at the pose")
    ap.add_argument("--yes", action="store_true", help="skip the typed 'go' confirmation")
    args = ap.parse_args()
    offsets = ArmCalib.load().joint_offsets
    with ArmSession(open_bus(args.port), joint_offsets=offsets, confirm=not args.yes) as s:
        run(s, args)
    print(s.status)
    return 0 if s.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
