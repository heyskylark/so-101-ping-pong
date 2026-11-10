"""Move the arm to a named pose, hold it, optionally take camera snapshots there, and return to where it started.

  uv run scripts/arm_pose.py ready --port /dev/ttyACM0                         # hold until Enter
  uv run scripts/arm_pose.py ready --port /dev/ttyACM0 --hold 5 --snapshot overhead wrist

Used to aim the cameras at the true bounce point (ready pose: blade centre 0.40 m in front of the base axis, 0.25
m up, level). The move is checked in the sim and runs at 0.5 rad/s; the calibrated joint offsets are applied if
measurements/calibration/arm.json exists. Snapshots go to sim/build/snapshots/<camera>_<pose>.jpg with the paddle
face's pixel position (red rubber) printed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from pingpong_sim.real.arm import RATE, ArmSession, open_bus
from pingpong_sim.real.camera_model import ArmCalib
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pose", choices=sorted(POSES))
    ap.add_argument("--port", required=True)
    ap.add_argument("--hold", type=float, help="seconds to hold (default: until Enter)")
    ap.add_argument("--snapshot", nargs="*", default=[], help="cameras to photograph at the pose")
    ap.add_argument("--yes", action="store_true", help="skip the typed 'go' confirmation")
    args = ap.parse_args()
    offsets = ArmCalib.load().joint_offsets
    with ArmSession(open_bus(args.port), joint_offsets=offsets, confirm=not args.yes) as s:
        print("start pose [deg]: " + ", ".join(f"{j} {np.degrees(v):+.1f}" for j, v in zip(JOINTS, s.start)))
        s.move_to(POSES[args.pose])
        err = np.degrees(np.abs(s.read().q - POSES[args.pose]))
        print(f"at {args.pose}; |error| per joint [deg]: " + ", ".join(f"{j} {e:.1f}" for j, e in zip(JOINTS, err)))
        if args.snapshot:
            hold_q = s.read().q
            import threading

            t = threading.Thread(target=snapshot, args=(args.snapshot, args.pose))
            t.start()
            while t.is_alive():  # keep commanding the pose (and the watchdog) while the cameras work
                s.follow(np.repeat(hold_q[None], int(0.2 * RATE), axis=0))
        if args.hold is not None:
            s.follow(np.repeat(s.read().q[None], int(args.hold * RATE), axis=0))
        elif not args.snapshot:
            input("holding; Enter to return ")
        s.move_to(s.start)
    print(s.status)
    return 0 if s.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
