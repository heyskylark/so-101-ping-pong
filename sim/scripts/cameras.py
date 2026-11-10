"""Camera bring-up: find the devices, name them, check exposure and ball detection.

  uv run scripts/cameras.py list                                   # capture devices and the current assignment
  uv run scripts/cameras.py assign overhead /dev/v4l/by-path/...   # -> measurements/calibration/rig.json
  uv run scripts/cameras.py assign wrist /dev/v4l/by-path/... --exposure-ms 2
  uv run scripts/cameras.py preview overhead                       # live picture + detection; keys below
  uv run scripts/cameras.py color white                            # ball colour (default orange)

Preview keys: + / - exposure (0.5 ms steps, saved on quit), m colour mask, q quit. It prints frames per second, the
timestamp source ("buffer" = the driver's capture-side stamp, needed for timing calibration; "arrival" = fallback),
the delay from that stamp to the frame reaching us, and how often the ball is found with its diameter.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import cv2
import numpy as np

from pingpong_sim.real.cameras import RIG, WHITE, BallColor, BallDetector, CameraConfig, RigConfig, UvcCamera, \
    v4l2_devices


def cmd_list(args) -> int:
    rig = RigConfig.load()
    devs = v4l2_devices()
    if not devs:
        print("no capture devices (cameras plugged in?)")
    for path, card in devs:
        names = [n for n, c in rig.cameras.items() if c.device == path]
        print(f"  {path}  {card}" + (f"  <- {', '.join(names)}" if names else ""))
    for n, c in rig.cameras.items():
        if c.device not in [d for d, _ in devs]:
            print(f"  {n}: {c.device} (not present)")
    return 0


def cmd_assign(args) -> int:
    rig = RigConfig.load()
    old = rig.cameras.get(args.name)
    rig.cameras[args.name] = CameraConfig(device=args.device, exposure_ms=args.exposure_ms or
                                          (old.exposure_ms if old else 2.0))
    rig.save()
    print(f"{args.name} -> {args.device} ({rig.cameras[args.name].exposure_ms} ms); saved {RIG}")
    return 0


def cmd_color(args) -> int:
    rig = RigConfig.load()
    rig.ball = WHITE if args.color == "white" else BallColor()
    rig.save()
    print(f"ball colour {args.color}: HSV {rig.ball}; saved {RIG}")
    return 0


def cmd_preview(args) -> int:
    rig = RigConfig.load()
    if args.name not in rig.cameras:
        raise SystemExit(f"{args.name} has no device: `cameras.py assign {args.name} <device>`")
    cfg = rig.cameras[args.name]
    det = BallDetector(rig.ball)
    found: list = []
    cam = UvcCamera(args.name, cfg, lambda f: det(f) or False)
    cam.start()
    show_mask, last_print, t0 = False, time.perf_counter(), time.perf_counter()
    frames0 = 0
    try:
        while cam.error is None:
            found += [r for _, r in cam.drain()]
            f = cam.last_frame
            if f is not None:
                img = f.copy()
                hit = found[-1] if found else False
                if hit:
                    u, v, d = hit
                    cv2.circle(img, (int(u), int(v)), max(3, int(d / 2)), (0, 255, 0), 2)
                if show_mask:
                    img = cv2.cvtColor(det.mask(f), cv2.COLOR_GRAY2BGR)
                cv2.putText(img, f"{args.name} {cfg.exposure_ms:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
                cv2.imshow(f"preview {args.name}", cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
            k = cv2.waitKey(15) & 0xFF
            if k == ord("q"):
                break
            if k == ord("m"):
                show_mask = not show_mask
            if k in (ord("+"), ord("=")) or k == ord("-"):
                cfg = replace(cfg, exposure_ms=max(0.1, cfg.exposure_ms + (0.5 if k != ord("-") else -0.5)))
                from pingpong_sim.real.cameras import set_controls

                set_controls(cfg.device, cfg.exposure_ms)
            now = time.perf_counter()
            if now - last_print > 2.0:
                hits = [r for r in found if r]
                fps = (cam.frames - frames0) / (now - last_print)
                lat = np.median(cam.latency) * 1e3 if cam.latency else float("nan")
                print(f"{fps:5.1f} fps, stamps: {cam.stamp_source}, stamp -> us {lat:.0f} ms, ball in "
                      f"{100 * len(hits) / max(len(found), 1):.0f} % of frames"
                      + (f", diameter {np.median([h[2] for h in hits]):.1f} px" if hits else ""))
                frames0, last_print, found = cam.frames, now, []
            if args.seconds and now - t0 > args.seconds:
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    if cam.error:
        raise SystemExit(f"{args.name}: {cam.error}")
    if cfg.exposure_ms != rig.cameras[args.name].exposure_ms:
        rig.cameras[args.name] = cfg
        rig.save()
        print(f"exposure {cfg.exposure_ms} ms saved to {RIG}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("assign")
    p.add_argument("name", choices=("overhead", "wrist"))
    p.add_argument("device")
    p.add_argument("--exposure-ms", type=float)
    p = sub.add_parser("preview")
    p.add_argument("name", choices=("overhead", "wrist"))
    p.add_argument("--seconds", type=float, default=0.0)
    p = sub.add_parser("color")
    p.add_argument("color", choices=("orange", "white"))
    args = ap.parse_args()
    return {"list": cmd_list, "assign": cmd_assign, "preview": cmd_preview, "color": cmd_color}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
