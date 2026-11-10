"""Camera bring-up: find the devices, name them, check exposure and ball detection.

  uv run scripts/cameras.py list                                   # capture devices and the current assignment
  uv run scripts/cameras.py assign overhead /dev/v4l/by-path/...   # -> measurements/calibration/rig.json
  uv run scripts/cameras.py assign wrist /dev/v4l/by-path/... --exposure-ms 3
  uv run scripts/cameras.py preview overhead                       # live picture + detection; keys below
  uv run scripts/cameras.py mask wrist                             # pixels the detector should ignore (no ball in view)
  uv run scripts/cameras.py color white                            # ball colour (default orange)

Preview keys: + / - exposure (0.1 ms steps, saved at once), m colour mask (ignored pixels in blue), q quit. It prints
frames per second, the timestamp source ("buffer" = the driver's capture-side stamp, needed for timing calibration;
"arrival" = fallback), the delay from that stamp to the frame reaching us, and how often the ball is found with its
diameter.

`mask`: with no ball in view, records for a few seconds which pixels look ball-coloured (for the wrist camera, the
blade's edge, fixed in its picture; for the overhead one, orange clutter) and saves them, grown by a few pixels, to
measurements/calibration/ignore_<name>.png. Redo it after changing the light or the exposure.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import cv2
import numpy as np

from pingpong_sim.real.cameras import RIG, WHITE, BallColor, BallDetector, CameraConfig, RigConfig, UvcCamera, \
    ignore_path, load_ignore, set_controls, v4l2_devices


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
    ignore = load_ignore(args.name)
    det = BallDetector(rig.ball, ignore=ignore)
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
                    if ignore is not None:
                        img[ignore > 0] = (img[ignore > 0] * 0.5 + np.array([120, 0, 0])).astype(np.uint8)
                cv2.putText(img, f"{args.name} {cfg.exposure_ms:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
                cv2.imshow(f"preview {args.name}", cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
            k = cv2.waitKey(15) & 0xFF
            if k == ord("q"):
                break
            if k == ord("m"):
                show_mask = not show_mask
            if k in (ord("+"), ord("=")) or k == ord("-"):
                cfg = replace(cfg, exposure_ms=round(max(0.1, cfg.exposure_ms + (0.1 if k != ord("-") else -0.1)), 1))
                set_controls(cfg.device, cfg.exposure_ms)
                rig.cameras[args.name] = cfg
                rig.save()
                print(f"exposure {cfg.exposure_ms:.1f} ms (saved)")
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
    return 0


def cmd_mask(args) -> int:
    rig = RigConfig.load()
    if args.name not in rig.cameras:
        raise SystemExit(f"{args.name} has no device: `cameras.py assign {args.name} <device>`")
    det = BallDetector(rig.ball)
    acc = None
    n = 0

    def add(frame):
        nonlocal acc, n
        m = det.mask(frame, raw=True).astype(np.float32) / 255
        acc = m if acc is None else acc + m
        n += 1
        return None

    print(f"{args.name}: no ball in view, light as for play; recording {args.seconds:.0f} s")
    cam = UvcCamera(args.name, rig.cameras[args.name], add)
    cam.start()
    time.sleep(args.seconds)
    cam.stop()
    cam.join(2.0)
    if cam.error or not n:
        raise SystemExit(f"{args.name}: {cam.error or 'no frames'}")
    hit = (acc / n >= 0.02).astype(np.uint8) * 255  # ball-coloured in at least 2 % of the frames
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.grow + 1, 2 * args.grow + 1))
    ignore = cv2.dilate(hit, k)
    cv2.imwrite(str(ignore_path(args.name)), ignore)
    print(f"{n} frames; ignoring {100 * (ignore > 0).mean():.2f} % of the picture "
          f"({(hit > 0).sum()} px seen ball-coloured, grown by {args.grow} px) -> {ignore_path(args.name)}")
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
    p = sub.add_parser("mask")
    p.add_argument("name", choices=("overhead", "wrist"))
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--grow", type=int, default=6, help="px added around the ball-coloured pixels")
    args = ap.parse_args()
    return {"list": cmd_list, "assign": cmd_assign, "preview": cmd_preview, "color": cmd_color,
            "mask": cmd_mask}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
