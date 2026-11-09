# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["opencv-python-headless", "numpy"]
# ///
"""Track the orange ping pong ball in slow-motion drop videos.

    uv run measurements/ball/track.py <video.mov> [<video.mov> ...] [--out DIR]

Writes one CSV per video to DIR (default: measurements/ball/tracks/<stem>.csv) with
columns:

    t_video      presentation timestamp of the frame [s, video time] (NOT frame index:
                 ~15% of capture frames are missing from the files)
    x_px, y_px   ball centroid in the displayed (rotated, portrait) image, y down [px]
    diameter_px  equivalent-circle diameter of the blob = 2*sqrt(area/pi) [px]
    width_px     blob bounding-box width (motion blur is vertical, so this is the
                 blur-free diameter estimate) [px]
    height_px    blob bounding-box height [px]
    flags        bitmask: 1 = several plausible ball blobs in the frame,
                          2 = blob size deviates >20% from the video's median ball size
                              (partial occlusion by hand / merging)
                          4 = jump from the previous detection (re-acquired track)

Frames with no plausible ball are omitted.

Detection: HSV threshold (5,150,150)-(22,255,255), connected components, then shape
gating (area >= 1500 px, bounding-box fill ratio ~pi/4, aspect ratio) removes the yellow
tape measure marks and skin; candidates that sit at the same place/size in > 25% of the
frames (the orange box on the shelf, the ball at rest) are dropped as static. If several
blobs survive, the one whose diameter is closest to the video median ball diameter and
closest to the previous detection is chosen.

Also writes DIR/<stem>_scene.csv (one row, medians over 15 frames) with the image
orientation of the red paddle blade and of the yellow tape measure, used by analyze.py
to measure the blade tilt of the angled drops:

    blade_cx_px, blade_cy_px          blade centroid
    blade_major_px, blade_minor_px    ellipse axes from second moments (4*sqrt(eigval))
    blade_axis_cw_deg                 major-axis rotation from horizontal, clockwise on
                                      screen (right side lower = positive)
    tape_cw_deg                       tape-measure rotation from vertical, clockwise on
                                      screen (top leaning right = positive); camera roll
                                      reference
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HSV_LO = (5, 150, 150)
HSV_HI = (22, 255, 255)
MIN_AREA = 1500  # ball is >= 60 px across in every video; tape-measure marks are < 40 px
FILL_RANGE = (0.55, 0.95)  # blob area / bbox area; a disc is pi/4 = 0.785
STATIC_TOL_PX = 6.0
STATIC_FRAC = 0.25
ASPECT_RANGE = (0.45, 1.6)  # width / height; vertical motion blur makes it < 1
HERE = Path(__file__).resolve().parent


def candidates(frame: np.ndarray) -> list[tuple[float, float, float, int, int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < MIN_AREA:
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        fill = a / float(w * h)
        asp = w / float(h)
        if not (FILL_RANGE[0] <= fill <= FILL_RANGE[1] and ASPECT_RANGE[0] <= asp <= ASPECT_RANGE[1]):
            continue
        out.append((cent[i][0], cent[i][1], 2.0 * np.sqrt(a / np.pi), int(w), int(h)))
    return out


def scene(frame: np.ndarray) -> tuple[float, ...] | None:
    """Orientation of the red paddle blade (largest red blob) and of the yellow tape."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 120, 50), (4, 255, 255)) | cv2.inRange(hsv, (170, 120, 50), (180, 255, 255))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(red)
    yel = cv2.inRange(hsv, (22, 100, 120), (35, 255, 255))
    m, ylab, ystats, _ = cv2.connectedComponentsWithStats(yel)
    if n < 2 or m < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mo = cv2.moments((lab == i).astype(np.uint8), binaryImage=True)
    mu20, mu02, mu11 = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"], mo["mu11"] / mo["m00"]
    # image y is down, so a positive angle is a clockwise rotation on screen
    axis = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
    ev = np.linalg.eigvalsh(np.array([[mu20, mu11], [mu11, mu02]]))
    j = 1 + int(np.argmax(ystats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(ylab == j)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()[:2]
    if vy < 0:
        vx, vy = -vx, -vy
    tape = float(np.degrees(np.arctan2(-vx, vy)))  # top leaning right = clockwise = positive
    return (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"], 4 * np.sqrt(ev[1]), 4 * np.sqrt(ev[0]), axis, tape)


N_SCENE_FRAMES = 15


def track(video: Path) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video))  # OpenCV applies the rotation metadata
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scene_at = set(np.linspace(0, max(nframes - 1, 0), N_SCENE_FRAMES).astype(int).tolist())
    frames, scenes = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if len(frames) in scene_at and (s := scene(frame)) is not None:
            scenes.append(s)
        frames.append((t, candidates(frame)))
    cap.release()
    sc = np.median(np.array(scenes), axis=0) if scenes else np.full(6, np.nan)
    return select(frames), sc


def select(frames: list) -> np.ndarray:
    """Pick the ball among the per-frame candidates -> rows (t, x, y, d, w, h, flags)."""
    # Static orange things (tape-measure markings, the box on the shelf, the ball while
    # resting on the paddle) sit at a fixed spot for many frames; the flying ball never
    # does. Drop candidates that recur at the same place/size in > STATIC_FRAC of frames.
    flat = np.array([(fi, *c[:3]) for fi, (_, cs) in enumerate(frames) for c in cs])
    if len(flat) == 0:
        return np.empty((0, 7))
    static = np.zeros(len(flat), bool)
    for j, (_, x, y, d) in enumerate(flat):
        near = (np.abs(flat[:, 1] - x) < STATIC_TOL_PX) & (np.abs(flat[:, 2] - y) < STATIC_TOL_PX)
        near &= np.abs(flat[:, 3] / d - 1) < 0.15
        static[j] = len(np.unique(flat[near, 0])) > STATIC_FRAC * len(frames)
    keep: list[list] = [[] for _ in frames]
    for (fi, *_), s, c in zip(flat, static, [c for _, cs in frames for c in cs]):
        if not s:
            keep[int(fi)].append(c)
    frames = [(t, keep[i]) for i, (t, _) in enumerate(frames)]

    all_d = [c[2] for _, cs in frames for c in cs]
    if not all_d:
        return np.empty((0, 7))
    # Ball size: mode of the remaining candidate diameters.
    hist, edges = np.histogram(all_d, bins=np.arange(20, 400, 4))
    k = int(np.argmax(hist))
    sel = [d for d in all_d if edges[max(k - 3, 0)] <= d <= edges[min(k + 4, len(edges) - 1)]]
    d_ball = float(np.median(sel))

    rows = []
    prev = None
    for t, cs in frames:
        cs = [c for c in cs if 0.6 * d_ball <= c[2] <= 1.5 * d_ball]
        if not cs:
            continue
        flags = 1 if len(cs) > 1 else 0

        def cost(c):
            size = abs(np.log(c[2] / d_ball)) * 300.0
            dist = 0.0 if prev is None else np.hypot(c[0] - prev[0], c[1] - prev[1])
            return size + dist

        c = min(cs, key=cost)
        if abs(c[2] / d_ball - 1) > 0.2:
            flags |= 2
        if prev is not None and np.hypot(c[0] - prev[0], c[1] - prev[1]) > 3 * d_ball:
            flags |= 4
        rows.append((t, c[0], c[1], c[2], c[3], c[4], flags))
        prev = c
    return np.array(rows)


def run(video: Path, out_dir: Path) -> str:
    r, sc = track(video)
    out = out_dir / f"{video.stem}.csv"
    with out.open("w") as f:
        f.write("t_video,x_px,y_px,diameter_px,width_px,height_px,flags\n")
        for t, x, y, d, w, h, fl in r:
            f.write(f"{t:.6f},{x:.2f},{y:.2f},{d:.2f},{int(w)},{int(h)},{int(fl)}\n")
    with (out_dir / f"{video.stem}_scene.csv").open("w") as f:
        f.write("blade_cx_px,blade_cy_px,blade_major_px,blade_minor_px,blade_axis_cw_deg,tape_cw_deg\n")
        f.write(",".join(f"{v:.3f}" for v in sc) + "\n")
    return f"{video.name}: {len(r)} detections -> {out}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "tracks")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(len(args.videos), 12)) as ex:
        for msg in ex.map(run, args.videos, [args.out] * len(args.videos)):
            print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
