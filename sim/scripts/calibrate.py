"""Camera and arm calibration for the real rig (docs/hardware_setup.md, "Calibration").

Order (from sim/; the arm steps need the joint map and the camera devices in rig.json):

  uv run scripts/calibrate.py selftest                      # the solvers on synthetic data (no hardware)
  uv run scripts/calibrate.py targets --tag 70.1 --square 30.0 --ruler 100.0   # caliper readings of the prints
  uv run scripts/calibrate.py intrinsics overhead           # wave the board in front of the camera
  uv run scripts/calibrate.py intrinsics wrist
  uv run scripts/calibrate.py overhead --port /dev/ttyACM0  # the arm shows the paddle tag to the overhead camera
  uv run scripts/calibrate.py lift --paddle-height-cm 30.5   # ruler at the ready hold (arm_pose.py ready): lift offset
  uv run scripts/calibrate.py face --front-cm 29.9 --rear-cm 30.7 --length-cm 15.1 \
      [--left-cm 30.4 --right-cm 30.0 --width-cm 15.0]       # same hold: the face's lean; sideways: the roll zero
  uv run scripts/calibrate.py wrist --port /dev/ttyACM0     # board standing in front; the arm moves the wrist camera
  uv run scripts/calibrate.py timing overhead --port /dev/ttyACM0
  uv run scripts/calibrate.py timing wrist --port /dev/ttyACM0   # board where it stood for `wrist`
  uv run scripts/calibrate.py gravity --port /dev/ttyACM0   # drop the ball a few times: checks the whole chain
  uv run scripts/calibrate.py static --port /dev/ttyACM0 --free   # ball resting on the blade, then on a support:
                                                            # each camera's position / range-from-size errors

Results: measurements/calibration/camera_<name>.json, arm.json; raw data of each step in
measurements/calibration/raw/<time>_<step>.npz. Targets (TODO.md): overhead pose <= 3 mm / 0.3 deg, wrist mount
<= 1 mm / 0.5 deg, timestamps <= +-2 ms; the gravity check should show a tilt < 0.5 deg and |g| within 2 %.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from pingpong_sim.camera_sim import OVERHEAD
from pingpong_sim.kinematics import ArmKinematics
from pingpong_sim.real.arm import RATE, TABLE_Z, ArmModel, ArmSession, PlanBuilder, open_bus
from pingpong_sim.real.calib_solve import (
    WRIST_BOARD_CENTRE,
    Links,
    angle_deg,
    board_pnp,
    fit_gravity,
    fit_intrinsics,
    fit_overhead,
    fit_timing,
    fit_wrist,
    leave_one_out,
    measure_ball,
    overhead_poses,
    project,
    rot,
    rot_z,
    rvec,
    table_board,
    tag_on_face,
    wrist_poses,
)
from pingpong_sim.real.camera_model import CALIB_DIR, ArmCalib, CameraCalib, apply_wrist_mount, nominal, to_measurements
from pingpong_sim.real.cameras import MAX_BALL_PX, MIN_BALL_PX, BallCameras, RigConfig, UvcCamera
from pingpong_sim.real.targets import SPEC, BoardDetector, TagDetector, load_spec
from pingpong_sim.ball import BALL_RADIUS
from pingpong_sim.robot import READY_JOINT_POS, READY_X
from pingpong_sim.servo_bus import JOINTS

RAW = CALIB_DIR / "raw"
SNAPSHOTS = Path(__file__).resolve().parents[1] / "build" / "snapshots"  # as arm_pose.py
Q_READY = np.array([READY_JOINT_POS[j] for j in JOINTS])


def stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def save_raw(step: str, **arrays) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{stamp()}_{step}.npz"
    np.savez_compressed(path, **{k: np.asarray(v, dtype=object) if isinstance(v, list) else v for k, v in arrays.items()})
    return path


def nominal_overhead() -> tuple[np.ndarray, np.ndarray]:
    """Where the overhead camera should be (sim placement): optical centre and axis."""
    a = np.array(OVERHEAD.axis, float)
    return np.array(OVERHEAD.pos, float), a / np.linalg.norm(a)


def camera_calib(name: str) -> CameraCalib:
    return CameraCalib.load(name) if CameraCalib.path(name).exists() else nominal(name)


def start_camera(name: str, process) -> UvcCamera:
    rig = RigConfig.load()
    if name not in rig.cameras:
        raise SystemExit(f"no device for {name} in rig.json: run `scripts/cameras.py assign {name} <device>`")
    cam = UvcCamera(name, rig.cameras[name], process)
    cam.start()
    t = time.perf_counter()
    while cam.frames == 0 and cam.error is None and time.perf_counter() - t < 5:
        time.sleep(0.05)
    if cam.error or cam.frames == 0:
        raise SystemExit(f"{name}: {cam.error or 'no frames'}")
    return cam


def hold(session: ArmSession, seconds: float) -> list[tuple[float, np.ndarray]]:
    """Hold the current goal for `seconds`, returning the encoder readings (t, true angles)."""
    q = session.read().q
    rows: list[tuple[float, np.ndarray]] = []
    session.follow(np.repeat(q[None], int(seconds * RATE), axis=0), on_tick=lambda k, g, r: rows.append((r.t, r.q)))
    return rows


def skipped_picture(cam: UvcCamera, step: str, i: int) -> Path:
    """Saves the camera's current frame at a pose where the target wasn't found (to see why); returns its path."""
    path = SNAPSHOTS / f"{step}_skipped_{i + 1:02d}.jpg"
    if cam.last_frame is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cam.last_frame)
    return path


def check_tour(session: ArmSession, poses: np.ndarray) -> None:
    b = PlanBuilder(session.read().q)
    for q in poses:
        b.move(q, name="tour")
    b.move(session.start, name="to_start")
    problems = session.check(b.build())
    if problems:
        raise SystemExit("pose tour check failed:\n  " + "\n  ".join(problems))


def safe_tour(session: ArmSession, poses: np.ndarray, via: np.ndarray) -> list[tuple[np.ndarray, int | None]]:
    """Waypoints (pose, its index in `poses` or None) for a tour of the poses and back toward the start. Each pose
    clears the table, but the joint-space line between two of them need not: a move the plan check refuses goes
    over `via` (the ready pose: high and clear) instead, and a pose not reachable even so is left out."""
    def ok(a: np.ndarray, b: np.ndarray) -> bool:
        pb = PlanBuilder(a)
        pb.move(b, name="tour")
        return not session.check(pb.build())

    at = session.read().q
    tour: list[tuple[np.ndarray, int | None]] = []
    for i, q in enumerate(poses):
        if ok(at, q):
            tour.append((q, i))
        elif ok(at, via) and ok(via, q):
            tour += [(via, None), (q, i)]
            print(f"  pose {i + 1}: reached over the ready pose (the direct move dips too close to the table)")
        else:
            print(f"  pose {i + 1}: left out (no checked move reaches it)")
            continue
        at = q
    pb = PlanBuilder(at)
    pb.move(session.start, name="to_start")
    if session.check(pb.build()):
        tour.append((via, None))
    return tour


# ---------------------------------------------------------------------------------------------------- targets

def cmd_targets(args) -> int:
    spec = json.loads(SPEC.read_text())
    m = spec.setdefault("measured_mm", {})
    if args.ruler is not None:
        scale = args.ruler / 100.0
        print(f"printed 100 mm ruler: {args.ruler:.2f} mm (scale {scale:.4f})")
        if abs(scale - 1) > 0.005:
            print("  more than 0.5 % off: the printer scaled the page. Reprint with scaling off, or the measured "
                  "sizes below still make the calibration right.")
        m["ruler_100"] = args.ruler
    if args.tag is not None:
        m["tag_black_square"] = args.tag
    if args.square is not None:
        m["board_square"] = args.square
    if args.marker is not None:
        m["board_marker"] = args.marker
    SPEC.write_text(json.dumps(spec, indent=2) + "\n")
    s = load_spec()
    print(f"using: tag {s['tag_mm']:.2f} mm, board squares {s['square_mm']:.2f} mm, markers {s['marker_mm']:.2f} mm")
    return 0


# ---------------------------------------------------------------------------------------------------- intrinsics

def intrinsics_sessions(camera: str) -> list[Path]:
    return sorted(RAW.glob(f"*_intrinsics_{camera}.npz"))


def coverage(uv: np.ndarray, W: int, H: int) -> np.ndarray:
    """Board corners per cell of an 8 x 4 grid over the picture (rows top to bottom)."""
    g, _, _ = np.histogram2d(uv[:, 0], uv[:, 1], bins=[8, 4], range=[[0, W], [0, H]])
    return g.T.astype(int)


def fit_and_save_intrinsics(camera: str, W: int, H: int, sessions: list[Path]) -> None:
    det = BoardDetector(min_corners=12)
    obj, img = [], []
    for path in sessions:
        r = np.load(path, allow_pickle=True)
        for ids, uv in zip(r["ids"], r["uv"]):
            obj.append(det.corners3d[np.asarray(ids, int)])
            img.append(np.asarray(uv, float))
    fit = fit_intrinsics(obj, img, (W, H))
    c = camera_calib(camera)
    c.width, c.height, c.K, c.D = W, H, fit["K"].tolist(), fit["D"].tolist()
    xy = c.undistort(np.array([[0, fit["K"][1, 2]], [W, fit["K"][1, 2]], [fit["K"][0, 2], 0], [fit["K"][0, 2], H]]))
    th = np.degrees(np.arctan(np.linalg.norm(xy, axis=1)))
    cov = coverage(np.concatenate(img), W, H)
    c.steps["intrinsics"] = {"date": stamp(), "views": len(obj), "rms_px": fit["rms_px"],
                             "raw": [p.name for p in sessions], "coverage": cov.tolist()}
    print(f"RMS {fit['rms_px']:.2f} px over {len(obj)} views from {len(sessions)} session(s) (worst view "
          f"{max(fit['per_view_px']):.2f} px); f {fit['K'][0, 0]:.1f} / {fit['K'][1, 1]:.1f} px, centre "
          f"{fit['K'][0, 2]:.1f}, {fit['K'][1, 2]:.1f}; D {np.round(fit['D'], 4).tolist()}; field of view "
          f"{th[0] + th[1]:.0f} x {th[2] + th[3]:.0f} deg")
    print("board corners per picture cell (empty cells: the lens model is extrapolated there):")
    print("\n".join("  " + " ".join(f"{v:4d}" for v in row) for row in cov))
    if not fit["monotonic"]:
        raise SystemExit("the fitted lens curve folds back near the edges: add views at the edges and corners")
    print("saved", c.save())


def cmd_intrinsics(args) -> int:
    """Collect views (unless --refit), then fit on every saved session of this camera (only this one with --fresh)."""
    if args.refit:
        sessions = intrinsics_sessions(args.camera)
        if not sessions:
            raise SystemExit(f"no saved views for {args.camera}")
        c = camera_calib(args.camera)
        fit_and_save_intrinsics(args.camera, c.width, c.height, sessions)
        return 0
    det = BoardDetector(min_corners=12)
    cam = start_camera(args.camera, det)
    W, H = cam.cfg.width, cam.cfg.height
    print(f"{args.camera}: hold the board still in a spot, then move it; {args.views} views are taken automatically "
          "(cover the whole picture, corners and edges included, tilted up to ~45 deg, near and far). q to stop.")
    views, feats, last = [], [], None
    show = True
    try:
        while len(views) < args.views:
            for _, (ids, uv) in cam.drain():
                f = np.array([uv[:, 0].mean() / W, uv[:, 1].mean() / H, np.ptp(uv[:, 0]) / W, np.ptp(uv[:, 1]) / H])
                still = last is not None and set(ids) == set(last[0]) and \
                    np.abs(uv - last[1][[list(last[0]).index(i) for i in ids]]).mean() < 1.0
                last = (ids, uv)
                if still and (not feats or min(np.abs(np.array(feats) - f).sum(1)) > 0.12):
                    views.append((ids, uv))
                    feats.append(f)
                    print(f"  view {len(views)}/{args.views}: {len(ids)} corners")
            if show and cam.last_frame is not None:
                img = cam.last_frame.copy()
                for ids, uv in views:
                    for u, v in uv:
                        cv2.circle(img, (int(u), int(v)), 2, (0, 200, 0), -1)
                if last is not None:
                    for u, v in last[1]:
                        cv2.circle(img, (int(u), int(v)), 4, (0, 0, 255), 1)
                try:
                    cv2.imshow(f"intrinsics {args.camera}", cv2.resize(img, (W // 2, H // 2)))
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    show = False  # no display: progress is printed
            else:
                time.sleep(0.03)
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    if len(views) < 10:
        raise SystemExit(f"only {len(views)} views; need >= 10")
    raw = save_raw(f"intrinsics_{args.camera}", ids=[i for i, _ in views], uv=[u for _, u in views])
    sessions = [raw] if args.fresh else intrinsics_sessions(args.camera)
    fit_and_save_intrinsics(args.camera, W, H, sessions)
    return 0


# ---------------------------------------------------------------------------------------------------- overhead

def overhead_sessions() -> list[Path]:
    return sorted(p for p in RAW.glob("*_overhead.npz") if "_intrinsics_" not in p.name and "_timing_" not in p.name)


def cmd_overhead(args) -> int:
    calib = CameraCalib.load("overhead")
    if "intrinsics" not in calib.steps:
        raise SystemExit("calibrate the overhead camera's lens first: calibrate.py intrinsics overhead")
    tag = TagDetector()
    prev = ArmCalib.load()
    have_prev = calib.R is not None and prev.tag_R is not None
    off_prev = np.array(prev.joint_offsets) if have_prev else np.zeros(len(JOINTS))
    if not args.refit:
        cam = start_camera("overhead", tag)
        try:
            with ArmSession(open_bus(args.port)) as s:  # raw angles: the offsets are being fitted
                links = Links(s.model.m)
                cp, ca = nominal_overhead()
                camera = None
                if have_prev:  # plan through the real lens with the previous estimate of the camera and the tag
                    cp, ca = np.array(calib.p), np.array(calib.R)[:, 2]
                    camera = {"K": calib.Km, "D": calib.Dm, "R": np.array(calib.R), "p": cp, "width": calib.width,
                              "height": calib.height, "tag_pts": tag.object_points, "tag_R": np.array(prev.tag_R),
                              "tag_p": np.array(prev.tag_p), "offsets": off_prev}
                poses = overhead_poses(links, s.model, cp, ca, n=args.poses, camera=camera) - off_prev  # -> raw
                tour = safe_tour(s, poses, Q_READY - off_prev)
                qs, corners = [], []
                for q, i in tour:
                    s.move_to(q)
                    if i is None:  # a detour or the way home
                        continue
                    hold(s, 1.5)  # the arm hunts ~+-1 deg for over a second after a move
                    cam.drain()
                    rows = hold(s, 1.0)
                    dets = [c for _, c in cam.drain()]
                    if len(dets) < 5:
                        print(f"  pose {i + 1}/{len(poses)}: tag seen in {len(dets)} frames, skipped (picture: "
                              f"{skipped_picture(cam, 'overhead', i)})")
                        continue
                    qs.append(np.mean([r[1] for r in rows], axis=0))
                    corners.append(np.median(np.array(dets), axis=0))
                    print(f"  pose {i + 1}/{len(poses)}: tag in {len(dets)} frames")
                s.move_to(s.start)
        finally:
            cam.stop()
        if qs:
            save_raw("overhead", qs=np.array(qs), corners=np.array(corners))
    else:
        links = Links(ArmModel().m)
    # Pool the sessions (the camera and the tag must not have moved between them; --fresh: the newest only).
    sessions = overhead_sessions()[-1:] if args.fresh else overhead_sessions()
    data = [np.load(p) for p in sessions]
    qs = np.concatenate([d["qs"] for d in data]) if data else np.zeros((0, 5))
    corners = np.concatenate([d["corners"] for d in data]) if data else np.zeros((0, 4, 2))
    if len(qs) < 8:
        raise SystemExit(f"tag seen at only {len(qs)} poses (need >= 8): is the tag on the back face, the camera "
                         "where docs/hardware_setup.md puts it?")
    # Pan and lift aren't fitted here (see calib_solve.FIT_JOINTS): keep the lift offset gravity found, else 0.
    fixed = np.array(prev.joint_offsets) * np.array([0, 1, 0, 0, 0]) if "gravity" in prev.steps else None
    fit = fit_overhead(links, calib.Km, calib.Dm, qs, corners, tag.object_points, fixed_offsets=fixed)
    report_overhead(fit)
    if fixed is None:
        print("shoulder_lift offset held at 0: the overhead camera can't see it; `calibrate.py gravity` sets it")

    def sub(idx):
        f = fit_overhead(links, calib.Km, calib.Dm, qs[idx], corners[idx], tag.object_points, fixed_offsets=fixed)
        return f["p"], f["R"], f["offsets"]

    unc = leave_one_out(sub, len(qs))
    ok = unc["pos_m"] <= 0.003 and unc["rot_deg"] <= 0.3
    print(f"uncertainty (leave-one-pose-out, {len(qs)} poses from {len(sessions)} session(s)): camera "
          f"{1e3 * unc['pos_m']:.1f} mm / {unc['rot_deg']:.2f} deg (target 3 mm / 0.3 deg: "
          f"{'met' if ok else 'NOT met, run again to add poses'}); offsets [deg] lift/elbow/wrist_flex/roll "
          + "/".join(f"{v:.2f}" for v in unc["offsets_deg"][1:]))
    calib.R, calib.p = fit["R"].tolist(), fit["p"].tolist()
    calib.steps["extrinsics"] = {"date": stamp(), "poses": len(qs), "rms_px": fit["rms_px"],
                                 "raw": [p.name for p in sessions], "uncertainty_mm": 1e3 * unc["pos_m"],
                                 "uncertainty_deg": unc["rot_deg"]}
    arm = ArmCalib.load()
    arm.joint_offsets = fit["offsets"].tolist()
    arm.tag_R, arm.tag_p = fit["tag_R"].tolist(), fit["tag_p"].tolist()
    arm.steps["overhead"] = {"date": stamp(), "poses": qs.tolist(), "per_view_px": fit["per_view_px"].tolist(),
                             "offsets_uncertainty_deg": unc["offsets_deg"].tolist()}
    print("saved", calib.save(), arm.save())
    return 0


def report_overhead(fit: dict) -> None:
    cp, _ = nominal_overhead()
    print(f"RMS {fit['rms_px']:.2f} px (worst pose {fit['per_view_px'].max():.2f} px); camera at "
          f"{np.round(fit['p'], 3).tolist()} m ({1e3 * np.linalg.norm(fit['p'] - cp):.0f} mm from the planned spot); "
          "joint offsets [deg] " + ", ".join(f"{j} {np.degrees(o):+.2f}" for j, o in zip(JOINTS, fit["offsets"])))


# ---------------------------------------------------------------------------------------------------- wrist

def wrist_usable(calib: CameraCalib):
    """Board corners the wrist mount fit can use: beyond the blade strip and the holder (the half that looks above
    the blade), inside the picture, in grid cells the lens fit had data in."""
    cov = np.array(calib.steps.get("intrinsics", {}).get("coverage", np.ones((4, 8))))
    W, H = calib.width, calib.height

    def usable(uv: np.ndarray) -> np.ndarray:
        u, v = uv[:, 0], uv[:, 1]
        ok = (u >= 780) & (u < W - 30) & (v > 30) & (v < H - 30)
        ci = np.clip((u / (W / 8)).astype(int), 0, 7)
        ri = np.clip((v / (H / 4)).astype(int), 0, 3)
        return ok & (cov[ri, ci] >= 3)

    return usable


def board_obs_overhead(board: BoardDetector, seconds: float = 1.5) -> tuple[np.ndarray, np.ndarray] | None:
    """Board corners (ids, median pixels) the overhead camera sees over `seconds`; None if it doesn't see it."""
    cam = start_camera("overhead", board)
    try:
        time.sleep(seconds)
        dets = [d for _, d in cam.drain()]
    finally:
        cam.stop()
    if len(dets) < 3:
        return None
    per_id: dict[int, list] = {}
    for ids, uv in dets:
        for k, c in zip(ids, uv):
            per_id.setdefault(int(k), []).append(c)
    keep = np.array(sorted(k for k, v in per_id.items() if len(v) >= len(dets) // 2))
    return keep, np.array([np.median(per_id[k], axis=0) for k in keep])


def board_world(K, D, R_wc, p_wc, board_pts, ids, uv) -> tuple[np.ndarray, np.ndarray] | None:
    """World-from-board pose of a board lying printed side up, from one camera's view (world-from-camera R_wc,
    p_wc); of the two planar PnP solutions, the one facing up."""
    for R_cb, t_cb in board_pnp(K, D, board_pts, ids, uv):
        R_wb = np.asarray(R_wc) @ R_cb
        if -R_wb[2, 2] > 0.7:  # board z points into the board
            return R_wb, np.asarray(R_wc) @ t_cb + np.asarray(p_wc)
    return None


def board_from_overhead(board: BoardDetector, seconds: float = 1.5) -> tuple[np.ndarray, np.ndarray, int] | None:
    """The board's world pose (world-from-board R, origin) measured by the calibrated overhead camera, and the
    number of corners it used; None if the overhead camera isn't calibrated or doesn't see the board."""
    oc = CameraCalib.load("overhead")
    obs = board_obs_overhead(board, seconds) if oc.R is not None else None
    pose = board_world(oc.Km, oc.Dm, oc.R, oc.p, board.corners3d, *obs) if obs is not None else None
    return None if pose is None else (*pose, len(obs[0]))


# ---------------------------------------------------------------------------------------------------- table

def lift_from_table(offsets, overhead: CameraCalib, model, qs, corners, board_pts, ids, uv):
    """The shoulder-lift offset from a board lying on the table (level with the arm's base): the one whose overhead
    camera pose (refitted with that offset from the tag data qs, corners) sees the board level about the lift axis.
    The overhead fit can't see this offset (it trades off against the camera's pitch); a wrong one tilts the whole
    world about the lift axis, the table with it. Returns the offset (rad), that overhead fit and the board pose."""
    from scipy.optimize import brentq

    tag = TagDetector()
    links = Links(model)
    base = np.array(offsets, float)
    cache = {}

    def at(lift: float):
        d = base.copy()
        d[1] = lift
        f = fit_overhead(links, overhead.Km, overhead.Dm, qs, corners, tag.object_points, fixed_offsets=d)
        pose = board_world(overhead.Km, overhead.Dm, f["R"], f["p"], board_pts, ids, uv)
        if pose is None:
            raise SystemExit("the board doesn't come out facing up: is it lying printed side up?")
        cache[lift] = (f, *pose)
        n = -pose[0][:, 2]
        return float(np.arctan2(n[0], n[2]))  # tilt about the lift axis (robot y at pan 0)

    lo, hi = np.radians(-15.0), np.radians(15.0)
    flo, fhi = at(lo), at(hi)
    if flo * fhi > 0:
        raise SystemExit(f"the board's tilt about the lift axis doesn't cross zero for lift offsets within +-15 deg "
                         f"({np.degrees(flo):+.2f} / {np.degrees(fhi):+.2f} deg)")
    lift = brentq(at, lo, hi, xtol=np.radians(0.01))
    if lift not in cache:
        at(lift)
    return (lift, *cache[lift])


def cmd_table(args) -> int:
    oc = CameraCalib.load("overhead")
    if oc.R is None or "extrinsics" not in oc.steps:
        raise SystemExit("calibrate the overhead camera first: calibrate.py overhead")
    arm = ArmCalib.load()
    model = ArmModel().m
    board = BoardDetector(min_corners=8)
    if args.refit:
        raws = sorted(RAW.glob("*_table.npz"))
        if not raws:
            raise SystemExit("no saved table view")
        r = np.load(raws[-1])
        ids, uv = r["ids"], r["uv"]
    else:
        print("Lay the board flat on the table, printed side up, roughly 35-50 cm in front of the arm base, where the "
              "overhead camera sees it; the table under it and under the arm's base must be one flat surface. The arm "
              "doesn't move.")
        if not args.yes:
            input("Enter when the board is there ")
        obs = board_obs_overhead(board, 3.0)
        if obs is None or len(obs[0]) < 12:
            raise SystemExit("the overhead camera doesn't see enough of the board")
        ids, uv = obs
        save_raw("table", ids=ids, uv=uv)
    sessions = overhead_sessions()
    qs = np.concatenate([np.load(p)["qs"] for p in sessions])
    corners = np.concatenate([np.load(p)["corners"] for p in sessions])
    before = board_world(oc.Km, oc.Dm, oc.R, oc.p, board.corners3d, ids, uv)
    if before is not None:
        nb = -before[0][:, 2]
        print(f"now: board tilted {np.degrees(np.arctan2(nb[0], nb[2])):+.2f} deg about the lift axis, "
              f"{np.degrees(np.arctan2(-nb[1], nb[2])):+.2f} deg sideways, centre "
              f"{100 * (before[0] @ board.corners3d.mean(0) + before[1])[2]:.1f} cm above the base plane "
              f"(lift offset {np.degrees(arm.joint_offsets[1]):+.2f} deg)")
    if "lift" in arm.steps:
        print(f"lift offset set from the paddle height ({arm.steps['lift']['lift_offset_deg']:+.2f} deg): only "
              "checked here (the real table under the board came out ~2.6 deg off level with the arm's base)")
        return 0
    lift, f, R_wb, p_wb = lift_from_table(arm.joint_offsets, oc, model, qs, corners, board.corners3d, ids, uv)
    n = -R_wb[:, 2]
    centre = R_wb @ board.corners3d.mean(0) + p_wb
    print(f"lift offset from the table: {np.degrees(lift):+.2f} deg; overhead camera refitted at "
          f"{np.round(f['p'], 3).tolist()} m, RMS {f['rms_px']:.2f} px")
    print(f"checks: the board's centre is {100 * centre[2]:.1f} cm above the base plane (its thickness, a few mm, if "
          f"the base and the board sit on the same flat surface); sideways tilt {np.degrees(np.arctan2(-n[1], n[2])):+.2f}"
          " deg (the table vs the base, side to side; not corrected)")
    oc.R, oc.p = f["R"].tolist(), f["p"].tolist()
    oc.steps["extrinsics"]["lift_from_table"] = True
    arm.joint_offsets = f["offsets"].tolist()
    arm.tag_R, arm.tag_p = f["tag_R"].tolist(), f["tag_p"].tolist()
    arm.steps["table"] = {"date": stamp(), "lift_offset_deg": float(np.degrees(lift)),
                          "board_height_m": float(centre[2]), "side_tilt_deg": float(np.degrees(np.arctan2(-n[1], n[2])))}
    print("saved", oc.save(), arm.save())
    return 0


READY_MAX_AGE_S = 1800.0  # a ruler reading belongs to the hold it was taken in


def cmd_lift(args) -> int:
    """The shoulder-lift offset from the paddle's height at ready, measured with a ruler from the table to the top
    face's centre while `arm_pose.py ready` holds the arm: the overhead fit can't see this offset (it trades off
    against the camera's pitch), and the table and gravity steps measure it less directly. The pose is the one the
    encoders read during that hold (arm_pose.py saves it), not the goal: the servos sag ~1 deg per joint below it."""
    from scipy.optimize import brentq

    oc = CameraCalib.load("overhead")
    if oc.R is None or "extrinsics" not in oc.steps:
        raise SystemExit("calibrate the overhead camera first: calibrate.py overhead")
    holds = sorted(RAW.glob("*_ready.npz"))
    if not holds or time.time() - holds[-1].stat().st_mtime > READY_MAX_AGE_S:
        raise SystemExit("no ready hold from the last 30 min: run `arm_pose.py ready --port <port>`, measure the "
                         "paddle height during the hold, then run this again")
    raw = np.load(holds[-1])["q_raw"]
    arm = ArmCalib.load()
    model = ArmModel()
    prev = np.array(arm.joint_offsets, float)
    h = args.paddle_height_cm / 100.0 + TABLE_Z  # ruler from the table -> robot frame
    tag = TagDetector()
    links = Links(model.m)
    sessions = overhead_sessions()
    qs = np.concatenate([np.load(p)["qs"] for p in sessions])
    corners = np.concatenate([np.load(p)["corners"] for p in sessions])
    lo, hi = np.radians(-20.0), np.radians(30.0)

    def face(off: np.ndarray, lift: float) -> tuple[np.ndarray, float]:
        o = off.copy()
        o[1] = lift
        return model.face_pose(raw + o)

    # The height gives the lift offset for given elbow / wrist offsets, and the overhead refit moves those with the
    # lift offset (the poses differ in pan, so it isn't purely a camera rotation; 0.2 deg per deg): repeat until
    # both agree. One pass from 4 deg away left the lift offset 1 deg off.
    off = prev.copy()
    for _ in range(20):
        lift = brentq(lambda a: face(off, a)[0][2] - h, lo, hi, xtol=np.radians(0.001))
        d = off.copy()
        d[1] = lift
        f = fit_overhead(links, oc.Km, oc.Dm, qs, corners, tag.object_points, fixed_offsets=d)
        settled = np.abs(f["offsets"] - d).max() < np.radians(0.01)
        off = f["offsets"]
        if settled:
            break
    per_3mm = abs(lift - brentq(lambda a: face(off, a)[0][2] - h - 0.003, lo, hi))
    tilt = face(off, lift)[1]
    print(f"{holds[-1].name}: paddle at {args.paddle_height_cm:.1f} cm -> lift offset {np.degrees(lift):+.2f} deg (was "
          f"{np.degrees(prev[1]):+.2f}; +-{np.degrees(per_3mm):.2f} deg per 3 mm of ruler error); during the hold the "
          f"face was tipped {tilt:+.1f} deg (+ = tip up)")
    print(f"overhead camera refitted at {np.round(f['p'], 3).tolist()} m, RMS {f['rms_px']:.2f} px; offsets [deg] "
          + ", ".join(f"{j} {np.degrees(x):+.2f}" for j, x in zip(JOINTS, f["offsets"])))
    oc.R, oc.p = f["R"].tolist(), f["p"].tolist()
    oc.steps["extrinsics"]["lift_from"] = "paddle height"
    arm.joint_offsets = f["offsets"].tolist()
    arm.tag_R, arm.tag_p = f["tag_R"].tolist(), f["tag_p"].tolist()
    arm.steps["lift"] = {"date": stamp(), "method": "paddle height at ready", "paddle_height_cm": args.paddle_height_cm,
                         "hold": holds[-1].name, "lift_offset_deg": float(np.degrees(lift)), "hold_tilt_deg": tilt}
    print("saved", oc.save(), arm.save())
    print("to check it: `arm_pose.py ready` again; the height it prints from the encoders should match the ruler "
          "within ~3 mm")
    return 0


def cmd_face(args) -> int:
    """How far the hitting face really leans forward of where the encoders and the calibration put it
    (`ArmCalib.face_tilt`, which run_robot.py plans with), from a ruler on the blade's front and rear edge while
    `arm_pose.py ready` holds the arm. The tag fit assumes the tag flat on a blade that sits square in its holder;
    1 mm of ruler error on each reading is ~0.5 deg. Measured from where dropped balls go (analyze_run.py) the same
    lean read three times larger on 2026-11-12: the ruler is the reference."""
    holds = sorted(RAW.glob("*_ready.npz"))
    if not holds or time.time() - holds[-1].stat().st_mtime > READY_MAX_AGE_S:
        raise SystemExit("no ready hold from the last 30 min: run `arm_pose.py ready --port <port>`, measure the "
                         "blade's front and rear edge heights during the hold, then run this again")
    arm = ArmCalib.load()
    model = ArmModel()
    model.face_pose(np.load(holds[-1])["q_raw"] + np.array(arm.joint_offsets, float))
    n = model.d.site_xmat[model.face].reshape(3, 3)[:, 2]
    seen = math.atan2(args.rear_cm - args.front_cm, args.length_cm)
    enc = float(n[0] / n[2])
    arm.face_tilt = [seen - enc, arm.face_tilt[1]]
    arm.steps["face_tilt"] = {"date": stamp(), "method": "ruler at the ready hold", "hold": holds[-1].name,
                              "front_cm": args.front_cm, "rear_cm": args.rear_cm, "length_cm": args.length_cm}
    print(f"{holds[-1].name}: the blade leans {math.degrees(seen):+.2f} deg forward by the ruler, {math.degrees(enc):+.2f} "
          f"by the encoders (the servos' sag): face tilt {math.degrees(seen - enc):+.2f} deg (+-"
          f"{math.degrees(math.atan2(0.14, args.length_cm)):.1f} deg for 1 mm on each reading)")
    side = (args.left_cm, args.right_cm, args.width_cm)
    if any(v is not None for v in side):
        if any(v is None for v in side):
            raise SystemExit("--left-cm, --right-cm and --width-cm go together")
        roll_from_ruler(args, arm, model, np.load(holds[-1])["q_raw"], holds[-1].name)
    print("saved", arm.save())
    return 0


def roll_from_ruler(args, arm: ArmCalib, model: ArmModel, q_raw: np.ndarray, hold: str) -> None:
    """The wrist_roll zero from a ruler on the blade's left and right edge at the ready hold. The tag fit gets it
    from a tag assumed flat on the blade; a ball dropped on the waiting blade then left sideways at ~0.12 m/s
    whichever side it landed (2026-11-13), as off a face leaning ~2-3 deg. Unlike the forward lean (a sag the
    policy is trained with) this is a plain offset of the joint, so it goes into `joint_offsets`: every mode gets
    a level blade. The wrist camera mount and the tag placement are turned back by the same angle about the roll
    axis (they were fitted with the old zero and the camera has not moved). `calibrate.py overhead` refits the
    zero from the tag: run this again after it."""
    from scipy.optimize import brentq

    j = JOINTS.index("wrist_roll")
    off = np.array(arm.joint_offsets, float)

    def lean(d: float) -> float:  # the face normal's lean toward +y (left) with the roll d further than believed
        q = q_raw + off
        q[j] += d
        model.face_pose(q)
        n = model.d.site_xmat[model.face].reshape(3, 3)[:, 2]
        return float(n[1] / n[2])

    seen = math.atan2(args.right_cm - args.left_cm, args.width_cm)  # left edge lower: the normal leans left
    enc = lean(0.0)
    d = brentq(lambda x: lean(x) - seen, math.radians(-15), math.radians(15), xtol=1e-6)
    move_roll_zero(arm, model, d, {"hold": hold, "left_cm": args.left_cm, "right_cm": args.right_cm,
                                   "width_cm": args.width_cm})
    print(f"{hold}: the blade leans {math.degrees(seen):+.2f} deg to the left by the ruler, {math.degrees(enc):+.2f} by the "
          f"encoders (+-{math.degrees(math.atan2(0.14, args.width_cm)):.1f} deg for 1 mm on each reading). The arm now "
          "holds the blade level; check with the ruler at the next hold.")


def move_roll_zero(arm: ArmCalib, model: ArmModel, d: float, record: dict) -> None:
    """Move the wrist_roll zero by d [rad] (the true roll is d further than was believed) in `arm` (not saved
    here), and turn the wrist camera mount (saved) and the tag placement back by d about the roll axis."""
    j = JOINTS.index("wrist_roll")
    jid = model.m.joint("wrist_roll").id
    axis, c = model.m.jnt_axis[jid].astype(float), model.m.jnt_pos[jid].astype(float)
    back = cv2.Rodrigues(-d * axis / np.linalg.norm(axis))[0]  # the link frame turns by d: what is fixed to the paddle turns back
    wrist = CameraCalib.load("wrist")
    if wrist.R is not None:
        wrist.R, wrist.p = (back @ np.array(wrist.R)).tolist(), (c + back @ (np.array(wrist.p) - c)).tolist()
        wrist.steps["mount"]["roll_zero_moved_deg"] = wrist.steps["mount"].get("roll_zero_moved_deg", 0.0) + math.degrees(d)
        print("saved", wrist.save())
    if arm.tag_R is not None:
        arm.tag_R, arm.tag_p = (back @ np.array(arm.tag_R)).tolist(), (c + back @ (np.array(arm.tag_p) - c)).tolist()
    off = np.array(arm.joint_offsets, float)
    off[j] += d
    arm.joint_offsets = off.tolist()
    arm.face_tilt = [arm.face_tilt[0], 0.0]
    moves = arm.steps.setdefault("roll_zero_moves", [])
    moves.append({"date": stamp(), "moved_deg": math.degrees(d), **record})
    print(f"wrist_roll zero moved by {math.degrees(d):+.2f} deg (offset now {math.degrees(off[j]):+.2f})")


def cmd_roll(args) -> int:
    """Move the wrist_roll zero by a given angle: the fine adjustment after `face --left-cm ...`, from which way
    dropped balls leave the waiting blade (analyze_run.py: they go to the side the face leans to). Positive turns
    the held blade's left edge up."""
    arm = ArmCalib.load()
    move_roll_zero(arm, ArmModel(), math.radians(args.left_up_deg), {"why": args.why})
    print("saved", arm.save())
    return 0


def capture_wrist(args, calib, board, arm, model, links, R_lc, p_lc, usable):
    """The wrist mount step's arm run: board placement (measured by the overhead camera when it sees it), planned
    poses, board corners at each. Returns qs (true angles), views [(ids, uv)], the board pose used for planning and
    whether the overhead camera measured it."""
    R_wb, p_wb = table_board(board.corners3d)
    print(f"Lay the board flat on the table, printed side up, its centre about {100 * WRIST_BOARD_CENTRE[0]:.0f} cm "
          "in front of the arm base's axis on the arm's centre line, long side across (left-right). Tape it down or "
          "weigh the margins if it curls; nothing may stick up above it. Don't move it until `timing wrist` is done.")
    if not args.yes:
        input("Enter when the board is there (the arm in its folded rest, out of the overhead camera's way) ")
    seen = board_from_overhead(board)
    if seen is None:
        print("the overhead camera doesn't see the board: planning for the spot above instead")
    else:
        R_wb, p_wb, n_c = seen
        centre = R_wb @ board.corners3d.mean(0) + p_wb
        print(f"overhead camera: board centre at x {100 * centre[0]:.1f} cm, y {100 * centre[1]:+.1f} cm, "
              f"z {100 * centre[2]:.1f} cm ({n_c} corners); planning the poses around it")
    camera = {"K": calib.Km, "D": calib.Dm, "R": R_lc, "p": p_lc, "board_R": R_wb, "board_p": p_wb,
              "board_pts": board.corners3d, "usable": usable, "offsets": arm.joint_offsets}
    poses = wrist_poses(links, model, camera, n=args.poses)
    cam = start_camera("wrist", board)
    qs, views = [], []
    try:
        with ArmSession(open_bus(args.port), joint_offsets=arm.joint_offsets) as s:
            check_tour(s, poses)
            for i, q in enumerate(poses):
                s.move_to(q)
                hold(s, 1.5)  # the arm hunts ~+-1 deg for over a second after a move
                cam.drain()
                rows = hold(s, 1.0)
                dets = [d for _, d in cam.drain()]
                if len(dets) < 5:
                    print(f"  pose {i + 1}/{len(poses)}: board seen in {len(dets)} frames, skipped (picture: "
                          f"{skipped_picture(cam, 'wrist', i)})")
                    continue
                per_id: dict[int, list] = {}
                for ids, uv in dets:
                    for k, c in zip(ids, uv):
                        per_id.setdefault(int(k), []).append(c)
                keep = sorted(k for k, v in per_id.items() if len(v) >= len(dets) // 2)
                qs.append(np.mean([r[1] for r in rows], axis=0))
                views.append((np.array(keep), np.array([np.median(per_id[k], axis=0) for k in keep])))
                print(f"  pose {i + 1}/{len(poses)}: {len(keep)} corners")
            s.move_to(s.start)
    finally:
        cam.stop()
    return qs, views, R_wb, p_wb, seen


def cmd_wrist(args) -> int:
    calib = CameraCalib.load("wrist")
    if "intrinsics" not in calib.steps:
        raise SystemExit("calibrate the wrist camera's lens first: calibrate.py intrinsics wrist")
    board = BoardDetector(min_corners=8)
    arm = ArmCalib.load()
    model = ArmModel()
    links = Links(model.m)
    R_lc, p_lc = (np.array(calib.R), np.array(calib.p)) if calib.R is not None else (links.cad_cam_R, links.cad_cam_p)
    usable = wrist_usable(calib)
    if args.refit:
        # Re-solve the newest recording with the current joint offsets (the lift offset was set after it).
        raws = sorted(p for p in RAW.glob("*_wrist.npz"))
        r = np.load(raws[-1], allow_pickle=True) if raws else None
        if r is None or "offsets" not in r:
            raise SystemExit("no wrist recording with its joint offsets saved")
        qs = list(np.asarray(r["qs"]) - r["offsets"] + np.array(arm.joint_offsets))
        views = [(np.asarray(i), np.asarray(u)) for i, u in zip(r["ids"], r["uv"])]
        m0 = calib.steps.get("mount", {})
        R_wb, p_wb = ((np.array(m0["board_R"]), np.array(m0["board_p"])) if "board_R" in m0
                      else table_board(board.corners3d))
        seen, raw = None, raws[-1]
        print(f"refitting {raw.name}: {len(views)} views")
    else:
        qs, views, R_wb, p_wb, seen = capture_wrist(args, calib, board, arm, model, links, R_lc, p_lc, usable)
        raw = save_raw("wrist", qs=np.array(qs), ids=[v[0] for v in views], uv=[v[1] for v in views],
                       offsets=np.array(arm.joint_offsets))
    # Only corners where the lens model is calibrated and the blade doesn't cut the board.
    views_ok, qs_ok = [], []
    for q, (ids, uv) in zip(qs, views):
        m = usable(uv)
        if m.sum() >= 8:
            views_ok.append((ids[m], uv[m]))
            qs_ok.append(q)
    if len(views_ok) < 8:
        raise SystemExit(f"board usable at only {len(views_ok)} poses (need >= 8): is it where the step said?")
    qs_ok = np.array(qs_ok)
    fit = fit_wrist(links, calib.Km, calib.Dm, qs_ok, views_ok, board.corners3d, mount0=(R_lc, p_lc),
                    board0=(R_wb, p_wb))
    d_p = 1e3 * np.linalg.norm(fit["p"] - links.cad_cam_p)
    print(f"RMS {fit['rms_px']:.2f} px over {len(views_ok)} poses; mount {d_p:.1f} mm / "
          f"{angle_deg(fit['R'], links.cad_cam_R):.2f} deg from CAD")
    if seen is not None:
        c_fit = fit["board_R"] @ board.corners3d.mean(0) + fit["board_p"]
        c_ov = R_wb @ board.corners3d.mean(0) + p_wb
        print(f"board as the wrist fit puts it vs as the overhead camera measured it: {1e3 * np.linalg.norm(c_fit - c_ov):.1f}"
              f" mm / {angle_deg(fit['board_R'], R_wb):.2f} deg apart (the two calibrations agree when these are small)")
    print("per pose: fit error [px] / median disagreement between the arm's and the camera's turn to the other "
          "poses [deg] (the arm's own pose error is ~1-2 deg; much more means the board or the camera moved):")
    for i, (e, dis) in enumerate(zip(fit["per_view_px"], fit["view_disagree_deg"])):
        print(f"  {i + 1:2d}: {e:5.1f} px  {dis:4.1f} deg" + ("   <-- moved?" if dis > 3.0 else ""))
    med = float(np.median(fit["view_disagree_deg"]))
    if med > 2.5 or fit["rms_px"] > 12.0:
        raise SystemExit(f"fit error {fit['rms_px']:.1f} px, median disagreement {med:.1f} deg: not saved. Check the "
                         "board lies still and the wrist camera and paddle don't wiggle in the holder, then run again.")
    calib.R, calib.p = fit["R"].tolist(), fit["p"].tolist()
    calib.steps["mount"] = {"date": stamp(), "poses": len(views_ok), "rms_px": fit["rms_px"], "raw": raw.name,
                            "disagree_deg_median": med,
                            "board_R": fit["board_R"].tolist(), "board_p": fit["board_p"].tolist(),
                            "pose_q": qs_ok.tolist()}
    print("saved", calib.save())
    return 0


# ---------------------------------------------------------------------------------------------------- timing

def wave(q0: np.ndarray, seconds: float) -> np.ndarray:
    """Sinusoids on pan (0.9 Hz, 0.10 rad) and wrist_flex (1.4 Hz, 0.15 rad) around q0, tapered in and out: the
    target moves ~0.2-0.3 m/s, so 1 ms of timestamp error is ~0.3 px."""
    t = np.arange(int(seconds * RATE)) / RATE
    taper = np.clip(np.minimum(t, t[-1] - t) / 1.0, 0, 1)
    q = np.repeat(q0[None], len(t), axis=0)
    q[:, 0] += 0.10 * taper * np.sin(2 * np.pi * 0.9 * t)
    q[:, 3] += 0.15 * taper * np.sin(2 * np.pi * 1.4 * t)
    return q


def cmd_timing(args) -> int:
    name = args.camera
    calib = CameraCalib.load(name)
    arm = ArmCalib.load()
    model = ArmModel()
    links = Links(model.m)

    def wave_clear(q: np.ndarray) -> bool:
        """The wave around q keeps the payload >= 4 cm above the table (the arm's own check wants 2) and the blade
        clear of the arm."""
        for qq in wave(q, args.seconds)[::10]:
            model.forward(qq)
            if model.clearance() < 0.04 or model.blade_hits():
                return False
        return True

    if name == "overhead":
        if calib.R is None or arm.tag_R is None:
            raise SystemExit("run `calibrate.py overhead` first")
        det = TagDetector()
        R_wc, p_wc = np.array(calib.R), np.array(calib.p)
        R_lt, p_lt = np.array(arm.tag_R), np.array(arm.tag_p)
        poses = np.array(arm.steps["overhead"]["poses"]) + np.array(arm.joint_offsets)  # raw -> true angles
        # The pose whose tag is nearest the image centre, of those whose wave stays clear of the table.
        centre = np.array([calib.width / 2, calib.height / 2])
        order = np.argsort([np.linalg.norm(project(calib.Km, calib.Dm, R_wc, p_wc, _tag(
            links, q, R_lt, p_lt, det.object_points)).mean(0) - centre) for q in poses])
        safe = [poses[i] for i in order if wave_clear(poses[i])]
        if not safe:
            raise SystemExit("no overhead-step pose has room for the wave above the table")
        q0 = safe[0]

        def predict(q, extra, k):
            return project(calib.Km, calib.Dm, R_wc, p_wc, _tag(links, q, R_lt, p_lt, det.object_points))
    else:
        if calib.R is None or "mount" not in calib.steps:
            raise SystemExit("run `calibrate.py wrist` first (and leave the board where it was)")
        det = BoardDetector(min_corners=8)
        R_lc, p_lc = np.array(calib.R), np.array(calib.p)
        m = calib.steps["mount"]
        safe = [q for q in np.array(m["pose_q"]) if wave_clear(q)]
        if not safe:
            raise SystemExit("no wrist-step pose has room for the wave above the table")
        q0 = safe[0]
    cam = start_camera(name, det)
    rows: list[tuple[float, np.ndarray]] = []
    try:
        with ArmSession(open_bus(args.port), joint_offsets=arm.joint_offsets) as s:
            # Check the whole thing (there, the wave, back) before moving at all.
            motion = wave(s.clamp(q0), args.seconds)
            b = PlanBuilder(s.read().q)
            b.move(motion[0], name="from_start")
            b.add(motion, name="wave")
            b.move(s.start, name="to_start")
            problems = s.check(b.build())
            if problems:
                raise SystemExit("wave check failed (the arm didn't move):\n  " + "\n  ".join(problems))
            s.move_to(motion[0])
            hold(s, 0.5)
            cam.drain()
            dets = []
            dropped0 = cam.dropped

            def tick(k, g, r):  # drain every tick: the camera's result queue holds only 64
                rows.append((r.t, r.q))
                dets.extend(cam.drain())

            s.follow(motion, on_tick=tick)
            dets.extend(cam.drain())
            lost = cam.dropped - dropped0  # before the move back: nobody drains the queue then
            s.move_to(s.start)
    finally:
        cam.stop()
    if lost:
        print(f"warning: {lost} detections lost to a full camera queue during the motion")
    t_enc, q_enc = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    dets = [(st, d) for st, d in dets if t_enc[0] + 0.2 < st < t_enc[-1] - 0.2]
    print(f"{len(dets)} frames with the target during the {args.seconds:.0f} s motion "
          f"({len(dets) / max(t_enc[-1] - t_enc[0] - 0.4, 1e-6):.1f} per second; the camera runs at ~29.5)")
    raw = save_raw(f"timing_{name}", t_enc=t_enc, q_enc=q_enc, stamps=np.array([st for st, _ in dets]),
                   observed=[d for _, d in dets], offsets=np.array(arm.joint_offsets))
    if len(dets) < 60:
        raise SystemExit(f"only {len(dets)} frames with the target during the motion")
    stamps = np.array([st for st, _ in dets])
    if name == "overhead":
        observed = [d for _, d in dets]
        rows_f = np.array([d[:, 1].mean() / calib.height for d in observed])
        fit = fit_timing(t_enc, q_enc, stamps, rows_f, observed, predict)
    else:
        board_pts = det.corners3d
        observed = [uv for _, (ids, uv) in dets]
        idss = [ids for _, (ids, uv) in dets]
        rows_f = np.array([uv[:, 1].mean() / calib.height for uv in observed])

        def predict(q, extra, k):
            R_l, p_l = links(q)
            R_wb, p_wb = rot(extra[:3]), extra[3:]
            return project(calib.Km, calib.Dm, R_l[0] @ R_lc, R_l[0] @ p_lc + p_l[0], board_pts[idss[k]] @ R_wb.T + p_wb)

        extra0 = np.concatenate([rvec(np.array(m["board_R"])), np.array(m["board_p"])])
        fit = fit_timing(t_enc, q_enc, stamps, rows_f, observed, predict, extra0=extra0)
    print(f"{name}: capture = buffer stamp {1e3 * fit['offset']:+.1f} ms + {1e3 * fit['readout']:.1f} ms x row; RMS "
          f"{fit['rms_px']:.2f} px over {len(stamps)} frames (at the target's mean row {rows_f.mean():.2f}: "
          f"{1e3 * (fit['offset'] + fit['readout'] * rows_f.mean()):+.1f} ms, the well-determined part)")
    calib.stamp_offset, calib.readout = fit["offset"], fit["readout"]
    calib.steps["timing"] = {"date": stamp(), "frames": len(stamps), "rms_px": fit["rms_px"], "raw": raw.name}
    print("saved", calib.save())
    return 0


def _tag(links: Links, q: np.ndarray, R_lt, p_lt, tag_pts) -> np.ndarray:
    R_l, p_l = links(q)
    return tag_pts @ (R_l[0] @ R_lt).T + R_l[0] @ p_lt + p_l[0]


# ---------------------------------------------------------------------------------------------------- gravity

def split_arcs(t: np.ndarray, X: np.ndarray, min_points: int = 6, min_s: float = 0.12) -> list[np.ndarray]:
    """Indices of free-flight arcs: split at gaps > 0.1 s and at bounces (a height minimum with the motion turning
    from down to up)."""
    cuts = [0]
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > 0.1:
            cuts.append(i)
        elif 1 < i < len(t) - 2 and X[i, 2] < X[i - 2, 2] and X[i, 2] < X[i + 2, 2] and X[i, 2] < 0.35:
            cuts.append(i)
    cuts.append(len(t))
    arcs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        idx = np.arange(a + 1, b - 1)  # drop the samples at the contacts
        if len(idx) >= min_points and t[idx[-1]] - t[idx[0]] >= min_s:
            arcs.append(idx)
    return arcs


class EncoderPoses:
    """The wrist camera's pose at any time of a recording, for `to_measurements`: encoder angles interpolated at that
    time, then forward kinematics. (An ArmHistory holds only the last 0.32 s and assumes evenly spaced ticks: fed a
    whole recording first, it gave every detection the pose at the end.)"""

    def __init__(self, kin: ArmKinematics, enc_t: np.ndarray, enc_q: np.ndarray):
        self.kin, self.t, self.q = kin, np.asarray(enc_t, float), np.asarray(enc_q, float)

    def at(self, t: torch.Tensor, keys=None) -> dict[str, torch.Tensor]:
        tt = float(t[0])
        q = torch.tensor([[np.interp(tt, self.t, self.q[:, j]) for j in range(self.q.shape[1])]], dtype=torch.float32)
        R, p = self.kin.wrist_camera(q)
        return {"wR": R, "wp": p}


def gravity_measurements(dets: np.ndarray, cams: list[str], enc_t: np.ndarray, enc_q: np.ndarray,
                         calibs: dict[str, CameraCalib], model) -> dict[str, np.ndarray]:
    """Raw detections (N, 5: camera index, stamp, u, v, d; times on the encoder clock) of the cameras in `calibs`
    -> measurement arrays t, R, p, z, sigma, cam (index into list(calibs)), sorted by time; wrist poses from the
    encoder log (interpolated at each capture time) through the arm model."""
    from pingpong_sim.real.camera_model import PixelDetection

    kin = ArmKinematics(model, prefix="")
    if "wrist" in calibs:
        apply_wrist_mount(kin, calibs["wrist"])
    poses = EncoderPoses(kin, enc_t, enc_q)
    parts = []
    for ci, name in enumerate(calibs):
        pix = [PixelDetection(name, *map(float, r[1:])) for r in dets if cams[int(r[0])] == name]
        if not pix:
            continue
        m = to_measurements(pix, {name: calibs[name]}, poses)
        part = {k: m[k][:, 0].numpy().astype(float) for k in ("t", "R", "p", "z", "sigma")}
        part["cam"] = np.full(len(part["t"]), ci)
        parts.append(part)
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("t", "R", "p", "z", "sigma", "cam")}
    order = np.argsort(out["t"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def gravity_arcs(m: dict) -> list[np.ndarray]:
    u = np.concatenate([m["z"][:, :2], np.ones((len(m["t"]), 1))], 1)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    X = m["p"] + np.einsum("nij,nj->ni", m["R"], (0.02 / np.sin(m["z"][:, 2] / 2))[:, None] * u)
    return split_arcs(m["t"], X)


def arc_list(m: dict, arcs: list[np.ndarray]) -> list[dict]:
    return [{k: m[k][i] for k in ("t", "R", "p", "z", "sigma", "cam")} for i in arcs]


def fit_g(m: dict, arcs: list[np.ndarray], n_cams: int = 1) -> dict:
    """Gravity direction with |g| = 9.81 and a size scale per camera."""
    return fit_gravity(arc_list(m, arcs), fix_magnitude=True, n_cams=n_cams)


def tilts_deg(g: np.ndarray) -> tuple[float, float]:
    """Gravity's tilt about the robot's forward (x) axis and about the lift (y) axis."""
    return float(np.degrees(np.arctan2(g[1], -g[2]))), float(np.degrees(np.arctan2(-g[0], -g[2])))


def load_gravity_recordings(newest_only: bool = False) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """All gravity recordings pooled (each shifted by 1000 s; `newest_only`: just the last): detections (N, 5,
    camera indices into the returned names), camera names, encoder times and raw angles."""
    names: list[str] = []
    D, T, Q = [], [], []
    paths = sorted(RAW.glob("*_gravity.npz"))
    for k, path in enumerate(paths[-1:] if newest_only else paths):
        g = np.load(path)
        rec = [str(n) for n in g["cameras"]]
        for n in rec:
            if n not in names:
                names.append(n)
        d = g["dets"].copy()
        d = d[(d[:, 4] >= MIN_BALL_PX) & (d[:, 4] <= MAX_BALL_PX)]  # recordings from before the size limits
        d[:, 0] = [names.index(rec[int(i)]) for i in d[:, 0]]
        d[:, 1] += 1000.0 * k
        D.append(d)
        T.append(g["enc_t"] + 1000.0 * k)
        Q.append(g["enc_q"])
    return np.concatenate(D), names, np.concatenate(T), np.concatenate(Q)


def cmd_gravity(args) -> int:
    arm = ArmCalib.load()
    model = ArmModel().m
    if args.cameras is None:
        args.cameras = [n for n in ("overhead", "wrist") if CameraCalib.path(n).exists() and (
            (c := CameraCalib.load(n)).R is not None and (n == "overhead" or "disagree_deg_median" in c.steps.get(
                "mount", {})))]  # a wrist mount from the checked (flat-board) step only
    if not args.refit:
        calibs = {n: CameraCalib.load(n) for n in args.cameras}
        for n, c in calibs.items():
            if c.R is None:
                raise SystemExit(f"{n}: no pose yet (calibrate.py overhead / wrist)")
        cams = BallCameras(RigConfig.load(), args.cameras)
        enc, dets = [], []
        with ArmSession(open_bus(args.port), joint_offsets=arm.joint_offsets) as s:
            s.move_to(Q_READY)
            cams.start()
            t0 = s.bus.now()
            print(f"Drop the ball straight down from rest, 20-30 cm above the paddle (open your fingers, no throw, no "
                  f"spin), let it bounce once or twice, catch it, repeat at different spots over the paddle; "
                  f"{args.seconds:.0f} s. (Tosses across the picture bias it: the overhead camera's range from the "
                  "ball's size drifts across its picture.)")
            try:
                def tick(k, goal, r):
                    enc.append((r.t - t0, r.q - np.asarray(arm.joint_offsets)))  # raw angles: offsets may change
                    dets.extend(cams.drain())

                s.follow(np.repeat(Q_READY[None], int(args.seconds * RATE), axis=0), on_tick=tick)
            finally:
                cams.stop()
            s.move_to(s.start)
        names = list(args.cameras)
        save_raw("gravity", cameras=np.array(names), enc_t=np.array([e[0] for e in enc]),
                 enc_q=np.array([e[1] for e in enc]),
                 dets=np.array([(names.index(d.camera), d.stamp - t0, d.u, d.v, d.d) for d in dets]).reshape(-1, 5))
    dets, names, enc_t, enc_q_raw = load_gravity_recordings(newest_only=args.fresh)
    use = [n for n in names if n in args.cameras]
    calibs = {n: CameraCalib.load(n) for n in use}
    m = gravity_measurements(dets, names, enc_t, enc_q_raw + np.array(arm.joint_offsets), calibs, model)
    arcs = gravity_arcs(m)
    if len(arcs) < 3:
        raise SystemExit(f"only {len(arcs)} usable ball flights in {len(m['t'])} detections")
    lift_src = arm.steps.get("lift") or arm.steps.get("table")
    if lift_src and not args.no_lift:
        print(f"lift offset {lift_src['lift_offset_deg']:+.2f} deg (from the {'paddle height' if 'lift' in arm.steps else 'table'})"
              " kept: much more precise than ball flights; gravity only checks it")
    if not args.no_lift and "overhead" in calibs and not lift_src:
        sessions = overhead_sessions()
        qs = np.concatenate([np.load(p)["qs"] for p in sessions])
        corners = np.concatenate([np.load(p)["corners"] for p in sessions])
        others = {n: c for n, c in calibs.items() if n != "overhead"}
        lift, f = lift_from_gravity(arm, calibs["overhead"], dets, names, enc_t, enc_q_raw, model, qs, corners,
                                    others=others)
        oc = calibs["overhead"]
        oc.R, oc.p = f["R"].tolist(), f["p"].tolist()
        oc.steps["extrinsics"]["lift_from_gravity"] = True
        arm.joint_offsets = f["offsets"].tolist()
        arm.tag_R, arm.tag_p = f["tag_R"].tolist(), f["tag_p"].tolist()
        arm.steps["gravity"] = {"date": stamp(), "lift_offset_deg": float(np.degrees(lift))}
        oc.save()
        arm.save()
        calibs = {n: CameraCalib.load(n) for n in use}
        m = gravity_measurements(dets, names, enc_t, enc_q_raw + np.array(arm.joint_offsets), calibs, model)
        arcs = gravity_arcs(m)
    fit = fit_g(m, arcs, n_cams=len(calibs))
    tx, ty = tilts_deg(fit["g"])
    # Uncertainty: leave one flight out.
    jk = []
    for i in range(len(arcs)):
        fi = fit_g(m, [a for j, a in enumerate(arcs) if j != i], n_cams=len(calibs))
        jk.append([*tilts_deg(fi["g"]), *fi["size_scale"]])
    jk = np.array(jk)
    se = np.sqrt(len(arcs) - 1) * jk.std(axis=0)
    print(f"gravity from {len(arcs)} flights ({', '.join(use)}; |g| held at 9.81): tilt about the forward axis "
          f"{tx:+.2f} +- {se[0]:.2f} deg, about the lift axis {ty:+.2f} +- {se[1]:.2f} deg; normalized RMS per "
          f"flight median {np.median(fit['per_arc_rms']):.2f}")
    for i, n in enumerate(calibs):
        c = calibs[n]
        c.size_scale = float(c.size_scale * fit["size_scale"][i])
        c.steps["size_scale"] = {"date": stamp(), "flights": len(arcs), "uncertainty": float(se[2 + i])}
        c.save()
        print(f"  {n}: detected ball diameter / true {c.size_scale:.3f} +- {se[2 + i]:.3f} (saved; ranges from the "
              "apparent size use it)")
    if "gravity" in arm.steps:
        arm.steps["gravity"]["lift_uncertainty_deg"] = float(se[1])
        arm.save()
    worst = max(abs(tx) + se[0], abs(ty) + se[1])
    print("targets: tilts < 0.5 deg. " + ("Met." if worst < 0.5 else
          "Not met yet: record more flights (`gravity` pools every recording) or, if the forward-axis tilt is "
          "clearly nonzero, level the arm base / table side to side."))
    return 0


def lift_from_gravity(arm: ArmCalib, overhead: CameraCalib, dets, names, enc_t, enc_q_raw, model, qs, corners,
                      others: dict | None = None):
    """The shoulder-lift offset the overhead calibration can't see: the one whose overhead camera pose (refitted
    with that offset from the tag data qs, corners) makes the ball flights fall straight down about the lift axis.
    `others`: more calibrated cameras (wrist: its pose follows the arm, so it moves with the offset too); with two
    cameras the flights are triangulated instead of ranged from the ball's apparent size. Returns the offset (rad)
    and that overhead fit."""
    from scipy.optimize import brentq

    tag = TagDetector()
    links = Links(model)
    base = np.array(arm.joint_offsets)
    others = others or {}
    cache = {}

    def at(lift: float):
        d = base.copy()
        d[1] = lift
        f = fit_overhead(links, overhead.Km, overhead.Dm, qs, corners, tag.object_points, fixed_offsets=d)
        c = CameraCalib(**{**overhead.__dict__, "R": f["R"].tolist(), "p": f["p"].tolist()})
        cals = {"overhead": c, **others}
        m = gravity_measurements(dets, names, enc_t, enc_q_raw + f["offsets"], cals, model)
        g = fit_g(m, gravity_arcs(m), n_cams=len(cals))["g"]
        cache[lift] = f
        return float(np.arctan2(-g[0], -g[2]))  # tilt about the lift axis (robot y at pan 0)

    lo, hi = np.radians(-12.0), np.radians(12.0)
    flo, fhi = at(lo), at(hi)
    if flo * fhi > 0:
        raise SystemExit(f"gravity tilt about the lift axis doesn't cross zero for lift offsets within +-12 deg "
                         f"({np.degrees(flo):+.2f} / {np.degrees(fhi):+.2f} deg)")
    lift = brentq(at, lo, hi, xtol=np.radians(0.02))
    if lift not in cache:
        at(lift)
    f = cache[lift]
    print(f"lift offset from gravity: {np.degrees(lift):+.2f} deg (camera refitted: {np.round(f['p'], 3).tolist()} m, "
          f"RMS {f['rms_px']:.2f} px)")
    return lift, f


STATIC_X = READY_X  # m, blade centre over the bounce target


def cmd_static(args) -> int:
    """Resting-ball check of each camera's ball position, especially the range from the apparent size: the ball
    rests on the level blade at several heights (true centre from the arm's kinematics), then, with `--free`, sits
    on a support you measure (true height from the ruler; the blade is out of the picture). Separates a range error
    that depends on distance (the colour mask's fringe matters more on a smaller blob) from one that only shows
    with the blade under the ball (blade pixels joining the blob)."""
    arm = ArmCalib.load()
    names = [n for n in ("overhead", "wrist") if CameraCalib.path(n).exists() and CameraCalib.load(n).R is not None]
    if not args.refit:
        model = ArmModel()
        cams = BallCameras(RigConfig.load(), names)
        enc, dets, segs = [], [], []
        with ArmSession(open_bus(args.port), joint_offsets=arm.joint_offsets) as s:
            t0 = s.bus.now()

            def record(q, kind, value):
                t_a = s.bus.now() - t0
                cams.drain()

                def tick(k, goal, r):
                    enc.append((r.t - t0, r.q - np.asarray(arm.joint_offsets)))
                    dets.extend(cams.drain())

                s.follow(np.repeat(q[None], int(args.seconds * RATE), axis=0), on_tick=tick)
                segs.append((t_a, s.bus.now() - t0, kind, value))

            cams.start()
            try:
                seed = Q_READY
                for h in args.heights:
                    q = model.ik(STATIC_X, h, seed)
                    seed = q
                    s.move_to(q)
                    input(f"blade level at {100 * h:.1f} cm: set the ball on the blade centre, let it settle, hands "
                          "out of the picture, then Enter ")
                    record(q, 0, h)
                    print(f"  recorded {args.seconds:.0f} s")
                if args.free:
                    s.move_to(s.start)
                    print("Blade away. Now the ball on a support (a cup, a tube) under the overhead camera, around "
                          "where the blade was, at different heights; measure from the table to the ball's top.")
                    while (ans := input("ball top above the table [cm] (blank: done): ").strip()):
                        record(s.read().q, 1, float(ans) / 100 + TABLE_Z - BALL_RADIUS)  # centre, robot frame
                        print(f"  recorded {args.seconds:.0f} s")
            finally:
                cams.stop()
            s.move_to(s.start)
        save_raw("static", cameras=np.array(names), enc_t=np.array([e[0] for e in enc]),
                 enc_q=np.array([e[1] for e in enc]), segments=np.array(segs).reshape(-1, 4),
                 dets=np.array([(names.index(d.camera), d.stamp - t0, d.u, d.v, d.d) for d in dets]).reshape(-1, 5))
    paths = sorted(RAW.glob("*_static.npz"))
    if not paths:
        raise SystemExit("no static recording yet")
    report_static(np.load(paths[-1]), arm)
    return 0


def report_static(z, arm: ArmCalib) -> None:
    names = [str(n) for n in z["cameras"]]
    calibs = {n: CameraCalib.load(n) for n in names}
    model = ArmModel()
    enc_t, enc_q = z["enc_t"], z["enc_q"] + np.array(arm.joint_offsets)
    m = gravity_measurements(z["dets"], names, enc_t, enc_q, calibs, model.m)
    u = np.concatenate([m["z"][:, :2], np.ones((len(m["t"]), 1))], 1)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    d = np.einsum("nij,nj->ni", m["R"], u)  # world ray directions
    rng = BALL_RADIUS / np.sin(m["z"][:, 2] / 2)  # range from the apparent size (size_scale applied)
    det_t, det_cam, det_px = z["dets"][:, 1], np.array([names[int(i)] for i in z["dets"][:, 0]]), z["dets"][:, 4]
    print("per segment and camera: detections, ball diameter [px], true / size-based range, position error "
          "measured - true [mm] (x, y, z), ray direction error [deg]")
    rows = []
    for t_a, t_b, kind, value in z["segments"]:
        w_enc = (enc_t > t_a + 0.5) & (enc_t < t_b)
        lab = f"blade at {100 * value:.1f} cm" if kind == 0 else f"support, centre at {100 * value:.1f} cm"
        if kind == 0:
            cs = []
            for q in enc_q[w_enc]:
                model.forward(q)
                cs.append(model.d.site_xpos[model.face] + BALL_RADIUS * model.d.site_xmat[model.face].reshape(3, 3)[:, 2])
            c_true = np.median(cs, axis=0)
        for ci, name in enumerate(calibs):
            sel = (m["cam"] == ci) & (m["t"] > t_a + 0.5) & (m["t"] < t_b)
            if sel.sum() < 5:
                print(f"  {lab:28s} {name:8s}: {sel.sum()} detections")
                continue
            p_cam, dd, r_m = m["p"][sel], d[sel], rng[sel]
            d_px = np.median(det_px[(det_cam == name) & (det_t > t_a + 0.5) & (det_t < t_b)])
            X = np.median(p_cam + r_m[:, None] * dd, axis=0)
            if kind == 0:
                to = c_true - np.median(p_cam, axis=0)
                r_t = np.linalg.norm(to)
                ray = np.degrees(np.arccos(np.clip(np.median(dd, axis=0) / np.linalg.norm(np.median(dd, axis=0)) @ to / r_t, -1, 1)))
                err = 1e3 * (X - c_true)
            else:
                s_t = (value - p_cam[:, 2]) / dd[:, 2]  # along the ray to the measured height
                r_t, ray = float(np.median(s_t)), float("nan")
                err = np.array([np.nan, np.nan, 1e3 * (X[2] - value)])
            ratio = r_t / np.median(r_m)
            rows.append((name, kind, r_t, ratio))
            print(f"  {lab:28s} {name:8s}: {sel.sum():4d} det, {d_px:5.1f} px, range {1e3 * r_t:4.0f} mm, true/size "
                  f"{ratio:.3f}, error {err[0]:+5.1f} {err[1]:+5.1f} {err[2]:+5.1f}, ray {ray:.2f} deg")
    for name in calibs:
        for kind, what in ((0, "on the blade"), (1, "on a support")):
            r = np.array([(rt, ra) for n, k, rt, ra in rows if n == name and k == kind])
            if len(r) >= 2 and np.ptp(r[:, 0]) > 0.02:
                slope = np.polyfit(r[:, 0], r[:, 1], 1)[0]
                print(f"{name} {what}: true/size {r[:, 1].min():.3f}-{r[:, 1].max():.3f}, change per 10 cm of range "
                      f"{0.1 * slope:+.3f} (0 = the size scale is a constant)")


# ---------------------------------------------------------------------------------------------------- selftest

def cmd_selftest(args) -> int:
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'pass' if cond else 'FAIL'}] {name}: {msg}")

    W, H = 1280, 720
    K = np.array([[702.0, 0, 646.0], [0, 699.0, 352.0], [0, 0, 1]])
    D = np.array([-0.035, 0.012, -0.004, 0.0006])
    spec = load_spec()
    board = BoardDetector(spec)
    tag = TagDetector(spec)

    def in_image(uv, Xc):
        return (Xc[:, 2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

    print("intrinsics (30 board views, 0.3 px noise)")
    obj, img = [], []
    while len(obj) < 30:
        R = rot(rng.normal(size=3) * 0.8) @ rot_z(rng.uniform(-np.pi, np.pi))  # steep tilts pin the focal length
        c = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.15, 0.15), rng.uniform(0.3, 0.6)])
        P = board.corners3d - board.corners3d.mean(0)
        Xc = P @ R.T + c
        if np.mean(Xc[:, 2]) < 0.2:
            continue
        uv = project(K, D, np.eye(3), np.zeros(3), Xc)
        m = in_image(uv, Xc)
        if m.sum() < 12:
            continue
        obj.append(P[m])
        img.append(uv[m] + rng.normal(scale=0.3, size=(m.sum(), 2)))
    fit = fit_intrinsics(obj, img, (W, H))
    df = np.abs(np.diag(fit["K"])[:2] - np.diag(K)[:2]).max()
    dc = np.abs(fit["K"][:2, 2] - K[:2, 2]).max()
    # Focal within 0.5 %; a principal point a few px off is mostly absorbed by the extrinsic fit.
    check("intrinsics", df < 3.5 and dc < 5 and fit["rms_px"] < 0.5,
          f"focal off {df:.2f} px, centre off {dc:.2f} px, RMS {fit['rms_px']:.2f} px (2D, 0.42 expected)")

    print("overhead (planned poses, 0.5 px noise, joint offsets 0.6-1.0 deg; lift held at its true value, as after "
          "the gravity step)")
    model = ArmModel()
    links = Links(model.m)
    cp, ca = nominal_overhead()
    cp_true = cp + np.array([0.02, -0.015, 0.01])
    z_c = ca
    x_c = np.array([-ca[2], 0.0, ca[0]])
    x_c /= np.linalg.norm(x_c)
    R_wc = np.stack([x_c, np.cross(z_c, x_c), z_c], 1) @ rot(np.radians([1.0, -2.0, 1.5]))
    R_lt, p_lt = tag_on_face(links, np.radians(27), 0.003, -0.004)
    offs = np.radians([0.0, 3.0, -0.6, 0.5, 1.0])
    poses = overhead_poses(links, model, cp, ca, n=20)
    qs, cs = [], []
    for q in poses:
        Xw = _tag(links, q, R_lt, p_lt, tag.object_points)
        uv = project(K, D, R_wc, cp_true, Xw)
        if in_image(uv, (Xw - cp_true) @ R_wc).all():
            qs.append(q - offs)  # the encoders read true - offset
            cs.append(uv + rng.normal(scale=0.5, size=uv.shape))
    qs, cs = np.array(qs), np.array(cs)
    fit = fit_overhead(links, K, D, qs, cs, tag.object_points, fixed_offsets=offs * np.array([0, 1, 0, 0, 0]))
    dp = 1e3 * np.linalg.norm(fit["p"] - cp_true)
    da = angle_deg(fit["R"], R_wc)
    do = np.degrees(np.abs(fit["offsets"] - offs)).max()
    check("overhead", dp < 3 and da < 0.3 and do < 0.4,
          f"{len(qs)} poses, camera off {dp:.2f} mm / {da:.3f} deg, joint offsets off {do:.2f} deg max, "
          f"RMS {fit['rms_px']:.2f} px")

    print("lift offset from gravity (true lift offset 3.0 deg, unknown to the overhead fit; 8 ball flights seen by "
          "the overhead camera, 0.5 px noise)")
    from pingpong_sim.real.calib_solve import _fly

    cal = CameraCalib(name="overhead", mount="world", width=W, height=H, K=K.tolist(), D=D.tolist(),
                      R=R_wc.tolist(), p=cp_true.tolist(), stamp_offset=0.0, readout=0.0)
    rows = []
    for k in range(8):
        p0 = np.array([0.40 + rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.30])
        v0 = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(1.6, 2.2)])
        ts = np.arange(0, 0.4, 1 / 30) + rng.uniform(0, 1 / 30)
        Pw = _fly(p0, v0, np.array([0, 0, -9.81]), ts)
        Xc = (Pw - cp_true) @ R_wc
        uv = cal.project(Xc)
        th = np.arccos(Xc[:, 2] / np.linalg.norm(Xc, axis=1))
        s_r, s_t = cal.lens_scales(th)
        dpx = 2 * np.arcsin(0.02 / np.linalg.norm(Xc, axis=1)) * np.sqrt(s_r * s_t)
        for t, (u, v), dd in zip(ts + 1.5 * k, uv, dpx):
            if 0 <= u < W and 0 <= v < H:
                rows.append((0, t, u + rng.normal(scale=0.5), v + rng.normal(scale=0.5), dd + rng.normal(scale=0.5)))
    enc_t = np.arange(0, 13, 1 / RATE)
    enc_q = np.repeat((Q_READY - offs)[None], len(enc_t), axis=0)
    guess = ArmCalib(joint_offsets=(offs * np.array([1, 0, 1, 1, 1])).tolist())  # lift unknown: 0
    R_tb, p_tb = table_board(board.corners3d, np.array([0.45, 0.02, 0.004]))
    Xb = board.corners3d @ R_tb.T + p_tb
    uvb = project(K, D, R_wc, cp_true, Xb)
    mb = in_image(uvb, (Xb - cp_true) @ R_wc)
    lift_t, f, R_fb, p_fb = lift_from_table(guess.joint_offsets, cal, model.m, qs, cs, board.corners3d,
                                            np.flatnonzero(mb), uvb[mb] + rng.normal(scale=0.3, size=(mb.sum(), 2)))
    dz = 1e3 * abs((R_fb @ board.corners3d.mean(0) + p_fb)[2] - 0.004)
    check("lift from table", abs(np.degrees(lift_t - offs[1])) < 0.3 and dz < 3,
          f"lift offset {np.degrees(lift_t):+.2f} deg (true +3.00), board height off {dz:.2f} mm")
    lift, f = lift_from_gravity(guess, cal, np.array(rows), ["overhead"], enc_t, enc_q, model.m, qs, cs)
    dp = 1e3 * np.linalg.norm(f["p"] - cp_true)
    check("lift from gravity", abs(np.degrees(lift - offs[1])) < 0.3 and dp < 3,
          f"lift offset {np.degrees(lift):+.2f} deg (true +3.00), camera then off {dp:.2f} mm / "
          f"{angle_deg(f['R'], R_wc):.3f} deg")

    print("wrist mount (board flat on the table, module turned 90 deg from CAD like the real one, 0.4 px noise)")
    kin = links.kin
    R_lc = links.cad_cam_R @ rot_z(math.pi / 2) @ rot(np.radians([0.6, -0.5, 0.4]))
    p_lc = links.cad_cam_p + np.array([0.0012, -0.0008, 0.0010])
    R_wb, p_wb = table_board(board.corners3d, np.array([0.41, 0.012, 0.002]))
    R_wb = R_wb @ rot_z(np.radians(4.0))  # not quite square

    def usable(uv):
        return (uv[:, 0] >= 780) & (uv[:, 0] < W - 30) & (uv[:, 1] > 30) & (uv[:, 1] < H - 30)

    plan_board = table_board(board.corners3d)
    plan_cam = {"K": K, "D": D, "R": R_lc @ rot(np.radians([3.0, -2.0, 4.0])), "p": p_lc + 0.01,  # rough guess
                "board_R": plan_board[0], "board_p": plan_board[1], "board_pts": board.corners3d, "usable": usable}
    qs, views = [], []
    for q in wrist_poses(links, model, plan_cam, n=24):
        R_l, p_l = links(q)
        R_c, p_c = R_l[0] @ R_lc, R_l[0] @ p_lc + p_l[0]
        Xw = board.corners3d @ R_wb.T + p_wb
        uv = project(K, D, R_c, p_c, Xw)
        m = in_image(uv, (Xw - p_c) @ R_c) & usable(uv)
        if m.sum() >= 8:
            qs.append(q)
            views.append((np.flatnonzero(m), uv[m] + rng.normal(scale=0.4, size=(m.sum(), 2))))
    fit = fit_wrist(links, K, D, np.array(qs), views, board.corners3d, board0=plan_board)
    dp = 1e3 * np.linalg.norm(fit["p"] - p_lc)
    da = angle_deg(fit["R"], R_lc)
    check("wrist", dp < 1 and da < 0.3, f"{len(qs)} poses, mount off {dp:.2f} mm / {da:.3f} deg, RMS {fit['rms_px']:.2f} px")
    del kin

    print("timing (overhead, wave motion, 30 fps, 0.5 px noise)")
    off_true, ro_true = -0.032, 0.022
    q0 = poses[0]
    t_enc = np.arange(0, 10, 1 / RATE)
    motion = wave(q0, 10.0)
    q_enc = motion + rng.normal(scale=np.radians(0.03), size=motion.shape)  # ~encoder resolution (0.088 deg)
    stamps, rows_f, observed = [], [], []
    for tc in np.arange(0.5, 9.5, 1 / 30):
        q = np.array([np.interp(tc, t_enc, motion[:, j]) for j in range(5)])
        uv = project(K, D, R_wc, cp_true, _tag(links, q, R_lt, p_lt, tag.object_points))
        row = uv[:, 1].mean() / H
        stamps.append(tc - off_true - ro_true * row)
        rows_f.append(row)
        observed.append(uv + rng.normal(scale=0.5, size=uv.shape))

    def predict(q, extra, k):
        return project(K, D, R_wc, cp_true, _tag(links, q, R_lt, p_lt, tag.object_points))

    fit = fit_timing(t_enc, q_enc, np.array(stamps), np.array(rows_f), observed, predict)
    # What the filter uses is the capture time at the rows the ball shows up on: offset + readout x row.
    mid = float(np.mean(rows_f))
    eff, eff_true = fit["offset"] + fit["readout"] * mid, off_true + ro_true * mid
    check("timing", abs(eff - eff_true) < 0.001,
          f"capture time at the mean row off {1e3 * abs(eff - eff_true):.2f} ms (offset {1e3 * fit['offset']:.1f} vs "
          f"{1e3 * off_true:.0f}, readout {1e3 * fit['readout']:.1f} vs {1e3 * ro_true:.0f} ms), RMS {fit['rms_px']:.2f} px")

    print("gravity (4 drops seen by two cameras, frame tilted 0.8 deg)")
    g_true = rot(np.radians([0.8, 0.0, 0.0])) @ np.array([0, 0, -9.81])
    R_l0, p_l0 = links(Q_READY)
    R2 = R_l0[0] @ R_lc
    p2 = R_l0[0] @ p_lc + p_l0[0]
    from pingpong_sim.real.calib_solve import _fly

    arcs = []
    for k in range(4):
        p0 = np.array([0.4 + rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.6 + rng.uniform(-0.05, 0.05)])
        v0 = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(-0.3, 0.3)])
        ts = np.sort(np.concatenate([np.arange(0, 0.28, 1 / 30), np.arange(0.011, 0.28, 1 / 30)]))
        P = _fly(p0, v0, g_true, ts)
        Rs = np.array([R_wc if i % 2 == 0 else R2 for i in range(len(ts))])
        ps = np.array([cp_true if i % 2 == 0 else p2 for i in range(len(ts))])
        zz = measure_ball(P, Rs, ps)
        sig = np.tile([0.002, 0.002, 0.0015], (len(ts), 1))
        arcs.append({"t": ts + 10 * k, "R": Rs, "p": ps, "z": zz + rng.normal(size=zz.shape) * sig, "sigma": sig})
    fit = fit_gravity(arcs)
    check("gravity", abs(fit["tilt_deg"] - 0.8) < 0.2 and abs(fit["magnitude"] - 9.81) < 0.15,
          f"tilt {fit['tilt_deg']:.2f} deg (true 0.80), |g| {fit['magnitude']:.2f} m/s^2")
    print("all passed" if ok else "SOME FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("targets")
    p.add_argument("--tag", type=float, help="paddle tag black square [mm]")
    p.add_argument("--square", type=float, help="board square [mm]")
    p.add_argument("--marker", type=float, help="board marker [mm] (default: scaled with the squares)")
    p.add_argument("--ruler", type=float, help="printed 100 mm ruler [mm]")
    p = sub.add_parser("intrinsics")
    p.add_argument("camera", choices=("overhead", "wrist"))
    p.add_argument("--views", type=int, default=30)
    p.add_argument("--fresh", action="store_true", help="fit only this session's views (default: pool all saved ones)")
    p.add_argument("--refit", action="store_true", help="no camera: refit from the saved views")
    p = sub.add_parser("overhead")
    p.add_argument("--port", help="servo bus serial port (not needed with --refit)")
    p.add_argument("--poses", type=int, default=24)
    p.add_argument("--fresh", action="store_true", help="fit only the newest session (default: pool all saved ones)")
    p.add_argument("--refit", action="store_true", help="no arm: refit from the saved sessions")
    p = sub.add_parser("lift")
    p.add_argument("--paddle-height-cm", type=float, required=True,
                   help="ruler: table to the paddle top face's centre, read while `arm_pose.py ready` holds the arm")
    p = sub.add_parser("face")
    p.add_argument("--front-cm", type=float, required=True,
                   help="ruler: table to the blade's top surface at its front edge (the tip), during the ready hold")
    p.add_argument("--rear-cm", type=float, required=True, help="the same at the rear edge (handle side)")
    p.add_argument("--length-cm", type=float, required=True, help="distance along the blade between the two points")
    p.add_argument("--left-cm", type=float, help="table to the blade's top surface at its left edge (looking along "
                   "the arm, from the base), same hold; with --right-cm and --width-cm: corrects the wrist_roll zero")
    p.add_argument("--right-cm", type=float, help="the same at the right edge")
    p.add_argument("--width-cm", type=float, help="distance across the blade between those two points")
    p = sub.add_parser("roll")
    p.add_argument("--left-up-deg", type=float, required=True,
                   help="turn the held blade's left edge up by this much (negative: down): balls dropped on the "
                        "waiting blade leave toward the side that is low")
    p.add_argument("--why", required=True, help="what the adjustment is based on (kept in arm.json)")
    p = sub.add_parser("table")
    p.add_argument("--yes", action="store_true", help="don't wait for Enter")
    p.add_argument("--refit", action="store_true", help="no camera: solve from the saved table view")
    p = sub.add_parser("wrist")
    p.add_argument("--port", help="servo bus serial port (not needed with --refit)")
    p.add_argument("--poses", type=int, default=24)
    p.add_argument("--yes", action="store_true", help="don't wait for Enter after the board placement")
    p.add_argument("--refit", action="store_true", help="no arm: re-solve the newest recording with the current "
                   "joint offsets")
    p = sub.add_parser("timing")
    p.add_argument("camera", choices=("overhead", "wrist"))
    p.add_argument("--port", required=True)
    p.add_argument("--seconds", type=float, default=12.0)
    p = sub.add_parser("gravity")
    p.add_argument("--port", help="servo bus serial port (not needed with --refit)")
    p.add_argument("--cameras", nargs="+", default=None,
                   help="cameras to use (default: every camera with a calibrated pose; two cameras triangulate the "
                        "flights, one ranges them from the ball's apparent size, much less precisely)")
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--refit", action="store_true", help="no arm: refit from the newest saved recording")
    p.add_argument("--no-lift", action="store_true", help="only check gravity; don't solve the lift offset")
    p.add_argument("--fresh", action="store_true", help="analyse only the newest recording (default: pool all)")
    p = sub.add_parser("static")
    p.add_argument("--port", help="servo bus serial port (not needed with --refit)")
    p.add_argument("--heights", type=float, nargs="+", default=[0.28, 0.31, 0.34],
                   help="blade face heights [m] for the ball resting on it (level blade over the bounce point: "
                        "below 0.27 the shoulder lift nears its limit)")
    p.add_argument("--seconds", type=float, default=4.0, help="recorded per placement")
    p.add_argument("--free", action="store_true", help="then the ball on a support you measure (blade away)")
    p.add_argument("--refit", action="store_true", help="no arm: report on the newest saved recording")
    args = ap.parse_args()
    torch.set_num_threads(1)
    return {"selftest": cmd_selftest, "targets": cmd_targets, "intrinsics": cmd_intrinsics, "overhead": cmd_overhead,
            "lift": cmd_lift, "face": cmd_face, "roll": cmd_roll, "table": cmd_table, "wrist": cmd_wrist,
            "timing": cmd_timing, "gravity": cmd_gravity, "static": cmd_static}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
