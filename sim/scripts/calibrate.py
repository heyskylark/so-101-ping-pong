"""Camera and arm calibration for the real rig (docs/hardware_setup.md, "Calibration").

Order (from sim/; the arm steps need the joint map and the camera devices in rig.json):

  uv run scripts/calibrate.py selftest                      # the solvers on synthetic data (no hardware)
  uv run scripts/calibrate.py targets --tag 70.1 --square 30.0 --ruler 100.0   # caliper readings of the prints
  uv run scripts/calibrate.py intrinsics overhead           # wave the board in front of the camera
  uv run scripts/calibrate.py intrinsics wrist
  uv run scripts/calibrate.py overhead --port /dev/ttyACM0  # the arm shows the paddle tag to the overhead camera
  uv run scripts/calibrate.py wrist --port /dev/ttyACM0     # board standing in front; the arm moves the wrist camera
  uv run scripts/calibrate.py timing overhead --port /dev/ttyACM0
  uv run scripts/calibrate.py timing wrist --port /dev/ttyACM0   # board where it stood for `wrist`
  uv run scripts/calibrate.py gravity --port /dev/ttyACM0   # drop the ball a few times: checks the whole chain

Results: measurements/calibration/camera_<name>.json, arm.json; raw data of each step in
measurements/calibration/raw/<time>_<step>.npz. Targets (TODO.md): overhead pose <= 3 mm / 0.3 deg, wrist mount
<= 1 mm / 0.5 deg, timestamps <= +-2 ms; the gravity check should show a tilt < 0.5 deg and |g| within 2 %.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from pingpong_sim.camera_sim import OVERHEAD
from pingpong_sim.controller import ArmHistory
from pingpong_sim.kinematics import ArmKinematics
from pingpong_sim.real.arm import RATE, ArmModel, ArmSession, PlanBuilder, open_bus
from pingpong_sim.real.calib_solve import (Links, angle_deg, fit_gravity, fit_intrinsics, fit_overhead, fit_timing,
                                           fit_wrist, leave_one_out, measure_ball, overhead_poses, project, rot, rot_z,
                                           rvec, tag_on_face, wrist_board_placement, wrist_poses)
from pingpong_sim.real.camera_model import (CALIB_DIR, ArmCalib, CameraCalib, apply_wrist_mount, nominal,
                                            to_measurements)
from pingpong_sim.real.cameras import MAX_BALL_PX, MIN_BALL_PX, BallCameras, RigConfig, UvcCamera
from pingpong_sim.real.targets import SPEC, BoardDetector, TagDetector, load_spec
from pingpong_sim.robot import READY_JOINT_POS
from pingpong_sim.servo_bus import JOINTS

RAW = CALIB_DIR / "raw"
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


def check_tour(session: ArmSession, poses: np.ndarray) -> None:
    b = PlanBuilder(session.read().q)
    for q in poses:
        b.move(q, name="tour")
    b.move(session.start, name="to_start")
    problems = session.check(b.build())
    if problems:
        raise SystemExit("pose tour check failed:\n  " + "\n  ".join(problems))


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
    return sorted(p for p in RAW.glob("*_overhead.npz") if "_intrinsics_" not in p.name)


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
                check_tour(s, poses)
                qs, corners = [], []
                for i, q in enumerate(poses):
                    s.move_to(q)
                    hold(s, 1.5)  # the arm hunts ~+-1 deg for over a second after a move
                    cam.drain()
                    rows = hold(s, 1.0)
                    dets = [c for _, c in cam.drain()]
                    if len(dets) < 5:
                        print(f"  pose {i + 1}/{len(poses)}: tag seen in {len(dets)} frames, skipped")
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


def cmd_wrist(args) -> int:
    calib = CameraCalib.load("wrist")
    if "intrinsics" not in calib.steps:
        raise SystemExit("calibrate the wrist camera's lens first: calibrate.py intrinsics wrist")
    board = BoardDetector(min_corners=8)
    arm = ArmCalib.load()
    model = ArmModel()
    links = Links(model.m)
    q_ready = Q_READY  # true angles; the session applies the joint offsets
    R_lc, p_lc = (np.array(calib.R), np.array(calib.p)) if calib.R is not None else (links.cad_cam_R, links.cad_cam_p)
    R_wb, p_wb, centre = wrist_board_placement(links, q_ready, R_lc, p_lc, board.corners3d, dist=0.40, above=0.16)
    usable = wrist_usable(calib)
    camera = {"K": calib.Km, "D": calib.Dm, "R": R_lc, "p": p_lc, "links": links, "board_R": R_wb, "board_p": p_wb,
              "board_pts": board.corners3d, "usable": usable}
    poses = wrist_poses(q_ready, model, n=args.poses, camera=camera)
    print(f"Stand the board upright, long side horizontal, printed side facing the wrist camera (square to the line "
          f"from the camera to the board). Its centre: {100 * centre[0]:.0f} cm in front of the arm base's axis, "
          f"{100 * -centre[1]:.0f} cm to the robot's right of its centre line, {100 * centre[2]:.0f} cm above the "
          "table. Don't move it until `timing wrist` is done.")
    if not args.yes:
        input("Enter when the board is there ")
    cam = start_camera("wrist", board)
    try:
        with ArmSession(open_bus(args.port), joint_offsets=arm.joint_offsets) as s:
            check_tour(s, poses)
            qs, views = [], []
            for i, q in enumerate(poses):
                s.move_to(q)
                hold(s, 1.5)  # the arm hunts ~+-1 deg for over a second after a move
                cam.drain()
                rows = hold(s, 1.0)
                dets = [d for _, d in cam.drain()]
                if len(dets) < 5:
                    print(f"  pose {i + 1}/{len(poses)}: board seen in {len(dets)} frames, skipped")
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
    raw = save_raw("wrist", qs=np.array(qs), ids=[v[0] for v in views], uv=[v[1] for v in views])
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
    fit = fit_wrist(links, calib.Km, calib.Dm, qs_ok, views_ok, board.corners3d, mount0=(R_lc, p_lc))
    d_p = 1e3 * np.linalg.norm(fit["p"] - links.cad_cam_p)
    print(f"RMS {fit['rms_px']:.2f} px over {len(views_ok)} poses; mount {d_p:.1f} mm / "
          f"{angle_deg(fit['R'], links.cad_cam_R):.2f} deg from CAD; board centre "
          f"{np.round(fit['board_R'] @ board.corners3d.mean(0) + fit['board_p'], 3).tolist()} m")
    calib.R, calib.p = fit["R"].tolist(), fit["p"].tolist()
    calib.steps["mount"] = {"date": stamp(), "poses": len(views_ok), "rms_px": fit["rms_px"], "raw": raw.name,
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
    links = Links(ArmModel().m)
    if name == "overhead":
        if calib.R is None or arm.tag_R is None:
            raise SystemExit("run `calibrate.py overhead` first")
        det = TagDetector()
        R_wc, p_wc = np.array(calib.R), np.array(calib.p)
        R_lt, p_lt = np.array(arm.tag_R), np.array(arm.tag_p)
        poses = np.array(arm.steps["overhead"]["poses"])
        # The pose whose tag is nearest the image centre.
        centre = np.array([calib.width / 2, calib.height / 2])
        q0 = poses[int(np.argmin([np.linalg.norm(project(calib.Km, calib.Dm, R_wc, p_wc, _tag(links, q + np.array(
            arm.joint_offsets), R_lt, p_lt, det.object_points)).mean(0) - centre) for q in poses]))]
        q0 = q0 + np.array(arm.joint_offsets)

        def predict(q, extra, k):
            return project(calib.Km, calib.Dm, R_wc, p_wc, _tag(links, q, R_lt, p_lt, det.object_points))
    else:
        if calib.R is None or "mount" not in calib.steps:
            raise SystemExit("run `calibrate.py wrist` first (and leave the board where it was)")
        det = BoardDetector(min_corners=8)
        R_lc, p_lc = np.array(calib.R), np.array(calib.p)
        m = calib.steps["mount"]
        q0 = np.array(m["pose_q"])[0]
    cam = start_camera(name, det)
    rows: list[tuple[float, np.ndarray]] = []
    try:
        with ArmSession(open_bus(args.port), joint_offsets=arm.joint_offsets) as s:
            s.move_to(q0)
            hold(s, 0.5)
            motion = wave(s.read().q, args.seconds)
            b = PlanBuilder(motion[0])
            b.add(motion, name="wave")
            problems = s.check(b.build())
            if problems:
                raise SystemExit("wave check failed:\n  " + "\n  ".join(problems))
            cam.drain()
            s.follow(motion, on_tick=lambda k, g, r: rows.append((r.t, r.q)))
            dets = cam.drain()
            s.move_to(s.start)
    finally:
        cam.stop()
    t_enc, q_enc = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    dets = [(st, d) for st, d in dets if t_enc[0] + 0.2 < st < t_enc[-1] - 0.2]
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
    raw = save_raw(f"timing_{name}", t_enc=t_enc, q_enc=q_enc, stamps=stamps, observed=observed)
    print(f"{name}: capture = buffer stamp {1e3 * fit['offset']:+.1f} ms + {1e3 * fit['readout']:.1f} ms x row; RMS "
          f"{fit['rms_px']:.2f} px over {len(stamps)} frames (stamp latency median "
          f"{1e3 * np.median(t_enc[-1] - stamps):.0f} ms behind the last encoder read)")
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


def gravity_measurements(dets: np.ndarray, cams: list[str], enc_t: np.ndarray, enc_q: np.ndarray,
                         calibs: dict[str, CameraCalib], model) -> dict[str, np.ndarray]:
    """Raw detections (N, 5: camera index, stamp, u, v, d; times on the encoder clock) of the cameras in `calibs`
    -> measurement arrays t, R, p, z, sigma, cam (index into list(calibs)), sorted by time; wrist poses from the
    encoder log through the arm model."""
    from pingpong_sim.real.camera_model import PixelDetection

    kin = ArmKinematics(model, prefix="")
    if "wrist" in calibs:
        apply_wrist_mount(kin, calibs["wrist"])
    hist = ArmHistory(kin, 1, 1 / RATE, "cpu")
    for t, q in zip(enc_t, enc_q):
        hist.push(float(t), torch.tensor(q, dtype=torch.float32)[None])
    parts = []
    for ci, name in enumerate(calibs):
        pix = [PixelDetection(name, *map(float, r[1:])) for r in dets if cams[int(r[0])] == name]
        if not pix:
            continue
        m = to_measurements(pix, {name: calibs[name]}, hist.hist)
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


def load_gravity_recordings() -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """All gravity recordings pooled (each shifted by 1000 s): detections (N, 5, camera indices into the returned
    names), camera names, encoder times and raw angles."""
    names: list[str] = []
    D, T, Q = [], [], []
    for k, path in enumerate(sorted(RAW.glob("*_gravity.npz"))):
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
        args.cameras = [n for n in ("overhead", "wrist")
                        if CameraCalib.path(n).exists() and CameraCalib.load(n).R is not None]
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
            print(f"Toss the ball gently across the space above the paddle (arcs peaking 10-20 cm above it, not "
                  f"touching the paddle), let it land beyond, repeat; {args.seconds:.0f} s.")
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
    dets, names, enc_t, enc_q_raw = load_gravity_recordings()
    use = [n for n in names if n in args.cameras]
    calibs = {n: CameraCalib.load(n) for n in use}
    m = gravity_measurements(dets, names, enc_t, enc_q_raw + np.array(arm.joint_offsets), calibs, model)
    arcs = gravity_arcs(m)
    if len(arcs) < 3:
        raise SystemExit(f"only {len(arcs)} usable ball flights in {len(m['t'])} detections")
    if not args.no_lift and "overhead" in calibs:
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
    lift, f = lift_from_gravity(guess, cal, np.array(rows), ["overhead"], enc_t, enc_q, model.m, qs, cs)
    dp = 1e3 * np.linalg.norm(f["p"] - cp_true)
    check("lift from gravity", abs(np.degrees(lift - offs[1])) < 0.3 and dp < 3,
          f"lift offset {np.degrees(lift):+.2f} deg (true +3.00), camera then off {dp:.2f} mm / "
          f"{angle_deg(f['R'], R_wc):.3f} deg")

    print("wrist mount (board in front, 0.4 px noise)")
    kin = links.kin
    R_lc = links.cad_cam_R @ rot(np.radians([0.6, -0.5, 0.4]))
    p_lc = links.cad_cam_p + np.array([0.0012, -0.0008, 0.0010])
    R_l0, p_l0 = links(Q_READY)
    cam_ax = (R_l0[0] @ R_lc)[:, 2]
    bc = (R_l0[0] @ p_lc + p_l0[0]) + 0.45 * cam_ax
    bz = -cam_ax
    bx = np.cross([0, 0, 1.0], bz)
    bx /= np.linalg.norm(bx)
    R_wb = np.stack([bx, np.cross(bz, bx), bz], 1)
    p_wb = bc - R_wb @ board.corners3d.mean(0)
    qs, views = [], []
    for q in wrist_poses(Q_READY, model, n=24):
        R_l, p_l = links(q)
        R_c, p_c = R_l[0] @ R_lc, R_l[0] @ p_lc + p_l[0]
        Xw = board.corners3d @ R_wb.T + p_wb
        uv = project(K, D, R_c, p_c, Xw)
        m = in_image(uv, (Xw - p_c) @ R_c)
        if m.sum() >= 8:
            qs.append(q)
            views.append((np.flatnonzero(m), uv[m] + rng.normal(scale=0.4, size=(m.sum(), 2))))
    fit = fit_wrist(links, K, D, np.array(qs), views, board.corners3d)
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
    p = sub.add_parser("wrist")
    p.add_argument("--port", required=True)
    p.add_argument("--poses", type=int, default=24)
    p.add_argument("--yes", action="store_true", help="don't wait for Enter after the board placement")
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
    args = ap.parse_args()
    torch.set_num_threads(1)
    return {"selftest": cmd_selftest, "targets": cmd_targets, "intrinsics": cmd_intrinsics, "overhead": cmd_overhead,
            "wrist": cmd_wrist, "timing": cmd_timing, "gravity": cmd_gravity}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
