"""Calibration solvers: data in, parameters out (no hardware). Used by scripts/calibrate.py and its `selftest`.

All pose fits minimize reprojection error in pixels through the fisheye lens model:

- `fit_intrinsics`: OpenCV fisheye calibration from ChArUco views.
- `fit_overhead`: overhead camera pose (world-from-camera), the paddle tag's pose on the paddle link and joint zero
  offsets (lift, elbow, wrist_flex, wrist_roll; pan is fixed: it trades off against the camera's yaw), from the
  tag's corners at ~20 arm poses and the encoder angles there.
- `fit_wrist`: wrist camera mount (paddle-link-from-camera) and the board's pose, from board corners seen by the
  wrist camera at ~20 arm poses.
- `fit_timing`: timestamp offset and rolling-shutter readout (capture = stamp + offset + readout * row / height),
  from target corners seen during a sinusoidal arm motion against the encoder log; optionally refits the static
  target's pose.
- `fit_gravity`: the gravity vector in the robot frame from ball flights (drag included): its tilt from -z checks
  the whole chain (extrinsics, encoders, timing).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares

from pingpong_sim.ball import BALL_RADIUS, DRAG_K, G
from pingpong_sim.kinematics import ArmKinematics

FIT_JOINTS = (1, 2, 3, 4)  # joint offsets fitted by the overhead calibration (not pan)


def rot(rvec: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0]


def rvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(rvec(Ra.T @ Rb))))


def project(K: np.ndarray, D: np.ndarray, R_wc: np.ndarray, p_wc: np.ndarray, X_w: np.ndarray) -> np.ndarray:
    """World points (N, 3) seen by a camera at (R_wc, p_wc) -> pixels (N, 2)."""
    Xc = (np.asarray(X_w, float).reshape(-1, 3) - p_wc) @ R_wc
    Xc[:, 2] = np.maximum(Xc[:, 2], 1e-6)
    uv, _ = cv2.fisheye.projectPoints(Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.asarray(D, float).reshape(4, 1))
    return uv.reshape(-1, 2)


class Links:
    """Paddle-link poses in float64 (finite differences in the fits need them)."""

    def __init__(self, mj_model, prefix: str = ""):
        self.kin = ArmKinematics(mj_model, prefix=prefix, dtype=torch.float64)
        s = mj_model.site(prefix + "blade_face_base")
        R = np.zeros(9)
        import mujoco

        mujoco.mju_quat2Mat(R, s.quat)
        self.back_R, self.back_p = R.reshape(3, 3), np.array(s.pos, float)  # back face site in the link frame
        self.cad_cam_R, self.cad_cam_p = self.kin.cam_R.numpy().copy(), self.kin.cam_p.numpy().copy()

    def __call__(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R, p = self.kin.link(torch.as_tensor(np.atleast_2d(q), dtype=torch.float64))
        return R.numpy(), p.numpy()


def pnp(K, D, obj: np.ndarray, uv: np.ndarray, square: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    """Camera-from-object pose from 2D-3D matches through the fisheye model."""
    xy = cv2.fisheye.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), K, np.asarray(D, float).reshape(4, 1))
    flags = cv2.SOLVEPNP_IPPE_SQUARE if square else cv2.SOLVEPNP_ITERATIVE
    ok, r, t = cv2.solvePnP(np.asarray(obj, float), xy.reshape(-1, 2), np.eye(3), None, flags=flags)
    return (rot(r), t.ravel()) if ok else None


# ---------------------------------------------------------------------------------------------------- intrinsics

def fit_intrinsics(obj: list[np.ndarray], img: list[np.ndarray], size: tuple[int, int],
                   f0: float = 719.0) -> dict:
    """Views of (N_i, 3) board points and their (N_i, 2) pixels -> K, D, RMS reprojection error (px)."""
    K = np.array([[f0, 0, size[0] / 2], [0, f0, size[1] / 2], [0, 0, 1]], float)
    D = np.zeros((4, 1))
    # OpenCV 5's fisheye.calibrate takes each view as a (1, N, 3) / (1, N, 2) array.
    o = [np.asarray(a, np.float64).reshape(1, -1, 3) for a in obj]
    i = [np.asarray(a, np.float64).reshape(1, -1, 2) for a in img]
    flags = cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW | cv2.CALIB_USE_INTRINSIC_GUESS  # OpenCV 5 names
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10)
    rms, K, D, rv, tv = cv2.fisheye.calibrate(o, i, size, K, D, flags=flags, criteria=crit)
    per_view = []
    for oo, ii, r, t in zip(o, i, rv, tv):
        uv, _ = cv2.fisheye.projectPoints(oo.reshape(-1, 1, 3), r, t, K, D)
        per_view.append(float(np.sqrt(np.mean(np.sum((uv.reshape(-1, 2) - ii.reshape(-1, 2)) ** 2, -1)))))
    return {"K": K, "D": D.ravel(), "rms_px": float(rms), "per_view_px": per_view}


# ---------------------------------------------------------------------------------------------------- overhead

def _tag_world(links: Links, q: np.ndarray, R_lt, p_lt, tag_pts) -> np.ndarray:
    R_l, p_l = links(q)
    return np.einsum("nij,kj->nki", R_l, tag_pts @ R_lt.T + p_lt) + p_l[:, None]  # (N, 4, 3)


def tag_on_face(links: Links, theta: float, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """The tag glued flat on the back face: rotated by theta about the face normal and shifted (dx, dy) in the face
    plane from the face centre -> link-from-tag rotation and position."""
    return links.back_R @ rot_z(theta), links.back_p + links.back_R @ np.array([dx, dy, 0.0])


def fit_overhead(links: Links, K, D, qs: np.ndarray, corners: np.ndarray, tag_pts: np.ndarray,
                 fit_offsets: bool = True, offset_prior_deg: float = 2.0) -> dict:
    """Tag corners (N, 4, 2) at encoder angles qs (N, 5) -> camera pose, tag placement on the back face, joint
    offsets. The tag lies flat on the blade (its normal is the face normal): with a free tag orientation the
    wrist_roll offset would be unobservable (a roll offset = the tag turned about the paddle axis), yet it is the
    paddle's sideways tilt."""
    n = len(qs)

    def unpack(x):
        d = np.zeros(5)
        if fit_offsets:
            d[list(FIT_JOINTS)] = x[9:13]
        R_lt, p_lt = tag_on_face(links, x[6], x[7], x[8])
        return rot(x[0:3]), x[3:6], R_lt, p_lt, d

    def res(x):
        R_wc, p_wc, R_lt, p_lt, d = unpack(x)
        X = _tag_world(links, qs + d, R_lt, p_lt, tag_pts)
        r = (project(K, D, R_wc, p_wc, X.reshape(-1, 3)) - corners.reshape(-1, 2)).ravel()
        if fit_offsets:
            r = np.concatenate([r, x[9:13] / math.radians(offset_prior_deg)])
        return r

    # Initial guesses: the tag centred on the back face, its rotation in the face unknown (8 tries); the camera
    # from each try's per-view PnP estimates (the view nearest their median position).
    views = [pnp(K, D, tag_pts, c, square=True) for c in corners]
    R_l, p_l = links(qs)
    best = None
    for k in range(8):
        theta = k * math.pi / 4
        R_lt, p_lt = tag_on_face(links, theta, 0.0, 0.0)
        cams = []
        for i, v in enumerate(views):
            if v is None:
                continue
            R_ct, t_ct = v
            R_wt, p_wt = R_l[i] @ R_lt, R_l[i] @ p_lt + p_l[i]
            R_wc = R_wt @ R_ct.T
            cams.append((R_wc, p_wt - R_wc @ t_ct))
        if not cams:
            raise ValueError("the tag pose could not be solved in any view")
        p0 = np.median([c[1] for c in cams], axis=0)
        R0 = cams[int(np.argmin([np.linalg.norm(c[1] - p0) for c in cams]))][0]
        x0 = np.concatenate([rvec(R0), p0, [theta, 0.0, 0.0], np.zeros(4 if fit_offsets else 0)])
        sol = least_squares(res, x0, method="lm", max_nfev=400)
        if best is None or sol.cost < best.cost:
            best = sol
    sol = least_squares(res, best.x, method="lm", max_nfev=4000)
    R_wc, p_wc, R_lt, p_lt, d = unpack(sol.x)
    r = res(sol.x)[: n * 8].reshape(n, 4, 2)
    per_view = np.sqrt((r ** 2).sum(-1).mean(-1))
    return {"R": R_wc, "p": p_wc, "tag_R": R_lt, "tag_p": p_lt, "offsets": d,
            "rms_px": float(np.sqrt((r ** 2).sum(-1).mean())), "per_view_px": per_view}


# ---------------------------------------------------------------------------------------------------- wrist

def fit_wrist(links: Links, K, D, qs: np.ndarray, views: list[tuple[np.ndarray, np.ndarray]],
              board_pts: np.ndarray, mount0: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Board corners seen by the wrist camera: views [(ids, uv)] at encoder angles qs (true angles) -> camera mount
    (paddle-link-from-camera) and board pose (world-from-board)."""
    R_l, p_l = links(qs)
    R_lc0, p_lc0 = mount0 if mount0 is not None else (links.cad_cam_R, links.cad_cam_p)
    # Board pose from the first view with a PnP solution, through the initial mount.
    for i, (ids, uv) in enumerate(views):
        v = pnp(K, D, board_pts[ids], uv)
        if v is not None:
            R_cb, t_cb = v
            R_wc = R_l[i] @ R_lc0
            p_wc = R_l[i] @ p_lc0 + p_l[i]
            R_wb0, p_wb0 = R_wc @ R_cb, R_wc @ t_cb + p_wc
            break
    else:
        raise ValueError("the board pose could not be solved in any view")
    obs = np.concatenate([uv for _, uv in views])

    def res(x):
        R_lc, p_lc, R_wb, p_wb = rot(x[0:3]), x[3:6], rot(x[6:9]), x[9:12]
        out = []
        for i, (ids, _) in enumerate(views):
            R_wc = R_l[i] @ R_lc
            p_wc = R_l[i] @ p_lc + p_l[i]
            out.append(project(K, D, R_wc, p_wc, board_pts[ids] @ R_wb.T + p_wb))
        return (np.concatenate(out) - obs).ravel()

    x0 = np.concatenate([rvec(R_lc0), p_lc0, rvec(R_wb0), p_wb0])
    sol = least_squares(res, x0, method="lm", max_nfev=4000)
    r = res(sol.x).reshape(-1, 2)
    return {"R": rot(sol.x[0:3]), "p": sol.x[3:6], "board_R": rot(sol.x[6:9]), "board_p": sol.x[9:12],
            "rms_px": float(np.sqrt((r ** 2).sum(-1).mean()))}


# ---------------------------------------------------------------------------------------------------- timing

def fit_timing(t_enc: np.ndarray, q_enc: np.ndarray, stamps: np.ndarray, rows: np.ndarray, observed: list[np.ndarray],
               predict, extra0: np.ndarray | None = None, search: tuple[float, float] = (-0.15, 0.06)) -> dict:
    """Detections (stamp, row fraction v/height, observed pixels (m, 2)) of a target whose pixels `predict(q,
    extra, k)` gives for encoder angles q (5,) at the capture time (k: detection index, `extra`: static target
    parameters refitted alongside) -> offset and readout (s)."""
    extra0 = np.zeros(0) if extra0 is None else np.asarray(extra0, float)

    def q_at(t):
        return np.stack([np.interp(t, t_enc, q_enc[:, j]) for j in range(q_enc.shape[1])], axis=-1)

    def res(x):
        a, b, extra = x[0], x[1], x[2:]
        tc = stamps + a + b * rows
        qs = q_at(tc)
        return np.concatenate([(predict(qs[k], extra, k) - observed[k]).ravel() for k in range(len(stamps))])

    grid = np.arange(search[0], search[1], 0.002)
    costs = [np.sum(res(np.concatenate([[a, 0.02], extra0])) ** 2) for a in grid]
    a0 = grid[int(np.argmin(costs))]
    lo = np.concatenate([[search[0], 0.0], np.full(len(extra0), -np.inf)])
    hi = np.concatenate([[search[1], 0.05], np.full(len(extra0), np.inf)])
    sol = least_squares(res, np.concatenate([[a0, 0.02], extra0]), bounds=(lo, hi), x_scale="jac")
    r = res(sol.x).reshape(-1, 2)
    return {"offset": float(sol.x[0]), "readout": float(sol.x[1]), "extra": sol.x[2:],
            "rms_px": float(np.sqrt((r ** 2).sum(-1).mean())), "grid": (grid, np.array(costs))}


# ---------------------------------------------------------------------------------------------------- gravity

def _fly(p0: np.ndarray, v0: np.ndarray, g: np.ndarray, ts: np.ndarray, h: float = 0.002) -> np.ndarray:
    """Positions at times ts (>= 0, sorted) of a ball from (p0, v0) under gravity g and quadratic drag (RK4)."""
    out, p, v, t = [], p0.astype(float), v0.astype(float), 0.0

    def acc(v):
        return g - DRAG_K * np.linalg.norm(v) * v

    for tk in ts:
        while t < tk - 1e-12:
            s = min(h, tk - t)
            k1p, k1v = v, acc(v)
            k2p, k2v = v + 0.5 * s * k1v, acc(v + 0.5 * s * k1v)
            k3p, k3v = v + 0.5 * s * k2v, acc(v + 0.5 * s * k2v)
            k4p, k4v = v + s * k3v, acc(v + s * k3v)
            p = p + s / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            v = v + s / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            t += s
        out.append(p.copy())
    return np.array(out)


def measure_ball(p: np.ndarray, R: np.ndarray, cp: np.ndarray) -> np.ndarray:
    """The filter's measurement model [x/z, y/z, angular diameter] (numpy, one point per camera pose)."""
    X = np.einsum("nji,nj->ni", R, p - cp)
    rho = np.linalg.norm(X, axis=1)
    return np.stack([X[:, 0] / X[:, 2], X[:, 1] / X[:, 2], 2 * np.arcsin(np.minimum(BALL_RADIUS / rho, 1.0))], 1)


def fit_gravity(arcs: list[dict]) -> dict:
    """Ball flights, each {t (n,), R (n, 3, 3), p (n, 3) camera poses, z (n, 3), sigma (n, 3)} -> gravity vector
    (3,), its magnitude and tilt from -z (deg), per-arc normalized RMS."""
    inits = []
    for a in arcs:  # monocular 3D points -> straight-line start
        u = np.concatenate([a["z"][:, :2], np.ones((len(a["t"]), 1))], 1)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        rng = BALL_RADIUS / np.sin(a["z"][:, 2] / 2)
        X = a["p"] + np.einsum("nij,nj->ni", a["R"], rng[:, None] * u)
        t = a["t"] - a["t"][0]
        A = np.stack([np.ones_like(t), t, 0.5 * t * t], 1)
        coef = np.linalg.lstsq(A, X, rcond=None)[0]  # p0, v0, a (a is fitted jointly below)
        inits.append(np.concatenate([coef[0], coef[1]]))

    def res(x):
        g = x[:3]
        out = []
        for k, a in enumerate(arcs):
            p0, v0 = x[3 + 6 * k: 6 + 6 * k], x[6 + 6 * k: 9 + 6 * k]
            P = _fly(p0, v0, g, a["t"] - a["t"][0])
            out.append(((measure_ball(P, a["R"], a["p"]) - a["z"]) / a["sigma"]).ravel())
        return np.concatenate(out)

    x0 = np.concatenate([[0.0, 0.0, -G]] + inits)
    sol = least_squares(res, x0, x_scale="jac")
    g = sol.x[:3]
    r = res(sol.x)
    per_arc, i = [], 0
    for a in arcs:
        m = 3 * len(a["t"])
        per_arc.append(float(np.sqrt(np.mean(r[i:i + m] ** 2))))
        i += m
    tilt = float(np.degrees(np.arccos(np.clip(-g[2] / np.linalg.norm(g), -1, 1))))
    return {"g": g, "magnitude": float(np.linalg.norm(g)), "tilt_deg": tilt, "per_arc_rms": per_arc}


# ---------------------------------------------------------------------------------------------------- poses

def overhead_poses(links: Links, model, cam_p: np.ndarray, cam_axis: np.ndarray, n: int = 20, seed: int = 0,
                   max_angle_deg: float = 40.0, tries: int = 20000) -> np.ndarray:
    """Arm poses (n, 5) showing the paddle's back face (the tag) to a camera near `cam_p` looking along `cam_axis`:
    face normal within max_angle of the camera direction, tag within max_angle of the optical axis, 0.2-0.6 m away,
    payload clear of the table and blade clear of the arm (`model`: arm.ArmModel). Spread out, ordered for short
    moves from the ready pose."""
    rng = np.random.default_rng(seed)
    lo, hi = model.range[:, 0] + 0.1, model.range[:, 1] - 0.1
    box_lo = np.array([-0.35, -1.3, -0.2, -1.3, -2.75])
    box_hi = np.array([0.35, 0.1, 1.6, 1.3, 2.75])
    cos_max = math.cos(math.radians(max_angle_deg))
    axis = cam_axis / np.linalg.norm(cam_axis)
    good, feats = [], []
    q = rng.uniform(np.maximum(box_lo, lo), np.minimum(box_hi, hi), size=(tries, 5))
    q[:, 4] = np.sign(q[:, 4]) * rng.uniform(1.9, 2.7, tries)  # back face up: roll well past +-90 deg
    R_l, p_l = links(q)
    c = np.einsum("nij,j->ni", R_l, links.back_p) + p_l
    nrm = np.einsum("nij,j->ni", R_l, links.back_R[:, 2])
    to_cam = cam_p - c
    dist = np.linalg.norm(to_cam, axis=1)
    ok = (np.einsum("ni,ni->n", nrm, to_cam) / dist > cos_max) & (-(to_cam @ axis) / dist > cos_max)
    ok &= (dist > 0.2) & (dist < 0.6) & (c[:, 2] > 0.15)
    for i in np.flatnonzero(ok):
        model.forward(q[i])
        if model.clearance() < 0.06 or model.blade_hits():
            continue
        good.append(q[i])
        feats.append(np.concatenate([c[i] / 0.1, nrm[i]]))
        if len(good) >= 50 * n:
            break
    if len(good) < n:
        raise RuntimeError(f"only {len(good)} poses show the tag to the camera; is the camera guess right?")
    good, feats = np.array(good), np.array(feats)
    chosen = [int(np.argmax(feats[:, 2]))]  # farthest-point sampling over position and normal
    d = np.linalg.norm(feats - feats[chosen[0]], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(feats - feats[k], axis=1))
    return order_poses(good[chosen])


def wrist_poses(ready: np.ndarray, model, n: int = 24, seed: int = 0) -> np.ndarray:
    """Arm poses around the ready pose for the wrist camera to see a board in front of the arm from varied
    viewpoints (pan +-0.25, lift/elbow +-0.12, wrist_flex +-0.25, wrist_roll +-0.4 rad), checked like above."""
    rng = np.random.default_rng(seed)
    span = np.array([0.25, 0.12, 0.12, 0.25, 0.4])
    out = []
    while len(out) < n:
        q = ready + rng.uniform(-span, span)
        model.forward(q)
        if model.clearance() > 0.06 and not model.blade_hits():
            out.append(q)
    return order_poses(np.array(out), ready)


def order_poses(qs: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
    """Nearest-neighbour tour (max joint distance) from `start` (default: the first pose)."""
    left = list(range(len(qs)))
    cur = start if start is not None else qs[left.pop(0)]
    order = [] if start is not None else [0]
    while left:
        k = min(left, key=lambda i: np.abs(qs[i] - cur).max())
        left.remove(k)
        order.append(k)
        cur = qs[k]
    return qs[order]
