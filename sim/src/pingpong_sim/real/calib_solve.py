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

# Joint offsets the overhead calibration fits. Not pan or lift: everything the camera sees (the tag) is beyond both,
# so a pan offset is the same picture as the camera turned about the vertical, and a lift offset the same as the
# camera turned about the shoulder's lift axis (26 real poses: lift free -> camera +-23 mm / 2.2 deg). The lift
# offset is found from gravity instead (`calibrate.py gravity`: the tilt of ball flights about that axis).
FIT_JOINTS = (2, 3, 4)


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

def lens_monotonic(D, theta_max: float = 1.25) -> bool:
    """The fisheye curve theta_d(theta) keeps increasing out to theta_max (rad): it doesn't fold back."""
    k = np.asarray(D, float).ravel()
    t = np.linspace(0, theta_max, 400)
    td = t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6 + k[3] * t**8)
    return bool(np.all(np.diff(td) > 0))


def fit_intrinsics(obj: list[np.ndarray], img: list[np.ndarray], size: tuple[int, int],
                   f0: float = 719.0, n_dist: int = 2) -> dict:
    """Views of (N_i, 3) board points and their (N_i, 2) pixels -> K, D, RMS reprojection error (px).
    `n_dist` distortion terms (k1..k4): 2 by default. With 4, the outer terms are free where the board didn't
    reach and folded the lens curve back on itself (first real fit); on 70 pooled views 2, 3 and 4 terms agreed
    to 0.01 deg along the ball's path with the same held-out error."""
    K = np.array([[f0, 0, size[0] / 2], [0, f0, size[1] / 2], [0, 0, 1]], float)
    D = np.zeros((4, 1))
    # OpenCV 5's fisheye.calibrate takes each view as a (1, N, 3) / (1, N, 2) array.
    o = [np.asarray(a, np.float64).reshape(1, -1, 3) for a in obj]
    i = [np.asarray(a, np.float64).reshape(1, -1, 2) for a in img]
    flags = cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW | cv2.CALIB_USE_INTRINSIC_GUESS  # OpenCV 5 names
    for k, f in zip((2, 3, 4), (cv2.CALIB_FIX_K2, cv2.CALIB_FIX_K3, cv2.CALIB_FIX_K4)):
        if n_dist < k:
            flags |= f
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 300, 1e-12)
    rms, K, D, rv, tv = cv2.fisheye.calibrate(o, i, size, K, D, flags=flags, criteria=crit)
    per_view = []
    for oo, ii, r, t in zip(o, i, rv, tv):
        uv, _ = cv2.fisheye.projectPoints(oo.reshape(-1, 1, 3), r, t, K, D)
        per_view.append(float(np.sqrt(np.mean(np.sum((uv.reshape(-1, 2) - ii.reshape(-1, 2)) ** 2, -1)))))
    return {"K": K, "D": D.ravel(), "rms_px": float(rms), "per_view_px": per_view, "monotonic": lens_monotonic(D)}


# ---------------------------------------------------------------------------------------------------- overhead

def _tag_world(links: Links, q: np.ndarray, R_lt, p_lt, tag_pts) -> np.ndarray:
    R_l, p_l = links(q)
    return np.einsum("nij,kj->nki", R_l, tag_pts @ R_lt.T + p_lt) + p_l[:, None]  # (N, 4, 3)


def tag_on_face(links: Links, theta: float, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """The tag glued flat on the back face: rotated by theta about the face normal and shifted (dx, dy) in the face
    plane from the face centre -> link-from-tag rotation and position."""
    return links.back_R @ rot_z(theta), links.back_p + links.back_R @ np.array([dx, dy, 0.0])


def fit_overhead(links: Links, K, D, qs: np.ndarray, corners: np.ndarray, tag_pts: np.ndarray,
                 fit_offsets: bool = True, offset_prior_deg: float = 2.0, joints: tuple = FIT_JOINTS,
                 fixed_offsets: np.ndarray | None = None) -> dict:
    """Tag corners (N, 4, 2) at encoder angles qs (N, 5) -> camera pose, tag placement on the back face, offsets
    of `joints` (the others held at `fixed_offsets`, default 0). The tag lies flat on the blade (its normal is the
    face normal): with a free tag orientation the wrist_roll offset would be unobservable (a roll offset = the tag
    turned about the paddle axis), yet it is the paddle's sideways tilt."""
    n = len(qs)
    joints = list(joints) if fit_offsets else []
    nj = len(joints)
    d_fixed = np.zeros(5) if fixed_offsets is None else np.asarray(fixed_offsets, float).copy()

    def unpack(x):
        d = d_fixed.copy()
        d[joints] = x[9:9 + nj]
        R_lt, p_lt = tag_on_face(links, x[6], x[7], x[8])
        return rot(x[0:3]), x[3:6], R_lt, p_lt, d

    def res(x):
        R_wc, p_wc, R_lt, p_lt, d = unpack(x)
        X = _tag_world(links, qs + d, R_lt, p_lt, tag_pts)
        r = (project(K, D, R_wc, p_wc, X.reshape(-1, 3)) - corners.reshape(-1, 2)).ravel()
        return np.concatenate([r, x[9:9 + nj] / math.radians(offset_prior_deg)])

    # Initial guesses: the tag centred on the back face, its rotation in the face unknown (8 tries); the camera
    # from each try's per-view PnP estimates (the view nearest their median position).
    views = [pnp(K, D, tag_pts, c, square=True) for c in corners]
    R_l, p_l = links(qs + d_fixed)
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
        x0 = np.concatenate([rvec(R0), p0, [theta, 0.0, 0.0], d_fixed[joints]])
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

def board_pnp(K, D, board_pts: np.ndarray, ids: np.ndarray, uv: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Both planar PnP solutions (camera-from-board R, t) of one view, the better one first."""
    xy = cv2.fisheye.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), K, np.asarray(D, float).reshape(4, 1))
    n_sol, rvs, tvs, _ = cv2.solvePnPGeneric(np.asarray(board_pts[ids], float), xy.reshape(-1, 2), np.eye(3), None,
                                             flags=cv2.SOLVEPNP_IPPE)
    return [(rot(rv), tv.ravel()) for rv, tv in zip(rvs[:n_sol], tvs[:n_sol])]


def hand_eye(R_l: np.ndarray, p_l: np.ndarray, R_c: list, t_c: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form camera mount X (link-from-camera) from link poses (world-from-link) and board poses
    (camera-from-board) of a fixed board: A X = X B over view pairs (rotation axes aligned, then translation by
    least squares). Also returns each view's median disagreement (deg) between the link's and the camera's turn
    to the other views: the turns must match whatever X is, so a large value means the board, the camera or the
    paddle moved, not a fitting problem."""
    n = len(R_c)
    a, b, rows = [], [], []
    dis = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            RA = R_l[j].T @ R_l[i]
            RB = R_c[j] @ R_c[i].T
            dis[i, j] = abs(angle_deg(RA, np.eye(3)) - angle_deg(RB, np.eye(3)))
            if j > i and angle_deg(RA, np.eye(3)) > 5:
                a.append(rvec(RA))
                b.append(rvec(RB))
                rows.append((RA, R_l[j].T @ (p_l[i] - p_l[j]), t_c[j] - RB @ t_c[i]))
    U, _, Vt = np.linalg.svd(np.array(b).T @ np.array(a))
    R_x = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    lhs = np.concatenate([RA - np.eye(3) for RA, _, _ in rows])
    y = np.concatenate([R_x @ tB - tA for _, tA, tB in rows])
    p_x = np.linalg.lstsq(lhs, y, rcond=None)[0]
    return R_x, p_x, np.nanmedian(dis + np.diag(np.full(n, np.nan)), axis=1) if n > 1 else np.zeros(n)


def fit_wrist(links: Links, K, D, qs: np.ndarray, views: list[tuple[np.ndarray, np.ndarray]],
              board_pts: np.ndarray, mount0: tuple[np.ndarray, np.ndarray] | None = None,
              board0: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Board corners seen by the wrist camera: views [(ids, uv)] at encoder angles qs (true angles) -> camera mount
    (paddle-link-from-camera) and board pose (world-from-board). Starting guesses: the closed-form hand-eye mount
    (the real module sits ~90 deg about its axis from the CAD frame, out of reach of a local fit from CAD) with the
    board where it puts it; and `mount0` with `board0` (where the board was meant to stand) turned 0/90/180/270 deg
    in its plane and both planar PnP solutions of the views with the most corners. The lowest-error fit wins.
    `view_disagree_deg`: see hand_eye."""
    R_l, p_l = links(qs)
    R_lc0, p_lc0 = mount0 if mount0 is not None else (links.cad_cam_R, links.cad_cam_p)
    mid = board_pts.mean(0)
    pnps = [board_pnp(K, D, board_pts, ids, uv) for ids, uv in views]
    R_x, p_x, disagree = hand_eye(R_l, p_l, [s[0][0] for s in pnps], [s[0][1] for s in pnps])
    starts = []
    if len(views) >= 3:
        boards = [(R_l[i] @ R_x @ s[0][0], R_l[i] @ (R_x @ s[0][1] + p_x) + p_l[i]) for i, s in enumerate(pnps)]
        Rm = np.mean([b[0] for b in boards], axis=0)
        U, _, Vt = np.linalg.svd(Rm)
        starts.append((R_x, p_x, U @ Vt, np.mean([b[1] for b in boards], axis=0)))
    if board0 is not None:
        R_b, p_b = board0
        centre = R_b @ mid + p_b
        for k in range(4):
            R = R_b @ rot_z(k * math.pi / 2)
            starts.append((R_lc0, p_lc0, R, centre - R @ mid))
    for i in np.argsort([-len(ids) for ids, _ in views])[:3]:
        R_wc = R_l[i] @ R_lc0
        p_wc = R_l[i] @ p_lc0 + p_l[i]
        for R_cb, t_cb in pnps[i]:
            starts.append((R_lc0, p_lc0, R_wc @ R_cb, R_wc @ t_cb + p_wc))
    obs = np.concatenate([uv for _, uv in views])

    def res(x):
        R_lc, p_lc, R_wb, p_wb = rot(x[0:3]), x[3:6], rot(x[6:9]), x[9:12]
        out = []
        for i, (ids, _) in enumerate(views):
            R_wc = R_l[i] @ R_lc
            p_wc = R_l[i] @ p_lc + p_l[i]
            out.append(project(K, D, R_wc, p_wc, board_pts[ids] @ R_wb.T + p_wb))
        return (np.concatenate(out) - obs).ravel()

    best = None
    for R_m, p_m, R_wb0, p_wb0 in starts:
        sol = least_squares(res, np.concatenate([rvec(R_m), p_m, rvec(R_wb0), p_wb0]), method="lm", max_nfev=4000)
        if best is None or sol.cost < best.cost:
            best = sol
    r = res(best.x).reshape(-1, 2)
    return {"R": rot(best.x[0:3]), "p": best.x[3:6], "board_R": rot(best.x[6:9]), "board_p": best.x[9:12],
            "rms_px": float(np.sqrt((r ** 2).sum(-1).mean())), "view_disagree_deg": disagree.tolist(),
            "per_view_px": [float(np.sqrt((rr ** 2).sum(-1).mean()))
                            for rr in np.split(r, np.cumsum([len(i) for i, _ in views])[:-1])]}


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


def _arc_init(a: dict, k: float = 1.0) -> np.ndarray:
    """Start (p0, v0) of a flight from its monocular 3D points (range from the apparent size, scale k)."""
    u = np.concatenate([a["z"][:, :2], np.ones((len(a["t"]), 1))], 1)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rng = BALL_RADIUS / np.sin(a["z"][:, 2] / (2 * k))
    X = a["p"] + np.einsum("nij,nj->ni", a["R"], rng[:, None] * u)
    t = a["t"] - a["t"][0]
    A = np.stack([np.ones_like(t), t, 0.5 * t * t], 1)
    coef = np.linalg.lstsq(A, X, rcond=None)[0]
    return np.concatenate([coef[0], coef[1]])


def fit_gravity(arcs: list[dict], fix_magnitude: bool = False, n_cams: int = 1) -> dict:
    """Ball flights, each {t (n,), R (n, 3, 3), p (n, 3) camera poses, z (n, 3), sigma (n, 3), cam (n,) camera
    index} -> gravity vector (3,), its magnitude and tilt from -z (deg), per-arc normalized RMS.
    `fix_magnitude`: |g| = 9.81 and a size scale per camera (detected / true angular diameter) are fitted instead
    of the magnitude: with one camera, range comes from the apparent size, and a biased size scales every distance.
    The Jacobian is sparse (each flight's start depends only on its own points): finite differences perturb all
    flights at once."""
    from scipy.sparse import lil_matrix

    nk = n_cams if fix_magnitude else 0
    ng = 2 if fix_magnitude else 3
    head = ng + nk
    inits = [_arc_init(a, 1.0) for a in arcs]

    def gvec(x):
        if not fix_magnitude:
            return x[:3]
        a, b = x[0], x[1]  # tilts about y and x
        return G * np.array([math.sin(a), -math.sin(b) * math.cos(a), -math.cos(a) * math.cos(b)])

    def res(x):
        g = gvec(x)
        out = []
        for k, a in enumerate(arcs):
            o = head + 6 * k
            P = _fly(x[o:o + 3], x[o + 3:o + 6], g, a["t"] - a["t"][0])
            z = measure_ball(P, a["R"], a["p"])
            if fix_magnitude:
                z[:, 2] *= x[ng + a.get("cam", np.zeros(len(a["t"]), int))]
            out.append(((z - a["z"]) / a["sigma"]).ravel())
        return np.concatenate(out)

    n_res = sum(3 * len(a["t"]) for a in arcs)
    S = lil_matrix((n_res, head + 6 * len(arcs)), dtype=int)
    row = 0
    for k, a in enumerate(arcs):
        m = 3 * len(a["t"])
        S[row:row + m, :head] = 1
        S[row:row + m, head + 6 * k: head + 6 * k + 6] = 1
        row += m
    x0 = np.concatenate([[0.0, 0.0] if fix_magnitude else [0.0, 0.0, -G], np.ones(nk)] + inits)
    sol = least_squares(res, x0, x_scale="jac", jac_sparsity=S, method="trf")
    g = gvec(sol.x)
    r = res(sol.x)
    per_arc, i = [], 0
    for a in arcs:
        m = 3 * len(a["t"])
        per_arc.append(float(np.sqrt(np.mean(r[i:i + m] ** 2))))
        i += m
    tilt = float(np.degrees(np.arccos(np.clip(-g[2] / np.linalg.norm(g), -1, 1))))
    return {"g": g, "magnitude": float(np.linalg.norm(g)), "tilt_deg": tilt, "per_arc_rms": per_arc,
            "size_scale": sol.x[ng:ng + nk] if fix_magnitude else None}


# ---------------------------------------------------------------------------------------------------- poses

def overhead_poses(links: Links, model, cam_p: np.ndarray, cam_axis: np.ndarray, n: int = 20, seed: int = 0,
                   max_angle_deg: float = 40.0, tries: int = 20000, camera: dict | None = None,
                   margin_px: float = 40.0, min_tag_px: float = 110.0) -> np.ndarray:
    """Arm poses (n, 5) showing the paddle's back face (the tag) to a camera near `cam_p` looking along `cam_axis`:
    face normal within max_angle of the camera direction, 0.2-0.6 m away, payload clear of the table and blade
    clear of the arm (`model`: arm.ArmModel). With `camera` (K, D, R, p world-from-camera, width, height, tag_pts,
    tag_R, tag_p), each candidate's tag corners are projected through the real lens: all four inside the picture
    with `margin_px`, the tag at least `min_tag_px` across (without it, the tag within max_angle of the axis: the
    first real run lost half its poses that way, the lens is only 51 deg tall). Spread over picture position, face
    angle and arm shape (lift, elbow: separates the joint offsets from the camera pose), ordered for short moves."""
    rng = np.random.default_rng(seed)
    lo, hi = model.range[:, 0] + 0.1, model.range[:, 1] - 0.1
    box_lo = np.array([-0.35, -1.3, -0.2, -1.3, -2.75])
    box_hi = np.array([0.35, 0.1, 1.6, 1.3, 2.75])
    cos_max = math.cos(math.radians(max_angle_deg))
    axis = cam_axis / np.linalg.norm(cam_axis)
    q = rng.uniform(np.maximum(box_lo, lo), np.minimum(box_hi, hi), size=(tries, 5))
    q[:, 4] = np.sign(q[:, 4]) * rng.uniform(1.9, 2.7, tries)  # back face up: roll well past +-90 deg
    # 0.1 rad inside the ranges in the angles that will be sent (true - calibrated offsets).
    raw_off = np.zeros(5) if camera is None or "offsets" not in camera else np.asarray(camera["offsets"])
    q = q[((q - raw_off) >= lo).all(1) & ((q - raw_off) <= hi).all(1)]
    tries = len(q)
    R_l, p_l = links(q)
    c = np.einsum("nij,j->ni", R_l, links.back_p) + p_l
    nrm = np.einsum("nij,j->ni", R_l, links.back_R[:, 2])
    to_cam = cam_p - c
    dist = np.linalg.norm(to_cam, axis=1)
    ok = (np.einsum("ni,ni->n", nrm, to_cam) / dist > cos_max) & (dist > 0.2) & (dist < 0.6) & (c[:, 2] > 0.15)
    uvc = np.zeros((tries, 2))
    if camera is None:
        ok &= -(to_cam @ axis) / dist > cos_max
    else:
        W, H = camera["width"], camera["height"]
        X = np.einsum("nij,kj->nki", R_l, camera["tag_pts"] @ np.asarray(camera["tag_R"]).T + camera["tag_p"]) + p_l[:, None]
        Xc = (X.reshape(-1, 3) - camera["p"]) @ np.asarray(camera["R"])
        front = (Xc[:, 2] > 0.05).reshape(tries, 4).all(1)
        uv = project(camera["K"], camera["D"], np.asarray(camera["R"]), np.asarray(camera["p"]), X.reshape(-1, 3))
        uv = uv.reshape(tries, 4, 2)
        inside = ((uv[..., 0] > margin_px) & (uv[..., 0] < W - margin_px) & (uv[..., 1] > margin_px)
                  & (uv[..., 1] < H - margin_px)).all(1)
        span = np.minimum(np.linalg.norm(uv[:, 0] - uv[:, 2], axis=1), np.linalg.norm(uv[:, 1] - uv[:, 3], axis=1))
        ok &= front & inside & (span > min_tag_px)
        uvc = uv.mean(1) / np.array([W, H])
    good, feats = [], []
    for i in np.flatnonzero(ok):
        model.forward(q[i])
        if model.clearance() < 0.06 or model.blade_hits():
            continue
        good.append(q[i])
        feats.append(np.concatenate([c[i] / 0.1, nrm[i], 2 * uvc[i], q[i, 1:3] / 0.5]))
        if len(good) >= 50 * n:
            break
    if len(good) < n:
        raise RuntimeError(f"only {len(good)} poses show the tag to the camera; is the camera guess right?")
    good, feats = np.array(good), np.array(feats)
    chosen = [int(np.argmax(feats[:, 2]))]  # farthest-point sampling
    d = np.linalg.norm(feats - feats[chosen[0]], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(feats - feats[k], axis=1))
    return order_poses(good[chosen])


def leave_one_out(fit_fn, n: int) -> dict:
    """Jackknife spread of a fit: `fit_fn(indices)` -> (camera position (3,), rotation (3, 3), offsets (5,)).
    Standard errors of the position (m), rotation (deg) and offsets (deg)."""
    full = fit_fn(list(range(n)))
    dp, da, do = [], [], []
    for i in range(n):
        p, R, d = fit_fn([j for j in range(n) if j != i])
        dp.append(np.linalg.norm(p - full[0]))
        da.append(angle_deg(R, full[1]))
        do.append(np.degrees(d - full[2]))
    s = math.sqrt(n - 1)
    return {"pos_m": s * float(np.sqrt(np.mean(np.square(dp)))), "rot_deg": s * float(np.sqrt(np.mean(np.square(da)))),
            "offsets_deg": s * np.sqrt(np.mean(np.square(do), axis=0))}


WRIST_BOARD_CENTRE = np.array([0.35, 0.0, 0.002])  # flat on the table: 35 cm out on the arm's centre line


def table_board(board_pts: np.ndarray, centre: np.ndarray = WRIST_BOARD_CENTRE) -> tuple[np.ndarray, np.ndarray]:
    """World-from-board pose of the board lying printed side up on the table, centred at `centre`, its long side
    across the arm (along robot y). OpenCV's board z points into the board, so down."""
    R = np.diag([1.0, -1.0, -1.0])
    if np.ptp(board_pts[:, 0]) > np.ptp(board_pts[:, 1]):  # board x is the long side: turn it across
        R = R @ rot_z(math.pi / 2)
    return R, centre - R @ board_pts.mean(0)


def wrist_poses(links: Links, model, camera: dict, n: int = 24, seed: int = 0, tries: int = 60000,
                min_corners: int = 18, max_view_deg: float = 40.0, max_dist: float = 0.36) -> np.ndarray:
    """Arm poses (n, 5) for the wrist camera to see a board lying on the table (a hand-held or propped board
    drifts a few degrees over the run, which the fit can't tell from a mount error): the paddle rolled over and
    tipped down so the half of the picture beyond the blade looks at the board. `camera`: K, D, R, p
    (paddle-link-from-camera, rough), board_R, board_p, board_pts, usable (pixels (N, 2) -> bool: beyond the blade,
    in the calibrated part of the lens), offsets (joint offsets: candidates stay 0.1 rad inside the ranges in the
    angles that will be sent). Each pose: >= `min_corners` usable corners, camera 0.18-`max_dist` m from the
    board centre within `max_view_deg` of its normal (the first flat-board run found the board at only 7 of 24
    poses, all <= 62 deg / 0.41 m; ArUco markers stop decoding when foreshortened further), payload >= 6 cm above
    the table, blade clear of the arm. Spread over joint angles and camera position (separates the mount from the
    board pose), ordered for short moves."""
    rng = np.random.default_rng(seed)
    lo, hi = model.range[:, 0] + 0.1, model.range[:, 1] - 0.1
    box_lo = np.array([-0.6, -1.3, -0.2, -1.4, -3.0])
    box_hi = np.array([0.6, 0.2, 1.7, 1.4, 3.0])
    q = rng.uniform(np.maximum(box_lo, lo), np.minimum(box_hi, hi), size=(tries, 5))
    off = np.asarray(camera.get("offsets", np.zeros(5)))
    q = q[((q - off) >= lo).all(1) & ((q - off) <= hi).all(1)]
    R_l, p_l = links(q)
    R_c = R_l @ camera["R"]
    c = np.einsum("nij,j->ni", R_l, camera["p"]) + p_l
    X = camera["board_pts"] @ camera["board_R"].T + camera["board_p"]
    ctr = X.mean(0)
    up = -camera["board_R"][:, 2]  # printed side's normal
    d = ctr - c
    dist = np.linalg.norm(d, axis=1)
    ok = ((dist > 0.18) & (dist < max_dist) & (-(d @ up) / dist > math.cos(math.radians(max_view_deg)))
          & (np.einsum("ni,ni->n", R_c[:, :, 2], d) / dist > 0.6))
    good, feats = [], []
    for i in np.flatnonzero(ok):
        if (((X - c[i]) @ R_c[i])[:, 2] < 0.05).any():
            continue
        if camera["usable"](project(camera["K"], camera["D"], R_c[i], c[i], X)).sum() < min_corners:
            continue
        model.forward(q[i])
        if model.clearance() < 0.06 or model.blade_hits():
            continue
        good.append(q[i])
        feats.append(np.concatenate([q[i] / 0.5, c[i] / 0.1]))
    if len(good) < n:
        raise RuntimeError(f"only {len(good)} poses see the board well; is the camera mount guess right?")
    good, feats = np.array(good), np.array(feats)
    chosen = [0]  # farthest-point sampling
    dd = np.linalg.norm(feats - feats[0], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(dd))
        chosen.append(k)
        dd = np.minimum(dd, np.linalg.norm(feats - feats[k], axis=1))
    return order_poses(good[chosen])


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
