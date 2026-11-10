"""Ball state estimation from camera detections: an extended Kalman filter shared by the simulator and deployment.

Pure torch, batched over worlds (B = 1 on the real arm). Frames: everything in the robot base/world frame, z up.

Measurements. One detection = one camera at one instant, any camera, fixed or moving: the caller passes the camera
pose at the capture time (for the wrist camera: forward kinematics of the encoder angles, interpolated with
`TickHistory`). The detection is the undistorted ball centre in normalized image coordinates (x/z, y/z; OpenCV
camera axes: x right, y down, z forward) plus the ball's apparent angular diameter alpha = 2 asin(r / range), which
gives range. One camera alone therefore fixes the ball in 3D (range only coarsely), and the filter fuses whichever
cameras see the ball, whenever they see it.

Flight model: gravity + quadratic drag (Cd 0.50, measured), integrated with RK4 in steps of at most `max_step`.

Paddle-aware prediction. Propagation checks every step against the paddle face (centre, normal, velocity from the
arm's forward kinematics, `TickHistory`): a crossing on the blade reflects the ball with the nominal restitution
(normal) and tangential keep factor, and inflates the velocity covariance by their uncertainty. So the estimate
predicts through a bounce that the cameras have not reported yet (their latency is tens of ms), and measurements
after the bounce update a state that already moves upward.

Time. The filter state lives at the time of the last processed measurement (never older than `now - hold`).
`estimate(now)` predicts a copy to `now`, compensating the camera latency. Measurements arriving out of order (from
the other camera, captured before the state time) are fused at the state time through the backward-predicted
measurement (Bar-Shalom's approximate out-of-sequence update without process noise).

Track management. Untracked worlds initialize from any single detection (range from the apparent size, velocity 0
with a large variance). Detections failing the chi-square gate are dropped; `reinit_after` consecutive failures
restart the track from the next one (a ball caught, held, or hit by the holder). A track with no accepted
measurement for `lost_after` is dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from pingpong_sim.ball import BALL_RADIUS, DRAG_K, G


_PADDLE_KEYS = ("center", "normal", "vel")  # what propagation reads from the paddle history


@dataclass
class BallFilterCfg:
    accel_noise: float = 0.5  # m/s^2, white acceleration noise (Cd +-0.03 is ~0.06 m/s^2; spin, model error)
    restitution: float = 0.58  # normal, measured on the blade centre (measurements/ball)
    restitution_std: float = 0.06
    tangential_keep: float = 0.5  # fraction of the tangential speed kept (measured 0.37-0.41, sim 0.6)
    tangential_std: float = 0.2
    paddle_speed_std: float = 0.05  # m/s, paddle velocity error at impact (encoder lag, gear flex)
    blade_radius: float = 0.075  # m, around the face centre
    gate_chi2: float = 16.27  # 3 dof, 99.9 %
    reinit_after: int = 3  # consecutive gate failures
    init_vel_std: float = 3.0  # m/s
    lost_after: float = 0.5  # s without an accepted measurement
    hold: float = 0.2  # s, the state is advanced to at least now - hold (bounds the prediction to now)
    max_step: float = 0.02  # s, RK4 step
    stamp_std: float = 0.002  # s, timestamp error per camera (calibration residual): adds (H v)^2 stamp_std^2 to R


def _accel(v: torch.Tensor) -> torch.Tensor:
    a = -DRAG_K * v.norm(dim=-1, keepdim=True) * v
    a[..., 2] -= G
    return a


def flight(p: torch.Tensor, v: torch.Tensor, h: torch.Tensor):
    """One RK4 step of length h (B,; may be negative) of gravity + drag flight."""
    h = h[:, None]
    k1p, k1v = v, _accel(v)
    k2p = v + 0.5 * h * k1v
    k2v = _accel(k2p)
    k3p = v + 0.5 * h * k2v
    k3v = _accel(k3p)
    k4p = v + h * k3v
    k4v = _accel(k4p)
    return p + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p), v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)


def _transition(v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """State transition Jacobian (B, 6, 6) of a flight step of length h, second order in h, at velocity v."""
    B = v.shape[0]
    speed = v.norm(dim=-1).clamp_min(1e-6)[:, None, None]
    eye = torch.eye(3, device=v.device).expand(B, 3, 3)
    Jd = -DRAG_K * (speed * eye + v[:, :, None] * v[:, None, :] / speed)  # d(accel)/dv
    h = h[:, None, None]
    F = torch.eye(6, device=v.device).repeat(B, 1, 1)
    F[:, :3, 3:] = h * eye + 0.5 * h * h * Jd
    F[:, 3:, 3:] = eye + h * Jd + 0.5 * h * h * (Jd @ Jd)
    return F


def _process_noise(q: float, h: torch.Tensor) -> torch.Tensor:
    """Covariance (B, 6, 6) of white acceleration noise of density q^2 over a step h >= 0."""
    h = h[:, None, None]
    eye = torch.eye(3, device=h.device)
    top = torch.cat([h**3 / 3 * eye, h**2 / 2 * eye], dim=2)
    bottom = torch.cat([h**2 / 2 * eye, h * eye], dim=2)
    return q * q * torch.cat([top, bottom], dim=1)


def orthonormalize(R: torch.Tensor) -> torch.Tensor:
    """Nearest-ish rotation to a (B, 3, 3) matrix close to one (Gram-Schmidt on the first two columns)."""
    x = R[:, :, 0] / R[:, :, 0].norm(dim=-1, keepdim=True)
    y = R[:, :, 1] - (R[:, :, 1] * x).sum(-1, keepdim=True) * x
    y = y / y.norm(dim=-1, keepdim=True)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=2)


class TickHistory:
    """Ring buffer of per-world tensors sampled on a common clock every `dt`, linearly interpolated at per-world
    times (clamped to the buffer). Holds the paddle state (centre, normal, velocity) and camera poses."""

    def __init__(self, num_envs: int, length: int, dt: float, shapes: dict[str, tuple[int, ...]], device):
        self.n, self.L, self.dt = num_envs, length, dt
        self.buf = {k: torch.zeros(num_envs, length, *s, device=device) for k, s in shapes.items()}
        self.count = 0  # samples pushed
        self.t_last = 0.0

    def push(self, t: float, **values: torch.Tensor) -> None:
        i = self.count % self.L
        for k, v in values.items():
            self.buf[k][:, i] = v
        self.count += 1
        self.t_last = t

    def fill(self, env_ids: torch.Tensor, **values: torch.Tensor) -> None:
        """Overwrite the whole history of env_ids with one sample (after a reset)."""
        for k, v in values.items():
            self.buf[k][env_ids] = v[:, None].expand_as(self.buf[k][env_ids])

    def hold_last(self, env_ids: torch.Tensor) -> None:
        """Overwrite the whole history of env_ids with their latest sample (after a reset)."""
        last = (self.count - 1) % self.L
        for b in self.buf.values():
            b[env_ids] = b[env_ids, last][:, None].expand_as(b[env_ids])

    def samples(self, t: torch.Tensor, keys=None):
        """The two samples around per-world times t (B,): dicts of values (`keys`, default all) at the earlier and
        the later tick, and the time from the earlier tick to t (B,), clamped to the buffer."""
        n_valid = min(self.count, self.L)
        t_first = self.t_last - (n_valid - 1) * self.dt
        x = ((t - t_first) / self.dt).clamp(0, n_valid - 1)
        i0 = x.floor().long().clamp(max=n_valid - 2) if n_valid > 1 else torch.zeros_like(x, dtype=torch.long)
        start = self.count - n_valid
        rows = torch.arange(self.n, device=t.device)
        i1 = (i0 + 1).clamp(max=n_valid - 1)
        bufs = self.buf if keys is None else {k: self.buf[k] for k in keys}
        a = {k: b[rows, (start + i0) % self.L] for k, b in bufs.items()}
        c = {k: b[rows, (start + i1) % self.L] for k, b in bufs.items()}
        return a, c, (x - i0) * self.dt

    def at(self, t: torch.Tensor, keys=None) -> dict[str, torch.Tensor]:
        a, c, tau = self.samples(t, keys)
        w = (tau / self.dt).clamp(0, 1)
        return {k: a[k] + w.view(-1, *([1] * (a[k].dim() - 1))) * (c[k] - a[k]) for k in a}


def measurement_sigma(theta: torch.Tensor, alpha: torch.Tensor, focal, pixel_noise: float, diameter_noise: float,
                      angle_floor: float, size_floor: float) -> torch.Tensor:
    """Standard deviations (B, 3) the filter assumes for a detection [x/z, y/z, alpha] at angle `theta` from the
    optical axis with angular diameter `alpha` (rad), for a lens of `focal` px/rad: pixel noise as an angle plus a
    calibration floor, stretched by the tan projection (sec^2 theta); diameter noise plus a relative size floor.
    Shared by the simulated cameras and the real ones."""
    sec2 = 1 / torch.cos(theta) ** 2
    s_ang = torch.sqrt(torch.as_tensor((pixel_noise / focal) ** 2 + angle_floor**2)) * sec2  # focal: float or (B,)
    s_size = torch.sqrt((diameter_noise / focal) ** 2 + (size_floor * alpha) ** 2)
    return torch.stack([s_ang, s_ang, s_size], dim=1)


def project(p: torch.Tensor, cam_R: torch.Tensor, cam_p: torch.Tensor):
    """Measurement model. Ball centres p (B, 3) seen from cameras (world-from-camera R, centre) -> [x/z, y/z, alpha]
    (B, 3), its Jacobian w.r.t. p (B, 3, 3), and whether the ball is in front of the camera (B,)."""
    pc = ((p - cam_p)[:, None, :] @ cam_R).squeeze(1)  # camera frame
    X, Y, Z = pc.unbind(-1)
    front = Z > 0.02
    Zs = torch.where(front, Z, torch.full_like(Z, 0.02))
    rho = pc.norm(dim=-1).clamp_min(BALL_RADIUS * 1.05)
    s = BALL_RADIUS / rho
    z = torch.stack([X / Zs, Y / Zs, 2 * torch.asin(s)], dim=-1)
    zero = torch.zeros_like(X)
    dxn = torch.stack([1 / Zs, zero, -X / Zs**2], -1)
    dyn = torch.stack([zero, 1 / Zs, -Y / Zs**2], -1)
    dal = (-2 * s / (rho * torch.sqrt(1 - s * s)))[:, None] * pc / rho[:, None]
    Hc = torch.stack([dxn, dyn, dal], dim=1)  # (B, 3, 3) w.r.t. the camera-frame point
    return z, Hc @ cam_R.transpose(1, 2), front


# -- one world on the CPU ------------------------------------------------------------------------------------------
# The same propagation (flight, transition, process noise, paddle history, bounce) in float64 numpy for B = 1:
# what the robot runs every tick. tests/test_fast_paths.py keeps it equal to the batched torch version.


def _accel_np(v: np.ndarray) -> np.ndarray:
    a = -DRAG_K * math.sqrt(v @ v) * v
    a[2] -= G
    return a


def flight_np(p: np.ndarray, v: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    k1v = _accel_np(v)
    k2p = v + 0.5 * h * k1v
    k2v = _accel_np(k2p)
    k3p = v + 0.5 * h * k2v
    k3v = _accel_np(k3p)
    k4p = v + h * k3v
    k4v = _accel_np(k4p)
    return p + h / 6 * (v + 2 * k2p + 2 * k3p + k4p), v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)


_EYE3, _EYE6 = np.eye(3), np.eye(6)


def _transition_np(v: np.ndarray, h: float) -> np.ndarray:
    speed = max(math.sqrt(v @ v), 1e-6)
    Jd = -DRAG_K * (speed * _EYE3 + np.outer(v, v) / speed)
    F = _EYE6.copy()
    F[:3, 3:] = h * _EYE3 + 0.5 * h * h * Jd
    F[3:, 3:] = _EYE3 + h * Jd + 0.5 * h * h * (Jd @ Jd)
    return F


def _process_noise_np(q: float, h: float) -> np.ndarray:
    Q = np.zeros((6, 6))
    Q[:3, :3] = h**3 / 3 * _EYE3
    Q[:3, 3:] = Q[3:, :3] = h**2 / 2 * _EYE3
    Q[3:, 3:] = h * _EYE3
    return q * q * Q


class _HistoryNp:
    """World 0 of a TickHistory as numpy, interpolated like TickHistory.at."""

    def __init__(self, h: TickHistory, keys):
        self.n_valid = min(h.count, h.L)
        self.t_first = h.t_last - (self.n_valid - 1) * h.dt
        self.start, self.L, self.dt = h.count - self.n_valid, h.L, h.dt
        self.buf = {k: h.buf[k][0].double().numpy() for k in keys}

    def at(self, t: float) -> dict[str, np.ndarray]:
        n = self.n_valid
        x = min(max((t - self.t_first) / self.dt, 0.0), n - 1)
        i0 = min(math.floor(x), n - 2) if n > 1 else 0
        i1 = min(i0 + 1, n - 1)
        w = min(max((x - i0), 0.0), 1.0)
        a, c = (self.start + i0) % self.L, (self.start + i1) % self.L
        return {k: b[a] + w * (b[c] - b[a]) for k, b in self.buf.items()}


def _bounce_np(cfg: BallFilterCfg, p, v, p1, v1, P, h: float, a: dict, b: dict):
    s0 = (p - a["center"]) @ a["normal"] - BALL_RADIUS
    s1 = (p1 - b["center"]) @ b["normal"] - BALL_RADIUS
    f = min(max(s0 / max(s0 - s1, 1e-9), 0.0), 1.0)
    pc, vc = p + f * (p1 - p), v + f * (v1 - v)
    c = a["center"] + f * (b["center"] - a["center"])
    nrm = a["normal"] + f * (b["normal"] - a["normal"])
    nrm = nrm / max(math.sqrt(nrm @ nrm), 1e-9)
    vp = a["vel"] + f * (b["vel"] - a["vel"])
    d = pc - c
    lateral = math.sqrt(max((d - (d @ nrm) * nrm) @ (d - (d @ nrm) * nrm), 0.0))
    rel = vc - vp
    vn = rel @ nrm
    vt = rel - vn * nrm
    if not (s0 > -0.015 and s1 < 0 and vn < 0 and lateral < cfg.blade_radius and h > 0):
        return p1, v1, P
    e, kt = cfg.restitution, cfg.tangential_keep
    vout = vp - e * vn * nrm + kt * vt
    pb, vb = flight_np(pc, vout, (1 - f) * h)
    nn = np.outer(nrm, nrm)
    M = -e * nn + kt * (_EYE3 - nn)
    J = _EYE6.copy()
    J[3:, 3:] = M
    Qv = (cfg.restitution_std * vn) ** 2 * nn + cfg.tangential_std**2 * (vt @ vt) * (_EYE3 - nn) \
        + cfg.paddle_speed_std**2 * _EYE3
    Pb = J @ P @ J.T
    Pb[3:, 3:] += Qv
    return pb, vb, Pb


def _propagate_one(cfg: BallFilterCfg, x: torch.Tensor, P: torch.Tensor, t0: float, t1: float,
                   paddle: TickHistory | None):
    dt = max(t1 - t0, 0.0)
    if dt <= 0.0:
        return x, P
    n_steps = max(1, math.ceil(dt / cfg.max_step - 1e-9))
    h = dt / n_steps
    xs, Ps = x[0].double().numpy(), P[0].double().numpy()
    hist = _HistoryNp(paddle, _PADDLE_KEYS) if paddle is not None else None
    t = t0
    pad = hist.at(t) if hist is not None else None
    Q = _process_noise_np(cfg.accel_noise, h)
    for _ in range(n_steps):
        p, v = xs[:3], xs[3:]
        p1, v1 = flight_np(p, v, h)
        F = _transition_np(0.5 * (v + v1), h)
        Ps = F @ Ps @ F.T + Q
        t = t + h
        if hist is not None:
            pad_next = hist.at(t)
            p1, v1, Ps = _bounce_np(cfg, p, v, p1, v1, Ps, h, pad, pad_next)
            pad = pad_next
        xs = np.concatenate([p1, v1])
    return (torch.tensor(xs, dtype=x.dtype)[None], torch.tensor(Ps, dtype=P.dtype)[None])


class BallFilter:
    def __init__(self, cfg: BallFilterCfg, num_envs: int, device):
        self.cfg, self.n, self.dev = cfg, num_envs, device
        self.x = torch.zeros(num_envs, 6, device=device)
        self.P = torch.eye(6, device=device).repeat(num_envs, 1, 1)
        self.t = torch.zeros(num_envs, device=device)  # state time
        self.tracked = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.rejects = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_meas = torch.full((num_envs,), -math.inf, device=device)  # time of the last accepted detection
        self.accepted = torch.zeros(num_envs, dtype=torch.long, device=device)  # counters for diagnostics
        self.rejected = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.tracked[ids] = False
        self.rejects[ids] = 0
        self.last_meas[ids] = -math.inf

    # -- propagation -------------------------------------------------------------------------------------------------

    def propagate(self, x, P, t0, t1, paddle: TickHistory | None):
        """Predict (x, P) from t0 to t1 >= t0 (B,), bouncing off the paddle face (`paddle` holds center, normal,
        vel). Worlds with t1 == t0 are returned unchanged. One world on the CPU takes the float64 numpy path (same
        steps; batched torch pays ~5 us of dispatch per tiny op, 10x the robot's control budget)."""
        if self.n == 1 and x.device.type == "cpu":
            return _propagate_one(self.cfg, x, P, float(t0[0]), float(t1[0]), paddle)
        return self._propagate_batched(x, P, t0, t1, paddle)

    def _propagate_batched(self, x, P, t0, t1, paddle: TickHistory | None):
        cfg = self.cfg
        dt = (t1 - t0).clamp_min(0)
        dt_max = float(dt.max()) if dt.numel() else 0.0
        if dt_max <= 0.0:
            return x, P
        n_steps = max(1, math.ceil(dt_max / cfg.max_step - 1e-9))
        h = dt / n_steps
        t = t0
        pad = paddle.at(t, _PADDLE_KEYS) if paddle is not None else None
        for _ in range(n_steps):
            p, v = x[:, :3], x[:, 3:]
            p1, v1 = flight(p, v, h)
            F = _transition(0.5 * (v + v1), h)
            P = F @ P @ F.transpose(1, 2) + _process_noise(cfg.accel_noise, h)
            t = t + h
            if paddle is not None:
                pad_next = paddle.at(t, _PADDLE_KEYS)
                p1, v1, P = self._bounce(p, v, p1, v1, P, h, pad, pad_next)
                pad = pad_next
            x = torch.cat([p1, v1], dim=1)
        return x, P

    def _bounce(self, p, v, p1, v1, P, h, a, b):
        """Bounce within a step of length h from (p, v) to (p1, v1), paddle states a and b at its ends."""
        cfg = self.cfg
        s0 = ((p - a["center"]) * a["normal"]).sum(-1) - BALL_RADIUS
        s1 = ((p1 - b["center"]) * b["normal"]).sum(-1) - BALL_RADIUS
        f = (s0 / (s0 - s1).clamp_min(1e-9)).clamp(0, 1)[:, None]
        pc, vc = p + f * (p1 - p), v + f * (v1 - v)
        c = a["center"] + f * (b["center"] - a["center"])
        nrm = a["normal"] + f * (b["normal"] - a["normal"])
        nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        vp = a["vel"] + f * (b["vel"] - a["vel"])
        d = pc - c
        lateral = (d - (d * nrm).sum(-1, keepdim=True) * nrm).norm(dim=-1)
        rel = vc - vp
        vn = (rel * nrm).sum(-1, keepdim=True)
        vt = rel - vn * nrm
        # Crossing the face, or already touching it (a resting ball sits ~1 mm in), while closing in on it.
        hit = (s0 > -0.015) & (s1 < 0) & (vn[:, 0] < 0) & (lateral < cfg.blade_radius) & (h > 0)
        e, kt = cfg.restitution, cfg.tangential_keep
        vout = vp - e * vn * nrm + kt * vt
        rem = (1 - f) * h[:, None]
        pb, vb = flight(pc, vout, rem[:, 0])
        # Velocity map v -> v_out about the normal; restitution, friction and paddle-speed uncertainty.
        nn = nrm[:, :, None] * nrm[:, None, :]
        eye = torch.eye(3, device=p.device)
        M = -e * nn + kt * (eye - nn)
        J = torch.eye(6, device=p.device).repeat(p.shape[0], 1, 1)
        J[:, 3:, 3:] = M
        vt2 = (vt * vt).sum(-1)[:, None, None]
        Qv = ((cfg.restitution_std * vn[:, :, None]) ** 2) * nn + (cfg.tangential_std**2) * vt2 * (eye - nn) \
            + cfg.paddle_speed_std**2 * eye
        Pb = J @ P @ J.transpose(1, 2)
        Pb[:, 3:, 3:] += Qv
        hk = hit[:, None]
        return torch.where(hk, pb, p1), torch.where(hk, vb, v1), torch.where(hit[:, None, None], Pb, P)

    # -- measurements ------------------------------------------------------------------------------------------------

    def update(self, t_meas, cam_R, cam_p, z, sigma, valid, paddle: TickHistory | None):
        """Fuse one detection per world: capture time (B,), camera world-from-camera rotation (B, 3, 3; OpenCV
        axes) and centre (B, 3), measurement [x/z, y/z, alpha] (B, 3), its standard deviations (B, 3), and which
        worlds have a detection (B,)."""
        cfg = self.cfg
        if not bool(valid.any()):
            return
        upd = valid & self.tracked
        dt = t_meas - self.t
        fwd = dt >= 0
        # In-order detections: move the state to the capture time.
        t_to = torch.where(upd & fwd, t_meas, self.t)
        x_f, P_f = self.propagate(self.x, self.P, self.t, t_to, paddle)
        # Out-of-order ones: predict the measurement at the (earlier) capture time from the current state.
        dt_b = torch.where(upd & ~fwd, dt, torch.zeros_like(dt))
        pb, vb = flight(x_f[:, :3], x_f[:, 3:], dt_b)
        Phi = _transition(x_f[:, 3:], dt_b)
        zh, Hp, front = project(pb, cam_R, cam_p)
        H = torch.cat([Hp, torch.zeros_like(Hp)], dim=2) @ Phi  # (B, 3, 6)
        # A timestamp error dt shifts the measurement by H v dt (ball motion; the wrist camera's own motion is
        # left out), so timestamp uncertainty is measurement noise along the image motion.
        Hv = (Hp @ vb[:, :, None])
        Rm = torch.diag_embed(sigma * sigma) + cfg.stamp_std**2 * (Hv @ Hv.transpose(1, 2))
        S = H @ P_f @ H.transpose(1, 2) + Rm
        innov = (z - zh)[:, :, None]
        S_inv = torch.linalg.inv(S)
        nis = (innov.transpose(1, 2) @ S_inv @ innov)[:, 0, 0]
        ok = upd & front & (nis < cfg.gate_chi2)
        K = P_f @ H.transpose(1, 2) @ S_inv
        x_u = x_f + (K @ innov)[:, :, 0]
        IKH = torch.eye(6, device=self.dev) - K @ H
        P_u = IKH @ P_f @ IKH.transpose(1, 2) + K @ Rm @ K.transpose(1, 2)  # Joseph form
        okk = ok[:, None]
        self.x = torch.where(okk, x_u, x_f)
        self.P = torch.where(ok[:, None, None], P_u, P_f)
        self.t = t_to
        self.last_meas = torch.where(ok, torch.maximum(self.last_meas, t_meas), self.last_meas)
        failed = upd & ~ok
        self.rejects = torch.where(ok, torch.zeros_like(self.rejects), self.rejects + failed.long())
        self.accepted += ok.long()
        self.rejected += failed.long()
        # (Re)start tracks.
        start = valid & (~self.tracked | (failed & (self.rejects >= cfg.reinit_after)))
        if start.any():
            x0, P0 = self._init(cam_R, cam_p, z, sigma)
            self.x = torch.where(start[:, None], x0, self.x)
            self.P = torch.where(start[:, None, None], P0, self.P)
            self.t = torch.where(start, t_meas, self.t)
            self.last_meas = torch.where(start, t_meas, self.last_meas)
            self.rejects = torch.where(start, torch.zeros_like(self.rejects), self.rejects)
            self.tracked |= start

    def _init(self, cam_R, cam_p, z, sigma):
        cfg = self.cfg
        u = torch.cat([z[:, :2], torch.ones_like(z[:, :1])], dim=1)
        u = u / u.norm(dim=-1, keepdim=True)
        alpha = z[:, 2].clamp(1e-3, 3.0)
        rng = BALL_RADIUS / torch.sin(alpha / 2)
        p = cam_p + (cam_R @ (rng[:, None] * u)[:, :, None])[:, :, 0]
        s_lat = rng * sigma[:, :2].max(dim=1).values
        s_rng = rng * sigma[:, 2] / alpha
        uu = u[:, :, None] * u[:, None, :]
        Cc = s_lat[:, None, None] ** 2 * (torch.eye(3, device=self.dev) - uu) + s_rng[:, None, None] ** 2 * uu
        P = torch.zeros(self.n, 6, 6, device=self.dev)
        P[:, :3, :3] = cam_R @ Cc @ cam_R.transpose(1, 2)
        P[:, 3:, 3:] = cfg.init_vel_std**2 * torch.eye(3, device=self.dev)
        return torch.cat([p, torch.zeros_like(p)], dim=1), P

    # -- per control step --------------------------------------------------------------------------------------------

    def estimate(self, now: float, paddle: TickHistory | None):
        """Advance stale states to `now - hold`, drop lost tracks, and predict to `now`. Returns position (B, 3),
        velocity (B, 3), position and velocity standard deviations (B,; RMS over axes), time since the last
        accepted detection (B,), and the tracked flag (B,). Untracked worlds report zeros and age = inf."""
        cfg = self.cfg
        self.tracked &= (now - self.last_meas) < cfg.lost_after
        t_hold = torch.where(self.tracked, torch.clamp_min(self.t, now - cfg.hold), self.t)
        self.x, self.P = self.propagate(self.x, self.P, self.t, t_hold, paddle)
        self.t = t_hold
        t_now = torch.where(self.tracked, torch.full_like(self.t, now), self.t)
        x, P = self.propagate(self.x, self.P, self.t, t_now, paddle)
        trk = self.tracked[:, None]
        p = torch.where(trk, x[:, :3], torch.zeros_like(x[:, :3]))
        v = torch.where(trk, x[:, 3:], torch.zeros_like(x[:, 3:]))
        d = torch.diagonal(P, dim1=1, dim2=2)
        p_std = torch.where(self.tracked, (d[:, :3].mean(-1)).sqrt(), torch.zeros_like(self.t))
        v_std = torch.where(self.tracked, (d[:, 3:].mean(-1)).sqrt(), torch.zeros_like(self.t))
        age = torch.where(self.tracked, now - self.last_meas, torch.full_like(self.t, math.inf))
        return p, v, p_std, v_std, age, self.tracked.clone()
