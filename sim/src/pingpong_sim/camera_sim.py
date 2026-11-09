"""Simulated ball cameras: the detections the deployment pipeline will hand to `BallFilter`, without rendering.

Rig (2026-11-09 plan): two InnoMaker U20CAM-720P modules (the 32x32 UVC camera of the SO-ARM100 repo): 1280x720
MJPEG at 30 fps, rolling shutter, 120 deg diagonal / 102 deg horizontal lens with strong barrel distortion
(modelled as equidistant, r = f theta, 719 px/rad).
- `wrist`: on the paddle holder (robot.WRIST_CAM_POS), looking along the paddle at the blade centre from ~0.2 m.
- `overhead`: the SO-ARM100 overhead mount (0.55 m post, camera plate tilted 25 deg from straight down), standing
  24 in (0.61 m) in front of the arm base, looking back and down at the bounce.

Per frame, per world:
1. Timing. Frames start every 1/fps from a random phase. The ball is imaged at mid-exposure of the row it lands on
   (rolling shutter: rows are read over `readout`); the detection arrives `latency` after the readout ends.
2. Truth at the capture time: the ball integrated from the neighbouring control ticks (forward from the earlier
   one, backward from the later one; across a bounce the branch above the blade is right), and the camera pose
   (wrist: interpolated sim pose, plus a per-world mount error).
3. Visibility: inside the image, beyond `min_range`, not behind the blade (ray test against the face disc), large
   enough; then random dropouts. False detections: random pixel and size.
4. Measurement: pixel centre and diameter with noise (plus centroid noise from motion blur over the exposure),
   undistorted with the believed focal length into [x/z, y/z, alpha] for the filter.
5. Believed pose and time: the overhead camera's calibrated extrinsics (true pose + calibration error); the wrist
   camera's pose from the encoder angles at the believed capture time (forward kinematics, CAD mount). Timestamps
   carry a per-world bias and per-frame jitter.

Per-world parameters are drawn at `reset` from the ranges in `CameraSimCfg`.

Findings (scripted bouncer, scripts/run_bouncer.py --cameras, 2026-11-09): systematic errors dominate, not noise,
frame rate or latency. Hence the defaults: image long side along the overhead tilt and along the blade normal
(`long_axis="y"`; the ball stays in view ~80 % / ~55 % of frames vs ~45 % / ~25 %), and calibration residuals at
1.5x the deployment targets (overhead extrinsics 3 mm / 0.3 deg, wrist mount 1 mm / 0.5 deg, timestamps +-2 ms).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from pingpong_sim.ball import BALL_RADIUS
from pingpong_sim.ball_filter import TickHistory, flight, orthonormalize
from pingpong_sim.kinematics import ArmKinematics

TILT = math.radians(25.0)  # overhead camera plate, from straight down (Overhead_Cam_Mount_32x32 cam_mount_top.stl)


@dataclass
class CameraSimCfg:
    name: str
    mount: str  # "world" or "wrist"
    # World mount: nominal optical centre, optical axis and image +x direction (orthogonalized against the axis).
    pos: tuple[float, float, float] = (0.58, 0.0, 0.55)
    axis: tuple[float, float, float] = (-math.sin(TILT), 0.0, -math.cos(TILT))
    right: tuple[float, float, float] = (0.0, 1.0, 0.0)
    long_axis: str = "y"  # image axis along the sensor's long side: "x" (right) or "y" (down)
    width_px: int = 1280
    height_px: int = 720
    px_per_rad: float = 719.0
    # True placement around nominal (world mount) or mount error (wrist), uniform per axis / in angle.
    pos_range: tuple[float, float, float] = (0.05, 0.05, 0.03)
    rot_range_deg: float = 5.0
    # Calibration error of the believed extrinsics (world mount; the wrist uses the CAD mount + its mount error).
    calib_pos: float = 0.005
    calib_rot_deg: float = 0.5
    focal_error: float = 0.01  # relative
    fps: tuple[float, float] = (30.0, 30.0)
    exposure_ms: tuple[float, float] = (1.0, 8.0)
    readout_ms: tuple[float, float] = (15.0, 30.0)
    latency_ms: tuple[float, float] = (15.0, 60.0)  # readout end -> detection available
    stamp_bias_ms: tuple[float, float] = (-3.0, 3.0)  # per-camera timestamp calibration residual
    stamp_jitter_ms: float = 1.0
    pixel_noise: tuple[float, float] = (0.3, 1.5)  # px, centroid
    diameter_noise: tuple[float, float] = (0.5, 2.0)  # px
    blur_noise: float = 0.1  # px of centroid noise per px of blur
    assumed_pixel_noise: float = 1.0  # what the filter is told
    assumed_diameter_noise: float = 1.5
    assumed_angle_floor: float = 0.004  # rad (0.23 deg): extrinsic calibration residual, as noise
    assumed_size_floor: float = 0.02  # relative apparent-size error (focal length, blob edge)
    dropout: tuple[float, float] = (0.0, 0.1)
    false_rate: tuple[float, float] = (0.0, 0.02)
    min_range: float = 0.08  # m
    min_diameter_px: float = 4.0


OVERHEAD = CameraSimCfg(name="overhead", mount="world")
WRIST = CameraSimCfg(name="wrist", mount="wrist", pos_range=(0.0015, 0.0015, 0.0015), rot_range_deg=0.75,
                     calib_pos=0.0, calib_rot_deg=0.0)


def ideal(cfg: CameraSimCfg, latency_ms: float = 30.0) -> CameraSimCfg:
    """`cfg` without randomization or noise: nominal pose, exact calibration and timestamps, fixed timing."""
    return replace(cfg, pos_range=(0.0, 0.0, 0.0), rot_range_deg=0.0, calib_pos=0.0, calib_rot_deg=0.0,
                   focal_error=0.0, exposure_ms=(1.0, 1.0), readout_ms=(20.0, 20.0), latency_ms=(latency_ms, latency_ms),
                   stamp_bias_ms=(0.0, 0.0), stamp_jitter_ms=0.0, pixel_noise=(1e-3, 1e-3), diameter_noise=(1e-3, 1e-3),
                   blur_noise=0.0, dropout=(0.0, 0.0), false_rate=(0.0, 0.0))


def _uniform(lo, hi, n, dev):
    return lo + (hi - lo) * torch.rand(n, device=dev)


def _small_rotation(n: int, max_deg: float, dev) -> torch.Tensor:
    """Random rotations (n, 3, 3): uniform axis, angle uniform in [0, max_deg]."""
    axis = torch.randn(n, 3, device=dev)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    ang = torch.rand(n, device=dev) * math.radians(max_deg)
    K = torch.zeros(n, 3, 3, device=dev)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -axis[:, 2], axis[:, 1], axis[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -axis[:, 0], -axis[:, 1], axis[:, 0]
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return torch.eye(3, device=dev) + s * K + (1 - c) * (K @ K)


def _box(n, rng, dev):
    return (2 * torch.rand(n, 3, device=dev) - 1) * torch.tensor(rng, device=dev)


class CameraRig:
    """Camera simulation for `num_envs` worlds, ticking every `dt`. Call `record` once per control tick (after
    physics), then `detect`."""

    def __init__(self, cams: list[CameraSimCfg], num_envs: int, dt: float, kin: ArmKinematics, device,
                 blade_radius: float = 0.075, truth_ticks: int = 16, arm_ticks: int = 32):
        self.cams, self.n, self.dt, self.kin, self.dev = cams, num_envs, dt, kin, device
        self.blade_radius = blade_radius
        self.truth = TickHistory(num_envs, truth_ticks, dt, {"p": (3,), "v": (3,), "center": (3,), "normal": (3,),
                                                             "wR": (3, 3), "wp": (3,)}, device)
        # What the deployment computes from the encoders: the filter's paddle history and the wrist camera pose.
        self.arm = TickHistory(num_envs, arm_ticks, dt, {"center": (3,), "normal": (3,), "vel": (3,),
                                                        "wR": (3, 3), "wp": (3,)}, device)
        self.t_reset = torch.zeros(num_envs, device=device)
        self.params: list[dict[str, torch.Tensor]] = [dict() for _ in cams]
        self._prev_face: torch.Tensor | None = None
        self.reset(torch.arange(num_envs, device=device), now=0.0)

    def reset(self, env_ids: torch.Tensor, now: float) -> None:
        """Draw new camera parameters for env_ids; detections captured before `now` are discarded for them."""
        k, dev = len(env_ids), self.dev
        self.t_reset[env_ids] = now
        for c, cfg in enumerate(self.cams):
            P = self.params[c]
            put = lambda name, val: P.setdefault(name, torch.zeros(self.n, *val.shape[1:], device=dev)).__setitem__(  # noqa: E731
                env_ids, val)
            fps = _uniform(*cfg.fps, k, dev)
            put("period", 1.0 / fps)
            put("phase", torch.rand(k, device=dev) / fps)
            for name, rng in (("exposure", cfg.exposure_ms), ("readout", cfg.readout_ms), ("latency", cfg.latency_ms),
                              ("stamp_bias", cfg.stamp_bias_ms)):
                put(name, _uniform(*rng, k, dev) / 1e3)
            for name, rng in (("pixel_noise", cfg.pixel_noise), ("diameter_noise", cfg.diameter_noise),
                              ("dropout", cfg.dropout), ("false_rate", cfg.false_rate)):
                put(name, _uniform(*rng, k, dev))
            put("focal", cfg.px_per_rad * torch.ones(k, device=dev))
            put("focal_believed", cfg.px_per_rad * (1 + cfg.focal_error * (2 * torch.rand(k, device=dev) - 1)))
            dR, dp = _small_rotation(k, cfg.rot_range_deg, dev), _box(k, cfg.pos_range, dev)
            if cfg.mount == "world":
                z = torch.tensor(cfg.axis, device=dev)
                z = z / z.norm()
                x = torch.tensor(cfg.right, device=dev)
                x = x - (x @ z) * z
                x = x / x.norm()
                R0 = torch.stack([x, torch.cross(z, x, dim=0), z], dim=1)
                R = dR @ R0
                p = torch.tensor(cfg.pos, device=dev) + dp
                put("R", R)
                put("p", p)
                put("R_believed", _small_rotation(k, cfg.calib_rot_deg, dev) @ R)
                put("p_believed", p + _box(k, (cfg.calib_pos,) * 3, dev))
            else:  # mount error in the camera frame
                put("mount_R", dR)
                put("mount_p", dp)

    def record(self, now: float, ball_p, ball_v, face, normal, wrist_R, wrist_p, q_enc) -> None:
        """True ball state, true paddle face (centre, normal) and wrist camera pose (OpenCV axes) after this tick's
        physics, and the encoder angles (B, 5)."""
        self.truth.push(now, p=ball_p, v=ball_v, center=face, normal=normal, wR=wrist_R, wp=wrist_p)
        f, nrm, _, _ = self.kin.fk(q_enc)
        wR, wp = self.kin.wrist_camera(q_enc)
        vel = torch.zeros_like(f) if self._prev_face is None else (f - self._prev_face) / self.dt
        self._prev_face = f
        self.arm.push(now, center=f, normal=nrm, vel=vel, wR=wR, wp=wp)

    # -- truth at arbitrary times ------------------------------------------------------------------------------------

    def _ball_at(self, t: torch.Tensor):
        a, b, tau = self.truth.samples(t, ("p", "v", "center", "normal"))
        pf, vf = flight(a["p"], a["v"], tau)
        pb, vb = flight(b["p"], b["v"], tau - self.dt)
        w = (tau / self.dt).clamp(0, 1)[:, None]
        c = a["center"] + w * (b["center"] - a["center"])
        nrm = a["normal"] + w * (b["normal"] - a["normal"])
        sf, sb = ((pf - c) * nrm).sum(-1), ((pb - c) * nrm).sum(-1)
        use_b = ((sb > sf) & ((pf - pb).norm(dim=-1) > 1e-3))[:, None]
        return torch.where(use_b, pb, pf), torch.where(use_b, vb, vf), c, nrm

    def _camera_at(self, c: int, t: torch.Tensor):
        P = self.params[c]
        if self.cams[c].mount == "world":
            return P["R"], P["p"]
        s = self.truth.at(t, ("wR", "wp"))
        Rw = orthonormalize(s["wR"])
        return Rw @ P["mount_R"], s["wp"] + (Rw @ P["mount_p"][:, :, None])[:, :, 0]

    def _pixels(self, c: int, p: torch.Tensor, R: torch.Tensor, cp: torch.Tensor):
        """Equidistant projection: pixel offsets from the image centre along camera x, y (B, 2), angle from the axis,
        range."""
        pc = ((p - cp)[:, None, :] @ R)[:, 0]
        rxy = pc[:, :2].norm(dim=-1)
        theta = torch.atan2(rxy, pc[:, 2])
        scale = self.params[c]["focal"] * torch.where(rxy > 1e-9, theta / rxy.clamp_min(1e-9), 1 / pc[:, 2].clamp_min(1e-6))
        return pc[:, :2] * scale[:, None], theta, pc.norm(dim=-1)

    # -- detections --------------------------------------------------------------------------------------------------

    def detect(self, now: float) -> dict[str, torch.Tensor]:
        """Detections arriving in (now - dt, now], at most one per camera per world, stacked (C, B, ...) and sorted
        by timestamp per world (invalid last). Keys: t, R, p, z, sigma, valid. Per-camera diagnostics (unsorted,
        (C, B)) go to `self.diag`: arrived (a frame came), visible (ball in view and unoccluded, before dropouts),
        valid, false (false detection), err_px."""
        outs = [self._detect_cam(c, now) for c in range(len(self.cams))]
        st = {k: torch.stack([o[k] for o in outs]) for k in outs[0]}
        self.diag = {k: st.pop(k) for k in ("arrived", "visible", "in_image", "occluded", "false", "err_px")}
        self.diag["valid"] = st["valid"]
        key = torch.where(st["valid"], st["t"], torch.full_like(st["t"], math.inf))
        order = key.argsort(dim=0)
        return {k: torch.gather(v, 0, order.view(*order.shape, *([1] * (v.dim() - 2))).expand_as(v))
                for k, v in st.items()}

    def _detect_cam(self, c: int, now: float) -> dict[str, torch.Tensor]:
        cfg, P, n, dev = self.cams[c], self.params[c], self.n, self.dev
        W, H = cfg.width_px, cfg.height_px
        half = torch.tensor([W / 2, H / 2] if cfg.long_axis == "x" else [H / 2, W / 2], device=dev)
        row_axis = 1 if cfg.long_axis == "x" else 0  # rolling-shutter rows run along the short side
        lag = P["exposure"] + P["readout"] + P["latency"]
        frame = torch.floor((now - lag - P["phase"]) / P["period"])
        t_start = P["phase"] + frame * P["period"]
        arrived = (t_start + lag) > now - self.dt + 1e-9
        # Capture time of the ball's row (fixed point; the row barely moves within a frame).
        tc = t_start + P["exposure"] / 2 + P["readout"] / 2
        for _ in range(2):
            p, _, _, _ = self._ball_at(tc)
            R, cp = self._camera_at(c, tc)
            uv, _, _ = self._pixels(c, p, R, cp)
            row = ((uv[:, row_axis] + H / 2) / H).clamp(0, 1)
            tc = t_start + P["exposure"] / 2 + P["readout"] * row
        p, v, center, nrm = self._ball_at(tc)
        R, cp = self._camera_at(c, tc)
        uv, theta, rng = self._pixels(c, p, R, cp)
        diam = P["focal"] * 2 * torch.asin((BALL_RADIUS / rng.clamp_min(BALL_RADIUS * 1.01)).clamp(max=1.0))
        # Visibility.
        in_image = (theta < math.pi / 2) & (uv.abs() < half).all(-1)
        d = p - cp
        denom = (d * nrm).sum(-1)
        s = ((center - cp) * nrm).sum(-1) / torch.where(denom.abs() > 1e-9, denom, torch.full_like(denom, 1e-9))
        hit = cp + s[:, None] * d
        occluded = (s > 0) & (s < 1) & ((hit - center).norm(dim=-1) < self.blade_radius)
        visible = arrived & in_image & (rng > cfg.min_range) & ~occluded & (diam > cfg.min_diameter_px)
        visible &= tc >= self.t_reset
        # Motion blur over the exposure (ball and camera motion), from the image velocity.
        dt_b = torch.full_like(tc, 1e-3)
        p2, _, _, _ = self._ball_at(tc + dt_b)
        R2, cp2 = self._camera_at(c, tc + dt_b)
        uv2, _, _ = self._pixels(c, p2, R2, cp2)
        blur = (uv2 - uv).norm(dim=-1) / 1e-3 * P["exposure"]
        sig = torch.sqrt(P["pixel_noise"] ** 2 + (cfg.blur_noise * blur) ** 2)
        uv_m = uv + sig[:, None] * torch.randn(n, 2, device=dev)
        diam_m = diam + P["diameter_noise"] * torch.randn(n, device=dev)
        valid = visible & (torch.rand(n, device=dev) >= P["dropout"])
        false = arrived & (torch.rand(n, device=dev) < P["false_rate"]) & (tc >= self.t_reset)
        uv_m = torch.where(false[:, None], (2 * torch.rand(n, 2, device=dev) - 1) * half, uv_m)
        diam_m = torch.where(false, _uniform(5.0, 80.0, n, dev), diam_m)
        valid |= false
        # Undistort with the believed focal length -> normalized coordinates and angular diameter.
        fb = P["focal_believed"]
        r_px = uv_m.norm(dim=-1).clamp_min(1e-9)
        th = (r_px / fb).clamp(max=1.45)
        xy = uv_m / r_px[:, None] * torch.tan(th)[:, None]
        z = torch.cat([xy, (diam_m.clamp_min(1.0) / fb)[:, None]], dim=1)
        sec2 = 1 / torch.cos(th) ** 2
        s_ang = torch.sqrt((cfg.assumed_pixel_noise / fb) ** 2 + cfg.assumed_angle_floor**2) * sec2
        s_size = torch.sqrt((cfg.assumed_diameter_noise / fb) ** 2 + (cfg.assumed_size_floor * z[:, 2]) ** 2)
        sigma = torch.stack([s_ang, s_ang, s_size], dim=1)
        # Believed timestamp and camera pose.
        t_stamp = tc + P["stamp_bias"] + cfg.stamp_jitter_ms / 1e3 * torch.randn(n, device=dev)
        if cfg.mount == "world":
            Rb, pb = P["R_believed"], P["p_believed"]
        else:
            s_arm = self.arm.at(t_stamp, ("wR", "wp"))
            Rb, pb = orthonormalize(s_arm["wR"]), s_arm["wp"]
        return {"t": t_stamp, "R": Rb, "p": pb, "z": z, "sigma": sigma, "valid": valid, "arrived": arrived,
                "visible": visible, "in_image": arrived & in_image, "occluded": arrived & occluded,
                "false": false & arrived, "err_px": (uv_m - uv).norm(dim=-1)}
