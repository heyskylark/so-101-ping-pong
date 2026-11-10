"""Real cameras as the ball filter sees them, and the calibration files that describe them.

`CameraCalib` (measurements/calibration/camera_<name>.json): lens (OpenCV fisheye model: K, D for the frames as
captured), pose (world mount: world-from-camera; wrist mount: paddle-link-from-camera, OpenCV axes), timestamps
(capture time of a row = buffer timestamp + `stamp_offset` + `readout` * row / height) and the noise the filter
assumes (camera_sim.CameraSimCfg defaults). `ArmCalib` (arm.json): joint zero offsets and the paddle tag's pose
on the paddle link, from the overhead calibration.

A pixel detection (centre u, v and diameter d in px) becomes the filter's measurement [x/z, y/z, alpha]: the
centre undistorted to the normalized image plane; alpha, the ball's angular diameter, from d and the lens's local
scale at that angle (radial f dθd/dθ and tangential f θd / sin θ: the blob's area goes with their product).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from pingpong_sim.ball_filter import TickHistory, measurement_sigma, orthonormalize
from pingpong_sim.camera_sim import OVERHEAD, WRIST, CameraSimCfg
from pingpong_sim.kinematics import ArmKinematics

REPO = Path(__file__).resolve().parents[4]
CALIB_DIR = REPO / "measurements" / "calibration"


def _noise(cfg: CameraSimCfg) -> dict:
    return {"pixel": cfg.assumed_pixel_noise, "diameter": cfg.assumed_diameter_noise,
            "angle_floor": cfg.assumed_angle_floor, "size_floor": cfg.assumed_size_floor}


@dataclass
class CameraCalib:
    name: str
    mount: str  # "world" or "wrist"
    width: int
    height: int
    K: list  # 3x3
    D: list  # 4, OpenCV fisheye (theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8))
    R: list | None = None  # 3x3, world-from-camera (world mount) or paddle-link-from-camera (wrist mount)
    p: list | None = None  # 3
    stamp_offset: float = 0.0  # s, row 0 capture time (mid-exposure) - buffer timestamp
    readout: float = 0.0  # s, rolling shutter: last row is captured `readout` after row 0
    noise: dict = field(default_factory=lambda: _noise(OVERHEAD))
    steps: dict = field(default_factory=dict)  # calibration step -> {date, residuals, ...}

    @staticmethod
    def path(name: str) -> Path:
        return CALIB_DIR / f"camera_{name}.json"

    @classmethod
    def load(cls, name: str) -> CameraCalib:
        return cls(**json.loads(cls.path(name).read_text()))

    def save(self) -> Path:
        path = self.path(self.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    @property
    def focal(self) -> float:
        return math.sqrt(self.K[0][0] * self.K[1][1])

    @property
    def Km(self) -> np.ndarray:
        return np.array(self.K, float)

    @property
    def Dm(self) -> np.ndarray:
        return np.array(self.D, float).reshape(4, 1)

    # -- geometry
    def undistort(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) -> normalized image coordinates (N, 2) = (x/z, y/z) in the camera frame."""
        pts = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        return cv2.fisheye.undistortPoints(pts, self.Km, self.Dm).reshape(-1, 2)

    def project(self, xyz_cam: np.ndarray) -> np.ndarray:
        """Camera-frame points (N, 3) -> pixels (N, 2)."""
        pts = np.asarray(xyz_cam, np.float64).reshape(-1, 1, 3)
        uv, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), self.Km, self.Dm)
        return uv.reshape(-1, 2)

    def lens_scales(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Radial and tangential image scale (px/rad) at angles theta from the axis."""
        k = np.array(self.D, float).ravel()
        t2 = theta * theta
        td = theta * (1 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4)
        dtd = 1 + 3 * k[0] * t2 + 5 * k[1] * t2**2 + 7 * k[2] * t2**3 + 9 * k[3] * t2**4
        f = self.focal
        tang = np.where(theta > 1e-6, td / np.sin(np.maximum(theta, 1e-6)), 1.0)
        return f * dtd, f * tang

    def measure(self, uv: np.ndarray, d_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel detections (N, 2) and diameters (N,) -> measurements [x/z, y/z, alpha] (N, 3) and their assumed
        standard deviations (N, 3)."""
        xy = self.undistort(uv)
        theta = np.arctan(np.linalg.norm(xy, axis=1))
        s_r, s_t = self.lens_scales(theta)
        alpha = np.asarray(d_px, float) / np.sqrt(s_r * s_t)
        z = np.concatenate([xy, alpha[:, None]], axis=1)
        n = self.noise
        sigma = measurement_sigma(torch.as_tensor(np.minimum(theta, 1.45)), torch.as_tensor(alpha), self.focal,
                                  n["pixel"], n["diameter"], n["angle_floor"], n["size_floor"]).numpy()
        return z, sigma

    def capture_time(self, stamp: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(stamp, float) + self.stamp_offset + self.readout * np.asarray(v, float) / self.height


def nominal(name: str, width: int = 1280, height: int = 720) -> CameraCalib:
    """Uncalibrated guess: the sim's lens (equidistant, 719 px/rad) and CAD wrist mount; no world pose (the
    overhead camera needs its calibration)."""
    cfg = WRIST if name == "wrist" else OVERHEAD
    f = cfg.px_per_rad
    return CameraCalib(name=name, mount=cfg.mount, width=width, height=height,
                       K=[[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], D=[0.0, 0.0, 0.0, 0.0],
                       stamp_offset=-0.030, noise=_noise(cfg))


@dataclass
class ArmCalib:
    joint_offsets: list = field(default_factory=lambda: [0.0] * 5)  # rad, true = mapped encoder angle + offset
    tag_R: list | None = None  # paddle-link-from-tag rotation
    tag_p: list | None = None
    steps: dict = field(default_factory=dict)

    @staticmethod
    def path() -> Path:
        return CALIB_DIR / "arm.json"

    @classmethod
    def load(cls) -> ArmCalib:
        p = cls.path()
        return cls(**json.loads(p.read_text())) if p.exists() else cls()

    def save(self) -> Path:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return p


def apply_wrist_mount(kin: ArmKinematics, calib: CameraCalib) -> None:
    """Use the calibrated wrist camera mount (paddle-link-from-camera) in the kinematics (and so the controller's
    arm history, which the wrist camera's pose at capture time comes from)."""
    if calib.mount != "wrist" or calib.R is None:
        return
    kin.cam_R = torch.tensor(calib.R, dtype=kin.cam_R.dtype, device=kin.cam_R.device)
    kin.cam_p = torch.tensor(calib.p, dtype=kin.cam_p.dtype, device=kin.cam_p.device)


@dataclass
class PixelDetection:
    camera: str
    stamp: float  # s, buffer timestamp (bus clock)
    u: float
    v: float
    d: float  # diameter, px


def to_measurements(dets: list[PixelDetection], calibs: dict[str, CameraCalib], arm: TickHistory, t0: float = 0.0,
                    device="cpu") -> dict[str, torch.Tensor]:
    """Pixel detections from any cameras -> the controller's detection batch (N, 1, ...) sorted by capture time:
    t, R, p, z, sigma, valid. Times are relative to `t0` (the controller's clock: the filter keeps float32 times,
    which can't hold a raw monotonic clock to the millisecond). The wrist camera's pose at the capture time comes
    from the encoder arm history."""
    if not dets:
        e = torch.zeros(0, 1, device=device)
        return {"t": e, "R": torch.zeros(0, 1, 3, 3, device=device), "p": torch.zeros(0, 1, 3, device=device),
                "z": torch.zeros(0, 1, 3, device=device), "sigma": torch.zeros(0, 1, 3, device=device),
                "valid": torch.zeros(0, 1, dtype=torch.bool, device=device)}
    rows = []
    for d in dets:
        c = calibs[d.camera]
        t = float(c.capture_time(d.stamp, d.v)) - t0
        z, s = c.measure(np.array([[d.u, d.v]]), np.array([d.d]))
        rows.append((t, c, z[0], s[0]))
    rows.sort(key=lambda r: r[0])
    t = torch.tensor([[r[0]] for r in rows], dtype=torch.float32, device=device)
    R = torch.zeros(len(rows), 1, 3, 3, device=device)
    p = torch.zeros(len(rows), 1, 3, device=device)
    for i, (ti, c, _, _) in enumerate(rows):
        if c.mount == "world":
            R[i, 0] = torch.tensor(c.R, device=device)
            p[i, 0] = torch.tensor(c.p, device=device)
        else:
            s = arm.at(torch.tensor([ti], device=device), ("wR", "wp"))
            R[i, 0] = orthonormalize(s["wR"])[0]
            p[i, 0] = s["wp"][0]
    z = torch.tensor(np.array([r[2] for r in rows]), dtype=torch.float32, device=device)[:, None]
    sig = torch.tensor(np.array([r[3] for r in rows]), dtype=torch.float32, device=device)[:, None]
    return {"t": t, "R": R, "p": p, "z": z, "sigma": sig,
            "valid": torch.ones(len(rows), 1, dtype=torch.bool, device=device)}
