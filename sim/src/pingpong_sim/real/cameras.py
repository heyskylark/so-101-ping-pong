"""UVC cameras (InnoMaker U20CAM-720P) and ball detection.

`RigConfig` (measurements/calibration/rig.json): which device is which camera, exposure, ball colour. Devices are
best given as /dev/v4l/by-path/... (both cameras are the same model, so by-id names collide).

`UvcCamera` captures MJPEG 1280x720 at 30 fps in its own thread with manual exposure and white balance (set
through v4l2-ctl), and runs a per-frame `process(frame) -> result or None` there; results come out of `drain()` as
`(stamp, result)`. The stamp is the V4L2 buffer timestamp (uvcvideo: CLOCK_MONOTONIC when the frame's first USB
packet arrived), the same clock as time.perf_counter / the bus; if the driver gives none, the time the frame was
handed to us. The calibrated `stamp_offset` (camera_model) turns it into the capture time.

`BallDetector`: colour threshold in HSV, opening, then the roundest, fullest blob of plausible size: centre from
its moments (sub-pixel), diameter from its area.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pingpong_sim.real.camera_model import CALIB_DIR, PixelDetection

RIG = CALIB_DIR / "rig.json"


@dataclass
class CameraConfig:
    device: str  # /dev/v4l/by-path/... or /dev/videoN
    exposure_ms: float = 2.0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class BallColor:
    """HSV window (OpenCV: H 0-179, S and V 0-255)."""

    h: tuple[int, int] = (3, 25)  # orange
    s: tuple[int, int] = (110, 255)
    v: tuple[int, int] = (90, 255)


WHITE = BallColor(h=(0, 179), s=(0, 50), v=(190, 255))


@dataclass
class RigConfig:
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    ball: BallColor = field(default_factory=BallColor)

    @classmethod
    def load(cls) -> RigConfig:
        if not RIG.exists():
            return cls()
        d = json.loads(RIG.read_text())
        b = d.get("ball", {})
        return cls(cameras={k: CameraConfig(**v) for k, v in d.get("cameras", {}).items()},
                   ball=BallColor(**{k: tuple(v) for k, v in b.items()}))

    def save(self) -> None:
        RIG.parent.mkdir(parents=True, exist_ok=True)
        RIG.write_text(json.dumps(asdict(self), indent=2) + "\n")


def v4l2_devices() -> list[tuple[str, str]]:
    """(path, description) of the capture devices: by-path names when available."""
    from pathlib import Path

    out = []
    by_path = Path("/dev/v4l/by-path")
    links = sorted(by_path.glob("*video-index0")) if by_path.exists() else sorted(Path("/dev").glob("video*"))
    for p in links:
        try:
            info = subprocess.run(["v4l2-ctl", "-d", str(p), "--info"], capture_output=True, text=True, timeout=3).stdout
            card = next((ln.split(":", 1)[1].strip() for ln in info.splitlines() if "Card type" in ln), "?")
        except (OSError, subprocess.TimeoutExpired):
            card = "? (v4l2-ctl missing)"
        out.append((str(p), card))
    return out


def set_controls(device: str, exposure_ms: float) -> list[str]:
    """Manual exposure (units of 100 us) and fixed white balance; returns problems (control names vary by kernel).
    The U20CAM can refuse exposure writes ("Permission denied") while reporting manual mode; switching to auto
    and back clears that (seen after replugging, 2026-11-12). A refused write is a problem only if the camera
    then still reads another value."""
    def ctl(arg: str, value: str) -> subprocess.CompletedProcess:
        return subprocess.run(["v4l2-ctl", "-d", device, arg, value], capture_output=True, text=True)

    problems = []
    tries = [("auto_exposure", 1), ("exposure_time_absolute", max(1, round(exposure_ms * 10))),
             ("exposure_dynamic_framerate", 0), ("white_balance_automatic", 0), ("focus_automatic_continuous", 0)]
    for name, value in tries:
        r = ctl("-c", f"{name}={value}")
        if r.returncode == 0 or name not in ("auto_exposure", "exposure_time_absolute"):
            continue
        if ctl("-C", name).stdout.split(":")[-1].split()[:1] == [str(value)]:
            continue  # refused, but already there
        if name == "exposure_time_absolute":
            ctl("-c", "auto_exposure=3")
            ctl("-c", "auto_exposure=1")
            r = ctl("-c", f"{name}={value}")
            if r.returncode == 0:
                continue
        problems.append(f"{name}={value}: {r.stderr.strip()}")
    return problems


def ignore_path(name: str):
    """Pixels the ball detector skips for camera `name` (255 = ignore): `scripts/cameras.py mask` makes it."""
    return CALIB_DIR / f"ignore_{name}.png"


def load_ignore(name: str) -> np.ndarray | None:
    p = ignore_path(name)
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None


# Ball diameter limits (px, f ~830 px/rad): at most ~0.8 m from either camera the 40 mm ball is >= ~40 px; at the
# wrist camera's 0.08 m minimum range ~400 px. The first real gravity recording had ~1000 speckles of 4-15 px on the
# red rubber's edge (it reads orange-ish under this light) between the real ball's 50-180 px detections.
MIN_BALL_PX, MAX_BALL_PX = 20.0, 420.0


class BallDetector:
    """`ignore`: pixels never counted as ball (a camera's own ball-coloured clutter: for the wrist camera, the blade
    edge, which is fixed in its picture)."""

    def __init__(self, color: BallColor, min_d: float = MIN_BALL_PX, max_d: float = MAX_BALL_PX, min_fill: float = 0.55,
                 ignore: np.ndarray | None = None):
        self.color, self.min_d, self.max_d, self.min_fill = color, min_d, max_d, min_fill
        self.lo = np.array([color.h[0], color.s[0], color.v[0]], np.uint8)
        self.hi = np.array([color.h[1], color.s[1], color.v[1]], np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.keep = None if ignore is None else (ignore == 0).astype(np.uint8) * 255

    def mask(self, frame_bgr: np.ndarray, raw: bool = False) -> np.ndarray:
        """Ball-coloured pixels (255), after the ignore mask unless `raw`."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.lo, self.hi)
        if self.keep is not None and not raw:
            m = cv2.bitwise_and(m, self.keep)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)

    def __call__(self, frame_bgr: np.ndarray) -> tuple[float, float, float] | None:
        """(u, v, diameter) in px of the best ball candidate, or None."""
        contours, _ = cv2.findContours(self.mask(frame_bgr), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best, best_score = None, 0.0
        for c in contours:
            area = cv2.contourArea(c)
            d = 2 * np.sqrt(area / np.pi)
            if not (self.min_d <= d <= self.max_d):
                continue
            (_, _), r = cv2.minEnclosingCircle(c)
            fill = area / (np.pi * r * r + 1e-9)  # 1 for a disc; motion blur stretches it into a capsule
            if fill < self.min_fill:
                continue
            score = fill * np.sqrt(area)
            if score > best_score:
                m = cv2.moments(c)
                best, best_score = (m["m10"] / m["m00"], m["m01"] / m["m00"], d), score
        return best


class UvcCamera(threading.Thread):
    def __init__(self, name: str, cfg: CameraConfig, process, max_queue: int = 64):
        super().__init__(daemon=True, name=f"cam-{name}")
        self.name_, self.cfg, self.process = name, cfg, process
        self.out: queue.Queue = queue.Queue(maxsize=max_queue)
        self.stop_flag = threading.Event()
        self.frames = 0
        self.dropped = 0  # results lost to a full queue (the consumer didn't drain often enough)
        self.stamp_source = "?"
        self.latency: list[float] = []  # arrival - buffer stamp, s (last 100)
        self.last_frame: np.ndarray | None = None
        self.error: str | None = None
        self.record = None  # callable(name, stamp, frame) given every frame (FrameRecorder.put)

    def open(self) -> cv2.VideoCapture:
        problems = set_controls(self.cfg.device, self.cfg.exposure_ms)
        if problems:
            print(f"{self.name_}: could not set " + "; ".join(problems))
        cap = cv2.VideoCapture(self.cfg.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"{self.name_}: can't open {self.cfg.device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        # One buffer halves the frame rate (the driver drops the frame arriving while we decode: 14.8 vs 29.5 fps
        # measured). Decode + detection is ~3 ms a frame, so the thread keeps up and buffers don't fill with stale frames.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != (self.cfg.width, self.cfg.height):
            raise RuntimeError(f"{self.name_}: got {w}x{h}, wanted {self.cfg.width}x{self.cfg.height}")
        set_controls(self.cfg.device, self.cfg.exposure_ms)  # some drivers reset controls on stream start
        return cap

    def run(self) -> None:
        try:
            cap = self.open()
        except Exception as e:
            self.error = str(e)
            return
        try:
            while not self.stop_flag.is_set():
                if not cap.grab():
                    self.error = "grab failed"
                    break
                arrival = time.perf_counter()
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                # Accept the buffer timestamp if it's on our clock (a frame is never from the future or > 1 s old).
                stamp = ms / 1000.0 if 0 < arrival - ms / 1000.0 < 1.0 else arrival
                self.stamp_source = "buffer" if stamp != arrival else "arrival"
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                self.frames += 1
                self.last_frame = frame
                if self.record is not None:
                    self.record(self.name_, stamp, frame)
                self.latency = (self.latency + [arrival - stamp])[-100:]
                result = self.process(frame) if self.process else None
                if result is not None:
                    try:
                        self.out.put_nowait((stamp, result))
                    except queue.Full:
                        self.dropped += 1
        finally:
            cap.release()

    def drain(self) -> list:
        items = []
        while True:
            try:
                items.append(self.out.get_nowait())
            except queue.Empty:
                return items

    def stop(self) -> None:
        """Stop capturing and wait for the thread (a capture thread alive at interpreter exit aborts the process)."""
        self.stop_flag.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(2.0)


class FrameRecorder(threading.Thread):
    """Saves frames as JPEG, `<dir>/<camera>/<buffer stamp, s>.jpg`, off the capture threads (encoding a frame takes
    a few ms). Frames arriving while its queue is full are dropped and counted."""

    def __init__(self, out_dir, quality: int = 90, max_queue: int = 128):
        super().__init__(daemon=True, name="frame-recorder")
        self.dir, self.quality = Path(out_dir), quality
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.stop_flag = threading.Event()
        self.saved = 0
        self.dropped = 0

    def put(self, camera: str, stamp: float, frame: np.ndarray) -> None:
        try:
            self.q.put_nowait((camera, stamp, frame))
        except queue.Full:
            self.dropped += 1

    def run(self) -> None:
        while not (self.stop_flag.is_set() and self.q.empty()):
            try:
                camera, stamp, frame = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            d = self.dir / camera
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{stamp:.6f}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            self.saved += 1

    def stop(self) -> None:
        self.stop_flag.set()
        if self.is_alive():
            self.join(20.0)


class BallCameras:
    """The rig's cameras running ball detection; `drain()` -> PixelDetections since the last call."""

    def __init__(self, rig: RigConfig, names: list[str]):
        missing = [n for n in names if n not in rig.cameras]
        if missing:
            raise SystemExit(f"no device for {missing} in {RIG}: run `scripts/cameras.py assign <name> <device>`")
        self.cams = [UvcCamera(n, rig.cameras[n], BallDetector(rig.ball, ignore=load_ignore(n))) for n in names]
        self.recorder: FrameRecorder | None = None

    def start(self) -> None:
        if self.recorder is not None:
            self.recorder.start()
        for c in self.cams:
            c.start()
        t = time.perf_counter()
        while any(c.frames == 0 and c.error is None for c in self.cams):
            if time.perf_counter() - t > 5.0:
                raise RuntimeError("no frames after 5 s from " + ", ".join(c.name_ for c in self.cams if c.frames == 0))
            time.sleep(0.05)
        errors = [f"{c.name_}: {c.error}" for c in self.cams if c.error]
        if errors:
            raise RuntimeError("; ".join(errors))

    def drain(self) -> list[PixelDetection]:
        return [PixelDetection(c.name_, stamp, *r) for c in self.cams for stamp, r in c.drain()]

    def record_to(self, out_dir) -> None:
        """Save every frame of every camera under `out_dir` from `start()` on (what the detector saw)."""
        self.recorder = FrameRecorder(out_dir)
        for c in self.cams:
            c.record = self.recorder.put

    def stop(self) -> None:
        for c in self.cams:
            c.stop()
        for c in self.cams:
            c.join(timeout=1.0)
        if self.recorder is not None:
            self.recorder.stop()
            print(f"frames saved: {self.recorder.saved} in {self.recorder.dir} (dropped {self.recorder.dropped})")
