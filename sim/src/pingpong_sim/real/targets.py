"""The printed calibration targets (scripts/make_calib_targets.py): detection and 3D corner positions.

Sizes: measurements/calibration/targets.json; the caliper readings in `measured_mm` (calibrate.py targets) win
over the design sizes.

- Paddle tag (AprilTag tag36h11): corners in the tag frame, OpenCV's order (top-left, top-right, bottom-right,
  bottom-left as printed) at (-s/2, s/2), (s/2, s/2), (s/2, -s/2), (-s/2, -s/2), z = 0 (out of the paper).
- ChArUco board: its chessboard corners in the board frame (x along the squares row, z = 0 on the paper).
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from pingpong_sim.real.camera_model import CALIB_DIR

SPEC = CALIB_DIR / "targets.json"


def load_spec() -> dict:
    spec = json.loads(SPEC.read_text())
    m = spec.get("measured_mm", {})
    spec["tag_mm"] = m.get("tag_black_square", spec["tag"]["black_square_mm"])
    spec["square_mm"] = m.get("board_square", spec["board"]["square_mm"])
    spec["marker_mm"] = m.get("board_marker", spec["board"]["marker_mm"] * spec["square_mm"] / spec["board"]["square_mm"])
    return spec


def _params(thresh_win_max: int = 23) -> cv2.aruco.DetectorParameters:
    """`thresh_win_max`: largest adaptive-threshold window [px] (OpenCV's default 23; tried from 3 in steps of 10)."""
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.adaptiveThreshWinSizeMax = thresh_win_max
    return p


class TagDetector:
    def __init__(self, spec: dict | None = None):
        spec = spec or load_spec()
        self.id = spec["tag"]["paddle_id"]
        self.size = spec["tag_mm"] / 1000
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["tag"]["opencv_dictionary"]))
        # The paddle tag gets ~250 px across (cells ~30 px): windows up to 23 px miss it in a dim picture.
        self.det = cv2.aruco.ArucoDetector(d, _params(thresh_win_max=63))
        s = self.size / 2
        self.object_points = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)

    def __call__(self, frame: np.ndarray) -> np.ndarray | None:
        """The paddle tag's corners (4, 2) in px, or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.det.detectMarkers(gray)
        if ids is None:
            return None
        for c, i in zip(corners, ids.ravel()):
            if i == self.id:
                return c.reshape(4, 2).astype(float)
        return None


_DARKEN = [(255 * (np.arange(256) / 255.0) ** g).astype(np.uint8) for g in (3.0, 4.0)]  # gamma curves


class BoardDetector:
    def __init__(self, spec: dict | None = None, min_corners: int = 6):
        spec = spec or load_spec()
        b = spec["board"]
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b["opencv_dictionary"]))
        self.board = cv2.aruco.CharucoBoard((b["squares_x"], b["squares_y"]), spec["square_mm"] / 1000,
                                            spec["marker_mm"] / 1000, d)
        self.board.setLegacyPattern(b["legacy_pattern"])
        cp = cv2.aruco.CharucoParameters()
        self.det = cv2.aruco.CharucoDetector(self.board, cp, _params())
        self.corners3d = self.board.getChessboardCorners().astype(float)  # (N, 3)
        self.min_corners = min_corners

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(ids (n,), corners (n, 2) px) of the detected chessboard corners, or None if fewer than min_corners.
        A washed-out picture (bright light, a lamp in view: the markers' black cells turn grey and stop decoding)
        is tried again darkened: 2026-11-12, 0 -> 19 of 23 wrist views; corners move ~0.6 px."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        best = None
        for lut in (None, *_DARKEN):
            corners, ids, _, _ = self.det.detectBoard(gray if lut is None else cv2.LUT(gray, lut))
            if ids is not None and (best is None or len(ids) > len(best[0])):
                best = ids, corners
            if lut is None and best is not None and len(ids) >= self.min_corners:
                break
        if best is None or len(best[0]) < self.min_corners:
            return None
        return best[0].ravel().astype(int), best[1].reshape(-1, 2).astype(float)
