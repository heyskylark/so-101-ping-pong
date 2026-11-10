"""Printable calibration targets (US Letter, 600 dpi, exact scale) and their spec.

- Page 1: AprilTag tag36h11 ID 0 (the paddle's back face) and ID 1 (spare), 70 mm black square, each with its
  one-cell white border and a cut line around it.
- Page 2: ChArUco board, 5 x 7 squares of 30 mm, DICT_5X5_100 markers (lens calibration, wrist camera mount).
- A 100 mm ruler on each page: measure it after printing to confirm the scale (print with scaling off).

Writes build/calib_targets.pdf and measurements/calibration/targets.json (design sizes; `measured_mm` fields are
filled in by `scripts/calibrate.py targets` from caliper readings, and the calibration uses those).

Print:
  uv run scripts/make_calib_targets.py
  lp -d Brother_HL_L3220CDW -o print-scaling=none -o media=Letter build/calib_targets.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = ROOT / "sim" / "build" / "calib_targets.pdf"
SPEC = ROOT / "measurements" / "calibration" / "targets.json"

DPI = 600
PAGE_IN = (8.5, 11.0)
MARGIN_MM = 4.4  # printer's unprintable margin (IPP: 4.32 mm)
PX_PER_MM = DPI / 25.4

TAG_DICT, TAG_IDS, TAG_CELL_PX = "DICT_APRILTAG_36h11", (0, 1), 207  # 8 cells x 207 px = 70.1 mm black square
BOARD_DICT, BOARD_SQUARES, BOARD_SQUARE_PX, BOARD_MARKER_RATIO = "DICT_5X5_100", (5, 7), 709, 22.0 / 30.0


def mm(v: float) -> int:
    return int(round(v * PX_PER_MM))


def page() -> Image.Image:
    return Image.new("L", (int(PAGE_IN[0] * DPI), int(PAGE_IN[1] * DPI)), 255)


def font(size_pt: float):
    return ImageFont.load_default(size=int(size_pt * DPI / 72))


def ruler(img: Image.Image, x0: float, y0: float, vertical: bool = False) -> None:
    """100 mm ruler at (x0, y0) mm with 10 mm ticks (1 mm ticks along the first 10 mm); labelled if horizontal."""
    d = ImageDraw.Draw(img)
    w = 4
    if vertical:
        d.line([(mm(x0), mm(y0)), (mm(x0), mm(y0 + 100))], fill=0, width=w)
    else:
        d.line([(mm(x0), mm(y0)), (mm(x0 + 100), mm(y0))], fill=0, width=w)
    for t in range(0, 101):
        if t % 10 and t > 10:
            continue
        length = 4.0 if t % 10 == 0 else 2.0
        if vertical:
            d.line([(mm(x0), mm(y0 + t)), (mm(x0 + length), mm(y0 + t))], fill=0, width=w)
        else:
            d.line([(mm(x0 + t), mm(y0)), (mm(x0 + t), mm(y0 - length))], fill=0, width=w)
    if not vertical:
        d.text((mm(x0 + 25), mm(y0 + 1.5)), "100 mm: check with calipers", fill=0, font=font(8))


def dashed_rect(d: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, dash: int = 60) -> None:
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash))
        for i in range(0, n, 2):
            p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
            q = (a[0] + (b[0] - a[0]) * (i + 1) / n, a[1] + (b[1] - a[1]) * (i + 1) / n)
            d.line([p, q], fill=128, width=3)


def tag_page() -> tuple[Image.Image, dict]:
    img = page()
    d = ImageDraw.Draw(img)
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, TAG_DICT))
    side = 8 * TAG_CELL_PX
    d.text((mm(15), mm(12)), "AprilTag tag36h11. ID 0: glue flat on the paddle's BACK face (the side that never hits).",
           fill=0, font=font(11))
    d.text((mm(15), mm(19)), "ID 1: spare. Cut along the dashed line (keeps the white border). Printed at 100 %.",
           fill=0, font=font(11))
    for k, tag_id in enumerate(TAG_IDS):
        marker = cv2.aruco.generateImageMarker(dic, tag_id, side, borderBits=1)
        x, y = mm(40), mm(35 + k * 115)
        img.paste(Image.fromarray(marker), (x, y))
        q = TAG_CELL_PX  # one-cell white quiet zone
        dashed_rect(d, x - q, y - q, x + side + q, y + side + q)
        d.text((x + side + q + mm(5), y + side // 2 - mm(4)), f"ID {tag_id}\nblack square\n{side / PX_PER_MM:.1f} mm",
               fill=0, font=font(11))
    ruler(img, 15, 262)
    ruler(img, 12, 130, vertical=True)
    spec = {"family": "tag36h11", "opencv_dictionary": TAG_DICT, "paddle_id": TAG_IDS[0], "spare_ids": list(TAG_IDS[1:]),
            "black_square_mm": round(side / PX_PER_MM, 3)}
    return img, spec


def board_page() -> tuple[Image.Image, dict]:
    img = page()
    d = ImageDraw.Draw(img)
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_DICT))
    sx, sy = BOARD_SQUARES
    sq_mm = BOARD_SQUARE_PX / PX_PER_MM
    board = cv2.aruco.CharucoBoard((sx, sy), sq_mm / 1000, sq_mm * BOARD_MARKER_RATIO / 1000, dic)
    w, h = sx * BOARD_SQUARE_PX, sy * BOARD_SQUARE_PX
    pattern = board.generateImage((w, h), marginSize=0, borderBits=1)
    x, y = (img.width - w) // 2, mm(22)
    img.paste(Image.fromarray(pattern), (x, y))
    d.text((mm(15), mm(10)), f"ChArUco {sx} x {sy}, {sq_mm:.2f} mm squares, {BOARD_DICT}. Glue flat to a rigid board "
                             "(foam board, cardboard); keep it flat.", fill=0, font=font(10))
    ruler(img, 15, 262)
    marker_px = round(BOARD_SQUARE_PX * BOARD_MARKER_RATIO)  # what generateImage draws, to the pixel
    spec = {"opencv_dictionary": BOARD_DICT, "squares_x": sx, "squares_y": sy, "square_mm": round(sq_mm, 3),
            "marker_mm": round(marker_px / PX_PER_MM, 3), "legacy_pattern": False}
    return img, spec


def main() -> None:
    p1, tag = tag_page()
    p2, board = board_page()
    # Content must stay inside the printable area (no scaling: anything outside is clipped, not shrunk).
    for im in (p1, p2):
        box = Image.eval(im, lambda v: 255 - v).getbbox()
        lim = (mm(MARGIN_MM), mm(MARGIN_MM), im.width - mm(MARGIN_MM), im.height - mm(MARGIN_MM))
        if box[0] < lim[0] or box[1] < lim[1] or box[2] > lim[2] or box[3] > lim[3]:
            raise SystemExit(f"content {box} outside the printable area {lim}")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    p1.convert("1").save(OUT_PDF, save_all=True, append_images=[p2.convert("1")], resolution=DPI)
    SPEC.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(SPEC.read_text()) if SPEC.exists() else {}
    spec = {"tag": tag, "board": board, "page": {"paper": "Letter", "dpi": DPI},
            "measured_mm": old.get("measured_mm", {})}
    SPEC.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {OUT_PDF} and {SPEC}")
    print(f"tag black square {tag['black_square_mm']} mm; board squares {board['square_mm']} mm, markers "
          f"{board['marker_mm']} mm ({board['squares_x']} x {board['squares_y']})")


if __name__ == "__main__":
    main()
