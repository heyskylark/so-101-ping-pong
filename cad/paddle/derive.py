# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "scipy", "pyyaml", "ezdxf"]
# ///
"""Derive the paddle outline and rigid-body properties from measurements.yaml.

Outputs (next to this file):
  outline.csv   symmetric blade/handle outline in the measurements.yaml frame (x, y) [mm]
  outline.dxf   same outline as a closed spline, for Onshape "Insert DXF"
  derived.yaml  pencil offset, checks, fitted densities, mass split, CoM, inertia

Run: uv run cad/paddle/derive.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import ezdxf
import numpy as np
import yaml
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
PX_PER_MM = 10.0
# Not measurable without peeling the rubber: inverted rubber + sponge sheets are roughly 0.6-0.8 g/cm^3.
RUBBER_DENSITY_G_CM3 = 0.65


def rectify_paper(img: np.ndarray, paper_mm: tuple[float, float]) -> np.ndarray:
    """Warp the sheet of paper to a fronto-parallel image at PX_PER_MM."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    paper = ((hsv[..., 1] < 40) & (hsv[..., 2] > 150)).astype(np.uint8) * 255
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sheet = max(contours, key=cv2.contourArea)
    quad = cv2.approxPolyDP(sheet, 0.01 * cv2.arcLength(sheet, True), True).reshape(-1, 2).astype(np.float32)
    if len(quad) != 4:
        raise RuntimeError(f"paper outline is not a quadrilateral ({len(quad)} corners)")
    s, d = quad.sum(1), np.diff(quad, axis=1).ravel()
    rough = [quad[np.argmin(s)], quad[np.argmin(d)], quad[np.argmax(s)], quad[np.argmax(d)]]  # TL TR BR BL

    # Refine: robust line fit per edge, corners = edge intersections.
    pts = sheet.reshape(-1, 2).astype(np.float32)
    edges = []
    for a, b in zip(rough, rough[1:] + rough[:1]):
        length = np.linalg.norm(b - a)
        u = (b - a) / length
        n = np.array([-u[1], u[0]])
        t, dist = (pts - a) @ u, np.abs((pts - a) @ n)
        sel = pts[(t > 0.1 * length) & (t < 0.9 * length) & (dist < 15)]
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        edges.append((np.array([x0, y0]), np.array([vx, vy])))

    def intersect(e1, e2):
        (p, r), (q, s_) = e1, e2
        t = np.linalg.solve(np.array([r, -s_]).T, q - p)
        return p + t[0] * r

    corners = np.array([intersect(edges[i - 1], edges[i]) for i in range(4)], np.float32)
    w_mm, h_mm = paper_mm
    dst = np.float32([[0, 0], [w_mm, 0], [w_mm, h_mm], [0, h_mm]]) * PX_PER_MM
    measured_aspect = (np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])) / (
        np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])
    )
    if abs(measured_aspect / (h_mm / w_mm) - 1) > 0.01:
        raise RuntimeError(f"paper aspect {measured_aspect:.4f} does not match {h_mm / w_mm:.4f}")
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, H, (int(w_mm * PX_PER_MM), int(h_mm * PX_PER_MM)), flags=cv2.INTER_CUBIC)


def trace_mask(rect: np.ndarray) -> np.ndarray:
    """Filled region bounded by the inner edge of the pencil line."""
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    line = ((cv2.medianBlur(gray, 51).astype(float) - gray) > 11).astype(np.uint8)
    border = int(4 * PX_PER_MM)
    line[:border], line[-border:], line[:, :border], line[:, -border:] = 0, 0, 0, 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(line, 8)
    line = np.isin(labels, [i for i in range(1, n) if stats[i, 4] >= 40]).astype(np.uint8)

    # Bridge faint gaps in the stroke, flood the inside, then grow back to the line's inner edge.
    h, w = line.shape
    for r in (7, 10, 15, 20, 25):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        walls = cv2.dilate(line, kernel) * 255
        cv2.floodFill(walls, np.zeros((h + 2, w + 2), np.uint8), (w // 2, int(h * 0.7)), 128)
        inside = (walls == 128).astype(np.uint8)
        if inside[2, 2] == 0 and inside.sum() < 0.6 * h * w:
            inside = cv2.dilate(inside, kernel)
            inside[line > 0] = 0
            inside = cv2.morphologyEx(inside, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask = np.zeros_like(inside)
            cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, -1)
            return mask
    raise RuntimeError("pencil trace is not closed even after bridging 5 mm gaps")


def half_width_profile(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Align the handle axis with image rows; return x [mm from butt], symmetric half-width [mm], max centerline skew."""
    rows = np.nonzero(mask.any(1))[0]
    butt = rows[0]
    ys = np.arange(butt + 10 * PX_PER_MM, butt + 70 * PX_PER_MM, 2 * PX_PER_MM).astype(int)
    cx = [np.nonzero(mask[r])[0][[0, -1]].mean() for r in ys]
    slope = np.polyfit(ys, cx, 1)[0]
    M = cv2.getRotationMatrix2D((float(np.mean(cx)), float(np.mean(ys))), -np.degrees(np.arctan(slope)), 1.0)
    mask = cv2.warpAffine(mask, M, mask.shape[::-1], flags=cv2.INTER_NEAREST)

    rows = np.nonzero(mask.any(1))[0]
    x, hw, center = [], [], []
    for r in range(rows[0], rows[-1] + 1):
        cols = np.nonzero(mask[r])[0]
        x.append((r - rows[0]) / PX_PER_MM)
        hw.append((cols[-1] - cols[0] + 1) / 2 / PX_PER_MM)
        center.append((cols[-1] + cols[0]) / 2 / PX_PER_MM)
    center, hw, x = np.array(center), np.array(hw), np.array(x)
    body = (x > 10) & (x < x[-1] - 10)  # ignore the rounded ends, where a row is only a few pixels wide
    skew = float(np.abs(center[body] - np.median(center[body])).max())
    return x, hw, skew


def handle_section(x: np.ndarray, meas: dict) -> tuple[np.ndarray, np.ndarray]:
    """Caliper width/thickness of the handle at x (mm from butt), held constant past the end stations."""
    st = np.array(meas["handle"]["stations"], float)
    sx = meas["handle"]["shoulder_x"] - st[:, 0]
    order = np.argsort(sx)
    return np.interp(x, sx[order], st[order, 1]), np.interp(x, sx[order], st[order, 2])


def mass_properties(x, hw, meas):
    """Fit handle/blade wood densities to mass + balance point; return them with CoM and inertia about the CoM.

    Three materials, two measurements: the rubber density is assumed (RUBBER_DENSITY_G_CM3) and the two
    wood densities are solved from the total mass and the balance point.
    """
    dx = x[1] - x[0]
    core_t = meas["blade"]["wood_core_thickness"]
    rub_t = meas["blade"]["rubber_thickness_per_side"]
    shoulder = meas["handle"]["shoulder_x"]
    r = meas["handle"]["corner_radius"]

    in_handle = x <= shoulder
    w_h, t_h = handle_section(x, meas)
    face_w = 2 * hw
    # Volumes per slice [mm^3]
    v_handle = np.where(in_handle, w_h * t_h - (4 - np.pi) * r**2, 0.0) * dx
    v_core = np.where(in_handle, 0.0, face_w * core_t) * dx
    v_rub = np.where(x >= meas["blade"]["rubber_start_x"], face_w * 2 * rub_t, 0.0) * dx
    # Per-slice second moments <y^2>, <z^2> [mm^2]
    y2_handle, z2_handle = w_h**2 / 12, t_h**2 / 12
    y2_face = face_w**2 / 12
    z2_core = core_t**2 / 12
    z_in, z_out = core_t / 2, core_t / 2 + rub_t
    z2_rub = (z_out**3 - z_in**3) / (3 * rub_t)

    m, xc_meas = meas["mass"], meas["balance_point_x"]
    dm_r = RUBBER_DENSITY_G_CM3 * 1e-3 * v_rub
    A = np.array([[v_handle.sum(), v_core.sum()], [(v_handle * x).sum(), (v_core * x).sum()]])
    b = np.array([m - dm_r.sum(), m * xc_meas - (dm_r * x).sum()])
    rho_h, rho_c = np.linalg.solve(A, b)
    dm_h, dm_c = rho_h * v_handle, rho_c * v_core
    dm = dm_h + dm_c + dm_r
    xc = (dm * x).sum() / dm.sum()

    y2 = dm_h * y2_handle + (dm_c + dm_r) * y2_face
    z2 = dm_h * z2_handle + dm_c * z2_core + dm_r * z2_rub
    x2 = dm * (x - xc) ** 2
    to_si = 1e-3 * 1e-6  # g*mm^2 -> kg*m^2
    return {
        "assumed_rubber_density_g_cm3": RUBBER_DENSITY_G_CM3,
        "fitted_density_g_cm3": {"handle": round(float(rho_h * 1e3), 3), "blade_core": round(float(rho_c * 1e3), 3)},
        "mass_g": {
            "handle": round(float(dm_h.sum()), 1),
            "blade_core": round(float(dm_c.sum()), 1),
            "rubber_both_sides": round(float(dm_r.sum()), 1),
        },
        "com_x_mm": round(float(xc), 2),
        "inertia_about_com_kg_m2": {  # principal axes = paddle x, y, z by symmetry
            "ixx": float(f"{(y2.sum() + z2.sum()) * to_si:.4e}"),
            "iyy": float(f"{(x2.sum() + z2.sum()) * to_si:.4e}"),
            "izz": float(f"{(x2.sum() + y2.sum()) * to_si:.4e}"),
        },
    }


def outline_points(x, hw, eps_mm=0.1) -> np.ndarray:
    """Closed outline (x, y) [mm], counter-clockwise, simplified to eps_mm."""
    right = np.column_stack([x, hw])
    left = np.column_stack([x[::-1], -hw[::-1]])
    poly = np.vstack([right, left])
    simplified = cv2.approxPolyDP((poly * 100).astype(np.int32).reshape(-1, 1, 2), eps_mm * 100, True)
    return simplified.reshape(-1, 2) / 100.0


def main() -> None:
    meas = yaml.safe_load((HERE / "measurements.yaml").read_text())
    img = cv2.imread(str(HERE / meas["trace"]["photo"]))
    if img is None:
        raise FileNotFoundError(meas["trace"]["photo"])
    mask = trace_mask(rectify_paper(img, tuple(meas["trace"]["paper_mm"])))
    _, hw_raw, _ = half_width_profile(mask)

    # Pencil offset: constant inward shift (2D erosion) that makes the blade width match the calipers.
    offset = (2 * hw_raw.max() - meas["blade_max_width"]) / 2
    k = 2 * int(round(offset * PX_PER_MM)) + 1
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    x, hw, skew = half_width_profile(mask)
    hw = np.clip(savgol_filter(hw, 21, 3), 0, None)  # 2 mm window

    pts = outline_points(x, hw)
    np.savetxt(HERE / "outline.csv", pts, fmt="%.2f", delimiter=",", header="x_mm,y_mm", comments="")
    doc = ezdxf.new(units=ezdxf.units.MM)
    doc.modelspace().add_lwpolyline(pts.tolist(), close=True)
    doc.saveas(HERE / "outline.dxf")

    face = x >= meas["blade"]["rubber_start_x"]
    trace_vs_caliper = [
        round(float(2 * np.interp(meas["handle"]["shoulder_x"] - s, x, hw) - w), 2)
        for (s, w, _t) in meas["handle"]["stations"]
    ]
    derived = {
        "generated_by": "derive.py",
        "pencil_offset_mm": round(float(offset), 2),
        "checks": {
            "traced_length_mm": round(float(x[-1]), 1),
            "caliper_length_mm": meas["overall_length"],
            "outline_centerline_skew_mm": round(skew, 2),
            "trace_minus_caliper_handle_width_mm": trace_vs_caliper,
        },
        "blade_face_area_cm2": round(float(np.trapezoid(2 * hw[face], x[face]) / 100), 1),
        "blade_face_centroid_x_mm": round(float(np.trapezoid(2 * hw[face] * x[face], x[face]) / np.trapezoid(2 * hw[face], x[face])), 1),
        "blade_widest_x_mm": round(float(x[np.argmax(hw)]), 1),
        **mass_properties(x, hw, meas),
    }
    (HERE / "derived.yaml").write_text(
        "# Generated by derive.py from measurements.yaml; do not edit.\n" + yaml.safe_dump(derived, sort_keys=False)
    )
    print(yaml.safe_dump(derived, sort_keys=False))


if __name__ == "__main__":
    main()
