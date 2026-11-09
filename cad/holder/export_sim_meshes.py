# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["build123d", "numpy", "pyvista"]
# ///
"""Export the wrist payload meshes for the simulation, all in the holder frame, in mm.

Holder frame (see paddle_holder.fs): origin on the horn face centre, +z toward the paddle, x = blade normal with
the cap at +x, y across the blade. In the MuJoCo `gripper` body this frame sits at quat (0 1 0 0), like the stock
follower (see sim/src/pingpong_sim/robot.py).

Outputs in sim/src/pingpong_sim/assets/paddle/:
  holder_base.stl, holder_cap.stl   from paddle_holder_{base,cap}.step
  camera_mount.stl, camera.stl      stock hex-nut wrist camera mount + 32x32 UVC module, placed as in
                                    preview_on_arm.py (the pose its collision sweep checked)
  paddle.stl                        handle loft + blade (visual)
  blade_collision.stl               rubber-covered blade region (x >= rubber_start_x) as a prism; MuJoCo uses its
                                    convex hull, which is exact here because that region of the outline is convex

Run: uv run cad/holder/export_sim_meshes.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import preview_on_arm as pa  # noqa: E402

OUT = pa.REPO / "sim" / "src" / "pingpong_sim" / "assets" / "paddle"
RUBBER_START_X = 100.9  # cad/paddle/measurements.yaml blade.rubber_start_x, from the butt


def blade_prism():
    """Prism over the rubber-covered part of the outline, in the holder frame (mm)."""
    import pyvista as pv

    with open(pa.REPO / "cad" / "paddle" / "outline.csv") as f:
        pts = np.array([(float(x), float(y)) for x, y in list(csv.reader(f))[1:]])
    # Clip the closed outline at x = RUBBER_START_X, keeping the blade side.
    ring = np.vstack([pts, pts[:1]])
    kept = []
    for a, b in zip(ring[:-1], ring[1:]):
        if a[0] >= RUBBER_START_X:
            kept.append(a)
        if (a[0] - RUBBER_START_X) * (b[0] - RUBBER_START_X) < 0:
            t = (RUBBER_START_X - a[0]) / (b[0] - a[0])
            kept.append(a + t * (b - a))
    kept = np.array(kept)
    half = pa.BLADE_THICKNESS / 2
    # Holder frame: blade normal x, width y, paddle axis z (butt at BUTT_Z).
    top = np.column_stack([np.full(len(kept), half), kept[:, 1], pa.BUTT_Z + kept[:, 0]])
    bottom = top * [-1, 1, 1]
    return pv.PolyData(np.vstack([top, bottom])).delaunay_3d().extract_surface(algorithm="dataset_surface")


def main() -> None:
    import pyvista as pv
    from build123d import export_stl, import_step

    OUT.mkdir(parents=True, exist_ok=True)
    for part in ("base", "cap"):
        export_stl(import_step(HERE / f"paddle_holder_{part}.step"), str(OUT / f"holder_{part}.stl"), tolerance=0.02)
    export_stl(import_step(pa.fetch("camera_mount")).moved(pa.CAM_MOUNT), str(OUT / "camera_mount.stl"), tolerance=0.02)
    export_stl(pa.paddle_solid(), str(OUT / "paddle.stl"), tolerance=0.05)

    camera = pv.read(pa.fetch("camera")).scale(1000)  # upstream mesh is in m
    camera.transform(pa.matrix(pa.CAM_MOUNT) @ pa.quat_matrix(pa.CAMERA_POS, pa.CAMERA_QUAT), inplace=True)
    camera.save(OUT / "camera.stl")
    blade_prism().save(OUT / "blade_collision.stl")

    for f in sorted(OUT.glob("*.stl")):
        m = pv.read(f)
        print(f"{f.name:22s} bounds(mm) {np.round(m.bounds, 1)}  volume {m.volume / 1000:7.2f} cm3")


if __name__ == "__main__":
    main()
