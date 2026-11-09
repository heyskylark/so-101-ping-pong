# /// script
# requires-python = ">=3.11"
# dependencies = ["trimesh", "numpy", "scipy", "networkx", "shapely", "rtree", "manifold3d"]
# ///
"""SO-ARM100 overhead camera mount top, modified so the 32x32 camera board can be turned a quarter turn.

The stock `cam_mount_top.stl` (TheRobotStudio/SO-ARM100, Optional/Overhead_Cam_Mount_32x32_UVC_Module, Apache-2.0)
holds the board with its lens holder through a cross-shaped cutout (15 x 27 mm, long arms along the post), so the
picture's long edge can only lie across the arm. Our setup needs it along the view tilt (docs/camera_setup.md).
This adds the same cutout turned 90 deg about its centre (a plus-shaped opening); the board's square 27 mm screw
pattern already fits both ways. Nothing else changes.

Output: cad/overhead_mount/cam_mount_top_rotatable.stl (print it instead of cam_mount_top.stl).

Run: uv run cad/overhead_mount/make_rotatable_top.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import Polygon

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / ".cache" / "overhead"
URL = ("https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Optional/"
       "Overhead_Cam_Mount_32x32_UVC_Module/stl/cam_mount_top.stl")
OUT = HERE / "cam_mount_top_rotatable.stl"

# Camera plate of the stock part (mm, part frame): outward normal and the offset of its mid-plane along it
# (faces at -316.6 and -319.6: a 3 mm plate).
PLATE_NORMAL = np.array([0.42, -0.91, 0.0])
PLATE_MID = -318.1


def main() -> None:
    src = CACHE / "cam_mount_top.stl"
    if not src.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, src)
    mesh = trimesh.load(src)
    n = PLATE_NORMAL / np.linalg.norm(PLATE_NORMAL)
    u = np.array([0.0, 0.0, 1.0])
    w = np.cross(n, u)
    # The cutout is the 15 mm wide loop of the plate's mid-plane section (the others: plate outline, screw holes).
    loops = [np.c_[e @ w, e @ u] for e in mesh.section(plane_origin=n * PLATE_MID, plane_normal=n).discrete]
    cut = Polygon(next(p for p in loops if 14.0 < np.ptp(p[:, 0]) < 16.0))
    turned = affinity.rotate(cut, 90, origin=cut.centroid)
    prism = trimesh.creation.extrude_polygon(turned, height=12.0)
    prism.apply_translation([0.0, 0.0, -6.0])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = w, u, n, n * PLATE_MID
    prism.apply_transform(T)
    out = mesh.difference(prism, engine="manifold")
    assert out.is_watertight
    out.export(OUT)
    print(f"cutout {np.round(cut.bounds, 2)} + turned {np.round(turned.bounds, 2)} (plate coords, mm); "
          f"removed {mesh.volume - out.volume:.0f} mm3 -> {OUT.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
