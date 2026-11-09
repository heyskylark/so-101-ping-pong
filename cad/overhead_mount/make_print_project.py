# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy-stl"]
# ///

"""Build a sliced Bambu Studio file (X1C, 0.4 nozzle) for the rotatable overhead camera mount top only.

The middle and bottom sections of the SO-ARM100 overhead mount are unchanged; only the top section (camera plate)
is reprinted. The part lies on the flat back face of its post (the side away from the camera head), so the head
points up (~58 mm tall); the camera plate's underside then overhangs 25 deg past vertical and needs no support.
Tree supports on the build plate catch anything the slicer still flags.

Output: cad/overhead_mount/build/cam_mount_top_rotatable_x1c.gcode.3mf, which Bambu Studio opens into Preview,
ready for "Print plate".

Run: uv run cad/overhead_mount/make_print_project.py [--filament "Bambu PETG HF @BBL X1C"]
Needs the Flathub Bambu Studio (com.bambulab.BambuStudio) and the STL from make_rotatable_top.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from stl import mesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "holder"))
from make_print_project import APP, MACHINE, PROCESS, PROJECT_OVERRIDES, flatten, mark_modified  # noqa: E402

BUILD = HERE / "build"
PART = HERE / "cam_mount_top_rotatable.stl"
PROCESS_OVERRIDES = {
    "wall_loops": "3",
    "enable_support": "1",
    "support_type": "tree(auto)",
    "support_on_build_plate_only": "1",
}


def oriented_stl() -> Path:
    """The part on its post's back face (part -x down): rotate -90 deg about y, then onto the bed."""
    m = mesh.Mesh.from_file(str(PART))
    a = np.radians(-90.0)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    m.rotate_using_matrix(ry.T)  # numpy-stl applies the matrix transposed
    m.translate(np.array([0, 0, -m.min_[2]]))
    height = np.ptp(m.vectors[..., 2])
    if not 55 < height < 61:
        raise RuntimeError(f"orientation is wrong (height {height:.1f} mm)")
    out = BUILD / PART.name
    m.save(str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filament", default="Bambu PLA Basic @BBL X1C")
    args = ap.parse_args()
    BUILD.mkdir(exist_ok=True)
    configs = {"machine": flatten("machine", MACHINE),
               "process": flatten("process", PROCESS) | PROCESS_OVERRIDES | PROJECT_OVERRIDES,
               "filament": flatten("filament", args.filament)}
    paths = {}
    for kind, cfg in configs.items():
        paths[kind] = BUILD / f"{kind}.json"
        paths[kind].write_text(json.dumps(cfg, indent=1))
    stl = oriented_stl()
    out = "cam_mount_top_rotatable_x1c.gcode.3mf"
    subprocess.run([
        "flatpak", "run", f"--filesystem={BUILD}", "--command=bambu-studio", APP,
        "--load-settings", f"{paths['machine']};{paths['process']}",
        "--load-filaments", str(paths["filament"]),
        "--arrange", "1", "--orient", "0", "--ensure-on-bed",
        "--slice", "0", "--outputdir", str(BUILD), "--export-3mf", out, str(stl),
    ], check=True)
    mark_modified(BUILD / out, PROCESS_OVERRIDES)
    print("wrote", BUILD / out)


if __name__ == "__main__":
    main()
