# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy-stl"]
# ///

"""Build a sliced Bambu Studio file (X1C, 0.4 nozzle) for the holder halves.

Each half is laid on its flat outer face (layers parallel to the blade), arranged by Bambu Studio's CLI, given its
support blockers, and sliced with the system presets plus the overrides below. Output:
cad/holder/build/paddle_holder_x1c.gcode.3mf (or paddle_holder_<part>_x1c.gcode.3mf for one half), which Bambu
Studio opens straight into Preview, ready for "Print plate".

Run: uv run cad/holder/make_print_project.py [--parts cap] [--filament "Bambu PETG HF @BBL X1C"]
Needs the Flathub Bambu Studio (com.bambulab.BambuStudio).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import uuid
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from stl import mesh

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
APP = "com.bambulab.BambuStudio"
PROFILES = Path("/var/lib/flatpak/app") / APP / "current/active/files/share/BambuStudio/profiles/BBL"
MACHINE = "Bambu Lab X1 Carbon 0.4 nozzle"
PROCESS = "0.20mm Standard @BBL X1C"

# Strength over the stock profile. Only the base's horn boss needs support (everything else is teardropped,
# chamfered at 45 deg or bridged); tree supports from the build plate cover it.
PROCESS_OVERRIDES = {
    "wall_loops": "4",
    "top_shell_layers": "5",
    "bottom_shell_layers": "4",
    "sparse_infill_density": "40%",
    "sparse_infill_pattern": "gyroid",
    "enable_support": "1",
    "support_type": "tree(auto)",
    "support_on_build_plate_only": "1",
}
PROJECT_OVERRIDES = {"curr_bed_type": "Textured PEI Plate"}  # the plate on this X1C

# Rotation about y that puts each half's flat outer face (x = -16.5 base, x = +16.5 cap) on the bed.
ORIENT_Y_DEG = {"base": -90.0, "cap": 90.0}

# Support blockers per half: boxes in the holder frame (mm, [min corner, max corner]). The cap's nut trap
# (paddle_holder.fs: cap camera screw at x = 4.7, z = 23.4, NUT_* sizes, mouth relief) runs up from the cap's print
# bed, and without a blocker the slicer grows a support tree up it to the trap's roof, which blocks the nut.
SUPPORT_BLOCKERS = {"cap": [((1.0, -21.2, 19.8), (17.5, -16.9, 27.0))]}


def flatten(category: str, name: str) -> dict:
    """System preset with its `inherits` chain and `include` templates merged in."""
    data = json.loads((PROFILES / category / f"{name}.json").read_text())
    merged = flatten(category, data["inherits"]) if data.get("inherits") else {}
    for inc in data.get("include", []):
        merged.update({k: v for k, v in flatten(category, inc).items() if k not in ("name", "instantiation")})
    merged.update({k: v for k, v in data.items() if k not in ("inherits", "include")})
    return merged


def oriented_stl(part: str) -> tuple[Path, np.ndarray]:
    """Write the half lying on its flat outer face; returns the STL and the 4x4 holder-frame -> STL transform."""
    m = mesh.Mesh.from_file(str(HERE / f"paddle_holder_{part}.stl"))
    a = np.radians(ORIENT_Y_DEG[part])
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    m.rotate_using_matrix(ry.T)  # numpy-stl applies the matrix transposed
    shift = -m.min_[2]
    m.translate(np.array([0, 0, shift]))
    if np.ptp(m.vectors[..., 2]) > 40:
        raise RuntimeError(f"{part}: orientation is wrong (height {np.ptp(m.vectors[..., 2]):.1f} mm)")
    out = BUILD / f"paddle_holder_{part}.stl"
    m.save(str(out))
    t = np.eye(4)
    t[:3, :3] = ry
    t[2, 3] = shift
    return out, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filament", default="Bambu PLA Basic @BBL X1C")
    ap.add_argument("--parts", nargs="+", choices=("base", "cap"), default=["base", "cap"],
                    help="halves to put on the plate (e.g. --parts cap to reprint just the cap)")
    args = ap.parse_args()
    BUILD.mkdir(exist_ok=True)

    machine = flatten("machine", MACHINE)
    process = flatten("process", PROCESS) | PROCESS_OVERRIDES | PROJECT_OVERRIDES
    filament = flatten("filament", args.filament)
    paths = {}
    for kind, cfg in (("machine", machine), ("process", process), ("filament", filament)):
        paths[kind] = BUILD / f"{kind}.json"
        paths[kind].write_text(json.dumps(cfg, indent=1))
    blockers = {}
    stls = []
    for part in args.parts:
        stl, to_stl = oriented_stl(part)
        stls.append(stl)
        blockers[stl.name] = [(to_stl @ np.append(corner, 1))[:3] for box in SUPPORT_BLOCKERS.get(part, [])
                              for corner in box_corners(*box)]

    out = f"paddle_holder_{args.parts[0] + '_' if len(args.parts) == 1 else ''}x1c.gcode.3mf"
    run = ["flatpak", "run", f"--filesystem={BUILD}", "--command=bambu-studio", APP]
    # Pass 1 loads and arranges the STLs; pass 2 slices the project again with the support blockers added.
    subprocess.run([
        *run,
        "--load-settings", f"{paths['machine']};{paths['process']}",
        "--load-filaments", str(paths["filament"]),
        "--arrange", "1", "--orient", "0", "--ensure-on-bed",
        "--slice", "0", "--outputdir", str(BUILD), "--export-3mf", out,
        *map(str, stls),
    ], check=True)
    if any(blockers.values()):
        add_support_blockers(BUILD / out, blockers)
        subprocess.run([*run, "--slice", "0", "--outputdir", str(BUILD), "--export-3mf", out, str(BUILD / out)],
                       check=True)
    mark_modified(BUILD / out)
    print("wrote", BUILD / out)


def box_corners(lo, hi) -> list[np.ndarray]:
    """Corner i = 4*ix + 2*iy + iz, matching BOX_TRIANGLES."""
    return [np.array([(lo, hi)[ix][0], (lo, hi)[iy][1], (lo, hi)[iz][2]])
            for ix in (0, 1) for iy in (0, 1) for iz in (0, 1)]


BOX_TRIANGLES = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                 (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]  # outward normals


def add_support_blockers(path: Path, blockers: dict[str, list[np.ndarray]]) -> None:
    """Add support-blocker boxes (corners in STL coordinates, 8 per box) to the objects loaded from those STLs."""
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        files = {i.filename: z.read(i.filename) for i in infos}
    settings = files["Metadata/model_settings.config"].decode()
    model = files["3D/3dmodel.model"].decode()
    for stl_name, corners in blockers.items():
        if not corners:
            continue
        obj = next(o for o in ET.fromstring(settings).iter("object")
                   if any(m.get("value") == stl_name for m in o.findall("metadata")))
        obj_id = obj.get("id")
        matrix = obj.find("part").find("metadata[@key='matrix']").get("value")
        block = re.search(rf'<object id="{obj_id}"[^>]*>\s*<components>.*?</components>', model, re.S).group(0)
        comp = re.search(r'<component p:path="([^"]+)" objectid="(\d+)"[^>]*transform="([^"]+)"', block)
        sub_path, transform = comp.group(1), comp.group(3)
        offset = np.array([float(v) for v in transform.split()[9:12]])  # the CLI only translates the mesh
        sub_name = sub_path.lstrip("/")
        sub = files[sub_name].decode()
        new_id = max(int(i) for i in re.findall(r'<object id="(\d+)"', sub)) + 1
        meshes, components, parts = [], [], []
        for b in range(len(corners) // 8):
            box = [c - offset for c in corners[8 * b: 8 * b + 8]]
            vid = new_id + b
            verts = "".join(f'<vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}"/>' for v in box)
            tris = "".join(f'<triangle v1="{i}" v2="{j}" v3="{k}"/>' for i, j, k in BOX_TRIANGLES)
            meshes.append(f'<object id="{vid}" p:UUID="{uuid.uuid4()}" type="model"><mesh><vertices>{verts}'
                          f'</vertices><triangles>{tris}</triangles></mesh></object>')
            components.append(f'<component p:path="{sub_path}" objectid="{vid}" p:UUID="{uuid.uuid4()}" '
                              f'transform="{transform}"/>')
            parts.append(f'<part id="{vid}" subtype="support_blocker">'
                         f'<metadata key="name" value="support blocker {b + 1}"/>'
                         f'<metadata key="matrix" value="{matrix}"/></part>')
        files[sub_name] = sub.replace("</resources>", "".join(meshes) + "</resources>", 1).encode()
        model = model.replace(block, block.replace("</components>", "".join(components) + "</components>"), 1)
        start = settings.index(f'<object id="{obj_id}">')
        end = settings.index("</object>", start)
        settings = settings[:end] + "".join(parts) + settings[end:]
    files["3D/3dmodel.model"] = model.encode()
    files["Metadata/model_settings.config"] = settings.encode()
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, files[info.filename])
    tmp.replace(path)


def mark_modified(path: Path) -> None:
    """List the process overrides in the 3MF's `different_settings_to_system`.

    The CLI leaves it empty, and Bambu Studio's GUI then reloads every unlisted key from the system preset of the same
    name, silently dropping the overrides (and the supports) on the next "Slice plate". Entries: print, filament(s),
    printer.
    """
    member = "Metadata/project_settings.config"
    with zipfile.ZipFile(path) as z:
        items = [(info, z.read(info.filename)) for info in z.infolist()]
    tmp = path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info, data in items:
            if info.filename == member:
                cfg = json.loads(data)
                cfg["different_settings_to_system"][0] = ";".join(sorted(PROCESS_OVERRIDES))
                data = json.dumps(cfg, indent=4).encode()
            z.writestr(info, data)
    tmp.replace(path)


if __name__ == "__main__":
    main()
