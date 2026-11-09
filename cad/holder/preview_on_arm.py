# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["build123d", "numpy", "pyvista"]
# ///
"""Place the paddle holder (+ paddle, wrist camera mount, camera) on the upstream SO-101 assembly, replacing the
stock gripper, check wrist-roll clearance, and render.

Outputs in cad/holder/build/ (git-ignored):
  so101_paddle.step         whole arm with holder + paddle + camera mount, for Onshape / any CAD viewer
  so101_paddle_{view}.png   offscreen renders

The roll sweep turns everything on the wrist-roll horn (holder, camera mount, camera) a full turn about the roll
axis and reports the smallest gap to the rest of the arm in the upstream assembly pose.

Run: uv run cad/holder/preview_on_arm.py
"""

from __future__ import annotations

import csv
import math
import urllib.request
from pathlib import Path

import numpy as np
from build123d import (
    Axis,
    Compound,
    GeomType,
    Location,
    Plane,
    Polygon,
    RectangleRounded,
    Vector,
    export_step,
    export_stl,
    extrude,
    import_step,
    loft,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUILD = HERE / "build"
CACHE = REPO / "cad" / ".cache"
UPSTREAM = "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/"
FILES = {
    "assembly": "STEP/SO101/SO101%20Assembly.step",
    "follower": "STEP/SO101/Follower_Specific/Wrist_Roll_Follower_SO101.step",
    "camera_mount": "Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module/stl/SO-ARM101_camera_wrist_mount.step",
    "camera": "Simulation/SO101/assets/wrist_camera_so101_v1.stl",
}

# Camera mount frame -> holder frame (see paddle_holder.fs): flange face x=0 on y=-24, its screw axes
# (y=-8.15, z=10 / 18.1) onto the holder's at z=23.4, x=-3.4 / 4.7.
CAM_MOUNT = Location(Plane(origin=(-13.4, -24.0, 31.55), x_dir=(0, 1, 0), z_dir=(1, 0, 0)))
# Camera module (32x32 UVC, lens along +z) in the mount frame, from upstream so101_new_calib_camera.xml.
CAMERA_POS = (-49.41, -38.13, 17.5)
CAMERA_QUAT = (0.37978693649389456, -0.5964584309361936, -0.37978833153177, -0.596456240026065)

# Handle stations [x from butt, width, thickness] (mm) -- same table as paddle_holder.fs / measurements.yaml.
HANDLE = [[0, 33.88, 24.6], [4, 33.88, 24.6], [23, 30.5, 24.0], [43, 27.18, 23.6], [63, 26.8, 23.5], [83, 29.5, 23.5]]
BUTT_Z = 9.0
BLADE_THICKNESS = 6.6 + 2 * 2.0


def fetch(name: str) -> Path:
    path = CACHE / Path(FILES[name]).name.replace("%20", " ")
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(UPSTREAM + FILES[name], path)
    return path


def follower_location(follower_global, follower_local) -> Location:
    """Pose of the stock follower in the assembly, from its horn recess and an asymmetric Ø10 hole."""
    circles = [e for e in follower_global.edges() if e.geom_type == GeomType.CIRCLE]
    rim = sorted({tuple(round(c, 3) for c in e.arc_center) for e in circles if abs(e.radius - 10) < 0.01})
    a, b = Vector(*rim[0]), Vector(*rim[-1])
    u = (b - a).normalized()
    if (follower_global.bounding_box().center() - a).dot(u) < 0:
        u = -u
    bottom = a if (a - b).dot(u) < 0 else b  # local (0, -0.22, -0.05)
    hole = next(e.arc_center for e in circles if abs(e.radius - 5) < 0.01 and abs((e.arc_center - bottom).dot(u) - 6.0) < 0.2)
    v = hole - bottom
    w = (v - u * v.dot(u)).normalized()
    th = math.atan2(-16.0, -10.71)  # local in-plane direction of that hole from the horn axis
    ex = w * math.cos(th) - u.cross(w) * math.sin(th)
    origin = bottom + u.cross(ex) * 0.22 + u * 0.05
    loc = Location(Plane(origin=origin, x_dir=ex, z_dir=u))
    err = (follower_local.moved(loc).bounding_box().min - follower_global.bounding_box().min).length
    if err > 0.1:
        raise RuntimeError(f"follower pose fit is off by {err:.3f} mm")
    return loc


def paddle_solid():
    """Paddle in the holder frame: handle along +z from the pocket floor, blade normal along x."""
    sections = [RectangleRounded(t, w, 8).moved(Location((0, 0, BUTT_Z + x))) for x, w, t in HANDLE]
    handle = loft(sections)
    with open(REPO / "cad" / "paddle" / "outline.csv") as f:
        pts = [(float(y), BUTT_Z + float(x)) for x, y in list(csv.reader(f))[1:]]
    blade = extrude(Plane.YZ * Polygon(*pts, align=None), amount=BLADE_THICKNESS / 2, both=True)
    return handle + blade


def matrix(loc: Location) -> np.ndarray:
    t = loc.wrapped.Transformation()
    return np.array([[t.Value(i, j) for j in range(1, 5)] for i in range(1, 4)] + [[0, 0, 0, 1]])


def quat_matrix(pos, q) -> np.ndarray:
    w, x, y, z = q
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = pos
    return m


def main() -> None:
    import pyvista as pv

    BUILD.mkdir(exist_ok=True)
    solids = import_step(fetch("assembly")).solids()
    follower_local = import_step(fetch("follower"))
    follower = next(s for s in solids if abs(s.volume - follower_local.volume) < 0.01 * follower_local.volume)
    holder_loc = follower_location(follower, follower_local) * Location((0, -0.22, 0.95))

    # Drop the stock gripper: the follower and everything mounted beyond the horn face, including the gripper
    # servo's cable, which starts in the wrist but runs out past the horn face.
    horn = holder_loc.position
    axis_dir = holder_loc * Location((0, 0, 1))
    u = (axis_dir.position - horn).normalized()
    to_holder = holder_loc.inverse()
    arm = [s for s in solids if s is not follower and (s.bounding_box().center() - horn).dot(u) < 1.0
           and s.moved(to_holder).bounding_box().max.Z < 10]
    holder = [import_step(HERE / f"paddle_holder_{p}.step").moved(holder_loc) for p in ("base", "cap")]
    mount = import_step(fetch("camera_mount")).moved(holder_loc * CAM_MOUNT)
    paddle = paddle_solid().moved(holder_loc)
    print(f"arm parts kept: {len(arm)}, removed: {len(solids) - len(arm)}")
    # Fit check in the holder frame: the mount must only touch the seat.
    mount_h = import_step(fetch("camera_mount")).moved(CAM_MOUNT)
    for p in ("base", "cap"):
        hit = mount_h & import_step(HERE / f"paddle_holder_{p}.step")  # None when empty
        if hit is not None and hit.volume > 1e-3:
            bb = hit.bounding_box()
            print(f"camera mount overlaps {p}: {hit.volume:.3f} mm^3 in {bb.min} .. {bb.max}")

    export_step(Compound(arm + holder + [mount, paddle]), BUILD / "so101_paddle.step")
    groups = {"arm": Compound(arm), "holder": Compound(holder), "mount": mount, "paddle": paddle}
    meshes = {name: polydata(shape) for name, shape in groups.items()}
    camera = pv.read(fetch("camera")).scale(1000)
    meshes["camera"] = camera.transform(matrix(holder_loc * CAM_MOUNT) @ quat_matrix(CAMERA_POS, CAMERA_QUAT), inplace=False)

    roll_sweep(arm, meshes["holder"] + meshes["mount"] + meshes["camera"], horn, u)
    render(meshes, horn, u)


def polydata(shape, tolerance: float = 0.05):
    import pyvista as pv

    path = BUILD / "_tmp.stl"
    export_stl(shape, path, tolerance=tolerance, angular_tolerance=0.2)
    mesh = pv.read(path)
    path.unlink()
    return mesh


def roll_sweep(arm: list, moving, horn: Vector, roll_axis: Vector) -> None:
    """Smallest signed gap per arm part (mesh vertices vs surfaces, both ways) over a full turn of the wrist roll.

    Angles are from the upstream assembly pose. The servo horn and its screws turn with the holder, so the horn zone
    (r < 12.5 mm, from 6 mm behind the horn face to the pocket floor) is left out on both sides.
    """
    import pyvista as pv

    o, k = np.array(tuple(horn)), np.array(tuple(roll_axis))

    def cyl(p):  # (radius, axial) about the roll axis
        a = (p - o) @ k
        return np.linalg.norm(p - o - np.outer(a, k), axis=1), a

    def outside_horn(p):
        r, a = cyl(p)
        return ~((r < 12.5) & (a > -6) & (a < BUTT_Z))

    r_m, a_m = cyl(moving.points)
    reach = (r_m.max() + 2, a_m.min() - 2, a_m.max() + 2)
    moving_pts = moving.points[outside_horn(moving.points)]
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rotations = {deg: np.eye(3) + math.sin(math.radians(deg)) * K + (1 - math.cos(math.radians(deg))) * K @ K
                 for deg in range(0, 360, 3)}
    results = []
    for i, solid in enumerate(arm):
        if (solid.bounding_box().center() - horn).length > 200:
            continue
        part = polydata(solid, 0.1)
        r, a = cyl(part.points)
        near = (r < reach[0]) & (a > reach[1]) & (a < reach[2]) & outside_horn(part.points)
        if not near.any():
            continue
        part_pts = pv.PolyData(part.points[near])
        worst = (np.inf, 0)
        for deg, R in rotations.items():
            turned = moving.copy()
            turned.points = (moving.points - o) @ R.T + o
            g1 = pv.PolyData((moving_pts - o) @ R.T + o).compute_implicit_distance(part)["implicit_distance"].min()
            g2 = part_pts.compute_implicit_distance(turned)["implicit_distance"].min()
            worst = min(worst, (min(g1, g2), deg))
        c = solid.bounding_box().center() - horn
        results.append((worst[0], worst[1], i, c.dot(Vector(*k)), solid.volume))
    results.sort()
    print("roll sweep, closest arm parts (gap mm, at deg from assembly pose, part #, axial offset of part centre, volume):")
    for gap, deg, i, ax, vol in results[:6]:
        print(f"  {gap:6.2f} mm at {deg:3d} deg  part {i:3d}  axial {ax:6.1f}  vol {vol:8.0f}")


def render(meshes: dict, horn: Vector, roll_axis: Vector) -> None:
    import pyvista as pv

    pv.OFF_SCREEN = True
    colors = {"arm": "#f2c94c", "holder": "#2d9cdb", "mount": "#27ae60", "camera": "#333333", "paddle": "#c0392b"}
    focus = np.array(tuple(horn)) + 60 * np.array(tuple(roll_axis))
    for view, (az, el) in {"iso": (35, 25), "side": (0, 5), "wrist": (10, 15)}.items():
        p = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        for n, m in meshes.items():
            p.add_mesh(m, color=colors[n], smooth_shading=True, split_sharp_edges=True, specular=0.2)
        p.set_background("white")
        bounds = np.array(meshes["arm"].merge(meshes["paddle"]).bounds).reshape(3, 2)
        center = bounds.mean(1) if view != "wrist" else focus
        dist = 2.2 * np.ptp(bounds, axis=1).max() if view != "wrist" else 420
        d = np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)),
                      math.cos(math.radians(el)) * math.sin(math.radians(az)),
                      math.sin(math.radians(el))])
        p.camera_position = [tuple(center + dist * d), tuple(center), (0, 0, 1)]
        p.screenshot(BUILD / f"so101_paddle_{view}.png")
        p.close()
        print("wrote", BUILD / f"so101_paddle_{view}.png")


if __name__ == "__main__":
    main()
