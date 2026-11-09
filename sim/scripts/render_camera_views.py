"""Render what the overhead and wrist cameras show in both mounting orientations, for docs/camera_setup.md.

Arm at the ready pose, ghost balls along one bounce (blade to apex), overhead post 24 in in front of the base.
Pinhole approximation of the 102 x 59 deg lens (the real picture is wider and curved at the edges).

Run (from sim/): MUJOCO_GL=egl uv run scripts/render_camera_views.py   -> ../docs/camera_views.png
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from pingpong_sim.robot import READY_JOINT_POS, WRIST_CAM_POS, WRIST_CAM_QUAT, get_spec

OUT = Path(__file__).resolve().parents[2] / "docs" / "camera_views.png"
W, H, FOVY = 1280, 720, 59.0
TILT = math.radians(25)

spec = get_spec()
wb = spec.worldbody
spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=(0.55, 0.55, 0.6), rgb2=(0.45, 0.45, 0.5), width=512, height=512)
spec.add_material(name="grid", textures=["", "grid"], texrepeat=(8, 8))
wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1.0, 1.0, 0.01), material="grid", contype=0, conaffinity=0)
wb.add_light(pos=(0.3, 0.0, 1.5), dir=(0, 0, -1), diffuse=(0.8, 0.8, 0.8))
wb.add_light(pos=(0.8, 0.5, 1.0), dir=(-0.5, -0.3, -1), diffuse=(0.4, 0.4, 0.4))
# Overhead post (24 in in front of the base).
wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=(0.62, 0, 0.27), size=(0.019, 0.018, 0.27), rgba=(0.1, 0.1, 0.1, 1),
            contype=0, conaffinity=0)
# Ghost balls along one bounce (ball centre 0.27 m on the blade, apex 0.52 m), slightly offset in x to separate them.
for i, h in enumerate((0.0, 0.06, 0.13, 0.20, 0.25)):
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.02, 0, 0), pos=(0.40, 0.0, 0.27 + h),
                rgba=(1.0, 0.45, 0.0, 1.0 - 0.12 * i), contype=0, conaffinity=0)

# Overhead camera: optical axis 25 deg from straight down, toward the arm. MuJoCo cameras look along -z, +y up.
axis = np.array([-math.sin(TILT), 0.0, -math.cos(TILT)])
across = np.array([0.0, 1.0, 0.0])
along = np.cross(-axis, across)  # in the tilt plane, perpendicular to the axis (points toward the arm, upward)
for name, x_img in (("overhead_x", across), ("overhead_y", -along)):
    y_img = np.cross(-axis, x_img)
    wb.add_camera(name=name, pos=(0.58, 0.0, 0.55), xyaxes=(*x_img, *y_img), fovy=FOVY)

# Wrist camera: the model's (image up = blade normal) and the module turned 90 deg about its lens axis.
holder = spec.body("paddle_holder")
q90 = np.zeros(4)
mujoco.mju_axisAngle2Quat(q90, np.array([0.0, 0.0, 1.0]), math.pi / 2)
q_rolled = np.zeros(4)
mujoco.mju_mulQuat(q_rolled, np.array(WRIST_CAM_QUAT), q90)
holder.add_camera(name="wrist_y", pos=WRIST_CAM_POS, quat=q_rolled, fovy=FOVY)

m = spec.compile()
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
d = mujoco.MjData(m)
for j, v in READY_JOINT_POS.items():
    d.qpos[m.joint(j).qposadr[0]] = v
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, H, W)
views = [("overhead_x", "A  Overhead: arm comes in from the TOP edge  (avoid)"),
         ("overhead_y", "B  Overhead: arm comes in from the SIDE  (use this)"),
         ("wrist_cam", "A  Wrist: blade edge is a HORIZONTAL line  (avoid)"),
         ("wrist_y", "B  Wrist: blade edge is a VERTICAL line  (use this)")]
fig, axs = plt.subplots(2, 2, figsize=(16, 9.6))
for ax, (cam, title) in zip(axs.flat, views):
    r.update_scene(d, camera=cam)
    ax.imshow(r.render())
    ax.set_title(title, fontsize=12)
    ax.axis("off")
fig.suptitle("Camera previews (1280x720, pinhole approx. of the 102x59° lens). Orange: ball along one bounce, "
             "blade to apex (fading).", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=80)
print("wrote", OUT)
