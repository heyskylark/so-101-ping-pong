"""SO-101 follower with the ping pong paddle holder, wrist camera and paddle instead of the gripper.

Base model: MuJoCo Menagerie `robotstudio_so101` (vendored unmodified in assets/so101/, Menagerie commit 3abca2e),
edited here with MjSpec:

- The gripper servo, fixed jaw, moving jaw (+ `gripper` joint/actuator) and stock camera mount are removed. The
  `gripper` body stays: it is the wrist-roll link, its frame is the servo horn face.
- The wrist-roll link carries the holder halves, camera mount + camera, and a `paddle` child body. Their meshes are
  exported by cad/holder/export_sim_meshes.py in the holder frame, which sits in the `gripper` body at quat (0 1 0 0)
  exactly like the stock follower part (verified: the follower STL and STEP share a frame, and the holder frame is
  the follower frame shifted by (0, -0.22, 0.95) mm, which cancels the follower's offset in the body).

Holder frame: origin on the horn face, +z along the paddle (away from the wrist), +x = blade normal on the cap side,
y across the blade. At wrist_roll = 0 with the blade level the cap side faces up.

Masses are measured (1 g scale, 2026-11-08); paddle inertia is cad/paddle/derived.yaml.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from pingpong_sim.servo_model import add_gear_joints

ASSETS = Path(__file__).resolve().parent / "assets"
SO101_XML = ASSETS / "so101" / "so101.xml"
PADDLE_MESHES = ASSETS / "paddle"

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Holder frame in the `gripper` body (see module docstring): x_g = (x_h, -y_h, -z_h) + (0, 2 um, 0).
HOLDER_POS = (0.0, 2e-6, 0.0)
HOLDER_QUAT = (0.0, 1.0, 0.0, 0.0)

# Measured masses [kg].
HOLDER_BASE_MASS = 0.035  # incl. inserts and clamp screws
HOLDER_CAP_MASS = 0.025
CAMERA_ASSEMBLY_MASS = 0.022  # printed mount + UVC module + screws/washers, weighed together
CAMERA_MODULE_MASS = 0.010  # split of the 22 g: assumption (module ~10 g, mount + hardware ~12 g)

# Wrist camera (InnoMaker U20CAM-720P, the 32x32 UVC module): optical centre 20 mm out along the lens axis from the
# module's mesh origin, in the holder frame (cad/holder/preview_on_arm.py CAM_MOUNT, CAMERA_POS/QUAT). The lens
# looks along the paddle and 25 deg inward, at the blade centre; MuJoCo camera frame: looks along -z, image up (+y)
# = the cap-side blade normal. Which image axis is the module's long one is a camera_sim setting.
WRIST_CAM_POS = (0.0041, -0.06495, 0.01154)
WRIST_CAM_QUAT = (0.15321, 0.690309, 0.69031, -0.153207)

# Paddle (NIBIRU SPORT): 151 g, balance point 124.6 mm from the butt, butt 9 mm from the horn face.
# Inertia about the CoM from the 3-material fit (derived.yaml), paddle axes (x along, y width, z thickness)
# mapped to the holder frame (z along, y width, x thickness).
PADDLE_MASS = 0.151
PADDLE_COM = (0.0, 0.0, 0.009 + 0.1246)
PADDLE_DIAGINERTIA = (9.1786e-4, 7.7668e-4, 1.4833e-4)

# Blade: rubber surface +-5.3 mm from the mid-plane (6.6 mm core + 2 x 2.0 mm rubber); face centroid 171.6 mm
# from the butt (derived.yaml).
BLADE_HALF_THICKNESS = 0.0053
BLADE_CENTER_Z = 0.009 + 0.1716

# Servos: Feetech STS3215 12 V (ST-3215-C018) on a 12 V supply. The training model is the fitted servo model
# (servo_model.py, servo_actuator.py; fit to the arm's own recording). SERVO_PARAMS is the pre-fit starting point:
# BAM's 7.4 V STS3215 m6 params scaled to the 12 V datasheet (scripts/make_sts3215_12v_params.py).
# Firmware gains written by the ping-pong deploy code at startup: P per joint as below, I = 0, D = 0, Acceleration 0.
# Shoulder lift and elbow: 32 since 2026-11-12 (64 before, as in the recordings the servo model was fitted to). At
# 64 they hunted at ~3 Hz while moving (goal - encoder 1.4 deg peak to peak, 0.3 at 32) and the elbow swung 1.9 deg
# under a wrist stroke's reaction (1.1 at 32); they hang 0.1-0.3 deg lower (scripts/tune_hold.py).
SERVO_PARAMS = ASSETS / "actuators" / "sts3215_12v_m6.json"
FIRMWARE_P = {"shoulder_pan": 16, "shoulder_lift": 32, "elbow_flex": 32, "wrist_flex": 32, "wrist_roll": 64}

# Ready pose: cap-side face level and up, blade centre READY_X in front of the base axis and READY_Z above the base
# mounting face, paddle pointing straight ahead (IK: real.arm.ArmModel.ik). wrist_roll is 2.8 deg off zero because
# Menagerie's roll zero is the stock follower's calibration. Close in and high (upper arm 22 deg from vertical,
# forearm in line with it, wrist bent 60 deg) since 2026-11-12: holding the paddle there loads shoulder lift with
# 0.17 N m and the elbow with 0.63, against 0.67 and 0.90 at the old 0.40 / 0.25 m pose, where two servos' gears
# wore out. Everything that takes the bounce point (bouncer, controller, task) defaults to it.
READY_X, READY_Z = 0.28, 0.31
READY_JOINT_POS = {
    "shoulder_pan": -0.0006,
    "shoulder_lift": -1.1826,
    "elbow_flex": 0.1331,
    "wrist_flex": 1.0496,
    "wrist_roll": 0.0487,
}

# Removed with the gripper.
_REMOVED_BODIES = ("moving_jaw_so101_v1", "camera_mount")
_REMOVED_ACTUATORS = ("gripper",)


def _mesh(spec: mujoco.MjSpec, name: str) -> str:
    spec.add_mesh(name=name, file=str(PADDLE_MESHES / f"{name}.stl"), scale=(1e-3, 1e-3, 1e-3))
    return name


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(SO101_XML))

    for name in _REMOVED_ACTUATORS:
        spec.delete(spec.actuator(name))
    for name in _REMOVED_BODIES:
        spec.delete(spec.body(name))
    wrist_roll = spec.body("gripper")
    for geom in list(wrist_roll.geoms):
        spec.delete(geom)
    for site in list(wrist_roll.sites):
        spec.delete(site)
    # Mass now comes from the payload geoms (holder halves, camera) and the paddle body.
    wrist_roll.explicitinertial = False
    wrist_roll.mass = 0.0
    wrist_roll.inertia = (0.0, 0.0, 0.0)
    # Meshes of the removed gripper parts and stock camera mount. Menagerie meshes are unnamed in the spec; MuJoCo
    # names them after the file stem at compile time.
    used = {g.meshname for g in spec.geoms if g.meshname}
    for mesh in [m for m in spec.meshes if (m.name or Path(m.file).stem) not in used]:
        spec.delete(mesh)

    holder = wrist_roll.add_body(name="paddle_holder", pos=HOLDER_POS, quat=HOLDER_QUAT)
    visual = dict(type=mujoco.mjtGeom.mjGEOM_MESH, contype=0, conaffinity=0, group=2)
    yellow = (1.0, 0.82, 0.12, 1.0)
    dark = (0.1, 0.1, 0.1, 1.0)
    holder.add_geom(name="holder_base", meshname=_mesh(spec, "holder_base"), mass=HOLDER_BASE_MASS, rgba=yellow, **visual)
    holder.add_geom(name="holder_cap", meshname=_mesh(spec, "holder_cap"), mass=HOLDER_CAP_MASS, rgba=yellow, **visual)
    holder.add_geom(name="camera_mount", meshname=_mesh(spec, "camera_mount"),
                    mass=CAMERA_ASSEMBLY_MASS - CAMERA_MODULE_MASS, rgba=yellow, **visual)
    holder.add_geom(name="camera", meshname=_mesh(spec, "camera"), mass=CAMERA_MODULE_MASS, rgba=dark, **visual)
    holder.add_camera(name="wrist_cam", pos=WRIST_CAM_POS, quat=WRIST_CAM_QUAT, fovy=59.0)

    paddle = holder.add_body(name="paddle", pos=(0, 0, 0))
    paddle.explicitinertial = True
    paddle.mass = PADDLE_MASS
    paddle.ipos = PADDLE_COM
    paddle.inertia = PADDLE_DIAGINERTIA
    paddle.add_geom(name="paddle_visual", meshname=_mesh(spec, "paddle"), mass=0.0, rgba=(0.75, 0.1, 0.1, 1.0), **visual)
    paddle.add_geom(
        name="blade_collision",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=_mesh(spec, "blade_collision"),
        mass=0.0,
        group=3,
        contype=1,
        conaffinity=1,
        condim=3,
        rgba=(0.9, 0.2, 0.2, 0.4),
    )
    # Face sites: z axis = outward face normal, on the rubber surface at the face centroid. +x is the cap side,
    # which faces up at wrist_roll = 0 with the blade level.
    paddle.add_site(name="blade_face_cap", pos=(BLADE_HALF_THICKNESS, 0, BLADE_CENTER_Z), quat=_quat_z_to((1, 0, 0)), group=3)
    paddle.add_site(name="blade_face_base", pos=(-BLADE_HALF_THICKNESS, 0, BLADE_CENTER_Z), quat=_quat_z_to((-1, 0, 0)), group=3)
    # Compliant gear joint in series with each servo joint (servo_model.py). Part of the robot model so mjlab
    # indexes it as an entity joint; the servo model sets its stiffness per world (rigid until it does).
    add_gear_joints(spec, ARM_JOINTS)
    return spec


def joint_inertia(m: mujoco.MjModel, prefix: str = "robot/") -> list[float]:
    """Inertia [kg m^2] the arm presents to each servo joint at the ready pose (ARM_JOINTS order)."""
    d = mujoco.MjData(m)
    for name, q in READY_JOINT_POS.items():
        d.qpos[m.joint(prefix + name).qposadr[0]] = q
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    dofs = [m.joint(prefix + name).dofadr[0] for name in ARM_JOINTS]
    return [float(M[i, i]) for i in dofs]


def _quat_z_to(direction) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) rotating +z onto `direction`."""
    z = np.array([0.0, 0.0, 1.0])
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    axis = np.cross(z, d)
    s, c = np.linalg.norm(axis), float(z @ d)
    if s < 1e-12:
        return (1.0, 0.0, 0.0, 0.0) if c > 0 else (0.0, 1.0, 0.0, 0.0)
    half = np.arctan2(s, c) / 2
    axis /= s
    return (float(np.cos(half)), *(float(v) for v in np.sin(half) * axis))


def get_fixed_blade_spec(tilt_deg: float = 0.0) -> mujoco.MjSpec:
    """Test fixture: the blade collision mesh fixed in the world, cap face up, face centre at the origin, tilted by
    `tilt_deg` about the world x axis (the paddle's long axis lies along x)."""
    spec = mujoco.MjSpec()
    spec.add_mesh(name="blade_collision", file=str(PADDLE_MESHES / "blade_collision.stl"), scale=(1e-3,) * 3)
    # Holder frame -> world: -90 deg about y maps holder +x (cap face normal) to world +z, then the tilt about x.
    q0 = np.array([np.sqrt(0.5), 0, -np.sqrt(0.5), 0])
    half = np.radians(tilt_deg) / 2
    q_tilt = np.array([np.cos(half), np.sin(half), 0, 0])
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, q_tilt, q0)
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, np.array([BLADE_HALF_THICKNESS, 0, BLADE_CENTER_Z]), quat)
    body = spec.worldbody.add_body(name="blade", pos=-pos, quat=quat)
    body.add_geom(name="blade_collision", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="blade_collision", condim=3)
    return spec


def get_robot_cfg(randomize: bool = False):
    """The arm with the fitted servo model; `randomize` draws a different servo per world at each reset."""
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    from pingpong_sim.servo_actuator import ServoActuatorCfg

    servos = ServoActuatorCfg(target_names_expr=ARM_JOINTS, firmware_p=dict(FIRMWARE_P), randomize=randomize)
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(joint_pos={**READY_JOINT_POS, ".*_gear": 0.0}, joint_vel={".*": 0.0}),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=(servos,)),
    )
