"""Build the mjlab scene (SO-101 + paddle, ball) and check it: masses, actuators, ready pose, wrist-roll collision
sweep, renders.

Outputs in sim/build/ (git-ignored): inspect_{front,side,top}.png

Run (from sim/): MUJOCO_GL=egl uv run scripts/inspect_model.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mujoco
import numpy as np
from mjlab.scene import Scene, SceneCfg

from pingpong_sim.ball import get_ball_cfg, scene_spec_fn
from pingpong_sim.robot import ARM_JOINTS, READY_JOINT_POS, get_robot_cfg

BUILD = Path(__file__).resolve().parents[1] / "build"


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Menagerie <option> fields are not propagated by attach (set per task)
        scene = Scene(SceneCfg(entities={"robot": get_robot_cfg(), "ball": get_ball_cfg()}, spec_fn=scene_spec_fn),
                      device="cpu")
    m = scene.compile()
    d = mujoco.MjData(m)

    print("bodies:")
    for b in range(1, m.nbody):
        print(f"  {m.body(b).name:28s} {m.body_mass[b] * 1e3:7.1f} g")
    roll = m.body("robot/gripper").id
    paddle = m.body("robot/paddle").id
    print(f"wrist-roll payload {m.body_subtreemass[roll] * 1e3:.1f} g, arm total "
          f"{m.body_subtreemass[m.body('robot/base').id] * 1e3:.1f} g")
    print("actuators:", [(m.actuator(i).name, mujoco.mjtGain(m.actuator_gaintype[i]).name) for i in range(m.nu)])

    for j, q in READY_JOINT_POS.items():
        d.qpos[m.jnt_qposadr[m.joint(f"robot/{j}").id]] = q
    face = m.site("robot/blade_face_cap").id
    ball = m.joint("ball/ball_joint").id
    mujoco.mj_forward(m, d)
    p, n = d.site_xpos[face].copy(), d.site_xmat[face].reshape(3, 3)[:, 2].copy()
    d.qpos[m.jnt_qposadr[ball]:m.jnt_qposadr[ball] + 3] = p + [0, 0, 0.12]
    mujoco.mj_forward(m, d)
    com = d.xipos[paddle]
    print(f"ready pose: blade face centre {np.round(p, 4)} normal {np.round(n, 4)}; paddle CoM {np.round(com, 4)}")
    print(f"  contacts at ready pose: {d.ncon}")

    # Wrist-roll sweep at the ready pose: any blade contact with the arm's collision boxes?
    roll_q = m.jnt_qposadr[m.joint("robot/wrist_roll").id]
    lo, hi = m.jnt_range[m.joint("robot/wrist_roll").id]
    hits = []
    for q in np.linspace(lo, hi, 181):
        d.qpos[roll_q] = q
        mujoco.mj_forward(m, d)
        for c in d.contact[: d.ncon]:
            names = sorted(m.geom(g).name for g in (c.geom1, c.geom2))
            hits.append((round(float(np.degrees(q))), names, round(float(c.dist) * 1e3, 1)))
    print(f"  wrist-roll sweep {np.degrees(lo):.0f}..{np.degrees(hi):.0f} deg: {len(hits)} contacts", hits[:5])

    for j, q in READY_JOINT_POS.items():
        d.qpos[m.jnt_qposadr[m.joint(f"robot/{j}").id]] = q
    mujoco.mj_forward(m, d)
    render(m, d, p)


def render(m: mujoco.MjModel, d: mujoco.MjData, target: np.ndarray) -> None:
    BUILD.mkdir(exist_ok=True)
    import imageio.v3 as iio

    r = mujoco.Renderer(m, 720, 960)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.22, 0.0, 0.18]
    cam.distance = 0.95
    for name, az, el in [("front", 160, -15), ("side", 90, -5), ("top", 180, -80)]:
        cam.azimuth, cam.elevation = az, el
        r.update_scene(d, cam, opt)
        iio.imwrite(BUILD / f"inspect_{name}.png", r.render())
    r.close()
    print("wrote", ", ".join(f"build/inspect_{v}.png" for v in ("front", "side", "top")))


if __name__ == "__main__":
    main()
