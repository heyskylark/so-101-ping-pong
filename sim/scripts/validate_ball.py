"""Replay the drop experiments (measurements/ball/) in MuJoCo Warp with the final ball model and compare.

1. Centre drops from the filmed release heights (4-25.25 in) onto the fixed level blade, with drag: e per impact
   (all rebounds) vs the measured e(v) fit, and the first impact speed of the 25.25 in drop (measured 3.40 m/s).
2. Drops onto the blade tilted 16.5 deg (the angled videos): normal e and the kept tangential speed fraction
   (measured: 0.54-0.58 and 0.37-0.41).

Run (from sim/): uv run scripts/validate_ball.py
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from pingpong_sim.ball import BALL_RADIUS, SIM_DT, add_air, add_paddle_contact, get_ball_cfg
from pingpong_sim.robot import get_fixed_blade_spec

INCH = 0.0254
HEIGHTS_IN = (4, 8, 12, 16, 19, 25.25)
REPLICATES = 8
G = 9.81


def measured_e(v):
    return 0.578 + 0.373 * np.exp(-v / 0.54)


def simulate(tilt_deg: float, start: np.ndarray, seconds: float):
    """Ball released at rest from `start` (n, 3) above the blade; returns ball positions and velocities."""
    n = len(start)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def scene_spec(spec):
        add_air(spec)
        add_paddle_contact(spec, "blade/blade_collision", "blade/blade")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene = Scene(SceneCfg(num_envs=n, entities={"blade": EntityCfg(spec_fn=lambda: get_fixed_blade_spec(tilt_deg)),
                                                     "ball": get_ball_cfg()}, spec_fn=scene_spec), device)
    m = scene.compile()
    sim = Simulation(num_envs=n, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=SIM_DT, integrator="euler",
                                                                     cone="elliptic", impratio=10.0)),
                     model=m, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    sim.reset()
    scene.reset()
    root = torch.zeros(n, 13, device=device)
    root[:, :3] = torch.tensor(start, device=device, dtype=torch.float32)
    root[:, 3] = 1.0
    scene["ball"].write_root_state_to_sim(root)
    sim.forward()
    jnt = m.joint("ball/ball_joint").id
    qa, va = int(m.jnt_qposadr[jnt]), int(m.jnt_dofadr[jnt])
    ps, vs = [], []
    for _ in range(int(seconds / SIM_DT)):
        sim.step()
        ps.append(sim.data.qpos[:, qa:qa + 3].clone())
        vs.append(sim.data.qvel[:, va:va + 3].clone())
    return torch.stack(ps).cpu().numpy(), torch.stack(vs).cpu().numpy()


def impacts(dist: np.ndarray, vn: np.ndarray, vt: np.ndarray):
    """Split a track into contacts (dist < 0, ball bottom below the face plane) and return, per contact, the normal
    and tangential speeds just before and after, extrapolated ballistically to the face plane."""
    out = []
    touching = dist < 0
    k = 1
    while k < len(dist):
        if touching[k] and not touching[k - 1]:
            start = k - 1
            end = k
            while end < len(dist) and touching[end]:
                end += 1
            if end >= len(dist):
                break
            vin = np.sqrt(vn[start] ** 2 + 2 * G * dist[start])
            vout = np.sqrt(max(vn[end] ** 2 + 2 * G * dist[end], 0.0)) if vn[end] > 0 else 0.0
            out.append((vin, vout, vt[start], vt[end]))
            k = end
        k += 1
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    heights = np.repeat(np.array(HEIGHTS_IN) * INCH, REPLICATES)
    start = np.zeros((len(heights), 3))
    start[:, 0] = rng.uniform(-0.01, 0.01, len(heights))
    start[:, 2] = BALL_RADIUS + heights + rng.uniform(0, 0.003, len(heights))
    p, v = simulate(0.0, start, 2.5)
    rows = []
    first_25 = []
    for i, h in enumerate(heights):
        hits = impacts(p[:, i, 2] - BALL_RADIUS, v[:, i, 2], v[:, i, 0])
        rows += [(vin, vout / vin) for vin, vout, _, _ in hits if vin > 0.2]
        if h == 25.25 * INCH and hits:
            first_25.append(hits[0][0])
    rows = np.array(rows)
    print("centre drops, level blade (sim with drag vs measured fit):")
    print("  v_in bin   n   sim e          measured fit")
    for lo, hi in [(0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.6)]:
        sel = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if sel.any():
            vm = rows[sel, 0].mean()
            print(f"  {lo:.1f}-{hi:.1f} {sel.sum():4d}   {rows[sel, 1].mean():.3f}±{rows[sel, 1].std():.3f}   "
                  f"{measured_e(vm):.3f} at {vm:.2f} m/s")
    print(f"  25.25 in first impact: {np.mean(first_25):.2f} m/s (measured 3.40; no-drag {np.sqrt(2 * G * 25.25 * INCH):.2f})")

    tilt = 16.5
    n_t = 16
    start = np.zeros((n_t, 3))
    start[:, 2] = BALL_RADIUS / np.cos(np.radians(tilt)) + 12 * INCH + rng.uniform(0, 0.003, n_t)
    p, v = simulate(tilt, start, 0.6)
    normal = np.array([0.0, -np.sin(np.radians(tilt)), np.cos(np.radians(tilt))])
    tangent = np.array([0.0, np.cos(np.radians(tilt)), np.sin(np.radians(tilt))])
    res = []
    for i in range(n_t):
        dist = p[:, i] @ normal - BALL_RADIUS
        hits = impacts(dist, v[:, i] @ normal, v[:, i] @ tangent)
        if hits:
            vin, vout, tin, tout = hits[0]
            res.append((vout / vin, tout / tin))
    res = np.array(res)
    print(f"blade tilted {tilt} deg, 12 in drop: normal e {res[:, 0].mean():.3f}±{res[:, 0].std():.3f} (measured 0.54-0.58), "
          f"tangential kept {res[:, 1].mean():.3f}±{res[:, 1].std():.3f} (measured 0.37-0.41, +-0.1)")


if __name__ == "__main__":
    main()
