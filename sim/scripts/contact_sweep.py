"""Ball-on-blade restitution in MuJoCo Warp vs contact parameters, timestep and impact speed.

A static blade (the paddle's blade collision mesh, face up) and the ball; one world per (solref, speed, replicate).
The ball starts above the face with a downward speed; each replicate starts at a random height within one
timestep of travel, so the contact begins at a random phase of the step (as in a real rollout). e = vz_out / vz_in,
from the ball's ballistic speed back at the face after it leaves (no air drag here).

Usage (from sim/):
  uv run scripts/contact_sweep.py --dt 0.001 --timeconst 0.01 0.014 --dampratio 0.15 0.2
  uv run scripts/contact_sweep.py --dt 0.001 --direct --stiffness 2e4 5e4 --damping 10 20
"""

from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from pingpong_sim.ball import BALL_RADIUS, get_ball_cfg
from pingpong_sim.robot import get_fixed_blade_spec

SPEEDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)  # m/s at impact
REPLICATES = 16
GAP = 0.004  # m, minimum start height of the ball's bottom above the face


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--direct", action="store_true", help="solref as (-stiffness, -damping)")
    ap.add_argument("--timeconst", type=float, nargs="+", default=[0.01])
    ap.add_argument("--dampratio", type=float, nargs="+", default=[0.2])
    ap.add_argument("--stiffness", type=float, nargs="+", default=[2e4])
    ap.add_argument("--damping", type=float, nargs="+", default=[20.0])
    ap.add_argument("--solimp", type=float, nargs=5, default=[0.9, 0.95, 0.001, 0.5, 2.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.direct:
        refs = [(-k, -b) for k, b in itertools.product(args.stiffness, args.damping)]
    else:
        refs = list(itertools.product(args.timeconst, args.dampratio))
    combos = [(r, v) for r in refs for v in SPEEDS for _ in range(REPLICATES)]
    n = len(combos)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene = Scene(SceneCfg(num_envs=n, env_spacing=0.5,
                               entities={"blade": EntityCfg(spec_fn=get_fixed_blade_spec), "ball": get_ball_cfg()}), device)
    m = scene.compile()
    sim = Simulation(num_envs=n, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=args.dt, integrator="euler",
                                                                     cone="elliptic", impratio=10.0)),
                     model=m, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    sim.expand_model_fields(("geom_solref", "geom_solimp"))
    for g in (m.geom("ball/ball_geom").id, m.geom("blade/blade_collision").id):
        sim.model.geom_solref[:, g] = torch.tensor([c[0] for c in combos], device=device, dtype=torch.float32)
        sim.model.geom_solimp[:, g] = torch.tensor(args.solimp, device=device)
    sim.reset()
    scene.reset()

    speeds = np.array([v for _, v in combos])
    gap = GAP + np.random.default_rng(args.seed).uniform(0, 1, n) * speeds * args.dt
    root = torch.zeros(n, 13, device=device)
    root[:, 2] = torch.tensor(BALL_RADIUS + gap, device=device)
    root[:, 3] = 1.0
    root[:, 9] = torch.tensor(-np.sqrt(speeds**2 - 2 * 9.81 * gap), device=device)
    scene["ball"].write_root_state_to_sim(root)
    sim.forward()

    # Ball state straight from qpos/qvel (entity root_link_* is from the start of the step, one step behind).
    jnt = m.joint("ball/ball_joint").id
    zadr, vadr = int(m.jnt_qposadr[jnt]) + 2, int(m.jnt_dofadr[jnt]) + 2
    steps = int(0.06 / args.dt)
    zs, vs = [], []
    for _ in range(steps):
        sim.step()
        zs.append(sim.data.qpos[:, zadr].clone())
        vs.append(sim.data.qvel[:, vadr].clone())
    z = torch.stack(zs).cpu().numpy() - BALL_RADIUS
    vz = torch.stack(vs).cpu().numpy()

    e = np.full(n, np.nan)
    pen = -z.min(axis=0)
    duration = np.full(n, np.nan)
    for i in range(n):
        touching = np.nonzero(z[:, i] < 0)[0]
        if len(touching) == 0:
            continue
        after = np.nonzero((np.arange(steps) > touching[-1]) & (z[:, i] > 0) & (vz[:, i] > 0))[0]
        if len(after):
            e[i] = np.sqrt(vz[after[0], i] ** 2 + 2 * 9.81 * z[after[0], i]) / speeds[i]
            duration[i] = (touching[-1] - touching[0] + 1) * args.dt

    print(f"dt {args.dt * 1e3:.1f} ms, solimp {args.solimp}, {REPLICATES} random contact phases per cell")
    print("solref             " + "  ".join(f"{v:9.1f}" for v in SPEEDS)
          + "   impact m/s | penetration mm, contact ms")
    per = len(SPEEDS) * REPLICATES
    for i0 in range(0, n, per):
        cells = [slice(i0 + k * REPLICATES, i0 + (k + 1) * REPLICATES) for k in range(len(SPEEDS))]
        row = "  ".join(f"{np.nanmean(e[c]):.3f}±{np.nanstd(e[c]):.3f}" for c in cells)
        block = slice(i0, i0 + per)
        print(f"{str(tuple(round(x, 5) for x in combos[i0][0])):18s} {row}   "
              f"{np.nanmax(pen[block]) * 1e3:5.2f}, {np.nanmean(duration[block]) * 1e3:4.1f}")


if __name__ == "__main__":
    main()
