"""GPU check of the arm + servo model: hold the ready pose, then a 0.3 rad step on each joint.

Reports the error after a 1 s hold at the ready pose and, per joint, the step's 10-90 % rise time, peak speed,
overshoot and final value. With --randomize, every joint's step runs in REPEATS worlds with randomized servos
(servo_actuator.DEFAULT_RANDOMIZATION) and the spread is reported.

Run (from sim/): uv run scripts/servo_check.py [--randomize]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from pingpong_sim.ball import SIM_DT
from pingpong_sim.harness import make_harness, step
from pingpong_sim.robot import ARM_JOINTS, READY_JOINT_POS
from pingpong_sim.servo_actuator import encoder_pos

DT = SIM_DT
STEP = 0.3  # rad
HOLD_S, STEP_S = 1.0, 1.0
REPEATS = 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--randomize", action="store_true")
    args = ap.parse_args()
    J = len(ARM_JOINTS)
    reps = REPEATS if args.randomize else 1
    n = reps * (1 + J)  # world r*(1+J): holds; r*(1+J)+1+k: steps joint k
    h = make_harness(num_envs=n, dt=DT, randomize=args.randomize)
    dev = h.sim.device
    ids, names = h.robot.find_joints(ARM_JOINTS, preserve_order=True)
    ready = torch.tensor([READY_JOINT_POS[j] for j in names], device=dev).repeat(n, 1)

    root = torch.zeros(n, 13, device=dev)  # park the ball away from the arm
    root[:, :3] = torch.tensor([1.0, 1.0, 0.02], device=dev)
    root[:, 3] = 1.0
    h.ball.write_root_state_to_sim(root)
    h.robot.write_joint_state_to_sim(ready, torch.zeros_like(ready), joint_ids=ids)
    h.robot.set_joint_position_target(ready, joint_ids=ids)
    h.sim.forward()
    step(h, int(HOLD_S / DT))
    q = encoder_pos(h.robot)
    hold = np.degrees((q - ready).cpu().numpy().reshape(reps, 1 + J, J)[:, 0])
    print("ready-pose error after 1 s hold [deg], median (min..max) over worlds:")
    for k, j in enumerate(names):
        print(f"  {j:14s} {np.median(hold[:, k]):+6.2f} ({hold[:, k].min():+.2f}..{hold[:, k].max():+.2f})")

    target = ready.clone()
    for r in range(reps):
        for k in range(J):
            target[r * (1 + J) + 1 + k, k] += STEP
    h.robot.set_joint_position_target(target, joint_ids=ids)
    start = q.clone().cpu().numpy()
    traj = []
    for _ in range(int(STEP_S / DT)):
        step(h)
        traj.append(encoder_pos(h.robot).clone())
    traj = torch.stack(traj).cpu().numpy()
    t = np.arange(len(traj)) * DT
    print(f"{STEP} rad step response, median (min..max) over worlds:")
    for k, j in enumerate(names):
        rows = [r * (1 + J) + 1 + k for r in range(reps)]
        x = (traj[:, rows, k] - start[rows, k]) / STEP  # (T, reps)
        t10 = np.array([t[np.argmax(c >= 0.1)] for c in x.T])
        t90 = np.array([t[np.argmax(c >= 0.9)] if (c >= 0.9).any() else np.nan for c in x.T])
        rise, over = 1e3 * (t90 - t10), 100 * (x.max(axis=0) - 1)
        speed = np.abs(np.gradient(traj[:, rows, k], DT, axis=0)).max(axis=0)

        def fmt(v, f):
            return f"{np.nanmedian(v):{f}} ({np.nanmin(v):{f}}..{np.nanmax(v):{f}})"

        print(f"  {j:14s} rise10-90 {fmt(rise, '.0f')} ms  peak speed {fmt(speed, '.2f')} rad/s  "
              f"overshoot {fmt(over, '.1f')} %  final {fmt(100 * x[-1], '.1f')} %")


if __name__ == "__main__":
    main()
