"""Identify a linear tracking model of the servoed arm for feedforward: q'' = A (u - q) - B q' + c.

u is the goal vector sent to the servos, q the encoder angles (5 joints); A, B are 5x5 (joints couple through the
arm's inertia), c is the gravity term. Drives the fitted (nominal) servo model with smooth random multi-sine goals
on all joints at once around the ready pose and fits A, B, c by least squares. The bouncer inverts it:
u = q_d + A^-1 (q_d'' + B q_d' - c).

The servos are nonlinear (current limit, stiction, rate-limited target), so R^2 is ~0.7 on lift/elbow/wrist_flex;
the model mainly cancels the 3-4 Hz resonance of the P-servoed arm (wn ~27 rad/s, zeta ~0.4) that a bounce
stroke excites.

Run (from sim/): uv run scripts/identify_tracking.py   -> src/pingpong_sim/assets/actuators/tracking_model.json
"""

from __future__ import annotations

import json

import numpy as np
import torch

from pingpong_sim.ball import SIM_DT
from pingpong_sim.harness import make_harness, step
from pingpong_sim.robot import ARM_JOINTS, ASSETS, READY_JOINT_POS
from pingpong_sim.servo_actuator import encoder_pos

SECONDS = 20.0
AMP = 0.12  # rad per joint
FREQS = (0.4, 0.9, 1.7, 2.6, 3.8)  # Hz
WORLDS = 8
OUT = ASSETS / "actuators" / "tracking_model.json"


def main() -> None:
    torch.set_grad_enabled(False)
    h = make_harness(num_envs=WORLDS)
    dev = h.sim.device
    ids, _ = h.robot.find_joints(ARM_JOINTS, preserve_order=True)
    q_ready = torch.tensor([READY_JOINT_POS[j] for j in ARM_JOINTS], device=dev)
    root = torch.zeros(WORLDS, 13, device=dev)  # park the ball
    root[:, :3] = torch.tensor([1.0, 1.0, 0.02])
    root[:, 3] = 1.0
    h.ball.write_root_state_to_sim(root)
    q0 = q_ready.repeat(WORLDS, 1)
    h.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), joint_ids=ids)
    h.robot.reset()

    rng = np.random.default_rng(0)
    phase = rng.uniform(0, 2 * np.pi, (WORLDS, 5, len(FREQS)))
    steps = int(SECONDS / SIM_DT)
    t = np.arange(steps) * SIM_DT
    ramp = np.clip(t / 1.0, 0, 1)[:, None, None]
    u = sum(np.sin(2 * np.pi * f * t[:, None, None] + phase[None, :, :, i]) for i, f in enumerate(FREQS))
    u = u * AMP / len(FREQS) * 2 * ramp + q_ready.cpu().numpy()
    q = np.zeros_like(u)
    ut = torch.tensor(u, dtype=torch.float32, device=dev)
    for k in range(steps):
        h.robot.set_joint_position_target(ut[k], joint_ids=ids)
        step(h)
        q[k] = encoder_pos(h.robot).cpu().numpy()

    qd = np.gradient(q, SIM_DT, axis=0)
    qdd = np.gradient(qd, SIM_DT, axis=0)
    sl = slice(int(2.0 / SIM_DT), steps - 1)
    e, v, y = (x[sl].reshape(-1, 5) for x in (u - q, qd, qdd))
    X = np.concatenate([e, -v, np.ones((len(e), 1))], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    A, B, c = coef[:5].T, coef[5:10].T, coef[10]
    r2 = 1 - np.var(y - X @ coef, axis=0) / np.var(y, axis=0)
    sag = np.linalg.solve(A, c)
    OUT.write_text(json.dumps(dict(joints=list(ARM_JOINTS), A=A.tolist(), B=B.tolist(), c=c.tolist(),
                                   r2=r2.round(3).tolist(), ready_sag_rad=sag.tolist(),
                                   source="scripts/identify_tracking.py, fitted servo model, nominal"), indent=1) + "\n")
    wn = np.sqrt(np.diag(A))
    print("R^2 per joint", r2.round(3))
    print("diagonal wn [rad/s]", wn.round(1), " zeta", (np.diag(B) / (2 * wn)).round(2))
    print("sag at the ready pose [deg]", np.degrees(sag).round(2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
