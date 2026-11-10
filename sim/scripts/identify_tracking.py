"""Identify a linear tracking model of the servoed arm for feedforward: q''(t) = A (u(t - d) - q) - B q' + c.

u is the goal vector sent to the servos, q the encoder angles (5 joints); A, B are 5x5 (joints couple through the
arm's inertia), c is the gravity term, d a command delay. The bouncer inverts it:
u(t) = q_d + A^-1 (q_d'' + B q_d' - c), evaluated d ahead (its `lead`).

Two sources:
- default: drives the fitted (nominal) servo model with smooth random multi-sine goals on all joints at once
  around the ready pose (d = 0). The servos are nonlinear (current limit, stiction, rate-limited target), so R^2
  is ~0.7 on lift/elbow/wrist_flex; the model mainly cancels the 3-4 Hz resonance of the P-servoed arm (wn ~27
  rad/s, zeta ~0.4) that a bounce stroke excites.
- `--from <stroke recordings>` (`record_servos.py strokes`): the bouncer's own strokes on the arm. Fits A, B, c by
  least squares on smoothed derivatives for each delay on a grid, picks the delay whose model, simulated from the
  recorded goals, best reproduces held-out series, and compares with the model in use. Real bounces showed the
  paddle peaking at ~60 % of the commanded speed and 50-90 ms behind it: a regime the nominal model doesn't have.

Run (from sim/): uv run scripts/identify_tracking.py [--from DIR ...]
    -> src/pingpong_sim/assets/actuators/tracking_model.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def simulate(A, B, c, d_steps: int, t: np.ndarray, u: np.ndarray, q0: np.ndarray, qd0: np.ndarray) -> np.ndarray:
    """q(t) of the tracking model driven by goals u (N, 5) sampled at t (100 Hz), from (q0, qd0); goals delayed by
    d_steps samples (held at u[0] before the start); 10 Euler substeps per sample."""
    q, qd = q0.astype(float).copy(), qd0.astype(float).copy()
    out = np.zeros_like(u)
    for k in range(len(t)):
        out[k] = q
        if k + 1 == len(t):
            break
        uk = u[max(k - d_steps, 0)]
        h = (t[k + 1] - t[k]) / 10
        for _ in range(10):
            qdd = A @ (uk - q) - B @ qd + c
            qd = qd + qdd * h
            q = q + qd * h
    return out


def fit_recording(dirs: list[Path], out: Path = OUT) -> None:
    """Refit on stroke recordings: the model in use, with each joint's stiffness row (A, c) and damping row (B)
    scaled and the goals delayed, the scales fitted by simulating the model from the recorded goals (output error:
    strokes excite mostly the elbow, too little for a full 5x5 equation-error fit, which went unstable), the delay
    on a 10 ms grid; held-out series decide, and the model in use is kept unless the refit predicts them better."""
    from scipy.optimize import least_squares

    series = []  # (t, u, q) per stroke series
    for d in dirs:
        z = np.load(d / "log.npz")
        meta = json.loads((d / "meta.json").read_text())
        t, u, q, seg = z["t_read"], z["goal"], z["pos"], z["seg"]
        for i in range(len(meta["segments"])):
            k = seg == i
            if k.sum() > 50:
                series.append((t[k] - t[k][0], u[k], q[k]))
    print(f"{len(series)} stroke series from {len(dirs)} recording(s)")
    rng = np.random.default_rng(0)
    test = sorted(rng.choice(len(series), max(1, len(series) // 4), replace=False).tolist())
    train = [i for i in range(len(series)) if i not in test]
    old = json.loads(OUT.read_text())
    A0, B0, c0 = np.array(old["A"]), np.array(old["B"]), np.array(old["c"])
    d0 = int(round(old.get("delay_s", 0.0) * 100))

    def model(x):
        a, b = np.exp(x[:5]), np.exp(x[5:])
        return a[:, None] * A0, b[:, None] * B0, a * c0

    def sims(A, B, c, ds, idx):
        for i in idx:
            t, u, q = series[i]
            yield t, q, simulate(A, B, c, ds, t, u, q[0], (q[1] - q[0]) / max(t[1] - t[0], 1e-3))

    def residuals(A, B, c, ds, idx):
        return np.concatenate([q - qs for _, q, qs in sims(A, B, c, ds, idx)])

    def report(A, B, c, ds, idx):
        errs, peaks = [], []
        for t, q, qs in sims(A, B, c, ds, idx):
            errs.append(q - qs)
            peaks.append(np.abs(np.gradient(q[:, 2], t)).max() / max(np.abs(np.gradient(qs[:, 2], t)).max(), 1e-6))
        e = np.concatenate(errs)
        return np.degrees(np.sqrt((e ** 2).mean(0))), float(np.median(peaks))

    rms0, pk0 = report(A0, B0, c0, d0, test)
    print(f"model in use ({old.get('source', '?')}, delay {10 * d0} ms): held-out RMS [deg] {rms0.round(2)} "
          f"(mean {rms0.mean():.2f}); elbow peak speed real/model {pk0:.2f}")
    fit_idx = train[:: max(1, len(train) // 8)]  # a subset keeps the simulation-based fit quick
    best = None
    for ds in range(0, 9):
        sol = least_squares(lambda x: residuals(*model(x), ds, fit_idx).ravel(), np.zeros(10),
                            bounds=(np.log(0.2), np.log(5.0)), x_scale=0.3, max_nfev=60)
        A, B, c = model(sol.x)
        rms, pk = report(A, B, c, ds, test)
        print(f"  delay {10 * ds:3d} ms: stiffness x {np.exp(sol.x[:5]).round(2)}, damping x "
              f"{np.exp(sol.x[5:]).round(2)}; held-out RMS [deg] {rms.round(2)} (mean {rms.mean():.2f}); elbow peak "
              f"real/model {pk:.2f}")
        if np.isfinite(rms).all() and (best is None or rms.mean() < best[0]):
            best = (rms.mean(), A, B, c, ds, sol.x)
    score, A, B, c, ds, x = best
    if score > 0.9 * rms0.mean():  # 2026-11-10 real strokes: 0.57 vs 0.59 deg, not worth a different feedforward
        print(f"the refit (held-out mean {score:.2f} deg) isn't 10 % better than the model in use ({rms0.mean():.2f}): "
              "kept")
        return
    sag = np.linalg.solve(A, c)
    out.write_text(json.dumps(dict(joints=list(ARM_JOINTS), A=A.tolist(), B=B.tolist(), c=c.tolist(),
                                   delay_s=ds / 100, stiffness_scale=np.exp(x[:5]).round(3).tolist(),
                                   damping_scale=np.exp(x[5:]).round(3).tolist(), ready_sag_rad=sag.tolist(),
                                   source=f"scripts/identify_tracking.py --from {' '.join(d.name for d in dirs)} "
                                          f"(scaled from: {old.get('source', '?')})"), indent=1) + "\n")
    print(f"chose delay {10 * ds} ms (held-out mean {score:.2f} vs {rms0.mean():.2f} deg); sag at ready [deg] "
          f"{np.degrees(sag).round(2)}")
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", nargs="+", type=Path, help="stroke recordings (record_servos.py strokes)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.src:
        return fit_recording(args.src, args.out)
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
    args.out.write_text(json.dumps(dict(joints=list(ARM_JOINTS), A=A.tolist(), B=B.tolist(), c=c.tolist(),
                                   r2=r2.round(3).tolist(), ready_sag_rad=sag.tolist(),
                                   source="scripts/identify_tracking.py, fitted servo model, nominal"), indent=1) + "\n")
    wn = np.sqrt(np.diag(A))
    print("R^2 per joint", r2.round(3))
    print("diagonal wn [rad/s]", wn.round(1), " zeta", (np.diag(B) / (2 * wn)).round(2))
    print("sag at the ready pose [deg]", np.degrees(sag).round(2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
