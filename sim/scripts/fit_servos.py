"""Fit the servo model (sim/src/pingpong_sim/servo_model.py) to a servo recording with CMA-ES.

Fit windows: the 'ready' and 'near_low' poses (steps, reversals, chirps, holds). Held-out windows: the 'far_high'
pose and the bounce-like motion. The cost is the mean squared encoder-angle error over all joints, plus
--band-weight x the squared difference of each joint's 2-8 Hz motion amplitude (phase-free).

Outputs:
  src/pingpong_sim/assets/actuators/sts3215_12v_fit.json   fitted params (BAM format + gear compliance, delay, scales)
  ../measurements/servo/fit_<time>/report.json             RMS before/after, params, sensitivity, cost history
  ../measurements/servo/fit_<time>/*.png                   held-out step responses, cost history

Run (from sim/): uv run scripts/fit_servos.py [--run ../measurements/servo/<recording>] [--gens 150]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from cmaes import CMA  # noqa: E402

from pingpong_sim import sysid  # noqa: E402
from pingpong_sim.robot import ARM_JOINTS, ASSETS, SERVO_PARAMS  # noqa: E402
from pingpong_sim.servo_model import PARAMS, bam_baseline, decode, encode, param_dim  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "measurements" / "servo" / "2026-11-08_212508"  # D = 0
OUT_PARAMS = ASSETS / "actuators" / "sts3215_12v_fit.json"
J = len(ARM_JOINTS)


def batch(values: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([np.asarray(v[k], float) for v in values]) for k in values[0]}


def evaluate(windows: sysid.Windows, cands: list[dict]) -> np.ndarray:
    """Simulated encoder angles (C, W, R, J) for each candidate on all windows."""
    r = sysid.Replay(windows, len(cands))
    return r.run(batch(cands)).cpu().numpy()


def param_names() -> list[str]:
    return [f"{p.name}[{j}]" if p.per_joint else p.name for p in PARAMS for j in (ARM_JOINTS if p.per_joint else [None])]


def coupling(w: sysid.Windows, sim: np.ndarray | None) -> dict[str, list[float]]:
    """Mean peak-to-peak motion [deg] of every joint during the step windows of `excited` (real if sim is None)."""
    out = {}
    for excited in ("shoulder_lift", "elbow_flex", "wrist_flex"):
        rows = np.nonzero(((w.joint == excited) & (w.kind == "steps")).all(axis=1))[0]
        x = w.pos if sim is None else sim
        out[excited] = np.degrees(np.ptp(x[rows, sysid.BURN_IN:], axis=1)).mean(0).round(2).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band-weight", type=float, default=1.0,
                    help="weight of the phase-free 2-8 Hz amplitude term (0: pointwise error only)")
    args = ap.parse_args()
    out = REPO / "measurements" / "servo" / f"fit_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    out.mkdir(parents=True)

    w = sysid.load_windows(args.run)
    fit, val = w.split == "fit", w.split == "val"
    print(f"{args.run.name}: {fit.sum()} fit windows, {val.sum()} held-out windows, 2 s each")
    base = bam_baseline(json.loads(SERVO_PARAMS.read_text()), J)

    # Reference points: the pre-fit model, and "the arm does exactly what it is told" (goal as the prediction).
    goal_at_read = np.take_along_axis(w.goal[:, int(sysid.MAX_DELAY / sysid.DT):], w.read_step[..., None], axis=1)
    rms = {"goal_as_prediction": {s: sysid.rms_by_joint(goal_at_read[m], w.pos[m]) for s, m in (("fit", fit), ("val", val))}}
    sim_base = evaluate(w, [base])[0]
    rms["before"] = {s: sysid.rms_by_joint(sim_base[m], w.pos[m]) for s, m in (("fit", fit), ("val", val))}
    for k, v in rms.items():
        print(f"{k:18s} RMS [deg] fit {v['fit'].round(2)}  held-out {v['val'].round(2)}")

    # CMA-ES in the unit cube. Start: the pre-fit model with mid-range gear stiffness and delay (rigid gears and
    # zero delay sit outside / on the search bounds).
    start = dict(base, gear_stiffness=np.full(J, 150.0), delay=0.02, feedback_delay=0.006, max_current=2.0)
    opt = CMA(mean=encode(start, J), sigma=args.sigma, bounds=np.tile([0.0, 1.0], (param_dim(J), 1)),
              population_size=args.pop, seed=args.seed)
    replay = sysid.Replay(w.subset(fit), args.pop)
    history, best_x, best_c = [], None, np.inf
    t0 = time.time()
    for g in range(args.gens):
        xs = np.stack([opt.ask() for _ in range(args.pop)])
        costs = replay.cost(replay.run(decode(xs, J)), args.band_weight).cpu().numpy()
        opt.tell(list(zip(xs, costs)))
        i = int(np.argmin(costs))
        if costs[i] < best_c:
            best_c, best_x = float(costs[i]), xs[i].copy()
        history.append((g, float(costs.min()), float(np.median(costs)), best_c))
        if g % 5 == 0 or g == args.gens - 1:
            print(f"gen {g:3d}  best sqrt(cost) {np.degrees(np.sqrt(best_c)):.3f} deg  gen median "
                  f"{np.degrees(np.sqrt(np.median(costs))):.3f}  sigma {opt._sigma:.3f}  {time.time() - t0:5.0f} s",
                  flush=True)
        if g >= 40 and history[-40][3] - best_c < 0.003 * best_c:
            print(f"converged at gen {g} (best improved < 0.3 % over 40 generations)")
            break
    best = {k: np.asarray(v) for k, v in decode(best_x[None], J).items()}
    best = {k: v[0] for k, v in best.items()}
    sim_fit = evaluate(w, [best])[0]
    rms["after"] = {s: sysid.rms_by_joint(sim_fit[m], w.pos[m]) for s, m in (("fit", fit), ("val", val))}
    print(f"{'after':18s} RMS [deg] fit {rms['after']['fit'].round(2)}  held-out {rms['after']['val'].round(2)}")
    coup = {"real": coupling(w, None), "before": coupling(w, sim_base), "after": coupling(w, sim_fit)}
    print("p2p motion of each joint [deg] during another joint's steps (real / before / after):")
    for excited in coup["real"]:
        print(f"  {excited:14s} " + "  ".join(f"{k} {coup[k][excited]}" for k in coup))
    band = {k: np.degrees(sysid.band_amplitude(torch.tensor(x[None])).numpy()[0].mean(0))
            for k, x in (("real", w.pos), ("before", sim_base), ("after", sim_fit))}
    print("2-8 Hz motion RMS per joint [deg], mean over windows: "
          + "  ".join(f"{k} {v.round(3)}" for k, v in band.items()))

    # Sensitivity: cost increase on the fit windows when each parameter moves by 10 % of its (log) range.
    names, d = param_names(), param_dim(J)
    probes = [best_x]
    for i in range(d):
        for s in (-0.1, 0.1):
            x = best_x.copy()
            x[i] = np.clip(x[i] + s, 0, 1)
            probes.append(x)
    r = sysid.Replay(w.subset(fit), len(probes))
    pc = r.cost(r.run(decode(np.stack(probes), J)), args.band_weight).cpu().numpy()
    sens = {n: float(max(pc[1 + 2 * i], pc[2 + 2 * i]) / pc[0] - 1) for i, n in enumerate(names)}

    values = {n: v for n, v in zip(names, np.concatenate([np.atleast_1d(best[p.name]) for p in PARAMS]))}
    print("\nparam                          before        fitted   cost +% at 10 % of range")
    flat_base = np.concatenate([np.atleast_1d(np.asarray(base[p.name], float)) for p in PARAMS])
    for n, b in zip(names, flat_base):
        unit = "ms" if n == "delay" else ""
        f = 1e3 if unit == "ms" else 1.0
        print(f"  {n:28s} {b * f:12.4g}  {values[n] * f:12.4g} {unit:3s} {100 * sens[n]:7.1f}")

    json_out = sysid.params_to_json(best, json.loads(SERVO_PARAMS.read_text()))
    json_out.update(q_offset=0.0, fitted_from=str(args.run.relative_to(REPO)), fit_report=str(out.relative_to(REPO)))
    OUT_PARAMS.write_text(json.dumps(json_out, indent=4) + "\n")
    report = dict(run=str(args.run.relative_to(REPO)), windows=dict(fit=int(fit.sum()), val=int(val.sum())),
                  coupling_p2p_deg=coup, band_rms_deg={k: v.round(4).tolist() for k, v in band.items()},
                  band_weight=args.band_weight,
                  rms_deg={k: {s: v.round(4).tolist() for s, v in d_.items()} for k, d_ in rms.items()},
                  joints=list(ARM_JOINTS), params={n: float(v) for n, v in values.items()},
                  params_before={n: float(b) for n, b in zip(names, flat_base)}, sensitivity=sens,
                  history=history, pop=args.pop, seed=args.seed)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    plots(w, sim_base, sim_fit, history, out)
    print("\nwrote", OUT_PARAMS.relative_to(REPO), "and", out.relative_to(REPO))


def plots(w: sysid.Windows, sim_base: np.ndarray, sim_fit: np.ndarray, history: list, out: Path) -> None:
    t = (w.read_step[0] - w.read_step[0, 0]) * sysid.DT
    fig, axes = plt.subplots(J, 1, figsize=(13, 2.6 * J))
    for ax, (j, joint) in zip(axes, enumerate(ARM_JOINTS)):
        cands = [i for i in range(len(w.names)) if w.split[i] == "val" and (w.joint[i] == joint).all()
                 and (w.kind[i] == "steps").all()]
        if not cands:
            continue
        i = cands[len(cands) // 2]
        ax.plot(t, np.degrees(w.pos[i, :, j]), "k", lw=1.4, label="real")
        ax.plot(t, np.degrees(sim_base[i, :, j]), lw=0.9, label="sim before fit")
        ax.plot(t, np.degrees(sim_fit[i, :, j]), lw=0.9, label="sim after fit")
        ax.axvspan(0, sysid.BURN_IN * 0.01, color="0.9")
        ax.set_ylabel(f"{joint} [deg]")
        ax.set_title(w.names[i], fontsize=8, loc="left")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("s (held-out far_high steps; grey = burn-in)")
    fig.tight_layout()
    fig.savefig(out / "heldout_steps.png", dpi=80)
    plt.close(fig)
    h = np.array(history)
    plt.figure(figsize=(6, 3.5))
    plt.plot(h[:, 0], np.degrees(np.sqrt(h[:, 3])), label="best so far")
    plt.plot(h[:, 0], np.degrees(np.sqrt(h[:, 2])), label="generation median", alpha=0.6)
    plt.xlabel("generation")
    plt.ylabel("sqrt(cost) [deg]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "cost_history.png", dpi=80)
    plt.close()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
