"""Replay servo recordings (scripts/record_servos.py) in MuJoCo Warp for system identification.

The recording is cut into windows of WINDOW samples (2 s at 100 Hz). Each window starts from the measured joint
angles and velocities (gear springs relaxed, firmware target = the delayed goal) and replays the recorded goal
stream at the sim timestep: goal k takes effect at t_cmd[k] + delay, the supply voltage follows the logged one
(median-filtered: the voltage register glitches). Simulated encoder angles are sampled at the read times t_read[k]
and compared with the measured ones after a BURN_IN.

One simulation holds n_candidates x n_windows worlds, world e = candidate e // W, window e % W, so a whole
CMA-ES population is scored in one rollout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from scipy.ndimage import median_filter

from pingpong_sim.robot import ARM_JOINTS, FIRMWARE_P, get_spec
from pingpong_sim.servo_model import PARAMS, ServoModel, motorize

DT = 0.001
WINDOW = 200  # samples (2 s)
BURN_IN = 20  # samples (0.2 s) excluded from the cost: gear deflection and firmware target start unknown
MAX_DELAY = 0.05  # s, upper bound of the delay parameter
FIT_POSES = ("ready", "near_low")


@dataclass
class Windows:
    names: list[str]  # "<segment>@<t>" per window
    split: np.ndarray  # (W,) "fit" / "val"
    goal: np.ndarray  # (W, T + D, J) goal at every sim step, D = MAX_DELAY / DT history steps first
    vin: np.ndarray  # (W, T) supply voltage
    read_step: np.ndarray  # (W, R) sim step of each read
    pos: np.ndarray  # (W, R, J) measured encoder angles
    q0: np.ndarray  # (W, J)
    dq0: np.ndarray  # (W, J)
    kind: np.ndarray  # (W, R) segment kind of each sample
    joint: np.ndarray  # (W, R) excited joint of each sample ("" if none)

    @property
    def steps(self) -> int:
        return self.goal.shape[1] - int(round(MAX_DELAY / DT))

    def subset(self, mask: np.ndarray) -> Windows:
        return Windows(**{k: (np.asarray(v)[mask] if k != "names" else [n for n, m in zip(v, mask) if m])
                          for k, v in self.__dict__.items()})


def load_windows(run_dir: Path) -> Windows:
    log = np.load(run_dir / "log.npz")
    meta = json.loads((run_dir / "meta.json").read_text())
    assert tuple(meta["joints"]) == ARM_JOINTS, meta["joints"]
    segs = meta["segments"]
    t_cmd, t_read, goal, pos = log["t_cmd"], log["t_read"], log["goal"], log["pos"]
    seg, phase = log["seg"], log["phase"]
    vin = median_filter(log["volt"], size=(51, 1)).mean(axis=1)

    def label(k: int) -> str:
        if phase[k] != 0:
            return ""
        s = segs[seg[k]]
        if s.get("kind") == "bounce" or s.get("pose") == "far_high":
            return "val"
        return "fit" if s.get("pose") in FIT_POSES else ""

    labels = np.array([label(k) for k in range(len(t_cmd))])
    hist = int(round(MAX_DELAY / DT))
    out = {k: [] for k in Windows.__dataclass_fields__}
    k = 1
    while k + WINDOW < len(t_cmd):
        w = slice(k, k + WINDOW)
        if labels[k] == "" or (labels[w] != labels[k]).any() or np.diff(t_read[w]).max() > 0.05:
            k += 1
            continue
        t0 = t_cmd[k]
        T = int(round((t_read[k + WINDOW - 1] - t0) / DT)) + 2
        times = t0 + (np.arange(T + hist) - hist) * DT
        idx = np.clip(np.searchsorted(t_cmd, times, side="right") - 1, 0, None)
        out["goal"].append(goal[idx])
        out["vin"].append(vin[idx[hist:]])
        out["read_step"].append(np.round((t_read[w] - t0) / DT).astype(int))
        out["pos"].append(pos[w])
        out["q0"].append(pos[k])
        out["dq0"].append((pos[k + 1] - pos[k - 1]) / (t_read[k + 1] - t_read[k - 1]))
        out["kind"].append(np.array([segs[s].get("kind", "") for s in seg[w]]))
        out["joint"].append(np.array([segs[s].get("joint", "") for s in seg[w]]))
        out["names"].append(f"{segs[seg[k]]['name']}@{t_read[k]:.1f}")
        out["split"].append(labels[k])
        k += WINDOW
    T = min(len(g) for g in out["goal"]) - hist
    out["goal"] = [g[: T + hist] for g in out["goal"]]
    out["vin"] = [v[:T] for v in out["vin"]]
    assert max(r.max() for r in out["read_step"]) < T, "read beyond the shortest window"
    return Windows(**{k: (np.stack(v) if k != "names" else v) for k, v in out.items()})


def make_spec() -> mujoco.MjSpec:
    spec = get_spec()  # includes the gear joints
    motorize(spec, ARM_JOINTS)
    return spec


class Replay:
    """n_candidates x n_windows worlds replaying `windows`."""

    def __init__(self, windows: Windows, n_candidates: int, device: str = "cuda:0"):
        self.w, self.P = windows, n_candidates
        self.W = len(windows.names)
        self.N = self.P * self.W
        self.mj_model = make_spec().compile()
        self.sim = Simulation(
            num_envs=self.N,
            cfg=SimulationCfg(njmax=40, nconmax=1, mujoco=MujocoCfg(
                timestep=DT, integrator="euler", cone="elliptic", disableflags=("contact",))),
            model=self.mj_model, device=device)
        self.sim.expand_model_fields(ServoModel.EXPANDED_FIELDS)
        dev = self.sim.device
        self.servo = ServoModel(self.sim.model, self.sim.data, self.mj_model, list(ARM_JOINTS),
                                [FIRMWARE_P[j] for j in ARM_JOINTS], self.N, dev)
        self.ctrl = torch.tensor([self.mj_model.actuator(j).id for j in ARM_JOINTS], device=dev)
        self.win = torch.arange(self.N, device=dev) % self.W
        self.goal = torch.tensor(windows.goal, dtype=torch.float32, device=dev)
        self.vin = torch.tensor(windows.vin, dtype=torch.float32, device=dev)
        self.read_step = torch.tensor(windows.read_step, device=dev)
        self.pos = torch.tensor(windows.pos, dtype=torch.float32, device=dev)
        self.hist = int(round(MAX_DELAY / DT))

    def expand(self, cand: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Candidate params (P, ...) -> per-world params (N, ...)."""
        return {k: np.repeat(np.asarray(v), self.W, axis=0) for k, v in cand.items()}

    @torch.no_grad()
    def run(self, cand: dict[str, np.ndarray]) -> torch.Tensor:
        """Simulated encoder angles at the read times, (P, W, R, J)."""
        per_env = self.expand(cand)
        self.sim.reset()
        self.servo.set_params(per_env)
        dev = self.sim.device
        rows = torch.arange(self.N, device=dev)
        lag = torch.tensor(np.round(per_env["delay"] / DT).astype(int), device=dev)
        q0 = torch.tensor(self.w.q0, dtype=torch.float32, device=dev)[self.win]
        dq0 = torch.tensor(self.w.dq0, dtype=torch.float32, device=dev)[self.win]
        self.servo.teleport(q0, dq0)
        self.sim.data.ctrl[:, self.ctrl] = 0.0
        self.servo.set_target(self.goal[self.win, self.hist - lag])
        self.sim.forward()
        T = self.w.steps
        traj = torch.empty(T, self.N, len(ARM_JOINTS), device=dev)
        for s in range(T):
            self.servo.vin = self.vin[self.win, s][:, None]
            self.sim.data.ctrl[:, self.ctrl] = self.servo.apply(self.goal[self.win, self.hist + s - lag], DT)
            self.sim.step()
            traj[s] = self.servo.encoder()
        sim_at_reads = traj[self.read_step[self.win], rows[:, None]]  # (N, R, J)
        return sim_at_reads.reshape(self.P, self.W, *sim_at_reads.shape[1:])

    def cost(self, sim_pos: torch.Tensor, band_weight: float = 0.0) -> torch.Tensor:
        """Per candidate: mean squared encoder error [rad^2] after the burn-in, plus band_weight x the mean squared
        difference of the 2-8 Hz motion amplitude (phase-free: matches how much the joints hunt, not when)."""
        err = sim_pos[:, :, BURN_IN:] - self.pos[None, :, BURN_IN:]
        cost = err.pow(2).mean(dim=(1, 2, 3))
        if band_weight:
            amp_sim, amp_real = band_amplitude(sim_pos), band_amplitude(self.pos[None])
            cost = cost + band_weight * (amp_sim - amp_real).pow(2).mean(dim=(1, 2))
        return torch.nan_to_num(cost, nan=1e3)


def band_amplitude(pos: torch.Tensor, lo: float = 2.0, hi: float = 8.0, rate: float = 100.0) -> torch.Tensor:
    """RMS [rad] of the lo-hi Hz part of each window/joint after the burn-in: (..., W, R, J) -> (..., W, J)."""
    x = pos[..., BURN_IN:, :]
    x = x - x.mean(dim=-2, keepdim=True)
    spec = torch.fft.rfft(x, dim=-2)
    f = torch.fft.rfftfreq(x.shape[-2], 1.0 / rate).to(x.device)
    band = ((f >= lo) & (f <= hi)).to(x.dtype)[:, None]
    n = x.shape[-2]
    return torch.sqrt(2.0 * (spec.abs().pow(2) * band).sum(dim=-2) / n**2)


def rms_by_joint(sim_pos: np.ndarray, real_pos: np.ndarray) -> np.ndarray:
    """RMS error [deg] per joint, (W, R, J) arrays, after the burn-in."""
    e = sim_pos[:, BURN_IN:] - real_pos[:, BURN_IN:]
    return np.degrees(np.sqrt((e**2).mean(axis=(0, 1))))


def params_to_json(values: dict[str, np.ndarray], base: dict) -> dict:
    """Fitted values in BAM's params format plus our extras (gear compliance, delay, per-joint friction scale)."""
    out = dict(base)
    for p in PARAMS:
        v = np.asarray(values[p.name], dtype=float)
        out[p.name] = v.tolist() if p.per_joint else float(v)
    out["joints"] = list(ARM_JOINTS)
    return out
