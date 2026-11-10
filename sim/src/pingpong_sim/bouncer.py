"""Scripted ball bouncer: predict, aim, hit. No learning; vectorized over worlds (torch).

Every control step, per world:

1. **Predict** where and when the ball's centre next falls through the hitting plane (z = hit_z + ball radius),
   integrating gravity + quadratic drag (Cd 0.50, measured) from the ball's position and velocity.
2. **Aim.** Choose the outgoing velocity: vertical so the ball rises `apex` above the plane (closed form with
   drag), horizontal so it comes down `steer_gain` of the way back to the target point. The paddle face normal is
   along (v_out - v_in): an impact only changes the normal velocity component (tangential friction ignored). The
   paddle speed s along the normal follows from the measured restitution: v_out_n = -e v_in_n + (1 + e) s.
3. **Hit.** A stroke of fixed duration `stroke_time` with constant acceleration, reaching paddle speed s along the
   normal at the impact pose (from IK). The stroke turns one joint only (`stroke_joint`, the elbow): with the
   forearm near level, the face is ~0.32 m out from the elbow axis at about its height, so it moves almost
   straight up. A multi-joint stroke that keeps the face level needs elbow and wrist (different firmware gains)
   to counter-rotate in step; a small timing mismatch between them cancelled most of the face speed. The face
   tilts a few degrees during the single-joint stroke but is on plan at impact. Before it, a cubic Hermite brings
   the arm from wherever it is to the cocked pose at rest; after impact, a follow-through returns it to ready.
4. **Feedforward.** The servo goal inverts the identified tracking model (scripts/identify_tracking.py),
   u = q_d + A^-1 (q_d'' + B q_d' - c), evaluated `lead` seconds ahead (the command delay: by default the tracking
   model's own `delay_s`, 0 for the nominal one; run_bouncer sweeps it).

Replanning refines the approach until `lock` seconds before the stroke starts; the stroke itself is never changed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch

from pingpong_sim.ball import BALL_RADIUS, DRAG_K, G
from pingpong_sim.ball_filter import flight, flight_np
from pingpong_sim.kinematics import ArmKinematics
from pingpong_sim.robot import ASSETS

TRACKING_MODEL = ASSETS / "actuators" / "tracking_model.json"  # scripts/identify_tracking.py


@dataclass
class BouncerCfg:
    target_xy: tuple[float, float] = (0.40, 0.0)  # m, where the ball should land on the blade (ready pose centre)
    hit_z: float = 0.25  # m, height of the blade face at impact
    apex: float = 0.25  # m, ball rise above the hitting plane
    restitution: float = 0.58  # measured on this paddle (measurements/ball)
    steer_gain: float = 0.7  # fraction of the landing-point error corrected per bounce
    reach: float = 0.06  # m, impact points are clipped to this radius around target_xy
    max_tilt_deg: float = 20.0
    stroke_time: float = 0.12  # s, constant-acceleration stroke ending at impact
    stroke_joint: int = 2  # elbow_flex
    lead: float | None = None  # s, goals are evaluated this far ahead of the plan (None: the tracking model's delay)
    lock: float = 0.03  # s, no replanning this close to the stroke start
    # s, added to every predicted impact time. Real rig 2026-11-09: predictions 19-28 ms early (replayed logs) with
    # the arm on its plan -> +0.025; the sim servo lags its plan ~10-20 ms, where early predictions helped.
    impact_bias: float = 0.0
    follow: float = 0.15  # s, follow-through back to the ready pose after impact
    min_approach: float = 0.04  # s, shortest approach segment
    predict_dt: float = 0.02  # s, RK4 flight steps (crossing time by interpolation: ~0.2 ms error)
    predict_horizon: float = 1.2  # s
    feedforward: bool = True  # invert the identified tracking model (else send the plan as the goal)


def apex_speed(h: float) -> float:
    """Upward speed that reaches height h under gravity + quadratic drag: h = ln(1 + k v^2 / g) / (2 k)."""
    return math.sqrt(G / DRAG_K * (math.exp(2 * DRAG_K * h) - 1))


def hermite(t, t0, T, p0, v0, p1, v1):
    """Cubic Hermite position, velocity and acceleration at times t (B,), segments starting at t0 (B,) of length
    T (B,); held at (p1, v1, 0) after the end."""
    s = ((t - t0) / T).clamp(0, 1)[:, None]
    Tc = T[:, None]
    s2, s3 = s * s, s * s * s
    p = (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * Tc * v0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * Tc * v1
    v = ((6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * Tc * v0 + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 2 * s) * Tc * v1) / Tc
    inside = ((t - t0) / T < 1)[:, None].to(p.dtype)
    a = inside * ((12 * s - 6) * p0 + (6 * s - 4) * Tc * v0 + (-12 * s + 6) * p1 + (6 * s - 2) * Tc * v1) / Tc**2
    return p, v, a


class ScriptedBouncer:
    def __init__(self, cfg: BouncerCfg, kin: ArmKinematics, q_ready: torch.Tensor, num_envs: int,
                 lead: torch.Tensor | None = None):
        self.cfg, self.kin = cfg, kin
        dev = q_ready.device
        self.n, self.dev = num_envs, dev
        self.q_ready = q_ready.expand(num_envs, 5).clone()
        tm = json.loads(TRACKING_MODEL.read_text())
        lead_s = cfg.lead if cfg.lead is not None else float(tm.get("delay_s", 0.0))
        self.lead = lead if lead is not None else torch.full((num_envs,), lead_s, device=dev)
        self.target = torch.tensor(cfg.target_xy, device=dev)
        self.z_ball = cfg.hit_z + BALL_RADIUS
        self.vz_out = apex_speed(cfg.apex)
        self.A_inv = torch.linalg.inv(torch.tensor(tm["A"], device=dev))
        self.B = torch.tensor(tm["B"], device=dev)
        self.c = torch.tensor(tm["c"], device=dev)
        z = torch.zeros(num_envs, device=dev)
        Ts = cfg.stroke_time
        # Plan: approach Hermite [a_t0, t_hit - Ts] -> stroke [t_hit - Ts, t_hit] -> follow-through. Start: at ready.
        self.t_hit = z - 1.0
        self.a_t0, self.a_T = z - 2.0, z + 1.0
        self.a_p0, self.a_v0 = self.q_ready.clone(), torch.zeros_like(self.q_ready)
        self.q_cock, self.q_hit = self.q_ready.clone(), self.q_ready.clone()
        self.Ts = Ts
        # Logged per world at each plan update (last plan wins).
        self.plan_speed = z.clone()  # paddle speed along the normal at impact [m/s]
        self.plan_tilt = z.clone()  # face tilt from vertical [rad]
        self.plan_ik_err = z.clone()  # m

    def reset(self, env_ids: torch.Tensor, now: float) -> None:
        """Drop the plans of env_ids: hold the ready pose until the next plan (episode reset at time `now`)."""
        self.t_hit[env_ids] = now - 1.0
        self.a_t0[env_ids] = now - 2.0
        self.a_T[env_ids] = 1.0
        for buf in (self.a_p0, self.q_cock, self.q_hit):
            buf[env_ids] = self.q_ready[env_ids]
        self.a_v0[env_ids] = 0.0
        self.plan_speed[env_ids] = 0.0
        self.plan_tilt[env_ids] = 0.0
        self.plan_ik_err[env_ids] = 0.0

    def evaluate(self, t: torch.Tensor):
        """Planned joint angles, velocities and accelerations at times t (B,)."""
        Ts = self.Ts
        t_s0 = self.t_hit - Ts
        zero = torch.zeros_like(self.q_ready)
        pa, va, aa = hermite(t, self.a_t0, self.a_T, self.a_p0, self.a_v0, self.q_cock, zero)
        tau = (t - t_s0).clamp(0, Ts)[:, None]
        dq = self.q_hit - self.q_cock
        ps, vs, as_ = self.q_cock + dq * (tau / Ts) ** 2, dq * 2 * tau / Ts**2, dq * 2 / Ts**2
        qd_hit = dq * 2 / Ts
        pf, vf, af = hermite(t, self.t_hit, torch.full_like(t, self.cfg.follow), self.q_hit, qd_hit, self.q_ready, zero)
        approach, stroke = (t < t_s0)[:, None], (t < self.t_hit)[:, None]
        pick = lambda a, s, f: torch.where(approach, a, torch.where(stroke, s, f))  # noqa: E731
        return pick(pa, ps, pf), pick(va, vs, vf), pick(aa, as_.expand_as(aa), af)

    def goal(self, q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor) -> torch.Tensor:
        """Servo goal that makes the tracking model follow (q, qd, qdd): u = q + A^-1 (qdd + B qd - c)."""
        if not self.cfg.feedforward:
            return q
        return q + (qdd + qd @ self.B.T - self.c) @ self.A_inv.T

    def predict(self, p: torch.Tensor, v: torch.Tensor):
        """First downward crossing of the hitting plane: time from now (B,; inf if none within the horizon),
        ball position and velocity there. One world on the CPU takes the float64 numpy path (same steps)."""
        if self.n == 1 and p.device.type == "cpu":
            return self._predict_one(p, v)
        return self._predict_batched(p, v)

    def _predict_one(self, p: torch.Tensor, v: torch.Tensor):
        dt, z = self.cfg.predict_dt, self.z_ball
        ps, vs = p[0].double().numpy(), v[0].double().numpy()
        for k in range(int(self.cfg.predict_horizon / dt)):
            pn, vn = flight_np(ps, vs, dt)
            if ps[2] >= z and pn[2] < z and vn[2] < 0:
                f = (ps[2] - z) / max(ps[2] - pn[2], 1e-9)
                return (torch.tensor([(k + f) * dt], dtype=p.dtype), torch.tensor(ps + f * (pn - ps), dtype=p.dtype)[None],
                        torch.tensor(vs + f * (vn - vs), dtype=v.dtype)[None])
            ps, vs = pn, vn
        return torch.tensor([math.inf], dtype=p.dtype), p.clone(), v.clone()

    def _predict_batched(self, p: torch.Tensor, v: torch.Tensor):
        cfg = self.cfg
        dt = cfg.predict_dt
        h = torch.full((self.n,), dt, device=self.dev)
        t_hit = torch.full((self.n,), math.inf, device=self.dev)
        p_hit, v_hit = p.clone(), v.clone()
        found = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        for k in range(int(cfg.predict_horizon / dt)):
            p_next, v_next = flight(p, v, h)
            cross = (~found) & (p[:, 2] >= self.z_ball) & (p_next[:, 2] < self.z_ball) & (v_next[:, 2] < 0)
            f = ((p[:, 2] - self.z_ball) / (p[:, 2] - p_next[:, 2]).clamp_min(1e-9))[:, None]
            t_hit = torch.where(cross, (k + f[:, 0]) * dt, t_hit)
            p_hit = torch.where(cross[:, None], p + f * (p_next - p), p_hit)
            v_hit = torch.where(cross[:, None], v + f * (v_next - v), v_hit)
            found |= cross
            if k % 8 == 7 and bool(found.all()):  # host sync only every 8 steps
                break
            p, v = p_next, v_next
        return t_hit, p_hit, v_hit

    def step(self, now: float, ball_p: torch.Tensor, ball_v: torch.Tensor, obs_time: float | None = None,
             prediction: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """Servo goals (B, 5) for this control step, given the (estimated) ball state measured at `obs_time`
        (default: now). Predicting from the measurement time is what the real Kalman filter does with its known
        latency. `prediction`: `predict(ball_p, ball_v)` if the caller already has it."""
        cfg = self.cfg
        t_cmd = now + self.lead
        q_c, qd_c, _ = self.evaluate(t_cmd)
        dt_hit, p_in, v_in = prediction if prediction is not None else self.predict(ball_p, ball_v)
        t_hit = (now if obs_time is None else obs_time) + dt_hit + cfg.impact_bias
        # Refine during the approach (until `lock` before the stroke); start a new plan once the follow-through is
        # done; never during the stroke or the follow-through.
        approaching = t_cmd < self.t_hit - self.Ts - cfg.lock
        recovered = t_cmd > self.t_hit + cfg.follow
        replan = (torch.isfinite(dt_hit) & (approaching | recovered)
                  & (t_hit - self.Ts - t_cmd > cfg.min_approach))
        if replan.any():
            self._plan(replan, t_cmd, q_c, qd_c, t_hit, p_in, v_in)
        return self.goal(*self.evaluate(t_cmd))

    def _plan(self, m, t_cmd, q_c, qd_c, t_hit, p_in, v_in) -> None:
        cfg = self.cfg
        # Impact point, clipped to the reachable patch.
        d = p_in[:, :2] - self.target
        r = d.norm(dim=1, keepdim=True).clamp_min(1e-9)
        xy = self.target + d * (cfg.reach / r).clamp(max=1.0)
        # Outgoing velocity: up to the apex, back toward the target.
        v_out = torch.zeros_like(v_in)
        v_out[:, :2] = cfg.steer_gain * (self.target - xy) / (2 * self.vz_out / G)
        v_out[:, 2] = self.vz_out
        n = v_out - v_in
        n = n / n.norm(dim=1, keepdim=True)
        tilt = torch.acos(n[:, 2].clamp(-1, 1))
        max_tilt = math.radians(cfg.max_tilt_deg)
        too_far = tilt > max_tilt
        if too_far.any():  # keep the azimuth, cap the tilt
            h = n[:, :2] / n[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-9)
            capped = torch.cat([h * math.sin(max_tilt), torch.full_like(n[:, 2:], math.cos(max_tilt))], dim=1)
            n = torch.where(too_far[:, None], capped, n)
            tilt = tilt.clamp(max=max_tilt)
        e = cfg.restitution
        speed = ((v_out * n).sum(1) + e * (v_in * n).sum(1)) / (1 + e)
        face = torch.cat([xy, torch.full_like(xy[:, :1], self.z_ball)], dim=1) - BALL_RADIUS * n
        q_hit, err = self.kin.ik(face, n, q_c, iters=6)
        # Single-joint stroke: joint displacement giving `stroke` metres of face travel along n at the impact pose.
        stroke = speed * self.Ts / 2
        face_h, _, origins, axes = self.kin.fk(q_hit)
        j = cfg.stroke_joint
        lever = (torch.cross(axes[:, j], face_h - origins[:, j], dim=1) * n).sum(1)  # m of face motion per rad
        sign = torch.where(lever < 0, -1.0, 1.0)
        lever = sign * lever.abs().clamp_min(0.05)  # guard against a stroke joint that barely moves the face
        q_cock = q_hit.clone()
        q_cock[:, j] = q_hit[:, j] - stroke / lever
        mk = m[:, None]
        self.a_t0 = torch.where(m, t_cmd, self.a_t0)
        self.a_T = torch.where(m, t_hit - self.Ts - t_cmd, self.a_T)
        self.a_p0 = torch.where(mk, q_c, self.a_p0)
        self.a_v0 = torch.where(mk, qd_c, self.a_v0)
        self.q_cock = torch.where(mk, q_cock, self.q_cock)
        self.q_hit = torch.where(mk, q_hit, self.q_hit)
        self.t_hit = torch.where(m, t_hit, self.t_hit)
        self.plan_speed = torch.where(m, speed, self.plan_speed)
        self.plan_tilt = torch.where(m, tilt, self.plan_tilt)
        self.plan_ik_err = torch.where(m, err, self.plan_ik_err)
