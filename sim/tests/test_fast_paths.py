"""The single-world CPU paths (what the robot runs) must match the batched torch paths (what training ran)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pingpong_sim.ball_filter import BallFilter, BallFilterCfg, TickHistory
from pingpong_sim.bouncer import BouncerCfg, ScriptedBouncer
from pingpong_sim.kinematics import ArmKinematics
from pingpong_sim.robot import ARM_JOINTS, READY_JOINT_POS, get_spec


@pytest.fixture(scope="module")
def bouncer() -> ScriptedBouncer:
    kin = ArmKinematics(get_spec().compile(), prefix="")
    q_ready = torch.tensor([READY_JOINT_POS[j] for j in ARM_JOINTS])
    return ScriptedBouncer(BouncerCfg(), kin, q_ready, 1)


def test_predict_matches_batched(bouncer):
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(200):
        p = torch.tensor([[0.4 + rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(0.15, 0.7)]],
                         dtype=torch.float32)
        v = torch.tensor([[rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-3, 3)]], dtype=torch.float32)
        t1, p1, v1 = bouncer._predict_one(p, v)
        t2, p2, v2 = bouncer._predict_batched(p, v)
        assert math.isinf(t1.item()) == math.isinf(t2.item())
        if math.isfinite(t2.item()):
            hits += 1
            assert t1.item() == pytest.approx(t2.item(), abs=1e-4)
            assert torch.allclose(p1, p2, atol=1e-4) and torch.allclose(v1, v2, atol=1e-4)
    assert hits > 50  # most draws cross the plane within the horizon


def test_chain_matches_batched(bouncer):
    kin = bouncer.kin
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = torch.tensor(rng.uniform(-1.5, 1.5, (1, 5)), dtype=torch.float32)
        one = kin._chain_one(q)
        batched = kin._chain(q.repeat(2, 1))  # B > 1: the torch path
        for a, b in zip(one, batched):
            assert torch.allclose(a[0], b[0], atol=1e-5)
        assert kin._chain(q) is one  # repeated angles reuse the last result


def _paddle(dt: float, z0: float, vz: float) -> TickHistory:
    """A level paddle face at x = 0.4 rising at vz: 32 ticks ending at t = 0."""
    h = TickHistory(1, 32, dt, {"center": (3,), "normal": (3,), "vel": (3,)}, "cpu")
    for k in range(32):
        t = (k - 31) * dt
        h.push(t, center=torch.tensor([[0.4, 0.0, z0 + vz * t]]), normal=torch.tensor([[0.0, 0.0, 1.0]]),
               vel=torch.tensor([[0.0, 0.0, vz]]))
    return h


@pytest.mark.parametrize("case", ["bounce", "free", "long"])
def test_propagate_matches_batched(case):
    cfg = BallFilterCfg()
    filt = BallFilter(cfg, 1, "cpu")
    paddle = _paddle(0.01, 0.25, 0.5)
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    P = torch.tensor(A @ A.T * 1e-3 + np.eye(6) * 1e-4, dtype=torch.float32)[None]
    if case == "bounce":  # falling onto the rising paddle within the step
        x = torch.tensor([[0.41, 0.01, 0.25 + 0.02 + 0.03, 0.1, 0.0, -2.2]], dtype=torch.float32)
        t0, t1 = -0.08, -0.01
    elif case == "free":
        x = torch.tensor([[0.40, 0.0, 0.60, 0.2, -0.1, 1.0]], dtype=torch.float32)
        t0, t1 = -0.05, 0.0
    else:  # the filter's longest prediction (hold = 0.2 s) through a bounce
        x = torch.tensor([[0.40, 0.0, 0.45, 0.0, 0.0, -0.5]], dtype=torch.float32)
        t0, t1 = -0.25, 0.0
    t0t, t1t = torch.tensor([t0]), torch.tensor([t1])
    x1, P1 = filt.propagate(x, P, t0t, t1t, paddle)
    x2, P2 = filt._propagate_batched(x, P, t0t, t1t, paddle)
    assert torch.allclose(x1, x2, atol=2e-4), (x1, x2)
    assert torch.allclose(P1, P2, rtol=2e-3, atol=1e-7)
    if case in ("bounce", "long"):
        assert x2[0, 5] > 0  # the paddle sent it back up
