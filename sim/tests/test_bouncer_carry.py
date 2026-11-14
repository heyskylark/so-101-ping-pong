"""The wrist-only bouncer's carry: the hold pose follows the ball along x (BouncerCfg.reach_x)."""

import pytest
import torch

from pingpong_sim.ball import BALL_RADIUS
from pingpong_sim.bouncer import BouncerCfg, ScriptedBouncer
from pingpong_sim.kinematics import ArmKinematics
from pingpong_sim.robot import ARM_JOINTS, READY_JOINT_POS, READY_X, READY_Z, get_spec


@pytest.fixture(scope="module")
def kin():
    return ArmKinematics(get_spec().compile(), prefix="")


def plan(kin, reach_x: float, ball_x: float) -> ScriptedBouncer:
    """A bouncer with its plan for a ball falling straight down ball_x ahead of the target."""
    q_ready = torch.tensor([READY_JOINT_POS[j] for j in ARM_JOINTS])
    b = ScriptedBouncer(BouncerCfg(wrist_only=True, stroke_joint=3, reach_x=reach_x), kin, q_ready, 1)
    b.step(0.0, torch.tensor([[READY_X + ball_x, 0.0, READY_Z + BALL_RADIUS + 0.25]]), torch.tensor([[0.0, 0.0, -0.5]]))
    assert float(b.t_hit[0]) > 0.0  # a plan was made
    return b


def test_hold_pose_follows_the_ball_level_and_at_height(kin):
    b = plan(kin, reach_x=0.04, ball_x=0.03)
    face, normal, _, _ = kin.fk(b.q_rest)
    ready, ready_normal, _, _ = kin.fk(b.q_ready)
    assert float(face[0, 0] - ready[0, 0]) == pytest.approx(0.03, abs=1e-3)
    assert float(face[0, 2] - ready[0, 2]) == pytest.approx(0.0, abs=1e-3)
    assert torch.allclose(normal, ready_normal, atol=2e-3)
    assert torch.equal(b.q_hit[:, 1:3], b.q_rest[:, 1:3])  # lift and elbow hold through the stroke: it is the wrist's
    # After the hit the arm stays under the ball instead of returning to the centre.
    q_after, _, _ = b.evaluate(b.t_hit + 1.0)
    assert torch.allclose(q_after, b.q_rest, atol=1e-5)


def test_carry_stops_at_its_reach(kin):
    b = plan(kin, reach_x=0.04, ball_x=0.10)
    face, _, _, _ = kin.fk(b.q_rest)
    ready, _, _, _ = kin.fk(b.q_ready)
    assert float(face[0, 0] - ready[0, 0]) == pytest.approx(0.04, abs=1e-3)


def test_without_reach_lift_and_elbow_hold_the_ready_pose(kin):
    b = plan(kin, reach_x=0.0, ball_x=0.03)
    assert torch.equal(b.q_hit[:, 1:3], b.q_ready[:, 1:3])
    assert torch.equal(b.q_rest, b.q_ready)
