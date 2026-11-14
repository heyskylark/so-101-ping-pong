"""BounceMemory: what a policy is told about this rally's bounces."""

import torch

from pingpong_sim.controller import MEMORY_SLOT, BounceMemory

G = 9.81
DT = 0.01
FACE = torch.tensor([0.28, 0.0, 0.31])


def run(mem: BounceMemory, ticks, lean=(0.0, 0.0), start=0.0) -> float:
    """Feed (tracked, age, position relative to the face, velocity) per tick to world 0 of `mem`; world 1 sees
    nothing. Returns the time of the last tick."""
    n = torch.tensor([[lean[0], lean[1], 1.0], [0.0, 0.0, 1.0]])
    paddle = lambda: (FACE.expand(2, 3), n, torch.tensor([[0.0, 0.0, 0.4], [0.0, 0.0, 0.0]]))  # noqa: E731
    now = start
    for k, (tracked, age, rel, v) in enumerate(ticks):
        now = start + k * DT
        p = torch.stack([FACE + torch.tensor(rel), torch.zeros(3)])
        mem.update(now, torch.tensor([tracked, False]), torch.tensor([age, torch.inf]), p,
                   torch.stack([torch.tensor(v), torch.zeros(3)]), torch.full((2,), G), paddle)
    return now


def bounce(vy_seen: float, lost_after: int | None = None, rel=(0.01, -0.02, 0.02)):
    """A ball falling at 2 m/s that the filter predicts to leave straight up at 1.3 m/s; the frames after the bounce
    then show it leaving with vy_seen sideways. `lost_after`: ticks after the bounce at which the track is lost."""
    ticks = [(True, 0.04, rel, (0.0, 0.0, -2.0))] * 3
    ticks.append((True, 0.04, rel, (0.0, 0.0, 1.3)))  # the predicted bounce
    for k in range(1, 20):
        if lost_after is not None and k >= lost_after:
            ticks.append((False, torch.inf, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        else:
            ticks.append((True, 0.04, rel, (0.0, vy_seen, 1.3 - G * k * DT)))
    return ticks


def slots(mem: BounceMemory, now: float) -> torch.Tensor:
    return mem.obs(now)[0, :-3].reshape(-1, MEMORY_SLOT)


def test_scored_bounce_reports_what_differed_from_the_prediction():
    mem = BounceMemory(3, 0.16, 2, "cpu")
    now = run(mem, bounce(vy_seen=0.1), lean=(0.02, -0.03))
    obs = mem.obs(now)
    s = slots(mem, now)[0]
    assert s[0] == 1.0
    assert torch.allclose(s[1], torch.tensor(now - 3 * DT), atol=1e-6)  # age: the bounce was the fourth tick
    assert torch.allclose(s[2:8], torch.tensor([0.02, -0.03, 0.4, 0.01, -0.02, 2.0]), atol=1e-6)
    assert torch.allclose(s[8:11], torch.tensor([0.0, 0.1, 1.3]), atol=1e-5)  # velocity out, taken back to the bounce
    assert torch.allclose(s[11:14], torch.tensor([0.0, 0.1, 0.0]), atol=1e-5)
    assert torch.allclose(obs[0, -3:], torch.tensor([0.0, 0.1, 0.1]), atol=1e-5)
    assert not obs[1].any()  # the other world saw no ball


def test_bounce_with_the_track_lost_is_kept_unscored():
    mem = BounceMemory(3, 0.16, 2, "cpu")
    now = run(mem, bounce(vy_seen=0.1, lost_after=5))
    obs = mem.obs(now)
    s = slots(mem, now)[0]
    assert s[0] == 0.5
    assert s[7] == 2.0  # what was known at the bounce is kept
    assert not s[8:].any()
    assert not obs[0, -3:].any()


def test_newest_first_and_cleared_by_reset():
    mem = BounceMemory(2, 0.16, 2, "cpu")
    now = run(mem, bounce(vy_seen=0.1))
    now = run(mem, bounce(vy_seen=-0.3), start=now + DT)
    now = run(mem, bounce(vy_seen=0.2), start=now + DT)
    s = slots(mem, now)
    assert torch.allclose(s[:, 12], torch.tensor([0.2, -0.3]), atol=1e-5)  # two slots: the first bounce is gone
    assert torch.allclose(mem.obs(now)[0, -3:], torch.tensor([0.0, 0.0, 0.3]), atol=1e-5)  # mean of all three
    mem.reset(torch.tensor([0]))
    assert not mem.obs(now).any()


def test_turn_away_from_the_blade_is_not_a_bounce():
    mem = BounceMemory(2, 0.16, 2, "cpu")
    now = run(mem, bounce(vy_seen=0.1, rel=(0.0, 0.2, 0.02)))  # off the table, 20 cm beside the blade
    assert not mem.obs(now).any()
