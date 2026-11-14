"""Start a task's training from a policy trained on a task with fewer observations.

The new task's actor and critic get the trained networks; the first layer's columns of the observation terms the old
task did not have are zero, so training starts from the old policy's behaviour and learns what the new inputs are
worth. (From scratch, inputs that are zero until the policy plays well start with a running std of 0, and the first
non-zero values reach the network as 100 sigma: the 2026-11-13 bounce-memory run stayed at 1.5 bounces where the same
task without the memory reached 5.) The new inputs' normalizer statistics are measured by running the old policy in
the new task.

Run (from sim/):
  uv run scripts/warm_start.py logs/rsl_rl/pingpong_gentle/<run>/model_2999.pt --from-task PingPong-Bounce-Gentle \
      --task PingPong-Bounce-Gentle-Memory --name warm_gentle_v3
  -> logs/rsl_rl/pingpong_gentle_memory/warm_gentle_v3/model_0.pt
  uv run scripts/train.py PingPong-Bounce-Gentle-Memory --env.scene.num-envs 4096 --agent.resume True \
      --agent.load-run warm_gentle_v3 --agent.load-checkpoint model_0.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import pingpong_sim.task  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

LOGS = Path(__file__).resolve().parents[1] / "logs" / "rsl_rl"
GROUPS = {"actor": "actor_state_dict", "critic": "critic_state_dict"}


def make_env(task: str, num_envs: int, device: str) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def layout(env: ManagerBasedRlEnv, group: str) -> dict[str, slice]:
    """Observation term -> its columns in the group's vector."""
    om, out, i = env.observation_manager, {}, 0
    for name, dims in zip(om.active_terms[group], om.group_obs_term_dim[group]):
        n = int(np.prod(dims))
        out[name] = slice(i, i + n)
        i += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="the trained policy (of --from-task)")
    ap.add_argument("--from-task", required=True)
    ap.add_argument("--task", required=True, help="the task to train: --from-task's observation terms and more")
    ap.add_argument("--name", required=True, help="run directory to create under the task's logs")
    ap.add_argument("--envs", type=int, default=256, help="worlds for measuring the new inputs' statistics")
    ap.add_argument("--steps", type=int, default=800, help="control steps for the same")
    ap.add_argument("--count", type=int, default=2_000_000,
                    help="samples the normalizers are taken to rest on (they keep adapting during training)")
    ap.add_argument("--action-std", type=float,
                    help="exploration noise to restart with (default: the trained policy's; training shrinks it to "
                         "~0.01, with which the policy barely tries anything new)")
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    old_env = make_env(args.from_task, 1, device)
    old = {g: layout(old_env, g) for g in GROUPS}
    old_env.close()

    env = make_env(args.task, args.envs, device)
    new = {g: layout(env, g) for g in GROUPS}
    venv = RslRlVecEnvWrapper(env)
    rl_cfg = load_rl_cfg(args.task)
    runner = MjlabOnPolicyRunner(venv, asdict(rl_cfg), device=device)
    src = torch.load(args.checkpoint, map_location=device, weights_only=False)

    added = {}
    for g, key in GROUPS.items():
        missing = [t for t in old[g] if t not in new[g]]
        if missing:
            raise SystemExit(f"{args.task} lacks {g} observation terms of {args.from_task}: {missing}")
        added[g] = [t for t in new[g] if t not in old[g]]
        net = getattr(runner.alg, g)
        sd = {k: v.clone() for k, v in net.state_dict().items()}
        for k, v in src[key].items():
            if k == "mlp.0.weight" or k.startswith("obs_normalizer._"):
                wide = torch.zeros_like(sd[k]) if k == "mlp.0.weight" else sd[k].clone()
                for t, cols in old[g].items():
                    wide[..., new[g][t]] = v[..., cols]
                sd[k] = wide
            elif k != "obs_normalizer.count":
                sd[k] = v.clone()
        if g == "actor" and args.action_std is not None:
            sd["distribution.std_param"] = torch.full_like(sd["distribution.std_param"], args.action_std)
        net.load_state_dict(sd)

    # The old policy in the new task: what the new inputs look like when it plays.
    policy = runner.get_inference_policy(device=device)
    obs = venv.get_observations()
    seen = {g: [] for g in GROUPS}
    with torch.inference_mode():
        for _ in range(args.steps):
            for g in GROUPS:
                seen[g].append(obs[g].clone())
            obs, *_ = venv.step(policy(obs))
    for g in GROUPS:
        x = torch.cat(seen[g])
        norm = getattr(runner.alg, g).obs_normalizer
        stds = []
        for t in added[g]:
            cols = new[g][t]
            norm._mean[0, cols] = x[:, cols].mean(0)
            norm._var[0, cols] = x[:, cols].var(0, unbiased=False)
            stds.append(x[:, cols].std(0))
        norm._std[:] = torch.sqrt(norm._var)
        norm.count.fill_(args.count)
        if stds:
            stds = torch.cat(stds)
            print(f"{g}: {x.shape[1] - len(stds)} -> {x.shape[1]} inputs; new terms {added[g]}, std of their inputs "
                  f"{float(stds.min()):.4f} .. {float(stds.max()):.4f}")
        else:
            print(f"{g}: {x.shape[1]} inputs, the same as {args.from_task}'s")

    out = LOGS / rl_cfg.experiment_name / args.name
    out.mkdir(parents=True, exist_ok=True)
    saved = runner.alg.save()  # a fresh optimizer: the old one's moments belong to the old network
    # The env's step counter goes on (reward curricula are staged on it; the iteration count starts again).
    steps = ((src.get("infos") or {}).get("env_state") or {}).get("common_step_counter", 0)
    saved["iter"], saved["infos"] = 0, {"env_state": {"common_step_counter": steps}}
    torch.save(saved, out / "model_0.pt")
    print("saved", out / "model_0.pt")
    env.close()


if __name__ == "__main__":
    main()
