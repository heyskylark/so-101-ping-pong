"""Export a trained bounce policy for deployment: `policy.onnx` (deterministic actor with its observation normalizer,
input "obs" (1, D), output "actions" (1, 5)) and `policy.json` (everything the deploy loop must reproduce: the actor
observation layout, action mapping, joint order, ready pose, joint limits, control rate).

Checks the export: runs the policy in the env, feeds the recorded actor observations to onnxruntime and compares
with the PyTorch actions.

Run (from sim/):
  uv run scripts/export_policy.py logs/rsl_rl/pingpong_residual/<run>/model_1999.pt --name residual_v1
  -> policies/residual_v1/{policy.onnx, policy.json}
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import pingpong_sim.task  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pingpong_sim.robot import ARM_JOINTS, FIRMWARE_P
from pingpong_sim.task.state import get

POLICIES = Path(__file__).resolve().parents[1] / "policies"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--name", required=True, help="output directory under sim/policies/")
    ap.add_argument("--task", default="PingPong-Bounce-Residual")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300, help="control steps recorded for the ONNX check")
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(args.task, play=True)  # no observation noise
    env_cfg.scene.num_envs = args.envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    venv = RslRlVecEnvWrapper(env)
    runner = MjlabOnPolicyRunner(venv, asdict(load_rl_cfg(args.task)), device=device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    out = POLICIES / args.name
    runner.export_policy_to_onnx(str(out), "policy.onnx")

    # Record actor observations and PyTorch actions, then replay them through onnxruntime.
    obs = venv.get_observations()
    xs, ys = [], []
    with torch.inference_mode():
        for _ in range(args.steps):
            a = policy(obs)
            xs.append(obs["actor"].cpu().numpy())
            ys.append(a.cpu().numpy())
            obs, *_ = venv.step(a)
    x, y = np.concatenate(xs).astype(np.float32), np.concatenate(ys)
    sess = ort.InferenceSession(str(out / "policy.onnx"), providers=["CPUExecutionProvider"])
    y_onnx = np.concatenate([sess.run(["actions"], {"obs": x[i:i + 1]})[0] for i in range(len(x))])
    diff = np.abs(y_onnx - y).max()

    st = get(env)
    om = env.observation_manager
    arm_cfg = env_cfg.actions["arm"]
    terms = []
    for name, dims, cfg in zip(om.active_terms["actor"], om.group_obs_term_dim["actor"], om._group_obs_term_cfgs["actor"]):
        h = max(cfg.history_length, 1)
        terms.append({"name": name, "dim": int(np.prod(dims)) // h, "history": h,
                      "doc": (inspect.getdoc(cfg.func) or "policy output of the previous steps (raw, unscaled)")
                      .split("\n\n")[0].replace("\n", " ")})
    spec = {
        "task_id": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve().relative_to(Path.cwd().resolve())),
        "control_dt": env.step_dt,
        "joints": list(ARM_JOINTS),
        "firmware_p": [FIRMWARE_P[j] for j in ARM_JOINTS],
        "q_ready": st.q_ready.tolist(),
        "joint_lower": st.kin.lo.tolist(),
        "joint_upper": st.kin.hi.tolist(),
        "action": {
            "dim": 5,
            "scale": arm_cfg.scale,
            "clip": arm_cfg.clip,
            "residual": arm_cfg.residual,
            "max_rate": arm_cfg.max_rate,  # rad/s per joint, or null: how fast a goal may move
            "goal": ("clip(bouncer_goal + scale * clip(action, -clip, clip), joint_lower, joint_upper)" if arm_cfg.residual
                     else "clip(q_ready + scale * clip(action, -clip, clip), joint_lower, joint_upper)")
                    + ("; then at most max_rate x control_dt from the last goal (from the encoders after a reset)"
                       if arm_cfg.max_rate is not None else ""),
        },
        "observation": {
            "dim": int(x.shape[1]),
            "layout": ("terms concatenated in this order; a term with history h contributes h frames, oldest first; "
                       "after a reset every frame equals the first value"),
            "terms": terms,
        },
        # Geometry and planning settings the deploy loop must match (observations are relative to these).
        "task": {k: getattr(st.cfg, k) for k in ("target_xy", "hit_z", "apex", "trust_vel_std", "use_bouncer")},
        # What the ball filter is told about each camera's detections (the rig's calibration files say otherwise
        # unless run_robot.py applies this).
        "camera_noise": {c.name: {"pixel": c.assumed_pixel_noise, "diameter": c.assumed_diameter_noise,
                                  "angle_floor": c.assumed_angle_floor, "size_floor": c.assumed_size_floor}
                         for c in st.cfg.cameras},
        "bouncer": {k: v for k, v in asdict(st.ctrl.bouncer.cfg).items() if isinstance(v, (int, float, bool, tuple))},
        "controller": {k: getattr(st.ctrl.cfg, k) for k in ("history_len", "history_stride", "memory_slots",
                                                            "memory_delay")},
        "filter": {"restitution": st.ctrl.cfg.filter.restitution},
        "onnx_check": {"samples": int(len(x)), "max_abs_diff": float(diff)},
    }
    (out / "policy.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"exported {out}/policy.onnx (obs {x.shape[1]}, actions 5); ONNX vs PyTorch over {len(x)} samples: "
          f"max |diff| {diff:.2e}")
    if diff > 1e-4:
        raise SystemExit("ONNX output differs from PyTorch")


if __name__ == "__main__":
    main()
