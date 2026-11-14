"""Run the scripted bouncer (pingpong_sim/bouncer.py) in many worlds and report how long it keeps the ball up.

Each world: arm at the ready pose, ball dropped from `--drop` above the hitting plane with a few cm of horizontal
offset, controller at 100 Hz. Ball state: the true state (optionally delayed and noisy), or with `--cameras` the
deployment pipeline: simulated cameras (pingpong_sim/camera_sim.py) -> BallFilter (pingpong_sim/ball_filter.py),
which also reports tracking error and camera visibility. A world fails when the ball falls 10 cm below the hitting
plane or drifts 15 cm from the target. Worlds are split into groups, one per `--lead` value.

Run (from sim/):
  uv run scripts/run_bouncer.py                                # fitted servos, true ball state
  uv run scripts/run_bouncer.py --randomize                    # randomized servos (training distribution)
  uv run scripts/run_bouncer.py --ball-latency 30 --ball-noise 2
  uv run scripts/run_bouncer.py --cameras overhead wrist       # camera rig + filter
  MUJOCO_GL=egl uv run scripts/run_bouncer.py --video          # + build/bouncer.mp4 of world 0
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from pingpong_sim.ball import BALL_RADIUS, SIM_DT
from pingpong_sim.ball_filter import BallFilter, BallFilterCfg
from pingpong_sim.bouncer import BouncerCfg, ScriptedBouncer
from pingpong_sim.camera_sim import OVERHEAD, WRIST, CameraRig, ideal
from pingpong_sim.controller import ArmHistory
from pingpong_sim.harness import make_harness, step
from pingpong_sim.kinematics import ArmKinematics
from pingpong_sim.robot import ARM_JOINTS, READY_JOINT_POS
from pingpong_sim.servo_actuator import encoder_pos

CONTROL_DT = 0.01
TRUST_VEL_STD = 0.5  # m/s, RMS over axes (fresh track: 3; after two frames: ~0.1)
BUILD = Path(__file__).resolve().parents[1] / "build"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--lead", type=float, nargs="+", default=None,
                    help="s, plan look-ahead per group (default: the tracking model's delay)")
    ap.add_argument("--apex", type=float, default=0.25)
    ap.add_argument("--stroke-time", type=float, help="s, the stroke's constant-acceleration time (default 0.12)")
    ap.add_argument("--follow", type=float, help="s, follow-through back to ready after impact (default 0.15)")
    ap.add_argument("--impact-bias", type=float, default=0.0, help="s, added to every predicted impact time")
    ap.add_argument("--stroke-joint", choices=["elbow_flex", "wrist_flex"], default="elbow_flex",
                    help="the one joint that makes the stroke")
    ap.add_argument("--drop", type=float, default=0.30, help="ball release height above the hitting plane [m]")
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--ball-latency", type=float, default=0.0, help="ms, rounded to control steps")
    ap.add_argument("--ball-noise", type=float, default=0.0, help="mm position noise (velocity noise scaled to it)")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-feedforward", action="store_true", help="send the plan itself as the servo goal")
    ap.add_argument("--cameras", nargs="*", choices=["overhead", "wrist"], default=[],
                    help="ball state from these simulated cameras through the ball filter")
    ap.add_argument("--overhead-long", choices=["x", "y"], default="y",
                    help="overhead image long side across the arm (x) or along the view tilt (y, recommended)")
    ap.add_argument("--wrist-long", choices=["x", "y"], default="y",
                    help="wrist image long side across the blade (x) or along the blade normal (y, recommended)")
    ap.add_argument("--ideal-cameras", action="store_true",
                    help="no camera noise or randomization: exact calibration, fixed 30 ms latency")
    ap.add_argument("--predictor-gravity", type=float,
                    help="gravity [m/s^2] the ball filter and the impact predictor assume (the sim ball keeps 9.81): "
                         "> 9.81 predicts impacts early, as the real rig did (~28 ms)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="other BouncerCfg fields, e.g. reach=0.03 max_tilt_deg=8 steer_gain=0.5 min_approach=0.08")
    ap.add_argument("--goal-rate", type=float, nargs="+", metavar="DEG_S",
                    help="limit on how fast the servo goals may move: one value, or one per joint")
    ap.add_argument("--pose", type=float, nargs=2, metavar=("X", "Z"),
                    help="ready pose with the blade centre X m in front of the base and Z m up, level (default "
                         "robot.READY_JOINT_POS: 0.28, 0.31); the bounce target moves with it")
    ap.add_argument("--overhead-pos", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="overhead camera optical centre [m], aimed at the bounce point (default 0.58 0 0.55, 25 deg tilt)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    if args.lead is None:
        import json

        from pingpong_sim.bouncer import TRACKING_MODEL

        args.lead = [float(json.loads(TRACKING_MODEL.read_text()).get("delay_s", 0.0))]
    n_groups = len(args.lead)
    n = args.envs - args.envs % n_groups
    h = make_harness(num_envs=n, randomize=args.randomize)
    dev, m = h.sim.device, h.mj_model
    if args.predictor_gravity is not None:
        import pingpong_sim.ball_filter as ball_filter_mod
        import pingpong_sim.bouncer as bouncer_mod

        ball_filter_mod.G = bouncer_mod.G = args.predictor_gravity
    timing = {k: v for k, v in (("stroke_time", args.stroke_time), ("follow", args.follow)) if v is not None}
    timing["impact_bias"] = args.impact_bias
    overrides = {}
    for kv in args.set:  # e.g. reach=0.03 max_tilt_deg=8
        key, val = kv.split("=")
        overrides[key] = type(getattr(BouncerCfg(), key))(float(val))
    cfg = BouncerCfg(**{**dict(apex=args.apex, feedforward=not args.no_feedforward,
                               stroke_joint=ARM_JOINTS.index(args.stroke_joint), **timing), **overrides})
    kin = ArmKinematics(m, device=dev)
    ids, _ = h.robot.find_joints(ARM_JOINTS, preserve_order=True)
    q_ready = torch.tensor([READY_JOINT_POS[j] for j in ARM_JOINTS], device=dev)
    if args.pose:  # a ready pose other than robot.READY_JOINT_POS: blade centre (x, z), face level
        from pingpong_sim.real.arm import ArmModel

        seeds = (q_ready.cpu().numpy(), np.array([0.0, -1.2, 0.2, 1.0, 0.05]), np.array([0.0, -1.5, 1.2, 0.3, 0.05]))
        sols = []
        for seed in seeds:
            try:
                sols.append(ArmModel().ik(args.pose[0], args.pose[1], seed))
            except RuntimeError:
                pass
        q = min(sols, key=lambda s: abs(s[2]))  # the straighter-elbow solution
        print("ready pose [deg]:", np.degrees(q).round(1))
        q_ready = torch.tensor(q, device=dev, dtype=torch.float32)
        cfg = replace(cfg, target_xy=(args.pose[0], 0.0), hit_z=args.pose[1])
    group = torch.arange(n, device=dev) * n_groups // n
    lead = torch.tensor(args.lead, device=dev)[group]
    ctrl = ScriptedBouncer(cfg, kin, q_ready, n, lead=lead)

    # Start: arm at ready, ball above the target with a few cm of offset, at rest.
    q0 = q_ready.repeat(n, 1)
    h.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), joint_ids=ids)
    h.robot.set_joint_position_target(q0, joint_ids=ids)
    h.robot.reset()
    z_plane = cfg.hit_z + BALL_RADIUS
    root = torch.zeros(n, 13, device=dev)
    root[:, :2] = torch.tensor(cfg.target_xy, device=dev) + 0.02 * torch.randn(n, 2, device=dev)
    root[:, 2] = z_plane + args.drop
    root[:, 3] = 1.0
    h.ball.write_root_state_to_sim(root)
    h.sim.forward()
    jb = m.joint("ball/ball_joint").id
    qa, va = int(m.jnt_qposadr[jb]), int(m.jnt_dofadr[jb])

    steps_per_ctrl = int(round(CONTROL_DT / SIM_DT))
    lag = int(round(args.ball_latency / 1e3 / CONTROL_DT))
    history: deque = deque(maxlen=lag + 1)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    t_fail = torch.full((n,), args.seconds, device=dev)
    bounces = torch.zeros(n, dtype=torch.long, device=dev)
    prev_vz = torch.zeros(n, device=dev)
    apex_cur = torch.zeros(n, device=dev)
    apexes, impacts, speeds, tilts = [], [], [], []
    motion, q_prev = [], q_ready.repeat(n, 1)
    frames, renderer, md = [], None, None
    if args.video:
        renderer, md = mujoco.Renderer(m, 480, 640), mujoco.MjData(m)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.30, 0.0, 0.30]
        cam.distance, cam.azimuth, cam.elevation = 1.1, 140, -15
    target = torch.tensor(cfg.target_xy, device=dev)
    rig = filt = None
    if args.cameras:
        over = replace(OVERHEAD, long_axis=args.overhead_long)
        if args.overhead_pos:  # e.g. the tower on a stand, moved with the bounce point: aimed at 8 cm above the blade
            aim = np.array([*cfg.target_xy, cfg.hit_z + 0.08]) - np.array(args.overhead_pos)
            over = replace(over, pos=tuple(args.overhead_pos), axis=tuple(float(v) for v in aim / np.linalg.norm(aim)))
        cams = {"overhead": over, "wrist": replace(WRIST, long_axis=args.wrist_long)}
        if args.ideal_cameras:
            cams = {k: ideal(c) for k, c in cams.items()}
        arm = ArmHistory(kin, n, CONTROL_DT, dev)
        rig = CameraRig([cams[c] for c in args.cameras], n, CONTROL_DT, arm.hist, dev)
        filt = BallFilter(BallFilterCfg(), n, dev)
        site = m.site("robot/blade_face_cap").id
        wcam = m.camera("robot/wrist_cam").id
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=dev))  # MuJoCo camera axes -> OpenCV
        stats = {"err_p": [], "err_v": [], "tracked": [], "visible": [], "arrived": [], "valid": [],
                 "std_p": [], "false": [], "age": []}

    for k in range(int(args.seconds / CONTROL_DT)):
        now = k * CONTROL_DT
        p = h.sim.data.qpos[:, qa:qa + 3].clone()
        v = h.sim.data.qvel[:, va:va + 3].clone()
        if rig is not None:
            d = h.sim.data
            face = d.site_xpos[:, site]
            normal = d.site_xmat[:, site].reshape(n, 3, 3)[:, :, 2]
            wR = d.cam_xmat[:, wcam].reshape(n, 3, 3) @ flip
            arm.push(now, encoder_pos(h.robot))
            rig.record(now, p, v, face, normal, wR, d.cam_xpos[:, wcam])
            det = rig.detect(now)
            for c in range(len(args.cameras)):
                filt.update(det["t"][c], det["R"][c], det["p"][c], det["z"][c], det["sigma"][c], det["valid"][c], arm.hist)
            p_obs, v_obs, p_std, v_std, age, tracked = filt.estimate(now, arm.hist)
            # Plan only on a converged track (a fresh one has velocity 0 +- 3 m/s); otherwise nothing to hit.
            trusted = tracked & (v_std < TRUST_VEL_STD)
            q_cmd = ctrl.step(now, torch.where(trusted[:, None], p_obs, torch.full_like(p_obs, -1.0)), v_obs)
            if now > 1.0:  # tracking statistics over alive worlds, ball above the blade
                flying = alive & (p[:, 2] > z_plane)
                ok = flying & tracked
                stats["err_p"].append((p_obs - p).norm(dim=1)[ok])
                stats["err_v"].append((v_obs - v).norm(dim=1)[ok])
                stats["std_p"].append(p_std[ok])
                stats["age"].append(age[ok])
                stats["tracked"].append(tracked[flying].float())
                for key in ("visible", "arrived", "valid", "false"):
                    stats[key].append((rig.diag[key] & flying).sum(1))  # per camera
        else:
            history.append((p, v))
            p_obs, v_obs = history[0]
            if args.ball_noise:
                s = args.ball_noise / 1e3
                p_obs = p_obs + s * torch.randn_like(p_obs)
                v_obs = v_obs + 10 * s * torch.randn_like(v_obs)  # rough KF velocity noise for that position noise
            q_cmd = ctrl.step(now, p_obs, v_obs, obs_time=now - (len(history) - 1) * CONTROL_DT)
        if args.goal_rate:  # deg/s: the goal may not move faster than this (the arm can't, and it shakes trying)
            step_max = torch.tensor(np.radians(args.goal_rate) * CONTROL_DT, device=dev, dtype=torch.float32).expand(5)
            q_cmd = q_prev + torch.maximum(torch.minimum(q_cmd - q_prev, step_max), -step_max)
        q_prev = q_cmd.clone()
        motion.append(encoder_pos(h.robot)[:32].clone())
        h.robot.set_joint_position_target(q_cmd, joint_ids=ids)
        for _ in range(steps_per_ctrl):
            step(h)

        p = h.sim.data.qpos[:, qa:qa + 3]
        vz = h.sim.data.qvel[:, va + 2]
        apex_cur = torch.maximum(apex_cur, p[:, 2])
        bounced = alive & (prev_vz < 0) & (vz > 0) & (p[:, 2] < z_plane + 0.05)
        if bounced.any():
            bounces += bounced.long()
            idx = bounced.nonzero(as_tuple=True)[0]
            impacts.append(torch.stack([group[idx].float(), (p[idx, :2] - target).norm(dim=1)], 1))
            speeds.append(torch.stack([group[idx].float(), ctrl.plan_speed[idx], ctrl.plan_tilt[idx]], 1))
            if now > 1.0:
                apexes.append(torch.stack([group[idx].float(), apex_cur[idx] - z_plane], 1))
            apex_cur[idx] = 0.0
        prev_vz = vz.clone()
        failed = alive & ((p[:, 2] < z_plane - 0.10) | ((p[:, :2] - target).norm(dim=1) > 0.15))
        t_fail = torch.where(failed, torch.full_like(t_fail, now), t_fail)
        alive &= ~failed

        if renderer is not None and k % 3 == 0 and now < 10.0:
            md.qpos[:] = h.sim.data.qpos[0].cpu().numpy()
            mujoco.mj_forward(m, md)
            renderer.update_scene(md, cam)
            frames.append(renderer.render())

    report(args, group, n_groups, alive, t_fail, bounces, apexes, impacts, speeds)
    Q = torch.stack(motion).cpu().numpy()  # (T, 32, 5) encoder angles of the first worlds
    dQ = np.abs(np.diff(Q, axis=0)) / CONTROL_DT
    print("arm motion (first 32 worlds, whole run), joints pan / lift / elbow / wrist_flex / roll:\n"
          f"  speed p50 {np.degrees(np.percentile(dQ, 50, axis=(0, 1))).round(0)} p95 "
          f"{np.degrees(np.percentile(dQ, 95, axis=(0, 1))).round(0)} deg/s; range p1..p99 "
          f"{np.degrees(np.percentile(Q, 99, axis=(0, 1)) - np.percentile(Q, 1, axis=(0, 1))).round(0)} deg")
    if rig is not None:
        report_tracking(args, stats)
    if frames:
        import imageio.v3 as iio

        BUILD.mkdir(exist_ok=True)
        iio.imwrite(BUILD / "bouncer.mp4", np.stack(frames), fps=1.0 / (3 * CONTROL_DT))
        print("wrote build/bouncer.mp4 (world 0, lead", args.lead[0], "s)")


def report(args, group, n_groups, alive, t_fail, bounces, apexes, impacts, speeds) -> None:
    cat = lambda xs: torch.cat(xs).cpu().numpy() if xs else np.zeros((0, 3))  # noqa: E731
    apexes, impacts, speeds = cat(apexes), cat(impacts), cat(speeds)
    g, alive, t_fail, bounces = (x.cpu().numpy() for x in (group, alive, t_fail, bounces))
    print(f"{'randomized' if args.randomize else 'fitted'} servos, stroke {args.stroke_joint}, apex target {args.apex} m, "
          f"ball latency {args.ball_latency} ms, noise {args.ball_noise} mm, {args.seconds:.0f} s, {len(g)} worlds")
    print("lead ms | still bouncing | bounces p10/p50/p90 | time to failure p50 | apex mean +- std | "
          "impact offset p50/p90 | paddle speed, tilt p50")
    for i, lead in enumerate(args.lead):
        s = g == i
        a = apexes[apexes[:, 0] == i, 1] if len(apexes) else np.zeros(0)
        o = impacts[impacts[:, 0] == i, 1] if len(impacts) else np.zeros(0)
        sp = speeds[speeds[:, 0] == i] if len(speeds) else np.zeros((0, 3))
        pct = lambda x, q: np.percentile(x, q) if len(x) else np.nan  # noqa: E731
        print(f"{lead * 1e3:6.0f}  | {100 * alive[s].mean():6.1f} %       | "
              f"{pct(bounces[s], 10):3.0f} / {pct(bounces[s], 50):3.0f} / {pct(bounces[s], 90):3.0f}        | "
              f"{np.median(t_fail[s]):5.1f} s              | {np.mean(a) if len(a) else np.nan:.3f} +- "
              f"{np.std(a) if len(a) else np.nan:.3f} m    | {1e3 * pct(o, 50):4.0f} / {1e3 * pct(o, 90):4.0f} mm      | "
              f"{pct(sp[:, 1], 50):.2f} m/s, {np.degrees(pct(sp[:, 2], 50)):.1f} deg")


def report_tracking(args, stats) -> None:
    cat = lambda k: torch.cat(stats[k]).cpu().numpy()  # noqa: E731
    tot = lambda k: torch.stack(stats[k]).sum(0).cpu().numpy()  # noqa: E731
    ep, ev, sp, age, trk = cat("err_p"), cat("err_v"), cat("std_p"), cat("age"), cat("tracked")
    arrived, visible, false = tot("arrived"), tot("visible"), tot("false")
    kind = "ideal" if args.ideal_cameras else "randomized"
    print(f"tracking ({', '.join(args.cameras)}, {kind}; long side overhead {args.overhead_long}, wrist "
          f"{args.wrist_long}), alive worlds with the ball above the blade:")
    print(f"  tracked {100 * trk.mean():.1f} % of control steps; position error p50/p90/p99 "
          f"{1e3 * np.percentile(ep, 50):.1f} / {1e3 * np.percentile(ep, 90):.1f} / {1e3 * np.percentile(ep, 99):.1f} mm "
          f"(filter std p50 {1e3 * np.median(sp):.1f} mm); velocity error p50/p90 {np.percentile(ev, 50):.3f} / "
          f"{np.percentile(ev, 90):.3f} m/s; time since last detection p50/p90 {1e3 * np.median(age):.0f} / "
          f"{1e3 * np.percentile(age, 90):.0f} ms")
    for c, name in enumerate(args.cameras):
        print(f"  {name:8s}: ball in view in {100 * visible[c] / max(arrived[c], 1):5.1f} % of frames, "
              f"false detections {100 * false[c] / max(arrived[c], 1):.1f} %")


if __name__ == "__main__":
    main()
