"""Bounce the ball with the real arm (or its CPU MuJoCo stand-in), running an exported policy at 100 Hz.

  uv run scripts/run_robot.py --port /dev/ttyACM0 --policy policies/residual_v1
  uv run scripts/run_robot.py --port /dev/ttyACM0 --scripted          # the scripted bouncer alone
  uv run scripts/run_robot.py --sim --seconds 60                       # arm, ball and cameras simulated

Needs (real arm): the joint map (`record_servos.py map`), camera devices in measurements/calibration/rig.json
(`cameras.py assign`), and the calibration (`calibrate.py`): every camera's lens, the overhead camera's pose, the
wrist camera's mount (else the CAD mount, with a warning), timestamp offsets (else a nominal -30 ms, with a warning).

Startup: the arm session (gain profile, torque on, holds where it is), "go", cameras, a checked move to the ready
pose. Then each 10 ms tick:

1. encoders -> angles (calibrated joint offsets added) and velocities (backward difference over two ticks: the
   Present_Velocity register reads ~0.8x, lags ~20 ms and is noisy);
2. `BounceController.sense`, then the ball detections since the last tick (capture times from the buffer
   timestamps) -> `update` (filter, impact prediction, scripted bouncer);
3. goal: WAIT holds the ready pose until the ball is tracked above the paddle (held or dropped: training starts
   episodes the same way), then PLAY runs the policy (`OnnxPolicy`) or the scripted bouncer. The ball lost (no
   detection for 0.5 s), below the blade or 0.25 m from the target ends the episode: RECOVER ramps back to ready
   (0.7 s), resets filter, bouncer and policy, then WAIT;
4. safety: goals clamped to the joint limits and to 0.15 rad per tick; a goal that would put the blade lower than
   12 cm above the table or tilt it past 45 deg is replaced by the last safe one; the watchdog (joint stuck > 0.5
   rad from its goal for 0.4 s, temperature, supply) aborts;
5. log (measurements/runs/<time>/: log.npz, detections, meta.json).

Ctrl-C: stop, return to the start pose, torque off, gains restored.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import pingpong_sim.task  # noqa: F401  (registers the tasks)
from mjlab.tasks.registry import load_env_cfg
from pingpong_sim.controller import BounceController, OnnxPolicy, controller_cfg_from_policy
from pingpong_sim.kinematics import ArmKinematics
from pingpong_sim.ball import BALL_RADIUS
from pingpong_sim.real.arm import RATE, ArmSession, PlanBuilder, open_bus
from pingpong_sim.real.camera_model import ArmCalib, CameraCalib, apply_wrist_mount, nominal, to_measurements
from pingpong_sim.robot import ARM_JOINTS, READY_JOINT_POS
from pingpong_sim.servo_bus import JOINTS

REPO = Path(__file__).resolve().parents[2]
MAX_STEP = 0.15  # rad per tick
MIN_BLADE_Z = 0.12  # m above the table
MAX_TILT_DEG = 45.0
LOST_AFTER = 0.5  # s without a detection
AWAY = 0.25  # m from the target
RECOVER_S = 0.7
REST = dict(speed=0.15, gap=0.03, lateral=0.06, hold=0.5)  # pop test: ball at rest on the blade [m/s, m, m, s]


def pop_goals(kin: ArmKinematics, q_ready: np.ndarray, speed: float, accel_time: float = 0.15,
              brake: float = 12.0, back: float = 0.5, drop: float = 0.12, lower: float = 0.6,
              pause: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Servo goals (feedforward through the bouncer's tracking model) and the plan (N, 5) at RATE for one pop: the
    face lowers `drop` over `lower` s (minimum jerk, < g: the ball stays on), holds `pause`, then goes straight up,
    level, its speed rising smoothly to `speed` over `accel_time`, then falling smoothly to 0 at `brake` m/s^2 on
    average (1.9x that at the peak: past g, a resting ball leaves near `speed`), then a minimum-jerk return to ready
    over `back` s. A level face reaches only ~0.105 m above ready (0.055-0.355 m at the target), hence the lowered
    start. Smooth profiles keep the feedforward offsets near the bouncer's (an abrupt switch from accelerating to
    braking spiked them to tens of degrees). Joint angles from IK."""
    import json as json_

    from pingpong_sim.bouncer import TRACKING_MODEL

    def smooth(x):
        x = np.clip(x, 0, 1)
        return 10 * x**3 - 15 * x**4 + 6 * x**5

    dt = 1.0 / RATE
    t_b = speed / brake
    t0 = lower + pause  # pop start
    t = np.arange(0.0, t0 + accel_time + t_b + back + 0.2, dt)
    tp = t - t0
    v = speed * (smooth(tp / accel_time) - smooth((tp - accel_time) / t_b))
    z_up = -drop + np.cumsum(v) * dt
    t_top = t0 + accel_time + t_b
    z_top = z_up[np.searchsorted(t, t_top)]
    z = np.where(t < t0, -drop * smooth(t / lower),
                 np.where(t <= t_top, z_up, z_top * (1 - smooth((t - t_top) / back))))
    face0, normal0, _, _ = kin.fk(torch.tensor(q_ready, dtype=torch.float32)[None])
    q, qs = torch.tensor(q_ready, dtype=torch.float32)[None], []
    for dz in z:
        q, _ = kin.ik(face0 + torch.tensor([[0.0, 0.0, float(dz)]]), normal0, q, iters=8)
        qs.append(q[0].numpy().astype(float))
    plan = np.array(qs)
    qd = np.gradient(plan, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    tm = json_.loads(TRACKING_MODEL.read_text())
    A_inv, B, c = np.linalg.inv(np.array(tm["A"])), np.array(tm["B"]), np.array(tm["c"])
    return plan + (qdd + qd @ B.T - c) @ A_inv.T, plan


def load_calibs(names: list[str]) -> tuple[dict[str, CameraCalib], list[str]]:
    calibs, warnings = {}, []
    for n in names:
        if CameraCalib.path(n).exists():
            c = CameraCalib.load(n)
        else:
            c = nominal(n)
            warnings.append(f"{n}: no calibration file, using the sim's lens")
        if "intrinsics" not in c.steps:
            warnings.append(f"{n}: lens not calibrated (calibrate.py intrinsics {n})")
        if c.mount == "world" and c.R is None:
            raise SystemExit(f"{n}: no pose: run `calibrate.py overhead` first")
        if c.mount == "wrist" and c.R is None:
            warnings.append(f"{n}: using the CAD mount (calibrate.py wrist)")
        if "timing" not in c.steps:
            warnings.append(f"{n}: timestamp offset not calibrated, using {1e3 * c.stamp_offset:.0f} ms "
                            f"(calibrate.py timing {n})")
        calibs[n] = c
    return calibs, warnings


class Run:
    def __init__(self, session: ArmSession, cameras, calibs: dict[str, CameraCalib], task_id: str,
                 policy_dir: Path | None, kin: ArmKinematics, apex: float | None = None, pop: float | None = None,
                 bouncer: dict | None = None):
        self.s, self.cams, self.calibs = session, cameras, calibs
        task = load_env_cfg(task_id).events["reset_episode"].params["task"]
        self.task = task
        self.kin = kin
        self.q_ready = torch.tensor([READY_JOINT_POS[j] for j in ARM_JOINTS])
        ccfg = task.controller_cfg()
        if policy_dir:  # the bounce height, stroke and history the policy was trained with
            ccfg = controller_cfg_from_policy(json.loads((policy_dir / "policy.json").read_text()), ccfg)
        ccfg = replace(ccfg, apex=apex if apex is not None else ccfg.apex,
                       bouncer=replace(ccfg.bouncer, **(bouncer or {})))
        self.ctrl = BounceController(ccfg, kin, self.q_ready, 1, 1.0 / RATE, "cpu")
        self.policy = OnnxPolicy(policy_dir, self.ctrl) if policy_dir else None
        self.pop = pop_goals(kin, self.q_ready.numpy(), pop)[0] if pop else None  # pop test: never plays
        self.pops = 0
        self.ids = torch.zeros(1, dtype=torch.long)
        self.log: dict[str, list] = {k: [] for k in ("t", "q", "dq", "goal", "base", "est_p", "est_v", "tracked",
                                                     "trusted", "state", "n_det", "work_ms", "unsafe")}
        self.dets: list[tuple] = []
        self.episodes = 0
        self.bounces = 0

    def safe(self, goal: np.ndarray) -> bool:
        face, normal, _, _ = self.kin.fk(torch.tensor(goal, dtype=torch.float32)[None])
        return bool(face[0, 2] > MIN_BLADE_Z and normal[0, 2] > np.cos(np.radians(MAX_TILT_DEG)))

    def run(self, seconds: float) -> None:
        s, ctrl = self.s, self.ctrl
        t0 = s.bus.now()
        hist_t, hist_q = [], []
        state, state_since, last_goal, rest_since = "WAIT", 0.0, self.q_ready.numpy().copy(), None
        recover_from = last_goal
        ctrl.reset(self.ids, 0.0)
        prev_vz, n = 0.0, 0
        start = s.bus.now()
        while True:
            r = s.read()
            now = r.t - t0
            if now > seconds:
                return
            w0 = time.perf_counter()
            hist_t, hist_q = (hist_t + [now])[-3:], (hist_q + [r.q])[-3:]
            dq = (hist_q[-1] - hist_q[0]) / (hist_t[-1] - hist_t[0]) if len(hist_t) == 3 else np.zeros(5)
            ctrl.sense(now, torch.tensor(r.q, dtype=torch.float32)[None], torch.tensor(dq, dtype=torch.float32)[None])
            dets = self.cams.drain()
            self.dets += [(d.stamp - t0, d.camera, d.u, d.v, d.d) for d in dets]
            ctrl.update(to_measurements(dets, self.calibs, ctrl.arm.hist, t0=t0))
            p, v = ctrl.est_p[0].numpy(), ctrl.est_v[0].numpy()
            tracked = bool(ctrl.tracked[0])
            above = tracked and p[2] > self.task.hit_z + 0.05
            # Episode state machine.
            if state == "WAIT":
                goal = self.q_ready.numpy()
                if self.pop is not None:  # pop test: launch a ball resting on the blade, never play
                    face, _, _, _ = self.kin.fk(torch.tensor(r.q, dtype=torch.float32)[None])
                    face = face[0].numpy()
                    gap = p[2] - face[2] - BALL_RADIUS
                    resting = (tracked and float(np.linalg.norm(v)) < REST["speed"] and -0.01 < gap < REST["gap"]
                               and np.hypot(*(p[:2] - face[:2])) < REST["lateral"])
                    rest_since = (rest_since if resting and rest_since is not None else now) if resting else None
                    if rest_since is not None and now - rest_since > REST["hold"]:
                        state, state_since, rest_since = "POP", now, None
                        self.pops += 1
                        print(f"  {now:6.1f} s: pop {self.pops}")
                elif above and float(ctrl.age[0]) < 0.1:
                    state, state_since = "PLAY", now
                    self.episodes += 1
                    if self.policy:
                        self.policy.reset()
            elif state == "POP":
                k = int(round((now - state_since) * RATE))
                goal = self.pop[min(k, len(self.pop) - 1)]
                if k >= len(self.pop):
                    state, state_since = "WAIT", now
            elif state == "PLAY":
                goal = (self.policy.act() if self.policy else ctrl.goal_base())[0].numpy()
                lost = not tracked or float(ctrl.age[0]) > LOST_AFTER
                low = tracked and p[2] < self.task.hit_z - 0.10
                away = tracked and np.hypot(*(p[:2] - np.array(self.task.target_xy))) > AWAY
                if prev_vz < -0.5 and v[2] > 0.5 and p[2] < self.task.hit_z + 0.1:
                    self.bounces += 1
                if lost or low or away:
                    print(f"  {now:6.1f} s: episode {self.episodes} ends ({'lost' if lost else 'low' if low else 'away'})")
                    state, state_since, recover_from = "RECOVER", now, last_goal.copy()
                    ctrl.reset(self.ids, now)
            else:  # RECOVER
                a = min(1.0, (now - state_since) / RECOVER_S)
                a = 10 * a**3 - 15 * a**4 + 6 * a**5
                goal = recover_from + a * (self.q_ready.numpy() - recover_from)
                # The ball back above the paddle and falling (e.g. it left the overhead view and came down on the
                # blade): play again once the arm is most of the way back to ready.
                again = (a > 0.5 and tracked and float(ctrl.age[0]) < 0.1 and v[2] < 0 and
                         p[2] > self.task.hit_z + 0.03 and np.hypot(*(p[:2] - np.array(self.task.target_xy))) < 0.12)
                if again:
                    state, state_since = "PLAY", now
                    self.episodes += 1
                    if self.policy:
                        self.policy.reset()
                elif a >= 1.0:
                    state, state_since = "WAIT", now
            prev_vz = float(v[2]) if tracked else 0.0
            # Safety.
            goal = last_goal + np.clip(goal - last_goal, -MAX_STEP, MAX_STEP)
            unsafe = not self.safe(goal)
            if unsafe:
                goal = last_goal
            sent = s.command(goal)
            last_goal = sent
            s.guard(sent, r, max_error=0.5)
            work = 1e3 * (time.perf_counter() - w0)
            for k, val in zip(self.log, (now, r.q, dq, sent, ctrl.goal_base()[0].numpy(), p, v, tracked,
                                          bool(ctrl.trusted[0]), state, len(dets), work, unsafe)):
                self.log[k].append(val)
            n += 1
            nxt = start + n / RATE
            if s.bus.now() > nxt:  # overran: skip the missed ticks rather than run late ones back to back
                n = int((s.bus.now() - start) * RATE) + 1
                nxt = start + n / RATE
            s.bus.sleep_until(nxt)

    def save(self, out: Path, meta: dict) -> None:
        out.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.array(v) for k, v in self.log.items()}
        np.savez_compressed(out / "log.npz", **arrays,
                            detections=np.array([(t, u, v, d) for t, _, u, v, d in self.dets]).reshape(-1, 4),
                            detection_camera=np.array([c for _, c, _, _, _ in self.dets]))
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")

    def summary(self) -> str:
        L = {k: np.array(v) for k, v in self.log.items()}
        if not len(L["t"]):
            return "nothing ran"
        dts = np.diff(L["t"])
        per_cam = {}
        for _, c, *_ in self.dets:
            per_cam[c] = per_cam.get(c, 0) + 1
        span = L["t"][-1] - L["t"][0]
        return (f"{len(L['t'])} ticks over {span:.0f} s: period p50 {1e3 * np.median(dts):.1f} ms, p99 "
                f"{1e3 * np.percentile(dts, 99):.1f} ms; work p50 {np.median(L['work_ms']):.1f} ms, p99 "
                f"{np.percentile(L['work_ms'], 99):.1f} ms\n"
                f"  detections/s: " + ", ".join(f"{c} {k / span:.1f}" for c, k in per_cam.items()) + "\n"
                f"  episodes {self.episodes}, pops {self.pops}, bounces (from the estimate) {self.bounces}, ticks tracked "
                f"{100 * L['tracked'].mean():.0f} %, unsafe goals replaced {int(L['unsafe'].sum())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="servo bus serial port, e.g. /dev/ttyACM0")
    src.add_argument("--sim", action="store_true", help="CPU MuJoCo arm, ball and cameras (virtual time)")
    pol = ap.add_mutually_exclusive_group()
    pol.add_argument("--policy", default="policies/residual_v1", help="exported policy directory")
    pol.add_argument("--scripted", action="store_true", help="the scripted bouncer alone")
    ap.add_argument("--cameras", nargs="+", default=["overhead", "wrist"])
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", help="log directory (default measurements/runs/<time>, sim: sim/build/runs/<time>)")
    ap.add_argument("--pop", type=float, metavar="SPEED",
                    help="pop test instead of play: a ball resting on the blade is launched straight up at SPEED "
                         "[m/s] (face level, brakes harder than g), again each time it settles: clean free flights")
    ap.add_argument("--apex", type=float, help="scripted bouncer: bounce height above the hitting plane [m] "
                    "(default: the task's 0.25; the overhead camera sees the ball up to ~0.2 m above the paddle)")
    ap.add_argument("--stroke-time", type=float, help="scripted bouncer: stroke time [s] (default 0.12)")
    ap.add_argument("--follow", type=float, help="scripted bouncer: follow-through time [s] (default 0.15)")
    ap.add_argument("--impact-bias", type=float, help="scripted bouncer: added to every predicted impact time [s] "
                    "(real runs: predictions 19-28 ms early -> ~0.025)")
    ap.add_argument("--ball-gravity", type=float,
                    help="EXPERIMENT: gravity [m/s^2] the ball filter and the bouncer's predictor assume (default "
                         "9.81). Real runs predicted impacts ~28 ms early, as if the ball fell at ~8.7 m/s^2")
    args = ap.parse_args()
    torch.set_num_threads(1)
    policy_dir = None if args.scripted or args.pop else Path(args.policy)
    bouncer = {k: v for k, v in (("stroke_time", args.stroke_time), ("follow", args.follow),
                                 ("impact_bias", args.impact_bias)) if v is not None}
    if policy_dir and (args.apex is not None or bouncer):
        raise SystemExit("--apex/--stroke-time/--follow/--impact-bias change the scripted bouncer the policy was "
                         "trained on: use them with --scripted")
    task_id = json.loads((policy_dir / "policy.json").read_text())["task_id"] if policy_dir else "PingPong-Bounce-Residual"

    if args.sim:
        from pingpong_sim.real.sim_backend import CpuWorld, SimBusAdapter, SimCameras

        world = CpuWorld(task_id, cameras=args.cameras)
        world.reset()
        bus = SimBusAdapter(world)
        cams = SimCameras(bus)
        calibs, warnings = cams.calibs(), []
        offsets = None
    else:
        from pingpong_sim.real.cameras import BallCameras, RigConfig

        calibs, warnings = load_calibs(args.cameras)
        bus = open_bus(args.port)
        cams = BallCameras(RigConfig.load(), args.cameras)
        offsets = ArmCalib.load().joint_offsets
    if args.ball_gravity is not None:
        import pingpong_sim.ball_filter as ball_filter_mod
        import pingpong_sim.bouncer as bouncer_mod

        ball_filter_mod.G = bouncer_mod.G = args.ball_gravity
        warnings.append(f"EXPERIMENT: ball filter and bouncer assume gravity {args.ball_gravity:.2f} m/s^2")
    for w in warnings:
        print("warning:", w)

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(args.out) if args.out else (REPO / "sim" / "build" / "runs" / stamp if args.sim
                                           else REPO / "measurements" / "runs" / stamp)
    session = ArmSession(bus, joint_offsets=offsets, confirm=not args.sim)
    run = None
    with session:
        kin = ArmKinematics(session.model.m, prefix="", device="cpu")
        if "wrist" in calibs:
            apply_wrist_mount(kin, calibs["wrist"])
        run = Run(session, cams, calibs, task_id, policy_dir, kin, apex=args.apex, pop=args.pop, bouncer=bouncer)
        if run.pop is not None:
            b = PlanBuilder(np.array([READY_JOINT_POS[j] for j in JOINTS]))
            b.add(run.pop, name="pop")
            problems = session.check(b.build())
            if problems:
                raise SystemExit("pop plan check failed:\n  " + "\n  ".join(problems))
        cams.start()
        try:
            q_ready = np.array([READY_JOINT_POS[j] for j in JOINTS])
            session.move_to(q_ready)
            if args.sim and args.pop:  # the "person" sets the ball down on the paddle centre
                face, _, _, _ = kin.fk(torch.tensor(q_ready, dtype=torch.float32)[None])
                world.launch()
                world.set_ball(face + torch.tensor([[0.0, 0.0, BALL_RADIUS + 0.001]]), torch.zeros(1, 3))
                world.release = world.now
            elif args.sim:
                world.launch()  # the "person" drops the first ball once the arm is ready
            what = (f"the pop test ({args.pop:.2f} m/s)" if args.pop else 'the scripted bouncer' if args.scripted
                    else policy_dir)
            print(f"at ready; running {what} for {args.seconds:.0f} s. " + (
                "Set the ball down on the paddle centre; it is launched each time it settles." if args.pop else
                "Hold a ball still above the paddle, or drop it there."))
            with torch.inference_mode():  # less dispatch per tensor op
                run.run(args.seconds)
            session.move_to(session.start)
        finally:
            cams.stop()
    print(session.status)
    if run is not None:
        print(run.summary())
        run.save(out, {"status": session.status, "policy": str(policy_dir), "task": task_id, "cameras": args.cameras,
                       "backend": "sim" if args.sim else "real", "port": args.port, "warnings": warnings,
                       "calibration": {k: v.steps for k, v in calibs.items()}, "created": stamp})
        print("saved", out)
    return 0 if session.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
