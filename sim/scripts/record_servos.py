"""Servo recording, option A (TODO.md): excite each joint of the SO-101 + paddle and log the response, for fitting
the servo model (BAM), command delay and gear compliance.

Two steps on the real arm, both from sim/ with the arm on USB and 12 V:

  uv run scripts/record_servos.py map    --port /dev/ttyACM0   # once: torque off, you move joints by hand
  uv run scripts/record_servos.py record --port /dev/ttyACM0   # ~9 min of motion

and the same plan against the simulation (no hardware):

  uv run scripts/record_servos.py record --sim

`map` checks each joint's direction against the sim convention and saves measurements/servo/joint_map.json.
`record` writes the ping-pong gain profile (robot.FIRMWARE_P, I = 0, D = 0), plans the motion from wherever the arm
rests, checks the plan in the sim (joint limits, table clearance, blade vs arm), asks for "go", runs it, returns to
the start pose, turns torque off and restores the servos' previous gains. Ctrl-C at any time: the arm freezes and
returns slowly to its start pose; a second Ctrl-C cuts torque at once (the arm drops).

Plan, at three level-blade poses (blade centre 0.40/0.25, 0.35/0.20, 0.45/0.30 m out/up), per joint, the others
holding: steps of 3 sizes both ways (delay, speed under load, overshoot), slow reversals (friction, backlash) and
a 0.3-5 Hz chirp (bandwidth); then 20 s of bounce-like multi-joint motion. Logs: goal and present position,
velocity, load, current, voltage, temperature, host timestamps of each command and read.

  uv run scripts/record_servos.py strokes --port /dev/ttyACM0   # ~40 s, no ball

`strokes` runs the scripted bouncer's own strokes for imaginary balls (dropped on the paddle, then bounced ideally
to 0.25 / 0.20 / 0.15 m at the centre and +-4 cm), with the joint offsets of the calibration, and logs the bouncer's
plan, the feedforward goal sent and the response (true angles): the fast stroke regime the slower recordings above
don't reach (real bounces: paddle peak at ~60 % of the commanded speed, later than the model).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from pingpong_sim.real.arm import (JOINT_MAP, RATE, V_RETREAT, WATCHDOG, Abort, ArmModel, ArmSession, Plan,
                                   PlanBuilder, Watchdog, check_plan, gain_profile, open_bus)
from pingpong_sim.robot import READY_JOINT_POS
from pingpong_sim.servo_bus import JOINTS, RAD_PER_TICK, JointMap, SimBus

REPO = Path(__file__).resolve().parents[2]
POSES = {"ready": (0.40, 0.25), "near_low": (0.35, 0.20), "far_high": (0.45, 0.30)}  # blade centre (x, z) [m]
STEP_AMPS = {j: (0.05, 0.15, 0.30) for j in JOINTS} | {"wrist_roll": (0.10, 0.30, 0.80)}
STEP_HOLD = 0.7  # s
CHIRP = dict(f0=0.3, f1=5.0, seconds=12.0, v_peak=1.5)  # amplitude min(A0, v_peak / (2 pi f))
CHIRP_A0 = {j: 0.12 for j in JOINTS} | {"wrist_roll": 0.30}
REVERSAL = dict(amp=0.05, period=4.0, cycles=2)
BOUNCE = dict(seconds=20.0, amp=0.06, freqs=(0.7, 1.3, 2.1, 3.4))


# Directions of positive motion in the sim convention (checked by `map`).
POSITIVE = {
    "shoulder_pan": "turn the arm clockwise seen from above (toward the robot's right)",
    "shoulder_lift": "tilt the upper arm forward (toward the paddle side)",
    "elbow_flex": "fold the forearm toward the upper arm (paddle tip goes down)",
    "wrist_flex": "tip the paddle down",
    "wrist_roll": "roll the camera side of the blade down",
}
# Sim start pose for --sim: the real arm's folded rest (2026-11-08: lift -109, elbow +97, wrist -101 deg), lift
# clipped to the sim's +-100 deg range. Payload ~180 mm above the table.
SIM_START = np.array([0.0, -1.70, 1.69, -1.76, 0.026])


# ---------------------------------------------------------------------------------------------------- plan

def step_offsets(amps) -> np.ndarray:
    seq = [a * s for a in amps for s in (1, 0, -1, 0)]
    return np.repeat(np.array(seq), int(STEP_HOLD * RATE))


def chirp_offsets(a0: float) -> np.ndarray:
    T, f0, f1 = CHIRP["seconds"], CHIRP["f0"], CHIRP["f1"]
    t = np.arange(int(T * RATE)) / RATE
    k = np.log(f1 / f0) / T
    f = f0 * np.exp(k * t)
    phase = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    amp = np.minimum(a0, CHIRP["v_peak"] / (2 * np.pi * f))
    taper = np.clip(np.minimum(t, T - t) / 0.5, 0, 1)
    return amp * taper * np.sin(phase)


def reversal_offsets() -> np.ndarray:
    a, p = REVERSAL["amp"], REVERSAL["period"]
    t = np.arange(int(p * REVERSAL["cycles"] * RATE)) / RATE
    return a * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * t / p))  # triangle wave from 0


def bounce_offsets(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(BOUNCE["seconds"] * RATE)) / RATE
    taper = np.clip(np.minimum(t, t[-1] - t) / 1.0, 0, 1)
    out = np.zeros((len(t), len(JOINTS)))
    for j in (1, 2, 3, 4):
        phases = rng.uniform(0, 2 * np.pi, len(BOUNCE["freqs"]))
        out[:, j] = sum(np.sin(2 * np.pi * f * t + ph) for f, ph in zip(BOUNCE["freqs"], phases))
    return BOUNCE["amp"] / len(BOUNCE["freqs"]) * 2 * taper[:, None] * out


def build_plan(start: np.ndarray, poses: dict[str, np.ndarray]) -> Plan:
    b = PlanBuilder(start)
    for name, pose in poses.items():
        b.move(pose, name=f"to_{name}", pose=name)
        b.hold(1.0, name=f"settle_{name}", pose=name)
        for j, joint in enumerate(JOINTS):
            common = dict(pose=name, joint=joint)
            b.offsets(pose, j, step_offsets(STEP_AMPS[joint]), name=f"{name}/{joint}/steps", kind="steps", **common)
            b.offsets(pose, j, reversal_offsets(), name=f"{name}/{joint}/reversal", kind="reversal", **common)
            b.offsets(pose, j, chirp_offsets(CHIRP_A0[joint]), name=f"{name}/{joint}/chirp", kind="chirp", **common)
            b.hold(0.5, name=f"{name}/{joint}/rest", pose=name)
    ready = poses["ready"]
    b.move(ready, name="to_ready_bounce", pose="ready")
    b.add(ready + bounce_offsets(np.random.default_rng(0)), name="ready/bounce", kind="bounce", pose="ready")
    b.hold(0.5, name="ready/rest", pose="ready")
    b.move(start, name="to_start")
    b.hold(0.5, name="end")
    return b.build()


# ---------------------------------------------------------------------------------------------------- run

class Recorder:
    def __init__(self, bus, joint_map: JointMap):
        self.bus, self.map = bus, joint_map
        self.rows: dict[str, list] = {k: [] for k in ("t_cmd", "t_read", "goal", "pos", "vel", "load", "volt",
                                                      "temp", "current", "seg", "phase")}
        self.read_failures = 0
        self.t0 = None
        self.watchdog = Watchdog()

    def state_rad(self):
        s = self.bus.read_state()
        return s, self.map.to_rad(s.position)

    def follow(self, q: np.ndarray, seg: np.ndarray, phase: int, watchdog: bool) -> None:
        """Track goal samples at RATE; time drives the index, so a slow loop never makes the goal jump."""
        start = self.bus.now()
        n = 0
        while True:
            k = int((self.bus.now() - start) * RATE)
            if k >= len(q):
                return
            goal = q[k]
            self.bus.write_goal(self.map.to_raw(goal))
            t_cmd = self.bus.now() - self.t0
            try:
                s = self.bus.read_state()
                self.read_failures = 0
            except IOError:
                self.read_failures += 1
                if watchdog and self.read_failures >= WATCHDOG["read_failures"]:
                    raise Abort(f"{self.read_failures} failed reads in a row")
                continue
            t_read = self.bus.now() - self.t0
            pos = self.map.to_rad(s.position)
            vel = np.array([self.map.sign[j] for j in JOINTS]) * s.velocity * RAD_PER_TICK
            for key, v in zip(self.rows, (t_cmd, t_read, goal, pos, vel, s.load, s.voltage, s.temperature, s.current,
                                          seg[k], phase)):
                self.rows[key].append(v)
            if watchdog:
                self.watchdog.check(goal, pos, s.temperature, s.voltage, t_read)
            n += 1
            self.bus.sleep_until(start + n / RATE)

    def retreat(self, target: np.ndarray) -> None:
        _, q_now = self.state_rad()
        b = PlanBuilder(q_now)
        b.move(target, v=V_RETREAT, name="retreat")
        b.hold(0.5, name="retreat_hold")
        plan = b.build()
        self.follow(plan.q, np.full(len(plan.q), -1), phase=1, watchdog=False)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.array(v) for k, v in self.rows.items()}


def summarize(log: dict[str, np.ndarray], plan: Plan) -> None:
    t = log["t_read"]
    dts = np.diff(log["t_cmd"])
    print(f"\nrecorded {len(t)} samples over {t[-1]:.0f} s; loop period median {1e3 * np.median(dts):.1f} ms, "
          f"p99 {1e3 * np.percentile(dts, 99):.1f} ms; read-cmd lag median {1e3 * np.median(log['t_read'] - log['t_cmd']):.1f} ms")
    err = log["goal"] - log["pos"]
    for i, j in enumerate(JOINTS):
        print(f"  {j:14s} |goal-pos| p50 {np.degrees(np.median(np.abs(err[:, i]))):5.2f} deg  "
              f"max {np.degrees(np.abs(err[:, i]).max()):5.1f} deg   peak speed {np.abs(log['vel'][:, i]).max():4.1f} rad/s   "
              f"peak |load| {np.abs(log['load'][:, i]).max() / 10:4.0f} %")
    print(f"  voltage {log['volt'].min():.1f}-{log['volt'].max():.1f} V, temperature max {log['temp'].max():.0f} C")


def save(out: Path, log: dict, plan: Plan, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "log.npz", **log, plan_q=plan.q, plan_seg=plan.seg)
    meta["segments"] = plan.segments
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=lambda o: o.tolist()) + "\n")
    print("saved", out)


def make_bus(args):
    return SimBus(SIM_START) if args.sim else open_bus(args.port)


def cmd_record(args) -> int:
    bus = make_bus(args)
    ident = bus.identify()
    cal = bus.read_calibration()
    for j in JOINTS[:-1]:  # wrist_roll is calibrated as a full turn (0..4095)
        if cal[j]["min"] == 0 and cal[j]["max"] == 4095:
            sys.exit(f"{j} has no position limits: calibrate the arm with LeRobot first")
    mid = {j: (cal[j]["min"] + cal[j]["max"]) / 2 for j in JOINTS}
    if args.sim:
        jmap = bus.map
    elif JOINT_MAP.exists():
        jmap = JointMap.load(JOINT_MAP, mid)
    else:
        sys.exit(f"no joint map at {JOINT_MAP}: run `record_servos.py map --port {args.port}` first")
    servo_limits = np.array([[cal[j]["min"], cal[j]["max"]] for j in JOINTS], float)
    original = bus.read_settings()
    if args.sim and args.d_gain:
        sys.exit("the sim models D = 0 only (BAM); --d-gain is for the real arm")
    gains = gain_profile(args.d_gain)
    print(f"gains for the run: P {'/'.join(str(gains[j]['P_Coefficient']) for j in JOINTS)}, I 0, D {args.d_gain}")

    kin = ArmModel()
    s0 = bus.read_state()
    start = jmap.to_rad(s0.position)
    seed = np.array([READY_JOINT_POS[j] for j in JOINTS])
    poses = {}
    for name, (x, z) in POSES.items():
        poses[name] = seed = kin.ik(x, z, seed)
    plan = build_plan(start, poses)
    print("start pose [deg]: " + ", ".join(f"{j} {np.degrees(v):+.1f}" for j, v in zip(JOINTS, start)))
    print(f"plan: {len(plan.segments)} segments, {plan.seconds / 60:.1f} min; supply {s0.voltage.mean():.1f} V")
    problems = check_plan(plan, kin, servo_limits, jmap, near_start={"start", f"to_{next(iter(POSES))}", "to_start", "end"})
    if problems:
        print("plan check FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("plan check passed: joint and servo limits, payload >= 20 mm above the table, blade clear of the arm")

    if not args.sim:
        if any(original[j]["Torque_Enable"] for j in JOINTS):
            print("\nTorque is on now. Writing the gains turns it off for a moment: the arm goes limp, so it must be "
                  "resting on something (or hold it) before you start.")
        print("\nThe arm will move through the plan. Keep clear of it; Ctrl-C stops it and brings it back here.")
        if input('type "go" to start: ').strip() != "go":
            return 1
    rec = Recorder(bus, jmap)
    status = "complete"
    torque_on = False
    first_sigint = []

    def on_sigint(signum, frame):
        # `uv run` may forward the terminal's Ctrl-C to us on top of the one we get directly: signals within 1 s of
        # the first count as the same press.
        now = time.monotonic()
        if not first_sigint:
            first_sigint.append(now)
            raise KeyboardInterrupt
        if now - first_sigint[0] > 1.0:
            bus.torque(False)
            print("\nsecond Ctrl-C: torque off")
            raise SystemExit(130)

    signal.signal(signal.SIGINT, on_sigint)
    try:
        bus.write_settings(gains)  # torque off while the arm rests
        rec.t0 = bus.now()
        # Enable torque holding the current position.
        bus.write_goal(bus.read_state().position)
        bus.torque(True)
        torque_on = True
        rec.follow(plan.q, plan.seg, phase=0, watchdog=True)
    except (Abort, KeyboardInterrupt) as e:
        status = f"aborted: {e}" if isinstance(e, Abort) else "aborted: Ctrl-C"
        if torque_on:
            print(f"\n{status} -> returning to the start pose")
            try:
                rec.retreat(start)
            except Exception as e2:  # bus trouble while returning: stop holding rather than fight it
                print(f"return failed ({e2}): torque off")
    except Exception as e:
        status = f"error: {type(e).__name__}: {e}"
        print(f"\n{status} -> torque off")
    finally:
        try:
            bus.torque(False)
        finally:
            try:
                bus.write_settings({j: {k: original[j][k] for k in gains[j]} for j in JOINTS})
                print("previous gains restored")
            except Exception as e:
                print(f"could NOT restore the previous gains ({e}); they were: "
                      + ", ".join(f"{j} " + "/".join(str(original[j][k]) for k in gains[j]) for j in JOINTS))
            bus.close()
    if not rec.rows["t_read"]:
        print(f"{status}; nothing recorded")
        return 2
    log = rec.arrays()
    summarize(log, plan)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(args.out) if args.out else (REPO / "sim" / "build" / "servo_dryrun" / stamp if args.sim
                                           else REPO / "measurements" / "servo" / stamp)
    save(out, log, plan, dict(
        backend="sim" if args.sim else "real", port=None if args.sim else args.port, status=status, rate_hz=RATE,
        joints=JOINTS, identify=ident, calibration=cal, joint_map={"sign": jmap.sign, "offset": jmap.offset, "mid": mid},
        gains_written=gains, gains_original=original, poses={k: v for k, v in poses.items()}, start=start,
        watchdog=WATCHDOG, created=stamp))
    return 0 if status == "complete" else 2


def cmd_map(args) -> int:
    if args.sim:
        sys.exit("map is for the real arm")
    bus = make_bus(args)
    bus.identify()
    cal = bus.read_calibration()
    mid = {j: (cal[j]["min"] + cal[j]["max"]) / 2 for j in JOINTS}
    jmap = JointMap(mid=mid)
    print("Torque goes OFF: rest the arm on something or hold it; you will move each joint by hand.")
    input("Enter when ready... ")
    bus.torque(False)
    try:
        for j in JOINTS:
            i = JOINTS.index(j)
            while True:
                before = bus.read_state().position[i]
                input(f"\n{j}: {POSITIVE[j]} by about 20 deg, hold it there, then press Enter ")
                delta = (bus.read_state().position[i] - before) * RAD_PER_TICK
                if abs(delta) < np.radians(5):
                    print(f"  only moved {np.degrees(delta):+.1f} deg; try again")
                    continue
                jmap.sign[j] = 1 if delta > 0 else -1
                print(f"  moved {np.degrees(abs(delta)):.0f} deg -> sign {jmap.sign[j]:+d}")
                break
        input("\nNow put the arm in LeRobot's middle pose: upper arm straight up, forearm level pointing forward, "
              "wrist straight (paddle level, cap side up). Enter when there ")
        q = jmap.to_rad(bus.read_state().position)
        print("  reads [deg]: " + ", ".join(f"{j} {np.degrees(v):+.1f}" for j, v in zip(JOINTS, q)))
        print("  The sim's zero pose is this pose (wrist_roll about +3 deg for a level blade). Several degrees off is"
              " hand placement; more than ~15 deg on a joint means its calibration differs: tell me.")
    finally:
        bus.close()
    jmap.save(JOINT_MAP)
    print("saved", JOINT_MAP)
    return 0


STROKE_SERIES = [(apex, dx, dy) for apex in (0.25, 0.20, 0.15) for dx, dy in ((0.0, 0.0), (0.04, 0.0), (-0.04, 0.0),
                                                                             (0.0, 0.04), (0.0, -0.04))]
STROKE_BOUNCES = 5


def stroke_series(kin_t, q_ready: np.ndarray, apex: float, dx: float, dy: float):
    """The scripted bouncer's goals for one imaginary ball: dropped from 0.25 m above the hitting plane at
    target + (dx, dy), then bounced ideally (to `apex`, steered back as the bouncer intends) STROKE_BOUNCES times.
    Returns feedforward goals and the bouncer's kinematic plan (N, 5) at RATE, and the impact times."""
    import torch

    from pingpong_sim.ball import BALL_RADIUS
    from pingpong_sim.ball_filter import flight_np
    from pingpong_sim.bouncer import BouncerCfg, ScriptedBouncer, apex_speed
    from pingpong_sim.ball import G

    cfg = BouncerCfg(apex=apex)
    b = ScriptedBouncer(cfg, kin_t, torch.tensor(q_ready, dtype=torch.float32), 1)
    b.reset(torch.zeros(1, dtype=torch.long), 0.0)
    tgt = np.array(cfg.target_xy)
    z_ball = cfg.hit_z + BALL_RADIUS
    p = np.array([tgt[0] + dx, tgt[1] + dy, z_ball + 0.25])
    v = np.zeros(3)
    dt = 1.0 / RATE
    goals, plan, impacts = [], [], []
    t = 0.0
    while len(impacts) < STROKE_BOUNCES or t < impacts[-1] + cfg.follow + 0.3:
        g = b.step(t, torch.tensor(p, dtype=torch.float32)[None], torch.tensor(v, dtype=torch.float32)[None])
        goals.append(g[0].numpy().astype(float))
        plan.append(b.evaluate(torch.tensor([t]))[0][0].numpy().astype(float))
        pn, vn = flight_np(p, v, dt)
        if len(impacts) < STROKE_BOUNCES and p[2] >= z_ball > pn[2] and vn[2] < 0:
            impacts.append(t + dt * (p[2] - z_ball) / max(p[2] - pn[2], 1e-9))
            vo = apex_speed(apex)
            pn = np.array([pn[0], pn[1], z_ball])
            vn = np.array([*(cfg.steer_gain * (tgt - pn[:2]) / (2 * vo / G)), vo])
        elif len(impacts) >= STROKE_BOUNCES and pn[2] < z_ball:
            pn, vn = np.array([5.0, 0.0, 0.0]), np.zeros(3)  # gone: the bouncer finishes its follow-through
        p, v = pn, vn
        t += dt
        if t > 15.0:
            raise RuntimeError(f"stroke series apex {apex} ({dx}, {dy}) did not finish")
    return np.array(goals), np.array(plan), impacts


def cmd_strokes(args) -> int:
    from pingpong_sim.kinematics import ArmKinematics
    from pingpong_sim.real.camera_model import ArmCalib

    offsets = None if args.sim else np.array(ArmCalib.load().joint_offsets)
    bus = make_bus(args)
    model = ArmModel()
    kin_t = ArmKinematics(model.m, prefix="", device="cpu")
    q_ready = np.array([READY_JOINT_POS[j] for j in JOINTS])
    goals, plans, seg, segments, impacts = [], [], [], [], []
    for i, (apex, dx, dy) in enumerate(STROKE_SERIES):
        g, pl, imp = stroke_series(kin_t, q_ready, apex, dx, dy)
        hold = np.repeat(q_ready[None], int(1.0 * RATE), axis=0)
        g, pl = np.concatenate([hold, g]), np.concatenate([hold, pl])
        n0 = sum(len(x) for x in goals)
        segments.append(dict(name=f"strokes/apex{apex:.2f}/{dx:+.2f},{dy:+.2f}", kind="strokes", pose="ready",
                             apex=apex, dx=dx, dy=dy, start=n0, n=len(g)))
        impacts += [n0 / RATE + 1.0 + x for x in imp]
        goals.append(g)
        plans.append(pl)
        seg.append(np.full(len(g), i))
    goals, plans, seg = np.concatenate(goals), np.concatenate(plans), np.concatenate(seg)
    print(f"{len(STROKE_SERIES)} series, {len(impacts)} strokes, {len(goals) / RATE:.0f} s")
    log = {k: [] for k in ("t_cmd", "t_read", "goal", "pos", "vel", "load", "volt", "temp", "seg", "phase")}
    status = "complete"
    with ArmSession(bus, joint_offsets=offsets, confirm=not args.sim) as s:
        b = PlanBuilder(s.read().q)
        b.move(q_ready, name="from_start")
        b.hold(1.0, name="settle")
        b.add(goals, name="strokes")
        b.move(s.start, name="to_start")
        problems = s.check(b.build())
        if problems:
            raise SystemExit("stroke plan check failed (the arm didn't move):\n  " + "\n  ".join(problems))
        print("plan check passed: joint and servo limits, payload above the table, blade clear of the arm")
        s.move_to(q_ready)
        t0 = s.bus.now()

        def tick(k, goal, r):
            for key, val in zip(log, (s.t_cmd - t0, r.t - t0, goal, r.q, r.dq, r.load, r.voltage, r.temperature,
                                      seg[k], 0)):
                log[key].append(val)

        try:
            s.follow(goals, on_tick=tick)
        finally:
            s.move_to(s.start)
    arrays = {k: np.array(v) for k, v in log.items()}
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(args.out) if args.out else (REPO / "sim" / "build" / "servo_dryrun" / f"{stamp}_strokes" if args.sim
                                           else REPO / "measurements" / "servo" / f"{stamp}_strokes")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "log.npz", **arrays, plan_goal=goals, plan_q=plans, plan_seg=seg,
                        impacts=np.array(impacts))
    (out / "meta.json").write_text(json.dumps(dict(
        backend="sim" if args.sim else "real", port=None if args.sim else args.port, status=status, rate_hz=RATE,
        joints=JOINTS, angles="true (joint offsets added)", joint_offsets=offsets, segments=segments,
        series=STROKE_SERIES, bounces=STROKE_BOUNCES, created=stamp), indent=1,
        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    err = arrays["goal"] - arrays["pos"]
    print(f"recorded {len(arrays['t_read'])} samples; supply {arrays['volt'].min():.1f}-{arrays['volt'].max():.1f} V, "
          f"temperature max {arrays['temp'].max():.0f} C; |goal - pos| max per joint [deg] "
          + ", ".join(f"{np.degrees(np.abs(err[:, i]).max()):.0f}" for i in range(5)))
    print("saved", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("map", "record", "strokes"))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="servo bus serial port, e.g. /dev/ttyACM0")
    src.add_argument("--sim", action="store_true", help="run against the MuJoCo Warp model")
    ap.add_argument("--out", help="output directory (default: measurements/servo/<time>, sim: sim/build/servo_dryrun/<time>)")
    ap.add_argument("--d-gain", type=int, default=0,
                    help="firmware D for every joint (default 0, what BAM's fits assume; LeRobot uses 32)")
    args = ap.parse_args()
    return {"map": cmd_map, "record": cmd_record, "strokes": cmd_strokes}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
