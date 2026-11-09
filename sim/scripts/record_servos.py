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
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from pingpong_sim.robot import FIRMWARE_P, READY_JOINT_POS, get_spec
from pingpong_sim.servo_bus import JOINTS, RAD_PER_TICK, FeetechBus, JointMap, SimBus

REPO = Path(__file__).resolve().parents[2]
JOINT_MAP = REPO / "measurements" / "servo" / "joint_map.json"
RATE = 100.0  # Hz, command/read loop
V_MOVE = 0.5  # rad/s, peak joint speed of transitions between poses
V_RETREAT = 0.3  # rad/s, return to the start pose
POSES = {"ready": (0.40, 0.25), "near_low": (0.35, 0.20), "far_high": (0.45, 0.30)}  # blade centre (x, z) [m]
STEP_AMPS = {j: (0.05, 0.15, 0.30) for j in JOINTS} | {"wrist_roll": (0.10, 0.30, 0.80)}
STEP_HOLD = 0.7  # s
CHIRP = dict(f0=0.3, f1=5.0, seconds=12.0, v_peak=1.5)  # amplitude min(A0, v_peak / (2 pi f))
CHIRP_A0 = {j: 0.12 for j in JOINTS} | {"wrist_roll": 0.30}
REVERSAL = dict(amp=0.05, period=4.0, cycles=2)
BOUNCE = dict(seconds=20.0, amp=0.06, freqs=(0.7, 1.3, 2.1, 3.4))
TABLE_Z = -0.0024  # base mounting face (bottom of the base mesh) in the robot frame
TABLE_MARGIN = 0.02  # m
LIMIT_MARGIN = 0.05  # rad inside the sim joint range
TICK_MARGIN = 10  # ticks inside the servo position limits
WATCHDOG = dict(error=0.35, seconds=0.4, temp=65.0, volt=10.0, read_failures=5)


def gain_profile(d_gain: int = 0) -> dict[str, dict]:
    """Firmware gains for the recording. Acceleration 0 = no firmware acceleration ramp (what the arm reads now;
    the sim doesn't model a ramp)."""
    return {j: {"P_Coefficient": FIRMWARE_P[j], "I_Coefficient": 0, "D_Coefficient": d_gain, "Acceleration": 0}
            for j in JOINTS}


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

@dataclass
class Plan:
    q: np.ndarray  # (T, 5) goal [rad] at RATE
    seg: np.ndarray  # (T,) segment index
    segments: list[dict] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return len(self.q) / RATE


class PlanBuilder:
    def __init__(self, start: np.ndarray):
        self.q = [start[None].copy()]
        self.seg = [np.array([0])]
        self.segments = [{"name": "start", "kind": "hold"}]

    @property
    def last(self) -> np.ndarray:
        return self.q[-1][-1]

    def add(self, q: np.ndarray, **info) -> None:
        self.segments.append(info)
        self.q.append(q)
        self.seg.append(np.full(len(q), len(self.segments) - 1))

    def move(self, target: np.ndarray, v: float = V_MOVE, **info) -> None:
        delta = target - self.last
        seconds = max(1.0, 1.875 * np.abs(delta).max() / v)  # minimum jerk: peak speed 1.875 delta / T
        s = np.linspace(0, 1, int(seconds * RATE) + 1)[1:]
        s = 10 * s**3 - 15 * s**4 + 6 * s**5
        self.add(self.last + s[:, None] * delta, kind="move", **info)

    def hold(self, seconds: float, **info) -> None:
        self.add(np.repeat(self.last[None], int(seconds * RATE), axis=0), kind="hold", **info)

    def offsets(self, base: np.ndarray, j: int, offsets: np.ndarray, **info) -> None:
        q = np.repeat(base[None], len(offsets), axis=0)
        q[:, j] += offsets
        self.add(q, **info)

    def build(self) -> Plan:
        return Plan(np.concatenate(self.q), np.concatenate(self.seg), self.segments)


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


# ---------------------------------------------------------------------------------------------------- sim checks

class Kinematics:
    """CPU MuJoCo model of the robot for IK and plan checks."""

    PAYLOAD = ("paddle_visual", "holder_base", "holder_cap", "camera_mount", "camera")

    def __init__(self):
        self.m = get_spec().compile()
        self.d = mujoco.MjData(self.m)
        self.adr = np.array([self.m.jnt_qposadr[self.m.joint(j).id] for j in JOINTS])
        self.range = np.array([self.m.jnt_range[self.m.joint(j).id] for j in JOINTS])
        self.face = self.m.site("blade_face_cap").id
        self.holder = self.m.body("paddle_holder").id
        self.blade = self.m.geom("blade_collision").id
        self.paddle = self.m.body("paddle").id
        self.verts = {}
        for name in self.PAYLOAD:
            g = self.m.geom(name).id
            mid = self.m.geom_dataid[g]
            v = self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
            self.verts[g] = v[:: max(1, len(v) // 1500)]

    def forward(self, q: np.ndarray) -> None:
        self.d.qpos[self.adr] = q
        mujoco.mj_forward(self.m, self.d)

    def ik(self, x: float, z: float, seed: np.ndarray) -> np.ndarray:
        """Joint angles putting the cap-side face centre at (x, 0, z), face level and up, paddle pointing along x."""

        def res(q):
            self.d.qpos[self.adr] = q
            mujoco.mj_kinematics(self.m, self.d)
            n = self.d.site_xmat[self.face].reshape(3, 3)[:, 2]
            axis = self.d.xmat[self.holder].reshape(3, 3)[:, 2]
            return np.r_[self.d.site_xpos[self.face] - (x, 0, z), 0.3 * (n - (0, 0, 1)), 0.1 * axis[1]]

        r = least_squares(res, seed, bounds=(self.range[:, 0], self.range[:, 1]), xtol=1e-12, ftol=1e-12)
        if r.cost > 1e-8:
            raise RuntimeError(f"no level-blade IK solution at x={x}, z={z} (cost {r.cost:.1e})")
        return r.x

    def clearance(self) -> float:
        """Lowest payload point (paddle, holder, camera) above the table [m]."""
        z = min((v @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g])[:, 2].min()
                for g, v in self.verts.items())
        return z - TABLE_Z

    def blade_hits(self) -> list[str]:
        """Arm geoms the blade collides with."""
        hits = []
        for c in self.d.contact[: self.d.ncon]:
            if self.blade in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.blade else c.geom1
                if self.m.geom_bodyid[other] != self.paddle:
                    hits.append(self.m.body(self.m.geom_bodyid[other]).name)
        return hits


def check_plan(plan: Plan, kin: Kinematics, servo_limits: np.ndarray, joint_map: JointMap) -> list[str]:
    """Problems with the plan; empty if it's safe to run."""
    problems = []
    lo, hi = kin.range[:, 0] + LIMIT_MARGIN, kin.range[:, 1] - LIMIT_MARGIN
    raw = np.array([joint_map.to_raw(q) for q in plan.q[:: int(RATE // 25)]])
    for i, j in enumerate(JOINTS):
        q = plan.q[:, i]
        motion = plan.q[1:, i] != plan.q[:-1, i]
        if motion.any() and (q.min() < lo[i] - 1e-9 or q.max() > hi[i] + 1e-9):
            # The start pose may sit outside the margin (wherever the arm rests); motion must not go further out.
            start = plan.q[0, i]
            if q.min() < min(lo[i], start) - 1e-6 or q.max() > max(hi[i], start) + 1e-6:
                problems.append(f"{j}: goal {np.degrees(q.min()):.0f}..{np.degrees(q.max()):.0f} deg leaves the sim "
                                f"range {np.degrees(kin.range[i, 0]):.0f}..{np.degrees(kin.range[i, 1]):.0f} deg")
        # As above: the resting start may sit past a limit (the servo clamps that goal); motion must not go further.
        r0 = raw[0, i]
        lo_t, hi_t = servo_limits[i, 0] + TICK_MARGIN, servo_limits[i, 1] - TICK_MARGIN
        if raw[:, i].min() < min(lo_t, r0) - 1 or raw[:, i].max() > max(hi_t, r0) + 1:
            problems.append(f"{j}: goal ticks {raw[:, i].min():.0f}..{raw[:, i].max():.0f} outside the servo limits "
                            f"{servo_limits[i, 0]:.0f}..{servo_limits[i, 1]:.0f}")
    kin.forward(plan.q[0])
    start_clear = kin.clearance()
    if start_clear < -0.01:
        problems.append(f"the start pose puts the payload {-start_clear * 1000:.0f} mm below the table: the joint map "
                        f"doesn't match the arm (rerun `map`)")
        return problems
    worst = {}
    near_start = {"start", f"to_{next(iter(POSES))}", "to_start", "end"}
    for k in range(0, len(plan.q), int(RATE // 50)):
        kin.forward(plan.q[k])
        seg = plan.segments[plan.seg[k]]["name"]
        # Leaving and returning to the start pose may pass as low as the start itself (the arm may rest low).
        floor = min(TABLE_MARGIN, start_clear - 0.005) if seg in near_start else TABLE_MARGIN
        c = kin.clearance()
        if c < floor and (seg not in worst or c < worst[seg][0]):
            worst[seg] = (c, k / RATE)
        hits = kin.blade_hits()
        if hits:
            problems.append(f"{seg} at {k / RATE:.1f} s: blade touches {sorted(set(hits))}")
            break
    for seg, (c, t) in worst.items():
        problems.append(f"{seg} at {t:.1f} s: payload {c * 1000:.0f} mm above the table (min {TABLE_MARGIN * 1000:.0f})")
    return problems


# ---------------------------------------------------------------------------------------------------- run

class Abort(Exception):
    pass


class Recorder:
    def __init__(self, bus, joint_map: JointMap):
        self.bus, self.map = bus, joint_map
        self.rows: dict[str, list] = {k: [] for k in ("t_cmd", "t_read", "goal", "pos", "vel", "load", "volt",
                                                      "temp", "current", "seg", "phase")}
        self.read_failures = 0
        self.over_since = None
        self.t0 = None
        # Single corrupted samples happen on this bus (e.g. one 80 C reading between 30 C ones): temperature and
        # voltage trip on the median of the last 0.5 s.
        self.temp_hist: deque = deque(maxlen=int(0.5 * RATE))
        self.volt_hist: deque = deque(maxlen=int(0.5 * RATE))

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
                self._check(goal, pos, s, t_read)
            n += 1
            self.bus.sleep_until(start + n / RATE)

    def _check(self, goal, pos, s, t) -> None:
        err = np.abs(goal - pos)
        if err.max() > WATCHDOG["error"]:
            self.over_since = self.over_since if self.over_since is not None else t
            if t - self.over_since > WATCHDOG["seconds"]:
                j = JOINTS[int(err.argmax())]
                raise Abort(f"{j} is {np.degrees(err.max()):.0f} deg from its goal for {WATCHDOG['seconds']} s")
        else:
            self.over_since = None
        self.temp_hist.append(s.temperature)
        self.volt_hist.append(s.voltage)
        if len(self.temp_hist) == self.temp_hist.maxlen:
            temp = np.median(np.array(self.temp_hist), axis=0)
            volt = np.median(np.array(self.volt_hist), axis=0)
            if temp.max() > WATCHDOG["temp"]:
                raise Abort(f"{JOINTS[int(temp.argmax())]} at {temp.max():.0f} C (0.5 s median)")
            if volt.min() < WATCHDOG["volt"]:
                raise Abort(f"supply {volt.min():.1f} V (0.5 s median)")

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
    if args.sim:
        return SimBus(SIM_START)
    try:
        bus = FeetechBus(args.port)
        bus.open()
    except Exception as e:  # serial.SerialException or ConnectionError
        ports = sorted(str(p) for p in Path("/dev/serial/by-id").glob("*")) if Path("/dev/serial/by-id").exists() else []
        sys.exit(f"can't open {args.port}: {e}\nserial devices: {ports or 'none (is the arm on USB?)'}")
    return bus


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

    kin = Kinematics()
    s0 = bus.read_state()
    start = jmap.to_rad(s0.position)
    seed = np.array([READY_JOINT_POS[j] for j in JOINTS])
    poses = {}
    for name, (x, z) in POSES.items():
        poses[name] = seed = kin.ik(x, z, seed)
    plan = build_plan(start, poses)
    print("start pose [deg]: " + ", ".join(f"{j} {np.degrees(v):+.1f}" for j, v in zip(JOINTS, start)))
    print(f"plan: {len(plan.segments)} segments, {plan.seconds / 60:.1f} min; supply {s0.voltage.mean():.1f} V")
    problems = check_plan(plan, kin, servo_limits, jmap)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("map", "record"))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="servo bus serial port, e.g. /dev/ttyACM0")
    src.add_argument("--sim", action="store_true", help="run against the MuJoCo Warp model")
    ap.add_argument("--out", help="output directory (default: measurements/servo/<time>, sim: sim/build/servo_dryrun/<time>)")
    ap.add_argument("--d-gain", type=int, default=0,
                    help="firmware D for every joint (default 0, what BAM's fits assume; LeRobot uses 32)")
    args = ap.parse_args()
    return cmd_map(args) if args.command == "map" else cmd_record(args)


if __name__ == "__main__":
    sys.exit(main())
