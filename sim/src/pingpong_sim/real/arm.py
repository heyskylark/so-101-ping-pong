"""The real arm (or a stand-in with the same bus interface) for scripts that move it: gain profile, minimum-jerk
moves checked in the sim before they run (joint and servo limits, table clearance, blade vs arm), a watchdog, and a
session that restores the servos when it ends.

`ArmSession` (a context manager) on enter: identifies the servos, loads the joint map, writes the ping-pong gain
profile (robot.FIRMWARE_P, I = 0, D = 0, Acceleration 0) and enables torque holding the current pose. On exit:
after Ctrl-C or a watchdog abort it returns slowly to the pose it started from; then torque off and the previous
gains restored. A second Ctrl-C within the return cuts torque at once (the arm drops).

Angles are the sim convention. With calibrated joint offsets (overhead calibration), session angles are true angles:
the servos are sent `true - offsets` through the joint map.

Clock: `bus.now()` (time.perf_counter: CLOCK_MONOTONIC on Linux, the clock of the camera buffer timestamps),
virtual time on the sim stand-ins.
"""

from __future__ import annotations

import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from pingpong_sim.robot import FIRMWARE_P, get_spec
from pingpong_sim.servo_bus import JOINTS, RAD_PER_TICK, FeetechBus, JointMap

REPO = Path(__file__).resolve().parents[4]
JOINT_MAP = REPO / "measurements" / "servo" / "joint_map.json"
RATE = 100.0  # Hz, command/read loop
V_MOVE = 0.5  # rad/s, peak joint speed of transitions between poses
V_RETREAT = 0.3  # rad/s, return to the start pose
TABLE_Z = -0.0024  # base mounting face (bottom of the base mesh) in the robot frame
TABLE_MARGIN = 0.02  # m
LIMIT_MARGIN = 0.05  # rad inside the sim joint range
TICK_MARGIN = 10  # ticks inside the servo position limits
WATCHDOG = dict(error=0.35, seconds=0.4, temp=65.0, volt=10.0, read_failures=5)


def gain_profile(d_gain: int = 0) -> dict[str, dict]:
    """Firmware gains the sim models. Acceleration 0 = no firmware acceleration ramp (the sim doesn't model one)."""
    return {j: {"P_Coefficient": FIRMWARE_P[j], "I_Coefficient": 0, "D_Coefficient": d_gain, "Acceleration": 0}
            for j in JOINTS}


class Abort(Exception):
    pass


# ---------------------------------------------------------------------------------------------------- plans

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


# ---------------------------------------------------------------------------------------------------- sim checks

class ArmModel:
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


def check_plan(plan: Plan, model: ArmModel, servo_limits: np.ndarray, joint_map: JointMap,
               near_start: set[str] = frozenset({"start", "from_start", "to_start", "end"}),
               offsets: np.ndarray | None = None) -> list[str]:
    """Problems with the plan; empty if it's safe to run. Segments named in `near_start` (leaving and returning to
    the start pose) may pass as low as the start pose itself (the arm may rest low). `offsets`: calibrated joint
    offsets (plan angles are true angles; the servos are sent plan - offsets)."""
    problems = []
    lo, hi = model.range[:, 0] + LIMIT_MARGIN, model.range[:, 1] - LIMIT_MARGIN
    off = np.zeros(len(JOINTS)) if offsets is None else offsets
    raw = np.array([joint_map.to_raw(q - off) for q in plan.q[:: int(RATE // 25)]])
    for i, j in enumerate(JOINTS):
        q = plan.q[:, i]
        motion = plan.q[1:, i] != plan.q[:-1, i]
        if motion.any() and (q.min() < lo[i] - 1e-9 or q.max() > hi[i] + 1e-9):
            # The start pose may sit outside the margin (wherever the arm rests); motion must not go further out.
            start = plan.q[0, i]
            if q.min() < min(lo[i], start) - 1e-6 or q.max() > max(hi[i], start) + 1e-6:
                problems.append(f"{j}: goal {np.degrees(q.min()):.0f}..{np.degrees(q.max()):.0f} deg leaves the sim "
                                f"range {np.degrees(model.range[i, 0]):.0f}..{np.degrees(model.range[i, 1]):.0f} deg")
        # As above: the resting start may sit past a limit (the servo clamps that goal); motion must not go further.
        r0 = raw[0, i]
        lo_t, hi_t = servo_limits[i, 0] + TICK_MARGIN, servo_limits[i, 1] - TICK_MARGIN
        if raw[:, i].min() < min(lo_t, r0) - 1 or raw[:, i].max() > max(hi_t, r0) + 1:
            problems.append(f"{j}: goal ticks {raw[:, i].min():.0f}..{raw[:, i].max():.0f} outside the servo limits "
                            f"{servo_limits[i, 0]:.0f}..{servo_limits[i, 1]:.0f}")
    model.forward(plan.q[0])
    start_clear = model.clearance()
    if start_clear < -0.01:
        problems.append(f"the start pose puts the payload {-start_clear * 1000:.0f} mm below the table: the joint map "
                        f"doesn't match the arm (rerun `record_servos.py map`)")
        return problems
    worst = {}
    for k in range(0, len(plan.q), int(RATE // 50)):
        model.forward(plan.q[k])
        seg = plan.segments[plan.seg[k]]["name"]
        floor = min(TABLE_MARGIN, start_clear - 0.005) if seg in near_start else TABLE_MARGIN
        c = model.clearance()
        if c < floor and (seg not in worst or c < worst[seg][0]):
            worst[seg] = (c, k / RATE)
        hits = model.blade_hits()
        if hits:
            problems.append(f"{seg} at {k / RATE:.1f} s: blade touches {sorted(set(hits))}")
            break
    for seg, (c, t) in worst.items():
        problems.append(f"{seg} at {t:.1f} s: payload {c * 1000:.0f} mm above the table (min {TABLE_MARGIN * 1000:.0f})")
    return problems


# ---------------------------------------------------------------------------------------------------- run

class Watchdog:
    """Aborts on a joint stuck far from its goal, overheating or a sagging supply. Single corrupted samples happen
    on this bus (e.g. one 80 C reading between 30 C ones): temperature and voltage trip on the median of 0.5 s."""

    def __init__(self):
        self.over_since = None
        self.temp_hist: deque = deque(maxlen=int(0.5 * RATE))
        self.volt_hist: deque = deque(maxlen=int(0.5 * RATE))

    def check(self, goal: np.ndarray, pos: np.ndarray, temperature: np.ndarray, voltage: np.ndarray, t: float,
              max_error: float = WATCHDOG["error"]) -> None:
        err = np.abs(goal - pos)
        if err.max() > max_error:
            self.over_since = self.over_since if self.over_since is not None else t
            if t - self.over_since > WATCHDOG["seconds"]:
                j = JOINTS[int(err.argmax())]
                raise Abort(f"{j} is {np.degrees(err.max()):.0f} deg from its goal for {WATCHDOG['seconds']} s")
        else:
            self.over_since = None
        self.temp_hist.append(temperature)
        self.volt_hist.append(voltage)
        if len(self.temp_hist) == self.temp_hist.maxlen:
            temp = np.median(np.array(self.temp_hist), axis=0)
            volt = np.median(np.array(self.volt_hist), axis=0)
            if temp.max() > WATCHDOG["temp"]:
                raise Abort(f"{JOINTS[int(temp.argmax())]} at {temp.max():.0f} C (0.5 s median)")
            if volt.min() < WATCHDOG["volt"]:
                raise Abort(f"supply {volt.min():.1f} V (0.5 s median)")


def open_bus(port: str):
    """The real bus on `port`; exits with the serial devices found if it can't be opened."""
    try:
        bus = FeetechBus(port)
        bus.open()
    except Exception as e:  # serial.SerialException or ConnectionError
        ports = sorted(str(p) for p in Path("/dev/serial/by-id").glob("*")) if Path("/dev/serial/by-id").exists() else []
        sys.exit(f"can't open {port}: {e}\nserial devices: {ports or 'none (is the arm on USB?)'}")
    return bus


def servo_calibration(bus) -> tuple[dict, dict[str, float]]:
    """The servos' LeRobot calibration and the mid ticks the joint map is relative to."""
    cal = bus.read_calibration()
    for j in JOINTS[:-1]:  # wrist_roll is calibrated as a full turn (0..4095)
        if cal[j]["min"] == 0 and cal[j]["max"] == 4095:
            sys.exit(f"{j} has no position limits: calibrate the arm with LeRobot first")
    return cal, {j: (cal[j]["min"] + cal[j]["max"]) / 2 for j in JOINTS}


@dataclass
class Reading:
    t: float  # bus clock at the read
    q: np.ndarray  # encoder angles, sim convention [rad]
    dq: np.ndarray  # rad/s (Present_Velocity)
    temperature: np.ndarray
    voltage: np.ndarray
    load: np.ndarray


class ArmSession:
    """See the module docstring. `joint_offsets` (rad, added to the mapped encoder angles) come from the overhead
    calibration; `confirm` asks for "go" before the first motion (real arm)."""

    def __init__(self, bus, joint_map: JointMap | None = None, joint_offsets: np.ndarray | None = None,
                 confirm: bool = True, gains: dict | None = None):
        self.bus, self._jmap, self.confirm = bus, joint_map, confirm
        self.offsets = np.zeros(len(JOINTS)) if joint_offsets is None else np.asarray(joint_offsets, float)
        self.gains = gains or gain_profile()
        self.model = ArmModel()
        self.watchdog = Watchdog()
        self.status = "running"
        self.torque_on = False
        self.read_failures = 0
        self._sigint: list[float] = []

    # -- lifecycle
    def __enter__(self) -> ArmSession:
        bus = self.bus
        self.ident = bus.identify()
        self.cal, mid = servo_calibration(bus)
        if self._jmap is not None:
            self.map = self._jmap
        elif hasattr(bus, "map"):  # sim stand-ins carry their own synthetic map
            self.map = bus.map
        elif JOINT_MAP.exists():
            self.map = JointMap.load(JOINT_MAP, mid)
        else:
            sys.exit(f"no joint map at {JOINT_MAP}: run `record_servos.py map --port <port>` first")
        self.servo_limits = np.array([[self.cal[j]["min"], self.cal[j]["max"]] for j in JOINTS], float)
        self.original = bus.read_settings()
        self.start = self.read().q
        if self.confirm:
            if any(self.original[j]["Torque_Enable"] for j in JOINTS):
                print("Torque is on now. Writing the gains turns it off for a moment: the arm goes limp, so it must be "
                      "resting on something (or hold it).")
            print("The arm will move. Keep clear of it; Ctrl-C stops it and brings it back to where it is now.")
            if input('type "go" to start: ').strip() != "go":
                raise SystemExit(1)
        signal.signal(signal.SIGINT, self._on_sigint)
        bus.write_settings(self.gains)  # torque off while the arm rests
        bus.write_goal(bus.read_state().position)  # hold where it is
        bus.torque(True)
        self.torque_on = True
        return self

    def _on_sigint(self, signum, frame):
        # `uv run` may forward the terminal's Ctrl-C to us on top of the one we get directly: signals within 1 s of
        # the first count as the same press.
        now = time.monotonic()
        if not self._sigint:
            self._sigint.append(now)
            raise KeyboardInterrupt
        if now - self._sigint[0] > 1.0:
            self.bus.torque(False)
            print("\nsecond Ctrl-C: torque off")
            raise SystemExit(130)

    def __exit__(self, exc_type, exc, tb) -> bool:
        bus = self.bus
        handled = exc_type in (Abort, KeyboardInterrupt)
        if exc_type is None:
            self.status = "complete"
        elif handled:
            self.status = f"aborted: {exc}" if exc_type is Abort else "aborted: Ctrl-C"
            if self.torque_on:
                print(f"\n{self.status} -> returning to the start pose")
                try:
                    self.move_to(self.start, v=V_RETREAT, watchdog=False, check=False)
                except Exception as e2:  # bus trouble while returning: stop holding rather than fight it
                    print(f"return failed ({e2}): torque off")
        else:
            self.status = f"error: {exc_type.__name__}: {exc}"
            print(f"\n{self.status} -> torque off")
        try:
            bus.torque(False)
            self.torque_on = False
        finally:
            try:
                bus.write_settings({j: {k: self.original[j][k] for k in self.gains[j]} for j in JOINTS})
            except Exception as e:
                print(f"could NOT restore the previous gains ({e}); they were: "
                      + ", ".join(f"{j} " + "/".join(str(self.original[j][k]) for k in self.gains[j]) for j in JOINTS))
            bus.close()
            signal.signal(signal.SIGINT, signal.default_int_handler)
        return handled

    # -- I/O
    def read(self) -> Reading:
        for _ in range(WATCHDOG["read_failures"]):
            try:
                s = self.bus.read_state()
                break
            except IOError:
                self.read_failures += 1
        else:
            raise Abort(f"{WATCHDOG['read_failures']} failed reads in a row")
        q = self.map.to_rad(s.position) + self.offsets
        sign = np.array([self.map.sign[j] for j in JOINTS])
        return Reading(self.bus.now(), q, sign * s.velocity * RAD_PER_TICK, s.temperature, s.voltage, s.load)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        """Inside the sim joint range and the servo limits, with margins (true angles)."""
        a, b = self.map.to_rad(self.servo_limits[:, 0]), self.map.to_rad(self.servo_limits[:, 1])
        tick = TICK_MARGIN * RAD_PER_TICK
        lo = np.maximum(self.model.range[:, 0] + LIMIT_MARGIN, np.minimum(a, b) + self.offsets + tick)
        hi = np.minimum(self.model.range[:, 1] - LIMIT_MARGIN, np.maximum(a, b) + self.offsets - tick)
        return np.clip(q, lo, hi)

    def command(self, q: np.ndarray) -> np.ndarray:
        """Send goals (true angles), clamped; returns what was sent."""
        q = self.clamp(np.asarray(q, float))
        self.bus.write_goal(self.map.to_raw(q - self.offsets))
        return q

    def guard(self, goal: np.ndarray, r: Reading, max_error: float = WATCHDOG["error"]) -> None:
        self.watchdog.check(goal, r.q, r.temperature, r.voltage, r.t, max_error)

    # -- motion
    def plan_move(self, target: np.ndarray, v: float = V_MOVE, hold: float = 0.3) -> Plan:
        b = PlanBuilder(self.read().q)
        b.move(np.asarray(target, float), v=v, name="from_start")
        b.hold(hold, name="settle")
        return b.build()

    def check(self, plan: Plan) -> list[str]:
        return check_plan(plan, self.model, self.servo_limits, self.map, offsets=self.offsets)

    def follow(self, q: np.ndarray, watchdog: bool = True, on_tick=None) -> None:
        """Track goal samples at RATE; time drives the index, so a slow loop never makes the goal jump.
        `on_tick(k, goal, reading)` after each read."""
        start = self.bus.now()
        n = 0
        while True:
            k = int((self.bus.now() - start) * RATE)
            if k >= len(q):
                return
            goal = self.command(q[k])
            r = self.read()
            if watchdog:
                self.guard(goal, r)
            if on_tick is not None:
                on_tick(k, goal, r)
            n += 1
            self.bus.sleep_until(start + n / RATE)

    def move_to(self, target: np.ndarray, v: float = V_MOVE, hold: float = 0.3, watchdog: bool = True,
                check: bool = True) -> None:
        plan = self.plan_move(target, v, hold)
        if check:
            problems = self.check(plan)
            if problems:
                raise Abort("move check failed: " + "; ".join(problems))
        self.follow(plan.q, watchdog=watchdog)
