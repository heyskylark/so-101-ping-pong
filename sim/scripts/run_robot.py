"""Bounce the ball with the real arm (or its CPU MuJoCo stand-in), running an exported policy at 100 Hz.

  uv run scripts/run_robot.py --port /dev/ttyACM0 --scripted          # the scripted bouncer alone (gentle task)
  uv run scripts/run_robot.py --port /dev/ttyACM0 --policy policies/<name>
  uv run scripts/run_robot.py --sim --scripted --seconds 60            # arm, ball and cameras simulated

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
import contextlib
import datetime as dt
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import pingpong_sim.task  # noqa: F401  (registers the tasks)
from mjlab.tasks.registry import load_env_cfg
from pingpong_sim.bouncer import BouncerCfg
from pingpong_sim.controller import BounceController, OnnxPolicy, controller_cfg_from_policy
from pingpong_sim.kinematics import ArmKinematics
from pingpong_sim.ball import BALL_RADIUS
from pingpong_sim.real.arm import RATE, ArmSession, PlanBuilder, open_bus, use_rig_tracking
from pingpong_sim.real.camera_model import (ArmCalib, CameraCalib, apply_face_tilt, apply_wrist_mount, nominal,
                                            to_measurements)
from pingpong_sim.robot import ARM_JOINTS, FIRMWARE_P, READY_JOINT_POS
from pingpong_sim.servo_bus import JOINTS

REPO = Path(__file__).resolve().parents[2]
MAX_STEP = 0.15  # rad per tick
MIN_BLADE_Z = 0.12  # m above the table
MAX_TILT_DEG = 45.0
LOST_AFTER = 0.5  # s without a detection
AWAY = 0.25  # m from the target
RECOVER_S = 0.7
REST = dict(speed=0.15, gap=0.03, lateral=0.06, hold=0.5)  # pop test: ball at rest on the blade [m/s, m, m, s]
TRIM = dict(rate=2.0, dead=np.radians(0.1), max=np.radians(4.0))  # pop test: goal trim at ready [1/s, rad, rad]
# A ball held still above the paddle [s, detections, px, relative size, s].
HELD = dict(window=0.2, n=4, px=16.0, size=0.05, gap=0.3)


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
                 bouncer: dict | None = None, restitution: float | None = None,
                 goal_rate: list[float] | None = None, paddle_lag: float = 0.0, witness: tuple[str, ...] = ()):
        if goal_rate and len(goal_rate) not in (1, 5):
            raise SystemExit("--goal-rate: one value, or one per joint")
        self.max_step = np.radians(goal_rate) / RATE * np.ones(5) if goal_rate else np.full(5, MAX_STEP)  # rad per tick
        self.s, self.cams, self.calibs = session, cameras, calibs
        task = load_env_cfg(task_id).events["reset_episode"].params["task"]
        self.task = task
        self.kin = kin
        self.q_ready = torch.tensor([READY_JOINT_POS[j] for j in ARM_JOINTS])
        if policy_dir is None:
            # Wait and rest with the real face level: `kin` may carry a measured face tilt (apply_face_tilt). A
            # policy keeps the pose it was trained around (its observations are relative to it).
            self.q_ready = kin.aim(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([task.target_xy[1]]),
                                   self.q_ready[None], iters=8)[0]
        ccfg = replace(task.controller_cfg(), paddle_lag=paddle_lag)
        if policy_dir:  # the bounce height, stroke and history the policy was trained with
            ccfg = controller_cfg_from_policy(json.loads((policy_dir / "policy.json").read_text()), ccfg)
        bouncer = dict(bouncer or {})
        if restitution is not None:
            bouncer["restitution"] = restitution
            ccfg = replace(ccfg, filter=replace(ccfg.filter, restitution=restitution))
        ccfg = replace(ccfg, apex=apex if apex is not None else ccfg.apex, bouncer=replace(ccfg.bouncer, **bouncer))
        self.ctrl = BounceController(ccfg, kin, self.q_ready, 1, 1.0 / RATE, "cpu")
        self.policy = OnnxPolicy(policy_dir, self.ctrl) if policy_dir else None
        self.pop = pop_goals(kin, self.q_ready.numpy(), pop)[0] if pop else None  # pop test: never plays
        self.pops = 0
        self.ids = torch.zeros(1, dtype=torch.long)
        self.log: dict[str, list] = {k: [] for k in ("t", "q", "dq", "goal", "base", "est_p", "est_v", "tracked",
                                                     "trusted", "state", "n_det", "work_ms", "unsafe", "plan_t_hit",
                                                     "plan_speed", "plan_tilt", "trim")}
        self.truth: dict[str, list] = {k: [] for k in ("ball_p", "ball_v", "face", "normal")}
        self.dets: list[tuple] = []
        self.witness, self.witness_dets = set(witness), []  # cameras that watch without feeding the filter
        self.episodes = 0
        self.t0 = 0.0
        self.bounces = 0
        self.held_at = -np.inf  # when the ball was last seen held still

    def safe(self, goal: np.ndarray) -> bool:
        face, normal, _, _ = self.kin.fk(torch.tensor(goal, dtype=torch.float32)[None])
        return bool(face[0, 2] > MIN_BLADE_Z and normal[0, 2] > np.cos(np.radians(MAX_TILT_DEG)))

    def held(self, now: float) -> bool:
        """The ball is being held still: every camera with HELD["n"] detections in the last HELD["window"] s saw it
        move less than HELD["px"] and change size by less than HELD["size"] (one stray detection aside); it stays
        held through a gap in the detections (HELD["gap"] s) until a camera sees it move. The filter knows no held
        ball (it has it falling at ~1 m/s and the bouncer strikes at that), so play waits for the release; a ball
        dropped or tossed in is never still."""
        def still(a: np.ndarray) -> bool:
            c = np.median(a, axis=0)
            off = np.maximum(np.abs(a[:, :2] - c[:2]).max(1) / HELD["px"], np.abs(a[:, 2] - c[2]) / (HELD["size"] * c[2]))
            a = np.delete(a, int(np.argmax(off)), axis=0)
            return bool(np.ptp(a[:, :2], axis=0).max() < HELD["px"] and np.ptp(a[:, 2]) < HELD["size"] * a[:, 2].mean())

        seen: dict[str, list] = {}
        for t, cam, u, v, d in reversed(self.dets):
            if t < now - HELD["window"] - 0.1:  # stamps trail `now` by the camera latency
                break
            seen.setdefault(cam, []).append((u, v, d))
        stills = [still(np.array(a)) for a in seen.values() if len(a) >= HELD["n"]]
        if stills:
            self.held_at = now if all(stills) else -np.inf
        return now - self.held_at < HELD["gap"]

    def run(self, seconds: float) -> None:
        s, ctrl = self.s, self.ctrl
        t0 = self.t0 = s.bus.now()  # the bus clock (time.perf_counter on the rig: the cameras' stamps too)
        hist_t, hist_q = [], []
        state, state_since, last_goal, rest_since = "WAIT", 0.0, self.q_ready.numpy().copy(), None
        trim, pop_shift = np.zeros(5), np.zeros(5)  # pop test: see WAIT
        recover_from = last_goal
        # What the bouncer's feedforward adds to hold a pose against gravity (the sim arm's sag at ready).
        ff_sag = (ctrl.bouncer.goal(self.q_ready[None], torch.zeros(1, 5), torch.zeros(1, 5))[0] - self.q_ready).numpy()
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
            if self.witness:
                self.witness_dets += [(d.stamp - t0, d.camera, d.u, d.v, d.d) for d in dets if d.camera in self.witness]
                dets = [d for d in dets if d.camera not in self.witness]
            self.dets += [(d.stamp - t0, d.camera, d.u, d.v, d.d) for d in dets]
            ctrl.update(to_measurements(dets, self.calibs, ctrl.arm.hist, t0=t0))
            p, v = ctrl.est_p[0].numpy(), ctrl.est_v[0].numpy()
            tracked = bool(ctrl.tracked[0])
            above = tracked and p[2] > self.task.hit_z + 0.05
            # Episode state machine.
            if state == "WAIT":
                # The servos sag ~1 deg per joint below a goal (no integral gain): the blade waits ~2 deg tipped down
                # and 1 cm low, so the first ball leaves forward (and a resting ball rolls off). Trim the goal slowly
                # until the encoders read ready.
                err = self.q_ready.numpy() - r.q
                trim = np.clip(trim + TRIM["rate"] / RATE * np.where(np.abs(err) > TRIM["dead"], err, 0.0),
                               -TRIM["max"], TRIM["max"])
                goal = self.q_ready.numpy() + trim
                if self.pop is not None:  # pop test: launch a ball resting on the blade, never play
                    face, _, _, _ = self.kin.fk(torch.tensor(r.q, dtype=torch.float32)[None])
                    face = face[0].numpy()
                    gap = p[2] - face[2] - BALL_RADIUS
                    resting = (tracked and float(np.linalg.norm(v)) < REST["speed"] and -0.01 < gap < REST["gap"]
                               and np.hypot(*(p[:2] - face[:2])) < REST["lateral"])
                    rest_since = (rest_since if resting and rest_since is not None else now) if resting else None
                    if rest_since is not None and now - rest_since > REST["hold"]:
                        state, state_since, rest_since = "POP", now, None
                        pop_shift = goal - self.pop[0]  # the pop's goals start from the trimmed hold, not a jump
                        self.pops += 1
                        print(f"  {now:6.1f} s: pop {self.pops}")
                elif above and float(ctrl.age[0]) < 0.1 and not self.held(now):
                    state, state_since = "PLAY", now
                    self.episodes += 1
                    if self.policy:
                        self.policy.reset()
            elif state == "POP":
                k = int(round((now - state_since) * RATE))
                goal = self.pop[min(k, len(self.pop) - 1)] + pop_shift
                if k >= len(self.pop):
                    state, state_since = "WAIT", now
            elif state == "PLAY":
                # The sag the WAIT trim measured on this arm replaces the one the feedforward expects (the sim's):
                # else the blade drops back ~2 deg tip-down and ~8 mm low as soon as play starts.
                goal = (self.policy.act() if self.policy else ctrl.goal_base())[0].numpy() + trim - ff_sag
                lost = not tracked or float(ctrl.age[0]) > LOST_AFTER
                low = tracked and p[2] < self.task.hit_z - 0.10
                away = tracked and np.hypot(*(p[:2] - np.array(self.task.target_xy))) > AWAY
                if prev_vz < -0.5 and v[2] > 0.5 and p[2] < self.task.hit_z + 0.1:
                    self.bounces += 1
                held = self.held(now)  # e.g. play began while the hand was still bringing the ball in
                if lost or low or away or held:
                    why = "lost" if lost else "low" if low else "away" if away else "held: waiting for the release"
                    print(f"  {now:6.1f} s: episode {self.episodes} ends ({why})")
                    state, state_since, recover_from = "RECOVER", now, last_goal.copy()
                    if lost or low or away:  # a held ball keeps its track: it is about to be dropped
                        ctrl.reset(self.ids, now, keep_trim=True)  # same arm: its learned tilt trim stays
            else:  # RECOVER
                a = min(1.0, (now - state_since) / RECOVER_S)
                a = 10 * a**3 - 15 * a**4 + 6 * a**5
                goal = recover_from + a * (self.q_ready.numpy() - recover_from)
                # The ball back above the paddle and falling (e.g. it left the overhead view and came down on the
                # blade): play again once the arm is most of the way back to ready.
                again = (a > 0.5 and tracked and float(ctrl.age[0]) < 0.1 and v[2] < 0 and
                         p[2] > self.task.hit_z + 0.03 and np.hypot(*(p[:2] - np.array(self.task.target_xy))) < 0.12)
                if again and not self.held(now):
                    state, state_since = "PLAY", now
                    self.episodes += 1
                    if self.policy:
                        self.policy.reset()
                elif a >= 1.0:
                    state, state_since = "WAIT", now
            prev_vz = float(v[2]) if tracked else 0.0
            # Safety.
            goal = last_goal + np.clip(goal - last_goal, -self.max_step, self.max_step)
            unsafe = not self.safe(goal)
            if unsafe:
                goal = last_goal
            sent = s.command(goal)
            last_goal = sent
            s.guard(sent, r, max_error=0.5)
            work = 1e3 * (time.perf_counter() - w0)
            bn = ctrl.bouncer  # the plan in force: predicted impact time, paddle speed along the normal, face tilt
            for k, val in zip(self.log, (now, r.q, dq, sent, ctrl.goal_base()[0].numpy(), p, v, tracked,
                                          bool(ctrl.trusted[0]), state, len(dets), work, unsafe, float(bn.t_hit[0]),
                                          float(bn.plan_speed[0]), float(bn.plan_tilt[0]), bn.trim[0].numpy().copy())):
                self.log[k].append(val)
            world = getattr(s.bus, "w", None)
            if world is not None:  # the simulated arm: what the encoders can't see
                for k, val in zip(self.truth, world.truth()):
                    self.truth[k].append(val[0].numpy().copy())
            n += 1
            nxt = start + n / RATE
            if s.bus.now() > nxt:  # overran: skip the missed ticks rather than run late ones back to back
                n = int((s.bus.now() - start) * RATE) + 1
                nxt = start + n / RATE
            s.bus.sleep_until(nxt)

    def save(self, out: Path, meta: dict) -> None:
        out.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.array(v) for k, v in self.log.items()}
        truth = {f"true_{k}": np.array(v) for k, v in self.truth.items() if v}  # sim runs only
        np.savez_compressed(out / "log.npz", **arrays, **truth, t0=self.t0,
                            detections=np.array([(t, u, v, d) for t, _, u, v, d in self.dets]).reshape(-1, 4),
                            detection_camera=np.array([c for _, c, _, _, _ in self.dets]),
                            witness_detections=np.array([(t, u, v, d) for t, _, u, v, d in self.witness_dets]).reshape(-1, 4),
                            witness_camera=np.array([c for _, c, _, _, _ in self.witness_dets]))
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
    pol.add_argument("--policy", help="exported policy directory (trained at the current ready pose)")
    pol.add_argument("--scripted", action="store_true", help="the scripted bouncer alone")
    ap.add_argument("--task", help="scripted bouncer: the task whose bouncer settings and cameras to use (default "
                    "PingPong-Bounce-Gentle: pan and wrist only, wrist camera only; PingPong-Bounce-Residual: the "
                    "full-arm bouncer with both cameras, which wore out two servos on the rig)")
    ap.add_argument("--cameras", nargs="+", help="default: the task's (the policy's task with --policy)")
    ap.add_argument("--witness", nargs="*", default=[], metavar="CAMERA",
                    help="rig: cameras that only watch (e.g. overhead): their detections are logged "
                         "(witness_detections) and with --record-frames their frames saved, but the ball filter never "
                         "sees them. For checking afterwards where the ball really went")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", help="log directory (default measurements/runs/<time>, sim: sim/build/runs/<time>)")
    ap.add_argument("--sim-flex", nargs="*", default=[], metavar="JOINT=K",
                    help="--sim: fixed stiffness [N m/rad] of the structure after these joints' encoders instead of "
                         "the task's draw, e.g. wrist_flex=6.5 (the log then holds the true face next to the encoders')")
    ap.add_argument("--sim-play", nargs="*", default=[], metavar="JOINT=DEG",
                    help="--sim: fixed free play [deg] in these joints' gears instead of the task's draw, e.g. "
                         "shoulder_lift=1.5")
    ap.add_argument("--sim-bounce-jitter", nargs=2, type=float, metavar=("LO", "HI"),
                    help="--sim: a new ball-blade damping ratio in this range every 0.15-0.3 s, so each bounce has its "
                         "own restitution (0.10 0.25: ~0.76-0.50), as the task PingPong-Bounce-Gentle-Rough trains with")
    ap.add_argument("--sim-zero-error", nargs="*", default=[], metavar="JOINT=DEG",
                    help="--sim: a calibration error of these joints' zeros (the controller believes the true angle "
                         "plus this), e.g. wrist_roll=2: a blade that leans 2 deg sideways when it is thought level")
    ap.add_argument("--record-frames", action="store_true",
                    help="save every camera frame as JPEG in <out>/frames/<camera>/<stamp>.jpg (~90 MB a minute per "
                         "camera; the log's detection times are these stamps - t0): what the detector saw")
    ap.add_argument("--pop", type=float, metavar="SPEED",
                    help="pop test instead of play: a ball resting on the blade is launched straight up at SPEED "
                         "[m/s] (face level, brakes harder than g), again each time it settles: clean free flights")
    ap.add_argument("--apex", type=float, help="scripted bouncer: bounce height above the hitting plane [m] "
                    "(default: the task's 0.15; the overhead camera sees the ball up to ~0.2 m above the paddle)")
    ap.add_argument("--stroke-time", type=float, help="scripted bouncer: stroke time [s] (default: the task's 0.10)")
    ap.add_argument("--follow", type=float, help="scripted bouncer: follow-through time [s] (default: the task's 0.08)")
    ap.add_argument("--stroke-joint", choices=["elbow_flex", "wrist_flex"],
                    help="scripted bouncer: the joint that makes the stroke (default: the task's, the elbow)")
    ap.add_argument("--impact-bias", type=float, help="scripted bouncer: added to every predicted impact time [s] "
                    "(real runs: predictions 19-28 ms early -> ~0.025)")
    ap.add_argument("--ball-gravity", type=float,
                    help="EXPERIMENT: gravity [m/s^2] the ball filter and the bouncer's predictor assume (default "
                         "9.81). Real runs predicted impacts ~28 ms early, as if the ball fell at ~8.7 m/s^2")
    ap.add_argument("--restitution", type=float,
                    help="normal restitution of the hitting rubber, for the ball filter's bounce prediction and the "
                         "bouncer's strike (default: the black side's 0.68, measured on the rig 2026-11-12; the "
                         "red side is 0.58). Policies trained before that day assume 0.58")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="scripted bouncer: other BouncerCfg fields, e.g. reach=0.03 max_tilt_deg=8 steer_gain=0.5")
    ap.add_argument("--goal-rate", type=float, nargs="+", metavar="DEG_S",
                    help="how fast the servo goals may move: one value, or one per joint (pan lift elbow wrist_flex "
                         "wrist_roll). Default 860 on every joint. The gentle bouncer's wrist strokes need ~850 on "
                         "wrist_flex (at 200 the limit held its goals back in 20-30 %% of the ticks and the strokes "
                         "came 30-50 ms late); lift and elbow only hold, so a low limit there costs nothing")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if not (args.policy or args.scripted or args.pop):
        raise SystemExit("--policy DIR, --scripted or --pop")
    policy_dir = None if args.scripted or args.pop else Path(args.policy)
    bouncer = {k: v for k, v in (("stroke_time", args.stroke_time), ("follow", args.follow),
                                 ("impact_bias", args.impact_bias)) if v is not None}
    if args.stroke_joint:
        bouncer["stroke_joint"] = JOINTS.index(args.stroke_joint)
    for kv in args.set:
        key, val = kv.split("=")
        bouncer[key] = type(getattr(BouncerCfg(), key))(float(val))
    if policy_dir and (args.apex is not None or bouncer):
        raise SystemExit("--apex/--stroke-time/--follow/--stroke-joint/--impact-bias/--set change the scripted bouncer "
                         "the policy was trained on: use them with --scripted")
    if policy_dir:
        spec = json.loads((policy_dir / "policy.json").read_text())
        task_id = spec["task_id"]
        if args.task and args.task != task_id:
            raise SystemExit(f"--task: the policy was trained on {task_id}")
        rate = spec["action"].get("max_rate")
        if rate is not None and args.goal_rate is None:  # the policy was trained with its goals moving this fast
            args.goal_rate = np.degrees(np.broadcast_to(rate, 5)).round(1).tolist()
        if np.abs(np.array(spec["q_ready"]) - [READY_JOINT_POS[j] for j in JOINTS]).max() > 1e-3:
            raise SystemExit(f"{policy_dir} was trained at another ready pose "
                             f"({np.degrees(spec['q_ready']).round(1)} deg): retrain at the current one")
        gains = [FIRMWARE_P[j] for j in JOINTS]
        if list(spec.get("firmware_p", gains)) != gains:
            raise SystemExit(f"{policy_dir} was trained with firmware gains {spec['firmware_p']}; the arm now runs "
                             f"{gains} (robot.FIRMWARE_P)")
    else:
        task_id = args.task or "PingPong-Bounce-Gentle"
    if args.cameras is None:
        args.cameras = [c.name for c in load_env_cfg(task_id).events["reset_episode"].params["task"].cameras]

    if args.sim:
        from pingpong_sim.real.sim_backend import CpuWorld, SimBusAdapter, SimCameras

        world = CpuWorld(task_id, cameras=args.cameras)
        if args.sim_flex or args.sim_play:
            ks = {kv.split("=")[0]: (float(kv.split("=")[1]),) * 2 for kv in args.sim_flex}
            play = {kv.split("=")[0]: (float(kv.split("=")[1]),) * 2 for kv in args.sim_play}
            flex = world.flex or {}
            world.flex = {"stiffness": {**flex.get("stiffness", {}), **ks},
                          "damping_ratio": flex.get("damping_ratio", (0.05, 0.05)),
                          "play_deg": {**(flex.get("play_deg") or {}), **play}}
        if args.sim_bounce_jitter:
            world.bounce = {"dampratio": tuple(args.sim_bounce_jitter), "every": (0.15, 0.3)}
        world.reset()
        bus = SimBusAdapter(world)
        cams = SimCameras(bus)
        calibs, warnings = cams.calibs(), []
        offsets = None
        if args.sim_zero_error:  # the controller's joint zeros are off: it believes true angle + error
            offsets = np.zeros(len(JOINTS))
            for kv in args.sim_zero_error:
                offsets[JOINTS.index(kv.split("=")[0])] = np.radians(float(kv.split("=")[1]))
    else:
        from pingpong_sim.real.cameras import BallCameras, RigConfig

        calibs, warnings = load_calibs(args.cameras)
        bus = open_bus(args.port)
        if set(args.witness) & set(args.cameras):
            raise SystemExit("--witness: a camera that tracks can't also be the witness")
        cams = BallCameras(RigConfig.load(), args.cameras + args.witness)
        offsets = ArmCalib.load().joint_offsets
        tracking = use_rig_tracking()
        if tracking:
            warnings.append(f"bouncer feedforward from the rig's tracking model ({tracking})")
    if args.ball_gravity is not None:
        import pingpong_sim.ball_filter as ball_filter_mod
        import pingpong_sim.bouncer as bouncer_mod

        ball_filter_mod.G = bouncer_mod.G = args.ball_gravity
        warnings.append(f"EXPERIMENT: ball filter and bouncer assume gravity {args.ball_gravity:.2f} m/s^2")
    if args.restitution is not None:
        warnings.append(f"ball filter and bouncer assume restitution {args.restitution:.2f}")
    for w in warnings:
        print("warning:", w)

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(args.out) if args.out else (REPO / "sim" / "build" / "runs" / stamp if args.sim
                                           else REPO / "measurements" / "runs" / stamp)
    if args.record_frames:
        if args.sim:
            raise SystemExit("--record-frames: the sim cameras have no pictures")
        cams.record_to(out / "frames")
    session = ArmSession(bus, joint_offsets=offsets, confirm=not args.sim)
    run = None

    def finish() -> None:  # also when closing the session fails (bus dead): the log is what explains the run
        print(session.status)
        if run is not None:
            print(run.summary())
            run.save(out, {"status": session.status, "policy": str(policy_dir), "task": task_id,
                           "cameras": args.cameras, "backend": "sim" if args.sim else "real", "port": args.port,
                           "warnings": warnings, "calibration": {k: v.steps for k, v in calibs.items()},
                           "options": {k: v for k, v in vars(args).items() if v is not None}, "created": stamp})
            print("saved", out)

    with contextlib.ExitStack() as stack:
        stack.callback(finish)
        stack.enter_context(session)
        kin = ArmKinematics(session.model.m, prefix="", device="cpu")
        if "wrist" in calibs:
            apply_wrist_mount(kin, calibs["wrist"])
        paddle_lag = 0.0
        if not args.sim and policy_dir is None:
            # The scripted bouncer plans with the measured face tilt and paddle lag. A policy was trained on an arm
            # whose blade sags and lags unseen in the same way (the task's arm_flex): it gets the raw view.
            arm = ArmCalib.load()
            apply_face_tilt(kin, arm, torch.tensor([READY_JOINT_POS[j] for j in ARM_JOINTS]))
            paddle_lag = arm.paddle_lag
            if any(arm.face_tilt):
                print(f"hitting face tilted {np.degrees(arm.face_tilt[0]):+.1f} deg forward, "
                      f"{np.degrees(arm.face_tilt[1]):+.1f} deg toward +y of the model (arm.json): planned for")
            if arm.paddle_lag:
                print(f"paddle and wrist camera taken {1e3 * arm.paddle_lag:.0f} ms behind the encoders (arm.json)")
        run = Run(session, cams, calibs, task_id, policy_dir, kin, apex=args.apex, pop=args.pop, bouncer=bouncer,
                  restitution=args.restitution, goal_rate=args.goal_rate, paddle_lag=paddle_lag,
                  witness=tuple(args.witness))
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
    return 0 if session.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
