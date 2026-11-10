"""The robot's surroundings on CPU MuJoCo: arm with the fitted servos, ball, simulated cameras. One world.

`CpuWorld`: the plant, built like the training env (same compiled scene and solver options, `ServoDrive` with
per-episode servo draws, ball launch, restitution draw, pushes, `CameraRig`). Used by scripts/sim2sim.py (with
the controller and ONNX policy in the loop) and, through two adapters, by the robot scripts' `--sim` mode:

- `SimBusAdapter`: the FeetechBus interface (ticks, joint map, gains, torque); virtual time, `sleep_until`
  advances the physics in control ticks. When a ball has been on the table or away for a second, it drops a new
  one above the paddle (a person dropping the next ball).
- `SimCameras`: the BallCameras interface: pixel detections (centre, diameter, buffer stamp) of a virtual lens
  matching `calibs()` (equidistant fisheye at the believed focal length, believed poses, exact timestamps), so the
  robot code runs its real undistortion and measurement path.
"""

from __future__ import annotations

import math
from dataclasses import replace

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

import pingpong_sim.task  # noqa: F401  (registers the tasks)
from pingpong_sim.ball import BALL_RADIUS, SIM_DT
from pingpong_sim.camera_sim import CameraRig
from pingpong_sim.controller import ArmHistory
from pingpong_sim.kinematics import ArmKinematics
from pingpong_sim.real.camera_model import CameraCalib, PixelDetection, _noise
from pingpong_sim.robot import ARM_JOINTS, FIRMWARE_P, READY_JOINT_POS
from pingpong_sim.servo_actuator import ServoDrive
from pingpong_sim.servo_bus import JOINTS, MAX_RES, MODEL_STS3215, RAD_PER_TICK, JointMap, State
from pingpong_sim.servo_model import ServoModel
from pingpong_sim.task.state import TaskCfg, launch_ball

DATA_FIELDS = ("qpos", "qvel", "qfrc_actuator", "qfrc_bias")


class Mirror:
    """float32 torch copies (1, ...) of MuJoCo model/data arrays, the layout the torch servo model indexes."""

    def __init__(self, src, names):
        self.src, self.names = src, names
        for k in names:
            setattr(self, k, torch.tensor(np.asarray(getattr(src, k)), dtype=torch.float32)[None].clone())

    def pull(self, names=None) -> None:
        for k in names or self.names:
            getattr(self, k)[0] = torch.from_numpy(np.asarray(getattr(self.src, k)))

    def push(self, names=None) -> None:
        for k in names or self.names:
            getattr(self.src, k)[...] = getattr(self, k)[0].numpy()


class CpuWorld:
    def __init__(self, task_id: str, cameras: list[str] | None = None):
        """`cameras`: names to simulate, always present (default: the task's, with its per-episode presence)."""
        env_cfg = load_env_cfg(task_id)
        self.task: TaskCfg = env_cfg.events["reset_episode"].params["task"]
        self.damp = env_cfg.events["ball_contact"].params["dampratio"]
        self.push_dv = env_cfg.events["push_ball"].params["dv"]
        self.push_every = env_cfg.events["push_ball"].interval_range_s
        self.term = {k: env_cfg.terminations[k].params for k in ("ball_dropped", "ball_away", "paddle_tilted")}
        self.episode_s = env_cfg.episode_length_s
        self.decimation = env_cfg.decimation
        self.dt = SIM_DT * self.decimation
        scene = Scene(replace(env_cfg.scene, num_envs=1), "cpu")
        m = self.m = scene.compile()
        env_cfg.sim.mujoco.apply(m)
        self.d = mujoco.MjData(m)
        names = [f"robot/{j}" for j in ARM_JOINTS]
        jids = [m.joint(n).id for n in names]
        self.act = [next(i for i in range(m.nu) if m.actuator_trnid[i, 0] == j) for j in jids]
        self.jrange = m.jnt_range[jids].copy()
        jb = m.joint("ball/ball_joint").id
        self.qa, self.va = int(m.jnt_qposadr[jb]), int(m.jnt_dofadr[jb])
        self.ball_body = int(m.jnt_bodyid[jb])
        paddle = m.body("robot/paddle").id
        self.robot_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith("robot/") and b != paddle}
        self.site = m.site("robot/blade_face_cap").id
        self.wcam = m.camera("robot/wrist_cam").id
        self.pair = m.pair("ball_blade").id
        self.flip = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        self.mm, self.dm = Mirror(m, ServoModel.EXPANDED_FIELDS), Mirror(self.d, DATA_FIELDS)
        servo = ServoModel(self.mm, self.dm, m, names, [FIRMWARE_P[j] for j in ARM_JOINTS], 1, "cpu")
        act_cfg = env_cfg.scene.entities["robot"].articulation.actuators[0]
        self.drive = ServoDrive(act_cfg, servo, list(ARM_JOINTS), SIM_DT, "cpu")
        self.kin = ArmKinematics(m, device="cpu")
        self.q_ready = torch.tensor([READY_JOINT_POS[j] for j in ARM_JOINTS])
        self.target = torch.tensor([*self.task.target_xy, self.task.hit_z + BALL_RADIUS])
        # The rig needs the encoder arm history for the wrist camera's believed pose (as the robot computes it).
        self.arm = ArmHistory(self.kin, 1, self.dt, "cpu")
        cams = list(self.task.cameras)
        if cameras is not None:
            cams = [replace(c, enable_prob=1.0) for c in cams if c.name in cameras]
        self.rig = CameraRig(cams, 1, self.dt, self.arm.hist, "cpu")
        self.ids = torch.zeros(1, dtype=torch.long)
        self.now = 0.0
        self.torque = True
        self.goal = self.q_ready[None].clone()
        self.hold_p = self.target[None].clone()
        self.release = math.inf
        self.next_push = math.inf

    # -- state
    def encoders(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.dm.pull()
        s = self.drive.servo
        return s.encoder().clone(), s.encoder_vel().clone()

    def truth(self):
        d = self.d
        p = torch.tensor(d.qpos[self.qa:self.qa + 3], dtype=torch.float32)[None]
        v = torch.tensor(d.qvel[self.va:self.va + 3], dtype=torch.float32)[None]
        face = torch.tensor(d.site_xpos[self.site], dtype=torch.float32)[None]
        normal = torch.tensor(d.site_xmat[self.site].reshape(3, 3)[:, 2], dtype=torch.float32)[None]
        return p, v, face, normal

    def released(self) -> bool:
        return self.now >= self.release

    def ball_touches_arm(self) -> bool:
        m, d = self.m, self.d
        for c in d.contact[:d.ncon]:
            b = {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
            if self.ball_body in b and (b - {self.ball_body}) <= self.robot_bodies and len(b) == 2:
                return True
        return False

    # -- episode
    def set_ball(self, p: torch.Tensor, v: torch.Tensor) -> None:
        self.d.qpos[self.qa:self.qa + 3] = p[0].numpy()
        self.d.qpos[self.qa + 3:self.qa + 7] = (1.0, 0.0, 0.0, 0.0)
        self.d.qvel[self.va:self.va + 6] = 0.0
        self.d.qvel[self.va:self.va + 3] = v[0].numpy()

    def launch(self) -> None:
        """A new ball: held above the target, or already flying (TaskCfg launch phase)."""
        self.hold_p, v0, release = launch_ball(self.task, self.target, 1, self.now, "cpu")
        self.release = float(release[0])
        self.set_ball(self.hold_p, v0)
        self.next_push = self.now + np.random.uniform(*self.push_every)
        mujoco.mj_forward(self.m, self.d)

    def reset(self, q_start: torch.Tensor | None = None, reset_cameras: bool = True) -> None:
        """New episode: servo draw, restitution draw, arm at rest at `q_start` (default ready + noise), ball launch,
        camera draw; records the first tick."""
        m, d = self.m, self.d
        self.drive.request_resample(self.ids)
        self.drive.mark_pending(self.ids)
        m.pair_solref[self.pair, 1] = self.damp[0] + (self.damp[1] - self.damp[0]) * np.random.rand()
        mujoco.mj_resetData(m, d)
        self.dm.pull()
        if q_start is None:
            q_start = self.q_ready[None] + (2 * torch.rand(1, 5) - 1) * self.task.joint_noise
        self.drive.servo.teleport(q_start, torch.zeros(1, 5))
        self.dm.push(("qpos", "qvel"))
        self.goal = q_start.clone()
        self.launch()
        if reset_cameras:
            self.rig.reset(self.ids, self.now)
        self.record(restart=True)

    def physics(self, goal: torch.Tensor | None = None) -> None:
        """One control tick of physics with servo goals (1, 5) (default: the last ones)."""
        m, d = self.m, self.d
        if goal is not None:
            self.goal = goal
        for _ in range(self.decimation):
            self.dm.pull()
            tau = self.drive.step(self.goal)
            self.mm.push()
            d.ctrl[self.act] = tau[0].numpy() if self.torque else 0.0
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        self.now += self.dt

    def events(self) -> None:
        """Hold the ball until its release; random pushes once released."""
        if not self.released():
            self.set_ball(self.hold_p, torch.zeros(1, 3))
            mujoco.mj_forward(self.m, self.d)
        elif self.now >= self.next_push:
            self.d.qvel[self.va:self.va + 3] += np.random.uniform(-self.push_dv, self.push_dv, 3)
            self.next_push = self.now + np.random.uniform(*self.push_every)

    def record(self, restart: bool = False) -> None:
        """Encoder arm history and camera truth for this tick (after physics and events)."""
        q, _ = self.encoders()
        self.arm.push(self.now, q)
        if restart:
            self.arm.restart(self.ids)
        p, v, face, normal = self.truth()
        d = self.d
        wR = torch.tensor(d.cam_xmat[self.wcam].reshape(3, 3), dtype=torch.float32)[None] @ self.flip
        self.rig.record(self.now, p, v, face, normal, wR, torch.tensor(d.cam_xpos[self.wcam], dtype=torch.float32)[None])
        if restart:
            self.rig.restart(self.ids)


# ---------------------------------------------------------------------------------------------------- adapters

class SimBusAdapter:
    """FeetechBus interface on a CpuWorld. Calibration is synthetic: mid 2047.5 ticks, limits = the sim joint
    ranges, sign +1, offset 0 (`map`)."""

    def __init__(self, world: CpuWorld, relaunch_after: float = 1.0):
        self.w = world
        self.map = JointMap(mid={j: MAX_RES / 2 for j in JOINTS})
        lo, hi = self.map.to_raw(world.jrange[:, 0]), self.map.to_raw(world.jrange[:, 1])
        self._limits = {j: (int(np.ceil(lo[i])), int(np.floor(hi[i]))) for i, j in enumerate(JOINTS)}
        self.relaunch_after = relaunch_after
        self._gone_since: float | None = None
        self.detections: list[dict] = []  # rig.detect output per tick, for SimCameras
        world.torque = False

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def now(self) -> float:
        return self.w.now

    def sleep_until(self, t: float) -> None:
        w = self.w
        while w.now + w.dt / 2 <= t:
            w.physics()
            w.events()
            w.record()
            self.detections.append(w.rig.detect(w.now))
            self._maybe_relaunch()

    def _maybe_relaunch(self) -> None:
        w = self.w
        p = w.truth()[0][0]
        gone = w.released() and (p[2] < 0.05 or (p[:2] - w.target[:2]).norm() > 0.5)
        if not gone:
            self._gone_since = None
        elif self._gone_since is None:
            self._gone_since = w.now
        elif w.now - self._gone_since > self.relaunch_after:
            w.launch()
            self._gone_since = None

    def identify(self) -> dict[str, dict]:
        return {j: {"model": MODEL_STS3215, "firmware": "sim"} for j in JOINTS}

    def read_calibration(self) -> dict[str, dict]:
        return {j: {"min": lo, "max": hi, "homing_offset": 0} for j, (lo, hi) in self._limits.items()}

    def read_settings(self) -> dict[str, dict]:
        return {j: {"P_Coefficient": FIRMWARE_P[j], "I_Coefficient": 0, "D_Coefficient": 0, "Acceleration": 0,
                    "Operating_Mode": 0, "Max_Torque_Limit": 1000, "Torque_Limit": 1000,
                    "Torque_Enable": int(self.w.torque)} for j in JOINTS}

    def write_settings(self, settings: dict[str, dict]) -> None:
        for j, regs in settings.items():
            p = regs.get("P_Coefficient", FIRMWARE_P[j])
            if p != FIRMWARE_P[j] or regs.get("I_Coefficient", 0) or regs.get("D_Coefficient", 0):
                raise ValueError(f"the sim models P={FIRMWARE_P[j]}, I=0, D=0 for {j} (robot.FIRMWARE_P)")
        self.w.torque = False

    def torque(self, enable: bool) -> None:
        if enable and not self.w.torque:
            self.w.drive.mark_pending(self.w.ids)  # firmware target syncs to the present position
        self.w.torque = enable

    def write_goal(self, raw: np.ndarray) -> None:
        q = self.map.to_rad(np.asarray(raw, float).round())  # the real bus sends whole ticks
        self.w.goal = torch.tensor(q, dtype=torch.float32)[None]

    def read_state(self) -> State:
        q, dq = self.w.encoders()
        n = len(JOINTS)
        return State(position=np.round(self.map.to_raw(q[0].numpy())), velocity=np.round(dq[0].numpy() / RAD_PER_TICK),
                     load=np.zeros(n), voltage=np.full(n, 12.0), temperature=np.full(n, 30.0), current=np.zeros(n))


class SimCameras:
    """BallCameras interface on the CpuWorld's camera rig (fed by SimBusAdapter.sleep_until)."""

    def __init__(self, bus: SimBusAdapter):
        self.bus, self.w = bus, bus.w
        self.cams = self.w.rig.cams
        self.size = {}
        for c in self.cams:  # image axes = the rig's camera axes; long side along `long_axis`
            self.size[c.name] = (c.width_px, c.height_px) if c.long_axis == "x" else (c.height_px, c.width_px)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def calibs(self) -> dict[str, CameraCalib]:
        rig, w = self.w.rig, self.w
        out = {}
        for i, c in enumerate(self.cams):
            P = rig.params[i]
            fb = float(P["focal_believed"][0])
            W, H = self.size[c.name]
            cal = CameraCalib(name=c.name, mount=c.mount, width=W, height=H,
                              K=[[fb, 0, W / 2], [0, fb, H / 2], [0, 0, 1]], D=[0.0, 0.0, 0.0, 0.0], noise=_noise(c))
            if c.mount == "world":
                cal.R, cal.p = P["R_believed"][0].tolist(), P["p_believed"][0].tolist()
            else:
                cal.R, cal.p = w.kin.cam_R.tolist(), w.kin.cam_p.tolist()
            out[c.name] = cal
        return out

    def drain(self) -> list[PixelDetection]:
        rig = self.w.rig
        out = []
        for det in self.bus.detections:
            for k in range(det["t"].shape[0]):
                if not bool(det["valid"][k, 0]):
                    continue
                c = int(det["cam"][k, 0])
                cfg, f = self.cams[c], float(rig.params[c]["focal"][0])
                W, H = self.size[cfg.name]
                uv = det["uv_px"][k, 0]
                # The rig's diameter is f x angular size; a real (equidistant) lens images the ball's area with the
                # radial x tangential scale f^2 theta / sin(theta).
                th = float(uv.norm()) / f
                d = float(det["diam_px"][k, 0]) * math.sqrt(th / math.sin(th) if th > 1e-6 else 1.0)
                out.append(PixelDetection(cfg.name, float(det["t"][k, 0]), W / 2 + float(uv[0]), H / 2 + float(uv[1]), d))
        self.bus.detections.clear()
        return out
