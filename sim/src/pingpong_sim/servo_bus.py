"""STS3215 servo bus for the SO-101 follower: the real Feetech bus and a MuJoCo stand-in with the same interface.

Joint angles are in the sim's convention (radians, Menagerie `robotstudio_so101` joints). The servos carry LeRobot's
calibration in EEPROM (Homing_Offset, Min/Max_Position_Limit), and LeRobot's degree mode reads
deg = (raw - mid) * 360 / 4095 with mid = (min + max) / 2. Menagerie's zero pose is that mid pose (upper arm up,
forearm and wrist level), so sim = sign * (raw - mid) * 2 pi / 4095 + offset, with sign +-1 and offset per joint
from the joint map (`record_servos.py map` checks the signs on the real arm).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
IDS = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "wrist_roll": 5}
MODEL_STS3215 = 777
MAX_RES = 4095  # LeRobot's degree conversion uses resolution - 1
RAD_PER_TICK = 2 * np.pi / MAX_RES

# STS/SMS control table: name -> (address, bytes). From LeRobot (v0.4.0, motors/feetech/tables.py).
REG = {
    "Firmware_Major_Version": (0, 1),
    "Firmware_Minor_Version": (1, 1),
    "Model_Number": (3, 2),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Torque_Limit": (16, 2),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Velocity": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Velocity": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Present_Current": (69, 2),
}
STATE_ADDR, STATE_LEN = 56, 15  # Present_Position .. Present_Current, one sync read


def _signed(value: int, sign_bit: int) -> int:
    """Feetech sign-magnitude decoding."""
    mag = value & ((1 << sign_bit) - 1)
    return -mag if value & (1 << sign_bit) else mag


@dataclass
class JointMap:
    """sim angle = sign * (raw - mid) * RAD_PER_TICK + offset."""

    mid: dict[str, float]
    sign: dict[str, int] = field(default_factory=lambda: {j: 1 for j in JOINTS})
    offset: dict[str, float] = field(default_factory=lambda: {j: 0.0 for j in JOINTS})

    def to_rad(self, raw: np.ndarray) -> np.ndarray:
        return np.array([self.sign[j] * (raw[i] - self.mid[j]) * RAD_PER_TICK + self.offset[j]
                         for i, j in enumerate(JOINTS)])

    def to_raw(self, q: np.ndarray) -> np.ndarray:
        return np.array([self.sign[j] * (q[i] - self.offset[j]) / RAD_PER_TICK + self.mid[j]
                         for i, j in enumerate(JOINTS)])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sign": self.sign, "offset": self.offset}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path, mid: dict[str, float]) -> JointMap:
        d = json.loads(path.read_text())
        return cls(mid=mid, sign={j: int(d["sign"][j]) for j in JOINTS},
                   offset={j: float(d["offset"][j]) for j in JOINTS})


@dataclass
class State:
    """Decoded servo state, raw units, one entry per joint in JOINTS order."""

    position: np.ndarray  # ticks
    velocity: np.ndarray  # ticks/s (Present_Velocity, assumed one tick per unit; the fit uses positions)
    load: np.ndarray  # 0.1 % of max torque
    voltage: np.ndarray  # V
    temperature: np.ndarray  # deg C
    current: np.ndarray  # A


class FeetechBus:
    """Real bus (feetech-servo-sdk), 1 Mbps, protocol 0."""

    def __init__(self, port: str, baudrate: int = 1_000_000, retries: int = 3):
        import scservo_sdk as scs

        self._scs = scs
        self.port = scs.PortHandler(port)
        self.packet = scs.PacketHandler(0)
        self.baudrate = baudrate
        self.retries = retries
        self._reader = None
        self._writer = None

    # -- connection
    def open(self) -> None:
        if not self.port.openPort():
            raise ConnectionError(f"can't open {self.port.getPortName()}")
        if not self.port.setBaudRate(self.baudrate):
            raise ConnectionError(f"can't set {self.baudrate} baud on {self.port.getPortName()}")
        scs = self._scs
        self._reader = scs.GroupSyncRead(self.port, self.packet, STATE_ADDR, STATE_LEN)
        for j in JOINTS:
            self._reader.addParam(IDS[j])
        self._writer = scs.GroupSyncWrite(self.port, self.packet, *REG["Goal_Position"])

    def close(self) -> None:
        self.port.closePort()

    @staticmethod
    def now() -> float:
        return time.perf_counter()

    @staticmethod
    def sleep_until(t: float) -> None:
        dt = t - time.perf_counter()
        if dt > 0:
            time.sleep(dt)

    # -- single registers
    def _flush(self) -> None:
        """Drop late replies left in the input buffer (an EEPROM write can delay a servo's answer past the SDK's
        timeout; the retry then reads that late reply and leaves its own behind for the next transaction)."""
        time.sleep(0.005)
        self.port.clearPort()

    def read(self, name: str, joint: str) -> int:
        addr, size = REG[name]
        fn = self.packet.read1ByteTxRx if size == 1 else self.packet.read2ByteTxRx
        for _ in range(self.retries):
            value, comm, err = fn(self.port, IDS[joint], addr)
            if comm == self._scs.COMM_SUCCESS:
                return int(value)
            self._flush()
        raise IOError(f"read {name} from {joint}: {self.packet.getTxRxResult(comm)}")

    def write(self, name: str, joint: str, value: int) -> None:
        addr, size = REG[name]
        fn = self.packet.write1ByteTxRx if size == 1 else self.packet.write2ByteTxRx
        for _ in range(self.retries):
            comm, err = fn(self.port, IDS[joint], addr, int(value))
            if comm == self._scs.COMM_SUCCESS:
                return
            self._flush()
        raise IOError(f"write {name}={value} to {joint}: {self.packet.getTxRxResult(comm)}")

    # -- setup
    def identify(self) -> dict[str, dict]:
        out = {}
        for j in JOINTS:
            model = self.read("Model_Number", j)
            if model != MODEL_STS3215:
                raise RuntimeError(f"{j} (id {IDS[j]}) reports model {model}, expected STS3215 ({MODEL_STS3215})")
            out[j] = {"model": model, "firmware": f"{self.read('Firmware_Major_Version', j)}."
                                                  f"{self.read('Firmware_Minor_Version', j)}"}
        return out

    def read_calibration(self) -> dict[str, dict]:
        cal = {}
        for j in JOINTS:
            lo, hi = self.read("Min_Position_Limit", j), self.read("Max_Position_Limit", j)
            cal[j] = {"min": lo, "max": hi, "homing_offset": _signed(self.read("Homing_Offset", j), 11)}
        return cal

    def read_settings(self) -> dict[str, dict]:
        names = ("P_Coefficient", "I_Coefficient", "D_Coefficient", "Acceleration", "Operating_Mode",
                 "Max_Torque_Limit", "Torque_Limit", "Torque_Enable")
        return {j: {n: self.read(n, j) for n in names} for j in JOINTS}

    def write_settings(self, settings: dict[str, dict]) -> None:
        """EEPROM writes (gains): torque must be off; opens and re-closes the lock. Verified by reading back."""
        for j, regs in settings.items():
            self.write("Torque_Enable", j, 0)
            self.write("Lock", j, 0)
            for name, value in regs.items():
                self.write(name, j, value)
            self.write("Lock", j, 1)
        time.sleep(0.05)  # let the servos finish writing flash
        self._flush()
        for j, regs in settings.items():
            got = {name: self.read(name, j) for name in regs}
            if got != {name: int(v) for name, v in regs.items()}:
                raise IOError(f"{j}: wrote {regs}, read back {got}")

    def torque(self, enable: bool) -> None:
        for j in JOINTS:
            self.write("Torque_Enable", j, 1 if enable else 0)

    # -- loop I/O
    def write_goal(self, raw: np.ndarray) -> None:
        self._writer.clearParam()
        for i, j in enumerate(JOINTS):
            v = int(round(raw[i]))
            self._writer.addParam(IDS[j], [v & 0xFF, (v >> 8) & 0xFF])
        comm = self._writer.txPacket()
        if comm != self._scs.COMM_SUCCESS:
            raise IOError(f"sync write goal: {self.packet.getTxRxResult(comm)}")

    def read_state(self) -> State:
        for _ in range(self.retries):
            comm = self._reader.txRxPacket()
            # A stale reply with the right ID but no data can pass as success: check every servo's payload length.
            if comm == self._scs.COMM_SUCCESS and all(len(self._reader.data_dict[IDS[j]]) == STATE_LEN for j in JOINTS):
                break
            self._flush()
        else:
            raise IOError(f"sync read state: {self.packet.getTxRxResult(comm)}")
        cols = {k: [] for k in ("p", "v", "l", "u", "t", "c")}
        for j in JOINTS:
            g = lambda name: self._reader.getData(IDS[j], *REG[name])  # noqa: E731
            cols["p"].append(g("Present_Position"))
            cols["v"].append(_signed(g("Present_Velocity"), 15))
            cols["l"].append(_signed(g("Present_Load"), 10))
            cols["u"].append(g("Present_Voltage") * 0.1)
            cols["t"].append(g("Present_Temperature"))
            cols["c"].append(_signed(g("Present_Current"), 15) * 0.0065)
        a = {k: np.array(v, dtype=float) for k, v in cols.items()}
        return State(a["p"], a["v"], a["l"], a["u"], a["t"], a["c"])


class SimBus:
    """MuJoCo Warp stand-in: the pingpong_sim robot with its fitted servo model, behind the FeetechBus interface.

    Calibration is synthetic: mid = 2047.5 for every joint and the limits are the sim joint ranges, so ticks map to
    sim radians with sign +1 and offset 0. Time is virtual: sleep_until() advances the simulation.
    """

    def __init__(self, start_q: np.ndarray, settle_s: float = 0.5):
        import torch

        from pingpong_sim.harness import make_harness, step
        from pingpong_sim.robot import FIRMWARE_P
        from pingpong_sim.servo_actuator import encoder_pos, encoder_vel

        self._torch, self._step = torch, step
        self._enc, self._enc_vel = encoder_pos, encoder_vel
        self.h = make_harness(num_envs=1)
        self.firmware_p = dict(FIRMWARE_P)
        dev = self.h.sim.device
        self._ids, names = self.h.robot.find_joints(JOINTS, preserve_order=True)
        assert tuple(names) == JOINTS
        lo, hi = self.h.mj_model.jnt_range[[self.h.mj_model.joint(f"robot/{j}").id for j in JOINTS]].T
        self.map = JointMap(mid={j: MAX_RES / 2 for j in JOINTS})
        lo_raw, hi_raw = self.map.to_raw(lo), self.map.to_raw(hi)
        self._limits = {j: (int(np.ceil(lo_raw[i])), int(np.floor(hi_raw[i]))) for i, j in enumerate(JOINTS)}
        # Park the ball away from the arm.
        root = torch.zeros(1, 13, device=dev)
        root[0, :3] = torch.tensor([1.0, 1.0, 0.02])
        root[0, 3] = 1.0
        self.h.ball.write_root_state_to_sim(root)
        q = torch.tensor(start_q, device=dev, dtype=torch.float32)[None]
        self.h.robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=self._ids)
        self.h.sim.forward()
        self._torque = False
        self._goal = q.clone()
        self.t = 0.0
        self._advance(settle_s)  # limp arm settles onto its rest pose
        self.t = 0.0

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def now(self) -> float:
        return self.t

    def sleep_until(self, t: float) -> None:
        if t > self.t:
            self._advance(t - self.t)

    def _advance(self, seconds: float) -> None:
        torch = self._torch
        n = max(1, int(round(seconds / self.h.dt)))
        for _ in range(n):
            if self._torque:
                self.h.robot.set_joint_position_target(self._goal, joint_ids=self._ids)
                self._step(self.h)
            else:
                self.h.robot.set_joint_position_target(self._enc(self.h.robot), joint_ids=self._ids)
                self.h.scene.write_data_to_sim()
                self.h.robot.write_ctrl_to_sim(torch.zeros(1, len(JOINTS), device=self.h.sim.device))
                self.h.sim.step()
                self.h.scene.update(dt=self.h.dt)
        self.t += n * self.h.dt

    def identify(self) -> dict[str, dict]:
        return {j: {"model": MODEL_STS3215, "firmware": "sim"} for j in JOINTS}

    def read_calibration(self) -> dict[str, dict]:
        return {j: {"min": lo, "max": hi, "homing_offset": 0} for j, (lo, hi) in self._limits.items()}

    def read_settings(self) -> dict[str, dict]:
        return {j: {"P_Coefficient": self.firmware_p[j], "I_Coefficient": 0, "D_Coefficient": 0, "Acceleration": 0,
                    "Operating_Mode": 0, "Max_Torque_Limit": 1000, "Torque_Limit": 1000,
                    "Torque_Enable": int(self._torque)} for j in JOINTS}

    def write_settings(self, settings: dict[str, dict]) -> None:
        for j, regs in settings.items():
            p = regs.get("P_Coefficient", self.firmware_p[j])
            if p != self.firmware_p[j] or regs.get("I_Coefficient", 0) or regs.get("D_Coefficient", 0):
                raise ValueError(f"SimBus models P={self.firmware_p[j]}, I=0, D=0 for {j} (robot.FIRMWARE_P)")
        self._torque = False

    def torque(self, enable: bool) -> None:
        if enable:
            self._goal = self._enc(self.h.robot).clone()
        self._torque = enable

    def write_goal(self, raw: np.ndarray) -> None:
        q = self.map.to_rad(np.asarray(raw, float).round())  # the real bus sends whole ticks
        self._goal = self._torch.tensor(q, device=self.h.sim.device, dtype=self._torch.float32)[None]

    def read_state(self) -> State:
        d = self.h.robot.data
        q = self._enc(self.h.robot)[0].cpu().numpy()
        qd = self._enc_vel(self.h.robot)[0].cpu().numpy()
        tau = d.qfrc_actuator[0, self._ids].cpu().numpy()
        raw = np.round(self.map.to_raw(q))
        return State(position=raw, velocity=np.round(qd / RAD_PER_TICK), load=np.round(tau / 2.94 * 1000),
                     voltage=np.full(len(JOINTS), 12.0), temperature=np.full(len(JOINTS), 30.0),
                     current=np.zeros(len(JOINTS)))
