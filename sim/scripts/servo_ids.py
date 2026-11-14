"""Replacement servos: which IDs answer on the bus, give a servo its joint's ID, calibrate its joint. The servos
never drive; `calibrate` has you move the arm by hand.

  uv run scripts/servo_ids.py scan --port /dev/ttyACM0
  uv run scripts/servo_ids.py set  --port /dev/ttyACM0 --from 6 --to 2
  uv run scripts/servo_ids.py calibrate --port /dev/ttyACM0 --joints shoulder_lift elbow_flex

IDs on this arm: 1 shoulder_pan, 2 shoulder_lift, 3 elbow_flex, 4 wrist_flex, 5 wrist_roll. `scan` pings every ID
at 1 Mbps. An ID that answers with a garbled packet is usually two servos set to the same ID talking at once: unplug
one of them (the servos pass the bus through, so unplugging one cable drops everything after it), `set` the other,
plug back in. `set` refuses if the old ID does not answer cleanly or the new one is taken; it switches the servo's
torque off and writes its EEPROM.

`calibrate` is LeRobot's follower calibration (`lerobot-calibrate`, SOFollower.calibrate) for the named joints
only: LeRobot's own command wants all six stock motors on the bus, and this arm has no gripper servo. Torque goes
off on every joint; you hold the arm in the middle pose (the homing offset is written so that pose reads 2047),
then move the named joints stop to stop (their position limits are written). The other joints keep their
calibration. `--lerobot-file` also updates LeRobot's calibration file for these joints; without it LeRobot would
write its old values back into the servos the next time it connects to this arm. Then `record_servos.py map`.

  uv run scripts/servo_ids.py limits --port /dev/ttyACM0 [--fix]

`limits` compares each joint servo's torque limit and overload settings with an arm joint's. A servo that was a
gripper (LeRobot's setup) keeps half the torque and an overload fallback of 25 %: on the shoulder or elbow it holds
the arm at first, then sags and hunts. `--fix` writes the arm-joint values (torque off, read back).

`scan` also prints each servo's voltage window: a 12 V STS3215 (C018 / C047) reports up to 14.0 V, a 7.4 V one
(C001 / C044 / C046, the SO-101 leader's) a lower limit, and must not stay on the 12 V bus. The gear ratio (1:345
on the follower; the leader's elbow and pan are 1:191, its wrist and gripper 1:147) is only on the sticker.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import scservo_sdk as scs

from pingpong_sim.servo_bus import IDS, MAX_RES, REG

ADDR_ID, ADDR_VMAX, ADDR_VMIN = 5, 14, 15
HALF_TURN = MAX_RES // 2  # 2047, as LeRobot
NAMES = {v: k for k, v in IDS.items()}


def encode_sign(value: int, sign_bit: int) -> int:
    """Feetech sign-magnitude encoding (the Homing_Offset register: sign in bit 11)."""
    if abs(value) >= 1 << sign_bit:
        raise ValueError(f"{value} does not fit {sign_bit} bits")
    return abs(value) | (1 << sign_bit if value < 0 else 0)


def decode_sign(value: int, sign_bit: int) -> int:
    mag = value & ((1 << sign_bit) - 1)
    return -mag if value & (1 << sign_bit) else mag


class Bus:
    def __init__(self, port: str):
        self.port, self.pk = scs.PortHandler(port), scs.PacketHandler(0)
        if not (self.port.openPort() and self.port.setBaudRate(1_000_000)):
            raise SystemExit(f"can't open {port} at 1 Mbps")

    def ping(self, i: int) -> str:
        """'ok', 'silent' or 'garbled'."""
        comm = None
        for _ in range(3):
            self.port.clearPort()
            try:
                _, comm, _ = self.pk.ping(self.port, i)
            except IndexError:  # the SDK reads the model number after the ping and took a stale packet for it
                comm = None
            if comm == scs.COMM_SUCCESS:
                return "ok"
            if comm != scs.COMM_RX_TIMEOUT:
                time.sleep(0.005)
        return "silent" if comm == scs.COMM_RX_TIMEOUT else "garbled"

    def read(self, i: int, addr: int, size: int = 1) -> int:
        fn = self.pk.read1ByteTxRx if size == 1 else self.pk.read2ByteTxRx
        for _ in range(5):
            self.port.clearPort()
            try:
                v, comm, _ = fn(self.port, i, addr)
            except IndexError:  # a stale packet (right ID, no data) passed as this read's answer
                comm = None
            if comm == scs.COMM_SUCCESS:
                return int(v)
            time.sleep(0.01)
        raise IOError(f"id {i}: no clean reply reading address {addr}")

    def write(self, i: int, addr: int, value: int, size: int = 1) -> None:
        """Write and read back. A servo answers an EEPROM write late (past the SDK's timeout when the value
        changes); that late reply, left in the buffer, would be taken for the next transaction's answer and shift
        every reply after it by one. So: wait out the write, drop what arrived, and trust only the read-back."""
        fn = self.pk.write1ByteTxRx if size == 1 else self.pk.write2ByteTxRx
        for _ in range(3):
            self.port.clearPort()
            fn(self.port, i, addr, int(value))
            time.sleep(0.03 if addr < 40 else 0.005)  # EEPROM below address 40
            if self.read(i, addr, size) == int(value):
                return
        raise IOError(f"id {i}: address {addr} does not read back {value}")

    def get(self, i: int, name: str) -> int:
        return self.read(i, *REG[name])

    def put(self, i: int, name: str, value: int) -> None:
        self.write(i, REG[name][0], value, REG[name][1])


def describe(bus: Bus, i: int) -> str:
    g = lambda name: bus.get(i, name)  # noqa: E731
    return (f"id {i} ({NAMES.get(i, 'not an arm joint')}): model {g('Model_Number')}, firmware "
            f"{g('Firmware_Major_Version')}.{g('Firmware_Minor_Version')}, voltage window "
            f"{bus.read(i, ADDR_VMIN) / 10:.1f}-{bus.read(i, ADDR_VMAX) / 10:.1f} V, supply now "
            f"{g('Present_Voltage') / 10:.1f} V, position {g('Present_Position')}, limits {g('Min_Position_Limit')}-"
            f"{g('Max_Position_Limit')}, homing offset {decode_sign(g('Homing_Offset'), 11)}, torque "
            f"{'on' if g('Torque_Enable') else 'off'}")


def cmd_scan(bus: Bus, args) -> int:
    state = {i: bus.ping(i) for i in range(0, 254)}
    for i, s in state.items():
        if s == "ok":
            print(describe(bus, i))
        elif s == "garbled":
            print(f"id {i}: garbled reply (two servos on this ID?)")
    missing = [f"{i} ({NAMES[i]})" for i in sorted(NAMES) if state[i] != "ok"]
    print("all five arm joints answer" if not missing else "not answering cleanly: " + ", ".join(missing))
    return 0 if not missing else 1


def cmd_set(bus: Bus, args) -> int:
    old, new = args.old, args.new
    if bus.ping(old) != "ok":
        raise SystemExit(f"id {old} does not answer cleanly ({bus.ping(old)}): exactly one servo with it must be connected")
    if bus.ping(new) != "silent":
        raise SystemExit(f"id {new} is already on the bus")
    print("before:", describe(bus, old))
    bus.put(old, "Torque_Enable", 0)
    bus.put(old, "Lock", 0)  # EEPROM writable
    bus.port.clearPort()  # the ID itself: the servo answers under the new one from here on, so no read-back
    bus.pk.write1ByteTxRx(bus.port, old, ADDR_ID, new)
    time.sleep(0.05)
    bus.put(new, "Lock", 1)
    if bus.ping(new) != "ok" or bus.ping(old) != "silent":
        raise SystemExit(f"after the write: id {new} {bus.ping(new)}, id {old} {bus.ping(old)}")
    print("after: ", describe(bus, new))
    return 0


def cmd_calibrate(bus: Bus, args) -> int:
    ids = [IDS[j] for j in args.joints]
    for i in IDS.values():
        if bus.ping(i) != "ok":
            raise SystemExit(f"id {i} ({NAMES[i]}) does not answer cleanly: `scan` first")
    for i in IDS.values():  # limp arm, so it can be posed
        bus.put(i, "Torque_Enable", 0)
    print(f"Torque is off on every joint. Calibrating: {', '.join(args.joints)}.")
    input("1. Hold the arm in the middle pose: upper arm straight up, forearm level and pointing forward (these "
          "joints near the middle of their travel). Press Enter while you hold it there ")
    for i in ids:
        bus.put(i, "Lock", 0)
        bus.put(i, "Operating_Mode", 0)  # position mode
        bus.put(i, "Homing_Offset", 0)
        bus.put(i, "Min_Position_Limit", 0)
        bus.put(i, "Max_Position_Limit", MAX_RES)
    homing = {}
    for i in ids:  # Present_Position = actual - Homing_Offset: make this pose read half a turn
        homing[i] = bus.get(i, "Present_Position") - HALF_TURN
        bus.put(i, "Homing_Offset", encode_sign(homing[i], 11))
    time.sleep(0.05)
    for i in ids:
        now = bus.get(i, "Present_Position")
        if abs(now - HALF_TURN) > 8:
            raise SystemExit(f"{NAMES[i]}: reads {now} after the homing write, expected {HALF_TURN}; nothing more written")
    print(f"   the middle pose now reads {HALF_TURN} on " + " and ".join(args.joints) + ".")
    print("2. Move each of these joints slowly through its whole travel, stop to stop (the other joints don't "
          "matter). Press Enter when done.")
    done = threading.Event()
    threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
    lo, hi = dict.fromkeys(ids, MAX_RES), dict.fromkeys(ids, 0)
    while not done.is_set():
        for i in ids:
            p = bus.get(i, "Present_Position")
            lo[i], hi[i] = min(lo[i], p), max(hi[i], p)
        print("   " + "   ".join(f"{NAMES[i]} {lo[i]:4d} .. {hi[i]:4d}" for i in ids), end="\r", flush=True)
        time.sleep(0.02)
    print()
    for i in ids:
        span = (hi[i] - lo[i]) * 360 / (MAX_RES + 1)
        if span < 120:
            raise SystemExit(f"{NAMES[i]} only moved {span:.0f} deg: limits not written (the homing offsets are). Run again.")
    for i in ids:
        bus.put(i, "Min_Position_Limit", lo[i])
        bus.put(i, "Max_Position_Limit", hi[i])
        bus.put(i, "Lock", 1)
        print("  ", describe(bus, i))
    if args.lerobot_file:
        path = Path(args.lerobot_file).expanduser()
        cal = json.loads(path.read_text())
        backup = path.with_name(path.name + ".bak")
        backup.write_text(json.dumps(cal, indent=4) + "\n")
        for i in ids:
            cal[NAMES[i]].update(homing_offset=homing[i], range_min=lo[i], range_max=hi[i])
        path.write_text(json.dumps(cal, indent=4) + "\n")
        print(f"updated {path} (previous version: {backup.name})")
    print(f"Next: uv run scripts/record_servos.py map --port {args.port}")
    return 0

# Torque and overload settings of an arm joint (address, size, value): what this arm's untouched joints hold. A
# servo that LeRobot set up as a gripper has 500 / 250 / 25 / 500: half the torque, and after 2 s above 20 % load
# (which holding the arm out is) it falls back to 25 % torque, lets the arm sag, recovers and starts over.
JOINT_LIMITS = {"Max_Torque_Limit": (16, 2, 1000), "Protection_Current": (28, 2, 310), "Protective_Torque": (34, 1, 20),
                "Protection_Time": (35, 1, 200), "Overload_Torque": (36, 1, 80), "Torque_Limit": (48, 2, 1000)}


def cmd_limits(bus: Bus, args) -> int:
    wrong = {}
    for i in sorted(NAMES):
        if bus.ping(i) != "ok":
            raise SystemExit(f"id {i} ({NAMES[i]}) does not answer cleanly: `scan` first")
        have = {n: bus.read(i, a, s) for n, (a, s, _) in JOINT_LIMITS.items()}
        diff = {n: v for n, v in have.items() if v != JOINT_LIMITS[n][2]}
        print(f"id {i} ({NAMES[i]}): " + ", ".join(f"{n} {v}" for n, v in have.items())
              + ("" if not diff else "   <-- not an arm joint's: " + ", ".join(diff)))
        if diff:
            wrong[i] = diff
    if not wrong:
        print("all five joints hold the arm-joint values")
        return 0
    if not args.fix:
        print("--fix writes " + ", ".join(f"{n} {v}" for n, (_, _, v) in JOINT_LIMITS.items()) + " to those servos")
        return 1
    for i, diff in wrong.items():
        bus.put(i, "Torque_Enable", 0)
        bus.put(i, "Lock", 0)  # EEPROM writable
        for n in diff:
            a, s, v = JOINT_LIMITS[n]
            bus.write(i, a, v, s)
        bus.put(i, "Lock", 1)
    time.sleep(0.2)  # late replies to the EEPROM writes would be read as the answers below (seen: garbage values)
    for i in wrong:
        have = {n: bus.read(i, a, s) for n, (a, s, _) in JOINT_LIMITS.items()}
        if any(v != JOINT_LIMITS[n][2] for n, v in have.items()):
            raise SystemExit(f"id {i} ({NAMES[i]}): reads back " + ", ".join(f"{n} {v}" for n, v in have.items()))
        print(f"id {i} ({NAMES[i]}): now " + ", ".join(f"{n} {v}" for n, v in have.items()))
    return 0



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("scan", "set", "calibrate", "limits"))
    ap.add_argument("--port", required=True)
    ap.add_argument("--fix", action="store_true", help="limits: write the arm-joint values where a servo differs")
    ap.add_argument("--from", dest="old", type=int, help="set: the servo's current ID")
    ap.add_argument("--to", dest="new", type=int, help="set: its new ID")
    ap.add_argument("--joints", nargs="+", choices=list(IDS), help="calibrate: the joints whose servo was replaced")
    ap.add_argument("--lerobot-file", help="calibrate: LeRobot's calibration file of this arm, updated for these "
                    "joints (~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json)")
    args = ap.parse_args()
    if args.command == "set" and (args.old is None or args.new is None):
        ap.error("set needs --from and --to")
    if args.command == "calibrate" and not args.joints:
        ap.error("calibrate needs --joints")
    bus = Bus(args.port)
    try:
        return {"scan": cmd_scan, "set": cmd_set, "calibrate": cmd_calibrate, "limits": cmd_limits}[args.command](bus, args)
    finally:
        bus.port.closePort()


if __name__ == "__main__":
    raise SystemExit(main())
