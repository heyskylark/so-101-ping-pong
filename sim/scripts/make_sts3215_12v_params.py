"""Derive starting BAM params for the 12 V STS3215 (ST-3215-C018) from BAM's fitted 7.4 V STS3215 m6 params.

BAM only ships an identified 7.4 V STS3215 (C001). The 12 V version has the same 1:345 gearbox and firmware but a
different motor, so only the motor terms change. BAM's DC motor model gives

    no-load speed  w0 = V / kt
    stall torque   ts = kt * V / R

and we keep BAM's fitted 7.4 V values, scaled by the datasheet ratios between the two versions:

    kt_12 = kt_74 * (12 / 7.4) * (w0_74 / w0_12)
    R_12  = R_74 * (kt_12 / kt_74) * (12 / 7.4) * (ts_74 / ts_12)

Friction, armature, firmware gain ratio, rate limit and delay stay at the 7.4 V fit (same gearbox and firmware).
q_offset is a per-servo calibration term of Rhoban's bench, so it is zeroed. These are starting values only:
servo recording option A (TODO.md §1) replaces them with a fit on our arm.

Datasheets:
  C001 7.4 V: stall 19.5 kg.cm, no-load 0.192 s/60 deg
  C018 12 V:  stall 30 kg.cm,   no-load 0.222 s/60 deg  (feetechrc.com ST-3215-C018 spec, 2023-07-20)

Run: uv run scripts/make_sts3215_12v_params.py   (from sim/)
"""

import json
import math
from pathlib import Path

import bam

KGCM = 9.80665e-2  # N.m per kg.cm
V74, V12 = 7.4, 12.0
STALL_74, STALL_12 = 19.5 * KGCM, 30.0 * KGCM
W0_74, W0_12 = (math.pi / 3) / 0.192, (math.pi / 3) / 0.222

SRC = Path(bam.__file__).parent / "params" / "feetech_sts3215_7_4V" / "m6.json"
DST = Path(__file__).resolve().parents[1] / "src" / "pingpong_sim" / "assets" / "actuators" / "sts3215_12v_m6.json"


def main() -> None:
    p = json.loads(SRC.read_text())
    kt = p["kt"] * (V12 / V74) * (W0_74 / W0_12)
    R = p["R"] * (kt / p["kt"]) * (V12 / V74) * (STALL_74 / STALL_12)
    print(f"kt {p['kt']:.4f} -> {kt:.4f}   R {p['R']:.4f} -> {R:.4f}")
    print(f"model no-load speed at 12 V {V12 / kt:.2f} rad/s (datasheet {W0_12:.2f}), "
          f"stall {kt * V12 / R:.2f} N.m (BAM 7.4 V fit: {p['kt'] * V74 / p['R']:.2f} vs datasheet {STALL_74:.2f})")
    p.update(kt=kt, R=R, q_offset=0.0)
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(p, indent=4) + "\n")
    print("wrote", DST)


if __name__ == "__main__":
    main()
