"""Bounces of a real run (measurements/runs/<time>), timed by the wrist camera, against the bouncer's logged plan.

  uv run scripts/analyze_run.py 2026-11-12_112158 [more runs]

The wrist camera rides on the blade, so the ball's distance from the blade in its picture is zero at contact
whatever the arm does: each bounce is a V in the ball's pixel column over time. A fit of two straight lines (the
fall taken out with the pixel scale from the ball's apparent size) gives the contact time and the ball's speed
relative to the blade before and after. Reported:

- restitution, from bounces off a nearly still blade (during a stroke the blade's own acceleration is in the lines);
- for struck bounces (logs with the plan, from 2026-11-12): planned impact time - actual, and the planned paddle
  speed against the blade's speed along its normal at contact (encoders).

- the ball's apparent size against its true one (`--save-size-scale` writes it to the calibration) and, as a sign
  check, where first bounces go against the calibrated face (`flights`; the face's tilt itself: calibrate.py face).

Speeds use the detected diameter as 40 mm, so they are off by the wrist camera's size scale error; ratios and times
are not affected.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from pingpong_sim.ball import BALL_RADIUS
from pingpong_sim.ball_filter import BallFilterCfg
from pingpong_sim.real.arm import ArmModel
from pingpong_sim.real.camera_model import ArmCalib, CameraCalib

REPO = Path(__file__).resolve().parents[2]
G = 9.81
WINDOW = 0.14  # s either side of the lowest point
STILL = 0.1  # m/s: blade speed below which a bounce counts for the restitution


def bounces(run: str, model: ArmModel, wrist: CameraCalib) -> tuple[list[dict], bool]:
    d = np.load(REPO / "measurements" / "runs" / run / "log.npz", allow_pickle=True)
    t, q, st = d["t"], d["q"], d["state"]
    has_plan = "plan_t_hit" in d.files
    w = d["detections"][d["detection_camera"] == "wrist"]
    tc = w[:, 0] + wrist.stamp_offset + wrist.readout * w[:, 2] / wrist.height  # capture times
    u, dia = w[:, 1], w[:, 3]
    face, normal = [], []
    for x in q:
        face.append(model.face_pose(x)[0])
        normal.append(model.d.site_xmat[model.face].reshape(3, 3)[:, 2].copy())
    vn = np.convolve((np.gradient(np.array(face), t, axis=0) * np.array(normal)).sum(1), np.ones(3) / 3, mode="same")
    rows = []
    for j in range(2, len(w) - 2):
        lo, hi = j, j
        while lo > 0 and tc[j] - tc[lo - 1] < WINDOW and tc[lo] - tc[lo - 1] < 0.06:
            lo -= 1
        while hi < len(w) - 1 and tc[hi + 1] - tc[j] < WINDOW and tc[hi + 1] - tc[hi] < 0.06:
            hi += 1
        if j - lo < 2 or hi - j < 2 or u[j] != u[lo:hi + 1].min() or min(u[lo], u[hi]) - u[j] < 25:
            continue  # not a V: the ball didn't come down to the blade and leave again
        T, U = tc[lo:hi + 1], u[lo:hi + 1]
        k = np.median(dia[lo:hi + 1]) / (2 * BALL_RADIUS)  # px per metre at the ball
        best = None
        for tb in np.arange(tc[j - 1], tc[j + 1], 0.001):
            x = T - tb
            A = np.column_stack([np.ones_like(x), np.where(x < 0, x, 0.0), np.where(x >= 0, x, 0.0)])
            y = U + 0.5 * G * k * x**2
            c = np.linalg.lstsq(A, y, rcond=None)[0]
            r = float(((A @ c - y) ** 2).sum())
            if best is None or r < best[0]:
                best = (r, tb, c)
        r, tb, c = best
        v_in, v_out = c[1] / k, c[2] / k
        if v_in > -0.3 or v_out < 0.2:
            continue
        i = min(int(np.searchsorted(t, tb)), len(t) - 1)
        row = dict(t=tb, state=str(st[i]), n=hi - lo + 1, v_in=v_in, v_out=v_out, e=-v_out / v_in, blade=vn[i],
                   rms_px=float(np.sqrt(r / (hi - lo + 1))))
        if has_plan:
            ip = max(0, i - 4)  # the plan in force just before contact (no replanning during a stroke)
            row.update(early=d["plan_t_hit"][ip] - tb, planned=d["plan_speed"][ip])
        rows.append(row)
    return rows, has_plan


def flights(run: str, model: ArmModel, wrist: CameraCalib, rows: list[dict]) -> list[dict]:
    """The ball's flights between two timed bounces on the blade while playing. Between them its height is known
    without the camera's range (a parabola through the two contacts, no drag: ~1 % high at the top), so each wrist
    detection's ray gives its true position. Per flight:

    - `ratio`: apparent / true angular diameter per detection. The filter takes range from the apparent size: a
      ball that looks 20 % big is put 20 % closer to the camera (4 cm short along the blade) and 20 % lower, and
      its next impact is predicted ~30 ms early.
    - `first`: a bounce off the waiting blade with nothing timed in the 0.8 s before: a ball dropped from a still
      hand, so the horizontal speed `v_h` of the flight is the blade's doing: the true face normal's lean from
      vertical, less `model_lean`, the normal the encoders give, is the face's tilt against the calibrated model
      (main() turns v_h into the lean).
    - `landed`: where the flight began, from the blade centre (x, y). Bounces off the table or a hand are far off.
    - `v_out`, `v_end`: vertical speed leaving the first bounce and arriving at the second, from the flight time
      and the blade's heights alone; `blade`: the blade's speed along its normal at the first bounce (encoders).
    """
    import torch

    from pingpong_sim.controller import ArmHistory
    from pingpong_sim.kinematics import ArmKinematics
    from pingpong_sim.real.camera_model import PixelDetection, apply_wrist_mount, to_measurements

    d = np.load(REPO / "measurements" / "runs" / run / "log.npz", allow_pickle=True)
    t, q = d["t"], d["q"]
    w = d["detections"][d["detection_camera"] == "wrist"]
    tc = w[:, 0] + wrist.stamp_offset
    kin = ArmKinematics(model.m, prefix="", device="cpu")
    apply_wrist_mount(kin, wrist)
    play = [r for r in rows if r["state"] == "PLAY"]
    out = []
    for a, b in zip(play[:-1], play[1:]):
        T = b["t"] - a["t"]
        seg = w[(tc > a["t"] + 0.02) & (tc < b["t"] - 0.02)]
        if not 0.2 < T < 0.5 or len(seg) < 4:
            continue
        i0, i1 = np.searchsorted(t, [a["t"] - 0.3, b["t"] + 0.05])
        hist = ArmHistory(kin, 1, 0.01, "cpu")  # the encoders' arm around this flight: the camera's pose per detection
        for i in range(i0, i1):
            hist.push(float(t[i]), torch.tensor(q[i], dtype=torch.float32)[None])
        m = to_measurements([PixelDetection("wrist", *x) for x in seg], {"wrist": wrist}, hist.hist, t0=0.0)
        ia = min(int(np.searchsorted(t, a["t"])), len(t) - 1)
        za, zb = model.face_pose(q[ia])[0][2], model.face_pose(q[min(int(np.searchsorted(t, b["t"])), len(t) - 1)])[0][2]
        ratio, ranges, taus, pts, cams = [], [], [], [], []
        for c in range(m["t"].shape[0]):
            tau = float(m["t"][c, 0]) - a["t"]
            h = za + (zb - za) * tau / T + BALL_RADIUS + 0.5 * G * tau * (T - tau)  # ball centre height
            z, R, cp = (m[key][c, 0].numpy() for key in ("z", "R", "p"))
            ray = R @ np.array([z[0], z[1], 1.0])
            s = (h - cp[2]) / ray[2] if abs(ray[2]) > 1e-3 else -1.0
            if s > 0:
                ranges.append(s * np.linalg.norm(ray))
                ratio.append(z[2] / (2 * np.arcsin(min(BALL_RADIUS / ranges[-1], 0.99))))
                taus.append(tau)
                pts.append(cp + s * ray)
                cams.append(cp)
        if len(taus) < 4:
            continue
        pts = np.array(pts)
        v_h, p0 = np.polyfit(taus, pts[:, :2], 1)  # horizontal velocity of the flight (x, y), where it began
        face = model.face_pose(q[ia])[0]
        n = model.d.site_xmat[model.face].reshape(3, 3)[:, 2]
        first = (abs(a["blade"]) < 0.05 and a.get("planned", 0.0) == 0.0
                 and not any(0 < a["t"] - r["t"] < 0.8 for r in rows))
        out.append(dict(ratio=np.array(ratio), range=np.array(ranges), first=first, model_lean=n[:2] / n[2],
                        t=a["t"], T=T, v_h=v_h, landed=p0 - face[:2], blade=a["blade"],
                        tau=np.array(taus), xyz=pts, cam=np.array(cams),
                        v_out=(zb - za) / T + 0.5 * G * T, v_end=0.5 * G * T - (zb - za) / T))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="directory names under measurements/runs")
    ap.add_argument("--after", type=float, default=0.0, help="ignore bounces before this time [s] (hand in view)")
    ap.add_argument("--save-size-scale", action="store_true",
                    help="save the wrist camera's ball size scale measured from these runs (redo after changing the "
                         "light on the ball, the exposure or the colour thresholds: the blob's size depends on them)")
    args = ap.parse_args()
    model, wrist = ArmModel(), CameraCalib.load("wrist")
    ratios, firsts, es = [np.zeros(0)], [], []
    for run in args.runs:
        rows, has_plan = bounces(run, model, wrist)
        rows = [r for r in rows if r["t"] >= args.after]
        print(f"{run}: {len(rows)} bounces timed by the wrist camera")
        print("    time  state    pts  in / out [m/s]    e    blade [m/s]" + ("  planned [m/s]  planned - actual [ms]" if has_plan else ""))
        for r in rows:
            print(f"  {r['t']:6.2f}  {r['state']:7s}  {r['n']:2d}   {r['v_in']:+.2f} / {r['v_out']:+.2f}   {r['e']:.2f}   {r['blade']:+.2f}"
                  + (f"         {r['planned']:.2f}          {1e3 * r['early']:+6.0f}" if has_plan else ""))
        still = np.array([r["e"] for r in rows if abs(r["blade"]) < STILL and r["rms_px"] < 15])
        if len(still):
            print(f"  restitution off the still blade: median {np.median(still):.2f} over {len(still)} bounces "
                  f"(range {still.min():.2f}-{still.max():.2f})")
        struck = [r for r in rows if has_plan and r["state"] == "PLAY" and r["planned"] > 0.15 and abs(r["early"]) < 0.2]
        if struck:
            early = 1e3 * np.array([r["early"] for r in struck])
            print(f"  struck bounces ({len(struck)}): planned impact - actual median {np.median(early):+.0f} ms (quartiles "
                  f"{np.percentile(early, 25):+.0f} .. {np.percentile(early, 75):+.0f}); planned speed median "
                  f"{np.median([r['planned'] for r in struck]):.2f}, blade at contact {np.median([r['blade'] for r in struck]):+.2f} m/s")
        fl = [f for f in flights(run, model, wrist, rows) if np.hypot(*f["landed"]) < 0.08]  # began on the blade
        ratios += [f["ratio"] for f in fl]
        firsts += [f for f in fl if f["first"]]
        # Bounces with a timed flight before and after: both speeds from the flight times, the blade's from the encoders.
        es += [(f1["v_out"] - f1["blade"]) / (f0["v_end"] + f1["blade"]) for f0, f1 in zip(fl[:-1], fl[1:])
               if abs(f1["t"] - f0["t"] - f0["T"]) < 1e-6]
    e = BallFilterCfg().restitution
    if len(es) >= 5:
        e = float(np.median(es))
        print(f"restitution, {len(es)} bounces with a timed flight before and after (speeds from the flight times, the "
              f"blade's from the encoders): {e:.2f} (quartiles {np.percentile(es, 25):.2f} .. {np.percentile(es, 75):.2f})")
    # A ball falling at v_in on a face leaning by a small angle a leaves with e v_in a along the normal's lean and,
    # of its v_in a along the face, keeps the fraction `tangential_keep` (the rubber grips): v_h = a v_in (e + keep).
    keep = BallFilterCfg().tangential_keep
    tilts = [f["v_h"] / (f["v_out"] * (1 + keep / e)) - f["model_lean"] for f in firsts]
    r = np.concatenate(ratios)
    if len(r) >= 20:
        new = wrist.size_scale * float(np.median(r))
        # Standard error of the median, from the quartiles (a few detections are far off: a hand, a blurred frame).
        se = 1.25 * (np.percentile(r, 75) - np.percentile(r, 25)) / 1.349 / np.sqrt(len(r))
        print(f"ball size, {len(r)} detections in flights between two timed bounces on the blade: apparent / true "
              f"diameter {np.median(r):.3f} (quartiles {np.percentile(r, 25):.3f} .. {np.percentile(r, 75):.3f}) with the "
              f"saved size scale {wrist.size_scale:.3f} -> the size scale should be {new:.3f} +- {wrist.size_scale * se:.3f}")
        if args.save_size_scale:
            wrist.size_scale = new
            wrist.steps["size_scale"] = {"date": dt.datetime.now().strftime("%Y-%m-%d_%H%M%S"), "runs": args.runs,
                                         "detections": int(len(r)), "method": "flights between timed bounces"}
            print("saved", wrist.save())
    elif args.save_size_scale:
        raise SystemExit(f"only {len(r)} detections in flights between timed bounces: nothing saved")
    if len(tilts) >= 5:
        tl = np.degrees(np.array(tilts))
        med = np.median(tl, axis=0)
        se = 1.25 * (np.percentile(tl, 75, axis=0) - np.percentile(tl, 25, axis=0)) / 1.349 / np.sqrt(len(tl))
        arm = ArmCalib.load()
        print(f"blade tilt, {len(tl)} first bounces off the waiting blade (where a dropped ball goes): as if the hitting "
              f"face leant {med[0]:+.1f} +- {se[0]:.1f} deg forward and {med[1]:+.1f} +- {se[1]:.1f} deg toward +y of the "
              f"calibrated model without its face tilt (saved, by ruler: {np.degrees(arm.face_tilt[0]):+.1f}, "
              f"{np.degrees(arm.face_tilt[1]):+.1f}). A sign check only: on 2026-11-12 this read three times a "
              f"ruler's lean (calibrate.py face)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
