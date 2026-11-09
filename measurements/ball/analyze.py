# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy", "scipy", "matplotlib", "pyyaml"]
# ///
"""Ball physics from the cached drop-video tracks (see README.md).

    uv run measurements/ball/analyze.py

Reads tracks/*.csv (made by track.py) and writes results.yaml plus diagnostic plots
into this directory.

Model (all videos)
------------------
* Time: t_real = t_video / K with K = 8 (240 fps capture on a 30 fps timebase). Checked
  per flight by fitting each flight's own scale.
* Vertical motion of a flight: gravity + quadratic air drag, closed form, parametrised by
  apex time t_a and apex height Y_a (world metres):
      up   (tau<0): Y = Y_a + vt^2/g * ln cos(g tau / vt)
      down (tau>0): Y = Y_a - vt^2/g * ln cosh(g tau / vt)
  with vt = sqrt(g / k), k = rho Cd A / (2 m).
* Camera: y_up_px = P0 + S * r(t) * Y / (1 + gamma * Y), a 1D pinhole projectivity of a
  vertical line (gamma absorbs camera pitch, ~ -pitch/distance). S [px/m] is the local
  vertical scale at the reference row P0 (the paddle contact row for paddle videos, so
  impact velocities use the scale *at the impact point*). r(t) = ball width(t)/width_ref
  corrects for depth changes (used for the 6 ft drops only).
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import brentq, least_squares

HERE = Path(__file__).resolve().parent
TRACKS = HERE / "tracks"

G = 9.80  # m/s^2
K = 8.0  # slow-motion factor (240 fps capture / 30 fps timebase)
BALL_M = 2.75e-3  # kg
BALL_D = 0.040  # m
BALL_A = math.pi * (BALL_D / 2) ** 2
RHO = 1.2  # kg/m^3
INCH = 0.0254

MIN_FLIGHT_PTS = 8
MIN_FLIGHT_HEIGHT_M = 0.003  # apex above contact
RES_TOL_PX = 3.0  # base residual tolerance when growing a flight
IMPACT_LEVEL_TOL_M = 0.020  # |impact height - paddle contact row|; larger = mis-segmented
FLIGHT_RMS_LIMIT_PX = 3.0  # per-flight rms limit (or 3x the video median, if larger)


def k_drag(cd: float) -> float:
    return RHO * cd * BALL_A / (2 * BALL_M)


def vterm(cd: float) -> float:
    return math.inf if cd <= 0 else math.sqrt(G / k_drag(cd))


def height(tau: np.ndarray, vt: float) -> np.ndarray:
    """Height relative to apex [m] at real time tau after apex."""
    if not math.isfinite(vt):
        return -0.5 * G * tau**2
    x = G * tau / vt
    up = (vt**2 / G) * np.log(np.cos(np.clip(x, -1.5, 0.0)))
    down = -(vt**2 / G) * np.log(np.cosh(np.clip(x, 0.0, None)))
    return np.where(tau < 0, up, down)


def clean(o):
    """numpy scalars/arrays -> plain Python for YAML."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    return o


def velocity(tau: np.ndarray, vt: float) -> np.ndarray:
    if not math.isfinite(vt):
        return -G * tau
    x = G * tau / vt
    return np.where(tau < 0, -vt * np.tan(np.clip(x, -1.5, 0.0)), -vt * np.tanh(np.clip(x, 0.0, None)))


def project(Y, S, gamma, P0, r=1.0):
    return P0 + S * r * Y / (1.0 + gamma * Y)


# --------------------------------------------------------------------------- data


@dataclass
class Track:
    name: str
    t: np.ndarray  # video seconds
    x: np.ndarray  # px
    yu: np.ndarray  # px, up positive (= -y_px)
    d: np.ndarray
    w: np.ndarray
    flags: np.ndarray


def load(name: str) -> Track:
    a = np.loadtxt(TRACKS / f"{name}.csv", delimiter=",", skiprows=1, ndmin=2)
    return Track(name, a[:, 0], a[:, 1], -a[:, 2], a[:, 3], a[:, 4], a[:, 6].astype(int))


def load_scene(name: str) -> dict:
    """Blade/tape orientation written by track.py (tracks/<name>_scene.csv)."""
    with (TRACKS / f"{name}_scene.csv").open() as fh:
        row = next(csv.DictReader(fh))
    return {k: float(v) for k, v in row.items()}


# --------------------------------------------------------------------------- segmentation


def bottom_minima(tr: Track, dball: float) -> list[int]:
    """Indices of bottom turning points (impacts): local minima of height with a clear
    descent before and ascent after."""
    y, t = tr.yu, tr.t
    thr = max(0.05 * dball, 2.5)
    out = []
    for i in range(2, len(y) - 2):
        lo, hi = max(0, i - 4), min(len(y), i + 5)
        if y[i] > y[lo:hi].min():
            continue
        if t[min(i + 2, len(t) - 1)] - t[max(i - 2, 0)] > 0.5:  # gap in detections
            continue
        if y[lo:i].max() - y[i] > thr and y[i + 1 : hi].max() - y[i] > thr:
            if out and i - out[-1] < 3:
                continue
            out.append(i)
    return out


def parab_fit(t, y):
    A = np.vstack([t**2, t, np.ones_like(t)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c, y - A @ c


def grow(t, y, idx: np.ndarray, seed: np.ndarray, can_left=True, can_right=True) -> np.ndarray:
    """Grow a contiguous free-fall window from `seed` (indices into idx) outward while the
    next point agrees with the parabola fitted to the current window."""
    lo, hi = int(seed[0]), int(seed[-1])
    while True:
        sel = idx[lo : hi + 1]
        c, res = parab_fit(t[sel], y[sel])
        rms = float(np.sqrt(np.mean(res**2))) if len(sel) > 3 else 0.0
        tol = max(RES_TOL_PX, 3.5 * rms)
        moved = False
        for side in ("L", "R"):
            j = lo - 1 if side == "L" else hi + 1
            if (side == "L" and not can_left) or (side == "R" and not can_right) or j < 0 or j >= len(idx):
                continue
            k = idx[j]
            # stop at detection gaps > 4 frame periods (track lost)
            nb = idx[lo] if side == "L" else idx[hi]
            if abs(t[k] - t[nb]) > 4 * (1 / 30) + 1e-6:
                continue
            pred = np.polyval(c, t[k])
            if abs(y[k] - pred) < tol:
                if side == "L":
                    lo = j
                else:
                    hi = j
                moved = True
        if not moved:
            break
    return idx[lo : hi + 1]


@dataclass
class Flight:
    idx: np.ndarray
    start_min: int | None  # bottom-minimum index that starts the flight (None = start of track)
    end_min: int | None
    kind: str  # "drop" (from hand), "bounce", "partial"
    ok: bool = True
    reject: str = ""
    params: dict = field(default_factory=dict)


def drop_seed(ys: np.ndarray, y_imp: float) -> np.ndarray:
    """Seed window for a hand drop ending in an impact at height y_imp: the part of the
    fall between 10% and 60% of the drop height after the highest point, which is surely
    free flight, clear of hand and paddle."""
    top = int(np.argmax(ys))
    H = ys[top] - y_imp
    band = np.where((np.arange(len(ys)) > top) & (ys > y_imp + 0.1 * H) & (ys < y_imp + 0.6 * H))[0]
    if len(band) >= 4:
        return np.arange(band[0], band[-1] + 1)
    n = len(ys)
    return np.arange(max(n - 7, 0), n - 1)


def segment_paddle(tr: Track, dball: float) -> tuple[list[Flight], list[int]]:
    mins = bottom_minima(tr, dball)
    bounds = [None] + mins + [None]
    flights = []
    t, y = tr.t, tr.yu
    for a, b in zip(bounds[:-1], bounds[1:]):
        i0 = 0 if a is None else a
        i1 = len(t) - 1 if b is None else b
        idx = np.arange(i0, i1 + 1)
        if len(idx) < 4:
            continue
        n = len(idx)
        if a is None and b is not None:  # drop from the hand
            sel = grow(t, y, idx, drop_seed(y[idx], y[b]))
            kind = "drop"
        elif a is not None and b is not None:  # rebound between two impacts
            q = max(1, n // 4)
            seed = np.arange(q, n - q) if n - 2 * q >= 3 else np.arange(1, n - 1)
            sel = grow(t, y, idx, seed)
            kind = "bounce"
        else:  # after the last impact
            sel = grow(t, y, idx, np.arange(1, min(8, n)))
            kind = "bounce"
        flights.append(Flight(sel, a, b, kind))
    return flights, mins


def segment_gaps(tr: Track, gap=0.3) -> list[Flight]:
    """6 ft drops: flights are the contiguous detection runs (floor contact is off-frame)."""
    t, y = tr.t, tr.yu
    cuts = np.where(np.diff(t) > gap)[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(t) - 1]
    out = []
    for s, e in zip(starts, ends):
        idx = np.arange(s, e + 1)
        if len(idx) < MIN_FLIGHT_PTS:
            continue
        n = len(idx)
        q = max(1, n // 4)
        sel = grow(t, y, idx, np.arange(q, n - q))
        out.append(Flight(sel, None, None, "partial"))
    return out


# --------------------------------------------------------------------------- fitting


def init_flight(t, y, S, P0):
    """apex time (real s) and apex height (m above P0) from a free parabola in video time."""
    c, _ = parab_fit(t, y)
    a, b, cc = c
    if a >= 0:  # not a valid free fall; fall back to the highest point
        i = int(np.argmax(y))
        return t[i] / K, (y[i] - P0) / S
    tv = -b / (2 * a)
    ya = cc - b * b / (4 * a)
    return tv / K, (ya - P0) / S


@dataclass
class VideoFit:
    S: float
    gamma: float
    cd: float
    flights: list[tuple[float, float]]  # (t_a, Y_a)
    cov: np.ndarray
    names: list[str]
    rms: float
    n: int
    x: np.ndarray


def fit_video(tr: Track, flights: list[Flight], P0: float, S0: float, cd: float,
              fit_gamma: bool, fit_cd: bool, r_of_t=None, gamma0: float = 0.0) -> VideoFit:
    t, y = tr.t, tr.yu
    fl = [f for f in flights if f.ok]
    inits = [init_flight(t[f.idx], y[f.idx], S0, P0) for f in fl]
    x0 = [S0] + ([gamma0] if fit_gamma else []) + ([cd] if fit_cd else [])
    for ta, ya in inits:
        x0 += [ta, ya]
    x0 = np.array(x0, float)
    ng = 1 + int(fit_gamma) + int(fit_cd)
    rr = [np.ones(len(f.idx)) if r_of_t is None else r_of_t(f, t[f.idx]) for f in fl]

    def unpack(p):
        S = p[0]
        gm = p[1] if fit_gamma else gamma0
        c = p[1 + int(fit_gamma)] if fit_cd else cd
        return S, gm, c

    def resid(p):
        S, gm, c = unpack(p)
        vt = vterm(max(c, 0.0))
        out = []
        for i, f in enumerate(fl):
            ta, ya = p[ng + 2 * i], p[ng + 2 * i + 1]
            Y = ya + height(t[f.idx] / K - ta, vt)
            out.append(project(Y, S, gm, P0, rr[i]) - y[f.idx])
        return np.concatenate(out)

    lb = np.full(len(x0), -np.inf)
    ub = np.full(len(x0), np.inf)
    lb[0] = 0.2 * S0
    ub[0] = 5 * S0
    if fit_cd:
        lb[1 + int(fit_gamma)] = 0.0
        ub[1 + int(fit_gamma)] = 3.0
    if fit_gamma:
        lb[1], ub[1] = -3.0, 3.0
    x0 = np.clip(x0, lb + 1e-9, ub - 1e-9)
    sol = least_squares(resid, x0, bounds=(lb, ub), x_scale="jac")
    r = sol.fun
    dof = max(len(r) - len(sol.x), 1)
    s2 = float(r @ r) / dof
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * s2
    except np.linalg.LinAlgError:
        cov = np.full((len(sol.x), len(sol.x)), np.nan)
    S, gm, c = unpack(sol.x)
    names = ["S"] + (["gamma"] if fit_gamma else []) + (["Cd"] if fit_cd else [])
    for i in range(len(fl)):
        names += [f"ta{i}", f"Ya{i}"]
    fpar = [(sol.x[ng + 2 * i], sol.x[ng + 2 * i + 1]) for i in range(len(fl))]
    return VideoFit(S, gm, c, fpar, cov, names, math.sqrt(s2), len(r), sol.x)


def single_flight_scale(tr: Track, f: Flight, P0, S0, gamma, cd) -> tuple[float, float]:
    """Fit one flight alone with a free scale S_f (K fixed) -> (S_f, sigma)."""
    vf = fit_video(tr, [Flight(f.idx, None, None, f.kind)], P0, S0, cd, False, False, gamma0=gamma)
    return vf.S, float(math.sqrt(max(vf.cov[0, 0], 0.0)))


# --------------------------------------------------------------------------- paddle videos

NOMINAL_RE = re.compile(r"c-(\d+(?:_\d+)?)in-(\d)")


def nominal_height_m(name: str) -> float | None:
    m = NOMINAL_RE.match(name)
    return None if not m else float(m.group(1).replace("_", ".")) * INCH


def impact_from(vf: VideoFit, i: int, cd: float, P0: float):
    """Impact between flight i and i+1 of a video fit: time where the two height curves
    meet. Returns (t_real, Y_imp, v_in, v_out)."""
    vt = vterm(cd)
    (ta1, ya1), (ta2, ya2) = vf.flights[i], vf.flights[i + 1]

    def diff(tt):
        return (ya1 + height(np.array([tt - ta1]), vt) - ya2 - height(np.array([tt - ta2]), vt))[0]

    lo, hi = ta1 + 1e-6, ta2 - 1e-6
    if not (hi > lo) or diff(lo) * diff(hi) > 0:
        return None
    ti = brentq(diff, lo, hi, xtol=1e-9)
    Y = ya1 + float(height(np.array([ti - ta1]), vt)[0])
    vin = float(velocity(np.array([ti - ta1]), vt)[0])
    vout = float(velocity(np.array([ti - ta2]), vt)[0])
    return ti, Y, vin, vout


def impact_with_errors(vf: VideoFit, i: int, cd: float, P0: float, ng: int):
    """Impact quantities in m/s (velocities measured with the local scale at Y_imp) and
    1-sigma errors from linear propagation of the fit covariance."""

    def quantities(x):
        S = x[0]
        gm = x[1] if "gamma" in vf.names else vf.gamma
        fl = [(x[ng + 2 * j], x[ng + 2 * j + 1]) for j in range(len(vf.flights))]
        tmp = VideoFit(S, gm, cd, fl, vf.cov, vf.names, vf.rms, vf.n, x)
        imp = impact_from(tmp, i, cd, P0)
        if imp is None:
            return None
        ti, Y, vin, vout = imp
        # the model is in world metres already; S/gamma only enter through the fitted
        # Y_a, so velocities come out in m/s directly.
        return np.array([ti, Y, vin, vout, -vout / vin])

    q0 = quantities(vf.x)
    if q0 is None:
        return None
    J = np.zeros((5, len(vf.x)))
    for j in range(len(vf.x)):
        h = 1e-6 * max(abs(vf.x[j]), 1e-3)
        xp = vf.x.copy()
        xp[j] += h
        qp = quantities(xp)
        if qp is None:
            return None
        J[:, j] = (qp - q0) / h
    C = J @ vf.cov @ J.T
    return q0, np.sqrt(np.clip(np.diag(C), 0, None))


def lateral_velocity(tr: Track, f: Flight, S: float) -> float:
    """x velocity [m/s] from a linear fit of x over the flight (horizontal scale ~ S)."""
    tt = tr.t[f.idx] / K
    c = np.polyfit(tt, tr.x[f.idx], 1)
    return float(c[0] / S)


def subset(tr: Track, keep: np.ndarray) -> Track:
    return Track(tr.name, tr.t[keep], tr.x[keep], tr.yu[keep], tr.d[keep], tr.w[keep], tr.flags[keep])


GAMMA_MIN_SPAN_M = 0.15  # fit the perspective term only if flights span this much height


def analyze_paddle(name: str, cd: float, fit_gamma: bool | None = None):
    tr = load(name)
    dball = float(np.median(tr.w))
    # drop detections far from the ball's (vertical) path: other orange objects picked up
    # while the ball was hidden in the hand
    keep = np.abs(tr.x - np.median(tr.x)) < 3 * dball
    n_off_path = int((~keep).sum())
    tr = subset(tr, keep)
    flights, mins = segment_paddle(tr, dball)
    if not mins:
        return {"name": name, "error": "no impacts found"}, None
    P0 = float(np.median(tr.yu[mins]))  # paddle contact row (ball centre)
    S_d = dball / BALL_D  # px/m from ball width
    # reject short / tiny flights
    for f in flights:
        ext = (tr.yu[f.idx].max() - P0) / S_d
        if len(f.idx) < MIN_FLIGHT_PTS:
            f.ok, f.reject = False, f"too few points ({len(f.idx)})"
        elif ext < MIN_FLIGHT_HEIGHT_M:
            f.ok, f.reject = False, f"too low ({ext * 1000:.1f} mm)"
    good = [f for f in flights if f.ok]
    if len(good) < 2:
        return {"name": name, "error": "fewer than 2 usable flights"}, None

    span = (max(tr.yu[f.idx].max() for f in good) - P0) / S_d
    if fit_gamma is None:
        fit_gamma = span > GAMMA_MIN_SPAN_M
    vf = fit_video(tr, flights, P0, S_d, cd, fit_gamma=fit_gamma, fit_cd=False)
    # flights the joint model cannot describe (rolling, mis-tracked) -> reject and refit
    vt = vterm(cd)
    frms = []
    for (ta, ya), f in zip(vf.flights, good):
        r = project(ya + height(tr.t[f.idx] / K - ta, vt), vf.S, vf.gamma, P0) - tr.yu[f.idx]
        frms.append(float(np.sqrt(np.mean(r**2))))
    lim = max(FLIGHT_RMS_LIMIT_PX, 3 * float(np.median(frms)))
    bad = [f for f, r in zip(good, frms) if r > lim]
    for f, r in zip(good, frms):
        if r > lim:
            f.ok, f.reject = False, f"not free flight (rms {r:.1f} px vs video median {np.median(frms):.1f})"
    if bad:
        good = [f for f in flights if f.ok]
        if len(good) < 2:
            return {"name": name, "error": "fewer than 2 usable flights"}, None
        vf = fit_video(tr, flights, P0, S_d, cd, fit_gamma=fit_gamma, fit_cd=False)
    ng = 1 + int(fit_gamma)

    # per-flight timescale check
    fl_out = []
    for j, f in enumerate(good):
        if len(f.idx) >= 12 and (tr.yu[f.idx].max() - tr.yu[f.idx].min()) / S_d > 0.01:
            Sf, sSf = single_flight_scale(tr, f, P0, vf.S, vf.gamma, cd)
            # K_f: slowdown implied by this flight's curvature if the true scale is S_d
            Kf = K * math.sqrt(S_d / Sf)
            ratio = Sf / vf.S
        else:
            Sf = sSf = Kf = ratio = None
        f.params = dict(S_f=Sf, S_f_err=sSf, K_f=Kf, S_f_over_S=ratio)
        ta, ya = vf.flights[j]
        fl_out.append(dict(
            kind=f.kind, n_points=int(len(f.idx)),
            t_video=[round(float(tr.t[f.idx[0]]), 4), round(float(tr.t[f.idx[-1]]), 4)],
            apex_height_mm=round(1000 * ya, 1),
            S_f_px_per_m=None if Sf is None else round(Sf, 1),
            S_f_err=None if sSf is None else round(sSf, 1),
            K_implied_by_ball_diameter=None if Kf is None else round(Kf, 3),
            vx_m_s=round(lateral_velocity(tr, f, vf.S), 3),
        ))
    # flag flights inconsistent with the video timescale (slow-mo ramp)
    for f, fo in zip(good, fl_out):
        r = f.params["S_f_over_S"]
        if r is not None:
            bad = abs(r - 1) > 0.15 and abs(f.params["S_f"] - vf.S) > 3 * f.params["S_f_err"]
            fo["timescale_ok"] = not bad
            if bad:
                f.reject = f"flight scale {r:.2f}x video scale (slow-mo ramp?)"

    impacts, rejected = [], []
    good_pos = {id(f): j for j, f in enumerate(good)}
    for a, b in zip(flights[:-1], flights[1:]):
        if a.end_min is None or a.end_min != b.start_min:
            continue
        tv = float(tr.t[a.end_min])
        if not (a.ok and b.ok):
            rejected.append(dict(video=name, t_video=round(tv, 3),
                                 reason="; ".join(r for r in (a.reject, b.reject) if r) or "neighbour flight rejected"))
            continue
        if a.reject or b.reject:
            rejected.append(dict(video=name, t_video=round(tv, 3), reason=a.reject or b.reject))
            continue
        i = good_pos[id(a)]
        res = impact_with_errors(vf, i, cd, P0, ng)
        if res is None:
            rejected.append(dict(video=name, t_video=round(tv, 3), reason="flight curves do not intersect"))
            continue
        (ti, Y, vin, vout, e), (sti, sY, svin, svout, se) = res
        y_imp_px = project(Y, vf.S, vf.gamma, P0)
        rec = dict(video=name, index=len(impacts), t_video=round(ti * K, 4), v_in=round(-vin, 4),
                   v_out=round(vout, 4), e=round(e, 4), v_in_err=round(svin, 4),
                   v_out_err=round(svout, 4), e_err=round(se, 4),
                   impact_row_offset_mm=round(1000 * Y, 1),
                   first=a.kind == "drop",
                   vx_in=fl_out[i]["vx_m_s"], vx_out=fl_out[i + 1]["vx_m_s"])
        if abs(Y) > IMPACT_LEVEL_TOL_M:
            rejected.append(dict(video=name, t_video=round(ti * K, 3),
                                 reason=f"impact {1000 * Y:.0f} mm off the paddle contact level"))
            continue
        impacts.append(rec)

    hnom = nominal_height_m(name)
    first = next((r for r in impacts if r["first"]), None)
    drop_check = None
    if first is not None:
        ta, ya = vf.flights[0]
        drop_check = dict(
            nominal_release_height_m=None if hnom is None else round(hnom, 4),
            v_free_fall_from_nominal=None if hnom is None else round(math.sqrt(2 * G * hnom), 3),
            v_in_measured=first["v_in"],
            fitted_fall_height_m=round(ya, 4),  # apex (release) height of the ball centre above contact
        )
        if hnom is not None and ya < 0.5 * hnom:
            drop_check["note"] = "release not captured: first tracked flight is already a rebound"
            first["first"] = False
    out = dict(
        name=name,
        n_detections_off_path_dropped=n_off_path,
        ball_width_px=round(dball, 1),
        S_ball_px_per_m=round(S_d, 1),
        S_v_px_per_m=round(vf.S, 1),
        S_v_err=round(math.sqrt(vf.cov[0, 0]), 1),
        S_v_over_S_ball=round(vf.S / S_d, 4),
        height_span_m=round(span, 3),
        gamma_per_m=round(vf.gamma, 4),
        gamma_err=None if not fit_gamma else round(math.sqrt(vf.cov[1, 1]), 4),
        gamma_note=None if fit_gamma else f"perspective term fixed at 0 (flights span < {GAMMA_MIN_SPAN_M} m)",
        fit_rms_px=round(vf.rms, 2),
        flights=fl_out,
        n_flights_found=len(flights),
        n_flights_used=len(good),
        flights_rejected=[dict(t_video=[round(float(tr.t[f.idx[0]]), 3), round(float(tr.t[f.idx[-1]]), 3)],
                               reason=f.reject) for f in flights if not f.ok],
        drop_check=drop_check,
    )
    diag = dict(tr=tr, flights=flights, vf=vf, P0=P0, cd=cd, impacts=impacts, ng=ng)
    return out, (impacts, rejected, diag)


# --------------------------------------------------------------------------- 6 ft drops


@dataclass
class DropSet:
    name: str
    tr: Track
    flights: list[Flight]
    wref: float  # median ball width [px]
    wfit: list[np.ndarray]  # per-flight linear width(t) fit (depth correction)

    @property
    def S_ball(self) -> float:
        return self.wref / BALL_D


def prep_6ft(name: str) -> DropSet:
    tr = load(name)
    flights = segment_gaps(tr)
    wref = float(np.median(tr.w[np.concatenate([f.idx for f in flights])]))
    wfit = [np.polyfit(tr.t[f.idx], tr.w[f.idx], 1) for f in flights]
    return DropSet(name, tr, flights, wref, wfit)


P0_CENTRE = -960.0  # image centre row (principal point) in y_up coordinates


def fit_6ft(sets: list[DropSet], fit_pitch: bool, pin_scale: bool, depth_corr: bool = True,
            fit_cd: bool = True, cd0: float = 0.45, pin_factor: float = 1.0) -> dict:
    """Joint fit of the 6 ft drops: shared Cd and camera-pitch term, per-video scale S_i,
    per-flight (t_a, Y_a). Perspective: gamma_i = c * wref_i/100 because gamma = -pitch/Z
    and Z is proportional to 1/ball width (same camera pitch in all three videos)."""
    nv = len(sets)
    nS = 0 if pin_scale else nv
    head = int(fit_cd) + int(fit_pitch) + nS

    def unpack(p):
        k = 0
        cd = p[k] if fit_cd else cd0
        k += int(fit_cd)
        c = p[k] if fit_pitch else 0.0
        k += int(fit_pitch)
        S = [pin_factor * ds.S_ball for ds in sets] if pin_scale else list(p[k : k + nv])
        return cd, c, S

    def per_flight(p):
        cd, c, S = unpack(p)
        vt = vterm(max(cd, 0.0))
        k = head
        out = []
        for v, ds in enumerate(sets):
            gm = c * ds.wref / 100.0
            for f, wf in zip(ds.flights, ds.wfit):
                tt = ds.tr.t[f.idx]
                r = np.polyval(wf, tt) / ds.wref if depth_corr else 1.0
                Y = p[k + 1] + height(tt / K - p[k], vt)
                out.append((ds, f, project(Y, S[v], gm, P0_CENTRE, r) - ds.tr.yu[f.idx], p[k], vt))
                k += 2
        return out

    x0 = ([cd0] if fit_cd else []) + ([0.0] if fit_pitch else []) + ([ds.S_ball for ds in sets] if not pin_scale else [])
    for ds in sets:
        for f in ds.flights:
            x0 += list(init_flight(ds.tr.t[f.idx], ds.tr.yu[f.idx], ds.S_ball, P0_CENTRE))
    x0 = np.array(x0, float)
    lb = np.full(len(x0), -np.inf)
    ub = np.full(len(x0), np.inf)
    k = 0
    if fit_cd:
        lb[k], ub[k] = 0.0, 3.0
        k += 1
    if fit_pitch:
        lb[k], ub[k] = -2.0, 2.0
        k += 1
    for _ in range(nS):
        lb[k], ub[k] = 0.5 * x0[k], 2.0 * x0[k]
        k += 1
    x0 = np.clip(x0, lb + 1e-9, ub - 1e-9)
    sol = least_squares(lambda p: np.concatenate([r for _, _, r, _, _ in per_flight(p)]), x0,
                        bounds=(lb, ub), x_scale="jac")
    dof = max(len(sol.fun) - len(sol.x), 1)
    s2 = float(sol.fun @ sol.fun) / dof
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * s2
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    cd, c, S = unpack(sol.x)
    res = dict(rms_px=round(math.sqrt(s2), 2), n_points=int(len(sol.fun)))
    k = 0
    if fit_cd:
        res.update(Cd=round(cd, 3), Cd_err_formal=round(float(sd[0]), 3))
        k += 1
    if fit_pitch:
        corr = cov[0, k] / (sd[0] * sd[k]) if fit_cd else float("nan")
        res.update(pitch_c=round(c, 4), pitch_c_err=round(float(sd[k]), 4),
                   implied_pitch_deg_if_f_1400px=round(math.degrees(-c * F_PX_ASSUMED * BALL_D / 100.0), 2),
                   corr_Cd_pitch=round(float(corr), 3))
        k += 1
    res["S_px_per_m"] = {ds.name: round(s, 1) for ds, s in zip(sets, S)}
    res["S_over_S_ball"] = {ds.name: round(s / ds.S_ball, 4) for ds, s in zip(sets, S)}
    if not pin_scale and fit_cd:
        res["corr_Cd_S"] = {ds.name: round(float(cov[0, k + v] / (sd[0] * sd[k + v])), 3) for v, ds in enumerate(sets)}
    flights = per_flight(sol.x)
    vmax = max(float(np.abs(velocity(ds.tr.t[f.idx] / K - ta, vt)).max()) for ds, f, _, ta, vt in flights)
    res["max_speed_m_s"] = round(vmax, 2)
    res["_residuals"] = [(ds.name, ds.tr.yu[f.idx], r) for ds, f, r, _, _ in flights]
    return res


# --------------------------------------------------------------------------- angled drops

F_PX_ASSUMED = 1400.0  # focal length [px] assumed for depth velocity (iPhone main, 1080p); +-10%
TILT_SYST_DEG = 2.0  # blade-tilt systematic: tape as roll reference, out-of-image-plane tilt


def analyze_angle(name: str, e_of_v, tilt_deg: float, tilt_err_deg: float):
    """Angled-paddle drops. Positions are converted to camera-frame metres with the
    per-frame apparent ball size (smoothed per flight), because after the bounce the ball
    moves strongly towards the camera (width grows ~25%), so a single px/m scale is wrong:
        X = (x - cx) D / w,  Y = (y_up - cy_up) D / w,  Z = f D / w   (f assumed)
    X is fitted linear, Y quadratic, Z linear in real time per flight; the gravity
    curvature of Y gives a scale correction kappa = g / |Y''| (absorbs the ball-width
    threshold bias) applied to X and Y velocities."""
    tr = load(name)
    dball = float(np.median(tr.w))
    mins = bottom_minima(tr, dball)
    t, x, y = tr.t, tr.x, tr.yu
    if not mins:
        return {"name": name, "error": "no impact found"}, None
    m = mins[0]
    idx_in = np.arange(0, m + 1)
    sel_in = grow(t, y, idx_in, drop_seed(y[idx_in], y[m]))
    idx_out = np.arange(m, len(t))
    sel_out = grow(t, y, idx_out, np.arange(1, min(9, len(idx_out))))

    def metric_fit(sel):
        tt = t[sel] / K
        ws = np.polyval(np.polyfit(tt, tr.w[sel], 1), tt)
        X = (x[sel] - 540.0) * BALL_D / ws
        Y = (y[sel] - P0_CENTRE) * BALL_D / ws
        Z = F_PX_ASSUMED * BALL_D / ws
        return np.polyfit(tt, X, 1), np.polyfit(tt, Y, 2), np.polyfit(tt, Z, 1)

    cx_in, cy_in, cz_in = metric_fit(sel_in)
    cx_out, cy_out, cz_out = metric_fit(sel_out)
    kap_in = (G / 2) / -cy_in[0]
    kap_out = (G / 2) / -cy_out[0]
    kap = float(np.mean([kap_in, kap_out]))
    roots = np.roots(cy_in - cy_out)
    roots = roots[np.isreal(roots)].real
    tm = t[m] / K
    ti = float(roots[np.argmin(np.abs(roots - tm))]) if len(roots) else tm

    def vel(cx, cy, cz):
        return np.array([kap * np.polyval(np.polyder(cx), ti), kap * np.polyval(np.polyder(cy), ti), cz[0]])

    v3_in, v3_out = vel(cx_in, cy_in, cz_in), vel(cx_out, cy_out, cz_out)
    res_3d = solve_tilt(v3_in, v3_out, e_of_v)
    res_2d = solve_tilt(np.r_[v3_in[:2], 0.0], np.r_[v3_out[:2], 0.0], e_of_v)
    res_img = measured_tilt_decomposition(v3_in, v3_out, tilt_deg, tilt_err_deg, e_of_v)
    out = dict(
        name=name,
        impact_t_video=round(ti * K, 4),
        n_points_in=int(len(sel_in)), n_points_out=int(len(sel_out)),
        kappa_gravity_scale_correction=dict(incoming=round(kap_in, 3), outgoing=round(kap_out, 3)),
        ball_width_px_in_out=[round(float(np.median(tr.w[sel_in])), 1), round(float(np.median(tr.w[sel_out])), 1)],
        v_in_xyz_m_s=[round(float(v), 3) for v in v3_in],
        v_out_xyz_m_s=[round(float(v), 3) for v in v3_out],
        axes="x = image right, y = up, z = away from camera (z from apparent-size change, f assumed 1400 px)",
        C_tilt_measured_from_image=res_img,
        trajectory_only_with_depth=res_3d,
        trajectory_only_image_plane=res_2d,
    )
    return out, dict(tr=tr, sel_in=sel_in, sel_out=sel_out)


def measured_tilt_decomposition(v_in, v_out, tilt_deg: float, tilt_err_deg: float, e_of_v) -> dict:
    """Closure C: blade tilt measured in the image (rotation of the blade's major axis
    relative to the level blade, corrected for camera roll with the tape measure); tilt
    axis assumed along the viewing direction, so n = (sin th, cos th, 0)."""

    def at(th_deg):
        th = math.radians(th_deg)
        n = np.array([math.sin(th), math.cos(th), 0.0])
        vni, vno = float(v_in @ n), float(v_out @ n)
        ti, to = v_in - vni * n, v_out - vno * n
        t_hat = ti / np.linalg.norm(ti)
        return dict(tilt_deg=round(th_deg, 2), v_n_in=round(-vni, 3), v_n_out=round(vno, 3),
                    e_normal=round(-vno / vni, 3), v_t_in=round(float(ti @ t_hat), 3),
                    v_t_out=round(float(to @ t_hat), 3), tangential_ratio=round(float(to @ t_hat) / float(ti @ t_hat), 3),
                    out_of_plane_tangential_m_s=round(float(np.linalg.norm(to - (to @ t_hat) * t_hat)), 3))

    c = at(tilt_deg)
    lo, hi = at(tilt_deg - tilt_err_deg), at(tilt_deg + tilt_err_deg)
    c.update(
        tilt_err_deg=round(tilt_err_deg, 2),
        e_normal_range=sorted([lo["e_normal"], hi["e_normal"]]),
        tangential_ratio_range=sorted([lo["tangential_ratio"], hi["tangential_ratio"]]),
        e_center_fit_at_this_v_n=round(float(e_of_v(c["v_n_in"])), 3),
        rolling_without_slip_reference=0.6,
    )
    return c


def solve_tilt(v_in: np.ndarray, v_out: np.ndarray, e_of_v) -> dict:
    """Blade normal n is assumed to lie in the plane of v_in and v_out (no initial spin, so
    friction keeps the tangential velocity direction). With n at angle th from -v_in:
        e(th) = -(v_out.n)/(v_in.n),  r(th) = (v_out.t)/(v_in.t)
    are not separately identifiable from one bounce. Two closures:
      A) e(th) = e_center(|v_n,in|) from the centre drops -> tilt, tangential ratio r
      B) r = 1 (no tangential loss)                     -> tilt, implied e
    """
    b1 = -v_in / np.linalg.norm(v_in)
    b2 = v_out - (v_out @ b1) * b1
    b2 /= np.linalg.norm(b2)

    def comps(th):
        n = math.cos(th) * b1 + math.sin(th) * b2
        tv = -math.sin(th) * b1 + math.cos(th) * b2
        return n, float(v_in @ n), float(v_out @ n), float(v_in @ tv), float(v_out @ tv)

    def e_of(th):
        _, vni, vno, _, _ = comps(th)
        return -vno / vni

    def r_of(th):
        _, _, _, vti, vto = comps(th)
        return vto / vti

    def pack(th, closure):
        n, vni, vno, vti, vto = comps(th)
        tilt = math.degrees(math.acos(min(1.0, abs(n[1]))))
        return dict(closure=closure, tilt_from_horizontal_deg=round(tilt, 1),
                    normal_xyz=[round(float(v), 3) for v in n],
                    v_n_in=round(-vni, 3), v_n_out=round(vno, 3), e_normal=round(-vno / vni, 3),
                    v_t_in=round(-vti, 3), v_t_out=round(-vto, 3), tangential_ratio=round(vto / vti, 3))

    out = {}
    ths = np.linspace(0.01, 1.2, 400)
    fa = np.array([e_of(th) - float(e_of_v(abs(comps(th)[1]))) for th in ths])
    fb = np.array([r_of(th) - 1.0 for th in ths])
    for key, fv, func, label in (
        ("A_e_from_center_fit", fa, lambda th: e_of(th) - float(e_of_v(abs(comps(th)[1]))), "e_n = e_center(v_n)"),
        ("B_no_tangential_loss", fb, lambda th: r_of(th) - 1.0, "tangential ratio = 1"),
    ):
        s = np.where(np.sign(fv[:-1]) != np.sign(fv[1:]))[0]
        if len(s) == 0:
            out[key] = dict(closure=label, solution=None)
            continue
        th = brentq(func, ths[s[0]], ths[s[0] + 1])
        out[key] = pack(th, label)
    return out


# --------------------------------------------------------------------------- drag bias of plain parabolas


def parabola_bias(cd: float, v_in: float = 3.5, e: float = 0.55, fs: float = 240.0) -> dict:
    """Synthetic bounce with drag, sampled at the 240 fps capture rate, analysed with one
    free parabola per flight (the simple method) -> bias in v_in, v_out, e."""
    vt = vterm(cd)
    h = -(vt**2) / (2 * G) * math.log(1 - (v_in / vt) ** 2)
    t1 = vt / G * math.acosh(math.exp(G * h / vt**2))
    vo = e * v_in
    tup = vt / G * math.atan(vo / vt)
    H2 = vt**2 / (2 * G) * math.log(1 + (vo / vt) ** 2)
    tdn = vt / G * math.acosh(math.exp(G * H2 / vt**2))
    ts = np.arange(0.0, t1 + tup + tdn, 1 / fs) + 0.37 / fs  # arbitrary phase
    tin = ts[ts < t1]
    tout = ts[(ts > t1) & (ts < t1 + tup + tdn)]
    yin = h + height(tin, vt)
    yout = H2 + height(tout - (t1 + tup), vt)
    ci = np.polyfit(tin, yin, 2)
    co = np.polyfit(tout, yout, 2)
    r = np.roots(ci - co)
    r = r[np.isreal(r)].real
    ti = float(r[np.argmin(np.abs(r - t1))])
    vin_f = -np.polyval(np.polyder(ci), ti)
    vout_f = np.polyval(np.polyder(co), ti)
    return dict(cd=cd, v_in_true=v_in, e_true=e, drop_height_m=round(h, 4),
                v_in_parabola=round(float(vin_f), 4), v_out_parabola=round(float(vout_f), 4),
                e_parabola=round(float(vout_f / vin_f), 4),
                v_in_bias_pct=round(100 * (vin_f / v_in - 1), 2),
                v_out_bias_pct=round(100 * (vout_f / vo - 1), 2),
                e_bias=round(float(vout_f / vin_f - e), 4))


# --------------------------------------------------------------------------- summary statistics


def ols(x: np.ndarray, y: np.ndarray, deg: int) -> dict:
    A = np.vander(x, deg + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    dof = max(len(y) - deg - 1, 1)
    s2 = float(res @ res) / dof
    cov = np.linalg.inv(A.T @ A) * s2
    return dict(coef=coef, err=np.sqrt(np.diag(cov)), resid_std=math.sqrt(s2), cov=cov)


def saturating_fit(v: np.ndarray, e: np.ndarray) -> dict:
    """e(v) = e_inf + A exp(-v / v0), least squares."""
    def f(p):
        return p[0] + p[1] * np.exp(-v / p[2]) - e

    sol = least_squares(f, [0.56, 0.35, 0.7], bounds=([0, 0, 0.05], [1, 2, 10]))
    dof = max(len(e) - 3, 1)
    s2 = float(sol.fun @ sol.fun) / dof
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * s2
    return dict(p=sol.x, err=np.sqrt(np.diag(cov)), resid_std=math.sqrt(s2))



def reason_category(r: str) -> str:
    for key, cat in (("too few points", "neighbour flight has < 8 points"),
                     ("too low", "neighbour flight apex < 3 mm (micro-bounce/rolling)"),
                     ("not free flight", "neighbour flight not described by free fall (rolling/mis-tracked)"),
                     ("slow-mo ramp", "flight timescale inconsistent (slow-mo ramp)"),
                     ("off the paddle", "impact far from paddle contact level (mis-segmentation)"),
                     ("intersect", "flight curves do not intersect"),
                     ("usable flights", "video unusable")):
        if key in r:
            return cat
    return r


# --------------------------------------------------------------------------- plots

HEIGHT_COLORS = {4: "tab:purple", 8: "tab:blue", 12: "tab:cyan", 16: "tab:green", 19: "tab:orange", 25.25: "tab:red"}


def nominal_in(name: str) -> float:
    m = NOMINAL_RE.match(name)
    return float(m.group(1).replace("_", ".")) if m else float("nan")


def plot_restitution(imps, lin, sat, const, bins, path):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axs[0]
    for h, col in HEIGHT_COLORS.items():
        sel = [r for r in imps if nominal_in(r["video"]) == h]
        if not sel:
            continue
        ax.errorbar([r["v_in"] for r in sel], [r["e"] for r in sel], yerr=[r["e_err"] for r in sel],
                    fmt="o", ms=4, color=col, alpha=0.75, label=f'{h:g} in drops', mfc="none" if h == 4 else col)
    vv = np.linspace(0, 3.6, 50)
    ax.plot(vv, lin["coef"][0] + lin["coef"][1] * vv, "k-",
            label=f'linear e = {lin["coef"][0]:.3f} {lin["coef"][1]:+.3f} v  (σ={lin["resid_std"]:.3f})')
    ax.axhline(const["mean"], color="k", ls="--", label=f'constant e = {const["mean"]:.3f} (σ={const["std"]:.3f})')
    es, A, v0 = sat["p"]
    vs = np.linspace(0.15, 3.6, 80)
    ax.plot(vs, es + A * np.exp(-vs / v0), "m-",
            label=f'e = {es:.3f} + {A:.3f} exp(-v/{v0:.2f})  (σ={sat["resid_std"]:.3f})')
    bc = [b for b in bins if b["n"] > 0]
    ax.errorbar([b["v_mean"] for b in bc], [b["e_mean"] for b in bc], yerr=[b["e_std"] for b in bc],
                fmt="s", color="k", ms=7, capsize=4, mfc="w", label="binned mean ± std")
    ax.set_xlabel("impact speed v_in [m/s]")
    ax.set_ylabel("coefficient of restitution e = v_out / v_in")
    ax.set_xlim(0, 3.7)
    ax.set_ylim(0.45, 0.95)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Ball on paddle centre (all rebounds)")
    ax = axs[1]
    v = np.array([r["v_in"] for r in imps])
    vo = np.array([r["v_out"] for r in imps])
    ax.plot(v, vo, "o", ms=4, alpha=0.7)
    ax.plot(vv, vv * (lin["coef"][0] + lin["coef"][1] * vv), "k-", label="linear e(v)")
    ax.plot(vs, vs * (es + A * np.exp(-vs / v0)), "m-", label="saturating e(v)")
    ax.plot(vv, const["mean"] * vv, "k--", label="constant e")
    ax.set_xlabel("v_in [m/s]")
    ax.set_ylabel("v_out [m/s]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Rebound speed")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_paddle_fits(diags: dict, path):
    names = sorted(diags, key=lambda n: (nominal_in(n), n))
    nc = 5
    nr = math.ceil(len(names) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 2.8 * nr))
    for ax, n in zip(axs.flat, names):
        d = diags[n]
        tr, vf, P0 = d["tr"], d["vf"], d["P0"]
        vt = vterm(d["cd"])
        ax.plot(tr.t, tr.yu - P0, ".", color="0.7", ms=2)
        good = [f for f in d["flights"] if f.ok]
        for (ta, ya), f in zip(vf.flights, good):
            ax.plot(tr.t[f.idx], tr.yu[f.idx] - P0, ".", ms=3)
            tt = np.linspace(tr.t[f.idx[0]] - 0.1, tr.t[f.idx[-1]] + 0.1, 200)
            ax.plot(tt, project(ya + height(tt / K - ta, vt), vf.S, vf.gamma, P0) - P0, "k-", lw=0.7)
        for r in d["impacts"]:
            ax.plot(r["t_video"], vf.S * r["impact_row_offset_mm"] / 1000, "rx")
        ax.set_title(f"{n}  rms {vf.rms:.2f}px", fontsize=9)
        ax.set_ylim(-30, None)
        ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[len(names):]:
        ax.axis("off")
    fig.supxlabel("video time [s]  (real = video / 8)")
    fig.supylabel("height above paddle contact row [px]")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_drag(v6: dict, cd_rows: list, CD: float, CD_err: float, path):
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, key in zip(axs[:2], ("joint_no_pitch", "joint_pitch_free")):
        seen: dict[str, str] = {}
        for name, yu, r in v6[key]["_residuals"]:
            col = seen.setdefault(name, f"C{len(seen)}")
            ax.plot(-yu, r, ".-", ms=3, lw=0.6, color=col, label=name if ax.get_legend_handles_labels()[1].count(name) == 0 else None)
        ax.set_title(f'6 ft drops, {key}: Cd={v6[key]["Cd"]:.2f}, rms={v6[key]["rms_px"]:.2f}px', fontsize=10)
        ax.set_xlabel("image row y [px] (down)")
        ax.set_ylabel("residual [px]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax = axs[2]
    xs = np.arange(len(cd_rows))
    ax.errorbar(xs, [r["Cd"] for r in cd_rows], yerr=[r["Cd_err"] for r in cd_rows], fmt="o", capsize=3,
                label="paddle drop videos (16-25 in), per video")
    ax.axhspan(CD - CD_err, CD + CD_err, color="tab:blue", alpha=0.2, label=f"adopted Cd = {CD:.2f} ± {CD_err:.2f}")
    for key, ls in (("joint_no_pitch", ":"), ("joint_pitch_free", "--"), ("joint_scale_pinned_to_ball_width_pitch_free", "-.")):
        ax.axhline(v6[key]["Cd"], color="k", ls=ls, lw=1, label=f"6 ft {key}: {v6[key]['Cd']:.2f}")
    ax.set_xticks(xs, [r["video"].replace("c-", "") for r in cd_rows], rotation=70, fontsize=7)
    ax.set_ylabel("Cd")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=7)
    ax.set_title("Drag coefficient estimates")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_timescale(videos: dict, path):
    fig, ax = plt.subplots(figsize=(12, 4))
    names = sorted(videos, key=lambda n: (nominal_in(n), n))
    for i, n in enumerate(names):
        v = videos[n]
        if "flights" not in v:
            continue
        ks = [f["K_implied_by_ball_diameter"] for f in v["flights"] if f["K_implied_by_ball_diameter"] is not None]
        ax.plot([i] * len(ks), ks, "o", ms=4, color="tab:blue")
        ax.plot(i, K * math.sqrt(v["S_ball_px_per_m"] / v["S_v_px_per_m"]), "k_", ms=12)
    ax.axhline(K, color="r", lw=1, label="K = 8 (240/30)")
    ax.set_xticks(range(len(names)), [n.replace("c-", "") for n in names], rotation=70, fontsize=7)
    ax.set_ylabel("slow-down factor implied by flight curvature\n(scale from ball width)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Per-flight timescale check (dots: single flights, bar: whole video joint fit)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_angled(angle_diags: dict, angle_out: dict, path):
    fig, axs = plt.subplots(1, len(angle_diags), figsize=(5.5 * len(angle_diags), 8.5), squeeze=False)
    for ax, (n, d) in zip(axs[0], angle_diags.items()):
        tr = d["tr"]
        ax.plot(tr.x, -tr.yu, ".", color="0.7", ms=3)
        ax.plot(tr.x[d["sel_in"]], -tr.yu[d["sel_in"]], ".", label="incoming fit window")
        ax.plot(tr.x[d["sel_out"]], -tr.yu[d["sel_out"]], ".", label="outgoing fit window")
        o = angle_out[n]
        ax.set_xlim(0, 1080)
        ax.set_ylim(1920, 0)
        ax.set_aspect("equal")
        c = o["C_tilt_measured_from_image"]
        txt = (f'v_in xyz {o["v_in_xyz_m_s"]}\nv_out xyz {o["v_out_xyz_m_s"]}\n'
               f'image tilt {c["tilt_deg"]}±{c["tilt_err_deg"]}°: e_n {c["e_normal"]} {c["e_normal_range"]}, '
               f'r_t {c["tangential_ratio"]} {c["tangential_ratio_range"]}')
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=7, va="top")
        ax.set_title(n)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px] (down)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def r3(x, n=4):
    return None if x is None else round(float(x), n)


def main() -> None:
    names = sorted(p.stem for p in TRACKS.glob("*.csv") if not p.stem.endswith("_scene"))
    six = [n for n in names if n.startswith("6ft")]
    paddle = [n for n in names if n.startswith("c-")]
    angled = [n for n in names if n.startswith("angle")]

    # ---- 1. drag from the 6 ft drops
    sets = [prep_6ft(n) for n in six]
    v6 = {
        "joint_no_pitch": fit_6ft(sets, fit_pitch=False, pin_scale=False),
        "joint_pitch_free": fit_6ft(sets, fit_pitch=True, pin_scale=False),
        "joint_scale_pinned_to_ball_width_pitch_free": fit_6ft(sets, fit_pitch=True, pin_scale=True),
        "joint_pitch_free_no_depth_correction": fit_6ft(sets, fit_pitch=True, pin_scale=False, depth_corr=False),
    }
    per_video_6 = {ds.name: fit_6ft([ds], fit_pitch=False, pin_scale=False) for ds in sets}

    # ---- 2. drag from the high paddle drops (whole fall from the hand is in frame)
    cd_rows, cd_rows_g0 = [], []
    for n in paddle:
        if (nominal_height_m(n) or 0) < 0.4:
            continue
        out, extra = analyze_paddle(n, 0.45, fit_gamma=True)
        if extra is None:
            continue
        d = extra[2]
        for fg, rows in ((True, cd_rows), (False, cd_rows_g0)):
            vf = fit_video(d["tr"], d["flights"], d["P0"], d["vf"].S, 0.45, fit_gamma=fg, fit_cd=True)
            sd = np.sqrt(np.clip(np.diag(vf.cov), 0, None))
            rows.append(dict(video=n, Cd=round(vf.cd, 3), Cd_err=round(float(sd[vf.names.index("Cd")]), 3),
                             gamma_per_m=round(vf.gamma, 4), rms_px=round(vf.rms, 2)))
    cds = np.array([r["Cd"] for r in cd_rows])
    CD = float(np.mean(cds))
    CD_err = float(np.std(cds, ddof=1) / math.sqrt(len(cds)))
    cds0 = np.array([r["Cd"] for r in cd_rows_g0])

    # ---- 3. paddle impacts
    videos, diags, impacts, rejected = {}, {}, [], []
    for n in paddle:
        out, extra = analyze_paddle(n, CD)
        videos[n] = out
        if extra is None:
            rejected.append(dict(video=n, t_video=None, reason=out["error"]))
            continue
        imps, rej, d = extra
        impacts += imps
        rejected += rej
        diags[n] = d
    # same segmentation without drag -> drag-correction size on the measured impacts
    nodrag = {}
    for n in diags:
        _, ex0 = analyze_paddle(n, 0.0)
        if ex0 is not None:
            for r in ex0[0]:
                nodrag[(n, round(r["t_video"], 1))] = r
    dv, de, vv_ = [], [], []
    for r in impacts:
        r0 = nodrag.get((r["video"], round(r["t_video"], 1)))
        if r0:
            dv.append(r["v_in"] / r0["v_in"] - 1)
            de.append(r["e"] - r0["e"])
            vv_.append(r["v_in"])
    dv, de, vv_ = map(np.array, (dv, de, vv_))
    hi = vv_ > 2.5

    # ---- 4. e(v) summary
    v = np.array([r["v_in"] for r in impacts])
    e = np.array([r["e"] for r in impacts])
    lin = ols(v, e, 1)
    sat = saturating_fit(v, e)
    const = dict(mean=float(e.mean()), std=float(e.std(ddof=1)), sem=float(e.std(ddof=1) / math.sqrt(len(e))))
    edges = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.6]
    bins = []
    for lo, hi_ in zip(edges[:-1], edges[1:]):
        s = (v >= lo) & (v < hi_)
        bins.append(dict(v_range=[lo, hi_], n=int(s.sum()),
                         v_mean=r3(v[s].mean(), 3) if s.any() else None,
                         e_mean=r3(e[s].mean()) if s.any() else None,
                         e_std=r3(e[s].std(ddof=1)) if s.sum() > 1 else None,
                         e_sem=r3(e[s].std(ddof=1) / math.sqrt(s.sum())) if s.sum() > 1 else None))
    per_height = {}
    for h in sorted({nominal_in(n) for n in paddle}):
        sel = [r for r in impacts if nominal_in(r["video"]) == h and r["first"]]
        allh = [r for r in impacts if nominal_in(r["video"]) == h]
        hn = h * INCH
        per_height[f"{h:g}in"] = dict(
            n_impacts=len(allh), n_first_impacts=len(sel),
            first_impact_v_in_mean=r3(np.mean([r["v_in"] for r in sel]), 3) if sel else None,
            first_impact_v_in_std=r3(np.std([r["v_in"] for r in sel], ddof=1), 3) if len(sel) > 1 else None,
            v_free_fall_from_nominal_height=round(math.sqrt(2 * G * hn), 3),
            first_impact_e_mean=r3(np.mean([r["e"] for r in sel])) if sel else None,
            first_impact_e_std=r3(np.std([r["e"] for r in sel], ddof=1)) if len(sel) > 1 else None,
        )
    vx = np.array([[r["vx_in"], r["vx_out"]] for r in impacts])
    lateral = dict(
        median_abs_vx_m_s=r3(np.median(np.abs(vx)), 3),
        p95_abs_vx_m_s=r3(np.percentile(np.abs(vx), 95), 3),
        max_abs_vx_m_s=r3(np.abs(vx).max(), 3),
        median_abs_vx_over_v_in=r3(np.median(np.abs(vx[:, 0]) / v), 3),
        median_abs_change_vx_at_impact_m_s=r3(np.median(np.abs(vx[:, 1] - vx[:, 0])), 3),
        note="x velocity from a linear fit per flight; image x only (depth motion not measured for centre drops)",
    )

    # ---- 5. timescale checks
    kf = [(n, f["K_implied_by_ball_diameter"], f.get("timescale_ok", True))
          for n, vo in videos.items() if "flights" in vo for f in vo["flights"]
          if f["K_implied_by_ball_diameter"] is not None]
    kvals = np.array([k for _, k, _ in kf])
    ratio = np.array([vo["S_v_over_S_ball"] for vo in videos.values() if "S_v_over_S_ball" in vo])
    # 6 ft drops with the scale pinned to the ball diameter: as measured, and x the blob-size
    # correction S_v/S_ball from the centre videos (the 6 ft framing differs from the paddle
    # videos, so S_v itself cannot be transferred; the ball-width threshold bias can)
    pin_corr = float(np.median([vo["S_v_over_S_ball"] for n, vo in videos.items()
                                if "S_v_over_S_ball" in vo and not n.startswith("c-4in")]))
    v6["joint_scale_pinned_to_ball_width_no_pitch"] = fit_6ft(sets, fit_pitch=False, pin_scale=True)
    v6["joint_scale_pinned_corrected_no_pitch"] = fit_6ft(sets, fit_pitch=False, pin_scale=True, pin_factor=pin_corr)
    per_video_6_pinned = {ds.name: fit_6ft([ds], fit_pitch=False, pin_scale=True) for ds in sets}
    cd6_pv = np.array([v["Cd"] for v in per_video_6_pinned.values()])

    # ---- 6. angled drops
    e_of_v = lambda vn: float(sat["p"][0] + sat["p"][1] * np.exp(-vn / sat["p"][2]))  # noqa: E731
    # blade tilt from the image: blade major-axis angle minus tape angle (camera roll),
    # relative to the same quantity for the level blade in the centre-drop videos
    rel = lambda s: s["blade_axis_cw_deg"] - s["tape_cw_deg"]  # noqa: E731
    level = np.array([rel(load_scene(n)) for n in paddle])
    level_ref, level_std = float(np.median(level)), float(np.std(level, ddof=1))
    angle_out, angle_diags, tilt_info = {}, {}, {}
    for n in angled:
        sc = load_scene(n)
        tilt = rel(sc) - level_ref
        tilt_err = math.hypot(level_std, TILT_SYST_DEG)
        tilt_info[n] = dict(blade_axis_cw_deg=r3(sc["blade_axis_cw_deg"], 2), tape_cw_deg=r3(sc["tape_cw_deg"], 2),
                            blade_aspect=r3(sc["blade_minor_px"] / sc["blade_major_px"], 3),
                            tilt_deg=r3(tilt, 2), tilt_err_deg=r3(tilt_err, 2))
        o, dg = analyze_angle(n, e_of_v, tilt, tilt_err)
        angle_out[n] = o
        if dg is not None:
            angle_diags[n] = dg

    # ---- 7. outputs
    rej_counts: dict[str, int] = {}
    for r in rejected:
        c = reason_category(r["reason"])
        rej_counts[c] = rej_counts.get(c, 0) + 1

    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    results = dict(
        meta=dict(
            generated_by="measurements/ball/analyze.py",
            slowdown_K=K, g=G, ball=dict(mass_kg=BALL_M, diameter_m=BALL_D), air_density=RHO,
            velocities="real-time m/s; v_in, v_out are speeds (positive); e = v_out / v_in",
        ),
        timescale_check=dict(
            method="each flight with >= 12 points and > 1 cm height fitted alone with a free scale S_f at K = 8; "
                   "K_implied = 8 * sqrt(S_ball / S_f) where S_ball = ball width / 40 mm",
            n_flights=int(len(kvals)),
            K_implied_median=r3(np.median(kvals), 3), K_implied_std=r3(np.std(kvals), 3),
            K_implied_min=r3(kvals.min(), 3), K_implied_max=r3(kvals.max(), 3),
            flights_flagged_inconsistent=[dict(video=n, K=k) for n, k, ok in kf if not ok],
            S_v_over_S_ball_median=r3(np.median(ratio)), S_v_over_S_ball_range=[r3(ratio.min()), r3(ratio.max())],
            conclusion="no slow-mo ramps inside the analysed flights: all flight curvatures agree within a few %; "
                       "K = 8 adopted. The gravity scale S_v is ~3% larger than the ball-width scale "
                       "(thresholded blob is slightly smaller than the ball), which is why K_implied clusters at "
                       f"{np.median(kvals):.2f} instead of 8.0.",
        ),
        drag=dict(
            adopted_Cd=round(CD, 3), adopted_Cd_err=round(CD_err, 3),
            adopted_from="mean of per-video fits of the 16-25 in paddle videos (gravity + quadratic drag, "
                         "free scale and perspective term); error = std/sqrt(n)",
            terminal_velocity_m_s=round(vterm(CD), 2),
            paddle_drops=dict(
                n_videos=len(cd_rows), Cd_mean=round(CD, 3), Cd_std=round(float(np.std(cds, ddof=1)), 3),
                Cd_mean_with_perspective_fixed_0=round(float(np.mean(cds0)), 3),
                Cd_std_with_perspective_fixed_0=round(float(np.std(cds0, ddof=1)), 3),
                per_video=cd_rows,
            ),
            six_ft_drops=dict(
                best_estimate=dict(
                    Cd=round(float(cd6_pv.mean()), 3),
                    Cd_err=round(float(cd6_pv.std(ddof=1) / math.sqrt(len(cd6_pv))), 3),
                    Cd_per_video=[round(float(c), 3) for c in cd6_pv],
                    method="K = 8, S_v pinned to the ball-diameter scale (ball width / 40 mm) per video, "
                           "no perspective term, per-video fits; error = std/sqrt(3)",
                ),
                blob_size_correction_factor=round(pin_corr, 4),
                variants={k: strip(vv) for k, vv in v6.items()},
                per_video_no_pitch={k: strip(vv) for k, vv in per_video_6.items()},
                per_video_scale_pinned={k: strip(vv) for k, vv in per_video_6_pinned.items()},
                literature_range=[0.40, 0.50],
                conclusion="DEGENERATE / not well constrained. The fast falling ball (~4.4 m/s, ~50 px motion "
                           "blur) has structured +-5 px centroid residuals; Cd trades off against the scale and "
                           "the camera-pitch (perspective) term: "
                           f"{v6['joint_no_pitch']['Cd']:.2f} (free scale, no perspective), "
                           f"{v6['joint_pitch_free']['Cd']:.2f} (free scale, perspective free), "
                           f"{v6['joint_scale_pinned_to_ball_width_no_pitch']['Cd']:.2f} / "
                           f"{v6['joint_scale_pinned_to_ball_width_pitch_free']['Cd']:.2f} (scale pinned to ball "
                           "diameter, without / with perspective), "
                           f"{v6['joint_scale_pinned_corrected_no_pitch']['Cd']:.2f} with the pinned scale x the "
                           f"blob-size correction {pin_corr:.3f} from the centre videos (fit rms worsens to "
                           f"{v6['joint_scale_pinned_corrected_no_pitch']['rms_px']:.2f} px). The best-constrained "
                           "drag value is the paddle-drop Cd (adopted); it lies in the literature range 0.40-0.50 "
                           "for a smooth sphere at Re ~ 1e4.",
            ),
            parabola_bias=dict(
                synthetic_3p5_m_s=parabola_bias(CD, 3.5, 0.55),
                synthetic_2p0_m_s=parabola_bias(CD, 2.0, 0.6),
                measured_drag_minus_nodrag=dict(
                    mean_rel_change_v_in_all=r3(dv.mean()), mean_change_e_all=r3(de.mean()),
                    mean_rel_change_v_in_above_2p5=r3(dv[hi].mean()) if hi.any() else None,
                    mean_change_e_above_2p5=r3(de[hi].mean()) if hi.any() else None,
                    note="same segmentation refitted with Cd = 0 (gravity-only, shared scale)",
                ),
            ),
        ),
        restitution=dict(
            n_impacts_used=len(impacts),
            n_impacts_rejected=len(rejected),
            rejected_by_reason=rej_counts,
            linear=dict(e0=r3(lin["coef"][0]), e0_err=r3(lin["err"][0]), e1_per_m_s=r3(lin["coef"][1]),
                        e1_err=r3(lin["err"][1]), resid_std=r3(lin["resid_std"]),
                        cov_e0_e1=r3(lin["cov"][0, 1], 6), form="e(v) = e0 + e1 * v_in"),
            saturating=dict(e_inf=r3(sat["p"][0]), A=r3(sat["p"][1]), v0_m_s=r3(sat["p"][2]),
                            errs=[r3(x) for x in sat["err"]], resid_std=r3(sat["resid_std"]),
                            form="e(v) = e_inf + A * exp(-v_in / v0)",
                            note="describes the flattening above ~2.5 m/s better than the linear fit; "
                                 "preferred for extrapolation to 3-4 m/s"),
            constant=dict(e=r3(const["mean"]), sem=r3(const["sem"]), std=r3(const["std"])),
            binned=bins,
            per_nominal_height=per_height,
            lateral_drift=lateral,
            note="formal per-impact errors (e_err) are much smaller than the scatter; the scatter (resid_std) "
                 "is the honest per-impact uncertainty (impact location on the blade, ball spin, paddle "
                 "vibration from the previous bounce).",
        ),
        angled_drops=dict(
            method="Velocities at impact from per-flight fits in camera-frame metres (apparent-size depth "
                   "correction). Primary result C: blade tilt measured in the image (blade major-axis rotation "
                   "minus tape-measure rotation, relative to the level blade of the centre videos; tilt axis "
                   "assumed along the viewing direction) -> normal/tangential components, e_n and tangential "
                   "ratio r_t = v_t,out / v_t,in, with ranges for tilt +- tilt_err. Trajectory-only closures "
                   "(blade normal in the plane of v_in, v_out): A assumes e_n = saturating centre fit -> tilt, "
                   "r_t; B assumes r_t = 1 -> tilt, e_n.",
            image_tilt=dict(level_reference_blade_minus_tape_deg=r3(level_ref, 2),
                            level_reference_std_deg=r3(level_std, 2), systematic_deg=TILT_SYST_DEG,
                            per_video=tilt_info),
            videos=angle_out,
            identifiability="From the trajectory alone one bounce gives 2 numbers in the bounce plane for 3 "
                            "unknowns (tilt, e_n, r_t); the image-measured tilt closes the system. r_t is very "
                            "sensitive to the tilt (see tangential_ratio_range). Ball spin is not observed. "
                            "The depth component comes from apparent-size change with an assumed focal length "
                            "(1400 px +-10%) and is small (|v_z| < 0.1 m/s), so the bounce is nearly in the "
                            "image plane. Only two videos, single bounce each.",
        ),
        videos=videos,
        impacts=impacts,
        rejected_impacts=rejected,
        caveats=[
            "Timebase: K = 8 assumed exact (240 fps capture on 30 fps timebase); per-frame PTS used (~15% of "
            "capture frames are missing). Flight curvatures agree with K = 8 to a few %.",
            "Scale: S_v from gravity per video at the paddle contact row, with a 1D perspective term when the "
            "flights span > 15 cm; absolute speeds carry ~1-2% scale uncertainty (S_v vs ball width differ 3-4%).",
            "e is a ratio of velocities at the same image location, so it is nearly independent of K and S_v; "
            "the drag correction raises e by ~0.007 on average and ~0.02 above 2.5 m/s "
            "(drag.parabola_bias.measured_drag_minus_nodrag).",
            "Impact point = intersection of incoming/outgoing flight curves; for fast impacts it lies up to "
            "~15 mm below the resting contact row (finite contact time, sponge/ball compression, blade/clamp "
            "compliance). Shown as impact_row_offset_mm.",
            "The paddle is held by a clamp at the handle; blade compliance and vibration after an impact can "
            "make e depend on the clamp, not only on ball+rubber. Low-speed rebounds (v < 0.6 m/s) happen while "
            "the blade may still ring from the previous impact and show the largest scatter.",
            "Impacts are near the blade centre but not exactly at one spot; lateral x velocities are small "
            "(see restitution.lateral_drift) but depth motion of centre drops is not measured.",
            "Spin is not measured; hand releases may impart small spin.",
            "Drag: Cd from the 6 ft drops alone is degenerate with camera perspective/scale; the adopted Cd comes "
            "from the 16-25 in paddle drops (max 3.4 m/s), whose fall from 0.64 m reproduces the drag-reduced "
            "impact speed (3.4 vs 3.55 m/s free fall).",
            "Release heights are tape-measure nominal; fitted fall heights of the ball centre are given in "
            "videos.*.drop_check.",
            "c-4in videos use a different (closer) camera position; flights are too low to fit the perspective "
            "term, so it is fixed at 0 there.",
        ],
    )
    with (HERE / "results.yaml").open("w") as fh:
        yaml.safe_dump(clean(results), fh, sort_keys=False, width=110, allow_unicode=True)

    plot_restitution(impacts, lin, sat, const, bins, HERE / "restitution.png")
    plot_paddle_fits(diags, HERE / "paddle_fits.png")
    plot_drag(v6, cd_rows, CD, CD_err, HERE / "drag.png")
    plot_timescale(videos, HERE / "timescale.png")
    if angle_diags:
        plot_angled(angle_diags, angle_out, HERE / "angled.png")

    print(f"impacts used {len(impacts)}, rejected {len(rejected)}: {rej_counts}")
    print(f"e(v) = {lin['coef'][0]:.3f} + {lin['coef'][1]:.3f} v  (resid {lin['resid_std']:.3f});  "
          f"constant {const['mean']:.3f} +- {const['std']:.3f}")
    print(f"Cd adopted {CD:.3f} +- {CD_err:.3f};  6ft variants: " +
          ", ".join(f"{k}={vv['Cd']:.2f}" for k, vv in v6.items()))
    for b in bins:
        print(b)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
