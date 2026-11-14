"""Ping pong ball: free 40 mm thin-shell sphere with air drag and the paddle-bounce contact fitted to drop videos.

Measurements (measurements/ball/results.yaml, 119 centre impacts on the clamped paddle, 240 fps):
- Mass: 4 balls = 11 g on a 1 g scale -> 2.75 +- 0.25 g each (ITTF: 2.7 g).
- Restitution on the blade centre: e(v) = 0.578 + 0.373 exp(-v / 0.54 m/s), scatter 0.037. Over the bounce
  task's impact speeds (1.5-3.5 m/s) that is 0.60 -> 0.57.
- Drag: Cd = 0.50 +- 0.03 from the 16-25 in falls (literature 0.40-0.50).
- Angled drops (16-17 deg blade tilt): normal e as above; 0.37-0.41 of the tangential speed kept (+-0.1).

- Black side (the hitting side since the 2026-11-12 reassembly; the drop videos above are of the red side):
  e = 0.68-0.70 at 1-2 m/s, from 21 bounces off the still blade seen by the wrist camera
  (sim/scripts/analyze_run.py on that day's runs).

Contact model: an explicit ball-blade contact pair (dynamic ball-paddle contacts excluded). MuJoCo's soft contact
is a linear spring-damper, so e is speed-independent, e = exp(-pi z / sqrt(1 - z^2)) for damping ratio z.
solref (0.01 s, 0.135) at a 1 ms step gives e = 0.69 +- 0.05 (scatter from the contact phase within a step) at
1-3.5 m/s (sim/scripts/contact_sweep.py): the black side. The red side's 0.58 +- 0.035 is damping ratio 0.19; slow
dribbles (< 1 m/s, red e 0.65-0.75) bounce lower than real. Tangential: with the friction constraint as soft as
the normal one the ball keeps 0.72 of its tangential speed; a stiff friction solref (0.002 s) makes it leave an
oblique hit rolling, keeping 1 / (1 + 2/3) = 0.6, the most loss this friction model can give (MuJoCo friction
stores no elastic energy). The measured 0.4 (+-0.1, red side) is below that. Validation:
sim/scripts/validate_ball.py (replays the red-side videos with RED_SOLREF).

Air: MuJoCo's ellipsoid fluid model. For a sphere f = -rho C_blunt pi r^2 |v| v, so C_blunt = Cd / 2. The model also
applies the Magnus lift from spin. On MuJoCo Warp 3.7 it requires the Euler integrator.
"""

from __future__ import annotations

import math

import mujoco

BALL_RADIUS = 0.020
BALL_MASS = 0.00275
# Thin spherical shell: I = 2/3 m r^2.
BALL_INERTIA = 2.0 / 3.0 * BALL_MASS * BALL_RADIUS**2

BALL_CD = 0.50
BALL_SOLREF = (0.01, 0.135)  # (timeconst [s], damping ratio): the black rubber; needs SIM_DT <= 1 ms
RED_SOLREF = (0.01, 0.19)  # the red rubber (the filmed drops, scripts/validate_ball.py)
BALL_SOLREF_FRICTION = (0.002, 1.0)  # stiff: sliding -> rolling within the contact
BALL_FRICTION = 1.0  # sliding; inverted rubber is grippy (the rolling limit is reached for mu > ~0.1), randomize

AIR_DENSITY = 1.204  # kg/m^3, 20 C
AIR_VISCOSITY = 1.8e-5  # Pa s

# Flight model used by controllers and the ball filter: a = g - DRAG_K |v| v.
G = 9.81
DRAG_K = AIR_DENSITY * BALL_CD * math.pi * BALL_RADIUS**2 / (2 * BALL_MASS)  # 1/m

SIM_DT = 0.001


def get_ball_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="ball_joint")
    body.explicitinertial = True
    body.mass = BALL_MASS
    body.inertia = (BALL_INERTIA,) * 3
    geom = body.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(BALL_RADIUS, 0, 0),
        mass=0.0,
        condim=3,
        priority=1,
        solref=BALL_SOLREF,
        friction=(BALL_FRICTION, 0.005, 0.0001),
        rgba=(1.0, 0.45, 0.05, 1.0),
    )
    # Fluid coefficients: blunt drag, slender drag (unused for a sphere), angular drag, Kutta lift, Magnus lift.
    geom.fluid_ellipsoid = 1
    geom.fluid_coefs = (BALL_CD / 2, 0.25, 1.5, 1.0, 1.0)
    return spec


def add_air(spec: mujoco.MjSpec) -> None:
    """Scene spec callback: air for the ball's fluid model (MjSpec.attach doesn't carry entity <option>s)."""
    spec.option.density = AIR_DENSITY
    spec.option.viscosity = AIR_VISCOSITY


def add_paddle_contact(spec: mujoco.MjSpec, blade_geom: str, blade_body: str, ball_prefix: str = "ball/",
                       solref: tuple[float, float] = BALL_SOLREF) -> None:
    """Scene spec callback: the fitted ball-blade contact as an explicit pair, replacing the dynamic contact.

    Names are scene-level (entity prefix included), e.g. blade_geom="robot/blade_collision",
    blade_body="robot/paddle".
    """
    spec.add_exclude(bodyname1=f"{ball_prefix}ball", bodyname2=blade_body)
    spec.add_pair(
        name="ball_blade",
        geomname1=f"{ball_prefix}ball_geom",
        geomname2=blade_geom,
        condim=3,
        solref=solref,
        solreffriction=BALL_SOLREF_FRICTION,
        friction=(BALL_FRICTION, BALL_FRICTION, 0.005, 0.0001, 0.0001),
    )


def scene_spec_fn(spec: mujoco.MjSpec) -> None:
    """Scene spec callback for scenes with the robot entity "robot" and the ball entity "ball"."""
    add_air(spec)
    add_paddle_contact(spec, "robot/blade_collision", "robot/paddle")


def get_ball_cfg():
    from mjlab.entity import EntityCfg

    return EntityCfg(spec_fn=get_ball_spec)
