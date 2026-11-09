# pingpong-sim

SO-101 + ping pong paddle + ball in mjlab (MuJoCo Warp). Python 3.12 (the BAM servo library supports only 3.12).

```bash
cd sim
uv sync
MUJOCO_GL=egl uv run scripts/inspect_model.py   # masses, ready pose, wrist-roll sweep, renders -> build/
uv run scripts/servo_check.py                   # GPU: hold ready pose, 0.3 rad step per joint
uv run scripts/validate_ball.py                 # GPU: replay the filmed drops (measurements/ball/) and compare
uv run scripts/contact_sweep.py --dt 0.001 --timeconst 0.01 --dampratio 0.18 0.2   # ball-blade restitution
uv run scripts/make_sts3215_12v_params.py       # regenerate the 12 V servo params
uv run scripts/record_servos.py record --sim    # servo recording plan against the sim (dry run) -> build/servo_dryrun/
uv run scripts/record_servos.py map --port /dev/ttyACM0      # real arm, once: joint directions -> measurements/servo/
uv run scripts/record_servos.py record --port /dev/ttyACM0   # real arm: ~8 min recording -> measurements/servo/<time>/
uv run scripts/plot_servo_log.py <recording dir>             # goal vs measured per joint and pose
uv run scripts/fit_servos.py                    # fit servo model, delays, gear compliance to a recording (CMA-ES, GPU)
uv run scripts/identify_tracking.py             # linear tracking model of the servoed arm -> assets/actuators/tracking_model.json
uv run scripts/run_bouncer.py [--randomize] [--ball-latency 30 --ball-noise 2]   # scripted bouncer, many worlds
uv run scripts/run_bouncer.py --cameras overhead wrist [--ideal-cameras] [--overhead-long y]  # ball state from simulated cameras + filter
MUJOCO_GL=egl uv run scripts/render_camera_views.py      # camera previews for docs/camera_setup.md
MUJOCO_GL=egl uv run scripts/run_bouncer.py --envs 2 --lead 0 --video            # + build/bouncer.mp4
```

## Layout

| Path | What |
|---|---|
| `src/pingpong_sim/robot.py` | Robot MjSpec: Menagerie SO-101 minus the gripper, plus holder, camera, paddle and gear joints; servo actuator cfg; ready pose |
| `src/pingpong_sim/ball.py` | Ball (40 mm shell, 2.75 g, Cd 0.50), fitted ball-blade contact pair, air; scene callback |
| `src/pingpong_sim/harness.py` | Scene + Simulation without an RL env, for checks and system identification |
| `src/pingpong_sim/servo_bus.py` | STS3215 bus (feetech-servo-sdk) and a MuJoCo stand-in with the same interface; tick ↔ sim-radian map |
| `src/pingpong_sim/servo_model.py` | STS3215 model (BAM m6 physics) with per-world params, series gear-compliance joints, delays, current limit |
| `src/pingpong_sim/servo_actuator.py` | mjlab actuator for the fitted servo model + per-reset randomization; `encoder_pos`/`encoder_vel` for observations |
| `src/pingpong_sim/kinematics.py` | Batched torch FK/IK of the arm (blade face centre + normal, wrist camera pose), exact vs MuJoCo |
| `src/pingpong_sim/bouncer.py` | Scripted bouncer: drag-aware impact prediction, aim from measured restitution, single-joint stroke, tracking feedforward |
| `src/pingpong_sim/ball_filter.py` | Ball EKF shared with deployment: per-camera pixel + apparent-size updates, paddle-aware prediction through bounces, latency and out-of-order handling |
| `src/pingpong_sim/camera_sim.py` | Simulated overhead + wrist cameras (30 fps rolling shutter, wide lens, latency, noise, blur, occlusion, dropouts, calibration error) producing the filter's detections |
| `src/pingpong_sim/sysid.py` | Replays a recording in 2 s windows (n candidates × n windows worlds) and scores the encoder error |
| `src/pingpong_sim/assets/so101/` | Menagerie `robotstudio_so101`, unmodified (commit 3abca2e, Apache-2.0) |
| `src/pingpong_sim/assets/paddle/` | Holder, camera, paddle meshes in the holder frame (mm), from `cad/holder/export_sim_meshes.py` |
| `src/pingpong_sim/assets/actuators/` | `sts3215_12v_fit.json`: fitted servo params (training model); `sts3215_12v_m6.json`: BAM 7.4 V fit scaled to 12 V (pre-fit start) |

## Decisions

- **Stack pinned by BAM:** `better-actuator-models[mjlab]==1.0.3` requires mjlab 1.3, mujoco / mujoco-warp 3.7,
  warp-lang 1.12.
- **Euler integrator:** MuJoCo Warp 3.7 doesn't support the fluid (air drag) model with implicit integrators.
  BAM's joint damping is still integrated implicitly by Euler.
- **Timestep 1 ms:** the ball-blade contact needs at least ~5 steps to give a repeatable bounce. At 2 ms the
  restitution scatters by ±10–30 % with the contact phase unless the contact is made very soft (10 mm penetration).
- **Ball-blade contact:** explicit pair, `solref (0.01, 0.19)` → e = 0.58 at any speed (measured 0.60–0.57 over
  1.5–3.5 m/s); stiff `solreffriction` so oblique hits leave rolling. See the `ball.py` docstring for the gaps.
- **Air drag:** MuJoCo's ellipsoid fluid model on the ball geom; for a sphere the blunt coefficient is Cd / 2.
  Its Magnus term gives spin lift. Scenes need `ball.scene_spec_fn` (air density + the contact pair).
- **Robot frame:** base mounting face at the origin, +x forward. Ready pose: cap-side blade face level and up, face
  centre at (0.40, 0, 0.25) m.
- **Servos:** the fitted model (`servo_actuator.ServoActuatorCfg`). Joint targets and observations are encoder
  angles (`encoder_pos`), not the entity's `joint_pos` (servo joints there are the motor side of the gear springs).
  RL env cfgs need the `servo_init` startup event (per-world fields) and `get_robot_cfg(randomize=True)`.
- **Ball state = the deployment pipeline:** training and benchmarks feed `BallFilter` with `CameraRig` detections
  (overhead + wrist UVC cameras), never the true state. The filter updates per camera in undistorted image
  coordinates + apparent size, predicts through bounces with the encoder-FK paddle, and models timestamp and
  calibration residuals as measurement noise (without that, two slightly inconsistent cameras wreck the velocity).
  Consumers act only on a converged track (velocity std < 0.5 m/s).
- **Camera defaults:** image long side along the overhead tilt / blade normal; calibration residuals at 1.5× the
  deployment targets (overhead 3 mm / 0.3°, wrist mount 1 mm / 0.5°, timestamps ±2 ms). Cost: ~57 ms per control
  tick at 4096 worlds (launch-bound torch, nearly independent of world count).
