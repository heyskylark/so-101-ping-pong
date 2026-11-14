# pingpong-sim

SO-101 + ping pong paddle + ball in mjlab (MuJoCo Warp). Python 3.12 (the BAM servo library supports only 3.12).

```bash
cd sim
uv sync
MUJOCO_GL=egl uv run scripts/inspect_model.py   # masses, ready pose, wrist-roll sweep, renders -> build/
uv run scripts/servo_check.py                   # GPU: hold ready pose, 0.3 rad step per joint
uv run scripts/validate_ball.py                 # GPU: replay the filmed drops (measurements/ball/) and compare
uv run scripts/contact_sweep.py --dt 0.001 --timeconst 0.01 --dampratio 0.12 0.135 0.15   # ball-blade restitution
uv run scripts/make_sts3215_12v_params.py       # regenerate the 12 V servo params
uv run scripts/record_servos.py record --sim    # servo recording plan against the sim (dry run) -> build/servo_dryrun/
uv run scripts/record_servos.py map --port /dev/ttyACM0      # real arm, once: joint directions -> measurements/servo/
uv run scripts/record_servos.py record --port /dev/ttyACM0   # real arm: ~8 min recording -> measurements/servo/<time>/
uv run scripts/record_servos.py strokes --port /dev/ttyACM0  # real arm, no ball: the bouncer's own strokes (~50 s) [--stroke-time --follow --apexes]
uv run scripts/plot_servo_log.py <recording dir>             # goal vs measured per joint and pose
uv run scripts/fit_servos.py --run <recording> [<strokes recording> ...] --stroke-weight 4   # servo model fit (CMA-ES, GPU)
uv run scripts/identify_tracking.py [--from <strokes recording>]   # tracking model for the bouncer's feedforward
uv run scripts/run_bouncer.py [--randomize] [--ball-latency 30 --ball-noise 2] [--apex --stroke-time --follow --impact-bias --predictor-gravity]   # scripted bouncer, many worlds
uv run scripts/run_bouncer.py --cameras overhead wrist [--ideal-cameras] [--overhead-long y]  # ball state from simulated cameras + filter
uv run scripts/run_bouncer.py --cameras wrist --stroke-joint wrist_flex --set wrist_only=1 [--goal-rate 200]   # pan + wrist only; prints arm motion
uv run scripts/run_bouncer.py --cameras overhead wrist --pose 0.28 0.31 --overhead-pos 0.47 0 0.78   # another ready pose / overhead camera placement
MUJOCO_GL=egl uv run scripts/render_camera_views.py      # camera previews for docs/camera_setup.md
MUJOCO_GL=egl uv run scripts/run_bouncer.py --envs 2 --lead 0 --video            # + build/bouncer.mp4
uv run scripts/eval_task.py --agent zero        # RL env with the scripted bouncer alone (residual task baseline)
uv run scripts/train.py PingPong-Bounce-Residual --env.scene.num-envs 4096       # PPO -> logs/rsl_rl/pingpong_residual/
uv run scripts/train.py PingPong-Bounce-Residual-Wrist --env.scene.num-envs 4096 # same, scripted stroke with the wrist -> pingpong_residual_wrist/
uv run scripts/train.py PingPong-Bounce-Gentle --env.scene.num-envs 4096         # the real arm's task: pan + wrist bouncer, wrist camera, capped residual, link flex -> pingpong_gentle/
uv run scripts/train.py PingPong-Bounce-Gentle-Memory --env.scene.num-envs 4096  # the same + the policy sees what this rally's bounces did vs predicted (BounceMemory) -> pingpong_gentle_memory/
uv run scripts/train.py PingPong-Bounce-Gentle-Carry --env.scene.num-envs 4096   # gentle + the bouncer carries the blade under the ball forward and back (lift, elbow, wrist; +-4 cm) -> pingpong_gentle_carry/
uv run scripts/train.py PingPong-Bounce-Gentle-Lead --env.scene.num-envs 4096    # memory task + the policy may move twice as far from the bouncer's goal and is paid for the ball's energy every step (short hops, a dead ball) -> pingpong_gentle_lead/
uv run scripts/warm_start.py <model.pt> --from-task PingPong-Bounce-Gentle --task PingPong-Bounce-Gentle-Memory --name <run>  # start a task with more observation terms from a trained policy; then train.py ... --agent.resume True --agent.load-run <run> --agent.load-checkpoint model_0.pt
uv run scripts/eval_task.py --checkpoint logs/rsl_rl/pingpong_residual/<run>/model_<n>.pt   # + bounce quality, residual size, timing
uv run scripts/play.py PingPong-Bounce-Residual --checkpoint-file <model.pt>     # interactive viewer
uv run scripts/export_policy.py logs/rsl_rl/pingpong_residual/<run>/model_<n>.pt --name <name>  # -> policies/<name>/ (ONNX + spec)
uv run scripts/sim2sim.py policies/<name> --episodes 96 --workers 16 [--belief-gravity 0.89]   # exported policy on CPU MuJoCo, deploy code path
uv run pytest tests                              # fast paths (one CPU world) == batched paths (training)
# Real rig (docs/hardware_setup.md, "Bring-up and calibration"):
uv run scripts/make_calib_targets.py            # printable AprilTag + ChArUco (Letter, exact scale) -> build/calib_targets.pdf
uv run scripts/cameras.py list|assign|preview|color     # camera devices, exposure, ball detection
uv run scripts/servo_ids.py scan|set|limits --port /dev/ttyACM0 [--from 6 --to 2] [--fix]   # which servo IDs answer; give a replacement servo its joint's ID; check (and fix) its torque / overload settings against an arm joint's
uv run scripts/calibrate.py selftest|targets|intrinsics|overhead|wrist|timing|gravity
uv run scripts/run_robot.py --port /dev/ttyACM0 [--policy policies/<name> | --scripted [--apex --stroke-time --follow --impact-bias --set KEY=VALUE]] [--record-frames]   # 100 Hz on the arm; gentle task: pan and wrist only (lift and elbow hold), wrist camera only; --record-frames saves every camera frame (~90 MB a minute)
uv run scripts/arm_pose.py ready --port /dev/ttyACM0 --readings   # load test: Enter prints the encoders' blade height; a ruler shows what bends unseen
uv run scripts/tune_hold.py --port /dev/ttyACM0 [--settings 64:0 32:16 16:32]   # shoulder lift and elbow with other firmware P:D gains: wobble after arriving, after a 3 deg step and under wrist strokes (~15 s per setting)
uv run scripts/run_robot.py --port /dev/ttyACM0 --pop 1.4   # pop test: a ball resting on the blade launched straight up, repeatedly
uv run scripts/run_robot.py --sim --scripted --seconds 60 [--sim-flex wrist_flex=15]  # same loop against the simulated arm, ball and cameras; --sim-flex fixes how soft the structure after a joint's encoder is [N m/rad] (the log then also holds the true face and ball). The rig's wrist: ~15 by the blade's sag (1.6 deg), ~7 by its lag in strokes (14 ms)
uv run scripts/analyze_run.py <run> [...] [--save-size-scale]   # real runs' bounces timed by the wrist camera: planned vs actual impact; from flights that begin on the blade: restitution (flight times), the ball's apparent size (the wrist camera's range scale), where first bounces go (a sign check of the face tilt; the tilt itself: calibrate.py face, by ruler)
uv run scripts/analyze_strokes.py <recording>    # dry strokes (record_servos.py strokes): blade speed and timing against the plan
```

## Layout

| Path | What |
|---|---|
| `src/pingpong_sim/robot.py` | Robot MjSpec: Menagerie SO-101 minus the gripper, plus holder, camera, paddle and gear joints; servo actuator cfg; ready pose |
| `src/pingpong_sim/ball.py` | Ball (40 mm shell, 2.75 g, Cd 0.50), fitted ball-blade contact pair, air; scene callback |
| `src/pingpong_sim/harness.py` | Scene + Simulation without an RL env, for checks and system identification |
| `src/pingpong_sim/servo_bus.py` | STS3215 bus (feetech-servo-sdk) and a MuJoCo stand-in with the same interface; tick ↔ sim-radian map |
| `src/pingpong_sim/servo_model.py` | STS3215 model (BAM m6 physics) with per-world params, series gear-compliance joints, delays, current limit |
| `src/pingpong_sim/servo_actuator.py` | `ServoDrive` (fitted servo model + command delay + disturbance + per-reset randomization, no mjlab) and the mjlab actuator around it; `encoder_pos`/`encoder_vel` for observations |
| `src/pingpong_sim/kinematics.py` | Batched torch FK/IK of the arm (blade face centre + normal, wrist camera pose, paddle link), exact vs MuJoCo; float64 numpy chain for one CPU world |
| `src/pingpong_sim/bouncer.py` | Scripted bouncer: drag-aware impact prediction, aim from measured restitution, single-joint stroke, tracking feedforward. `wrist_only`: aim and stroke with pan, wrist_flex and wrist_roll while lift and elbow hold (the blade tilts about the wrist: ~1 deg of lift/elbow motion instead of 40-60) |
| `src/pingpong_sim/ball_filter.py` | Ball EKF shared with deployment: per-camera pixel + apparent-size updates, paddle-aware prediction through bounces, latency and out-of-order handling |
| `src/pingpong_sim/camera_sim.py` | Simulated overhead + wrist cameras (30 fps rolling shutter, wide lens, latency, noise, blur, occlusion, dropouts, calibration error) producing the filter's detections |
| `src/pingpong_sim/controller.py` | What the robot runs, shared with training: `BounceController` (encoder arm history → ball filter on camera detections → scripted bouncer; actor observation terms) and `OnnxPolicy` (exported actor + `policy.json` layout → servo goals) |
| `src/pingpong_sim/task/` | mjlab bounce task: `state.py` (per-env simulated cameras → `BounceController`, launch phase, true-state bounce events), `mdp.py` (events, action term, observations, rewards, terminations, metrics), `env_cfg.py` (env + PPO cfgs); registers `PingPong-Bounce-Residual` / `-Direct` |
| `src/pingpong_sim/real/` | The real rig: `arm.py` (arm session: gain profile, checked minimum-jerk moves, watchdog, restore on exit), `cameras.py` (UVC capture threads with buffer timestamps, HSV ball detector, `rig.json`), `camera_model.py` (fisheye lens + pose + timing calibration files; pixel detections → filter measurements), `targets.py` (AprilTag / ChArUco detection), `calib_solve.py` (calibration solvers and pose planning), `sim_backend.py` (`CpuWorld` plant on CPU MuJoCo with bus/camera adapters) |
| `policies/<name>/` | Exported policies: `policy.onnx` (actor + normalizer) and `policy.json` (observation layout, action mapping, joint order, limits, task geometry) |
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
- **Ball-blade contact:** explicit pair, `solref (0.01, 0.135)` → e = 0.69 at any speed (the black rubber, measured
  0.68–0.70 on the rig; the red side is 0.58, damping ratio 0.19); stiff `solreffriction` so oblique hits leave
  rolling. See the `ball.py` docstring for the gaps.
- **Air drag:** MuJoCo's ellipsoid fluid model on the ball geom; for a sphere the blunt coefficient is Cd / 2.
  Its Magnus term gives spin lift. Scenes need `ball.scene_spec_fn` (air density + the contact pair).
- **Robot frame:** base mounting face at the origin, +x forward. Ready pose: cap-side blade face level and up, face
  centre at (0.28, 0, 0.31) m (`robot.READY_X`, `READY_Z`; the bouncer, controller and task default to it).
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
