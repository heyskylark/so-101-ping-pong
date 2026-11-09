# Hardware setup

Everything the sim assumes about the real rig, in the order you build it. Camera mounting details are in
[`camera_setup.md`](camera_setup.md); the plan and the measurements behind each number are in [`TODO.md`](../TODO.md).

## Parts

| Part | Notes |
|---|---|
| SO-101 follower arm, 5 servos | Feetech STS3215 **12 V / 30 kg·cm** on IDs 1–5 (pan, lift, elbow, wrist_flex, wrist_roll). The gripper servo (ID 6) and jaw come off. |
| 12 V supply + bus adapter | The arm's usual supply and USB servo board; the arm shows up as `/dev/ttyACM0` (or under `/dev/serial/by-id/`). A switch on the 12 V line you can reach. |
| Paddle holder | `cad/holder/` two-piece clamp (replaces the wrist-roll follower part), 4× M3x10 into M3 heat-set inserts. |
| Paddle | NIBIRU SPORT, clamped in the holder; the **cap-side face hits the ball** (faces up at the ready pose). |
| Wrist camera | InnoMaker U20CAM-720P on the stock hex-nut 32x32 mount, on the holder's side (2× M3x8 + nuts). |
| Overhead camera | Second U20CAM-720P on the SO-ARM100 overhead mount, **top section reprinted** as `cad/overhead_mount/cam_mount_top_rotatable.stl`. |
| Balls | 40 mm 3-star ABS (2.75 g). Orange is easier to separate from a white wall and the arm. |
| Lighting | Bright, diffuse, **flicker-free** LED (DC or high-frequency driver). |
| Calibration targets | One printed AprilTag on the paddle's back face, one printed ChArUco board (see [AprilTags](#apriltags-and-calibration-targets)). |
| Host | The training PC: USB for the servo board and both cameras; the control loop runs on the CPU at 100 Hz. |

## 1. Arm

1. Calibrate the arm with LeRobot as usual (it writes the homing offsets and position limits the bus code reads).
   The joint map that converts servo ticks to sim angles already exists (`measurements/servo/joint_map.json`,
   from `uv run scripts/record_servos.py map`); redo it only after re-calibrating with LeRobot or swapping a servo.
2. Remove the gripper servo and jaw; bolt the holder to the wrist-roll horn (same 4 M3 screws and roll zero as
   the stock follower part).
3. Clamp the paddle: handle waist in the holder's pocket, **cap-side face = hitting face**. Check the blade is
   flat in the holder; a few degrees of tilt walks the ball off the paddle (1° ≈ 3 cm drift per bounce at a
   25 cm apex).
4. Wrist camera on the holder, board turned so the picture's long edge is **perpendicular to the blade**
   (camera_setup.md, "Wrist camera").
5. Clamp the arm base to the table. The sim's origin is the base mounting face under the pan axis; "in front"
   means along the arm's straight-ahead direction at pan = 0.

Servo gains are written by the code at startup: the fitted P per joint (pan 16, lift 64, elbow 64, wrist_flex
32, roll 64), I = 0, D = 0, Acceleration 0. The model and the policy assume exactly these; don't change them in
LeRobot.

## 2. Layout

![Side view of the layout, to scale](hardware_layout.svg)

- **Bounce point:** 0.40 m in front of the base axis, blade 0.25 m above the base mounting face, ball apex
  ~0.25 m above the blade.
- **Overhead post:** 0.61 m in front of the base on the straight-ahead line, plate facing back toward the arm.
  Within ±5 cm / ±5° is fine (training covers that); calibration measures the exact pose.
- **Clear space:** nothing within ~0.3 m of the bounce point up to ~0.6 m height. Episodes in training end when
  the ball drifts 0.2 m from the target, so it lands anywhere in that circle; a towel underneath saves chasing balls.
- **Background:** matte and dark where the cameras look: a dark cloth on the table under the bounce (overhead
  view) and a dark board or cloth beyond the bounce in the wrist camera's view (it looks forward and across,
  toward the overhead post). No orange or white clutter in either picture.
- **Light:** bright enough for a **≤ 2 ms manual exposure** (camera_setup.md, "Image settings"). Use flicker-free
  LEDs: at 2 ms the rolling shutter turns mains flicker into bright and dark bands across the picture.

## 3. Wiring

- 12 V supply → servo board → arm; servo board USB → PC.
- Each camera on its own USB port. Two 720p MJPEG streams at 30 fps fit easily on USB 2; avoid unpowered hubs.
- Note the stable device names: `ls /dev/v4l/by-id/` (cameras) and `ls /dev/serial/by-id/` (servo board). The
  two cameras are the same model, so tell them apart by port path (`/dev/v4l/by-path/`) and keep them plugged
  into the same ports.

## 4. Cameras

Follow [`camera_setup.md`](camera_setup.md): mount orientation (long edge along the tilt / across the blade),
preview checks, manual exposure ≤ 2 ms, auto white balance off.

## AprilTags and calibration targets

**Yes, one AprilTag, and one ChArUco board.** Neither is used while bouncing (the policy only sees the ball, and the
arm's pose comes from the encoders); they are for calibration, which the sim says matters more than noise, frame rate or
latency.

| Target | Where | Used for | Accuracy needed |
|---|---|---|---|
| **AprilTag**, tag36h11 ID 0, ~70 mm black square | Glued flat on the paddle's **back face** (base side, the face that never hits); roughly centred, exact placement doesn't matter | Overhead camera position and angle in the robot frame; each camera's timestamp offset; joint zero offsets as a by-product | Overhead ≤ 3 mm / ≤ 0.3°; timestamps ≤ ±2 ms |
| **ChArUco board**, ~A4 / Letter, e.g. 7×5 squares of 30 mm | Printed, glued to a flat rigid board | Lens model of each camera (wave it in front of the camera); then standing upright on the table in front of the arm as the fixed target for the wrist camera mount | Wrist mount ≤ 1 mm / ≤ 0.5° |

Print both at 100 % scale on matte paper and measure the printed tag's black square and the board's square size with
calipers; those numbers go into the calibration.

Why the tag goes on the paddle rather than on the table: the arm moves it through ~20 poses where the encoders and
the model already say where the paddle is, so the overhead camera's pose comes out directly in the robot frame
(the tag's exact spot on the blade is solved along with it). A tag fixed to the table would need its own
position measured relative to the arm. The wrist camera can't see the paddle face (it's mounted on the paddle and
looks across the blade edge-on), hence the fixed board for it. To show the back-face tag to the overhead camera,
the arm rolls the paddle over (wrist_roll reaches ±157–163°).

Calibration steps (TODO.md §3; the scripts for these are not written yet):

1. **Lens** (each camera): ~30 views of the ChArUco board across the whole picture, corners included → fisheye
   lens model.
2. **Overhead camera pose:** the arm shows the paddle tag to the overhead camera in ~20 poses → camera pose in the
   robot frame (plus the tag's spot on the blade and per-joint zero offsets).
3. **Wrist camera mount:** board standing on the table; the arm moves so the wrist camera sees it from ~20 poses →
   camera pose on the holder (CAD value as the starting guess).
4. **Timestamp offsets:** the arm waves the paddle sinusoidally with the tag in view of each camera; matching the
   tag track against the encoder track gives each camera's capture-time offset (to ~1 ms) and the rolling-shutter
   readout time.
5. **Checks:** drop a ball and fit its flight; the acceleration must come out as straight down at 9.81 m/s² in the
   robot frame (a 1° frame tilt is as bad as a 1° paddle tilt). A ball resting on the level paddle must not roll.

## Safety

- Keep the 12 V switch within reach. The arm goes limp when power is cut; at the ready pose the paddle drops onto
  the table, so keep the area under it clear.
- Stay out of the arm's reach while it runs: strokes reach ~0.6 m/s at the blade and the pan joint can swing.
- Start a run with the ball in your hand above the paddle; the controller waits for a converged ball track
  before it strikes. Releasing a ball you're holding is part of training (held 0–1 s, then let go).

## Software status

| Piece | State |
|---|---|
| Servo bus, joint map, gain profile (`sim/src/pingpong_sim/servo_bus.py`, `record_servos.py`) | Done, used for the servo recordings |
| Ball filter, scripted bouncer, controller (`ball_filter.py`, `bouncer.py`, `controller.py`) | Done, shared with the sim; runs the policy on CPU MuJoCo (`sim/scripts/sim2sim.py`) |
| Trained policy, ONNX + spec (`sim/policies/<name>/`) | Done (`residual_v1`) |
| Camera capture + ball detection (MJPEG, undistort, blob → pixel centre and diameter, timestamps) | To do |
| 100 Hz run loop on the real arm (encoders + detections → controller → servo goals), with limits and a stop key | To do |
| Calibration scripts (steps 1–5 above) | To do |
