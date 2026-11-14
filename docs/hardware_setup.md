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
   **Swapping a servo** once the gripper is off (LeRobot's `lerobot-calibrate` then fails: it wants all six
   motors): a 12 V 1:345 STS3215 (C018 / C047); `uv run scripts/servo_ids.py scan`, `set --from <id> --to <joint
   id>` with only that one unset servo on the bus, `limits --fix` (a servo that was a gripper keeps LeRobot's
   gripper protection: half the torque and an overload fallback of 25 %; on the shoulder lift and elbow that
   showed as a ~1° wobble at 3 Hz while holding and shaking while settling, 2026-11-12), `calibrate --joints
   <joint>` (LeRobot's procedure for those joints: middle pose, then stop to stop, by hand),
   `record_servos.py map`, then calibration steps 5 and 6 (`overhead`, `lift`: the joint's zero offset changed).
2. Remove the gripper servo and jaw; bolt the holder to the wrist-roll horn (same 4 M3 screws and roll zero as
   the stock follower part).
3. Clamp the paddle: handle waist in the holder's pocket, **cap-side face = hitting face**. Check the blade is
   flat in the holder; a few degrees of tilt walks the ball off the paddle (1° ≈ 3 cm drift per bounce at a
   25 cm apex).
4. Wrist camera on the holder, board turned so the picture's long edge is **perpendicular to the blade**
   (camera_setup.md, "Wrist camera").
5. Clamp the arm base to the table. The sim's origin is the base mounting face under the pan axis; "in front"
   means along the arm's straight-ahead direction at pan = 0.

Servo gains are written by the code at startup: P per joint (pan 16, lift 32, elbow 32, wrist_flex 32, roll 64),
I = 0, D = 0, Acceleration 0. The model and the policy assume exactly these; don't change them in LeRobot. Lift
and elbow were 64 until 2026-11-12: at 64 they hunt at ~3 Hz whenever they move
(`uv run scripts/tune_hold.py --port /dev/ttyACM0` compares settings on the arm).

## 2. Layout

![Side view of the layout, to scale](hardware_layout.svg)

- **Bounce point:** 0.28 m in front of the base axis, blade 0.31 m above the base mounting face, ball apex
  ~0.15 m above the blade. (Until 2026-11-12: 0.40 m out, 0.25 m up; moved in to halve the load the shoulder and
  elbow servos hold.)
- **Overhead post:** 0.61 m in front of the base on the straight-ahead line, plate facing back toward the arm. It
  calibrates the arm (tag on the paddle) from there. For play it was placed for the old bounce point: the gentle
  task tracks the ball with the wrist camera alone.
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
| **ChArUco board**, 5×7 squares of 30 mm (DICT_5X5_100) | Printed, glued flat to a rigid board (foam board, stiff cardboard) | Lens model of each camera (wave it in front of the camera); then lying flat on the table in front of the arm as the fixed target for the wrist camera mount (a hand-held or propped board drifts a few degrees, which the fit can't tell from a mount error) | Wrist mount ≤ 1 mm / ≤ 0.5° |

Both print on US Letter at exact scale: `uv run scripts/make_calib_targets.py`, then
`lp -o print-scaling=none -o media=Letter build/calib_targets.pdf` (page 1: tag ID 0 plus a spare, page 2: the
board, each with a 100 mm ruler). Measure the ruler, the tag's black square and a board square with calipers;
`calibrate.py targets` records them and the calibration uses those numbers.

Why the tag goes on the paddle rather than on the table: the arm moves it through ~20 poses where the encoders and
the model already say where the paddle is, so the overhead camera's pose comes out directly in the robot frame
(the tag's exact spot on the blade is solved along with it). A tag fixed to the table would need its own
position measured relative to the arm. The wrist camera can't see the paddle face (it's mounted on the paddle and
looks across the blade edge-on), hence the fixed board for it. To show the back-face tag to the overhead camera,
the arm rolls the paddle over (wrist_roll reaches ±157–163°).

## Bring-up and calibration

All from `sim/`. Each step saves into `measurements/calibration/` (`rig.json`, `camera_<name>.json`, `arm.json`, raw
data in `raw/`), and later steps refuse to run until the ones they need are done. Every arm step plans its moves,
checks them in the sim (joint and servo limits, table clearance, blade vs arm) and asks for "go"; Ctrl-C returns
the arm to where it started, a second Ctrl-C cuts torque.

**Start every arm step from the folded rest** (the pose the limp arm falls into: upper arm back, forearm folded up,
paddle pointing up). With the paddle resting tip-down on the table, joint offsets of a few degrees (measured later
by step 5) put the tip "inside" the table in the model, and the check refuses to move.

Aim the overhead camera with the arm at the real ready pose:
`uv run scripts/arm_pose.py ready --port /dev/ttyACM0 --snapshot overhead wrist` moves there, saves both pictures
to `sim/build/snapshots/` with the paddle face's pixel position, and returns. Planned: paddle centre near
(715, 360) in the overhead picture (arm entering from the right), blade a vertical strip mid-picture in the wrist one.

| # | Command | What you do | Result |
|---|---|---|---|
| 0 | `uv run scripts/calibrate.py selftest` | nothing (no hardware) | the solvers recover known answers from synthetic data |
| 1 | `uv run scripts/calibrate.py targets --ruler 100.0 --tag 70.1 --square 30.0` | type your caliper readings | sizes used by all fits |
| 2 | `uv run scripts/cameras.py list`, then `cameras.py assign overhead <device>` and `assign wrist <device>` | plug in one camera at a time to tell them apart (use the `/dev/v4l/by-path/` names) | `rig.json` |
| 3 | `uv run scripts/cameras.py preview overhead` (and `wrist`) | check orientation (camera_setup.md), set exposure with +/- (≤ 2 ms, picture still bright), toss the ball: it should be found in most frames | exposure saved; ball colour via `cameras.py color` |
| 4 | `uv run scripts/calibrate.py intrinsics overhead` (and `wrist`) | wave the board: still for a moment at each spot, all over the picture including the corners, tilted up to ~45°, near and far | lens model, RMS ≲ 0.5 px |
| 5 | `uv run scripts/calibrate.py overhead --port /dev/ttyACM0` (run it again to add poses; sessions pool) | tag on the paddle's back face; the arm rolls the paddle over and shows it to the overhead camera in 24 poses | camera pose, elbow / wrist_flex / wrist_roll offsets, tag placement; prints its uncertainty |
| 6 | `uv run scripts/arm_pose.py ready --port /dev/ttyACM0 --hold 60`, ruler from the table to the paddle top face's centre, then `uv run scripts/calibrate.py lift --paddle-height-cm <cm>`; then `calibrate.py table` (check) and `calibrate.py wrist --port /dev/ttyACM0` | measure while the arm holds ready (the hold saves what the encoders read, and `lift` uses that: the servos hold ~1° per joint below their goal, ~1 cm at the blade); for `table`/`wrist`: board flat on the table, printed side up, ~35–50 cm in front of the base, long side across, taped down; `table` only reads the overhead camera; the arm then rolls the paddle over and looks down at the board in 24 poses planned around where the overhead camera measured it | `lift`: the shoulder-lift offset (the overhead fit can't see it: it trades off against the camera's pitch; here it was ~10°), the overhead pose refitted with it; `table`: the board's tilt and height as a check (the table vs the base); `wrist`: the wrist camera mount, each pose's arm-vs-camera disagreement |
| 7 | `uv run scripts/calibrate.py gravity --port /dev/ttyACM0` | toss the ball gently 10–20 cm above the paddle without touching it, 10–15 times; recordings pool | a check of the lift offset (gravity straight down), each camera's ball-size scale; both cameras triangulate the flights |
| 8 | `uv run scripts/calibrate.py timing overhead --port ...`, then `timing wrist` (board untouched since step 6), then `gravity --port ... --no-lift` | the arm waves the paddle for 12 s; then tosses as in step 7 | timestamp offsets and rolling-shutter readout; final check: gravity tilts < 0.5° |
| 9 | `uv run scripts/calibrate.py static --port /dev/ttyACM0 --free` (`--refit`: report again) | set the ball on the blade centre at each height the arm stops at (default 28, 31, 34 cm), hands out of the picture; then, blade away, the ball on a support at a few heights, typing the ruler reading to the ball's top | per camera: position error and true / size-based range on the blade vs on a support. A range error only on the blade means blade pixels join the ball's blob; one that changes with distance means the size measurement does |
| 10 | `uv run scripts/run_robot.py --port /dev/ttyACM0 --scripted --seconds 60` | release the ball 10–15 cm above the blade centre (the wrist camera sees ~20 cm up) | the scripted bouncer alone (gentle task: pan and wrist only, lift and elbow hold, wrist camera only): a first test of the whole loop |
| 11 | `uv run scripts/run_robot.py --port /dev/ttyACM0 --policy policies/gentle_v2 --goal-rate 400 200 300 860 600` | same | the trained policy (`gentle_v2`: `PingPong-Bounce-Gentle`, trained 2026-11-12/13 on an arm whose wrist sags and lags and whose gears have play like this one's); logs in `measurements/runs/<time>/`. A policy gets the raw encoder view: the face tilt and paddle lag of `arm.json` are for the scripted bouncer only. Policies trained at another ready pose or with other firmware gains (`residual_v*`, `direct_v3`, `gentle_v1`) are refused |
| 12 | `uv run scripts/arm_pose.py ready --port /dev/ttyACM0 --readings` | Enter, ruler at the blade's far edge; ~100 g on the blade centre, Enter, ruler; weight off, Enter, ruler; `q` | load test: what the ruler shows beyond the encoders' printed height change is bending the encoders can't see (the simulator's `arm_flex` range) |

After changing the paddle (flipped, re-taped tag, new rubber): redo `cameras.py color` and `cameras.py mask` for
both cameras (the wrist camera's ignore mask holds the blade edge it sees), then steps 5 (the tag fit finds the tag
anywhere on the back face), 6 (`lift`, `wrist`), 8 and 9. The ball's restitution off the new hitting rubber needs
re-measuring too (`analyze_run.py` on a run with a few free bounces; `ball.BALL_SOLREF`, `BallFilterCfg.restitution`,
`BouncerCfg.restitution`: 0.68 on the black side, 0.58 on the red).

After reseating or tightening the paddle holder on the wrist-roll horn (paddle and wrist camera are one piece on
it): move the old `*_overhead.npz` sessions out of `measurements/calibration/raw/` (they pool), then steps 5, 6
(`lift`, `table`, `wrist`) and the face's lean: during a ready hold, ruler from the table to the blade's top surface
at its front and rear edge, then `uv run scripts/calibrate.py face --front-cm <cm> --rear-cm <cm> --length-cm <cm>`
(the tag fit leaves the face ~1.6° tipped forward of the model; where dropped balls go is not a measure of it: it
read three times the ruler). Lens models, timestamps (`timing wrist` gave the same offset before and after:
3.0 and 2.2 ms), the ball's size scale and the ignore masks stay. A loose holder shows in the wrist_roll offset of
step 5 jumping between sessions (+1.7° to +10.1° on 2026-11-12).

In fast strokes the paddle and the camera on it follow the encoders ~14 ms late (`paddle_lag` in
`measurements/calibration/arm.json`, which `run_robot.py` applies to the blade and camera pose it believes and to
the stroke's timing). It is read from runs: the lag at which, replayed, the ball estimate after a struck bounce
stops predicting the next impact early. `calibrate.py timing` waves slowly and does not show it.

The holder works loose under the strokes: again on 2026-11-13, during an evening of policy runs, and nothing the rig logs
showed it (the encoders, the wrist camera on the paddle and the timing step all move with or before it). A phone
video at 120-240 fps from behind the arm does: with the holder loose the blade had barely started up when the ball
landed and did most of its swing in the 30-110 ms after, while the encoders had the swing on time; tight, it is in
full swing at contact (`measurements/video/2026-11-13_side_IMG_4109.MOV` and `..._IMG_4110.MOV`;
`sim/build/analysis/video_sheet.py` cuts frame sheets). Before a session: hold the wrist servo's body and rock the
blade's tip up and down by hand; any click or free travel means tighten first.

Why each step exists:

- **Lens:** fisheye model (barrel distortion beyond −17 % at the edges), fitted from ChArUco corners.
- **Overhead pose:** the tag's corners at ~24 arm poses, with the encoder angles there, fit the camera pose in the
  robot frame, where the tag sits on the blade, and the elbow, wrist_flex and wrist_roll zero offsets. The tag is
  assumed flat on the blade: that is what makes the wrist_roll offset (the paddle's sideways tilt) measurable. Pan
  and lift can't be measured this way: the tag is beyond both joints, so a pan or lift offset looks exactly like
  the camera turned about that joint's axis.
- **Wrist mount:** the board's corners seen from ~20 poses fit the camera's mount on the holder (CAD as the start)
  and where the board stands.
- **Timing:** the target's track against the encoders gives the capture time of each frame relative to the driver's
  buffer timestamp (to < 1 ms in the selftest) and the rolling-shutter readout across rows.
- **Gravity:** ball flights fitted with drag. A lift offset tilts the robot's idea of "level" about the lift axis,
  so ball flights seen through the overhead camera fall at a slant; step 6 finds the lift offset that makes them fall
  straight down (a planted 3.00° comes back as 2.96° in the selftest). The final check reports any tilt left: about
  the forward axis that means the arm base or table isn't level. Tilt matters: 1° ≈ 3 cm of drift per bounce.

## Running

`scripts/run_robot.py` loads the calibration, connects the arm (gain profile, torque on), starts both cameras and
moves to the ready pose, then runs at 100 Hz: encoders and ball detections go through the same controller code as
training (filter, impact prediction, scripted bouncer, observation layout from `policy.json`) into the ONNX policy.
The arm holds ready until the ball is tracked above the paddle and moving (dropped, or held and then released), plays
until the ball is lost, low or more than 25 cm away, then returns to ready and waits for the next ball; a ball tracked
back above the blade and
falling once the return is half done is played again. A ball lying still on the paddle is never played (training
starts every episode from a falling ball): drop it. Goals are limited to 0.15 rad per tick
and kept off the table (blade ≥ 12 cm up) and below 45° of tilt; the watchdog stops on a stuck joint, heat or low
supply. `--sim` runs the same loop against the simulated arm, ball and cameras: work per tick is ~2.5 ms median,
~5 ms p99 on this PC, inside the 10 ms tick with the bus I/O.

Scripted-bouncer options (the policy was trained on the defaults): `--apex` (bounce height above the hitting plane;
the overhead camera, ~0.51 m up, loses the ball above ~0.2 m; below ~0.2 m the stroke and follow-through no longer
fit between bounces, sim: 0.12 m keeps under 1 % of balls). `--ball-gravity` is an experiment: the gravity the ball
filter and the impact predictor assume (real runs predicted impacts ~28 ms early).

## Safety

- Keep the 12 V switch within reach. The arm goes limp when power is cut; at the ready pose the paddle drops onto
  the table, so keep the area under it clear.
- Stay out of the arm's reach while it runs: strokes reach ~0.6 m/s at the blade and the pan joint can swing.
- Start a run with the ball in your hand above the paddle. While both cameras see it still (a held ball: < 16 px
  and < 5 % size change over 0.2 s) the arm stays at ready, and play starts at the release; without that the
  scripted bouncer swings at a held ball (the filter has it falling at ~1 m/s; the same in `--sim`). Training
  holds the ball 0–1 s before letting go.

## Software status

| Piece | State |
|---|---|
| Servo bus, joint map, arm session (`servo_bus.py`, `real/arm.py`, `record_servos.py`) | Done, used for the servo recordings |
| Ball filter, scripted bouncer, controller (`ball_filter.py`, `bouncer.py`, `controller.py`) | Done, shared with training; single-world CPU fast paths (`tests/test_fast_paths.py`) |
| Trained policies, ONNX + spec (`sim/policies/<name>/`) | `residual_v1` (deploy first), `residual_v2`, `direct_v3` |
| Camera capture + ball detection (`real/cameras.py`, `scripts/cameras.py`) | Written; detection checked on synthetic frames. Needs the cameras: exposure, colour thresholds, buffer timestamps |
| Camera model and calibration files (`real/camera_model.py`) | Done; exercised by `run_robot.py --sim` |
| Calibration (`scripts/calibrate.py`, `real/calib_solve.py`) | Written; `selftest` passes. Needs the arm and cameras |
| 100 Hz loop (`scripts/run_robot.py`) | Runs end to end in `--sim` (bounces continuously, 100 % tracked). Needs the arm |
