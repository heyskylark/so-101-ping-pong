# Ball–paddle restitution and air drag from slow-motion drop videos

Source: `ping-pong-drop-vids/pingpong-drop-vids/*.mov` (iPhone 16 Pro, 240 fps slo-mo on a
30 fps timebase, portrait). Numbers live in `results.yaml`; this file covers the method.

## Rerun

```sh
uv run measurements/ball/track.py ping-pong-drop-vids/pingpong-drop-vids/*.mov   # ~1 min, rewrites tracks/
uv run measurements/ball/analyze.py                                              # ~5 s, from tracks/ only
```

`track.py <video> …` regenerates `tracks/<stem>.csv` (per-frame ball track: `t_video, x_px,
y_px, diameter_px, width_px, height_px, flags`) and `tracks/<stem>_scene.csv` (paddle-blade and
tape-measure orientation). `analyze.py` writes `results.yaml` and the plots below.

| file | content |
|---|---|
| `restitution.png` | e vs impact speed for all centre impacts, fits, binned means; v_out vs v_in |
| `paddle_fits.png` | every centre-drop video: track, fitted flights, impact points |
| `timescale.png` | slow-down factor implied by each flight's curvature (ramp check) |
| `drag.png` | 6 ft drop residuals for two model variants; Cd per paddle video |
| `angled.png` | angled-paddle trajectories and fit windows |

## Method

**Tracking.** HSV threshold on the orange ball, connected components, shape gating (area,
fill ratio, aspect), rejection of static orange blobs (tape marks, a box on the shelf), then
continuity. Times are the per-frame PTS (`CAP_PROP_POS_MSEC`), never frame index: ~15% of
capture frames are missing.

**Timebase and scale.** Real time = video time / K with K = 8. Each flight (≥ 12 points,
> 1 cm) is also fitted alone with a free scale; with the ball-width scale this implies
K = 7.89 ± 0.15 across 107 flights. None is inconsistent, so no slow-mo ramps fall inside the
analysed flights. The vertical scale S_v [px/m] is fitted per video from gravity. It runs ~3%
above the ball-width scale because the thresholded blob is slightly smaller than the ball. A
1D perspective term `y = P0 + S·Y/(1+γY)` is fitted when the flights span more than 15 cm,
so S_v is the scale *at the paddle contact row*.

**Flights and impacts.** Impacts are bottom turning points of the track. The points between
two impacts are grown outward from a seed window while they keep matching a parabola, which
drops contact frames, the hand-hold and rolling. A flight is rejected if it has fewer than 8
points, an apex under 3 mm, or a residual above 3 px (or 3× the video median). All flights of
a video are fitted jointly with gravity + quadratic drag in closed form, using per-flight apex
time and height and a shared S_v, γ and Cd. The impact time is where consecutive flight
curves intersect; v_in and v_out are the model velocities there and e = v_out/v_in. Errors
come from the fit covariance, but the scatter between impacts (`resid_std`) is the realistic
uncertainty.

**Drag.** Two independent attempts:
1. *6 ft floor drops*: a joint fit of the falling flight (4.4 m/s) and the rising rebound,
   with depth correction from ball width, K = 8. With S_v pinned to the ball-diameter scale
   and no perspective term, the per-video fits give Cd = 0.53, 0.23, 0.40, so
   **0.38 ± 0.09**. The fast, blurred ball has structured ±5 px residuals, and Cd trades off
   against the scale and the camera-pitch term: 0.37 with free scale, 0.77 with free
   perspective, 0.51 pinned with perspective, 0.23 pinned × the 1.037 blob-size correction
   (and a worse fit). **The 6 ft fit is degenerate.** Literature for a smooth sphere at
   Re ≈ 10⁴: 0.40–0.50.
2. *16–25 in paddle drops*: the whole fall from the hand is in frame (up to 3.4 m/s). A fit
   per video with free Cd, S_v and γ gives 15 values with mean **0.50 ± 0.03** (std 0.10);
   fixing γ = 0 gives 0.42. This is the adopted value (v_terminal ≈ 8.5 m/s). As a check, the
   25 in drops fall 0.64 m and hit at 3.40 m/s against 3.55 m/s for free fall, which matches
   this drag.

Drag matters little for e: refitting with Cd = 0 changes e by +0.007 on average and +0.02
above 2.5 m/s. A free parabola per flight at 3.5 m/s biases v_in by +2% and e by −0.018 (see
`drag.parabola_bias`).

**Angled paddle.** Impact velocities come from per-flight fits in camera-frame metres. The
apparent ball size gives the depth correction and the depth velocity, assuming f ≈ 1400 px.
The blade tilt is measured in the image as the rotation of the blade's major axis minus the
tape-measure rotation (camera roll), relative to the level blade in the centre videos. With
the tilt known, the bounce splits into a normal part (e_n) and a tangential part
(r_t = v_t,out/v_t,in). The trajectory-only closures (assume e_n, or assume r_t = 1) are also
reported, to show that the trajectory alone cannot identify both.

## Caveats

- Tilt axis assumed along the line of sight; the blade's apparent aspect changes between the
  setups, so ±2° is added as a systematic. r_t is very sensitive to the tilt: ±2° moves it by
  ±0.1.
- The clamped paddle is compliant. Fast impacts meet up to ~15 mm below the resting contact
  row, and low-speed rebounds (< 0.6 m/s) have the largest scatter, possibly from blade
  vibration left over from the previous impact.
- Spin is not measured. Centre drops are vertical to within |v_x| ≈ 0.02 m/s (median); depth
  motion is not measured for them.
- `c-4in-*` use a different, closer camera position with only low flights (γ fixed at 0, S_v
  less certain; `c-4in-2` S_v/S_ball = 0.83). In `c-4in-1` and `c-4in-2` the release is not
  captured, so only rebounds are used. `c-8in-1` has false detections on another orange object
  before release, removed by an x-gate.
- Release heights are nominal tape-measure values. The fitted fall heights of the ball centre
  are in `videos.*.drop_check`: within 3% of nominal at 25 in, but 21–46% short for the 4–8 in
  drops, where the hand released the ball lower.
