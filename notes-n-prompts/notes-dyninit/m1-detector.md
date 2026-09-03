# M1 — the static/dynamic detector

`src/init_detect.{h,cpp}`, `src/test/unittest_init_detect.cpp`,
`src/app/init_probe.cpp`. Observation only: nothing in `Estimator` calls it yet.

M0 established that no accelerometer statistic can see constant-velocity motion,
and that MH_01 and MH_02 are moving at 0.671 and 0.481 m/s when XIVO's
initializer fires. M1 builds the thing that notices.

## 1. Which cue, and why not the obvious ones

Four candidates were measured offline over all 17 sequences of the two datasets
XIVO is evaluated on
(`notes-n-prompts/notes-dyninit/harness/flow_diag.py`, 5 s horizon,
0.5 s windows, minimum over window placements). Margin is the moving minimum
divided by the static maximum, so it is the factor by which a single threshold
separates the two classes:

| cue | moving min (2) | static max (15) | margin |
|---|---|---|---|
| best-fit-rotation flow residual | 0.610 px | 0.090 px (V2_02) | **6.8x** |
| raw pixel disparity — OpenVINS' cue | 3.801 px | 0.765 px (room2) | 5.0x |
| accelerometer sample sd | 0.497 m/s² | 0.258 m/s² (V2_03) | 1.93x |
| gyro-de-rotated flow | 1.654 px | 1.211 px (V2_03) | 1.37x |

Three of the four are wrong in kind, not merely weaker:

* **`| |a| - |g| |`**, which is what the shipped `gravity_init_max_accel_dev`
  gate uses, is blind to constant velocity *in principle*. Specific force is
  `R'(a - g)`; at constant velocity `a = 0` and the magnitude reads exactly `|g|`.
  Pinned as a test (`ConstantVelocityDefeatsTheShippedGate`).
* **Accelerometer sample sd** does separate the classes, but only just, and only
  as a minimum over windows. It also cannot see a *constant acceleration*, since
  a constant world-frame acceleration with no rotation makes the specific force a
  constant vector — its variance is exactly zero
  (`StartsAtRestThenAcceleratesIsStatic` pins this).
* **Raw pixel disparity** cannot tell translation from rotation: rotation
  produces disparity at any depth. TUM-VI room2 shows 0.765 px of it with the rig
  essentially at rest, and that number grows with pan rate, so the 5.0x margin is
  not a property of the statistic but of how gently these particular sequences
  happen to be handled.
* **Gyro-de-rotated flow** removes the rotation but assumes an unbiased gyro. At
  turn-on EuRoC's ADIS16448 reads 0.079–0.085 rad/s while sitting still, which is
  essentially all of what it reads at all; over one 50 ms frame gap at f = 458 px
  that is 1.8 px of predicted motion that never happened, larger than the signal
  it is trying to isolate. And the gyro bias is one of the quantities dynamic
  initialization exists to estimate, so using it to decide whether to estimate it
  is circular.

What survives is to **fit the rotation from the images** and measure the flow it
fails to explain. Some rotation explains a purely rotating camera's flow exactly,
at any depth and any rate; no rotation explains parallax, because parallax depends
on depth and rotation does not. The residual is therefore a translation signal
owing nothing to the gyro and nothing to scene scale.

The fit is Wahba's problem on the **unit sphere** — `min_R Σ w_i |u2_i - R u1_i|²`,
solved in closed form as `R = U diag(1,1,det(UV')) V'` from the SVD of
`M = Σ w_i u2_i u1_i'` — with three Huber IRLS passes. On the sphere and not in
normalized image coordinates: normalized coordinates are the tangent of the field
angle, so on TUM-VI's ~180° fisheye they diverge toward the image edge, a
least-squares fit in them weights edge points by orders of magnitude more than
centre points, and the residual is not comparable across the image. Fitting in
normalized coordinates was in fact the first thing tried, and it put TUM-VI at
0.167–0.609 px — indistinguishable from MH_01's 0.610. Moving to the sphere and
scoring the residual in pixels through the real camera model dropped TUM-VI to
0.022–0.060 px. Same data, same tracks; the 6.8x margin above is entirely a
consequence of getting the parameterization right.

## 2. The decision rule

**Vision decides whenever it has an opinion. The accelerometer is the fallback
for when fewer than `min_tracks` survive, not a second trigger and not a veto.**

Both obvious combinations are worse, for opposite reasons:

* `imu AND flow` reintroduces exactly the blind spot the detector exists to
  remove: an accelerometer cannot see constant velocity at all, so requiring it to
  agree lets a steady glide through as static.
* `imu OR flow` false-positives on a rig that is stationary but steadily
  rotating. Gravity then sweeps through the body frame, and at 0.3 rad/s that
  alone is ~`9.81 · 0.15 / sqrt(12)` ≈ 0.42 m/s² of accelerometer sd — past any
  threshold MH_02's 0.497 has to clear. TUM-VI room4 averages 0.32 rad/s at init,
  so this is ordinary handheld behaviour, and the cost would be a bundle
  adjustment on a window with no parallax in it.

This was caught before the first synthetic test was run, and
`RotatingButStillIsStatic` now pins it from both sides: it asserts the verdict is
static *and* asserts `accel_sd > 0.35`, so the reason the rule is what it is
cannot quietly stop being true.

## 3. Results — 17 of 17, from the C++ implementation

`bin/init_probe` feeds the real dataset loader, the real camera model and the real
KLT, and reports the verdict at the instant `Ready()` first returns true — which
is the instant the estimator will ask. Default options: `window_sec` 0.5,
`horizon_sec` 2.0, `flow_thresh` 0.25 px, `min_tracks` 15.

```
sequence                     verdict    t_dec   flow_px accel_sd  pairs  bias_hint
MH_01_easy                   dynamic    1.000    2.0264   0.2036     19     0.2706
MH_02_easy                   dynamic    1.000    2.5289   0.3433     19     0.1716
MH_03_medium                  static    1.000    0.0271   0.3387     19     0.0777
MH_04_difficult               static    1.000    0.0971   0.1553     19     0.0740
MH_05_difficult               static    1.000    0.0164   0.0514     19     0.0791
V1_01_easy                    static    1.000    0.0110   0.6606     20     0.0792
V1_02_medium                  static    1.000    0.0140   0.0476     20     0.0740
V1_03_difficult               static    1.000    0.0245   0.0367     19     0.0720
V2_01_easy                    static    1.000    0.0136   0.2888     19     0.0861
V2_02_medium                  static    1.000    0.0462   0.2105     19     0.0794
V2_03_difficult               static    1.000    0.0261   0.9286     19     0.0849

room1                         static    1.003    0.0414   0.3078     19     0.0102
room2                         static    1.003    0.0460   0.3143     19     0.0115
room3                         static    1.003    0.0400   0.1456     19     0.0181
room4                         static    1.003    0.0513   0.1259     19     0.0179
room5                         static    1.003    0.0634   0.1267     19     0.0213
room6                         static    1.003    0.0644   0.2422     19     0.0158
```

**17 of 17 correct** — MH_01 and MH_02 dynamic, the other fifteen static.

Three things in this table are worth more than the classification itself.

**The margin at the horizon that matters is 20.9x, not 6.8x.** Moving minimum
2.026 px (MH_01) against static maximum 0.097 px (MH_04). The offline table's
6.8x was measured over a 5 s horizon, which gives min-over-windows ten times as
many placements to find a quiet one in; at the 1 s the detector actually decides
at, the classes are far further apart. The 0.25 px threshold sits 2.6x above the
loudest static sequence and 8.1x below the quietest moving one.

**At this horizon the accelerometer cue is not merely weak — it is inverted.**
Static V2_03 reads 0.929 m/s² and static V1_01 reads 0.661, while moving MH_01
reads 0.204. No threshold on accelerometer sd classifies these 17 sequences
correctly at a 1 s horizon; the best possible one gets 15 of 17 by calling
everything static. The offline 1.93x margin came from a 5 s window, where MH_01's
later motion is available. This retroactively settles the rule in §2 more sharply
than the reasoning did: had the accelerometer been given a vote, it would have
voted wrong on the two sequences this whole milestone is about.

**The gyro-bias hint independently reproduces two datasheets.** It reads
0.072–0.086 rad/s on the nine static EuRoC sequences — the ADIS16448's turn-on
gyro bias, and the same 0.079–0.085 the offline script measured a different way —
and 0.010–0.021 rad/s on TUM-VI's BMI160. On MH_01 and MH_02 it reads 0.27 and
0.17, which is *not* bias: with the rig translating, the flow the fit cannot
explain contaminates it. It is a seed for M3, labelled a hint, and nothing depends
on it.

Getting that agreement required a real fix. The first version compared the gyro's
body-frame rotation directly against the image-fitted camera-frame rotation, and
read 0.18–0.44 rad/s on TUM-VI — 20x the truth. Conjugating a rotation by the
extrinsics preserves *its own* angle, which is what made the omission look
harmless, but not the angle of a **product** of two rotations, and TUM-VI's camera
is flipped nearly 180° relative to the IMU. `Options::Rbc` now carries the
extrinsics and the hint agrees with the offline measurement on both datasets. The
verdict itself never needed extrinsics and still does not.

## 4. Unit tests — 11 of 11

`bin/unitTests_init_detect`, ctest name `InitDetect`. Sequence-level cases render
~400 points through the real EuRoC radtan model at 752x480 with bilinear
sub-pixel blob placement, then run the real `goodFeaturesToTrack` + KLT, so the
tracking and the projection are covered and not only the arithmetic. Camera path
and IMU stream come from the same exact closed-form integral, so any residual the
detector reports is parallax rather than a discretisation artifact.

| test | what it guards |
|---|---|
| `RecoversRotationExactly` | Wahba fit, noiseless, to 1e-12 |
| `ReturnsARotationNotAReflection` | near-coplanar bearings; `det R = +1` |
| `HuberSurvivesOutliers` | 10% garbage tracks; IRLS beats the plain fit |
| `RotatingButStillIsStatic` | 0.3 rad/s, no translation → static, *and* accel sd > 0.35 |
| `ConstantVelocityIsDynamic` | 0.6 m/s → dynamic, *and* accel sd is exactly 0 |
| `ConstantVelocityDefeatsTheShippedGate` | `\| \|a\|-\|g\| \|` < 1e-12 on that motion |
| `BiasedGyroButStillIsStatic` | 0.08 rad/s of pure gyro bias → static |
| `StartsAtRestThenAcceleratesIsStatic` | see below |
| `AlreadyMovingAndShakenIsDynamic` | the MH_01 signature; both cues agree |
| `UndecidedBeforeTheWindowIsFull` | no verdict is offered too early |
| `FallsBackToTheImuWhenThereIsNoTexture` | blank image → 0 pairs → IMU answers |

`StartsAtRestThenAcceleratesIsStatic` deserves the note. A rig at rest at t=0 that
then accelerates at 1.5 m/s² classifies **static**, and that is the intended
answer, not a miss. Both statistics are minimised over candidate windows, so the
earliest window decides — and in the earliest window the rig has moved 6 mm. The
instant being initialized *at* is the start of that window, and `v = 0` is exactly
right there; a bundle adjustment would be solving for a velocity that is zero on a
baseline of 6 mm. MH_01 and MH_02 are not this case: they are already moving at
0.67 and 0.48 m/s in their first window, which is why min-over-windows still
reports 2.03 and 2.53 px on them.

This test failed on first run at 0.245 px against the 0.25 threshold, which is
how the property got understood rather than assumed. (The run before that failed
for a duller reason: the acceleration was passed into the helper's `gyro_bias`
slot, so it simulated a perfectly static rig and the `static` verdict was
correct.)

## 5. Deliberately self-contained

`MotionDetector` runs its own corner detection and optical flow instead of
borrowing XIVO's `Tracker`. The tracker is entangled with the feature memory pool
and the group/feature lifecycle; running it before the filter exists and then
discarding its output would put pool and id state at risk. Keeping the pre-init
window out of that machinery is what makes "the static path is untouched" a
structural property rather than a hope. The cost is one duplicated KLT for as long
as the detector is undecided — under 1 s of frames.

## 6. What is not yet established

* **The IMU fallback is unexercised on real data.** All 17 sequences produce
  19–20 tracked frame pairs, so vision always has an opinion and
  `FallsBackToTheImuWhenThereIsNoTexture` is the only evidence the fallback works
  at all. Given §3's finding that accelerometer sd is inverted on these sequences
  at a 1 s horizon, the fallback should be understood as "do something defensible
  when blind", not as a calibrated second opinion.
* **The visual cue's own limitation is `|t|/depth`.** A far-field scene raises the
  static floor, exactly as it does for OpenVINS' disparity threshold. Neither
  dataset has one; nothing here measures that regime.
* `flow_thresh` and `imu_thresh` are set from these 17 sequences and are tuned
  properly in M5.

## 7. Regression control

`ctest`: 24 of 24 pass (23 before M1, plus `InitDetect`).

n=3 EuRoC stereo pass in this worktree against `euroc_m6_final/fast` members 0-2:
**66 of 66 trajectory files byte-identical, 165 of 165 metric cells identical.**
Details in `m1-control.txt`. The detector is not called from `Estimator`, so this
is a check that adding a translation unit to `xest` and relinking changed nothing,
not a check of the detector — but worth running, because this filter is chaotic
enough that a one-ulp change in the covariance update shows up in the third digit
of ATE.

## 8. Reproduce

```sh
# build
cd xivo-dyninit/build && make -j64

# unit tests
cd .. && ./bin/unitTests_init_detect          # or: cd build && ctest -R InitDetect

# the 17-sequence table
for s in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult \
         V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium \
         V2_03_difficult; do
  GLOG_minloglevel=2 ./bin/init_probe -cfg cfg/euroc_stereo.json \
      -dataset euroc -root ../data/euroc -seq $s
done
for s in room1 room2 room3 room4 room5 room6; do
  GLOG_minloglevel=2 ./bin/init_probe -cfg cfg/tumvi_stereo.json \
      -dataset tumvi -root ../data/tumvi -seq $s
done

# the offline cue comparison (venv on PATH, from the worktree root)
python3 notes-n-prompts/notes-dyninit/harness/flow_diag.py \
    --dataset euroc --root ../data/euroc --cfg cfg/euroc_stereo.json
python3 notes-n-prompts/notes-dyninit/harness/flow_diag.py \
    --dataset tumvi --root ../data/tumvi --cfg cfg/tumvi_stereo.json

# the regression control (from experiments/openvins, NOT the in-repo copy --
# the scripts resolve $WORKSPACE as $HERE/../..)
cd ../experiments/openvins
CPU_BASE=0 CPU_SPAN=60 ./sweep_xivo.sh --name fast --members 3 --mode stereo \
  --worktree xivo-dyninit --out ../results/dyninit_m1_control \
  --seqs "MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult \
V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult"
```
