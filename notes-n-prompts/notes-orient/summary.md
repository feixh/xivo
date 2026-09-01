# Orientation accuracy: summary

Branch `auto-orient`, forked from `auto` @ 9e3ec06. Two commits.

Everything below is the standard harness -- `experiments/openvins/run_xivo_reference.sh
--jitter 6`, TUM-VI room1-room6, both modes, 6-member ensemble perturbing `X.Vsb`
by `k*1e-6` m/s, 72 runs per row. Result dirs: baseline
`experiments/results/xivo_ref_jitter`, after M1 `experiments/results/orient_m1`,
final `experiments/results/orient_final`. OpenVINS reference is
`experiments/results/ov_accuracy`.

## Headline

| metric | mono before | mono after | | stereo before | stereo after | | OpenVINS mono / stereo |
|---|---|---|---|---|---|---|---|
| **orientation ATE [deg]** | 1.824 | **0.949** | -48% | 1.798 | **0.983** | -45% | 1.574 / 1.444 |
| RPE 8 m rotation [deg] | 0.5153 | 0.5149 | -0.0004 | 0.5074 | 0.5101 | +0.0027 | 0.645 / 0.584 |
| position ATE `ate_002` [m] | 0.0928 | 0.0957 | +0.0029 | 0.0636 | 0.0636 | +0.0000 | 0.0621 / 0.0677 |
| position ATE `ov_ate_pos` [m] | 0.0968 | 0.0963 | -0.0005 | 0.0688 | 0.0640 | -0.0048 | 0.0638 / 0.0697 |
| RPE 8 m position [m] | 0.0480 | 0.0489 | +0.0009 | 0.0292 | 0.0293 | +0.0001 | 0.0308 / 0.0265 |

**Primary goal: orientation ATE <= 1.42 deg in both modes. MET** -- 0.949 mono and
0.983 stereo, a 33% / 31% margin, and 40% / 32% better than the OpenVINS reference
runs.

**Non-regression constraints, all satisfied:**

1. RPE 8 m rotation <= 0.53 deg: **0.5149** mono, **0.5101** stereo. Mono is
   fractionally *better* than baseline; stereo is +0.0027, which is 0.4 sigma.
2. Position ATE not worse than baseline by more than 0.005 m: `ate_002` +0.0029
   mono, +0.0000 stereo. On the other position metric, `ov_ate_pos`, both modes
   *improve* (-0.0005 / -0.0048).
3. No sequence diverged: no row of either `summary.csv` has position ATE > 0.5 m or
   orientation ATE > 10 deg, over all 144 runs.

Per-sequence orientation ATE [deg]:

| | room1 | room2 | room3 | room4 | room5 | room6 | mean |
|---|---|---|---|---|---|---|---|
| mono before | 1.507 | 3.230 | 2.049 | 1.442 | 1.136 | 1.582 | 1.824 |
| mono after M1 | 1.225 | 0.920 | 1.423 | 0.841 | 0.955 | 0.713 | 1.013 |
| **mono final** | 1.060 | 0.675 | 1.402 | 0.895 | 1.042 | 0.618 | **0.949** |
| stereo before | 1.212 | 3.177 | 2.438 | 1.431 | 0.977 | 1.555 | 1.798 |
| stereo after M1 | 0.872 | 0.658 | 2.015 | 0.824 | 0.778 | 0.606 | 0.959 |
| **stereo final** | 0.854 | 0.659 | 2.012 | 0.873 | 0.821 | 0.677 | **0.983** |
| OpenVINS mono | 1.409 | 2.058 | 1.621 | 0.982 | 1.625 | 1.750 | 1.574 |
| OpenVINS stereo | 1.775 | 2.913 | 1.117 | 0.729 | 1.262 | 0.868 | 1.444 |

Ensemble sd of a 6-room mean is 0.04-0.08 deg for orientation ATE, so its standard
error is 0.016-0.034 deg.

## Which milestone bought what

### M1 (`f3c0ec4`) -- publish the pose in the gravity-aligned frame. **-0.81 deg mono, -0.84 deg stereo.**

The whole gap to OpenVINS was an output frame-convention bug, not attitude drift.
XIVO's spatial frame `S` is the body frame of the *first IMU sample*, so it is
tilted by whatever the rig's attitude was at startup (0.8-3.0 deg on room1-6), and
nothing ever applied the filter's own gravity-direction state `Rsg` to the published
pose. Standard VIO evaluation aligns **yaw and position only** -- roll and pitch are
observable and must not be aligned away -- so that tilt landed in the reported
orientation error undiminished, whereas OpenVINS' global frame is gravity-aligned by
construction.

Fix: a new `Estimator::gwb()` / `gwc()` returning `Rsg' * (Rsb, Tsb)`, used by the
pybind bindings the eval harness reads and by `src/app/vio.cpp`. The filter is
untouched -- `gsb()` keeps its meaning on the estimation path -- and the estimate is
bit-identical, which `ate_002` (full SE(3) Horn alignment, blind to a global
rotation) confirms to four decimals in both modes. Gated on `gravity_align_output`,
default true.

Decomposing the error confirms the mechanism exactly: the tilt component went
1.503 -> 0.419 deg mono while the yaw component was **unchanged** (0.910 -> 0.908),
and the *constant* part of the tilt error -- which is what a frame offset is -- went
1.440 -> 0.088. Details in `m1-gravity-aligned-output.md`.

### M2 (`475fa89`) -- fix the 4-DoF gauge about gravity, not the group's body z-axis. **A correctness fix; a wash on the benchmark.**

`SwitchRefGroup`'s `group_degrees_fixed == 4` branch zeroed `dW(2)` of the elected
gauge group. Because `SO3xR3::operator+=` is `Rsb *= SO3::exp(dW)`, `dW` is a
*body-frame* perturbation, so that fixed rotation about the group's **body z-axis**
while the unobservable direction is rotation about **gravity**. Measured over the
groundtruth, the body z-axis is a median 7-17 deg from vertical (p90 20-41, max 74):
the rig is hand-held and pitched at the room and is never level. So only
`cos(angle)` of the yaw gauge was ever fixed. The fix projects out `u = Rsb' n_s`
instead of the coordinate `e3`, which is the same congruence `P <- M P M'` and
reduces to the old code when the rig is level.

Measured effect on orientation ATE: mono 1.013 -> 0.949 (-0.064, 1.6 sigma), stereo
0.959 -> 0.983 (+0.024, 0.6 sigma). Averaged over the two modes, -0.020 +- 0.028 --
**statistically a wash**. It does move the component it is supposed to move (mono
yaw 0.908 -> 0.832, tilt unchanged at 0.42), it improves `rpe_ori` in mono, and it
breaks no constraint. I kept it because the old axis is demonstrably the wrong one
and the new code is a strict generalization of the old, not because the benchmark
can resolve it. **If the merge is tight on mono position ATE, this commit is
self-contained and can be dropped** -- it is the +0.0029 m on mono `ate_002`.
Details, and the falsified half of my hypothesis, in `m2-gauge-axis.md`.

## The most useful finding, which is not a change

**0.43 deg of the remaining orientation ATE is a property of TUM-VI, not of any
estimator.** `oridecomp.py` splits the posyaw-aligned orientation error into yaw and
tilt (validated: it reproduces `ov_eval`'s `ov_ate_ori_deg` to four significant
figures on XIVO *and* on OpenVINS). Tilt RMS per sequence:

| seq | XIVO final mono | XIVO final stereo | OpenVINS mono | OpenVINS stereo |
|---|---|---|---|---|
| room1 | 0.323 | 0.323 | 0.348 | 0.351 |
| room2 | 0.469 | 0.469 | 0.480 | 0.486 |
| room3 | 0.509 | 0.503 | 0.541 | 0.535 |
| room4 | 0.448 | 0.450 | 0.452 | 0.438 |
| room5 | 0.352 | 0.342 | 0.400 | 0.407 |
| room6 | 0.416 | 0.421 | 0.411 | 0.375 |
| **mean** | **0.420** | **0.419** | **0.439** | **0.432** |

Two unrelated estimators agree on each sequence's tilt error to within 0.01-0.05
deg, and on the XIVO side the number is immune to all ten things I screened --
including injecting the *measured true* accelerometer bias, which was my best theory
for it. It is on the evaluation side of the interface: the mocap attitude
groundtruth, the marker-to-IMU rotation, or the mocap/IMU time sync. See
`tilt-floor-is-the-benchmark.md`.

Consequently the only addressable part of the error is yaw, and there XIVO is now
far ahead:

| yaw RMS [deg] | mono | stereo |
|---|---|---|
| XIVO baseline / after M1 | 0.910 / 0.908 | 0.840 / 0.838 |
| XIVO final | **0.832** | **0.867** |
| OpenVINS | 1.507 | 1.358 |

An orientation ATE of ~0.42 deg is the floor for anybody on this benchmark under
this protocol. XIVO is now at 2.3x the floor; OpenVINS is at 3.6x.

## What was tried and rejected

Ten screened candidates, none kept; full reasoning, numbers and mechanisms in
`negative-results.md`. The prompt's seven leads:

| lead | verdict |
|---|---|
| 1. no FEJ | not attempted -- requires `src/feature.cpp`, the position agent's file. The single biggest remaining opportunity. |
| 2. bias random walk 3.33x too small vs kalibr | rejected: ori 1.095, and `rpe_ori` 0.5360 breaks the 0.53 limit |
| 3. gravity 9.8 vs 9.80766 | rejected, and **9.8 is the better number** -- the accelerometer's effective gravity is 9.75, so 9.8 is nearer the data than 9.80766 is, and both 9.80766 and 9.75 cost +0.012 m of position ATE |
| 4. `P.Wsg` = 3.01 is enormous | rejected: 1.3 sigma on ori, adverse on position, and M1 already collected the payoff |
| 5. attitude/bias init from a static start | rejected, **and the premise is false** -- the rooms start at 6-18 deg/s, there is no static window |
| 6. integration-scheme consistency | checked, nothing wrong: state, transition and covariance share the RK stages, attitude is integrated on SO(3) |
| 7. gauge fixing / `td` / owner-change | the gauge half is M2. `td` cannot explain the residual (it would have to be 13 ms wrong). |

The single most informative rejected measurement: the true IMU biases on these
sequences are 8-15x (accel) and 2.5-22x (gyro) the filter's *entire* uncertainty
budget, so the bias states are pinned at zero for the whole run -- and **loosening
them makes everything worse, three different ways**, always failing on `rpe_ori`.
Without FEJ the covariance is over-confident, so extra state freedom is spent
absorbing visual residuals rather than the DC bias. That is why lead 1 is the
enabling change and not an alternative to leads 2-4.

## Residual risks

* **M2 is a wash on the benchmark.** Kept on mechanism, not on measurement. It costs
  +0.0029 m on mono `ate_002` (1.1 sigma). Droppable as one commit.
* **room3 stereo, 2.012 deg, is 2x every other cell** and did not respond to
  anything. Its yaw error is 1.95 deg against 0.43-1.18 elsewhere, and its measured
  gyro bias is 2.7e-3 rad/s -- 22x the filter's budget and 4-5x every other
  sequence. It is an IMU-calibration-consistency problem, and FEJ is the change that
  would let the filter absorb it.
* **The result rests on `Rsg` converging.** M1 levels the published pose with the
  filter's live gravity estimate. On room1-6 the final `Rsg` is within 0.05-0.40 deg
  of truth, but a sequence that never excites gravity observability would publish a
  worse-levelled pose than the old convention did in its first seconds. The old
  behaviour is one config key away (`gravity_align_output: false`).
* **Only TUM-VI rooms were measured.** Outside room1-6 the mocap covers 7-55% of
  poses, so ATE there is end-to-end drift rather than a trajectory statistic; none of
  these numbers should be read as applying to corridor/magistrale.
* **`ov_ate_pos_m` moved** (favourably, both modes) because M1 changes the published
  frame. That is the position agent's headline metric; flagged for the merge.

## Files in this branch's diff

| file | why | overlap risk |
|---|---|---|
| `src/estimator.h` | new `gwb()` / `gwc()` accessors, `gravity_align_output_` member | mine |
| `src/estimator.cpp` | read the new config key; M2's gauge projection in `SwitchRefGroup` | **also the position agent's file** -- but the two hunks are in the config-reading block and in `SwitchRefGroup`, far from anything visual |
| `pybind11/pyxivo.cpp` | `gsb`/`gsc` bindings publish `gwb`/`gwc`; this is what the eval harness reads | low |
| `src/app/vio.cpp` | text dump publishes `gwb` | low |

**No config file was changed.** `cfg/eff_mono.json` and `cfg/eff_stereo.json` are
byte-identical to `auto` @ 9e3ec06 -- see `config-delta.md`.
