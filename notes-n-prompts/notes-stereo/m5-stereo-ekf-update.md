# M5 — the right camera in the EKF measurement update

Give every in-state feature two more measurement rows per frame, from its match
in the right image. M4 used the right observation *once*, to seed depth at
creation; M5 uses it *every frame*, as a real measurement.

Headline, and it is not the one I expected:

- **RPE_tra improves, robustly: 0.0208 → 0.0192–0.0201 m, in all 8 swept arms.**
- **ATE does not move above the noise floor.** Mean ATE@0.02 lands between
  0.0950 and 0.1114 m depending on knob settings, against 0.0998 for the same
  build with the update disabled. The spread *between* arms is larger than the
  gap to the control, so there is no ATE effect I can claim.
- This milestone also **changed which ATE number is the headline**, from
  `--max_difference 0.001` to `0.02`. §Which ATE explains why; short version,
  the 1 ms window silently excludes the first 23 s of every run.
- **RPE_rot is flat to three digits (0.6197 → 0.6189–0.6217).** My M4 note
  predicted this milestone would be the one to move it. That prediction was
  wrong, and §"Why RPE_rot did not move" below explains why — it is not that
  the update is weak, it is that RPE_rot is not vision-limited here.

Shipped setting: `R_scale = 1.0`, `mh_scale = 1.0`. Reasoning in §Tuning.

## What was built

`Feature::ComputeRightJacobian` (src/feature.cpp), called at the end of
`ComputeJacobian` whenever the feature has a right match and the rig is enabled.

The whole design rests on one observation. The rig extrinsics are fixed and live
*outside* the error state (documented at the top of `src/stereo.h`), so the
entire dependence of the right observation on the state flows through
`cache_.Xcn` — the same 3D point, in the current group's left-camera frame, that
the left rows were already linearized about. So the right rows reuse the left
camera's whole `dXcn_d*` chain verbatim; the two cameras differ only in the last
two links:

```
Xc1        = R_c1c0 Xcn + T_c1c0
dxp1_dXcn  = dxp1_dxc1 · dxc1_dXc1 · R_c1c0
```

and then the same `dXcn_d{Wsb,Tsb,Wbc,Tbc,Wsbr,Tsbr,x}` blocks are multiplied
through. Camera 1's *intrinsics* are not in the error state either (only
camera 0's would be, under `USE_ONLINE_CAMERA_CALIB`, which this build has
compiled out), which is consistent with holding the rig fixed.

Two ways the right rows are declined rather than computed:

- **no match this frame** — `Tracker::MatchStereo()` calls `ClearRightObs()` on
  every feature at the start of every stereo frame, so `has_right()` is a
  per-frame fact. `right_jac_valid_` is recomputed from scratch on every
  `ComputeJacobian`, so unlike `stereo_seeded_` it never persists.
- **the point is predicted behind camera 1** — a diverged pose or depth can do
  this while the tracker still matched something. `project` would divide by a
  negative depth and there is no meaningful linearization, so the feature
  contributes no right rows. Counted as `num_stereo_upd_rej_geom`.

The measurement height in `FilterUpdate` is therefore **data-dependent**: each
in-state feature contributes 2 rows, plus 2 more if it survived. The old fixed
`2 * i` stride is replaced by a running `row` cursor, with a `CHECK(row ==
total_size)` in debug builds.

### Why the right camera gets its own 2-dof gate

`GateStereoMeasurements()` (src/update.cpp) runs a separate Mahalanobis gate on
the right residual alone, rather than folding both cameras into one 4-dof
distance. Two reasons:

1. `MH_thresh_` defaults to 5.991, the 95% quantile of chi-squared with **2**
   degrees of freedom. Folding a second camera in would silently change the
   effective gate without changing the number.
2. A bad left→right *match* and a bad left *track* are independent failures. A
   wrong right match should cost the feature its two right rows, not its place
   in the state. `InvalidateRightJacobian()` does exactly that.

It is called from `FilterUpdate()` rather than from `OutlierRejection()` so that
it operates on exactly `in_current_ekf_update_` and **cannot be bypassed** by
the configurations that skip `MHGating` or `OnePointRANSAC` (this build has
`use_1pt_RANSAC: false`). `MHGating`, `OnePointRANSAC` and `HuberOnInnovation`
are deliberately left left-camera-only.

## The regression gate held

`stereo_update.enable = false` reproduces the recorded M4 numbers **exactly**,
all six digits, in all six rooms:

```
room1 0.104566   room2 0.062660   room3 0.100661
room4 0.057737   room5 0.107785   room6 0.047389
```

This mattered more than usual, because `ComputeRightJacobian` runs whenever a
right match exists, regardless of the flag. `GateStereoMeasurements` therefore
has to *explicitly disown* every feature's right rows when the flag is off, and
this is the test that the disowning is complete.

Also ruled out by tracing frame order: M4's seed and M5's update never consume
the same right observation twice. Seeding happens in
`InitializeJustCreatedTracks` at the end of frame *t*; `ClearRightObs()` runs at
the start of frame *t+1*, so the update at *t+1* uses a fresh match.

## Tests

`src/test/unittest_jacobians_stereo.cpp`, 15 tests, a separate binary from
`unitTests_Jacobians` because `CameraManager`'s registry is process-wide and
this one needs the real TUM-VI fisheye pair in slots 0/1 while that one installs
a perfect pinhole in slot 0.

Seven of them central-difference `J_r_` against an **independently written**
`ComputeRightPixel()` — written out from the nominal states rather than by
calling into `cache_`, so its numbers do not come from the code under test.
The rest pin down the parts a finite difference cannot see:
`AllOtherColumnsAreZero` (no stray write into another group's or feature's
slot), `NoBlockIsAccidentallyZero` (a block that is zero in both analytic and
numeric would pass silently — which is exactly the failure mode of forgetting
to fill one), `RightRowsAreNotTheLeftRows`, and the two decline paths.

`FillJacobianBlockCopiesEveryLiveBlock` is the coverage test deferred from M0,
where `FillJacobianBlock` wrote the reference group's translation block over its
rotation block — a bug invisible to any test that only inspected `J_`. It now
checks both cameras' row pairs.

**Verified the tests have teeth by ablation**, not by assuming:

| ablation | tests that fail |
|---|---|
| drop `* rig.Rc1c0()` from the chain | 7 finite-difference tests |
| project with camera 0's intrinsics | 8 |
| omit the `foff` (feature state) block | `FeatureState`, `NoBlockIsAccidentallyZero` |

Two of my own test bugs, worth recording because both were *geometry* errors,
not coding errors:

- `PointBehindTheRightCameraContributesNoRows` first tried `x_(2) = log(0.02)`,
  on the assumption that a very shallow depth puts the point behind camera 1.
  **It does not** — the baseline is 99.9% lateral, so any point in front of
  camera 0 is essentially always in front of camera 1. The test now moves the
  current body pose 10 m along the *reference* camera's optical axis,
  overshooting a 7.4 m point, and `ASSERT_LT(Xc1(2), 0.0)` checks the fixture
  itself rather than trusting it.
- The fixture originally drew two independent random poses. With a 190 px
  fisheye that puts the predicted point behind a camera most of the time, so
  there was nothing to differentiate. It now uses a *small* relative motion,
  which is also what 20 Hz actually looks like.

Full suite: only the two failures that predate M0
(`NumericalLinearAlgebra.SlowAndFastGivensMatch`,
`Triangulation.Angular_Reprojection_Error`).

## End-to-end sweep

Seed 0, ASLR off (`setarch -R`), all six rooms. Both ATE protocols tracked, per
the policy adopted in M4. Columns are means over the six rooms.

| arm | ATE @0.001 | ATE @0.02 | RPE_tra | RPE_rot |
|---|---|---|---|---|
| `enable=false` (= M4) | **0.0801** | **0.0998** | 0.02083 | 0.6197 |
| `R_scale=0.5` | 0.0739 | 0.0950 | 0.01937 | 0.6217 |
| `R_scale=1`   | 0.0760 | 0.1013 | 0.02010 | 0.6211 |
| `R_scale=2`   | 0.0886 | 0.1114 | 0.01999 | 0.6205 |
| `R_scale=4`   | 0.0778 | 0.0986 | 0.01987 | 0.6196 |
| `R_scale=8`   | 0.0805 | 0.0971 | 0.01928 | 0.6200 |
| `mh_scale=0.1` | 0.0785 | 0.0998 | 0.01948 | 0.6189 |
| `mh_scale=0.3` | 0.0787 | 0.0976 | 0.01916 | 0.6203 |
| `mh_scale=3`   | 0.0842 | 0.1069 | 0.01979 | 0.6210 |

Read this honestly:

- **RPE_tra is the one real effect.** Every one of the eight stereo-on arms
  beats the control, across a 16× range of `R_scale` and a 30× range of
  `mh_scale`. An effect that survives that much knob variation is not noise.
- **ATE is not resolvable.** `R_scale` 1 → 2 moves mean ATE@0.02 by 0.010 —
  twice the gap between the best arm and the control. Whatever the update does
  to ATE is smaller than the filter's sensitivity to how the update is weighted.
- **RPE_rot does not care at all.** Range across all nine rows: 0.6189–0.6217.
- The two ATE protocols disagree in sign on the `R_scale=1` arm (0.0801 →
  0.0760 improving, 0.0998 → 0.1013 worsening). That disagreement is what
  forced §Which ATE, and it turned out to be a property of the *metric*.

### Which ATE — the 1 ms window is not a random subsample

M4 adopted `--max_difference 0.001` as the headline because that is what
`run_and_eval_pyxivo.py` passes, and noted only that it associates ~25% of
frames. That undersells the problem. Measured on room1:

```
tol 0.001: 720 of 2818 frames matched, spanning 23.0 s .. 140.9 s, in 76 contiguous blocks
tol 0.020: 2771 of 2818 frames matched, spanning  0.0 s .. 140.9 s
image period 50.158 ms   GT period 8.333 ms   ratio 6.0192
```

The period ratio is 6.019, not something incommensurate, so the image-to-GT
phase offset **drifts slowly** rather than scattering. The matched frames
therefore arrive in long contiguous runs, and on room1 the phase does not come
within 1 ms until **23 seconds in** — so the entire initialization phase is
excluded from the score. That is precisely the part of the run with the largest
error (§Where the ATE actually is: first-decile aligned error 0.11–0.26 m
against ~0.06 m mid-run). A metric that omits the worst 16% of the run and
90% of the rest will disagree in sign with one that does not.

**So the headline moves to `--max_difference 0.02`** — 98% of frames, the whole
span — with `0.001` still reported alongside for comparability with the README's
monocular baseline and the < 0.06 m exit criterion. Neither is discarded, but
they are no longer treated as equally trustworthy, and the earlier
"accept only if both improve" rule is replaced by "0.02 decides, 0.001 is
reported".

### The noise floor is not the seed

Ran `R_scale=1` on all six rooms at `XIVO_RANDOM_SEED` 0, 1, 2, 3:

| seq | seed 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| room1 | 0.079327 | 0.087491 | 0.093194 | 0.079905 |
| room3 | 0.097552 | 0.096121 | 0.097356 | 0.098789 |
| room2/4/5/6 | — | *byte-identical to seed 0* | | |

(ATE@0.001.) Four of six rooms do not depend on the seed at all — with
`use_1pt_RANSAC: false` the RNG has almost nothing to do. 6-room mean across the
four seeds: 0.0760 / 0.0771 / 0.0783 / 0.0763, **std ≈ 0.001 m**; at 0.02,
0.1013 / 0.1001 / 0.1023 / 0.0998, again ≈ 0.001.

That is a much *tighter* floor than the ±0.005 I had been quoting — and it is
the wrong floor to use. The `R_scale` sweep varies a knob whose true value is 1
by symmetry (two cameras of the same model, same tracker), yet moves the 6-room
mean over 0.0950–0.1114, **std ≈ 0.006, and non-monotonically**. The filter is
a deterministic but chaotic function of its configuration. Sizing an error bar
from the seed would make a 0.005 m difference look like 5σ when it is under 1σ
of the uncertainty that actually matters for a tuning decision.

Rule adopted for M6: **size the error bar from replicates that perturb something
the answer should not depend on**, and accept a change only if it exceeds that
spread or holds across a whole family of settings. M5's RPE_tra result passes
that test (8/8 arms); its ATE result does not.

### Tuning decision

Ship `R_scale = 1.0`, `mh_scale = 1.0`, i.e. treat a right pixel exactly like a
left pixel and gate it at the same quantile.

`R_scale = 0.5` is nominally the best arm on both protocols, and I am not
shipping it. It would mean asserting the right camera's pixels are *twice as
trustworthy* as the left camera's, for two cameras of the same model, same
resolution, same distortion family, tracked by the same KLT. There is no
mechanism for that; its margin over `R_scale = 1` is 0.006 on ATE@0.02, i.e.
about 1σ of the config-perturbation spread that the same sweep measures; and
`R_scale = 2` and `4` bracket it in the wrong order, which is what fitting noise
looks like. Revisit in M6, where `sigma_px`, `visual_meas_std` and `R_scale`
should be swept *jointly*.

### The gate barely fires

room1: 78248 right measurements used of 78781 offered — geom=189, mh=344, i.e.
**0.44% rejected** by a gate set at the 95% quantile of a 2-dof chi-squared,
where a correctly calibrated filter would reject ~5%. So the predicted
innovation covariance `S = Jr P Jr' + R` is substantially larger than the
residuals actually are: the noise model is conservative. Tightening the gate
(`mh_scale=0.1`, a 10× tighter threshold) still changes nothing above noise,
which says the *rejections* are not where the error is.

## Why RPE_rot did not move

My M4 note asserted: *"rotation is constrained by where features appear across
the field, not by how well their depth is known, so only a continuous
second-camera measurement can move it."* The second-camera measurement is now
there and rotation did not move. Four hypotheses, tested in order.

**1. A time offset between the estimate and ground truth.** Swept
`evaluate_rpe.py --offset` from −30 ms to +30 ms on room1. RPE_rot is 0.532 at
offset 0 and rises steeply either side (0.745 at +5 ms, 1.041 at −5 ms). The
timing is already at its optimum. *Dead.* Note the slope though: ~100 deg of
RPE_rot per second of timing error, which sets up hypothesis 2.

**2. The metric's own quantization floor.** The estimate is at image stamps
(20 Hz); ground truth is 120 Hz and associated by *nearest neighbour*, so the
tool charges the estimator for `ω · (δ₁ − δ₀)` with each `δ` up to half a GT
interval (4.17 ms). Built the trajectory a *perfect* estimator would output —
ground truth lerp'd/slerp'd to the image stamps — and scored it with the
shipped command:

| seq | RPE_rot of a perfect estimate | RPE_tra |
|---|---|---|
| room1 | 0.2847 deg | 0.0038 m |
| room2 | 0.2818 | 0.0043 |
| room3 | 0.3039 | 0.0049 |
| room6 | 0.2095 | — |

**So ~0.28 deg of the 0.62 deg is the metric, not the estimator.** In quadrature
that leaves ~0.55 deg of real rotation error, and the 0.5 deg exit criterion
corresponds to a true error of ~0.41 deg. Real, but a 25% reduction, not a 20%
one. *Partially explains, does not dismiss.*

**3. An unmodeled body-frame offset between the estimate and ground truth.**
The eval-mode saver writes `gsb`, the IMU body pose; ground truth is
`mav0/mocap0/data.csv`, and TUM-VI ships no marker extrinsic. If the mocap
frame were a different rigid frame on the rig, RPE_rot would be inflated by a
*conjugation* (`X' A' X A`, whose angle is nonzero) and ATE by a
rotation-dependent position offset that Horn alignment cannot absorb. Solved for
the best-fit `g_bm` per sequence (Kabsch on relative-rotation axes, then linear
least squares on positions):

| seq | rotation of `g_bm` | translation of `g_bm` |
|---|---|---|
| room1 | 0.036 deg | (0.017, −0.034, 0.010) |
| room2 | 0.073 | (0.025, 0.001, −0.036) |
| room3 | 0.096 | (−0.038, −0.005, 0.007) |
| room4 | 0.047 | (0.028, −0.023, 0.008) |
| room5 | 0.049 | (0.005, −0.021, −0.002) |
| room6 | 0.170 | (0.005, 0.017, 0.043) |

The rotation is under 0.2 deg everywhere — the mocap data is already in the IMU
frame — and applying it changes RPE_rot by 0.001 deg. The 4 cm translations
point in *inconsistent directions* across the six sequences, which a real rigid
extrinsic of one rig cannot do; they are the fit absorbing estimator drift.
Applying them makes RPE_tra twice as bad and ATE no better. *Dead, cleanly.*

**4. Gyroscope scale / misalignment.** Binned the per-pair rotation error by how
much the rig actually rotated over that 1 s (room1, and the same binning of the
perfect-estimate floor for comparison):

| \|GT rel. rotation\| | our err (rmse) | floor | err/rot |
|---|---|---|---|
| 0–10 deg | 0.287 | 0.129 | 0.059 |
| 10–20 | 0.417 | 0.215 | 0.027 |
| 20–40 | 0.497 | 0.276 | 0.016 |
| 40–60 | 0.543 | 0.279 | 0.011 |
| 60–90 | 0.576 | 0.313 | 0.008 |
| 90–130 | 0.513 | 0.295 | 0.005 |

Best-fit proportional model: `err ≈ 0.63% of rotation` for us, `0.34%` for the
floor, so ~0.5% excess — consistent with a gyro scale/misalignment error of
that size. But the error is **not** purely proportional: the 0–10 deg bin still
shows 0.287 deg against a floor of 0.129, i.e. ~0.26 deg of
rotation-*independent* error. So RPE_rot decomposes roughly as

```
0.28 deg  metric quantization  (cannot be reduced)
0.26 deg  rotation-independent attitude jitter
~0.5%·|rot| gyro scale/misalignment  (Cg, which this build does not estimate)
```

*Confirmed as the shape of the problem, and it is entirely an IMU/attitude
problem — which is why a second camera did nothing for it.*

The lever this hands M6 is concrete: `USE_ONLINE_IMU_CALIB` is commented out in
`src/CMakeLists.txt`, so `Cg` (gyro scale and misalignment) is not estimated at
all. Note it cannot be enabled alone — in `feature.cpp` the `Cg` Jacobian block
is nested inside `#ifdef USE_ONLINE_TEMPORAL_CALIB` — so M6 would enable both,
and would need a monocular baseline rebuilt with the same flags to keep the
README comparison fair.

## Where the ATE actually is

Two more diagnostics, since M5 failed to move ATE and M6 has to.

**It is not a scale error.** Aligning with Sim3 instead of SE3 changes nothing
(room1: 0.1412 → 0.1412) and the recovered scale is within ±2% and of
*inconsistent sign* across rooms (+0.5%, +0.2%, −0.8%, +2.2%, −0.3%, −0.8%).

**It is not localized excursions.** Split room1's aligned error into deciles of
time:

```
enable=false  0.161 0.164 0.186 0.136 0.063 0.094 0.109 0.097 0.130 0.203
R_scale=1     0.114 0.141 0.180 0.090 0.067 0.070 0.083 0.094 0.102 0.136
R_scale=2     0.260 0.265 0.250 0.156 0.057 0.091 0.158 0.190 0.212 0.276
R_scale=8     0.093 0.108 0.142 0.079 0.053 0.059 0.059 0.084 0.099 0.133
```

The worst 10% of samples carry only 20–27% of the sum of squares, and excising
them barely moves the RMSE — the error is spread out, not spiky. The shape is
"high at both ends, low in the middle", which after a global Horn fit is what a
*bent* trajectory looks like, not what accumulating drift looks like.

And note **where the arms differ: in the first decile** (0.093 to 0.260 across
arms whose only difference is how much a right pixel is trusted). The divergence
between arms originates in the initialization phase and then persists. That is
the most actionable thing in this note: **M6 should look at initialization
first**, and stereo gives it something monocular did not have — metric depth
from the very first frame.

It also closes the loop on §Which ATE. The first three deciles of room1 are
0–42 s; the 1 ms protocol starts scoring at 23 s. So the metric M4 chose as its
headline is *blind to most of the region where the arms differ*, which is
exactly why it and the dense protocol disagreed in sign here.

## Distance to target

Exit criteria are mean ATE < 0.06 m and mean RPE_rot < 0.5 deg over room1–6.

| | now (M5, shipped setting) | target |
|---|---|---|
| mean ATE @0.02 | 0.1013 | < 0.06 |
| mean ATE @0.001 | 0.0760 | < 0.06 |
| mean RPE_rot | 0.6211 | < 0.5 |
| mean RPE_tra | 0.0201 | — |

Neither criterion is met, and — unlike after M4, where I expected this milestone
to close the ATE gap — M5 gives no reason to think more measurement information
will. The two leads that this milestone actually produced are:

1. **Initialization.** The error is present from the first seconds, is not
   drift, is not scale, and is where the arms diverge. Stereo hands M6 metric
   depth at frame one, which monocular initialization could not use.
2. **`Cg`, the gyro scale/misalignment matrix, is not estimated in this build.**
   ~0.5% of the rotation error is proportional to rotation, which is that
   signature. Enabling it requires `USE_ONLINE_IMU_CALIB` *and*
   `USE_ONLINE_TEMPORAL_CALIB` (the `Cg` Jacobian block is nested inside the
   latter's `#ifdef`), plus a monocular baseline rebuilt with the same flags so
   the README comparison stays honest.

Of the 0.6211 deg RPE_rot, 0.28 deg is the metric's own quantization and cannot
be removed; the 0.5 deg criterion corresponds to a *true* rotation error of
about 0.41 deg against today's ~0.55 deg.

## Reproducibility

Re-ran `R_scale=1` and `R_scale=2` on room1 after the whole sweep: identical to
six digits (0.079327 / 0.141899). So the 0.079-vs-0.142 gap between those two
arms is a genuine deterministic sensitivity of the filter to a 2× change in one
measurement variance, not a build or scheduling artifact. Worth stating plainly:
a filter this sensitive to that knob is a filter whose ATE is dominated by
something other than measurement information.

Note also that another job was running monocular sweeps in the same working
tree during this milestone. `lib/pyxivo.*.so` was checksummed and snapshotted,
and the reruns above confirm the binary did not change under the sweep.

## Files

```
src/feature.h                        Jr(), inn_r(), right_jac_valid(),
                                     InvalidateRightJacobian(),
                                     FillRightJacobianBlock()
src/feature.cpp                      ComputeRightJacobian();
                                     FillJacobianBlock refactored onto
                                     FillJacobianBlockFrom()
src/update.cpp                       GateStereoMeasurements();
                                     variable-height H_/inn_/diagR_ assembly
src/estimator.h / .cpp               stereo_update{enable,R_scale,mh_scale},
                                     three counters, config validation
src/test/unittest_jacobians_stereo.cpp   15 tests
src/CMakeLists.txt                   unitTests_jacobians_stereo target
pybind11/pyxivo.cpp                  three counter bindings
scripts/pyxivo.py                    print_stereo_stats reports them
scripts/make_stereo_cfg.py           emits the stereo_update block
cfg/tumvi_stereo.json                regenerated
```
