# Monocular position accuracy on TUM-VI room1-6 — final report

Branch `auto-position`, forked from `auto` @ `9e3ec06`. Five commits.

Evidence standard throughout: full `--jitter 6` ensembles (6 sequences x 6
perturbed replicates = 36 runs per mode) via
`experiments/openvins/run_xivo_reference.sh`, `CPU_BASE=64 CPU_SPAN=60`. Headline
metric `ate_002`; the 6-room mean has an sd of ~0.0067 m (mono) / 0.0045 m
(stereo), so nothing under ~0.005 m is treated as a result.

## Targets

| target | required | achieved | |
|---|---|---|---|
| mono position ATE (`ate_002`) | <= 0.061 | **0.0563** | **met** |
| mono RPE 8 m position | <= 0.030 | **0.0271** | **met** |

## Non-regression constraints

| constraint | limit | achieved | |
|---|---|---|---|
| stereo position ATE (`ate_002`) | <= 0.064 | **0.0492** | met, improved 23% |
| RPE 8 m rotation, mono | <= 0.53 deg | **0.505** | met |
| RPE 8 m rotation, stereo | <= 0.53 deg | **0.508** | met |
| orientation ATE, mono | <= 1.85 deg | **1.756** | met |
| orientation ATE, stereo | <= 1.85 deg | **1.754** | met |
| no sequence diverges | -- | worst of 72 runs 0.0973 m | met |

## Before / after -- all five metrics, both modes, 6-room means

Before = the pristine `cfg/eff_*.json` on this same tree
(`experiments/results/position_base_full`). After =
`experiments/results/position_final`. Both are full both-modes `--jitter 6` runs.

| metric | mono before | mono after | stereo before | stereo after |
|---|---|---|---|---|
| ATE position, `ate_002` [m] | 0.0935 | **0.0563** (-40%) | 0.0637 | **0.0492** (-23%) |
| ATE position, `ov_ate_pos_m` [m] | 0.0974 | 0.0638 | 0.0689 | 0.0568 |
| ATE orientation, ov_eval [deg] | 1.829 | 1.756 | 1.796 | 1.754 |
| RPE 8 m position [m] | 0.0481 | **0.0271** (-44%) | 0.0292 | **0.0217** (-26%) |
| RPE 8 m rotation [deg] | 0.515 | 0.505 | 0.508 | 0.508 |

Orientation improves slightly in both modes even though nothing in the
orientation model was touched -- a better-conditioned visual update leaves less
error for the attitude to absorb.

### Per-sequence mono position ATE (`ate_002`, m)

| | room1 | room2 | room3 | room4 | room5 | room6 | mean |
|---|---|---|---|---|---|---|---|
| before | 0.0762 | 0.1001 | 0.1381 | 0.0805 | 0.1065 | 0.0594 | 0.0935 |
| after | **0.0578** | **0.0467** | **0.0628** | **0.0393** | **0.0949** | **0.0363** | **0.0563** |
| delta | -0.018 | -0.053 | -0.075 | -0.041 | -0.012 | -0.023 | -0.037 |

Every sequence improves. The two that were worst (room3 0.138, room2 0.100) gain
the most, and the spread across sequences collapses from 0.079 m to 0.059 m -- the
result is not one sequence carrying the mean.

Per-sequence mono RPE 8 m after: 0.0218 / 0.0188 / 0.0318 / 0.0273 / 0.0343 /
0.0283.

room5 is the one sequence still clearly above the rest (0.0949). It is also the
only one whose ATE the front-end changes barely moved, and it was already the
outlier at baseline relative to its RPE -- consistent with an association or GT
issue on that sequence rather than a filtering one, but not proven.

## Which milestone bought what

Mono `ate_002` / RPE-8m position, on the base each was measured against.

| milestone | change | ate_002 | rpe8 | worth |
|---|---|---|---|---|
| baseline | pristine config | 0.0928 | 0.0480 | -- |
| M1 | `use_OOS` + tuned `OOS` block (`pose_window` 20 is the key) | 0.0875 | 0.0418 | -0.0053 |
| M2 | negative results only (see below) | -- | -- | 0 |
| M4a | `tracker_cfg.histogram_method=CLAHE` (+ `grayscale`) | 0.0717 | 0.0354 | **-0.0158** |
| M4b | `subpix_refine`, `augment_every`, `min_observations`, `oos_meas_std` stacked | 0.0605 | 0.0326 | -0.0112 |
| M3 | FEJ | 0.0845 on the M1 base | 0.0419 | 0 (noise) |
| M5 | `consistent_init.enable` | **0.0563** | **0.0271** | -0.0042 (-0.0138 on the M1 base) |
| M6 | OOS scratch-buffer split | 0.0563 | 0.0271 | 0, but -302 MB RSS |

Two of these are the substance.

**CLAHE (-0.0158) is a measurement-supply fix.** TUM-VI's fisheye frames measure
mean intensity 62.9 inside r < 80 px of the principal point and 33.2 outside
r > 270 px (measured, 40 frames of room1 cam0, post 16->8 bit), while FAST uses a
single global threshold of 20. Detection was therefore roughly twice as strict at
the edge of the image as at the centre -- and the edge of the image is where the
parallax is. `grayscale` alone is inert (+0.0002), which is the control that makes
this attributable to the equalization rather than to finally running the front end
on one channel instead of the BGR that `cv::imread` was handing it.

**Consistent initialization (-0.0042 here, -0.0138 on the OOS base) is a
covariance-consistency fix.** `FillCovarianceBlock` imported the depth
sub-filter's 3x3 -- a covariance *conditional* on an exactly-known pose, since
`SubfilterUpdate` has no `Hx P Hx'` term -- and zeroed every cross-covariance,
asserting the feature was independent of the group it was anchored to. Replaced by
the standard delayed-initialization augmentation
(`P_ff = Hl^-1 (sigma^2 I + Hx P Hx') Hl^-T`, `P_xf = -P Hx' Hl^-T`).

## Negative results, in one place

Each of these is a full `--jitter 6` mono ensemble, not a single run.

| tried | result | note |
|---|---|---|
| **FEJ** (`fej.mode` 1 and 2, `fej.oos`) | **0.0845 vs 0.0875; 0.0562 vs 0.0563 on the final base** | Implemented, correct, byte-identical when off, all unit tests pass -- and inside the noise floor. `SwitchRefGroup` already zeroes the gauge group's 4 unobservable DOF, and `max_group_lifetime` is 60 frames, so anchors barely age. `m3-fej.md`. |
| `MH_thresh=9.21` (99% gate) | +0.0031 without M5, 0.0000 with it | M5 removes the *penalty* for a looser gate but yields no gain. |
| `MH_max_strikes` 3 (defer instead of destroy) | negative | `m2`. |
| `tracker_cfg.epipolar_rejection` (RANSAC F-matrix, after OpenVINS) | +0.0030 on `combo`, +0.0049 on the final base | Removes the same correspondences CLAHE already fixes. |
| `tracker_cfg.max_theta_deg` (radial FOV mask) | negative | `m2`. |
| `num_features_min/max` 180/240 | +0.0047 | The census says 76-85 of 90 state slots are occupied; the filter was short of *good* features, not of features. |
| `histogram_method=HISTOGRAM` | 0.0799 vs CLAHE's 0.0717 | OpenVINS's own TUM-VI config picks HISTOGRAM; on XIVO CLAHE is 0.008 m better. |
| `clahe_clip_limit=4` | 0.0623 vs 0.0605 | The OpenVINS default of 10 is right here. |
| `consistent_init.min_views=3` / `meas_std=1.5` / `max_var=100` | all within noise | All three guards stay at their defaults. |
| `subfilter.ready_steps=1` | -0.0045 on the M1 base, +0.0005 on the final one | A coin flip once the rest is in; left at 2. |
| `OOS.pose_window=30`, `OOS.max_observations=30` | 0.0602, 0.0634 vs 0.0605 | 20/15 is not a knife edge on one side and is better on the other. |
| `use_depth_opt`, `triangulation.method=l1_angular`, `visual_meas_std` 0.5/1.0/1.5, `num_gauge_xy_features=0`, in-state depth caps | all within noise or worse | `m0`. |
| **`consistent_init` alone, without the OOS block** | **+0.0057** | The one interaction in the whole delta. See below. |

## Residual risks

1. **`consistent_init` and the `OOS` block must merge together.** Enabled without
   the out-of-state window, consistent initialization is *worse* than the shipped
   `FillCovarianceBlock` (0.0985 vs 0.0928). An honest, larger initial covariance
   lets the filter correct a new feature more, which only pays if there are enough
   good measurements to do the correcting. Splitting these two keys across a merge
   would silently give back more than the feature is worth. Recorded at the top of
   `config-delta.md` and in the config comments themselves.
2. **The gain is concentrated in one front-end knob.** CLAHE is 43% of the total
   ATE improvement. It is a contrast normalization tuned to nothing (OpenVINS's own
   parameters) and the mechanism is measured, not inferred -- but it is a *dataset*
   property. On a well-exposed, non-fisheye dataset the -0.0158 will not be there,
   and the rest of the stack (-0.021) is what carries over.
3. **room5 is unexplained.** 0.0949 m against a 0.0563 mean, and it is the sequence
   the front-end changes moved least. Worth a look before this is called done on
   TUM-VI, but it is not a regression: it improved (0.1065 -> 0.0949).
4. **`OOS.pose_window: 20` is a capacity trade.** The window occupies group slots
   (`EKF_MAX_GROUPS` 45) that would otherwise hold anchors. It measures fine at 20
   and 30 on room1-6, which are 100 s indoor loops; on a long outdoor sequence the
   group budget is tighter and this is the first knob I would re-measure.
5. **FEJ ships off.** It is correct, tested, and free when off, and it is the right
   thing to reach for on a longer sequence than a 100 s mocap room -- where
   `SwitchRefGroup`'s explicit gauge fixing does less and anchors are relatively
   older. It is *not* validated as an improvement anywhere, so turning it on is a
   new experiment, not a safe default.
6. **Throughput was not measured properly** -- that is the third agent's job and I
   ran no `--timing` pass. The contended FPS column of the `--jitter 6` runs is
   52.5 mono / 28.0 stereo after vs 70.8 / 35.2 before, i.e. of order -25%, almost
   all of it the OOS update and CLAHE. Treat the magnitude as indicative only.
   Peak RSS *is* reliable and is +3.3 MB mono / +33.9 MB stereo (141.2 / 177.2 vs
   137.9 / 143.3) after M6; before M6 the same config cost +306 MB.

## Commits

| | |
|---|---|
| `3313951` | Front-end and gating knobs the tuning needed, all default-off |
| `2358016` | First-estimates Jacobians for groups, features and the out-of-state rows |
| `a4aa1ec` | Consistent feature initialization: mono position ATE 0.0605 -> 0.0563 |
| `e6c51ca` | Stop giving every pooled feature its own OOS scratch buffer: -302 MB |
| `193f7eb` | Ship the tuned mono/stereo config: mono position ATE 0.0928 -> 0.0563 |

The first four are code and are no-ops on any config that does not opt in; every
number above comes from the fifth. Merging the code without the config is
bit-identical to `auto` -- verified by `cmp` on dumped trajectories.

## Notes in this directory

* `m0-baseline-and-config-sweeps.md` -- the baseline, the noise floor, and the
  first round of config-only sweeps.
* `m1-oos-pose-window.md` -- why the MSCKF path was inert and what turning it on
  is worth.
* `m2-frontend-and-gating.md` -- the BGR discovery, the front-end knobs, the MH
  gate experiments.
* `m3-fej.md` -- FEJ: implemented, verified, neutral, and why.
* `m4-oos-and-frontend-stack.md` -- the one-knob-at-a-time table and the stack; the
  measured vignetting numbers.
* `m5-consistent-init.md` -- the covariance bug, the fix, the attribution arms.
* `m6-oos-buffer-memory.md` -- the 302 MB column-major page-touching bug.
* `config-delta.md` -- every changed key, old -> new, why, plus the keys
  deliberately left alone. **Read this before merging the configs.**
* `sweep.sh`, `patch_cfg.py` -- the sweep driver. Note `patch_cfg.py` strips JSON
  comments, so it must not be used on a config that is being kept.
