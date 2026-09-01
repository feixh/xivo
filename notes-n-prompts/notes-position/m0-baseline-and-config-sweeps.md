# M0 -- baseline reproduction and the first round of config sweeps

## Baseline, reproduced in this worktree

`sweep.sh base_mono` (no keys changed) reproduces the mono half of
`experiments/results/xivo_ref_jitter` **bit-exactly**, so the worktree is a
faithful fork and the fast pass below is directly comparable to it.

| metric | mono 6-room mean |
|---|---|
| `ate_002` | **0.0928** |
| `ov_rpe8_pos_m` | **0.0480** |

Per-sequence mono `ate_002`: room1 0.0762, room2 0.1001, room3 0.1343,
room4 0.0805, room5 0.1065, room6 0.0594.

OpenVINS mono reference for context: `ate_002` **0.0621**
(`experiments/results/ov_accuracy/summary.md`). Target is 0.061.

## Measurement protocol used for every sweep below

`notes-n-prompts/notes-position/sweep.sh TAG KEY=VAL ...` -- mono only, all six
rooms, `--jitter 6` (36 runs), `CPU_BASE=64 CPU_SPAN=60`. That is exactly the
mono half of a full harness run, ~3 min wall. Configs are backed up and restored
by an `EXIT` trap, so no sweep can leave a dirty `cfg/`.

Per the task brief, sd of the 6-room mono mean is 0.0067 m, so a delta under
~0.005 m is not a result.

## Round 1: all five arms negative

The three leads handed to me (visual_meas_std inherited from stereo, `use_OOS`,
"switch to inverse depth") were each tested first. All three are dead ends.

| tag | keys | `ate_002` | `ov_rpe8_pos_m` | verdict |
|---|---|---|---|---|
| baseline | -- | 0.0928 | 0.0480 | -- |
| `vms10` | `visual_meas_std=1.0` | 0.0981 | 0.0478 | worse |
| `vms15` | `visual_meas_std=1.5` | 0.1023 | 0.0494 | worse |
| `oos_on` | `use_OOS=true` | 0.0964 | 0.0468 | worse on ATE |
| `tri_l1` | `triangulation.method=l1_angular` | 0.0982 | 0.0463 | worse |
| `depthopt` | `use_depth_opt=true` | **diverges** | -- | unusable |

### Lead 1 refuted: `visual_meas_std: 0.75` is not a stereo leftover

The *comment* above the key in `cfg/eff_mono.json` is indeed a copy-paste of the
stereo one, but the value is not: 0.75 is what the shipped
`cfg/tumvi_mono_ctl.json` has, and it measures better than both 1.0 (+0.0053)
and 1.5 (+0.0095). Monotone in the wrong direction, so the optimum is at or
below 0.75. (`0.5` is tested in round 2.)

### Lead 2 refuted: `use_OOS` costs ATE

+0.0036 m ATE for -0.0012 m RPE-8m -- i.e. it very slightly improves *local*
accuracy and slightly worsens global. Not within noise on ATE, and it is also
+347 MB steady RSS / -11% FPS (see the memory notes), so it is not worth
buying on a metric it does not move.

### Lead 3 was already done, and the two ways to redo it both lose

XIVO's feature state is **already** an anchored inverse-depth-like
parameterization: `Feature::x_` is `(X/Z, Y/Z, log Z)` in the *reference group's*
camera frame (`1/Z` under `USE_INVDEPTH`), documented in `src/feature.h`. There
is no XYZ representation to switch away from.

The two adjacent knobs both lose:
* `triangulation.method=l1_angular` (the code default; the shipped configs
  override it to `direct_linear_transform_svd`) is +0.0054 on ATE.
* `use_depth_opt=true` -- the multi-view Gauss-Newton `Feature::RefineDepth`
  path -- **diverges**: room1 ensemble members score ATE 13913.89 / 2384.24 /
  24863.13 m, and two runs (`mono/room1_r3`, `mono/room4_r1`) died on signal 6.
  It also crashes `score_openvins.py` (`ValueError: invalid literal for int()`
  on an empty `stats.txt` field from the aborted runs), so
  `experiments/results/position_depthopt` has no `summary.md`.
  This is a real HEAD bug, not a tuning miss; do not re-enable the flag as-is.

## What round 1 rules out, and what it points at

Nothing in the *measurement-noise / feature-representation* dimension is
mistuned. The remaining hypotheses are structural, and come from reading
OpenVINS' front end and SLAM-feature lifecycle:

1. **Depth window.** XIVO caps admissible depth at `max_depth: 5.0` (enforced in
   `Criteria::Candidate`/`CandidateStrict`, `src/options.cpp:12-31`) and
   `triangulation.zmax: 5.0`. OpenVINS accepts `min_dist 0.10` .. `max_dist 60`
   with a depth/baseline *ratio* gate instead. TUM-VI `room` is a 5-10 m space,
   so every far wall/ceiling point -- the low-parallax, high-leverage heading
   anchors -- is structurally rejected.
2. **Front-end quality.** OpenVINS runs `equalizeHist`, 5x5 grid FAST with a
   per-cell cap, `cornerSubPix` on every new detection, and a fisheye corner
   mask. XIVO has none of those.
3. **Outlier rejection model.** XIVO's `Tracker::OutlierRejection`
   (`src/tracker.cpp:932`) fits a **homography** on distorted pixels -- the wrong
   model for a room with real depth variation, since it rejects exactly the
   high-parallax features. OpenVINS fits a fundamental matrix on undistorted
   normalized coordinates. XIVO also ships with `do_outlier_rejection: false`.
4. **Promotion is too eager and rejection too final.** XIVO promotes after
   `subfilter.ready_steps: 2` (~3-4 frames); OpenVINS needs 2.0 s *and* >11
   observations. XIVO **destroys** a feature on a *single* MH failure at a 95%
   gate -- with 90 in-state features that is ~4-5 good features killed per frame
   by chance; OpenVINS marginalizes on the second strike.
5. **Depth is over-confident at promotion.** `Feature::SubfilterUpdate`'s
   innovation covariance `S = H P H' + Rtri I` omits *all* pose uncertainty
   (`src/feature.cpp:377`), and `Feature::FillCovarianceBlock` zeroes every
   cross-covariance row/column when the feature enters the state.

Confirmed *not* the gap: robust cost (neither system has one --
`Estimator::HuberOnInnovation` at `src/estimator.cpp:1610` is dead code, never
called) and the propagator (both are high-order).
