# Plan: stereo + IMU support in XIVO

## Objective

Add genuine stereo visual-inertial odometry to XIVO (currently monocular-only)
and beat the shipped monocular baseline by a wide margin.

| metric (mean over room1-6) | mono baseline | exit criterion |
|---|---|---|
| ATE RMSE | 0.121 m | **< 0.06 m** |
| RPE rotational | 0.622 deg | **< 0.50 deg** |

Baseline source: `RESULTS.md`, config `sweep_dlt_nodesc`, seed 0.

## Why stereo should win here

Three specific, mechanical reasons — not just "more data":

1. **Scale becomes directly observable.** Monocular VIO recovers scale only
   through IMU excitation. On the TUM-VI *room* sequences the motion is
   handheld and smooth, so scale is weakly excited and drifts. A calibrated
   101 mm baseline pins metric scale from the very first frame.
2. **Depth initialization stops being a guess.** Today a new feature is born at
   `initial_z: 2.5 m` with `initial_std_z: 1.0` (`cfg/sweep_dlt_nodesc.json`),
   then refined by a depth subfilter needing `ready_steps: 2` observations plus
   parallax. Stereo triangulates true metric depth at first detection with a
   covariance derived from the actual geometry. Bad initial depth is the main
   driver of the outlier/rejection cascade in `MHGating`.
3. **Twice the constraints per instate feature.** Each feature contributes 4
   measurement rows (left + right) instead of 2, at no cost in state dimension.
   This directly tightens rotation, which is what RPE-rotational measures.

## Key architectural findings (from code survey)

Established by reading the source; these shape the design.

- `CameraManager` is a **hard singleton** (`src/camera_manager.cpp:6-13`), no
  teardown, with **34 global `Camera::instance()` call sites**. Because they are
  all *static* calls, widening to `instance(cam_id)` is the least invasive
  possible change.
- `Track` is a bare `std::vector<Vec2>` of observations with **no per-observation
  camera tag** (`src/feature.h:30`), and `FeatureAdj` maps **group id → a single
  `Vec2`** (`src/feature.h:21`). So right-camera pixels must NOT go through
  `Track`/`FeatureAdj` — they need a dedicated slot on `Feature`.
- The Jacobian chain in `Feature::ComputeJacobian` computes `cache_.dXcn_d*` for
  every state block, then applies `dxp_dXcn` last (`src/feature.cpp`). A right
  camera differs only by a **fixed** rigid transform, so
  `dXc1_d(state) = Rc1c0 * dXcn_d(state)` — **the entire existing chain is
  reusable with one extra 3x3 multiply.** This is the crux that makes stereo
  cheap here.
- Stereo extrinsics will be held **fixed, outside the EKF state**. TUM-VI ships
  factory calibration good to ~1e-3, and keeping them out avoids touching
  `Index`, `kMotionSize`, `kFullSize`, and the covariance layout in `core.h`.
- **cam0/cam1 timestamps are bit-identical** in all six sequences (verified by
  diffing the timestamp columns) — hardware-synchronized, so no temporal
  interpolation or pair-matching logic is required.
- Both cameras are `equidistant` (fisheye), intrinsics + `T_cn_cnm1` available
  in `data/tumvi/dataset-room*/dso/camchain.yaml`.
- `use_OOS: false` and `use_depth_opt: false` in the winning config, so the
  MSCKF/OOS and ceres depth-optimization paths are dormant and need no stereo
  work.

### Pre-existing bug found during the survey

`src/feature.cpp:688-689` in `Feature::FillJacobianBlock`:

```cpp
H.block<2, 3>(offset, goff) = J_.block<2, 3>(0, goff);
H.block<2, 3>(offset, goff) = J_.block<2, 3>(0, goff + 3);   // <-- same dest
```

The destination is `goff` both times. The reference-group **rotation** block is
overwritten by the **translation** block, and the translation block at `goff+3`
is never written at all. Every EKF update has been using a malformed
reference-group Jacobian. `OnePointRANSAC` reads `J_` directly so it was never
fatal, which is why this survived. Fixed in M0 and measured independently, since
it perturbs the mono baseline too.

## Design

```
        cam0 img ──┐
                   ├──> StereoTracker ──> Feature{ xp (left), xp_r (right) }
        cam1 img ──┘         │                        │
                             │ KLT L->R + epipolar    │
                             │ gate + circular check  │
                             ▼                        ▼
                    StereoTriangulate          4-row EKF update
                    (metric depth + cov)       [left 2 rows; right 2 rows]
```

- **Camera registry** — `CameraManager::Create(cfg, cam_id)` /
  `instance(cam_id)`; `instance()` stays == `instance(0)`. Zero changes at the
  34 existing call sites.
- **Stereo extrinsics** — new `StereoRig` holding `SE3 gc0c1` (and inverse),
  loaded from the config, mirroring `camchain.yaml`'s `T_cn_cnm1`.
- **Feature** — gains `Vec2 xp_r_`, `bool has_right_`, `number_t stereo_depth_`,
  and a right-camera Jacobian block `J_r_`.
- **Tracker** — `UpdateStereo(left, right)`: existing temporal KLT on left
  (unchanged, so mono behaviour is preserved bit-exactly when stereo is off),
  then left→right KLT, gated by epipolar residual and a right→left circular
  consistency check.
- **Depth init** — on first stereo observation, triangulate; set `x_(2)` to
  `log(z)` and shrink the depth variance. Falls back to the current
  `initial_z` path when no right match exists.
- **Measurement update** — instate features with a right match contribute 4
  rows; the assembly in `Estimator::FilterUpdate` and `MHGating` becomes
  variable-height per feature.

All stereo behaviour sits behind a config flag (`stereo: true`) so the mono path
stays byte-reproducible — that is the regression guarantee for every milestone.

## Milestones

Each milestone = build clean + tests pass + git commit. Notes accumulate in
`notes-n-prompts/notes-stereo/`.

| # | Milestone | Deliverable | Verification |
|---|---|---|---|
| **M0** | Baseline + Jacobian fix | Fix `FillJacobianBlock`; record baseline | Unit tests pass; mono ATE re-measured on 6 seqs, quantifying the fix in isolation |
| **M1** | Multi-camera foundation | `CameraManager` registry; `StereoRig`; stereo config + `camchain.yaml` converter | New unit test: both cameras load, project/unproject round-trip < 1e-9; **mono result byte-identical** |
| **M2** | Stereo data path | Stereo `DataLoader`, `VisualMeasStereo` through `pybind11`, python runner | Loads 2821 synced pairs for room1; right images reach the tracker |
| **M3** | Stereo tracking | `Tracker::UpdateStereo`, epipolar + circular gating | Unit test on real room1 frames: match rate, epipolar residual distribution; visual sanity dump |
| **M4** | Stereo depth init | Metric triangulation at first observation + covariance | Unit test vs analytic depth; **first end-to-end ATE** on room1 |
| **M5** | Stereo EKF update | 4-row measurement model, right Jacobian | Numerical-vs-analytic Jacobian test (mirrors existing `unittest_jacobians_instate`); full 6-seq eval |
| **M6** | Tuning | Sweep stereo noise/gating/feature-count; hit exit criteria | Full 6-seq eval per config; documented sweep table |
| **M7** | Report | `notes-n-prompts/report-stereo.md` | Final numbers vs exit criteria |

## Testing strategy

- **Regression invariant:** with `stereo: false`, output must stay byte-identical
  to the M0 baseline. Checked at every milestone — this is what makes it safe to
  refactor the camera singleton and the measurement assembly.
- **Unit tests** in `src/test/` beside the existing ones (`unittest_jacobians_*`,
  `unittest_camera*`). Stereo Jacobians get the same finite-difference treatment
  the mono Jacobians already have.
- **End-to-end** via `../run_eval.sh` (ATE + RPE, seed 0 for determinism).
- Known pre-existing failures (`Triangulation.Angular_Reprojection_Error`,
  `NumericalLinearAlgebra.SlowAndFastGivensMatch`) are baseline, not regressions.

## Risks

| Risk | Mitigation |
|---|---|
| Refactoring the camera singleton breaks mono | Byte-identical regression gate at every milestone |
| Wrong stereo extrinsics convention (`T_cn_cnm1` direction) | Validate by triangulating real matches and checking depths are positive and in the 0.05-5 m config range before wiring into the filter |
| Fisheye epipolar geometry is not a straight line | Gate on angular/reprojection residual after unprojection, not on raw pixel rows |
| 4-row update destabilizes MH gating | Per-feature MH distance uses the correct block size; gating thresholds swept in M6 |
| 2x tracking cost | Cost is acceptable (mono is ~30 s/seq); parallelize L/R only if it becomes a problem |
| Fixed extrinsics slightly wrong | Optional M6 experiment: inflate stereo measurement noise, or add extrinsics to state only if evidence demands |

## Deliverable

Branch `auto-stereo` in the xivo package, one commit per milestone, plus
`report-stereo.md` and `notes-stereo/` in the notes directory.
