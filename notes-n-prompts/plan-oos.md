# Plan — out-of-state (MSCKF) update for XIVO

Target: `xivo` branch `auto-oos`, developed in the worktree
`/home/ubuntu/workspace/auto-slam-engineer/xivo-oos` (created from `auto`).

Exit criteria (monocular + IMU, TUM-VI room1-6, seed 0):
mean ATE < 0.06 m and mean rotational RPE < 0.5 deg.

## 1. What exists today

`use_OOS` is a config knob, but `src/estimator.cpp:120` does

```cpp
if (use_OOS_) { LOG(FATAL) << "MSCKF not implemented"; }
```

The MSCKF plumbing was deliberately deleted in commit `48ffac8` ("remove MSCKF
logic", Jul 2022). What survived that deletion:

| piece | file | state |
|---|---|---|
| `Feature::ComputeOOSJacobian` / `...Internal` | `src/oos.cpp` | present; per-observation Jacobians are correct (numerically verified by `unittest_jacobians_oos.cpp`, 6/6 pass) |
| left-nullspace projection | `src/helpers.cpp` `SlowGivens` / `Givens` | present but **wrong as called** (see below) |
| measurement compression | `src/helpers.cpp` `QR` | present but returns `rows`, so the caller's `topRows(rows)` is a no-op |
| `oos_features_`, `Roos_`, `use_OOS_`, `OOS_update_min_observations_` | `src/estimator.{h,cpp}` | declared, never used |
| the update itself, the feature bookkeeping, the group window | — | deleted |

Known defects in the surviving code (all must be fixed in M1):

1. `ComputeOOSJacobian` calls `SlowGivens(oos_.Hf, oos_.Hx, A)` on the **full**
   `2*kMaxGroup`-row buffers, not on the `2n` rows it just filled. Rows beyond
   `2n` hold stale values from the previous feature (`Hf` is never zeroed), so
   the nullspace basis, the projected Jacobian, the residual and the returned
   row count are all garbage.
2. `A = FullPivLU::kernel()` is **not orthonormal**. The projected noise
   covariance is `Aᵀ R A`, which only stays `σ²I` for orthonormal `A`; the
   update then feeds `UpdateJosephForm` a diagonal `Roos_` that does not match
   the actual noise. Householder-`Q` gives an orthonormal basis instead.
3. `oos_.inn = A.transpose() * oos_.inn` mixes a `(2n-3)×2n` matrix with the
   full `2*kMaxGroup` residual buffer — dimension-dependent on (1).

## 2. Milestones

### M0 — baseline and harness
* worktree `xivo-oos` on branch `auto-oos` (from `auto`), separate build tree.
* `run_eval_oos.sh` at the workspace root (same interface as `run_eval.sh`,
  drives the worktree).
* Fix `Feature::FillJacobianBlock`: both group blocks were written to `goff`, so
  the reference group's rotation Jacobian was overwritten by its translation
  Jacobian and the translation slot stayed zero. This is in-state code, but every
  OOS comparison would otherwise be measured against a filter with a corrupted
  measurement model.
* Record the baseline for `cfg/sweep_dlt_nodesc.json` (the best mono config in
  `RESULTS.md`, mean ATE 0.1209 before the fix).

### M1 — correct OOS measurement model, unit-tested
* `Feature::ComputeOOSJacobian`: work on the filled `2n` rows only; project onto
  the left nullspace of `Hf` with an orthonormal basis (Householder QR of `Hf`,
  columns `3..2n-1` of `Q`), producing `(2n-3)` rows of `Hx` and residual.
* Multi-view triangulation for OOS features: Gauss-Newton over **all**
  observations in in-state groups, in the existing `(x/z, y/z, log z)`
  parameterization w.r.t. the reference camera. `Feature::RefineDepth` is close
  but gates on the *sum* of per-view residual norms against `max_res_norm`
  (2.5 px), which auto-rejects any well-tracked feature with more than a couple
  of views; the OOS path needs a per-view (mean) gate. Keep `RefineDepth`
  untouched so the in-state admission path does not change.
* Tests (new `src/test/unittest_oos_update.cpp`):
  - `Aᵀ Hf ≈ 0`, `AᵀA = I`, row count `= 2n-3`;
  - noise-free synthetic scene + exact poses ⇒ projected residual ≈ 0;
  - synthetic multi-view triangulation recovers a known 3-D point;
  - `FillJacobianBlock` copies every block of `J_` (the M0 bug had no test).

### M2 — pipeline integration
* Config: `use_OOS`, plus an `OOS` block (`meas_std`, `min_observations`,
  `max_observations`, `chi2_thresh`, `refine`, `max_reproj_err`).
* `ProcessTracks`: a track dropped by the tracker while **not** in-state is
  currently destroyed on the spot. Instead, keep it (status `DROPPED`) and queue
  it in `oos_features_`.
* After in-state outlier rejection / group discarding (so group `sind()`s are
  final): refine depth, compute per-feature OOS Jacobians, chi-square gate each
  feature, stack the surviving blocks under the in-state rows in `H_`, `inn_`,
  `diagR_` (`Roos_` for the OOS rows), one joint `UpdateJosephForm`, then remove
  the OOS features from the graph.
* Instrumentation: per-frame counts of candidate/used/gated OOS features and
  total OOS rows, exposed through the estimator so a run can be summarised.
* End-to-end eval on all six rooms.

### M3 — pose window for the OOS update
An OOS residual only constrains groups that are in the state. Today a group is
in-state only while it owns in-state features, so a dropped track may have very
few in-state observations. The deleted implementation kept a genuine sliding
window: it added *every* new frame's group to the state and pruned old ones with
a stride (`oos_discard_step`). Measure the distribution of in-state observations
per candidate first (M2 instrumentation), then evaluate:
  a. no group-management change;
  b. always augment with the current pose + strided pruning, with
     `EKF_MAX_GROUPS` raised enough to hold both the window and the reference
     groups of in-state features.

### M4 — robustness and tuning
* Noise/gating sweep: `oos_meas_std`, `min_observations`, chi-square threshold,
  triangulation quality gates, window length/stride, `EKF_MAX_GROUPS`.
* Interaction with the existing outlier rejection (MH gating, 1-pt RANSAC) and
  with `use_depth_opt` / `triangulation.method`.
* Fix and (optionally) enable measurement compression (`QR` returns `rows`, so
  compression is currently a no-op; and it may only be applied to a block with
  homogeneous noise).
* Re-sweep the in-state knobs, which were tuned against the broken in-state
  Jacobian.
* Seed sensitivity of the final config.

### M5 — deliverables
* `notes-n-prompts/report-oos.md`, detailed notes in
  `notes-n-prompts/notes-oos/`, one git commit per milestone on `auto-oos`.

## 3. Risks

* **Observability.** OOS residuals are invariant to the global gauge after the
  nullspace projection, so they should not excite the unobservable directions;
  but they do touch group-pose blocks whose covariance is manipulated by the
  gauge-fixing code (`FixFeatureXY`, the 1-pt RANSAC zeroing). Watch for
  covariance collapse / indefiniteness.
* **Bad triangulations.** A wrongly triangulated OOS point produces a consistent
  but wrong constraint on a whole window of poses — much more damaging than one
  bad in-state measurement. Gating is not optional here.
* **Cost.** Raising `EKF_MAX_GROUPS` grows `kFullSize` and the `O(n³)` Joseph-form
  update; `kFullSize` is 293 today (23 + 6·15 + 3·30).
* **The RPE target.** Rotational RPE is remarkably config-insensitive in
  `RESULTS.md` (0.62 mean across every knob tried). Reaching < 0.5 deg most
  likely needs the extra pose constraints to actually tighten attitude, not just
  position; if it does not move, that is a finding to report with evidence.
