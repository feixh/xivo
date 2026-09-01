# Plan — scan the code and fix bugs (mono + IMU)

Scope: the monocular-camera + IMU path of the `xivo` package, delivered on branch
`auto-bugfix` in worktree `/home/ubuntu/workspace/auto-slam-engineer/xivo-bugfix`
(branched from `auto`).

## How bugs will be found

Five complementary passes, because each catches a different class of defect:

1. **Systematic static audit, subsystem by subsystem.** ~14 kLOC over `src/` and
   `common/` is small enough to read exhaustively. Split by subsystem and audited
   in parallel by sub-agents, each reporting `file:line`, why it is wrong, the
   mono+IMU impact, and a minimal fix. Findings go into a single bug register and
   are then **independently re-verified by me** before any code changes — an audit
   that reports plausible-but-wrong findings is worse than no audit.
2. **The repo's own failing tests.** Two unit tests fail on a clean build
   (`NumericalLinearAlgebra.SlowAndFastGivensMatch`,
   `Triangulation.Angular_Reprojection_Error`). A shipped test that fails is a
   bug report the authors already wrote; both are in numerical code the filter
   depends on.
3. **Differential / invariant testing.** For the numeric core (Jacobians,
   triangulation, Givens, camera models, integrators) assert the properties that
   must hold: analytic vs numerical Jacobians, `A' Hf = 0` for the marginalizer,
   project/unproject round-trips, RK4 vs Prince-Dormand agreement.
4. **Runtime instrumentation on real data.** Build with NaN/overflow assertions
   that actually fire (the existing `anynan()` guard is itself broken), then run
   all six TUM-VI rooms and look for state blow-ups, silently rejected updates,
   and covariance non-PSD.
5. **End-to-end regression.** ATE/RPE on room1-6 before and after every change.
   A "fix" that does not move — or that worsens — the metrics gets scrutinised,
   because the shipped config was tuned *against* the buggy code.

Fixes are ranked by (severity × confidence), and each lands with a regression
test that fails before it and passes after.

## Milestones

Each milestone = one git commit on `auto-bugfix`, with notes in
`notes-n-prompts/notes-bugfix/`.

| # | Milestone | Deliverable | Gate |
|---|---|---|---|
| M0 | Baseline | worktree built; unit-test and 6-room ATE/RPE baseline recorded | numbers reproduce the `auto` branch |
| M1 | Bug register | audited findings, triaged, each verified or rejected by hand | every entry has file:line + a reasoned severity |
| M2 | Filter-model fixes | the bugs that corrupt the EKF measurement/propagation model | e2e ATE improves or is neutral; new unit tests |
| M3 | Numerical robustness | NaN/overflow/termination defects (broken `anynan`, unbounded log-depth, gating loop) | adversarial tests; no hang/abort on any room |
| M4 | Remaining fixes + shipped test failures | lower-severity defects; the two failing unit tests | full unit suite green |
| M5 | Regression + retune | 6-room e2e on the fixed code; re-sweep config knobs | mean ATE/RPE reported vs baseline |
| M6 | Report | `notes-n-prompts/report-bugfix.md` | — |

## Subsystem split for the audit (M1)

| Area | Files |
|---|---|
| A EKF core | `estimator.cpp`, `estimator_accessors.cpp`, `estimator_process.cpp`, `core.h` |
| B Update & gating | `update.cpp`, `mm.cpp`, `oos.cpp` |
| C State lifecycle | `feature.{h,cpp}`, `group.{h,cpp}`, `graph.{h,cpp}`, `graphbase.cpp` |
| D Numerics | `common/*.h` (rodrigues, project, utils, cameras), `helpers.cpp`, `geometry.cpp`, `jac.h`, `rk4.cpp`, `princedormand.cpp`, `imu.cpp` |
| E Front end & config | `tracker.{h,cpp}`, `manager.cpp`, `loader.cpp`, `param.cpp`, `options.cpp` |

## Already-known defects (to confirm on this branch, not re-discover)

Found during earlier work on sibling branches; all four are present on `auto`:

1. `Feature::FillJacobianBlock` writes the reference-group rotation and
   translation Jacobians to the **same** column offset (`src/feature.cpp:688-689`),
   so the translation block stays zero and the rotation block holds the wrong
   matrix. Known worth ~16% mean ATE.
2. `anynan()` iterates `Derived::RowsAtCompileTime`, which is `-1` for dynamic
   matrices (`common/utils.h:94`) — every NaN guard on `MatX`/`VecX` is a no-op.
3. Feature log-depth is unbounded, so `exp()` overflows to `inf` and poisons the
   filter (aborts in Sophus quaternion normalisation).
4. `Estimator::MHGating` can loop forever, because `NaN < thresh` never becomes
   true so the inlier count never reaches `min_required_inliers_`.

## Risks

- **Config was tuned against buggy code.** Fixing a real bug can *raise* ATE
  under the old knobs. Mitigation: judge M2-M4 on model-correctness evidence
  (unit tests, invariants) and only re-tune in M5.
- **Determinism.** `XIVO_RANDOM_SEED=0` is required or every run differs; all
  comparisons use it, and seed-sensitivity is checked before claiming a win.
- **Non-goals.** Stereo, OOS-as-a-feature, loop closure and mapping are out of
  scope except where a defect in them also corrupts the mono path.
