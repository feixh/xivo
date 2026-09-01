# M1 — the out-of-state measurement model

Scope: `src/oos.cpp`, `src/feature.{h,cpp}`, `src/options.h`,
`src/test/unittest_oos_update.cpp`. No pipeline change yet (`use_OOS` still
aborts in the estimator), so end-to-end numbers are unchanged by this milestone.

## What the measurement is

A feature that the tracker drops without it ever having been in the state still
carries information: its `n` observations, taken from `n` different poses, are
all explained by one 3-D point. Triangulate the point, stack the `2n`
reprojection residuals

```
r = [ ... x_i - pi(g_i^-1 X) ... ]  ~=  Hx * dx + Hf * dX + noise
```

and then left-multiply by a matrix `A` whose columns span the left nullspace of
`Hf`. `A' Hf = 0`, so the (unknown, and correlated-with-nothing) point error
drops out and `A' r = (A' Hx) dx + A' noise` is a genuine EKF measurement of the
poses in the state. That is the MSCKF update.

## The three defects in the surviving code

1. **Stale rows.** `SlowGivens(oos_.Hf, oos_.Hx, A)` was handed the whole
   `2*kMaxGroup = 30`-row buffer while only `2n` rows had just been written.
   `Hf` is never zeroed, so rows `2n..29` held whatever the *previous* feature
   left there — the nullspace basis, the projected Jacobian and the returned row
   count were all garbage. Now `MarginalizeOOSPoint(rows)` takes an explicit row
   count and only ever touches `topRows(rows)`.
2. **Non-orthonormal basis.** `A = FullPivLU(Hf').kernel()` returns *a* basis of
   the nullspace, not an orthonormal one. The update feeds `UpdateJosephForm` a
   diagonal `Roos_`, i.e. it assumes `A' R A = sigma^2 I`, which only holds when
   `A' A = I`. Replaced by the last `2n-3` columns of the Householder `Q` of
   `Hf`: `Hf = Q [R; 0]`, so every column of `Hf` is in the span of the first
   three columns of `Q` and `A' Hf = 0` holds exactly — even when `Hf` is rank
   deficient (a feature with no parallax), in which case one row of information
   is simply thrown away instead of the basis silently becoming wrong.
3. **Dimension mismatch.** `oos_.inn = A' * oos_.inn` mixed a `(2n-3) x 2n`
   matrix with the 30-element buffer. Now residual and Jacobian are projected
   together, in place, over the same `rows`.

`ComputeOOSJacobianInternal` (the per-observation Jacobians) was *not* broken —
its perturbation conventions match the in-state code and `unittest_jacobians_oos`
verifies all four blocks numerically. It is untouched.

## Triangulation

`Feature::RefineOOSDepth` — Gauss-Newton over all selected observations in the
existing `(x/z, y/z, log z)` parameterization w.r.t. the reference camera.
Differences from the existing `Feature::RefineDepth` (which is left alone,
because it sits on the in-state admission path):

* gates on the **mean** per-view residual norm, not the sum. `RefineDepth`
  compares `sum_i |res_i|` against `max_res_norm` (2.5 px in the shipped
  configs), which rejects essentially every feature with more than ~3 views —
  exactly the well-tracked features an OOS update wants.
* keeps the best iterate instead of the last one, and evaluates the state
  produced by the final step (`iter <= max_iters`) instead of discarding it.
* includes the reference view. Its residual is not identically zero once
  `x_(0:2)` starts moving, and it is one of the rows that end up in the update,
  so it belongs in the cost being minimized.
* rejects on depth range (`zmin`/`zmax`) as well as on residual, and returns
  `false` (rather than a filter-poisoning NaN) when the normal equations are
  degenerate.

A bad triangulation is much more damaging here than a bad in-state measurement —
it constrains a whole window of poses consistently and wrongly — so the gate is
deliberately tighter than the in-state one (`max_mean_reproj_err` default 1.5 px).

## Observation selection

`Feature::SelectOOSObservations` keeps only observations from **in-state** groups
(a pose that is not in the state cannot be corrected, and is treated as known),
sorts them oldest-first (`GetObservationsOf` walks a hash map, so its order is
not something to depend on), and thins tracks longer than `max_observations` by
taking evenly spaced observations while always keeping the first and the last —
the parallax, and hence the depth information, lives at the ends of the track.
The function is idempotent, so running the refinement and the Jacobian each on
their own selection is guaranteed to use identical rows.

`ComputeOOSJacobian` no longer reads `Estimator::instance()`; everything comes
from an `OOSOptions` struct (`src/options.h`), which also makes it unit-testable
without an estimator.

## Tests — `bin/unitTests_OOSUpdate`, 13 cases

Synthetic scene: 5..12 in-state groups on a gently curving trajectory, one point
at 3.4 m, the perfect-pinhole camera, all observations exact.

| case | what it pins down |
|---|---|
| `RowCountIsTwoNMinusThree` | `2n-3` rows out; `oos_inn_size`/`Ho`/`ro` agree |
| `TooFewObservationsIsRejected` | `min_observations`; floating groups don't count |
| `MarginalizationAnnihilatesPointJacobian` | `A' Hf = 0` (a copy of `Hf` parked in the unused feature columns of `Hx` must come out zero) |
| `MarginalizationBasisIsOrthonormal` | `Hx1' Hx1 = Hx0' Pi Hx0` and `Hx1' r1 = Hx0' Pi r0` with `Pi` the orthogonal projector — invariant to the choice of basis, and false unless `A'A = I` |
| `NoiseFreeResidualIsZero` | exact data ⇒ residual 0 |
| `PointErrorIsMarginalizedOut` | a 10% depth error gives a >1 px raw innovation but a projected one 100x smaller |
| `RefineRecoversPoint` | 35% depth + bearing error ⇒ point recovered to 1e-4 m, residual ~0 |
| `RefineRejectsInconsistentTrack` | one 30 px outlier observation ⇒ `false` |
| `RefineRejectsOutOfRangeDepth`, `RefineNeedsTwoViews` | the other two rejection paths |
| `SelectionSkipsFloatingGroupsAndSortsByAge`, `SelectionThinsLongTracks` | selection, ordering, thinning, idempotence |
| `FillJacobianBlockCopiesEveryBlock` | regression test for the M0 bug |

Mutation-checked, not just green: replacing the Householder basis with the old
`FullPivLU::kernel()` makes exactly `MarginalizationBasisIsOrthonormal` fail
(and nothing else), which is the point — the LU kernel *is* a nullspace basis, so
the orthogonality test alone cannot see the bug that mattered.

Full suite after M1: same two pre-existing failures as the M0 baseline
(`SlowAndFastGivensMatch`, `Angular_Reprojection_Error`), nothing new. Note that
`ctest` reports everything as failing regardless — the test binaries load
`src/test/camera_configs.json` by relative path, so they have to be run from the
repository root, not from `build/`.
