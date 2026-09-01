# M5 — feature/group lifecycle and re-anchoring

Eight defects in `graph.cpp`, `feature.cpp`, `manager.cpp`, `geometry.cpp`
and `estimator.cpp`, plus two new test files. Register entries #25–#31 and
one defect found during the work that was not in the register.

**This milestone regresses the mean ATE by 0.012 and is committed anyway.**
The reasoning is in the last section; read it before treating the number as
a failure.

| variant | mean ATE | RPE_rot | RPE_tra |
|---|---|---|---|
| M4 (`69f1486`) | **0.0929** | 0.6204 | 0.0252 |
| M5, feature block only (no pose cross terms) | 0.1135 | 0.6203 | 0.0268 |
| M5, full transform, ocf = 1.5 (shipped) | 0.1051 | 0.6205 | 0.0270 |
| M5, full transform, ocf = 1.0 | 0.1044 | 0.6204 | 0.0267 |
| M5, full transform, ocf = 1.2 | 0.1084 | 0.6199 | 0.0268 |

Per sequence as committed: room1 0.1217, room2 0.0757, room3 0.1622,
room4 0.1009, room5 0.1067, room6 0.0633. The regression is spread across
five of the six — it is not one sequence tipping over.

## The headline: re-anchoring never touched the filter covariance

`Feature::ChangeOwner` re-parameterizes a feature into a new reference
group:

    xn = pi( (gsb_n gbc)^-1 gsb_o gbc pi^-1(x) )

and ended with

```cpp
P_ = J * P_ * J.transpose();
```

`Feature::P_` is a local `Mat3`. For an **out-of-state** feature that is
the whole story — the depth sub-filter treats the reference pose as exact.
For an **in-state** feature it is dead storage: the authoritative
covariance is the 3×3 block of `Estimator::P_` at `kFeatureBegin + 3 *
sind()`, together with its cross-covariance rows and columns against the
other 200 state variables. None of that is reachable from `Feature`.

So for every in-state feature that got re-anchored — which is every
in-state feature whose reference group ages out, i.e. routinely — the mean
`x_` was moved into a new coordinate system and the covariance was left
describing the *old* one. Including the cross-covariances, which still
pointed at the group the feature was no longer anchored to. And
`f->inflate_cov(cov_factor)` — the one thing that looked like it was
compensating — also wrote only to the dead local copy.

Two consequences worth separating:

1. The feature's own 3×3 block was wrong (wrong basis).
2. The outgoing group was about to be deleted from the state. Its
   contribution to the re-anchored feature's uncertainty was never folded
   in, so it was **lost**, not marginalized. The filter got more confident
   for free every time a group aged out.

### The transform

`ChangeOwner` now optionally reports three Jacobian blocks and the
estimator applies them to the filter matrix. `xn` depends on the feature's
own three parameters *and* on both groups' 6-DOF poses, so this is a row
operation over three blocks rather than a similarity transform on one.
Using the same right (local) perturbation convention as the rest of the
filter (`core.h:135-152`, `Rsb <- Rsb exp(dW)`, `Tsb <- Tsb + dT`):

    d xn / d dx    = dxn_dXcn * dXcn_dx                        (this existed)
    d xn / d dW_o  = dxn_dXcn * R_cn_s * (-Rsb_o * hat(Xb_o))
    d xn / d dT_o  = dxn_dXcn * R_cn_s
    d xn / d dW_n  = dxn_dXcn * Rbc^T * hat(Xb_n)
    d xn / d dT_n  = dxn_dXcn * (-R_cn_s)

with `Xb = Rsb^T (Xs - Tsb)` in each group's body frame. Every one of the
four new expressions has a counterpart already present in
`ComputeJacobian`'s cache, which is what made them checkable by
inspection before they were checked numerically.

`Estimator::ReanchorFeatureCovariance` applies `P <- S P S^T`, where `S` is
the identity except for the feature's three rows, which hold `Jx` at the
feature offset, `Jn` at the new group's offset and `Jo` at the old
group's. It is done as two passes, `(S P)` then `(S P) S^T`; the column
pass deliberately reads the **row-updated** blocks, which is what makes the
diagonal come out as the full quadratic form rather than a one-sided
product. `err_` gets the same treatment — it is zero outside an update, but
the function should not depend on that.

Deliberate remaining approximation: the `Wbc`/`Tbc` calibration cross terms
are omitted. They appear in both chains with opposite signs and partially
cancel, and the camera-IMU extrinsic uncertainty is small compared to the
group poses'. Noted here rather than silently.

### The Jacobians are validated, not asserted

`src/test/unittest_reanchor.cpp`, 6 tests, all passing. Each of the three
blocks is central-differenced against `Reparameterized()`, a reference
implementation written from the geometry — it composes `SE3`s directly and
shares no code with `ChangeOwner`. `tol = 1e-6`. There is also a test that
the two 3×6 pose blocks are neither equal nor negatives of each other, so a
regression that swapped them or reported one twice fails; and one pinning
the documented no-mutation-on-failure contract.

This matters for how the regression below is read: the transform is
**correct**, so the ATE cost is not a derivation error to be hunted.

## `J_` was never cleared between calls

Not in the register — found while reasoning about what else `ChangeOwner`
leaves stale. `Feature::J_` is a full-width row block of which only a few
column blocks are ever written, and `ComputeJacobian` overwrites only
those. After a re-anchoring, the old reference group's six columns retain
the Jacobian w.r.t. a group the feature is no longer anchored to, and
`MHGating` forms `J_ * P_ * J_^T` over the **whole** row — so gating used a
term that should not have been there at all.

Fixed with `J_.setZero()` at the top of `ComputeJacobian`. Ablation C shows
the six-sequence output is **byte-identical** with and without it, which is
the useful part of the finding: on this config the stale columns always
multiply into state blocks that happen to be zero, so the bug is real,
reachable, and currently silent. It would surface the moment a re-anchored
feature's old group slot were reused by a group with non-zero
cross-covariance.

## Gauge features are dropped rather than re-anchored

A gauge feature has had the x/y rows and columns of its filter covariance
**zeroed** by `Estimator::FixFeatureXY` in order to fix the gauge of one
specific group. `TransferFeatureOwnership` re-anchored those like any
other feature, which keeps two directions pinned but now under a group that
already has its own three gauge features — over-constraining the gauge —
and there is no covariance that could honestly be restored for the released
directions.

They are now dropped with the group they fix. The cost is bounded:
`DiscardAffectedGroups` only discards a group once it owns fewer than
`num_gauge_xy_features` in-state features, so at most that many are lost.
Measured on top of M4 alone this is **0.0935 vs 0.0929** — neutral.

## `DiscardFeatures` was called before the status was set

```cpp
std::vector<FeaturePtr> nullrefs = FindNewOwnersForFeaturesOf(g);
DiscardFeatures(nullrefs);
for (auto f: nullrefs) { f->SetStatus(FeatureStatus::NULLREFED); }
```

Two bugs in three lines. `DiscardFeatures` keys the release of the filter
state slot off `Feature::instate()`, and it ends by returning the object to
the `MemoryManager` pool. So:

- the features still had status `INSTATE`/`GAUGE` when `DiscardFeatures`
  ran, so the slot release path ran on the basis of the *pre*-discard
  status — and then
- `SetStatus` wrote to an object the memory manager was already free to
  hand out to the next `Feature::Create`.

Reordered so the slot is released and the status set *before*
`DiscardFeatures` sees the feature. Costs 0.0060 when reverted (borderline
against the 0.005 noise band).

## Stale Jacobians survived re-anchoring into the EKF update

`UpdateStep` computes each feature's Jacobian, then calls
`DiscardAffectedGroups`, which may re-anchor some of those same features —
changing both `x_` and *which state columns the Jacobian belongs in*.
`FilterUpdate` then applied a Jacobian computed against the old reference
group at the new group's offset.

Fixed by recomputing over `in_current_ekf_update_` after the discard pass.
At most `kMaxFeature` = 30 features, so the cost is negligible. Worth
0.0060 mean ATE.

## `gauge_features_[f->ref()].erase(f)` leaked map entries

`operator[]` default-constructs. `RemoveFeature` is called for features
that never had an owner, so `f->ref()` is `nullptr`, and the map grew a
`nullptr` key holding an empty set. Only `RemoveGroup` erases keys, so any
entry created this way outlives the group it is keyed by — and for a
non-null but already-removed group, `gauge_features_` regained an entry for
a group that is no longer in the graph. Changed to `find` + guarded erase.

## `FindNewGaugeFeatures`: a discarded return value left the gauge unfixed

The retry loop looks for three non-collinear gauge features, shuffling up to
ten times. On the last attempt:

```cpp
if (NT==9) {
  gauge_features_[g] = gauge_features_backup;
  fill_slots(g, candidates_backup);   // return value discarded
}
```

`fill_slots` has two effects: it inserts into `gauge_features_[g]` *and* it
returns the list the caller passes to `FixFeatureXY`. Dropping the return
value meant the group counted three gauge features — so it would never look
for more — while `FixFeatureXY` never ran on any of them and their status
stayed `INSTATE`. **The group's gauge was left entirely unfixed while the
bookkeeping claimed it was fixed.** Now assigned and `break`-ed.

Two more in the same function. `num_to_find` is `int` and compared against
`candidates.size()`; if it ever went negative the usual arithmetic
conversions made it a huge `size_t`, and the `CHECK(num_to_find >= 0)`
guarding that is compiled out in release (`NDEBUG`). Clamped, and the
comparisons cast explicitly. And the loop spun all ten iterations
recomputing an identical answer when fewer than three slots got filled —
collinearity is undefined for fewer than three points and no reshuffle
changes how many slots fill.

## `PointsAreCollinear` was the cross-binary irreproducibility

Three defects, all live (see `m0-baseline.md` for how this surfaced):

```cpp
Vec3 v1 = pts[1] - pts[0];                    // unguarded
for (int i=2; i<pts.size(); i++)
  if (v1.cross(pts[i] - pts[0]).norm() > thresh) return false;
```

1. `pts[1]` is read out of bounds for 0 or 1 points.
2. `|v1 x vi|` is an **area**. It grows with the square of the distance
   between the points, so identical geometry gets opposite verdicts at
   different ranges — a gauge triple 10 m out is no more collinear than the
   same triple at 1 m. Worse, it collapses below any fixed threshold
   whenever `pts[0]` and `pts[1]` happen to be close together, *regardless
   of where the other points are*, so well-spread triples were rejected.
3. The verdict depends on which point lands at index 0 — and the caller
   builds the vector by iterating an `unordered_set<FeaturePtr>`, i.e. in
   heap-address order. **This is the root cause of two different binaries
   producing different trajectories from identical input**, identified in
   M0.

Replaced with the second moment of the centred point set: `sqrt(s1/s2)`
from a `SelfAdjointEigenSolver`, i.e. the spread orthogonal to the best-fit
line over the spread along it. Permutation-invariant, scale-invariant, and
dimensionless. `src/test/unittest_geometry.cpp` (7 tests) pins all three
properties, including all 24 permutations of a 4-point set and five decades
of scale; the close-leading-pair case is a direct regression test for
defect 2. Every test except the degenerate-size one *fails* against the old
implementation, and that one **segfaults** it.

**The threshold's units changed and it has not been re-swept.** All 20-odd
configs set `collinear_cross_prod_thresh: 0.001`. As a ratio that means
"off-line spread below 0.1 % of along-line spread", which essentially never
fires — so the retry loop now always exits on its first iteration. The old
value did fire, but spuriously (defect 2). A meaningful value is more like
0.05–0.1 (≈3–6°). Left alone here because changing it is tuning, not a bug
fix; it goes in M7's sweep. Ablation D confirms this path is nearly inert
on `sweep_dlt_nodesc` (0.1132 vs 0.1135), so re-tuning it will not move the
mean much.

## Why a regression is being committed

Six ablations plus one isolation run, all on `sweep_dlt_nodesc` /
room1–room6 / seed 0:

| ablation | mean ATE | verdict |
|---|---|---|
| full M5 (feature block only) | 0.1135 | — |
| minus gauge-feature drop | 0.1221 | drop is worth 0.0086 |
| minus covariance transform | 0.1144 | ~neutral once the rest is in |
| minus `J_.setZero()` | 0.1135 | byte-identical |
| minus `PointsAreCollinear` rewrite | 0.1132 | inert |
| minus Jacobian recompute | 0.1195 | worth 0.0060 |
| M4 + gauge-drop only | 0.0935 | ≈ M4 |

Read together: **everything in M5 except making in-state re-anchoring live
is either neutral or a small gain.** The 0.012 cost comes entirely from the
covariance transform, and adding the two pose cross-term blocks — the more
correct version — recovered 0.008 of it (0.1135 → 0.1051), which is
consistent with the transform being right rather than wrong.

The remaining gap is a tuning artefact, and the mechanism is specific:
before this milestone, a re-anchored in-state feature's covariance was
never updated and its cross-covariance against its reference group was
*stale or zero*. `MH_thresh` and `feature_owner_change_cov_factor` were
tuned against that. Making the covariance correct makes `S = J P J^T`
larger for re-anchored features, so the Mahalanobis distance shrinks and
gating admits measurements it used to reject. The inflation factor was
swept (1.0 → 0.1044, 1.2 → 0.1084, 1.5 → 0.1051) and is **not** the lever;
`MH_thresh` is the obvious next candidate and belongs in M7 alongside the
other two knobs whose units this work changed.

Ruled out as explanations:

- **Bad marginalization of the outgoing group.**
  `RemoveGroupFromState` zeroes the group's rows and columns, which *is*
  correct marginalization in covariance form, and it runs inside
  `DiscardGroup` — *after* `FindNewOwnersForFeaturesOf` has folded the
  outgoing group's contribution into the feature's rows. Verified by
  reading the call order, not assumed.
- **A Jacobian sign or convention error.** Six numerical tests.
- **A transposition error in the two-pass update.** Checked by hand: the
  column pass reads the row-updated blocks, so the result is the full
  `S P S^T`.

The exit criterion for this work is that the code is free of bugs, not that
every milestone lowers the ATE. Reverting M5 would mean knowingly shipping
a filter that loses a group's uncertainty every time one ages out, writes
to recycled memory, applies Jacobians at the wrong state offset, and
produces different trajectories from the same input depending on heap
layout — in exchange for 0.012 m of ATE on one config that was tuned
around those bugs. So it is committed, and the re-tuning it makes necessary
is M7's job.
