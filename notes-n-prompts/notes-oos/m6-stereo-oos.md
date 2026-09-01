# M6 — the out-of-state update on a stereo rig

`auto-oos` was developed on the monocular control config; `auto` also carries the
stereo rig. Merging the two is more than a textual merge: a dropped track that
was seen by *both* cameras carries twice the measurements, and the out-of-state
update was throwing half of them away. This note is the design, the two
non-obvious constraints that shaped it, and what it bought.

## Where the right observations had to come from

`Feature` holds exactly one right observation — `xp_r_`, set by the matcher for
the *current* frame and cleared at the next one. That is all the in-state update
needs, since it linearizes about the current frame. The out-of-state update is
the opposite case: it fires when the tracker *drops* a track, and it revisits the
whole history of that track. By then `xp_r_` describes some other frame.

So the right pixel had to become a property of the (feature, group) **edge**, not
of the feature:

* `Observation` gained `has_right` / `xp_r` (`src/core.h`).
* `FeatureAdj` gained a side map `right : group id -> Vec2` (`src/feature.h`).
  A side map rather than changing `FeatureAdj`'s value type, because that type is
  iterated in a dozen places (`GetGroupsOf`, `FindNewOwner`, `SanityCheck`, the
  reanchoring path, ...) and none of them care about the right camera. `Add` /
  `Remove` keep the two in step; `GetObservationsOf` / `GetObservationOf` join
  them back together.
* `Graph::AddGroupToFeature` records it. That is the one place where "the current
  frame" and "the group being created for it" coincide, which is why reading
  `f->xp_r()` there is correct.

On TUM-VI room1, **89.7%** of the in-state views of a dropped track carry a right
observation, so this is nearly a doubling of the measurement.

## Two constraints that shaped the rest

**1. The noise has to stay isotropic.** `MarginalizeOOSPoint` projects the
stacked Jacobian onto the *orthonormal* left nullspace of `Hf` precisely so that
`A' (sigma^2 I) A = sigma^2 I` and the update can feed the filter a scalar
`Roos_` (see M1). A right-camera row with a different variance would break that
argument. Rather than carry a non-uniform `R` through the QR, the right rows are
**whitened as they are written**: `Hf`, `Hx` and `inn` all get a factor
`1/sqrt(OOS.stereo_R_scale)`. After that the stacked measurement really is
isotropic and every line of the existing algebra holds unchanged.
`OOS.stereo_R_scale` defaults to `stereo_update.R_scale`, so one knob governs
both updates.

**2. The Jacobian buffer must not grow.** `OOSJacobian` is `2*kMaxGroup` rows by
`kFullSize` columns and is a member of **every pooled `Feature`** — at 90/45 that
is ~400 kB per feature. Doubling it to fit `4n` rows would cost more than the
whole rest of the update. Instead the *view* budget is halved when the right rows
are in play (`SelectOOSObservations`), which is the response the design already
had for long tracks: thin the track, keeping both ends, where the parallax is. At
`kMaxGroup = 45` the cap is 22 views and the tuned config asks for 15, so it
never binds; a row-budget check in `ComputeOOSJacobian` makes sure no
`max_observations` / `kMaxGroup` combination can overrun the buffer silently
(those Eigen block writes are unchecked in a release build).

## The rows themselves

Because the rig is *fixed* and lives outside the error state, the entire
dependence of the right observation on the state still flows through `cache_.Xcn`
— the same 3-D point in the same left-camera frame the left rows were linearized
about. The right rows therefore reuse the left camera's whole `dXcn_d*` chain
verbatim; only the last two links differ:

```
Xc1        = Rc1c0 * Xcn + Tc1c0
dxp1_dXcn  = dxp1_dxc1 * dxc1_dXc1 * Rc1c0
```

which is exactly what `Feature::ComputeRightJacobian` does for the in-state
update. Crucially the right rows share the *same* `Hf` block structure as the
left ones — they constrain the same point — so they enter the same nullspace
projection. An n-view track yields `4n - 3` rows instead of `2n - 3`.

A view whose right observation is missing, or whose point is predicted behind the
right camera, contributes its two left rows and nothing else, so the row count is
data-dependent and `ComputeOOSJacobianInternal` now takes an explicit row cursor
and returns how many rows it wrote. (`oos_jac_counter_` used to double as the
observation counter and then as the row count; with a variable stride that no
longer works.)

`RefineOOSDepth` gets the right residuals too, with the same weighting: the point
that is marginalized should be the optimum of the measurements that marginalize
it. It is also what makes the depth gate see a bad right match.

## Why stereo does *not* relax the two-view floor

On a row count alone, one stereo view looks sufficient: 4 rows minus 3 for the
point leaves one. It is worthless. With a single group in the state, every row's
dependence on the state flows through that group's `Xcn`, whose 3-dimensional
column space is exactly what the marginalization annihilates — formally,
`Hf = dxp_dXcn * dXcn_dXs` with `dXcn_dXs` invertible, so `A' Hf = 0` implies
`A' dxp_dXcn = 0` and hence `A' Hx = 0`. The surviving row would have an
identically zero Jacobian and a nonzero innovation: a pure epipolar residual on a
rig that is not being estimated. The two-view floor is a property of the
marginalization, not of the monocular measurement.

## What it bought

Six-member ensembles, six rooms, ATE@0.001 (`merge/logs/e_*.log`):

| arm | OOS off | OOS on | delta | Welch t |
| --- | --- | --- | --- | --- |
| mono + IMU | 0.0784 ± 0.0061 | 0.0686 ± 0.0034 | −0.0098 | −3.46 |
| stereo + IMU | 0.0556 ± 0.0032 | **0.0453 ± 0.0024** | −0.0103 | −6.29 |

The out-of-state update is worth ~0.010 in both modes and stereo+OOS is the best
arm on every metric. **The right rows themselves are not, on this dataset,
worth anything measurable.** Running the same stereo OOS config with
`use_stereo: false` — i.e. the monocular measurement on the stereo pipeline, a
6-member ensemble, `merge/logs/e_st_on_monomeas.log`:

| metric | mono measurement | + right rows | delta | Welch t |
| --- | --- | --- | --- | --- |
| ATE@0.001 | 0.0449 ± 0.0015 | 0.0453 ± 0.0024 | +0.0005 | +0.40 |
| ATE@0.02 | 0.0565 ± 0.0027 | 0.0591 ± 0.0029 | +0.0025 | +1.55 |
| RPE_tra | 0.0133 | 0.0132 | −0.0001 | −1.15 |
| RPE_rot | 0.6216 | 0.6215 | −0.0001 | −0.56 |

Nothing there clears its own noise. It is not that the rows are inert — they do
what they are supposed to do. On room1 they take the update from 19 727 to
44 992 rows, drop `bad_triangulation` from 795 to 778 and raise the number of
accepted tracks from 2695 to 2762. They are also free: wall clock is 225 s vs
233 s on room1 and 219 s vs 216 s on room3, i.e. inside scheduling noise, since
this is one extra small QR per frame.

The reason it does not show up in ATE is visible in the same summary:
`too_short=10538` of `candidates=14115`, and mean in-state views per candidate
is **2.0**. What limits the out-of-state update on hand-held room sequences is
how many dropped tracks have enough in-state groups to constrain anything at
all, not how well the ones that qualify are triangulated — and the ones that
qualify already have the trajectory's own parallax. Better depth on an
already-well-determined depth buys nothing. Where the rows should matter is the
case this dataset does not contain much of: slow or forward motion, where the
baseline between views collapses and the rig's baseline is all there is. The
unit test for that case has to be synthetic for exactly this reason.

So `use_stereo` defaults to `true` because it is the better-posed measurement at
no cost, not because it was measured to help; the knob is there to turn it off.

## A measurement trap worth recording

The stock `evaluate_rpe.py` shows the stereo OOS arm **worse** in rotation:
+0.0008 deg at t = +2.99, which reads as a real regression. It is not. That
evaluator's nearest-neighbour timestamp matching scores a *perfect* trajectory at
~0.30 deg on these sequences, and the whole 0.62 deg it reports is dominated by
that artifact. Re-scored with `evaluate_rpe_interp.py` (via
`scripts/rpe_interp_dir.py`), the same 72 runs give +0.0005 deg at t = +1.34 —
indistinguishable from zero. Any rotational claim on TUM-VI has to come from the
interpolated evaluator.

## Tests

`src/test/unittest_oos_stereo.cpp` (`ctest -R OOSStereo`), a separate binary
because `CameraManager`'s registry and `StereoRig` are process-wide singletons
and this test needs the real fisheye pair in slots 0/1:

* row counts: `4n` filled, `4n - 3` after marginalization, and the three ways of
  having no right rows (`use_stereo: false`, no match, monocular) reproduce the
  monocular measurement;
* `A' Hf == 0` with stereo rows in the stack;
* the right rows against numeric differentiation of the predicted right pixel,
  w.r.t. the group pose, `gbc` and the point;
* whitening: at `stereo_R_scale = 4` the right rows are exactly halved and the
  left rows untouched;
* a low-parallax track (21 mm of motion, ~100 mm baseline, ⅓ px of matching
  error) whose depth only the right rows recover — and note that this test needs
  the pixel noise to be meaningful at all: with exact observations even a
  millimetre of parallax determines the depth exactly, and what ruins the
  monocular estimate is the *amplification* of a matching error by a short
  baseline;
* a 15 px right-match error failing the triangulation gate that the left
  observations alone pass;
* the halved view budget.
