# M4 — triangulation guards, gravity Jacobian, badtri units

Commit `69f1486`. Eight defects across `helpers.cpp`, `feature.cpp` and
`estimator.cpp`. Register entries #17–#24 plus the shipped failing test
#55.

Measured on `cfg/sweep_dlt_nodesc`, TUM-VI room1–room6, mono cam0 + IMU,
seed 0:

| variant | mean ATE | RPE_rot | RPE_tra |
|---|---|---|---|
| M2/M3 baseline | 0.1041 | 0.6207 | 0.0282 |
| **M4, all fixes** | **0.0929** | **0.6204** | **0.0252** |
| M4 minus badtri units | 0.1068 | 0.6203 | 0.0283 |
| M4 minus gravity Jacobian | 0.0969 | 0.6209 | 0.0292 |

Per sequence (all fixes): room1 0.1138, room2 0.0741, room3 0.1479,
room4 0.0829, room5 0.1004, room6 0.0383.

Read the ablations honestly: **the badtri unit fix is the whole gain.**
Remove it and the mean goes to 0.1068, i.e. *worse* than the M2 baseline
by 0.0027 — inside the ±0.005 cross-binary noise band, so the four
triangulation guards plus the gravity Jacobian are collectively neutral
on this config. The gravity revert costs 0.0040, also inside the band.
They are committed because they are correct, not because they were
measured.

## The gain: `initial_std_x/y_badtri` were never converted from pixels

```cpp
init_std_x_ = cfg_["initial_std_x"].asDouble();
init_std_x_ /= Camera::instance()->GetFocalLength();   // normalised
...
init_std_x_badtri_ = cfg_["initial_std_x_badtri"].asDouble();  // pixels!
```

Both pairs feed the identical sink — `Feature::Initialize`, i.e. the
covariance of `x_ = (X/Z, Y/Z, log Z)`, whose first two components are
normalised camera coordinates. Every shipped config gives the two pairs
the *same* numbers (1.0), which settles the question of intent: they must
be interpreted the same way. Without the division the initial x/y
variance was too large by `fl² ≈ 3.65e4` for TUM-VI cam0.

This is not a corner case. #52 establishes that the badtri branch in
`InitializeJustCreatedTracks` is taken **100 % of the time**: the
features it loops over are `new_features_`, which `manager.cpp:181` marks
`TrackStatus::CREATED` and which therefore have exactly one observation,
while `Triangulate` requires two (`manager.cpp:230` gates on
`f->size() == 2`). So `init_std_x_/y_` — the correctly-converted pair —
is dead code, and every feature in the system was initialised with a
36 500× inflated x/y variance.

That also retroactively makes the M3 `fl_` fix (#11) observable: `fl_`'s
only consumer is this conversion.

Note the knob's units changed. `initial_std_x_badtri` now means pixels
(matching `initial_std_x`), so 1.0 is 1 px rather than 1 normalised unit.
It has not been re-swept; that belongs to M7.

## The NaN chain in the live triangulation path

Three defects compose into one failure mode, and the live config
(`triangulation.method = 1`, DLT-SVD) hits all of it.

`DirectLinearTransformSVD` ends with

```cpp
X << V(0, 3), V(1, 3), V(2, 3);
X /= V(3, 3);
return true;
```

`V(3,3) → 0` is the point-at-infinity case — any zero-parallax pair,
which is routine for a nearly-static or purely-rotating hand-held camera.
The division yields `inf`/`NaN` and the unconditional `return true`
reports it as a good triangulation.

The caller's only line of defence was

```cpp
if (auto z = Xc1(2); z < options.zmin || z > options.zmax) { /* bad */ }
```

which is written as an *is-bad* test. Every comparison against NaN is
false, so a NaN sails past both branches of the `||` and lands in the
success path, where it sets `x_` to NaN, `x_(2)` to `log(NaN)`, and
`triangulation_successful_ = true`. `ClampLogDepth` cannot rescue that
(`std::clamp` on NaN returns NaN), and MH gating cannot reject it either
— `mh_dist < mh_thresh` is false for NaN, so the feature is *not* an
inlier, but `S.llt().solve()` on a NaN covariance has already been
computed. Rewritten as a negated is-good test with `allFinite()`.

Worth recording as a structural finding: **the live DLT-SVD path has no
cheirality, parallax or angular-reprojection check at all.** Those live
in `check_cheirality` / `check_parallax` /
`check_angular_reprojection`, which are called only from `L1Angular`,
`L2Angular` and `LinfAngular`. Method 1 gets the `zmin`/`zmax` range test
and nothing else — which is exactly why the NaN gate mattered.

## The shipped failing test was a real bug report

`Triangulation.Angular_Reprojection_Error` fails on a clean build of the
base branch, with an in-tree comment saying it "fails in RELEASE but
passes in DEBUG". That is the fingerprint of FMA/x87 contraction, and the
cause is:

```cpp
float theta0 = acos(Rf0.dot(Rf0_prime) / (Rf0.norm() * Rf0_prime.norm()));
```

The angular methods leave one of the two bearings **unchanged by
construction**, so that ratio is `1 + O(ulp)`. Whether it lands on
`1.0` or `1.0000000000000002` depends on whether the compiler contracts
the dot product; on the wrong side, `acos` returns NaN. Then
`std::max(NaN, x)` propagates the NaN and `NaN > max_theta_thresh` is
**false** — so the gate silently *accepted* the outlier it exists to
reject, and the reprojection check was a no-op on exactly the
measurements it was written for. `check_parallax` has the same shape.

Fixed with a `SafeAcos` that clamps to [-1, 1] (and preserves a genuine
NaN input), plus explicit `!std::isfinite(...)` guards on both gates.
Unit tests are now 34 pass / 1 fail, from 33 / 2.

## The gravity Jacobian used the wrong rotation

```cpp
Mat3 dV_dWsg = -Rsb * SO3::hat(g_);
```

The gravity term of the velocity dynamics is `Rsg · g`, not
`Rsb · anything`. XIVO perturbs on the **right** — `core.h:148` is
`Rsg *= dRsg` — so

    d/dδ [ Rsg · exp(δ) · g ] |δ=0 = −Rsg · hat(g) · δ

The convention is confirmed by the neighbouring block, `dV_dWsb = -Rsb *
SO3::hat(accel_calib)`, which is the same right-perturbation derivative
for the `Rsb · accel_calib` term. That is what makes `Wsg` unambiguously
the odd one out rather than a global sign/convention question. The two
expressions agree only while `Rsb == I`, i.e. at `t = 0`; after that the
column was rotated by `Rsb·Rsgᵗ` relative to the truth.

Costs 0.0040 mean ATE when reverted — inside the noise band, so this is
recorded as a correctness fix. It is also a *gravity-direction* fix, and
gravity direction is a rotation quantity, which is worth remembering
against the RPE_rot stretch goal: it did not help there either
(0.6204 vs 0.6209).

## Three more, dormant under this config

- `DirectLinearTransformAvg` (method 0) inverts a 2×2 that is exactly
  singular when the two bearings are parallel. `Mat2::inverse()` does not
  fail; it returns `inf`/`NaN`.
- `LinfAngular` computes `m - (m·n)n`, which is a projection onto the
  plane perpendicular to `n` only if `‖n‖ = 1`. `n_a`/`n_b` are raw cross
  products, so the correction was scaled by `‖n‖²`. Both `L1Angular` and
  `L2Angular` normalise; only `Linf` forgot. Also guarded against the
  degenerate `‖n‖ → 0`.
- `feature_owner_change_cov_factor` was read under the key
  `filter_owner_change_cov_factor`, which no config defines. 25 configs
  set the real name and were ignored. They all set it to `1.5`, which is
  also the hard-coded default, so this fix changes nothing numerically
  *yet* — the knob only starts to matter once M5 makes `inflate_cov` act
  on the filter covariance instead of a dead local copy.

That is the third dead config key found (after `use_prediction` and
`comparison_score_type`) and it is the reason cross-checking every key
against an actual reader is now a standing part of the audit.
