# M5 — consistent feature initialization: the change that met the target

Mono, 6 rooms, `--jitter 6`, `ate_002` / `ov_rpe8_pos_m`.

| arm | base | ate_002 | rpe8 |
|---|---|---|---|
| `combo_subpix` | — | 0.0605 | 0.0326 |
| **`c_ci`** | `combo_subpix` + `consistent_init.enable` | **0.0563** | **0.0271** |
| `c_ci_ready1` | `c_ci` + `subfilter.ready_steps=1` | 0.0568 | 0.0260 |

`c_ci` per sequence, ATE: 0.0578 / 0.0467 / 0.0628 / 0.0393 / 0.0949 / 0.0363.
RPE-8m: 0.0218 / 0.0188 / 0.0318 / 0.0273 / 0.0343 / 0.0283.

Both mono targets (`ate_002` <= 0.061, `rpe8_pos` <= 0.030) are met here for the
first time. -0.0042 on ATE is at the edge of the 0.0067 m sd of the 6-room mean,
but -0.0055 on RPE (a 17% relative cut) is well clear of it, and the two agree in
sign on five of six sequences, so this is a mechanism and not a lucky draw.

## The bug, stated precisely

`Estimator::AddFeatureToState` gave a promoted feature its covariance with
`Feature::FillCovarianceBlock` (`src/feature.cpp:1156`), which does two things:

1. copies the depth sub-filter's 3x3 `P_` into the feature's block of the big
   covariance, and
2. **zeroes every cross-covariance** between the new feature and all 561 other
   error-state dimensions.

Both are over-confident, for different reasons.

**(1) The sub-filter's 3x3 is a conditional covariance.**
`Feature::SubfilterUpdate` (`src/feature.cpp:365`) forms

```cpp
Mat2 S = H * P_ * H.transpose();
S(0,0) += Rtri; S(1,1) += Rtri;
```

`H` is d(pixel)/d(x_) only. There is no `Hx P Hx'` term, so the sub-filter treats
both the reference pose and the current pose as *exactly known*. The 3x3 it
converges to is therefore `cov(x_ | poses)`, not `cov(x_)`. Handing it to the
filter as if it were the marginal is a strict understatement of the depth
uncertainty, and the understatement is largest exactly when the anchor pose is
poorly known — which on a monocular run is always, early.

**(2) Zero cross-covariance is a false independence claim.** The feature's depth
was *inferred* from the anchor group's motion over the frames it was tracked. It
is strongly correlated with that group's pose, with the velocity, and with the
biases that shaped the propagation. Asserting zero says the filter may correct
the pose without correcting the feature, and vice versa — so a pose error that
the feature could have explained gets attributed to something else, usually the
biases.

The tree already knew about this: `APPROXIMATE_INIT_COVARIANCE` exists as an
alternative, and is commented out in `src/CMakeLists.txt:38`. Its implementation
is not usable (it fills the group block and leaves the rest), so it was replaced
rather than revived.

## The fix

The textbook delayed-initialization augmentation, in
`Estimator::InitializeFeatureCovariance`:

```
P_ff = Hl^-1 (sigma^2 I + Hx P Hx') Hl^-T
P_xf = -P Hx' Hl^-T
x   += Hl^-1 res
```

`Hl` (3x3) and `Hx` (3 x kFullSize) come from the new
`Feature::ComputeInitJacobian` (`src/oos.cpp`), which stacks the feature's
measurements over its in-state views by reusing `ComputeOOSJacobianInternal`,
adds back the anchor group's and the extrinsics' contribution *through* the 3D
point (in the OOS path the point is a free variable; here it is a function of the
anchor), and QR-reduces `Hl = Q [R; 0]` to keep the 3 invertible rows. Same
recipe as OpenVINS `StateHelper::initialize_invertible`.

Note what `Hx P Hx'` buys: the pose uncertainty the sub-filter dropped is now
*added* to the feature's block, and `P_xf` is nonzero by construction — the two
defects above are fixed by the same expression.

Guards, all falling back to `FillCovarianceBlock`:

* anchor group not in the state, or `sind() < 0`;
* fewer than `min_views` in-state views;
* `rank(Hl) < 3` at threshold 1e-6, i.e. no parallax;
* non-finite result, non-positive diagonal, or a diagonal entry above `max_var`;
* the residual correction `dx` pushing `exp(x(2)+dx(2))` outside `(min_z_, max_z_)`.

The `[census]` line reports `consistent-init:used/tried`. On room1 that is
**17469/17734 — 98.5%** of promotions take the new path, so the guards are not
quietly disabling it.

## Attribution: it needs the OOS window, and it is not additive

Three arms, the same one key (`consistent_init.enable=true`) on three bases:

| arm | base | ate_002 | rpe8 | delta ATE | delta RPE |
|---|---|---|---|---|---|
| `a_ci_only` | plain baseline (0.0928 / 0.0480) | 0.0985 | 0.0469 | **+0.0057** | -0.0011 |
| `a_ci_oos` | `oos_full` (0.0875 / 0.0418) | **0.0737** | **0.0317** | **-0.0138** | **-0.0101** |
| `c_ci` | `combo_subpix` (0.0605 / 0.0326) | **0.0563** | **0.0271** | **-0.0042** | **-0.0055** |

Read top to bottom that is one clean statement: **the consistent initial
covariance is worthless — slightly harmful — without the out-of-state window, and
it is the single largest effect of any change in this work once the window is
there** (-0.0138 m, well past the 0.0067 m sd). By the time CLAHE and sub-pixel
refinement are also in, part of the same benefit has already been collected by
other means, so the marginal value drops to -0.0042.

Why the sign flips is the mechanism, not an accident. A more honest (larger)
initial covariance means the filter trusts each new feature *less* and lets it be
corrected more. That is good if there are enough good measurements to do the
correcting and bad if there are not — a loose prior on a starved filter just adds
variance. On the plain baseline the MSCKF path was inert (`m1`) and detection was
starved in the periphery (`m4`), so the extra freedom had nothing to spend itself
on. `a_ci_oos` is the decisive arm: it adds *only* measurement supply, no
front-end change, and that alone turns +0.0057 into -0.0138.

The practical consequence for merging: **`consistent_init.enable` must not be
turned on without the `OOS` block.** They are one config, not two independent
knobs.

## Nothing further moves it

Eight increments on top of `c_ci` (0.0563 / 0.0271). None clears the 0.005 m
floor, in either direction:

| arm | key | ate_002 | rpe8 |
|---|---|---|---|
| `d_maxvar` | `consistent_init.max_var=100` | 0.0551 | 0.0264 |
| `d_mv3` | `consistent_init.min_views=3` | 0.0556 | 0.0268 |
| `d_fej1` | `fej.mode=1` | 0.0562 | 0.0261 |
| `d_mh99` | `MH_thresh=9.21` | 0.0567 | 0.0268 |
| `d_std15` | `consistent_init.meas_std=1.5` | 0.0573 | 0.0270 |
| `c_ci_ready1` | `subfilter.ready_steps=1` | 0.0568 | 0.0260 |
| `d_epi` | `tracker_cfg.epipolar_rejection` | 0.0612 | 0.0260 |

So all three of `consistent_init`'s own tuning knobs stay at their code defaults
(`min_views` 2, `meas_std` = `visual_meas_std`, `max_var` 1e4): tightening or
loosening the acceptance guards changes nothing, which is the reassuring outcome
— it means the 98.5% acceptance rate is not carrying marginal, barely-invertible
geometries that a stricter test would have to filter.

`d_epi` is the only clearly negative one, and it is the same +0.005 ATE /
-0.007 RPE trade the epipolar test showed in `m4`. It stays off.

## What this says about the coordinator's hypothesis

The coordinator's read was: one root cause (filter over-confidence) explains the
MH gate destroying good features, the bias states refusing to be loosened, and
`SubfilterUpdate`/`FillCovarianceBlock`; and FEJ is the precondition for fixing
it. Score:

* **Over-confidence: confirmed, and located.** It is in how a feature's
  covariance is *created*. Fixing that is worth -0.0042 ATE / -0.0055 RPE.
* **FEJ as the precondition: not confirmed.** FEJ is implemented and correct and
  worth 0.003 m, inside the noise floor (`m3`). It is not a precondition for
  anything here — `c_ci` was measured with `fej.mode=0`, and adding FEJ on top of
  `c_ci` does not help either.
* **The MH gate: half confirmed, and still not worth touching.** Loosening the
  gate to 99% costs +0.0031 ATE without consistent init (`c_mh99` 0.0636 vs
  0.0605) and **nothing at all with it** (`d_mh99` 0.0567 / 0.0268 vs `c_ci`
  0.0563 / 0.0271). So the coordinator's link is real in one direction: an
  over-confident covariance is what made a wider gate actively harmful, and
  fixing the covariance removes the penalty. But it does not turn into a gain —
  those 4-5 rejected features per frame are not features a better covariance
  would have saved, and `MH_max_strikes > 1` stays negative (`m2`). The gate is
  left at the shipped 5.991.

So: same disease, one of the three organs, and not the one FEJ treats.
