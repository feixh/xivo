# M3 — first-estimates Jacobians: implemented, measured, neutral

Mono, 6 rooms, `--jitter 6`, `ate_002` / `ov_rpe8_pos_m`. Measured against the
`oos_full` base (`use_OOS` + the tuned `OOS` block), whose reference point is
**0.0875 / 0.0418**.

| arm | keys | ate_002 | rpe8 |
|---|---|---|---|
| base | — | 0.0875 | 0.0418 |
| `fej1` | `fej.mode=1 fej.oos=false` | 0.0845 | 0.0419 |
| `fej1o` | `fej.mode=1 fej.oos=true` | 0.0878 | 0.0432 |
| `fej2o` | `fej.mode=2 fej.oos=true` | 0.0880 | 0.0436 |

`fej1` is -0.0030 on ATE, which is inside the 0.005 m noise floor and flat on
RPE. `fej.oos` and freezing the feature's own parametrization are both slightly
negative. **FEJ is not the missing piece on TUM-VI room1-6.** It stays in the
tree, off by default, because it is correct and cheap and it is the right thing
to reach for on a longer sequence.

## What was built

* `Group::FreezeFEJ()` / `Rsb_fej()` / `Tsb_fej()` (`src/group.h`) — records the
  pose a group had when it entered the state. Frozen once, never refreshed;
  `Group::Reset` clears the flag because groups come from a pool.
* `Feature::FreezeFEJ()` / `x_fej()` (`src/feature.h`) — same for the feature's
  own `(X/Z, Y/Z, log Z)`. `ChangeOwner` re-freezes, because re-anchoring
  re-expresses `x_` in a different frame and the old value is not even in the
  right coordinates.
* `Feature::RelinearizeFEJ` (`src/feature.cpp`) — a *second* pass that rebuilds
  the whole chain at the frozen point and overwrites the `J_` blocks. Deliberately
  not a parameterization of `ComputeJacobian`'s first pass: with `fej_mode_ == 0`
  not one instruction of the original changes, so FEJ-off is provably free.
* The OOS/MSCKF equivalent in `Feature::ComputeOOSJacobianInternal`
  (`src/oos.cpp`), guarded by `fej_oos_`. `cache_.Xs` is *not* substituted: an
  out-of-state point is not a state element, it is re-triangulated from the
  window every update and then projected out by the left nullspace of `Hf`, so it
  has no linearization point to freeze.
* Config: `fej.mode` (0/1/2) and `fej.oos`. `Estimator::AddGroupToState` and
  `AddFeatureToState` do the freezing.

In every case **the residual stays at the current estimate and only the Jacobian
moves**, which is what OpenVINS does (`UpdaterHelper.cpp:353-362`: compute the
residual, then overwrite only the Jacobian inputs with `Rot_fej()`/`pos_fej()`).
The current body pose needs no substitution because `ComputeJacobian` runs before
the measurement update, so `Rsb`/`Tsb` are still the propagated values -- the
mirror of OpenVINS refreshing `_imu`'s fej at every propagation.

## Verification that FEJ-off is a no-op

* `bin/unitTests_Jacobians`, `unitTests_jacobians_stereo`, `unitTests_OOSUpdate`,
  `unitTests_oos_stereo`, `unitTests_ekf_update`, `unitTests_determinism`: pass.
* room1 with the pristine mono config, dumped trajectory `cmp`-identical to
  `position_nochange/mono/room1_r0/dump/tumvi_room1_cam0`. Byte for byte.

## Why it is neutral here, which is the interesting part

**XIVO already fixes the gauge explicitly.** `Estimator::SwitchRefGroup`
(`src/estimator.cpp:1911`) does, for the gauge group:

```cpp
if (group_degrees_fixed_ == 4) {
  P_.block(offset+2, 0, 4, err_.size()).setZero();
  P_.block(0, offset+2, err_.size(), 4).setZero();
}
```

`offset+2` is the yaw component of that group's `Wsb` and `offset+3..5` its
`Tsb`: exactly the four unobservable directions of a monocular VIO -- global
position and global yaw. FEJ's entire purpose is to stop the linearized system
from spuriously acquiring information in those four directions. XIVO pins them by
construction instead. The two are alternative remedies for one disease, and the
one already in place is the blunter but more direct of the two.

**Anchors are short-lived.** FEJ only differs from relinearizing when a state
element's estimate has moved appreciably since it entered the state. XIVO's
`max_group_lifetime` is 60 frames (3 s at 20 Hz), features are re-anchored by
`ChangeOwner` when their group is dropped, and re-anchoring re-freezes. So the
frozen anchor is at most 3 s old and usually much less -- there is little drift
between freeze and use for FEJ to protect.

**Per sequence it is a wash.** `fej1`: 0.0773 / 0.0938 / 0.1203 / 0.0587 /
0.0962 / 0.0607, against the base's 0.0812 / 0.0967 / 0.1265 / 0.0616 / 0.1023 /
0.0568. Five sequences improve by 0.003-0.006 and one (room6) loses 0.004 -- a
uniform small shift, not a mechanism firing on some sequences and not others.
That is what "correct but not binding" looks like.

## Consequence for the working hypothesis

The coordinator's read was that FEJ is the precondition for fixing filter
over-confidence, and that over-confidence explains both the MH gate destroying
good features and the bias states refusing to be loosened. The first half of that
did not survive contact: FEJ is implemented, correct, and worth 0.003 m. If
over-confidence is the disease then its source is *not* the relinearization
point, and the next place to look is where a feature's covariance is created --
see `m5-consistent-init.md`.
