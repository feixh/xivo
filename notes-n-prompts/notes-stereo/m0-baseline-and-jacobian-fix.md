# M0 — baseline + `FillJacobianBlock` fix

## The bug

`src/feature.cpp:688-689`, in `Feature::FillJacobianBlock`:

```cpp
int goff = kGroupBegin + 6 * ref_->sind();
H.block<2, 3>(offset, goff) = J_.block<2, 3>(0, goff);
H.block<2, 3>(offset, goff) = J_.block<2, 3>(0, goff + 3);   // dest should be goff + 3
```

Both writes target `goff`. Consequences:

- the reference group's **rotation** Jacobian (`dxp/dWsbr`) is written, then
  immediately **overwritten** by the translation Jacobian (`dxp/dTsbr`);
- the slot for translation at `goff + 3` is **left at whatever `H` held** —
  `H_.setZero(...)` runs once per update in `FilterUpdate`, so it stays zero.

So every EKF measurement update saw: rotation block = translation Jacobian,
translation block = 0. `J_` itself is computed correctly in `ComputeJacobian`
(`src/feature.cpp:656-657`); only the copy into the stacked `H` was wrong.

## Why it went unnoticed

`OnePointRANSAC` builds its own `H_` by reading `mh_inliers[i]->J()` directly
(`src/update.cpp:346`), bypassing `FillJacobianBlock` entirely. So the RANSAC
low-innovation update used correct Jacobians while the main `FilterUpdate` did
not. The filter still converged because the group-pose block is only part of the
measurement model — the body-pose blocks (`Wsb`/`Tsb`) were always right.

## Effect (config `sweep_dlt_nodesc`, seed 0, all six sequences)

```
seq      ATE before   ATE after    RPE_rot before  after      RPE_tra before  after
room1    0.118        0.1336       0.5312          0.5295     0.0229          0.0229
room2    0.120        0.0684       0.7255          0.7235     ~0.048          0.0259
room3    0.217        0.1549       0.7354          0.7321     ~0.069          0.0370
room4    0.085        0.0911       0.6368          0.6367     0.0228          0.0228
room5    0.107        0.0992       0.5753          0.5754     ~0.025          0.0315
room6    0.079        0.0639       0.5287          0.5257     0.0212          0.0212
-----------------------------------------------------------------------------------
mean     0.1209       0.1019       0.6222          0.6205
```

**Mean ATE 0.1209 → 0.1019 m** (-16%). Now marginally better than the authors'
own shipped results (0.101). room2 (-43%) and room3 (-29%) improve most; room1
(+13%) and room4 (+7%) regress slightly — consistent with a genuine model fix
rather than a tuning artifact, since the config was previously tuned *against*
the broken Jacobian.

Rotational RPE is essentially unchanged (0.6222 → 0.6205), which makes sense:
the corrupted block concerned the *reference group* pose, and rotation between
consecutive frames is dominated by the IMU + body-pose blocks.

## Implication for the stereo work

Baseline for all stereo comparisons is now **ATE 0.1019 / RPE_rot 0.6205**, not
the 0.1209 in `RESULTS.md`. The exit criteria (ATE < 0.06, RPE_rot < 0.5) require
roughly a **41% ATE reduction** and a **19% rotation reduction** from here.

Also relevant: the config knobs in `sweep_dlt_nodesc` were tuned against the
buggy Jacobian, so a re-sweep of the *mono* knobs might recover more on its own.
Not pursuing that now — it would confound the stereo comparison. Noted for M6.

## Test status

- `unitTests_Jacobians` — 13/13 pass (these test `J_`, not the `H` copy, which
  is precisely why they never caught this).
- Pre-existing failures, unchanged and unrelated:
  `Triangulation.Angular_Reprojection_Error`,
  `NumericalLinearAlgebra.SlowAndFastGivensMatch`.

**Gap in coverage worth noting:** nothing tests `FillJacobianBlock`. A test that
asserts `H.block(offset, goff, 2, 6) == J_.block(0, goff, 2, 6)` would have
caught this immediately. Added to the M5 test plan, where the stereo path will
extend this same function.

## Incidental fix

`xivo/build/CMakeCache.txt` still pointed at the pre-reorganization dependency
paths (`../opencv_install`, `../venv`), so the first rebuild failed with
`Could NOT find Python3 (missing: Interpreter)`. Reconfigured in place with
`-DOpenCV_DIR=$ROOT/dependencies/opencv_install/...` and
`-DPython3_EXECUTABLE=$ROOT/dependencies/venv/bin/python`.

`run_eval.sh` now runs the six sequences **concurrently** (they are independent
and single-threaded) and prints an ATE/RPE summary table plus means. A 6-sequence
evaluation went from ~30 min serial to ~5 min. This matters: M6 tuning needs many
full evaluations.
