# M2 — filter-model fixes

Two defects in code that runs on every frame of the mono+IMU pipeline.
Commits `431da19` and `732f052` on `auto-bugfix`.

## Fix 1 — `Feature::FillJacobianBlock` dropped the reference-group translation Jacobian

`src/feature.cpp:688`. Before:

```cpp
H.block<2, 3>(offset, goff) = J_.block<2, 3>(0, goff);
H.block<2, 3>(offset, goff) = J_.block<2, 3>(0, goff + 3);   // same destination
H.block<2, 3>(offset, foff) = J_.block<2, 3>(0, foff);
```

The second line writes the *translation* Jacobian into the *rotation*
columns. Net effect per measurement row-pair:

| columns | should hold | actually held |
|---|---|---|
| `goff .. goff+2` (ref group rotation) | ∂r/∂Wsb_ref | ∂r/∂Tsb_ref |
| `goff+3 .. goff+5` (ref group translation) | ∂r/∂Tsb_ref | **0** |

Zero, not garbage, because `FilterUpdate` zeroes `H` once per update.

`Feature::ComputeJacobian` was always correct — it writes `goff` and
`goff + 3` — so `J_` held the right values and only the copy into the
stacked `H` was wrong. That is exactly why the shipped test suite missed
it: all 13 tests in `unittest_jacobians_instate.cpp` inspect `J_`, and
nothing called `FillJacobianBlock`.

The filter consequence is not subtle. `H` is the only thing the EKF sees:
it drives the innovation covariance, the Kalman gain, the MH gating
threshold, and the Joseph-form covariance update. With the reference
group's translation column zeroed, the filter believed a measurement of a
feature carried *no information about where its anchor pose is* — so it
never corrected anchor translation from vision, and it applied the
rotation correction with the wrong sensitivity matrix.

### Regression test

`InstateJacobiansTest.FillJacobianBlockCopiesEveryBlock`. Checks every
destination block of `H` against the matching columns of `J_`, at a
non-zero row `offset`, plus two guards that make the test non-vacuous:

```cpp
EXPECT_FALSE(f->J_.block(0, goff, 2, 3).isApprox(f->J_.block(0, goff + 3, 2, 3)));
EXPECT_FALSE(f->J_.block(0, goff + 3, 2, 3).isZero());
```

Without those, a test written against the buggy code would still pass.
Mutation-checked: restored the old two-lines-to-`goff` version and
confirmed the test fails, then restored the fix. `unitTests_Jacobians`
is 14/14.

(Implementation note: `EXPECT_TRUE(H.block<2, 3>(...)...)` does not
compile — the comma in `block<2, 3>` is read as a macro argument
separator. The test uses a `block_matches(int col)` lambda over the
dynamic-size `.block(r, c, 2, 3)` instead.)

## Fix 2 — `Estimator::AdaptInitialDepth` did not take a median

`src/manager.cpp:271`. `depth[depth.size() >> 1]` on an unsorted vector
is the middle element of the graph's traversal order, not the median.
Fixed with `std::nth_element` (only the middle element needs to be in
place).

Live in the mono config: `UpdateStep` calls it unconditionally every
frame, `median_weight` is 0.99, and the result becomes `init_z_`, the
initial depth of every subsequently created feature
(`manager.cpp:586`/`:590`).

## Measurement

`cfg/sweep_dlt_nodesc`, `XIVO_RANDOM_SEED=0`, TUM-VI room1–room6, mono
cam0 + IMU. Ablated so the gain can be attributed.

| | mean ATE | mean RPE_rot | mean RPE_tra |
|---|---|---|---|
| M0 baseline (`auto`) | 0.1261 | 0.6227 | 0.0364 |
| + fix 1 only (`results/bugfix/m2a_jac_only/`) | **0.1019** | 0.6205 | **0.0269** |
| + fix 1 and 2 (`results/bugfix/m2_jac_median/`) | 0.1041 | 0.6207 | 0.0282 |

Per sequence, fix 1 alone:

| seq | M0 ATE | M2a ATE |
|---|---|---|
| room1 | 0.1219 | 0.1336 |
| room2 | 0.1199 | 0.0684 |
| room3 | 0.2165 | 0.1549 |
| room4 | 0.0846 | 0.0911 |
| room5 | 0.1240 | 0.0992 |
| room6 | 0.0897 | 0.0639 |

**The whole −19% is fix 1.** Fix 2 moves the mean by +0.0022, which is
inside the ±0.005 binary-layout noise band established in
`m0-baseline.md`, so it is not evidence of anything either way; it is
committed as a correctness fix.

room1 and room4 regressing while room2/3/5/6 improve sharply is the
expected signature of a genuine model fix here: every threshold in
`sweep_dlt_nodesc.json` (MH gating χ², outlier thresholds, initial
covariances, `Rr`) was tuned by the sweep *against the broken `H`*.
Re-tuning against the corrected model is M6's job, and the two regressing
sequences are the reason it is worth doing rather than optional.

RPE_rot barely moves (0.6227 → 0.6205). The stretch goal of < 0.5° is a
rotation problem and these fixes were not rotation problems; that needs
its own investigation.
