# M1 — gating from the 25 columns a measurement can reach

Commit: `M1: gate on the 25x25 slice of P a measurement can reach`.

## The observation

`Feature` stores its measurement Jacobians full width:

```c++
Eigen::Matrix<number_t, 2, kFullSize> J_, J_r_;   // 2 x 564 each
```

but `Feature::ComputeJacobian` writes only these column blocks:

| block | columns | width |
| --- | --- | --- |
| `Wsb`, `Tsb` | 0..5 | 6 |
| `bg` | 9..11 | 3 |
| `Wbc`, `Tbc` | 15..20 | 6 |
| `td` | 23 | 1 |
| its reference group | `kGroupBegin + 6·gsind` | 6 |
| itself | `kFeatureBegin + 3·fsind` | 3 |

**25 of 564.** A measurement of one feature is structurally independent of `Vsb`,
`ba`, `Wsg`, of every group but its own reference, and of every other feature.
(`Vsb` and `ba` enter the *dynamics*, not the projection; `Wsg` only rotates
gravity. `bg` and `td` are nonzero only because of online temporal calibration —
they are the derivative of the td-corrected pose.)

Both Mahalanobis gates evaluated the quadratic form densely anyway:

```c++
Mat2 S = J * P_ * J.transpose();        // 2x564 * 564x564 * 564x2
```

once per in-state feature in `MHGating` (76 per frame) and again per right
observation in `GateStereoMeasurements` (74 per frame). Each of those reads all
**2.5 MB** of `P_`, so ~150 traversals of a 2.5 MB matrix per frame: 380 MB/frame
of pure bandwidth, to produce 4 numbers each time. It is not the flops that hurt
(1.5 MFLOP per call) but that `P_` does not fit in L2.

## The change

`src/core.h` gains the column-run description of a measurement, next to `Index`
and under the same `#ifdef`s as `ComputeJacobian` so the two cannot drift:

* `ColRun{start, len}` and `kJacSharedRuns` — the runs common to every
  measurement, with the layout-adjacent ones merged (`static_assert`s enforce the
  adjacency), ending in a sentinel that `MeasurementRuns(gsind, fsind, …)`
  replaces with the feature's actual group run before appending its feature run;
* `kJacCols` = 25 (computed, not written down), with `JacCompact` = 2x25 and
  `CovCompact` = 25x25 fixed-size types;
* `GatherCols` / `GatherCov` — the gather of a row block's columns and of `P`'s
  symmetric submatrix;
* `InnovationCov(J, P, gsind, fsind, R)` — `J P J^T + R I` from that slice. A
  free function, not an `Estimator` member, so the test can hand it an arbitrary
  `P` and compare against the dense product.

The three gates (`MHGating`, `GateStereoMeasurements`,
`OnePointRANSAC`'s per-hypothesis gate) now call it. Zero-but-present runs are
kept rather than trimmed: including a zero column changes no result, while
omitting a nonzero one silently corrupts the filter.

This is algebraically exact — every skipped product has a structurally zero
factor — but not bit-identical *as an expression*: compacting changes which
nonzero products land in the same accumulation block inside Eigen's gemm, so the
sums are reassociated at the 1e-16 level. On a filter with hard gating that is
normally enough to diverge a trajectory (see
`notes-bugfix/m6-numerics-and-plumbing.md`), so the accuracy check below was
planned as an ensemble.

It turned out not to need one, for a reason worth writing down: **`S` is used
only to make a decision.** All three call sites compute a Mahalanobis distance
and compare it to a fixed threshold — `S` never enters the state or covariance
update, which recompute nothing from it. A 1e-16 perturbation of `S` can
therefore change the trajectory only by flipping an accept/reject at a
gate, and a flip requires a distance within 1e-16 of the threshold. None
occurred: over both ensembles (2 settings x 8 members x 6 rooms, 96 runs) the
estimated trajectories are **identical to the baseline's in every digit the
output records** — all 96 `tumvi_<seq>_cam0` files match byte for byte
(`filecmp`, absolute paths).

That is worth stating precisely rather than as "bit-identical": `savers.py` writes
those files with `np.savetxt(fmt='%f')`, so they carry 6 decimals, ~1 µm and
~1e-6 in the quaternion. The files cannot distinguish an exactly-identical filter
state from one that differs at 1e-16, and the *argument* above is what says it is
the former. What the ensemble adds is the empirical half: no gate flipped in 96
runs, which is the only way this change could have shown up at all. Still a far
stronger non-degradation result than matching ensemble means, and specific to this
milestone — M2 changes the update arithmetic itself, and does not have it
everywhere (5 of its 96 runs diverge).

(Compare `tumvi_<seq>_cam0`, not `tumvi_<seq>_bench`: the latter is a two-line
header that is identical across every arm by construction.)

## Tests

Four new cases in `unitTests_jacobians_stereo` (the binary that has the real
fisheye pair and a `StereoRig`, so it can check both cameras' rows):

| test | what it pins |
| --- | --- |
| `MeasurementRunsCoverEveryLiveBlock` | every column the finite-difference tests find live is inside a run, and the runs sum to `kJacCols` |
| `NothingOutsideTheMeasurementRunsIsNonzero` | at slots 7/33, `J_` and `J_r_` are *exactly* zero outside the runs |
| `CompactInnovationCovMatchesTheDenseProduct` | compact vs dense `S`, both cameras, slots 0/0, 7/33 and 44/89, against a random dense SPD `P`: relative error < 1e-12 |
| `CompactInnovationCovDependsOnTheRightSlots` | reading a *wrong* slot changes the answer, so the test above cannot pass vacuously |

The first two are complements: the first alone would pass if the runs covered the
whole state, the second alone if they covered nothing. Together they pin
`kJacSharedRuns` to exactly what `ComputeJacobian` writes.

20/20 pass in that binary; the rest of the suite is unaffected.

## Speed

`sweeps/m1_gating.log`, arms interleaved, two repeats, baseline = the same commit
without M1 (worktree `xivo`, same capacity and flags):

| arm | seq | wall (s) | **FPS** | mh (ms) | stereo_gating (ms) | process_tracks (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| base_mono | room1 | 134.1 | 21.04 | 9.02 | — | 32.58 |
| m1_mono | room1 | 107.9 | **26.13** | 0.10 | — | 23.44 |
| base_mono | room6 | 127.7 | 20.64 | 9.31 | — | 33.76 |
| m1_mono | room6 | 102.7 | **25.68** | 0.11 | — | 24.25 |
| base_stereo | room1 | 228.7 | 12.34 | 9.07 | 8.34 | 52.66 |
| m1_stereo | room1 | 177.0 | **15.94** | 0.12 | 0.08 | 34.99 |
| base_stereo | room6 | 218.1 | 12.08 | 9.39 | 8.83 | 55.06 |
| m1_stereo | room6 | 168.9 | **15.60** | 0.13 | 0.09 | 36.62 |

Averaged over the two sequences: **mono +24.3% FPS, stereo +29.2% FPS**
(`harness/tab.py`, `FPSx` column). Per-frame timings are the estimator's own
`print_timing` averages, so they are directly attributable.

The gate cost itself is gone, not reduced: 9.0 ms → 0.11 ms for MH-gating
(**~80x**) and 8.3 ms → 0.09 ms for stereo gating. What is left is the gather
plus 2x25x25x2 flops, which is L1-resident. The savings land in
`process_tracks`, which contains both gates; nothing else moves, and RSS is
unchanged (the compact types are stack temporaries).

Note how much this says about where the time was: 9 ms of a 48 ms mono frame was
being spent reading a covariance to produce 76 scalars.

## Accuracy

Baseline arm `xivo-effbase` (detached at `d13ec97`) vs. this commit, 8-member
ensembles perturbed in `X.Vsb` (`run_ensemble_bugfix.sh`), 6 rooms each,
threads pinned to 1:

| | ATE | RPE_rot | RPE_tra | RPE_rot_i | RPE_tra_i |
| --- | --- | --- | --- | --- | --- |
| base_mono | 0.0796 ± 0.0063 | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| m1_mono | 0.0796 ± 0.0063 | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| base_stereo | 0.0551 ± 0.0031 | 0.6208 | 0.0139 | 0.5128 | 0.0132 |
| m1_stereo | 0.0551 ± 0.0031 | 0.6208 | 0.0139 | 0.5128 | 0.0132 |

Identical because the runs are identical: **all 96 trajectory files**
(`m*/tumvi_room*_cam0`, 2 settings x 8 members x 6 rooms) match byte-for-byte, for
the reason given above. `_i` is
`evaluate_rpe_interp.py`; the stock RPE numbers are reported alongside because
the stock evaluator's quantization floor is 0.28 deg (see
`notes-bugfix/`).

The stereo ensemble for this commit is measured from the frozen worktree
`xivo-effm1` rather than from `xivo-efficiency`, after an earlier attempt was
discarded: the first stereo arm was launched one second before M2's `pyxivo.so`
was relinked in that same worktree, so rooms 2-6 of every member imported the
*next* milestone's binary. The mono arm survived scrutiny only because
bit-identity to the baseline is impossible under M2's arithmetic, which proves it
ran the intended binary. **Rule adopted from here on: an arm is measured from a
worktree checked out at that commit and never built into again while a run is
live.**
