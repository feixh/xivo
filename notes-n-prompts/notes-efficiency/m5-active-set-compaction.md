# M5 — the update on the occupied extent, not on the capacity

Commit: `M5: restrict the EKF update to the occupied extent of the state`
(`e2e6f0b`).

## What was there

After M2 the update is one rank-`m` symmetric downdate of `P_`:

```c++
P.selfadjointView<Eigen::Lower>().rankUpdate(M.transpose(), -1.0);
for (int j = 1; j < P.cols(); ++j)
  P.block(0, j, j, 1) = P.block(j, 0, 1, j).transpose();
```

`P_` is `kFullSize` = 564 square, because `kFullSize` is the *compile-time
capacity*: 24 motion + 6x45 group + 3x90 feature. The census says the run-time
occupancy is nowhere near that:

| | feature slots | group slots | occupied dim |
| --- | --- | --- | --- |
| mono, mean over room1-6 | 74.6 / 90 | 7.9 / 45 | 295 / 564 |
| stereo, mean over room1-6 | 74.1 / 90 | 8.1 / 45 | 295 / 564 |

The feature slots are nearly full — the tracker is asked for 135–180 features and
the filter takes 90 — but the *group* slots are not: 8 of 45. Groups are keyframes,
and XIVO retires them aggressively. That one column is where the gap comes from:
37 unused group slots is 222 dimensions, 39% of the state, that the downdate reads
and writes on every frame for nothing.

So the arithmetic the downdate does is `564^2 * m / 2`, and the arithmetic it needs
is `295^2 * m / 2` — a 3.7x gap on the single most expensive step remaining in the
system (3.2 ms mono / 7.2 ms stereo per frame after M4). It also reads and writes
2.5 MB per update where 0.7 MB would do, and this step is bandwidth-bound, which is
exactly why M2's 17x arithmetic reduction only converted to 7x.

## The premise, and the one I got wrong first

The compaction is exact, not approximate, but the reason is not the obvious one.

**The wrong version.** "A slot that is not occupied is a zero row and column of
`P_`." `RemoveGroupFromState` and `RemoveFeatureFromState` do zero the whole row
and column; `Feature::FillCovarianceBlock` does too before writing its own block.
So the claim is *nearly* true, and I wrote it into `core.h`, into the header, and
into an `#ifndef NDEBUG` check — and the check aborted on the first update of the
first sequence:

```
Check failed: P_.row(i).cwiseAbs().maxCoeff() == 0 (1 vs. 0)
  row 30 is outside the occupied extent but nonzero
```

`estimator.cpp:347` is `P_.setIdentity(kFullSize, kFullSize)`, and only the motion
and calibration blocks get scaled from the config afterwards. Every group and
feature slot therefore starts at variance **1** and stays there until something
uses it. Row 30 is group slot 1, which nothing had touched yet.

**The right version.** What the update needs is not that the vacant part is zero
but that it is *uncorrelated*. Write `L` for the live set. For `i` outside it,
`P(i, ·)` is supported on `{i}` and `H(:, i)` is zero — a measurement cannot
reference an unoccupied slot, which is M1's property restated. Then

```
M(:, i)  =  H P e_i  =  P(i,i) H(:, i)  =  0
```

so `S = M H^T` does not see column `i`, `W = L^-1 M` has a zero column there,
`err_i = W(:, i)^T u = 0`, and `-W^T W` contributes exactly nothing to any entry in
row or column `i`. The vacant part of `P` comes out of the update bit-for-bit as it
went in — which is also what the dense form does, so skipping it is a
rearrangement and not a change of answer.

That the wrong premise and the right one lead to the same code is not luck; it is
why the bug was worth catching. Had I not checked, the note would have carried a
false statement about the filter, and the next person to reason from it — say, to
add a term that gives a vacant slot a cross-covariance — would have had no warning.

The check is still there, now testing the two conditions that actually matter
(`CheckLiveExtent` in `ekf_update.cpp`), and it is reachable from an optimized
build:

```
cmake -DCMAKE_CXX_FLAGS=-DXIVO_CHECK_OCCUPIED_STATE ..
```

It runs clean over all six rooms in both settings, ~0.4 ms per update, which is far
too much to leave on and far too little to justify a Debug build.

## Why two runs is all it takes

There is no gather and no scatter here, which is the part of the design worth
explaining, because the obvious implementation is a permutation.

`AddGroupToState` and `AddFeatureToState` both allocate the **lowest free slot**
(`for (index = 0; index < gsel_.size() && gsel_[index]; ++index);`). Live slots
therefore pack toward index 0, and the occupied region is bounded by two high-water
marks:

```
[0, kGroupBegin + kGroupSize * groups_used)   motion + intrinsics + groups
[kFeatureBegin, kFeatureBegin + kFeatureSize * features_used)
```

— two contiguous runs, merging into one when every group slot is in use, because
`kGroupBegin == kCameraBegin + kMaxCameraIntrinsics` makes the motion and group
regions adjacent. `StateRuns` in `core.h` is that pair; `Estimator::OccupiedState`
computes it from 135 bool tests over `gsel_` and `fsel_`.

Every step of the update is then expressed on run blocks of the existing `P_`:

| step | before | after |
| --- | --- | --- |
| `M = H P` | `H.block(...) * P.middleRows(run)`, full width | same, restricted to the live column runs |
| `S = M H^T` | sum over measurement runs | same; dense blocks sum over live runs instead of all 564 |
| `L^-1 M` | one solve, 564 columns | one solve per run |
| `P -= W^T W` | one 564-square `rankUpdate` | one `rankUpdate` per diagonal run + one gemm per off-diagonal pair |
| mirror | 563 scalar column copies | per-run column copies + one blocked transpose per pair |

The alternative — permute `P_` into a dense `dim x dim` scratch, update, permute
back — reaches the true occupied dimension (295) rather than the high-water extent,
but it needs a permutation vector, two O(N^2) copies per update, and a second index
space for `H` and `meas_blocks_` to live in. The extent version gets

```
extent 339 of 564  ->  (564/339)^2 = 2.77x   on the quadratic terms
ideal  295 of 564  ->  (564/295)^2 = 3.65x
```

Put as work removed rather than as a ratio of ratios: the extent form drops 64% of
the downdate's arithmetic and the permutation form would drop 73%, so this is
**~88% of the achievable saving** with no new index space, no per-update copies, and
no possibility of a permutation bug. `live-dim` is now printed in the census, so
this is measured rather than argued:

| | occupied-dim | **live-dim** | live-runs |
| --- | --- | --- | --- |
| mono room1 | 296.4 | 339.0 | 2 |
| mono room2 | 304.3 | 345.3 | 2 |
| mono room3 | 284.6 | 329.9 | 2 |
| mono room4 | 279.6 | 330.8 | 2 |
| mono room5 | 280.8 | 330.6 | 2 |
| mono room6 | 324.5 | 358.8 | 2 |
| stereo room1 | 296.1 | 339.8 | 2 |
| stereo room2 | 303.1 | 342.9 | 2 |
| stereo room3 | 283.7 | 330.9 | 2 |
| stereo room4 | 279.1 | 329.0 | 2 |
| stereo room5 | 278.5 | 329.7 | 2 |
| stereo room6 | 326.6 | 364.1 | 2 |

The extent is consistently ~45 dimensions above the occupancy — 15 feature slots'
worth — which is what a high-water mark costs when features are retired from the
middle of the range and re-allocated at the bottom. `live-runs` is 2 in every
update of every sequence, so the merged single-run case is dead code on TUM-VI and
exists for the capacity-saturated configuration.

Note also what `live-dim` is *not*: it is not 564 minus the unused capacity in some
average sense, it is the extent as of each individual update, averaged. The census
prints both so the difference between "how much is occupied" and "how much a
high-water mark has to cover" stays visible.

## Runs are a covering, not a characterization

A run may contain vacant slots — a feature freed below the high-water mark, before
the allocator refills it. Including such a slot costs arithmetic on a zero row and
column and changes nothing; *excluding* a live one would silently corrupt the
filter. The implementation errs in that direction by construction (it never trims a
run), and `OversizedRunsGiveTheSameAnswer` pins it by running the same update with
deliberately inflated marks.

This is the same discipline as M1's `kJacSharedRuns`, and for the same reason: for
this kind of optimization the two failure modes are not symmetric, so the code
should only be able to fail the cheap way.

## Tests

`unitTests_ekf_update` grows from 7 cases to 11. The fixture changed too: `MakeP`
now gives vacant slots a variance (`kVacant = 0.75`, deliberately not 1 so a result
that lands on 1 cannot pass by accident) with no cross terms, which is the shape
`Estimator` actually produces. The old fixture's exact zeros would have let the
wrong premise pass.

| test | what it pins |
| --- | --- |
| `CompactingToTheOccupiedExtentChangesNothing` | the milestone's claim: the compacted update vs. the same update over all 564 dimensions, at the census shape with a dense out-of-state block, 1e-13 — and the vacant tail *exactly* unchanged in both, since there the answer is "untouched", not "recomputed" |
| `OversizedRunsGiveTheSameAnswer` | tight marks vs. inflated ones, 1e-13 |
| `OccupiedState.RunsAgreeWithTheLiveIndexSet` | the runs enumerate exactly the indices the fixture fills, swept over every group occupancy 0..45 and six feature counts; plus ascending, disjoint, `nruns <= 2`, and `dim` consistent |
| `OccupiedState.TheWholeStateIsOneRun` | `WholeState()`, and that a saturated occupancy degenerates to it |

Every pre-existing case still runs, now through the compacted path, and their
`MatchesJoseph*` comparisons are against the **dense** Joseph form over all 564
dimensions — so the four of them are themselves compaction tests. Their vacant-slot
assertion became `== kVacant` instead of `== 0`.

21/21 targets pass under `ctest`.

## Speed

`sweeps/m5.log` — four arms interleaved inside each (sequence, repeat), two repeats,
one thread, `setarch -R`, from the frozen worktrees `xivo-effm4` (`48a5f54`) and
`xivo-effm5` (`e2e6f0b`).

| arm | seq | wall (s) | **FPS** | actual_update (ms) | update (ms) | visual_meas (ms) | track (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m4_mono | room1 | 37.2 | 75.90 | 3.20 | 3.24 | 8.02 | 3.82 |
| **m5_mono** | room1 | 33.5 | **84.12** | **1.86** | 1.89 | 6.69 | 3.84 |
| m4_stereo | room1 | 77.7 | 36.31 | 7.01 | 7.17 | 19.01 | 10.66 |
| **m5_stereo** | room1 | 70.1 | **40.25** | **4.34** | 4.49 | 16.32 | 10.66 |
| m4_mono | room6 | 33.8 | 77.95 | 3.33 | 3.36 | 7.69 | 3.44 |
| **m5_mono** | room6 | 30.4 | **86.75** | **2.02** | 2.05 | 6.37 | 3.43 |
| m4_stereo | room6 | 70.9 | 37.19 | 7.37 | 7.53 | 18.48 | 9.87 |
| **m5_stereo** | room6 | 64.2 | **41.06** | **4.78** | 4.94 | 15.86 | 9.84 |

| | vs. M4 | vs. baseline (chained) |
| --- | --- | --- |
| mono | **1.111x** | **4.09x** |
| stereo | **1.106x** | **3.31x** |

All of it lands in `actual_update`, and nothing else moves: `track` is 3.82 → 3.84
and 10.66 → 10.66, `propagation` 0.17 → 0.17, `mh` 0.10 → 0.10. That is the right
shape — M5 touches one function.

### 1.7x on the update, not the 2.8x the extent ratio suggests

`actual_update` goes **3.20 → 1.86 ms** mono (1.72x) and **7.01 → 4.34** stereo
(1.62x). `(564/339)² = 2.77x` is the ratio on the *downdate*, and quoting it for the
whole update was my own sloppiness — only one of the five steps is quadratic in the
state dimension. In multiply-add units at the mono shape (`m = 152` rows,
`live = 339`, `N = 564`):

| step | cost | at N=564 | at live=339 |
| --- | --- | --- | --- |
| `M = H P` | `m · 25 · live` | 2.14 | 1.29 |
| `S = M Hᵗ` | `m² · 25` | 0.58 | 0.58 |
| Cholesky of `S` | `m³/6` | 0.59 | 0.59 |
| `L⁻¹M` | `m² · live / 2` | 6.51 | 3.92 |
| `P -= WᵗW` | `m · live² / 2` | 24.2 | 8.73 |
| **total (MFLOP)** | | **34.0** | **15.1** |

So the arithmetic ratio is **2.25x**, not 2.77x: the triangular solve is only *linear*
in `live` and is a sixth of the work, and `S` plus its Cholesky (1.17 MFLOP) do not
know the state dimension at all — 3.4% of the update before M5, 7.7% after.

Measured 1.72x against a predicted 2.25x. The residual is where it should be for this
change: the downdate was one 564-square `rankUpdate` and is now two `rankUpdate`s on
339- and 315-sized blocks plus one 24x315 `gemm`, so the same flops run through
smaller kernels at lower efficiency, and the step is bandwidth-bound rather than
issue-bound to begin with. Trading 2.25x of arithmetic for 1.72x of time is a
reasonable exchange rate for a change that adds no data movement.

The forward-looking reading: `m³/6` and `m²·25` are now 8% of the update and the
solve is another 26%, both driven by the *measurement* count. A further milestone
aimed at this function would have to attack `m` — the 76 two-row blocks — not `N`.

Note also that `actual_update` is now *below* `track` on the stereo path (4.78 vs
9.84) — the first time in this work that the covariance update is not the largest
item. At the baseline it was 35.2 ms against 13.5. The remaining stereo frame is
24.4 ms: 9.8 track, 5.8 process_tracks (4.8 of it the update), ~1.8 propagation, and
~6.9 ms of PNG decode and the Python feed loop, which is a harness floor rather than
estimator cost. The mono frame is 11.5 ms, of which ~3.3 ms is that same floor.

## Accuracy

8-member ensembles from the frozen worktree `xivo-effm5` (`e2e6f0b`), 6 rooms each:

| | ATE | RPE_rot | RPE_tra | RPE_rot_i | RPE_tra_i |
| --- | --- | --- | --- | --- | --- |
| base_mono | 0.0796 ± 0.0063 | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| m4_mono | 0.0786 ± 0.0049 | 0.6205 | 0.0226 | 0.5126 | 0.0222 |
| **m5_mono** | **0.0797 ± 0.0063** | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| base_stereo | 0.0551 ± 0.0031 | 0.6208 | 0.0139 | 0.5128 | 0.0132 |
| m4_stereo | 0.0549 ± 0.0033 | 0.6208 | 0.0139 | 0.5128 | 0.0132 |
| **m5_stereo** | **0.0549 ± 0.0033** | 0.6208 | 0.0139 | 0.5128 | 0.0132 |

Stereo is unmoved to four decimals in all five metrics. Mono moves +0.0011 against
M4, which is a sixth of the 0.007 intrinsic sd of a 6-room mean — but the *direction*
is the interesting part, and it is not noise in the way that phrasing suggests.

### The divergence set shrank

| | identical | differing |
| --- | --- | --- |
| mono, M5 vs M4 | 45 / 48 | `m0,m1,m4 / room3` |
| mono, M5 vs baseline | **47 / 48** | `m7/room3` |
| stereo, M5 vs M4 | 47 / 48 | `m0/room3` |
| stereo, M5 vs baseline | 46 / 48 | `m0/room3`, `m6/room1` |

M4's mono ensemble differed from the baseline's on four runs (`m0,m1,m4,m7 / room3`,
inherited unchanged from M2). M5 differs from M4 on three of those four — and they
are exactly the three that go *back* to the baseline's trajectory. Per member:

| member | baseline | M4 | M5 |
| --- | --- | --- | --- |
| 1 | 0.0900 | 0.0821 | **0.0900** |
| 4 | 0.0772 | 0.0767 | **0.0772** |
| 7 | 0.0777 | 0.0786 | 0.0786 |

Members 0, 2, 3, 5, 6 are byte-identical across all three. So the M5 mono ensemble
*is* the baseline ensemble with one member changed, and the entire 0.0010 ATE
"improvement" that M2 introduced and M3/M4 carried was two members' worth of gate
flips on one sequence.

That is worth stating plainly because it cuts the other way too: had M5 been judged
on its mono ATE alone, it would read as a 0.0011 regression against M4. It is
neither. A trajectory cannot wander back to a bit-identical match with a run made
four commits earlier by accident — 21 000 poses at 6 decimals do not coincide — so
these divergences are single gate decisions flipping at a threshold, not error that
accumulates. Stereo shows the same thing in the other direction: it kept M4's one
divergence (`m6/room1`) and added one (`m0/room3`), so the set is a random walk with
no trend, exactly as it should be for reassociated arithmetic.

The five-run total across both settings after five milestones of rounding changes,
against 96 runs, is the number to carry into the report.

### Why this milestone can diverge at all

Unlike M4 (which touches no arithmetic and was byte-identical on all 96), M5
reassociates:

- the triangular solve becomes one `trsm` per run instead of one over 564 columns,
  so its internal panel blocking changes;
- `-WᵗW` becomes two `rankUpdate`s plus one off-diagonal `gemm` instead of one
  564-square `rankUpdate`;
- `err = Wᵗu` becomes two segmented `gemv`s.

What it does *not* change is the summation order inside `H P` — and that is why the
divergence rate is as low as it is. The dense-block path in `MeasurementTimesCov`
splits its summation index at the run boundary, which would reassociate, but the
census reports `oos:0` on every TUM-VI sequence in both settings: there are no dense
out-of-state blocks on this dataset, so that path never runs here. The unit test
`CompactingToTheOccupiedExtentChangesNothing` deliberately includes a 12-row dense
block so the case is covered even though the dataset does not reach it.
