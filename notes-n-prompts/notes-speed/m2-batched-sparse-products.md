# M2 -- batch the two block-sparse products in the EKF update

## How it was found

`harness/bench_update.cpp` reproduces the dense linear algebra of
`EkfUpdateDowndate` at the measured census (`rows` 283 stereo / 146 mono,
`live-dim` 331, `live-runs` 2, `kFullSize` 564) and times each piece:

```
rows=283          LLT 0.297   solve 0.769   rankUpdate+gemm 0.847   mirror 0.021   sum 1.93 ms
rows=146          LLT 0.063   solve 0.264   rankUpdate+gemm 0.530   mirror 0.020   sum 0.88 ms
```

but `actual-update` measured 4.04 ms (stereo) and 1.72 ms (mono). The missing
2.11 / 0.84 ms is `MeasurementTimesCov` + `CovTimesMeasurementT`, whose *flop*
count is trivial: 283 rows x 25 nonzero columns x 331 live columns is 2.3 MFLOP,
about 2% of the update.

They were slow because of call count, not arithmetic. Both looped

    for each row block b:  for each column run r of b:  for each live run cj:
        M.block(b.row, cj.start, b.rows, cj.len) += H.block(...) * P.block(...)

with `kJacRuns` = 6 (`Wsb Tsb`, `bg`, `Wbc Tbc`, `td`, the reference group, the
feature), `live.nruns` = 2, and -- crucially -- **one block per camera**:
`Estimator::ConstructJacobians` pushes a 2-row block for the left observation and
a second 2-row block for the right, so stereo has ~141 blocks for 283 rows. That
is 141 x 6 x 2 = 1692 gemm calls for `M = H P` and 141 x 6 = 846 for `S = M H^T`,
2538 per update, with M as small as 2 and K as small as 1. At the measured 2.11 ms
that is 0.83 us per call -- entirely Eigen's dispatch, packing and blocking
setup.

(The sampling profile had blamed 14.3% of stereo CPU on
`Eigen::internal::triangular_solve_vector`. The *magnitude* was right -- it
matched `actual-update` almost exactly -- but the symbol was a misattribution:
`addr2line` folds the whole gemm/trsm kernel family onto whichever outer template
symbol the address happens to land in, which is also why no `gebp` or
`rankUpdate` symbol appears anywhere in that profile. Believing the name would
have sent me to optimize the triangular solve, which is only 0.77 ms of 4.04.)

## The change

`src/ekf_update.cpp` only. Both products now walk maximal spans of consecutive
*sparse* blocks and batch three ways:

1. **The fixed shared runs are hoisted out of the block loop.**
   `kJacSharedRuns` ends with a sentinel that is replaced per feature by its
   reference group's run; everything before it -- `Wsb Tsb` (6), `bg` (3),
   `Wbc Tbc` (6), `td` (1) -- is *the same 16 columns for every visual
   measurement in the update*. So they are driven once over the whole span:
   4 gemms of (283 x 6) x (6 x 331) instead of 4 per block. That is 16 of the 25
   nonzero columns, i.e. most of the flops, moved onto a shape Eigen is good at.

2. **The group run is driven over merged spans.** Two adjacent blocks with the
   same `gsind` have identical group columns, and blocks cover the rows of `H` in
   order, so a run of them is a run of rows. This merges a feature's left and
   right blocks (always) and consecutive features that share a reference group
   (usually -- features enter the state together from the same group).

3. **Only the 3-column feature run is left per feature**, itself merged across
   the left/right pair.

Call count per stereo update falls from ~2538 to roughly 270.

## CORRECTION (written after M4, commit 801d45e)

**The section below is wrong and is kept only because the mistake is instructive.**
This change is a *reassociation*, not a bit-identical rewrite.

The mistake had two halves. The reasoning half: per-element run order and K are
indeed preserved, but that is not sufficient -- merging blocks changes **M**, and
Eigen's gemm is not shape-invariant in the last bit. A different M means a
different LHS packing and a different row-peeling path through `gebp_kernel`, so
the same K-ordered sum is rounded differently.

The evidence half is the worse of the two: **`md5sum` of
`dump/tumvi_<seq>_cam0` is not a bit-identity check.** That file is printed to
six decimals (`XIVO_DUMP_PRECISE=1` is what makes it exact), so a matching md5
means "agrees to 1e-6 m", which a rounding-level perturbation usually does. Four
matching md5s looked like proof and were not.

What actually settled it: the final `--jitter 6` ensemble has 72 runs, and 3 of
them (mono room2 member 5, mono room5 members 2 and 4) did *not* match M1's
trajectories. Bisecting by reverting one file at a time and rebuilding --
`scripts/pyxivo.py` to M3's, `src/tracker.cpp` to M2's, `src/ekf_update.cpp` to
M1's -- reproduced M1's md5 exactly on the last one. Then, to see it directly,
both forms were compiled into one binary and `M = H P` and `S = M H^T` compared
element by element on every update:

| room3 | differing elements of `M` | of `S` | max abs diff / max abs value |
|---|---|---|---|
| mono, update 1 | 1024 / 101520 | 15985 / 32400 | 2.1e-13 |
| mono, update 2 | 11840 / 101520 | 7036 / 32400 | 2.1e-16 |
| stereo, update 1 | 2063 / 203040 | 72494 / 129600 | 2.7e-11 |
| stereo, update 2 | 35642 / 203040 | 23490 / 129600 | 1.8e-16 |

So they differ on every update, at one ulp typically and never worse than 2e-13
of the matrix's own magnitude. The eye-catching *relative* differences (up to
3e-5) are all on elements where the sum cancels to ~1e-8; absolute agreement is
at the rounding level throughout. `unitTests_ekf_update`, which checks the fast
form against the dense `EkfUpdateJoseph` reference, passes -- this is rounding,
not an error.

The brief permits exactly this ("If a change cannot be bit-identical -- e.g.
reassociating a floating-point sum -- it is still allowed, but then you must prove
no accuracy regression with a full 6-member ensemble"), and the branch already
needed such a proof because of M1. See `summary.md` for the ensemble table:
every ATE mean moves less than one sd worse, every orientation and RPE mean
moves better.

Two lessons worth keeping:

1. Verify bit-identity on something that *is* the bits. `XIVO_DUMP_PRECISE=1`, or
   a checksum of `P_`, not a 6-decimal text dump.
2. "Same K, same order" is the right question to ask about Eigen and an
   incomplete answer. If you need bit-identity from a gemm, you have to keep the
   *shape*, not just the summation order.

## Why it is bit-identical, not just equal -- SUPERSEDED, SEE THE CORRECTION ABOVE

Two things had to hold and both do:

- **Per-element accumulation order is unchanged.** Every output element still
  receives its contributions in the order `Wsb Tsb`, `bg`, `Wbc Tbc`, `td`,
  group, feature, because phase A runs the fixed runs in `kJacSharedRuns` order
  before the group phase and the feature phase. `dst.setZero()` before the first
  contribution is kept rather than replaced by an assigning first term, so even
  the sign of a zero is preserved.
- **Per-gemm summation order over K is unchanged.** Merging blocks changes M (the
  number of rows) and never K: a merged group gemm still has K = 6, a merged
  feature gemm still has K = 3. Eigen's `gebp` accumulates a given output element
  over the packed k-panel in ascending k for any M and N, and both the old and
  the new shapes go through `GemmProduct` (all sizes are dynamic, so
  `CoeffBasedProduct` is never selected). Same kernel, same k order, same FMAs.

Merging blocks across the group is legitimate only because `H` is *exactly* zero
in the columns a block does not own -- guaranteed by `Feature::ComputeJacobian`
and pinned by `unitTests_jacobians_stereo`, which checks in both directions that
everything outside `kJacSharedRuns` is exactly zero and that every column the
finite-difference tests find live is inside one of them.

Verified empirically -- *and this is the step that was too weak, see the
CORRECTION above*: `md5sum dump/tumvi_<seq>_cam0` matches the previous commit
on room1 and room3, mono and stereo (all four). That file has six decimals, so
what this actually shows is agreement to 1e-6 m. `unitTests_ekf_update` (which
checks the fast form against the dense `EkfUpdateJoseph` reference),
`unitTests_Jacobians`, `unitTests_jacobians_stereo`, `unitTests_determinism` and
`unitTests_propagate_cov` all pass.

    mono   room1 b5185115fb76d44726cc6ee861ad6e73
    mono   room3 ee2210f7a5093e5baf852cbb22ab09ca
    stereo room1 6dc11ae2a241e8f5296690a708ba1dd0
    stereo room3 129c2d3f7fd637389847346455ea67c3

## Result

`actual-update`, room3:

| | M1 | M2 | |
|---|---|---|---|
| mono | 1.716 ms | 1.103 ms | -36% |
| stereo | 4.043 ms | 2.530 ms | -37% |

room1: mono 1.893 -> 1.189, stereo 4.504 -> 2.884.

Implied cost of the two products after batching: 2.53 - 1.93 = 0.59 ms stereo,
down from 2.11. What is left in `actual-update` is now the dense algebra
`bench_update.cpp` accounts for.

Paired wall clock, candidate on cpu 128 and `auto` @ 9e3ec06 on cpu 129 in the
same window, `--timing --no-score`, 2821 frames:

| | base | M1 | M2 | M2/base | FPS base -> M2 |
|---|---|---|---|---|---|
| mono room1 | 29.35 s | 23.75 s | 21.53 s | **1.363x** | 94.6 -> 131.0 |
| mono room3 | 28.58 s | 22.75 s | 21.08 s | **1.356x** | 98.9 -> 133.8 |
| stereo room1 | 66.93 s | 52.80 s | 48.24 s | **1.387x** | 42.2 -> 58.5 |
| stereo room3 | 63.09 s | 50.97 s | 46.67 s | **1.352x** | 44.7 -> 60.4 |

## What is left in the update, and what was rejected

Per-update budget after M2, stereo: LLT 0.30, triangular solve 0.77,
`rankUpdate` + off-diagonal gemm 0.85, the two products 0.59, mirror 0.02.

- **Transposed triangular solve.** `bench_update.cpp` measures the
  `L^T \ M^T` / `OnTheRight` form at 0.58 ms against 0.77. But it produces `W^T`
  in a layout where the downdate's `rankUpdate` would read strided rows of a
  column-major matrix, and it reassociates nothing but does change which panel
  Eigen packs -- so it is neither free nor obviously bit-identical. 0.19 ms of a
  16.5 ms frame; left alone.
- **Compacting `live` to the occupied set.** `live-dim` is 331 but
  `occupied-dim` is 285: 14% of the solve and the downdate multiplies vacant
  slots. Gathering `P` into a 285-square and scattering back costs 2 x 0.65 MB of
  traffic to save 14% of ~2 MB, which is a wash, and it would need the same
  treatment in `err` and the mirror.
- **Sequential (one-feature-at-a-time) updates.** Would make `S` 4x4 and remove
  the 283-square factorization entirely, but it is a different filter and
  changes accuracy. Out of scope by the brief.
