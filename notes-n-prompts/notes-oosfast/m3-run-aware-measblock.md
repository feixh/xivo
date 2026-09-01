# M3 — a `MeasBlock` that knows more than "dense"

Config key: none of its own; the run set is only attached when
`oos_fast.enable` is on, so with the key off this is dead code.

## The defect

`Estimator::FilterUpdate` pushed every out-of-state row block as

```c++
H_.block(offset, 0, n, err_.size()) = f->Ho();
meas_blocks_.push_back({offset, n, -1, -1});
```

`gsind = -1` means `MeasBlock::sparse()` is false, which in `ekf_update.cpp` means
"treat all of this block's columns as possibly nonzero". So the OOS rows opted out of
the block-sparse machinery entirely.

The coordinator's diagnosis said this made them pay for the *maximum* state while the
in-state rows paid for the active one. That is nearly right, and the correction
matters for sizing the win: `MeasurementTimesCov` and `CovTimesMeasurementT` are
already **live-extent aware** through `StateRuns` / `OccupiedStateRuns` (M5 of the
earlier efficiency work), and the dense path loops `live.nruns` x `live.nruns`
rather than over `kFullSize`. So a dense OOS block was paying for `live`, not for
564. The census puts `live-dim` at 488.5 of 564. It is a `488 x 488` read of `P_` —
1.9 MB — per OOS block, not a `564 x 564` one.

Sized rather than assumed, from the instrumented `[census]` line on mono room1-6:

```
rows:180.725 (right:0 oos:11.5191)   live-dim:488.52/564   live-runs:2
group-slots:30.9525/45   occupied-dim:463.434/564
```

11.5 OOS rows per update out of 180.7. Each OOS block is `2n-3` rows for a track with
`n` in-state views; views/candidate is 3.22, so a block is ~3-9 rows and there are
~2-4 blocks per update, each doing `nruns^2 = 4` gemms with a summation index of 488
where ~30 would do.

## The fix

`MeasBlock` gained a fifth field:

```c++
const RunSet *runs{nullptr};
```

and `ekf_update.cpp` a helper that both dense paths now use:

```c++
inline void DenseSumRuns(const MeasBlock &b, const StateRuns &live, RunSet &out) {
  out.Clear();
  if (b.runs != nullptr) { /* copy b.runs */ return; }
  for (int i = 0; i < live.nruns; ++i) out.Add(live.runs[i].start, live.runs[i].len);
}
```

so the *summation* index of `H P` and of `M H^T` comes from the block when it knows
its own columns and from `live` when it does not. The output index is unchanged
(`live` on both). Null keeps the previous behaviour exactly, which is what a
loop-closure block — whose sparsity nobody has worked out — still gets.

This is the "multi-run `MeasBlock`" the coordinator asked about, and yes, it had to
be a run *set*: the groups an OOS track was observed from are scattered slots, not
one contiguous block. `kMaxMeasRuns = kMaxGroup + 2`.

What is skipped is identically zero — each run is its own `dst.noalias() += A * B`
and a skipped run has `A` zero — but the retained runs are cut on different
boundaries than `live`'s, so the gemm shapes differ and the result is **reassociated,
not bit-identical**. I wrote "bit-identical" into the header first and it was wrong;
the corrected comment and `EkfUpdate.ADenseBlockWithRunsGivesTheSameAnswer` (which
bounds the difference rather than requiring equality) are the honest version.

## Measured

Mean over room1-6 mono, one core:

| timer | base | cand |
| --- | --- | --- |
| `actual-update` | 2.492 ms/frame | 2.322 ms/frame |
| `update` | 2.528 | 2.359 |

**−0.170 ms/frame.** Part of that is the run-aware summation and part is dropping the
two `Ho()`/`ro()` by-value copies per feature in the stacked-`H` fill.

Smaller than M1 or M2, as it should be: 11.5 of 180.7 rows.

## Why this is where `actual-update` stops

After M3 `actual-update` is 2.32 ms/frame and is the largest single estimator cost.
Its flop budget at `m = 181` rows and `n = 488` live columns is

| step | MFLOP |
| --- | --- |
| `M = H P` (sparse rows) | 4.4 |
| `S = M H^T` | 0.8 |
| `chol(S)` | 2.0 |
| `W = L^-1 M` | 8.0 |
| `P -= W^T W` (lower triangle) | **21.6** |
| mirror | 0.1 (writes) |

~37 MFLOP in 2.32 ms is ~16 GFLOP/s of double-precision Eigen at these shapes,
which is a respectable fraction of one core. The rank-`m` downdate is 58% of it and
it is irreducible without changing one of `m` or `n`:

* `m` (181) cannot be reduced by QR/measurement compression — see
  what-didnt-work.md; `m < n`, so there is nothing to compress.
* `n` (488) is already the live extent. `occupied-dim` is 463.4, so a true active-set
  compaction that permuted `P_` into contiguous occupied slots would save
  `1 - (463.4/488.5)^2 = 10%` of the downdate, ~0.14 ms/frame, at the cost of
  permuting a 2.5 MB matrix. Not worth it. (This gap used to be much larger — the
  earlier M5 note records `occupied-dim: 295/564` — but the merged config's OOS pose
  window keeps 31 group slots alive instead of 8, so the bounding runs are now tight.)
* Reducing the number of in-state features or the pose window is a tuning change and
  is explicitly out of scope.
