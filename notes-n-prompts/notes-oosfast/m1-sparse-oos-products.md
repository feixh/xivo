# M1 — an out-of-state measurement's column runs, and the products that use them

Config key: `oos_fast.enable` (`OOSOptions::fast_sparse`), default **false**.

## The shape nobody had written down

The in-state visual measurement already had its sparsity in the code:
`kJacSharedRuns` in `src/core.h` says a feature's two rows are nonzero in the
motion block, the extrinsics, its reference group's six columns and its own three.
25 columns of 564. `MeasBlock` carries that, and `MeasurementTimesCov` /
`CovTimesMeasurementT` exploit it.

The **out-of-state** (MSCKF) measurement had nothing. Its stacked Jacobian is
`2n-3 x kFullSize` after the nullspace projection, and every consumer treated all
564 columns as possibly nonzero. They are not. The stack is built one observation
at a time by `ComputeOOSJacobianInternal`, and observation `i` writes exactly

* `Index::Wbc .. Tbc+3` — six columns of body-to-camera extrinsics, shared;
* `kGroupBegin + kGroupSize * obs.g->sind()` — six columns, the group it was seen
  from;

and nothing else. The point columns live in a separate `Hf` and are annihilated.
The left-nullspace projection `A^T H` is a row operation, so it cannot create a
nonzero column. Therefore

> the marginalized OOS Jacobian is nonzero only in `{Wbc,Tbc}` plus the pose block
> of each in-state group the track was observed from.

The census says a track that reaches the OOS path has 4.72 observations, of which
3.22 are in the state, so that is `6 + 6*3.22 ≈ 25` columns — the same order as an
in-state measurement, not 564.

## The vocabulary

`src/core.h` gained a run-list type next to the existing `ColRun`:

```c++
struct RunSet {                     // ascending, disjoint, maximal
  ColRun runs[kMaxMeasRuns];        // kMaxMeasRuns = kMaxGroup + 2
  int nruns, dim;
  void Clear();
  void Add(int start, int len);     // inserts and coalesces
  int Compact(int col) const;       // global column -> index in the gathered form, or -1
};
```

plus `GatherRunCols`, `GatherRunCov`, `ScatterRunCols` and a debug predicate
`ColsWithinRuns`. `Add` is what makes the rest safe: it keeps the invariant, so two
adjacent group slots become one run of 12 and a repeated slot costs nothing.
`RunSetTest.AddKeepsRunsAscendingDisjointAndMaximal` pins that.

The two gather helpers take `Dst &&dst`, not `Dst dst`. That is not a style choice:
with `Dst dst` and an lvalue `MatX` argument the template deduces `Dst = MatX`, the
gather fills a *copy*, and the caller silently sees an untouched matrix. It cost me
an afternoon.

`Feature::OOSColumnRuns(views, extra_gsind)` (in `src/oos.cpp`) builds the set from
the observation list; `extra_gsind` is for `ComputeInitJacobian`, which also writes
the anchor group's block. The set is a **superset** of the true support, which is
all any of the users need.

## Where it is used, and what each was costing

### 1. `Feature::ComputeOOSJacobian` — the scratch clear

`ComputeOOSJacobianInternal` used to zero each row of the shared scratch `Hx` across
all 564 columns before writing its ~12 nonzero ones. With `fast_sparse` on it zeros
only the columns in the run set.

This opens a trap I nearly walked into. `MarginalizeOOSPoint` then did

```c++
oos_.Hx = A.transpose() * s.Hx.topRows(rows);
```

over the full width — which now reads the **previous feature's** values outside the
current run set. The projection had to be restricted to the runs too, with the rest
of `oos_.Hx` explicitly zeroed:

```c++
oos_.Hx.setZero(out_rows, kFullSize);
for (int i = 0; i < oos_.runs.nruns; ++i) {
  const ColRun &r = oos_.runs.runs[i];
  oos_.Hx.middleCols(r.start, r.len).noalias() =
      A.transpose() * s.Hx.block(0, r.start, rows, r.len);
}
```

That is a correctness requirement, not an optimization.
`OOSUpdateTest.FastSparseLeavesNoStaleScratchBehind` is the regression test: it runs
two features whose group slots are disjoint back to back and asserts the second one's
Jacobian has no columns outside its own runs.

### 2. `Estimator::OOSGating` — the actual win here

The Mahalanobis gate formed

```c++
MatX S = H * P_ * H.transpose();   // H is (2n-3) x 564
```

i.e. it read all 2.54 MB of `P_` for a 9-row measurement, once per candidate. With
the runs it gathers a `dim x dim` sub-covariance (`dim ≈ 25-36`, so ~5-10 kB) and
multiplies that. `H` and `r` are also taken as references now (`oos_Hx()`,
`oos_inn()`) rather than through `Ho()`/`ro()`, which returned by value and so
copied ~32 kB per call, twice per feature.

Measured, mean over room1-6 mono, one core:

| timer | base | cand |
| --- | --- | --- |
| `oos-jacobian` (includes the gate) | 0.332 ms/frame | 0.039 ms/frame |

**8.5x, −0.293 ms/frame.**

### 3. `Feature::ComputeInitJacobian` — see m2

### 4. The stacked update — see m3

## Numerical equivalence

The two forms compute the same matrices up to Eigen's gemm reassociation (a
`rows x 12 x 12` product instead of a `rows x 564 x 564` one packs differently), so
this is *not* bit-identical and I do not claim it is. What it is:

* `unitTests_OOSUpdate` `FastSparseOOSJacobianMatchesTheDenseForm` and
  `FastSparseInitJacobianMatchesTheDenseForm` compute both forms on the same input
  and require a relative difference below `1e-14`, plus `ColsWithinRuns` on both;
* end to end, with `XIVO_DUMP_PRECISE=1` (17 significant figures — see
  config-delta.md, this branch had to *port* that env var), the whole trajectory
  agrees to

  | | poses | max position difference |
  | --- | --- | --- |
  | mono room1 | 2818 | 4.4e-14 m |
  | stereo room3 | 2582 | 3.4e-14 m |

  which is far inside the tolerance of every gate and threshold in the filter, so
  no gating decision flipped anywhere in either run.

The first attempt at this check produced *identical* md5s and I nearly published it.
It was worthless: `scripts/savers.py` on this branch printed `%f`, six decimals, so
the md5 only proved agreement to 1e-6 m. The `XIVO_DUMP_PRECISE` port is what turned
it into evidence.

Later, in M5, I found that the *key-off* path is not bit-identical to HEAD either, and
that this is not fixable: HEAD's own trajectory moves by 4.3e-12 m when nothing changes
but a `MALLOC_MMAP_THRESHOLD_` setting. See the last section of
m5-shared-oos-buffer.md — the 4e-14 numbers above sit two orders of magnitude below the
tree's own allocator-layout noise, which is the right way to read them.
