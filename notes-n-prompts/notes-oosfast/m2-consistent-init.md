# M2 — `consistent_init` at 1.84 ms/frame was two full reads of `P_` per feature

Config key: none. Behaviour is selected by the same `oos_fast.enable` as M1.

## What the knob sweep said

The coordinator's knob-attribution sweep (one-core mono, room1-6, 6 runs per arm,
raw dirs `experiments/results/knobfps_*_mono/`) put `consistent_init.enable=false`
at 98.1 FPS against a merged baseline of 83.1, i.e. **−1.84 ms/frame**, for
+0.0042 m of ATE. Turning it off is not acceptable — it is the
triangulation-consistent covariance that keeps the filter honest at promotion — so
the question was where 1.84 ms goes for something that runs 6.3 times per frame.

## Where it went

`Estimator::InitializeFeatureCovariance` (`src/estimator.cpp`) is XIVO's port of
OpenVINS `StateHelper::initialize_invertible`. Its core is

```c++
Mat3 M   = Hx * P_ * Hx.transpose();          // Hx is 3 x 564
M.diagonal().array() += consistent_init_R_;
MatX Pxf = -P_ * (Hx.transpose() * Hl_inv.transpose());
```

Two lines, three flops per element, and **two complete traversals of a 564x564
covariance**. `P_` is 2.54 MB. At 6.3 promotions per frame that is ~32 MB of memory
traffic per frame, for a 3-row measurement whose Jacobian is nonzero in about 30
columns.

The flop count is not the problem — `3 x 564 x 564` is 1 MFLOP — the traffic is. It
is the same defect as the OOS gate in M1, one level up.

## The fix

`Hx` here has exactly the M1 support plus the anchor group, which is why
`OOSColumnRuns` takes an `extra_gsind`:

```c++
RunSet rs; const RunSet *runs = nullptr;
if (oos_options_.fast_sparse) { rs = Feature::OOSColumnRuns(views, ref->sind()); runs = &rs; }
```

`ComputeInitJacobian` takes the set and writes its result through
`ScatterRunCols(Q1.transpose() * Hc, *runs, *Hx_out)` — i.e. the QR side of it also
works in compacted coordinates, using `RunSet::Compact` to translate `Index::Wbc`,
`Index::Tbc` and each group offset into the gathered frame. Then

```c++
Eigen::Matrix<number_t, 3, -1> Hc(3, rs.dim);   GatherRunCols(Hx, rs, Hc);
auto Pcols = init_cov_Pcols_.leftCols(rs.dim);  GatherRunCols(P_.topRows(size), rs, Pcols);
MatX Pc(rs.dim, rs.dim);                        GatherRunCov(P_, rs, Pc);
M   = Hc * Pc * Hc.transpose();
Pxf = -Pcols * (Hc.transpose() * Hl_inv.transpose());
```

`Pc` is `dim x dim` (~30 x 30, 7 kB) and `Pcols` is `size x dim` (~564 x 30,
136 kB, and reused across calls through the `init_cov_Pcols_` member rather than
reallocated). Traffic per call goes from 5.1 MB to ~150 kB, a **34x** reduction.

Note that `Pxf` — the cross-covariance between the new feature and the *rest of the
state* — genuinely needs all `size` rows of `P_`. What it does not need is all
`size` columns, and columns are where the sparsity is. Gathering the columns first
turns one `564 x 564` gemm into a `564 x 30` one.

## Measured

`process-tracks` minus its sub-timers (`update`, `jacobian`, `MH-gating`,
`oos-jacobian`) is the part of the visual measurement that promotion lives in.
Mean over room1-6 mono, one core, `-mode runOnly`:

| | base | cand |
| --- | --- | --- |
| `process-tracks` total | 5.499 ms | 3.331 ms |
| of which `update` | 2.528 | 2.359 |
| of which `oos-jacobian` | 0.332 | 0.039 |
| of which `jacobian` + `MH-gating` | 0.185 | 0.193 |
| **remainder (promotion, triangulation, bookkeeping)** | **2.454** | **0.740** |

**−1.714 ms/frame**, i.e. ~93% of the sweep's 1.84 ms `consistent_init` bill, with
the feature still on and the accuracy it buys still paid for.

`consistent-init:17511/17743` in the `[census]` line is unchanged between the two
runs: the same 17511 promotions took the consistent path, so nothing was skipped to
get this.
