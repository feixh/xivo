# M3 -- bf16 kernels wired into the filter

What landed: `common/bf16_gemm.h` (the AVX512-BF16 GEMM behind an Eigen-facing
API), a 13-case unit test, and four call sites in the filter switched from Eigen
expressions to explicit kernel calls under a per-kernel precision knob.

Two questions were open at the end of M2, and both are now answered by
measurement:

1. Does the kernel reach microbenchmark speed once it is behind a real API and
   called at the filter's shapes? **Yes**, within 20%.
2. Can bf16 carry the covariance? **No -- not in this formulation.** The filter
   collapses. The reason is cancellation, and it is measured below rather than
   argued: the update changes the covariance by ~5e-3 relative, and a bf16
   product is wrong by ~3e-3 relative. The arithmetic error is the same size as
   the information.

That second answer is not a dead end. It says the low precision has to be
applied to the *correction*, not to the covariance -- which is what M4 does.

## The knob

```json
"kernel_precision": {"joseph": "f64", "innovation": "f64", "gating": "f64",
                     "batch_gating": false}
```

Defaults are exactly the shipped behaviour, so an unmodified config gets an
unmodified filter. Each name is a group of products, not a matrix:

| knob | products | reaches the state this frame? |
| --- | --- | --- |
| `innovation` | `H P`, `(H P) H^T` | **yes** -- via `S.ldlt()` -> `K` -> `err_` |
| `joseph` | `K H`, `(KH-I) P (KH-I)^T`, `K R K^T` | no -- covariance only |
| `gating` | `J_i P J_i^T` (MH), `J_r P J_r^T` (stereo), `H P H^T` (OOS) | no -- decides accept/reject |
| `batch_gating` | stacks the gating Jacobians into one product | no |

`f32` and `bf16` fall back to `f64` on a host without AVX512-BF16
(`bfgemm::effective`), so a config is portable.

## Kernel timings at the filter's real shapes

`bench/kernel_api.cpp`, against the shipped header, at the shapes the filter
actually reaches: n = 564 (= 24 + 6*45 + 3*90, the fixed capacity), m = 180, 90
features. Median of 5, pinned to an idle core. Full output in
`logs/m3_kernel_api.log`.

| shape | f64 ms | f32 ms | bf16 ms | f32 | bf16 | M0 microbench |
| --- | --- | --- | --- | --- | --- | --- |
| Joseph, `A P A^T` (2x n^3) | 13.33 | 7.23 | 4.11 | 1.84x | **3.24x** | 3.4x |
| Innovation, `H P` + `(H P) H^T` | 2.82 | 1.63 | 0.96 | 1.73x | **2.93x** | 2.7x |
| Gating, per-feature `Mul` | 9.88 | 10.52 | 10.49 | 0.94x | **0.94x** | -- |
| Gating, packed rhs | 9.93 | 10.50 | 2.28 | 0.95x | **4.36x** | 5.5x |
| Gating, batched | 2.36 | 1.66 | 1.17 | 1.43x | **2.03x** | -- |

Sanity check against the M0 profile: per-feature gating measures 9.88 ms here
against a profiled `MH-gating` of 9.0 ms, and Joseph + innovation measure 16.2 ms
against a profiled `actual-update` of 23.1 ms -- the missing 7 ms is `K H`,
`K R K^T`, the `ldlt` solve and assembling `H`, none of which the bench includes.
So the bench is measuring the right products at the right size.

Exit criterion for M3 (within ~20% of the microbenchmark) is met on all three
shapes. Two things in that table are worth more than the exit criterion:

**A naive per-feature bf16 sweep is no faster than fp64** (10.49 vs 9.88 ms).
Ninety 2xn Jacobians against the same P repacks P ninety times, and the packing
is O(n^2) while the useful work is O(2n^2) per feature. Precision alone, applied
without regard to arrangement, buys nothing. Hence two extra entry points in the
header: `PackRhs` packs P once and `MulRhs` reuses it (4.36x), and the batched
arrangement avoids the question entirely.

**Blocking is worth more than precision on the gating sweep.** Stacking the 180
Jacobian rows into a single product takes fp64 from 9.88 to 2.36 ms -- 4.2x,
with no change of precision and, as shown below, no change of result. bf16 on
top of that buys a further 2.03x. Those two wins are independent and are
reported separately throughout, because only one of them carries any numerical
risk.

## Accuracy arms, mono room1

`cfg/mkarms.sh` generates one config per arm from the shipped configs by
inserting a single `kernel_precision` line, so the arms differ in nothing else.
Baseline for comparison is the M0 fp64 run: 0.071380 / 0.078622 / 0.017578 /
0.527822.

| arm | joseph / innovation / gating / batch | ATE@0.001 | ATE@0.02 | RPE_tra | RPE_rot |
| --- | --- | --- | --- | --- | --- |
| `f64` | f64 f64 f64 false | 0.071380 | 0.078622 | 0.017578 | 0.527822 |
| `batch` | f64 f64 f64 **true** | 0.071380 | 0.078622 | 0.017578 | 0.527822 |
| `gate_bf16` | f64 f64 **bf16** true | 0.069831 | 0.084723 | 0.016564 | 0.527733 |
| `f32` | **f32 f32 f32** true | 0.064371 | 0.067842 | 0.016005 | 0.527984 |
| `jos_bf16` | **bf16** f64 bf16 true | *aborts mid-run* | -- | -- | -- |
| `inn_bf16` | f64 **bf16** bf16 true | 161973 | 238412 | 27437 | 12.8 |
| `bf16` | **bf16 bf16 bf16** true | 5137691 | 34944391 | 1819436 | 26.8 |

Single runs, so the differences among the first four are inside the +-0.007
single-run noise band and mean nothing on their own -- the ensembles in M5 are
what will decide. Two entries do mean something on their own:

**`f64` and `batch` are bit-identical to the baseline trajectory** (`cmp` on the
dumped `tumvi_room1_cam0`). That is a stronger statement than "the ATE matches":
the refactor -- explicit temporaries, `H P` computed once instead of twice, the
gating sweep restructured into one 180xn product -- changes not one bit of the
output. Eigen's k-panel blocking depends on k and the cache, not on the number of
rows, so the 180-row product accumulates each output element in exactly the order
the 90 separate 2-row products did. Every arm below is therefore a clean A/B
against fp64 with no refactor confound.

**Both n^3 groups are individually fatal.** `inn_bf16` (bf16 only in the gain
path) diverges by five orders of magnitude; `jos_bf16` (bf16 only in the
covariance rotation, nothing touching this frame's state) collapses even harder
-- it aborts. Before the abort the log is 7500 lines of
`SwitchGaugeXYFeatures: not enough instate features` and gauge features being
dropped, and the in-state-view histogram shows 113420 frames with zero in-state
features against 2530 for fp64. The filter is not "less accurate"; it never
holds a feature.

## Why: the update is a small difference of large numbers

Measured in the running filter (`XIVO_DIAG_UPDATE=1`, which prints every 40th
update; the diagnostic is read-only and env-guarded, so an ordinary run is
unaffected), mono room1, fp64:

```
DIAG    1 n=564 m=180 |P|=53.49  |dP|/|P|=2.2e-02  |joseph-short|/|dP|=5.7e-13
DIAG  241 n=564 m=178 |P|= 9.49  |dP|/|P|=6.5e-05  |joseph-short|/|dP|=5.0e-12
DIAG  681 n=564 m=196 |P|= 6.00  |dP|/|P|=1.9e-04  |joseph-short|/|dP|=1.8e-12
DIAG 1441 n=564 m=160 |P|= 4.44  |dP|/|P|=2.2e-01  |joseph-short|/|dP|=2.1e-15
```

`|dP|/|P|` -- how much one update moves the covariance -- ranges over
6e-5 .. 2e-1 and sits around 5e-3 in the steady state. Compare the bf16 error
column of the kernel table: 3.2e-3 on the Joseph product. **The rounding error
of the arithmetic is of the same order as, and often 10-50x larger than, the
change the update is trying to make.** An 8-bit significand cannot express a
0.05% difference between two O(1) matrices. fp32's 1.3e-7 is comfortably below
`|dP|/|P|` at every frame observed, which is why `f32` is stable.

This is not a property of the bf16 kernel, and no amount of tuning inside the
kernel fixes it. It is a property of writing `P <- A P A^T` at all: the output is
the input plus a small correction, so the precision has to match the input's
magnitude, not the correction's.

Two consequences also worth recording:

* `mindiag` of P is exactly 0 at every frame -- the gauge-fixed states carry a
  hard zero variance. Any "floor the diagonal at eps" guard would corrupt them,
  so that idea is dead.
* The asymmetry of `A P A^T` has to be repaired by the caller. A single
  `A A^T` through the bf16 kernel comes out *exactly* symmetric (Eigen's own
  does not: ~5e-17 at f64, ~5e-8 at f32), but a three-factor product computed as
  two products is asymmetric at every precision -- 1.9e-16 f64, 1.1e-7 f32,
  2.2e-3 bf16 -- because `C(i,j)` sums `(AP)_ik A_jk` while `C(j,i)` sums
  `(AP)_jk A_ik`. The result is assigned back into P and rotated again next
  frame, so it compounds. `Estimator::Symmetrize` runs whenever the effective
  precision is not f64; it is O(n^2) against an O(n^3) product.

## The way out, measured

The last column above is the useful one. `|joseph-short|/|dP|` compares the
Joseph form against the short form

```
P+ = P - K (H P),   symmetrized
```

which is algebraically identical when K is the optimal gain (`K S K^T =
K (H P)` follows from `K = P H^T S^-1`), and which XIVO's `ldlt` solve does
produce. Measured over the whole sequence the two forms agree to **1e-15 ..
1e-9 relative to the size of the update itself** -- i.e. to fp64 roundoff. The
Joseph form's extra robustness is buying nothing here that fp64 does not already
provide.

The short form drops `K H`, both `n^3` products and `K R K^T`, and adds one
`n x m x n`. At n = 564, m = 180 that is 1.33e8 multiply-adds against 5.49e8, a
**4.1x flop reduction** on the largest kernel of the frame -- larger than
anything precision can offer, and it composes with precision rather than
competing with it. And it changes the numerics
in exactly the direction this milestone showed was needed: the low-precision
product now computes `K (H P)`, a quantity whose own magnitude is `|dP|`, and
its error is 3e-3 of `|dP|` rather than 3e-3 of `|P|`. At the measured
`|dP|/|P| ~ 5e-3` that is a ~200x reduction in the error injected into the
covariance, for free.

M4 implements it behind `covariance_form: joseph | short`, keeps `joseph` as the
default until the ensembles say otherwise, and re-runs the precision grid on top
of it.

## Tests

`src/test/unittest_bf16_gemm.cpp`, 13 cases, run by `ctest -R BF16Gemm`; the
whole suite is 19/19. It links only gtest, so it builds and runs even when the
rest of the library does not. What each case is for:

| test | what would break without it |
| --- | --- |
| `F64IsEigenExactly` | the f64 path is a pass-through, not a reimplementation |
| `RealShapesAccuracy` | error bounds at the filter's actual n, m |
| `TransposeRoutesAreBitIdentical` | `at`/`bt` flags vs. materialised transposes |
| `GramProductIsExactlySymmetric` | the bf16 `A A^T` symmetry claim above |
| `ThreeFactorProductNeedsSymmetrizing` | records the asymmetry, and that `Symmetrize` fixes it |
| `RaggedShapes` | all of {1,2,3,5,6,7,31,32,33,37}^3 -- the MR=6/NR=32 tile padding |
| `AccumulateMatchesAssignThenAdd` | the `K R K^T` accumulate path |
| `BlockOperandsWithForeignStride` | callers pass blocks, so `Ref` must not copy |
| `PackedRhsMatchesMul` / `...TransposeVariants` | `PackRhs`/`MulRhs` == `Mul` |
| `Deterministic` | bit-repeatability, which every A/B in this project assumes |
| `PrecFromString` | a typo in a config must not silently select bf16 |
| `RoundToNearestEven` | ties-to-even, and unbiasedness over 1e5 samples |

Destinations are pre-filled with -7.0 before every call, so an untouched output
element cannot pass as a zero.
