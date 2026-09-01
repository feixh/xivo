# M6 — the motion model has nine dynamic rows and a block-diagonal noise

Commit: `M6: exploit the structure of the motion model in the propagation`
(`0f2ef77`).

M3 deferred the motion-to-structure correlation and left `propagation` at 0.17 ms
per call, which was the integrator's own arithmetic. This milestone is that
remainder. It is 0.17 -> **0.032 ms/call**, a 5.3x, against the plan's target of
"~0.03".

## What was there

Every Prince-Dormand stage — seven per step, ~3 steps per IMU sample, ~11
`Propagate` calls per image, so **231 stage evaluations per image** — ran this:

```c++
static MatX FK1, FK2, FK3, FK4, FK5, FK6, FK7;
static MatX PK1, PK2, PK3, PK4, PK5, PK6, PK7;
...
ComputeMotionJacobianAt(X0, gyro_accel);
FK2 = F_ + F_ * r_2_9 * (FK1) * dt;
P0  = P_.block<kMotionSize, kMotionSize>(0, 0) + r_2_9 * (PK1) * dt;
PK2 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();
```

with

```c++
Eigen::SparseMatrix<number_t> F_;   // 24x24
Eigen::SparseMatrix<number_t> G_;   // 24x12
```

and `ComputeMotionJacobianAt` rebuilding both from scratch each stage —
`F_.setZero()` followed by 39 `coeffRef` insertions, `G_.setZero()` followed by
18 — and the step closing with

```c++
F_.setIdentity();
F_ = F_ + FK * dt;        // a dense expression, assigned back into a SparseMatrix
AccumulateMotionStructureCorrelation();   // Fcross_ = F_ * Fcross_
```

That last pair is worth pausing on: `I + FK*dt` has no zeros left, so the
assignment materialized all 576 entries and `F_` finished each step as a *fully
dense* sparse matrix. The accumulation then ran a 24x24x24 product through the
sparse kernel at nnz = 576 — a dense product paying sparse indirection on every
term.

## The three structural facts

All three are properties of the model, not approximations, and each is pinned by
a test.

**1. `F` has nine nonzero rows.** `ComputeMotionJacobianAt` writes only Wsb, Tsb
and Vsb. Every other state in the filter — `bg`, `ba`, `Wbc`, `Tbc`, `Wsg`, `td`
— is a random walk, so its `d(xdot)/dx` is identically zero. The three that do
have dynamics are indices 0..8 of the `Index` enum, i.e. contiguous and leading,
so the nonzero part is `topRows<9>`:

```c++
constexpr int kMotionDynSize = Index::bg;
using MatMotionDyn = Eigen::Matrix<number_t, kMotionDynSize, kMotionSize>;
```

Two `static_assert`s keep it honest: that Wsb, Tsb and Vsb are all below
`kMotionDynSize`, and that none of the random-walk states is. Reordering the
enum breaks the build rather than the filter.

**2. `G Qimu G'` is four 3x3 blocks.** `Estimator` builds `Qimu_` as a diagonal
and then squares it, so it is block diagonal with each 3x3 block itself
diagonal. `G` is

```
dWsb/dng = -I,   dVsb/dna = -Rsb,   dbg/dnbg = I,   dba/dnba = I
```

and nothing else, so `G Qimu G'` has *exactly* zero cross terms and reduces to

```c++
[Wsb,Wsb] += Qg
[Vsb,Vsb] += Rsb Qa Rsb'
[bg ,bg ] += Qbg
[ba ,ba ] += Qba
```

18 nonzero entries out of 576. `AddMotionNoiseCov` writes those four blocks and
`G_` is deleted outright.

This is where the plan was wrong. It said `G Qimu G'` "does not depend on the
stage" and should be hoisted out of the seven-stage loop. It cannot be: `G`
carries `Rsb`, and each stage evaluates the Jacobian at a *different* composed
state, so `Rsb Qa Rsb'` changes every stage. Three of the four blocks are
stage-invariant and the fourth is not, which is why the change is a structural
rewrite rather than a hoist.

The four-block form silently drops precisely the terms a correlated `Qimu` would
contribute, so the premise is now checked where `Qimu_` is built:

```c++
// CHECK_EQ is a macro; the comma inside block<3,3>(3*a, 3*b) would be read as
// an argument separator, so the value is hoisted first.
const number_t cross = Qimu_.block<3, 3>(3 * a, 3 * b).cwiseAbs().maxCoeff();
CHECK_EQ(cross, 0) << ...;
```

**3. `P F' = (F P)'`.** `P` is symmetric, so the slope needs one product, not
two, and comes out exactly symmetric by construction:

```c++
A.noalias() = F * P;                        // 9x24
out.setZero();
out.topRows<kMotionDynSize>() = A;
out.leftCols<kMotionDynSize>() += A.transpose();
AddMotionNoiseCov(Rsb, Qimu, out);
```

The symmetry precondition is a `CHECK_LE` under `#ifndef NDEBUG` rather than a
comment, because the whole expression is wrong on a `P` that has drifted.

## What that buys, and what it does not

`ApplyMotionTransition` now writes nine rows instead of 24 (and mirrors the
matching nine columns), the stage matrices are fixed-size 9x24, and the
accumulator applies `(I + [Fdt;0]) Fcross` as `Fcross + [Fdt Fcross; 0]`.

The arithmetic, counted in multiply-adds per Prince-Dormand step:

| | M5 | M6 |
| --- | --- | --- |
| `FK_i` (7 stages) | 39·24 = 936 each | 9·9·24 = 1944 each |
| `PK_i` (7 stages) | 936 + 936 + 216 + 432 = 2520 each | 9·24·24 = 5184 each |
| `Fcross_` (once) | 576·24 = 13824 | 9·24·24 = 5184 |
| **per step** | **38.0 k** | **55.1 k** |

**M6 does 1.45x more arithmetic than M5 and runs 5.3x faster.** That is the whole
finding of this milestone, and it is the opposite of what the flop count predicts,
so it is worth being explicit about where the time was going instead:

* a sparse-times-dense product at nnz = 39 on a 24x24 operand is `nnz x cols`
  scalar FMAs with indirect addressing — unvectorizable, and cheaper in flops
  precisely because it cannot use the machine's width;
* `F_.setZero()` plus 39 `coeffRef` insertions per stage rebuilds an index
  structure — each insertion into compressed storage shifts the entries after it,
  so the cost is quadratic in the number already present — and there are 57 such
  branchy insertions across `F_` and `G_` per stage, none of them arithmetic;
* `F_ = F_ + FK*dt` is a dense-to-sparse conversion scanning all 576 entries once
  per step, and it leaves the accumulation product running at nnz = 576;
* the stage matrices were dynamic-size, so each product inside a larger expression
  materialized a heap-allocated `MatX` temporary — a handful per stage, hundreds
  per image, for matrices of 4.6 kB that a fixed-size type puts on the stack.
  (This one turns out to be the smallest of the four: the two scratch arms below
  bound it at 6% of the saving.)

Dividing measured time by the flop count: M5 ran the propagation at **~1.3
GFLOP/s** and M6 runs it at **~10.4 GFLOP/s**. The ceiling: this is an EPYC 9R14
(Zen 4) at 2.6 GHz, built with `-march=native`, so Eigen vectorizes with AVX-512 —
which Zen 4 issues as two 256-bit halves, giving the same 2 FMA x 4 doubles x 2
flops = 16 flops/cycle, ~42 GFLOP/s on one core. So the propagation went from **3%
of what this machine can do to 25%**, by taking sparse machinery off a problem
with 24 columns. (Both
figures are lower bounds on the covariance kernels themselves, since the
`propagation` timer also covers `ComposeMotion` and the Jacobian evaluation.)

### Which of the three changes paid

M6 bundles three things: sparse containers to dense, dynamic-size stage matrices
to fixed-size, and the structural narrowing. Two scratch worktrees isolate them,
both detached at M5 (`e2e6f0b`) and changed by nothing but the declarations. They
are measurement fixtures, not deliverables: uncommitted, on no branch, and marked
`ATTRIBUTION BUILD (not for delivery)` at the edit site.

`sweeps/m6_attrib2.log`, four arms interleaved, two repeats, mono room1, load
2-4:

| arm | `F_`, `G_` | `FK_i`, `PK_i`, `P0` | algebra | propagation (ms/call) | FPS | share of the saving |
| --- | --- | --- | --- | --- | --- | --- |
| M5 | `Eigen::SparseMatrix` | `MatX` | 24x24 | 0.170 | 85.3 | — |
| `xivo-effm6y` | dense fixed | `MatX` | 24x24 | 0.089 | 92.9 | **59%** |
| `xivo-effm6x` | dense fixed | fixed | 24x24 | 0.081 | 93.5 | 6% |
| M6 | — (structural) | fixed 9x24 | 9x24 | **0.032** | 98.3 | 36% |

The two repeats agree to the last digit reported (0.1702/0.1696, 0.0890/0.0886,
0.0808/0.0805, 0.0319/0.0317), so the split is not load noise.

Both scratch arms are *exactly* M5's arithmetic — dropping the sparse
representation only skips terms that are exactly zero, and it skips them in the
same order — and both were checked to produce a **bit-identical trajectory** to M5
on mono room1 before being timed. So the split is clean: the first two rows change
no numbers at all, and only the last row can move a trajectory.

**The sparse container was the largest single item — 59% of the saving, more than
the structural work it was hiding.** Replacing two `Eigen::SparseMatrix` members
with dense fixed-size ones, and nothing else, is a 1.9x on `propagation` and a
+8.8% on end-to-end FPS: a two-line diff. Making the seven stage matrices
fixed-size on top of that is worth 6%, and exploiting all three structural facts
— the part that took the design work and the tests — is 36%.

That ordering is the useful thing to carry forward. `Eigen::SparseMatrix` is the
wrong container for a 24x24 operand at any density: 39 nonzeros out of 576 is 7%
occupancy, which is exactly the regime where sparse storage looks obviously right
and is nevertheless slower than dense, because at 24 columns the dense product is
a handful of vectorized FMAs and the sparse one is a rebuild of an index
structure.

## Tests

`unitTests_propagate_cov` grows from 6 cases to 13.

| test | what it pins |
| --- | --- |
| `TheAccumulatedTransitionIsTheIdentityBelowTheDynamicRows` | the accumulated product, not one factor: `(I+A)(I+B) = I+A+B+AB` keeps the property only if every factor has it |
| `AccumulatingByTheDynamicRowsMatchesTheFullProduct` | `Fcross + [Fdt Fcross; 0]` equals `(I + [Fdt;0]) Fcross` over 30 steps |
| `MotionCovSlope.MatchesTheUnstructuredForm` | 3 seeds, 1e-13, against `F P + P F' + G Qimu G'` with `F` padded to 24x24 and `G` from `MotionNoiseJacobian` |
| `MotionCovSlope.TheSlopeIsExactlySymmetric` | exact equality element by element, which `P F' = (F P)'` makes structural |
| `MotionCovSlope.TheStructurallyZeroPartIsExactlyZero` | seeded with `Constant(1234.5)`, so "zero" means *cleared* — the caller reuses the matrix across steps |
| `MotionNoise.TheFourBlockFormMatchesGQGt` | `AddMotionNoiseCov` alone, 1e-14, plus that `G Qimu G'` really has 18 nonzeros |
| `MotionNoise.TheBlockDiagonalPremiseIsLoadBearing` | correlate the gyro and accel noise and the four-block form must now be *wrong* — otherwise the `Qimu_` check guards nothing |

The last one is the counterpart of M3's `TheAccumulationOrderMatters`: an
equality test whose reference shares the assumption under test proves nothing, so
each structural claim gets a companion that breaks it.

One existing test had to change its claim.
`OneStepIsBitIdenticalToTheOldUpperBlock` asserted that a single step's
correlation update matches the old code element for element; it now fails at
element (8,0) by one bit, because `ApplyMotionTransition` narrowed from a
24x24x540 product to 9x24x540 and Eigen picks its panel blocking from the shape.
Renamed to `OneStepMatchesTheOldUpperBlock` and split into the three claims that
are actually true:

* rows 9..23 are still bit-identical, and for a stronger reason than before — the
  old code multiplied them by rows of the identity, the new code does not touch
  them;
* rows 0..8 are bit-identical to the *narrowed* product, so the implementation is
  that product and nothing else;
* rows 0..8 agree with the old 24-row product to <1e-15 relative, not exactly.

21/21 targets pass under `ctest`.

## Speed

`sweeps/m6.log` — four arms interleaved inside each (sequence, repeat), two
repeats, one thread each, `setarch -R`, from the frozen worktrees `xivo-effm5`
(`e2e6f0b`) and `xivo-effm6` (`0f2ef77`), verified to carry identical compiler
flags and defines. Load average was 2-4 throughout, so unlike some earlier
batches this one is not competing with a neighbour.

| arm | seq | wall (s) | **FPS** | propagation (ms/call) | actual_update (ms) | track (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| m5_mono | room1 | 32.9 | 85.65 | 0.170 | 1.82 | 3.74 |
| **m6_mono** | room1 | 28.6 | **98.67** | **0.032** | 1.83 | 3.72 |
| m5_stereo | room1 | 68.9 | 40.95 | 0.170 | 4.29 | 10.43 |
| **m6_stereo** | room1 | 64.5 | **43.75** | **0.032** | 4.29 | 10.39 |
| m5_mono | room6 | 30.4 | 86.65 | 0.170 | 2.03 | 3.43 |
| **m6_mono** | room6 | 26.3 | **100.34** | **0.032** | 2.01 | 3.42 |
| m5_stereo | room6 | 64.3 | 41.01 | 0.171 | 4.79 | 9.85 |
| **m6_stereo** | room6 | 60.1 | **43.87** | **0.032** | 4.79 | 9.83 |

| | vs. M5 | vs. baseline (chained) |
| --- | --- | --- |
| mono | **1.155x** | **4.72x** |
| stereo | **1.069x** | **3.54x** |

Nothing outside the propagation moved: `actual_update`, `track` and
`process_tracks` all agree with M5 to the reported precision, which is the
signature a change confined to `Propagate` should have.

### The saving re-derives the call count

The absolute saving is the same in every column, as it must be — the propagation
does not know how many cameras there are:

| | wall M5 (s) | wall M6 (s) | frames | saved (ms/frame) | / 0.138 ms per call = calls/image |
| --- | --- | --- | --- | --- | --- |
| mono room1 | 32.94 | 28.59 | 2821 | 1.542 | 11.2 |
| mono room6 | 30.42 | 26.27 | 2636 | 1.574 | 11.4 |
| stereo room1 | 68.90 | 64.48 | 2821 | 1.567 | 11.4 |
| stereo room6 | 64.28 | 60.08 | 2636 | 1.593 | 11.5 |

Four independent measurements land on ~11.4 `Propagate` calls per image, against
the 10 IMU samples per image that 200 Hz / 20 Hz predicts plus one
`Propagate(true)` to reach the image timestamp. M3 derived 10.4 the same way from
a smaller saving; the two bracket 11 from either side, and M8 will settle it with
a directly instrumented count.

Mono is now **10.1 ms/frame** and stereo **22.9 ms/frame**, of which the whole EKF
prediction step is 0.36 ms in both. Propagation is finished as an optimization
target: even reducing it to zero would be another 3.5% mono and 1.6% stereo.

## Accuracy

8-member ensembles (`run_ensemble_bugfix.sh`, `X.Vsb` perturbed by k·1e-6 m/s),
6 rooms each, from `xivo-effm6` (`0f2ef77`):

| | ATE | RPE_rot | RPE_tra | RPE_rot_i | RPE_tra_i |
| --- | --- | --- | --- | --- | --- |
| base_mono | 0.0796 ± 0.0063 | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| m5_mono | 0.0797 ± 0.0063 | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| **m6_mono** | **0.0797 ± 0.0063** | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| base_stereo | 0.0551 ± 0.0031 | 0.6208 | 0.0139 | 0.5128 | 0.0132 |
| m5_stereo | 0.0549 ± 0.0033 | 0.6208 | 0.0139 | 0.5128 | 0.0132 |
| **m6_stereo** | **0.0549 ± 0.0033** | 0.6208 | 0.0139 | 0.5128 | 0.0132 |

Every figure in both settings agrees with M5's in every reported digit.

### The census, and one run that walked back to the baseline

| | identical | differing |
| --- | --- | --- |
| mono, M6 vs M5 | **48 / 48** | — |
| mono, M6 vs baseline | 47 / 48 | `m7/room3` |
| stereo, M6 vs M5 | 47 / 48 | `m0/room3` |
| stereo, M6 vs baseline | 47 / 48 | `m6/room1` |

Mono is the strongest result any milestone in this series has produced: **not one
of the 48 runs changed a single recorded digit**, and the mono ensemble is
therefore still the baseline ensemble with exactly one member (7) altered, as it
has been since M2.

Stereo moved one run, and it moved it *back*. The stereo census against the
baseline reads 47/48 at M4 (`m6/room1`), 46/48 at M5 (`m6/room1` plus a new
`m0/room3`), and 47/48 again at M6 — `m0/room3` is bit-identical to the baseline
once more, having been the one run M5 flipped:

| stereo member 0 | ATE | RPE_rot | RPE_rot_i |
| --- | --- | --- | --- |
| baseline | 0.0540 | 0.6214 | 0.5134 |
| M5 | 0.0541 | 0.6213 | 0.5132 |
| **M6** | **0.0540** | **0.6214** | **0.5134** |

This is the same phenomenon M5 showed on three mono runs, and it is the reason to
keep taking these censuses rather than only the ensemble means. A trajectory
cannot return to a *bit-identical* match with the baseline by accident over 2593
frames. So `m0/room3`'s divergence at M5 was one gating decision landing on the
other side of a threshold, and M6's last-bit perturbation put it back — not error
accumulating and partially cancelling.

### Why this milestone can diverge at all, given a 48/48 mono census

The interesting thing about mono 48/48 is that M6 is *not* bit-exact in principle.
Splitting the structural factor of the attribution table into its two halves gives
four edits, and three of them are exact:

* sparse to dense containers: exact. Skipping structural zeros skips them in the
  same order, which is why both scratch arms above reproduce M5 bit for bit.
* dynamic to fixed size: exact. Same kernels, same shapes.
* four-block `G Qimu G'`: exact. `(-1)·q·(-1) = q` and `(-Rsb)·Qa·(-Rsb)' =
  Rsb·Qa·Rsb'` hold bit for bit in IEEE arithmetic, term by term and in the same
  order.

but the fourth is not:

* the narrowed products. `F P` over nine rows and over 24 rows sum the same 24
  terms, but Eigen chooses its panel blocking from the operand shape. At 24
  columns that appears to come out the same; at 540 it does not, and the unit test
  measures it directly in the correlation product: one bit at element (8,0).

So the state does differ in its last bits from the first image onward, in both
settings. What the census says is that in 95 of 96 runs no accept/reject decision
came close enough to a threshold for a 1e-16 perturbation to flip it. That is the
same behaviour M3 and M5 showed, and it is what a filter that is locally
contracting between discrete decisions should do: the perturbation stays at the
level it was introduced at until a gate flips, and then the trajectory jumps.
A bit-identical *output file* is evidence that no gate flipped, not that the
internal state was identical.
