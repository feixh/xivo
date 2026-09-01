# M2 — the covariance update: Joseph form → block-sparse symmetric downdate

Commit: `M2: replace the Joseph-form update with a block-sparse symmetric downdate`
(`7d219bc`).

## What was there

`Estimator::UpdateJosephForm`, unchanged since the original release:

```c++
S_ = H_ * P_ * H_.transpose();                 // (1) H P, then · Hᵗ
for (i) S_(i,i) += diagR_(i);
K_.transpose() = S_.ldlt().solve(H_ * P_);     // (2) H P *again*
err_ = K_ * inn_;
I_KH_ = K_ * H_;                               // (3) N x m x N
for (i) I_KH_(i,i) -= 1;
P_ = I_KH_ * P_ * I_KH_.transpose();           // (4) two N x N x N
for (i) K_.block(0,i,kr,1) *= sqrt(diagR_(i));
P_.noalias() += K_ * K_.transpose();           // (5)
```

with `N = kFullSize = 564` and `m` = 153 rows mono / 303 stereo (census, M0).
Counting multiply-adds at 76 in-state features:

| step | product | MFLOP (mono) |
| --- | --- | --- |
| (1) | `H P` | 49 |
| (1) | `· Hᵗ` | 13 |
| (2) | `H P` again | 49 |
| (2) | `ldlt` factor + solve for `N` columns | 14 |
| (3) | `K H` | 49 |
| (4) | `I_KH · P`, then `· I_KHᵗ` | 359 |
| (5) | `K Kᵗ` (all `N²` entries, not a triangle) | 49 |
| | **total** | **~580** |

Measured: **22.8 ms/frame mono, 34.2 ms/frame stereo** (`actual-update`), the
single largest cost in the system once M1 removed the gates — 22.8 of a 47.5 ms
mono frame.

Two things are wrong with it beyond the repeated `H P`. Steps (3)–(5) build and
multiply three full 564x564 matrices to apply a rank-153 correction. And they do
it *densely*: `H` has 25 nonzero columns per row block, so `K H` and `I_KH` are
overwhelmingly determined by structure that the gemm does not know about.

## The algebra

The Joseph form exists to keep `P` symmetric and positive semidefinite for an
*arbitrary* gain. At the *optimal* gain its extra terms are redundant. Write
`M = H P` and `S = M Hᵗ + R`. Because `P` is symmetric, `P Hᵗ = Mᵗ`, so
`K = P Hᵗ S⁻¹ = Mᵗ S⁻¹` and therefore `K S Kᵗ = Mᵗ S⁻¹ M = K M`:

```
(I-KH) P (I-KH)ᵗ + K R Kᵗ
  = P - K H P - P Hᵗ Kᵗ + K (H P Hᵗ + R) Kᵗ
  = P - K M - Mᵗ Kᵗ + K S Kᵗ
  = P - Mᵗ S⁻¹ M
  = P - Wᵗ W,        W = L⁻¹ M,   S = L Lᵗ
```

so the whole update is one rank-`m` symmetric downdate of `P`, and the error state
is `err = Wᵗ u` with `u = L⁻¹ inn`. This is not an approximation and not a
different filter: it is the same expression, rearranged. It is what OpenVINS
does (`StateHelper::EKFUpdate`).

Going through the Cholesky factor matters. Forming `P - K M` directly leaves a
difference of two independently-rounded products, whose asymmetry has no bound;
`P - WᵗW` subtracts something that is symmetric positive semidefinite *by
construction*, which is the property the Joseph form was there to provide.
`rankUpdate` on `selfadjointView<Lower>` touches only the lower triangle, which
is then mirrored, so `P_` comes out **exactly** symmetric — a stronger guarantee
than the old code gave (it produced `A P Aᵗ + K Kᵗ`, symmetric only up to
rounding).

New cost:

| step | MFLOP (mono) |
| --- | --- |
| `M = H P`, block-sparse | 2 |
| `S = M Hᵗ`, block-sparse | 0.6 |
| `LLT` of `S`, and one `L⁻¹M` (not two solves) | 8 |
| `P -= WᵗW` (triangle only) | 24 |
| **total** | **~35** |

A 17x reduction in arithmetic; the measured speedup is smaller because what is
left — the 564x564 downdate — is bandwidth-bound, not flop-bound.

## The sparsity, and where it comes from

M1 established that a visual measurement's rows are nonzero in only 25 of 564
columns. `H P` therefore does not need the other 539 rows of `P`:

```c++
for (block b) {
  if (dense) dst = H.middleRows(...) * P;
  else for (run r : MeasurementRuns(b.gsind, b.fsind))
    dst.noalias() += H.block(b.row, r.start, b.rows, r.len) * P.middleRows(r.start, r.len);
}
```

and `S = M Hᵗ` gets the same treatment on the columns of the result. Which rows
are which is recorded as `H_` is filled, in `meas_blocks_` — three sites
(`FilterUpdate`'s in-state rows and its out-of-state rows, `CloseLoopInternal`,
`OnePointRANSAC`). Out-of-state rows span every group their track was observed
from and the left-nullspace projection mixes them, so they are marked dense; the
loop-closure rows touch two groups per block, likewise dense. Marking a block
dense is always *correct* and only costs speed, which is the right way for this
bookkeeping to fail.

The remaining 24 MFLOP is the downdate, which is irreducible at this state
dimension — it is the one step whose cost is set by `N²m`, not by the sparsity.
M5 (active-set compaction) is what attacks that, since only 296 of the 564
dimensions are occupied.

## Structure

The update moved out of `Estimator` into `src/ekf_update.{h,cpp}` as free
functions over `(P, H, inn, diagR, blocks, err)`. The reason is testability: as a
member function reading `P_`, `H_`, `S_`, `K_`, `I_KH_`, it could only be
exercised by running the whole filter, which is exactly the situation that let it
sit unexamined. As a free function the fast form can be checked against the slow
one on inputs chosen to stress it.

`EkfUpdateJoseph` is kept, for two jobs: it is the reference the new code is
tested against, and it is the fallback if `S` has no Cholesky factor. That
happens only if `P` has already gone indefinite (`R > 0` guarantees `S ≻ 0` for
any `P ⪰ 0`), in which case the more defensive form is the right one to use. It
is logged at WARNING rather than taken silently — if it ever fires on real data
that is a bug report, not a recovery. It has not fired on TUM-VI.

`S_`, `K_` and `I_KH_` are gone from `Estimator` (three 564-square scratch
matrices, ~2.5 MB each, allocated once and touched every frame).

## Tests

`unitTests_ekf_update`, built at the *real* `kFullSize` because the sparsity
pattern is defined in terms of `kGroupBegin`/`kFeatureBegin` and does not exist at
a toy size. `P` is generated SPD on the occupied slots and exactly zero on the
vacant ones, which is the shape `P_` actually has.

| test | what it pins |
| --- | --- |
| `MatchesJosephAtTheRealSize` | 76 features / 7 groups (the census shape): `P` and `err` agree with the Joseph form to 1e-9 relative |
| `MatchesJosephWithStereoRows` | same at the stereo row count |
| `MatchesJosephWithADenseOutOfStateBlock` | 40 features + a 12-row dense block, i.e. both paths in one `H` |
| `MatchesJosephOnASmallUpdate` | 4 features / 2 groups — barely above `min_required_inliers_` |
| `BlockSparseHPEqualsTheDenseProduct` | `MeasurementTimesCov` vs. dense `H*P`, 1e-13 — localizes a failure above |
| `ADenseBlockDescriptionGivesTheSameAnswer` | declaring everything dense reproduces the sparse answer, so no block silently drops columns |
| `RefusesAnIndefiniteInnovationCovariance` | the fallback trigger fires, and `P`/`err` are left untouched for the caller |

Each `MatchesJoseph*` case also asserts exact symmetry of `P` (every element, not
a norm), a positive variance on every live slot, and that vacant slots stay
exactly zero.

19/19 targets pass under `ctest`.

## Speed

`sweeps/m2_update.log` — three arms interleaved inside each (sequence, repeat),
two repeats, one thread each, `setarch -R`, all from frozen worktrees:
`xivo-effbase` (`d13ec97`), `xivo-effm1` (`8ac1ad6`), `xivo-effm2` (`7d219bc`).
Wall clock and FPS are the whole `-mode runOnly` process; the per-frame columns
are the estimator's own `print_timing` averages.

| arm | seq | wall (s) | **FPS** | actual_update (ms) | update (ms) | visual_meas (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| base_mono | room1 | 137.2 | 20.56 | 23.39 | 23.44 | 38.23 |
| m1_mono | room1 | 108.4 | 26.03 | 22.69 | 22.73 | 28.08 |
| **m2_mono** | room1 | 52.4 | **53.80** | **3.19** | 3.22 | 8.32 |
| base_stereo | room1 | 226.5 | 12.45 | 33.92 | 42.37 | 66.59 |
| m1_stereo | room1 | 178.2 | 15.83 | 33.96 | 34.13 | 49.49 |
| **m2_stereo** | room1 | 101.7 | **27.73** | **7.28** | 7.44 | 22.46 |
| base_mono | room6 | 125.3 | 21.03 | 23.02 | 23.06 | 37.33 |
| m1_mono | room6 | 102.4 | 25.74 | 23.26 | 23.30 | 28.53 |
| **m2_mono** | room6 | 48.5 | **54.35** | **3.38** | 3.41 | 8.17 |
| base_stereo | room6 | 215.3 | 12.24 | 35.27 | 44.19 | 68.23 |
| m1_stereo | room6 | 168.8 | 15.62 | 35.31 | 35.49 | 50.48 |
| **m2_stereo** | room6 | 98.6 | **26.72** | **7.80** | 7.99 | 23.79 |

Averaged over the two sequences (`harness/tab.py`, `FPSx`):

| | vs. baseline | vs. M1 |
| --- | --- | --- |
| mono | **2.60x** | 2.08x |
| stereo | **2.21x** | 1.75x |

The update itself: **22.7 -> 3.2 ms mono (7.1x)** and **34.0 -> 7.3 ms stereo
(4.7x)**. Predicted from the flop count was 17x, and the gap is where the note
above said it would be — what is left is the 564x564 `rankUpdate`, which reads and
writes 2.5 MB and is bandwidth-bound. The arithmetic reduction is real but only
about a third of it converts.

Two other readings of the table. The mono `update` column (3.4 ms) is now within
0.03 ms of `actual_update`, so the wrapper around the update — the Jacobian
assembly and inlier bookkeeping — costs essentially nothing; whereas at baseline
the *stereo* `update` (44.2) exceeded `actual_update` (35.3) by 8.9 ms, which was
the stereo gate M1 removed. And the stereo/mono FPS ratio moves from 0.59 at
baseline to 0.50 here: with the update no longer dominating, the second camera's
front-end and Jacobian work is a larger share of what remains, which is what M4
goes after.

RSS is unchanged to within 1% (449 -> 442 MB mono), as expected: the three 564²
scratch matrices this removes are 2.5 MB each against a 450 MB footprint that is
dominated by the loaded image sequence.

## Accuracy

8-member ensembles (`run_ensemble_bugfix.sh`, `X.Vsb` perturbed by k·1e-6 m/s),
6 rooms each, baseline arm from the frozen worktree `xivo-effbase` (`d13ec97`) and
this arm from `xivo-effm2` (`7d219bc`):

| | ATE | RPE_rot | RPE_tra | RPE_rot_i | RPE_tra_i |
| --- | --- | --- | --- | --- | --- |
| base_mono | 0.0796 ± 0.0063 | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| **m2_mono** | **0.0786 ± 0.0049** | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| base_stereo | 0.0551 ± 0.0031 | 0.6208 | 0.0139 | 0.5128 | 0.0132 |
| **m2_stereo** | **0.0549 ± 0.0033** | 0.6208 | 0.0139 | 0.5128 | 0.0132 |

Every RPE figure agrees to the 4 decimals reported, and both ATE means move by
less than a fifth of an ensemble sd. The reason the agreement is this tight is
more interesting than the table:

**91 of the 96 runs are byte-for-byte identical to the baseline's.** Only 5
diverged — mono room3 in members 0, 1, 4, 7, and stereo room1 in member 6 — and of
those, 3 changed the member's 6-room ATE at 4 decimals:

| mono member | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.0758 | 0.0900 | 0.0774 | 0.0781 | 0.0772 | 0.0719 | 0.0885 | 0.0777 |
| m2 | 0.0758 | **0.0821** | 0.0774 | 0.0781 | **0.0764** | 0.0719 | 0.0885 | **0.0786** |

This is what an algebraically-exact-but-reassociated change looks like on a
chaotic filter. `P - WᵗW` and the Joseph form differ at 1e-16 relative on every
update, and `err = Wᵗu` likewise, so the state differs in its last bits
immediately. In 91 runs that difference stayed below the 1e-6 the output records
(`savers.py` writes `tumvi_<seq>_cam0` with `fmt='%f'`) for the whole sequence; in
5 it reached a gate, flipped an accept/reject, and the trajectory changed
macroscopically from there.

So the honest reading of the smaller mono sd (0.0049 vs 0.0063) is *not* that the
new form is more accurate. It is that the baseline ensemble happened to contain a
0.0900 member and this one draws 0.0821 for the same initial condition — one
gate flip in room3. Both numbers are samples from the same ~0.03-wide
distribution ([[../notes-bugfix]] M6; the mean's intrinsic sd over 6 rooms is
~0.007), and 3 of 8 members moving is exactly the rate the chaos argument
predicts. The claim being defended is *no degradation*, and 91/96 runs unchanged
plus 5 relocated within the ensemble spread is a stronger form of it than two
means with overlapping error bars.

(Beware: compare `tumvi_<seq>_cam0`. `tumvi_<seq>_bench` is a two-line header
identical across all arms, and a `cmp` on a relative path from the wrong cwd
reports every file as differing.)
