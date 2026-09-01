# M3 — the motion-to-structure correlation, once per image instead of ~30 times

Commit: `M3: accumulate the motion transition, apply it once per image`
(`fead64f`).

## What was there

Both integrators ended every step with this:

```c++
// update the correlation between motion and structure state
P_.block<kMotionSize, kFullSize - kMotionSize>(0, kMotionSize) =
    F_ * P_.block<kMotionSize, kFullSize - kMotionSize>(0, kMotionSize);
P_.block<kFullSize - kMotionSize, kMotionSize>(kMotionSize, 0) =
    P_.block<kFullSize - kMotionSize, kMotionSize>(kMotionSize, 0) *
    F_.transpose();
```

`kMotionSize` is 24 and `kFullSize - kMotionSize` is 540, so that is two
24x540 blocks, 101 kB each, both rewritten in full. How often:

* `Propagate` runs once per IMU sample, plus once more to bring the state to the
  image timestamp;
* `PrinceDormand` subdivides each interval into steps of `stepsize` = 2 ms, so a
  5 ms IMU interval is ~3 `PrinceDormandStep` calls;
* TUM-VI is 200 Hz IMU / 20 Hz camera, so ~10 samples per image.

**~30 rewrites of 200 kB per image**, i.e. ~6 MB of traffic per frame — and
nothing reads either block until the visual update at the end of the frame. The
measured cost was `propagation` = 0.66 ms *per call* (the estimator's timer
averages per call, and `Propagate` is called ~11 times per image), so ~7 ms of
frame time, against 18.5 ms total after M2. It had been invisible next to a 22.8
ms update; once M2 removed that, it was the largest single item left.

## The change

The block update is linear in `P`, so `n` steps compose:

```
F_n (... (F_1 P)) == (F_n ... F_1) P
```

The integrators now call `AccumulateMotionStructureCorrelation()`, which does one
24x24 product (`Fcross_ = F_ * Fcross_`, ~28 kFLOP, L1-resident), and
`Propagate` applies the accumulated transition once, at the end of the last call
before an image:

```c++
if (visual_meas) { FlushMotionStructureCorrelation(); }
```

That is where it has to land. Everything downstream of a visual measurement —
the gates, the update Jacobians, the slot bookkeeping, `OnePointRANSAC`'s
backup/restore — reads the correlation blocks, and all three visual entry points
(`VisualMeasInternal`, `VisualMeasStereoInternal`,
`VisualMeasPointCloudInternal`) call `Propagate(true)` before touching any of
them. The `dt == 0` early return needed the same flush, since it skips the
integration but not the caller's read.

The one accessor that exposes the stale region is `Estimator::P()`. It already
returned a copy, so it applies the pending transition to that copy:

```c++
MatX P() const { MatX out = P_; ApplyMotionStructureCorrelation(out); return out; }
```

Every other covariance accessor (`Pstate`, `CameraCov`, the per-feature and
per-group blocks) reads a *diagonal* block, which the deferral does not touch.

The application itself also stopped computing the lower block as a separate
product and mirrors the upper one instead, halving what is left. Sequenced
against the deferral this is a small thing; on its own it would have been a 2x on
this line.

## What the mirror does *not* buy

I expected the mirror to be a conditioning improvement as well — two
independently-rounded products should drift apart. It does not: the test measures
the old form's asymmetry as exactly 0. `F U` and `Uᵗ Fᵗ` sum the same products in
the same order inside Eigen's gemm, so they agree bit for bit. The mirror is a
halving of the work, and it makes symmetry structural rather than a property of
the kernel's blocking, but it does not change the numbers. Worth recording
because the opposite claim is the intuitive one, and it would have gone into this
note unchecked.

## Tests

`unitTests_propagate_cov` (new target; needs nothing but `xest`, since
`ApplyMotionTransition` is a free function in `core.h` — same reason as M1's
`InnovationCov`). `F` is generated as `I + 0.002·N(0,1)` in rows 0..8 only, which
is the real shape: `ComputeMotionJacobianAt` writes `FK` only for Wsb, Tsb and
Vsb, the states with dynamics.

| test | what it pins |
| --- | --- |
| `DeferredTransitionMatchesPerStepApplication` | 30 steps (one image), deferred vs per-step, 1e-12 relative |
| `DeferredTransitionMatchesOverManyImages` | 300 steps — far past any real flush interval — 1e-10 |
| `OneStepIsBitIdenticalToTheOldUpperBlock` | with one step the accumulator *is* `F`, so every element must match exactly; catches an accumulator not initialized to the identity |
| `TheAccumulationOrderMatters` | the reversed order is wrong by >1e-6, so the equality tests above are not vacuous |
| `TheResultIsExactlySymmetric` | every element of the result, plus the finding above about the reference |
| `NothingOutsideTheCorrelationBlocksMoves` | the motion and structure diagonal blocks are untouched, exactly |

The order test is the important one. The matrices do not commute, but they are all
close to the identity, so `Fcross_ * F_` instead of `F_ * Fcross_` is wrong by
roughly `dt²·[FK_i, FK_j]` — small enough to look like rounding in an end-to-end
ATE, and it would have been indistinguishable from noise in an ensemble.

20/20 targets pass under `ctest`.

## Speed

`sweeps/m3m4.log` — four arms interleaved inside each (sequence, repeat), two
repeats, one thread each, `setarch -R`, from the frozen worktrees `xivo-effbase`
(`d13ec97`), `xivo-effm2` (`7d219bc`), `xivo-effm3` (`fead64f`) and `xivo-effm4`
(`48a5f54`).

| arm | seq | wall (s) | **FPS** | propagation (ms/call) | visual_meas (ms) | actual_update (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| base_mono | room1 | 134.6 | 20.96 | 0.66 | 37.39 | 23.11 |
| m2_mono | room1 | 52.3 | 53.93 | 0.66 | 8.29 | 3.19 |
| **m3_mono** | room1 | 36.8 | **76.61** | **0.17** | 7.88 | 3.16 |
| base_stereo | room1 | 225.1 | 12.53 | 0.66 | 66.16 | 33.77 |
| m2_stereo | room1 | 101.7 | 27.74 | 0.66 | 22.46 | 7.23 |
| **m3_stereo** | room1 | 85.8 | **32.89** | **0.17** | 21.89 | 7.18 |
| base_mono | room6 | 124.7 | 21.13 | 0.66 | 37.10 | 22.94 |
| m2_mono | room6 | 48.5 | 54.36 | 0.66 | 8.17 | 3.37 |
| **m3_mono** | room6 | 34.0 | **77.55** | **0.17** | 7.74 | 3.33 |
| base_stereo | room6 | 215.0 | 12.26 | 0.66 | 68.13 | 35.29 |
| m2_stereo | room6 | 97.3 | 27.09 | 0.66 | 23.34 | 7.75 |
| **m3_stereo** | room6 | 81.8 | **32.21** | **0.17** | 22.54 | 7.66 |

| | vs. baseline | vs. M2 |
| --- | --- | --- |
| mono | **3.66x** | 1.42x |
| stereo | **2.63x** | 1.19x |

`propagation` is **0.66 -> 0.17 ms per call**, a 3.9x on the timer that covers the
whole of `Propagate`. Not 30x, and it should not be: what the deferral removes is
the correlation rewrite, not the Prince-Dormand integration itself, and the 0.17 ms
that remains is that integration (seven stages of 24x24 products through
`Eigen::SparseMatrix`, plus `G Qimu G^T` recomputed in each). M6 is what goes after
the remainder.

The frame-time arithmetic is worth writing out, because it pins how many times
`Propagate` runs per image without a separate instrumented run — the timer reports a
mean *per call*, and the wall clock reports the total:

| | ms/frame at M2 | at M3 | of which `visual_meas` | left for propagation | implied calls/image |
| --- | --- | --- | --- | --- | --- |
| mono room1 | 18.54 | 13.05 | 0.41 | 5.08 | 10.4 |
| mono room6 | 18.40 | 12.89 | 0.43 | 5.08 | 10.4 |
| stereo room1 | 36.05 | 30.40 | 0.57 | 5.08 | 10.4 |

Three independent measurements landing on 10.4 calls per image, against the 10.0
IMU samples per image that 200 Hz / 20 Hz predicts plus one `Propagate(true)` to
reach the image timestamp — 11 by that count, and less than 11 because a few images
arrive with no IMU sample in between. So the ~7 ms/frame this milestone was aimed at
was real, and ~5.1 ms of it is gone.

The stereo speedup is smaller in *ratio* purely because the denominator is bigger:
the absolute saving is the same 5.1 ms/frame in all three columns, which is what it
should be — the propagation does not know how many cameras there are.

## Accuracy

8-member ensembles (`run_ensemble_bugfix.sh`, `X.Vsb` perturbed by k·1e-6 m/s),
6 rooms each, from the frozen worktree `xivo-effm3` (`fead64f`):

| | ATE | RPE_rot | RPE_tra | RPE_rot_i | RPE_tra_i |
| --- | --- | --- | --- | --- | --- |
| base_mono | 0.0796 ± 0.0063 | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| m2_mono | 0.0786 ± 0.0049 | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| **m3_mono** | **0.0786 ± 0.0049** | 0.6205 | 0.0226 | 0.5126 | 0.0222 |
| base_stereo | 0.0551 ± 0.0031 | 0.6208 | 0.0139 | 0.5128 | 0.0132 |
| m2_stereo | 0.0549 ± 0.0033 | 0.6208 | 0.0139 | 0.5128 | 0.0132 |
| **m3_stereo** | **0.0549 ± 0.0033** | 0.6208 | 0.0139 | 0.5128 | 0.0132 |

Every figure agrees with M2's to the reported precision except mono `RPE_tra`, at
0.0226 against 0.0227 — one unit in the last place reported, from a single member
(the min and max of that column, 0.0221 and 0.0232, are unchanged). The trajectory
census says why the agreement is this tight:

| | identical | differing |
| --- | --- | --- |
| mono, M3 vs M2 | 47 / 48 | `m4/room3` |
| stereo, M3 vs M2 | 48 / 48 | — |
| mono, M3 vs baseline | 44 / 48 | `m0,m1,m4,m7 / room3` |
| stereo, M3 vs baseline | 47 / 48 | `m6/room1` |

**95 of 96 runs are identical to M2's** in every digit the output records. The one
that is not is mono member 4 on room3, whose 6-room ATE moves 0.0764 -> 0.0767;
room3 is the sequence that also accounts for all four of M2's mono divergences from
the baseline, and mono member 4 is one of them. So the same single fragile
(sequence, initial condition) pair absorbs the reassociation again, and the
divergence set versus the *baseline* is the same 5 runs as M2's — not 5 new ones on
top.

That is the expected signature. `(F_n … F_1) P` and `F_n(…(F_1 P))` are the same
number in exact arithmetic and differ at ~1e-16 relative in floating point, so the
state differs in its last bits from the first image; on a chaotically-gated filter
that shows up as an occasional gate flip and nothing else. It is a weaker
statement than M1's (where the argument gives *exact* equality of the decisions)
and the same statement as M2's, at a fifth the divergence rate.

The 300-step unit test is what rules out the failure mode this milestone could
plausibly have had — an accumulated `Fcross_` drifting from the per-step product
over a long flush interval. 1e-10 at 10x the real interval, against a 1e-16
per-step difference, says the composition is stable, not merely correct once.
