# Plan: make XIVO faster at fixed EKF capacity

Requirements: `notes-n-prompts/requirements-efficiency.md`. Metric is **FPS**;
the constraint is that **ATE and RPE must not degrade**. Delivered on branch
`auto-efficiency` of the xivo package, developed in the worktree
`workspace/xivo-efficiency` (created from `auto`).

## The two settings, and what "fixed capacity" means

Capacity in XIVO is partly compile-time (`EKF_MAX_FEATURES` / `EKF_MAX_GROUPS`)
and partly config (`tracker_cfg.num_features_max`). Both are held fixed for all
of this work, at what the shipped configs ask for:

| | EKF state | tracker | config |
| --- | --- | --- | --- |
| 1. monocular + IMU | 90 features / 45 groups | 135–180 | `cfg/eff_mono.json` |
| 2. stereo + IMU | 90 features / 45 groups | 135–180 | `cfg/eff_stereo.json` |

`cfg/eff_mono.json` and `cfg/eff_stereo.json` are byte-identical to the shipped
`cfg/tumvi_mono_ctl.json` and `cfg/tumvi_stereo.json` apart from
`print_timing: true`. The two differ from each other in exactly three keys
(`stereo`, `stereo_init.enable`, `stereo_update.enable`), so they are a
controlled pair.

Error-state dimension at this capacity is **N = 564** = 24 motion
(`Index::End` with `USE_ONLINE_TEMPORAL_CALIB`) + 6·45 group + 3·90 feature.

Explicitly *out of scope*: lowering capacity, dropping features, loosening
gates, or any other accuracy-for-speed trade. `notes-stereo/cost-and-throughput.md`
already mapped that curve (60/120 is the real-time point); this work is about
making the *same* computation cost less.

## Baseline (measured, not assumed)

`cfg/eff_stereo.json`, room6, one core, `-mode runOnly`,
`notes-efficiency/harness/fps_one.sh`:

```
wall 215.3 s / 2636 frames = 81.7 ms/frame = 12.25 FPS
  actual-update      35.20   43%   dense EKF covariance update
  track              13.46   16%   KLT over two images
  MH-gating           9.29   11%
  stereo-gating       8.78   11%
  propagation         6.58    8%   0.660 ms x 9.97 IMU samples/frame
  PNG decode+Python   6.8     8%   wall - visual-meas - propagation
  jacobian, misc      1.6     2%
```

This agrees with the independent measurement in
`notes-stereo/cost-and-throughput.md` (11.3 FPS on room6), so the baseline is
reproducible across sessions and machine load.

## The finding the whole plan rests on

Every per-feature measurement Jacobian is **stored dense but is structurally
sparse**. `Feature` holds

```c++
Eigen::Matrix<number_t, 2, kFullSize> J_, J_r_;   // 2 x 564
```

of which only ~33 columns can ever be nonzero: the 24 motion columns, the 6
columns of the feature's *reference group*, and its own 3 feature columns. The
filter multiplies by the other 531 columns anyway:

- `MHGating` (update.cpp:66) evaluates `Mat2 S = J * P_ * J.transpose()` as a
  dense 2x564 x 564x564 product, once per in-state feature. That reads the whole
  2.5 MB of `P_` ~90 times per frame; it is memory-bandwidth bound and it needs
  only a 33x33 slice.
- `GateStereoMeasurements` (update.cpp:170) does the same thing again for the
  right-camera rows, which is why stereo gating costs the same order as MH
  gating.
- `UpdateJosephForm` (estimator.cpp:1540) computes `H_ * P_` **twice** --
  once for `S_` and again as the right-hand side of the `ldlt().solve()` -- and
  then spends two full N^3 products on `I_KH_ * P_ * I_KH_.transpose()`.

So the dominant 43% + 11% + 11% = 65% of the frame is dense linear algebra over
a matrix that is ~94% structural zero, plus one duplicated `H*P`.

## Milestones

Each milestone is a commit, is validated before the next one starts, and gets a
note under `notes-n-prompts/notes-efficiency/`.

### M0 — measurement harness and baseline
`notes-efficiency/harness/{fps_one.sh,fps_batch.sh,tab.py}`, the two configs,
and a **state census** printed with the timing block (mean in-state features,
in-state groups, and EKF measurement rows per frame) so that later decisions
about active-set compaction rest on occupancy data rather than on the capacity
number. Baseline numbers for both settings on room1 and room6.

### M1 — sparse Jacobian gating
Give `Feature` an explicit list of its structurally nonzero column blocks and
evaluate `S = J P J^T` from the corresponding submatrix of `P_`, in both
`MHGating` and `GateStereoMeasurements`.

Algebraically exact (the skipped terms are multiplications by exact zero), so it
is checkable by a much stronger test than ATE: the trajectory should be
bit-identical, or differ only by floating-point summation order. Target 18.1 ms
-> ~1 ms.

### M2 — restructure the EKF covariance update
1. Compute `M = H P` once and reuse it (removes a duplicated 114 MFLOP).
2. Compute `M` and `S = M H^T` block-sparsely, from the same column-block lists.
3. Replace `P <- I_KH * P * I_KH^T + K R K^T` (two N^3 products) with an
   O(m N^2) form. Two candidates, to be decided by measurement:
   - restructured Joseph: `AP = P - K M`, then `AP - (AP H^T) K^T + (K sqrt(R))(...)^T`,
     same algebra as today;
   - the symmetric downdate `P <- P - K M` with symmetry enforced by
     construction, which is what OpenVINS's `StateHelper::EKFUpdate` does.
   The second is cheaper; the first keeps Joseph's positive-semidefiniteness
   argument. Guard whichever is chosen with a covariance-health check.

Target 35.2 ms -> <= 10 ms. This changes rounding, so it is validated by
ensemble ATE/RPE, not by a single run.

### M3 — IMU propagation
`PrinceDormandStep` applies the 24x540 motion-to-structure correlation update
(`P_.block(0, kMotionSize) = F_ * P_.block(0, kMotionSize)`) on *every*
Prince-Dormand substep -- ~3 substeps x 9.97 samples = ~30 times per frame --
although nothing reads that block until the visual update. Accumulate the
transition matrix across substeps and samples and apply it once per frame
(exact, up to rounding). Plus: fixed-size 24x24 matrices instead of `MatX`, and
hoist the stage-invariant part of `G Qimu G^T` out of the 7-stage loop.
Target 6.6 ms -> ~1.5 ms.

*Delivered as the accumulate-and-flush part only.* The two smaller items (fixed
24x24 stage matrices, hoisting `G Qimu G^T`) were split out into M6 below, once
the correlation deferral turned out to account for essentially all of the 6.6 ms
on its own: `propagation` went 0.66 -> 0.17 ms/call, and what is left is the
integrator's own arithmetic, which is a different change with a different risk
profile and deserves its own commit and its own ensemble.

### M4 — front end
The left image pyramid is built twice per stereo frame (`Tracker::Update` caches
one in `pyramid_`, `MatchStereo` builds another from the same `img_`), and the
input image is `clone()`d every frame. Reuse and elide. Target 13.5 -> ~10 ms.

### M5 — active-set compaction (conditional on the M0 census)
`P_` is always the full 564x564 regardless of how many slots are occupied, and
removed slots are zeroed rather than excluded. If typical occupancy is materially
below 90/45, gathering the active submatrix for the update shrinks every N^2 and
N^3 cost quadratically. **Only** pursued if the census says the occupancy gap is
real; otherwise this milestone is recorded as measured-and-rejected.

*The census says it is real:* 7.3 of 45 group slots and 76 of 90 feature slots
occupied, `occupied-dim` 296 of 564. **Pursued**, in the cheap form: because both
allocators take the lowest free slot, the occupied region is two contiguous runs
(motion+groups, then features) described by two high-water marks, so no gather or
scatter is needed -- every operation in the update is expressed on the runs of the
existing `P_`. Measured extent is 339 of 564 rather than the 295 a general gather
would reach: ~88% of the achievable saving on the dominant term, for a fraction of
the machinery and with no risk of a permutation bug.

*Delivered.* `actual_update` 3.20 -> 1.86 ms mono and 7.01 -> 4.34 ms stereo; FPS
+11.1% mono and +10.6% stereo, taking the cumulative figures to **4.09x** and
**3.31x**. ATE/RPE unmoved (mono 0.0797 vs baseline 0.0796, stereo 0.0549 vs 0.0551);
47/48 mono runs are byte-identical to the *baseline*, up from 44/48 at M4.

### M6 — propagation internals
The leftovers of M3. `F_` and `G_` are `Eigen::SparseMatrix<number_t>` used only
in 24x24 and 24x15 dense products; `PrinceDormandStep`'s seven stage matrices are
dynamic `MatX` statics; and `G Qimu G^T` is recomputed in each of the seven
stages although it does not depend on the stage. Target: 0.17 -> ~0.03 ms/call,
i.e. ~1.9 -> ~0.3 ms/frame.

*Correction to the above, found while implementing it:* `G Qimu G^T` **does**
depend on the stage, so hoisting it out of the seven-stage loop would have been
wrong. `G` carries `Rsb`, and each stage evaluates the Jacobian at a different
composed state. What is true instead is that `Qimu_` is block diagonal (it is
built as a diagonal and then squared), so `G Qimu G^T` has exactly zero cross
terms and collapses to four 3x3 blocks -- of which three are constant and the
fourth, `Rsb Qa Rsb^T`, is the stage-dependent one. Two further structural facts
that the plan did not have: `F` has only nine nonzero rows (Wsb, Tsb, Vsb; every
other state is a random walk), and `P` is symmetric so `P F^T = (F P)^T`. The
milestone was implemented around those three instead.

*Delivered.* `propagation` **0.170 -> 0.032 ms/call**, a 5.3x, beating the target;
FPS +15.5% mono and +6.9% stereo, taking the cumulative figures to **4.72x** and
**3.54x**. ATE/RPE unmoved in every reported digit (mono 0.0797 ± 0.0063, stereo
0.0549 ± 0.0033, identical to M5); **48/48 mono runs bit-identical to M5** and
47/48 to the baseline, and stereo went *back* to 47/48 against the baseline by
un-flipping the one run M5 had flipped. Attribution, from two scratch worktrees:
59% of the saving is the sparse-to-dense container change alone (a two-line diff),
6% is fixed-size stage matrices, 36% is the structural work. M6 does 1.45x *more*
arithmetic than M5 and runs 5.3x faster.

### M7 — build and flags
Everything so far changed the arithmetic; this milestone changes only how it is
compiled, so every arm must be checked for numerical identity as well as speed.

The effective compile line is
`-O0 -std=c++17 -Wno-narrowing -Wno-register -fPIC -g -mtune=native -march=native
-funroll-loops -O3 -DNDEBUG`.

1. **`-DEIGEN_INITIALIZE_MATRICES_BY_ZERO`** (`CMakeLists.txt:105`) zero-fills
   every dense object and temporary, which at this size is megabytes per frame.
   Measure its cost, and separately settle whether it is load-bearing, since
   removing it is only safe if nothing reads before writing. A build that is
   bit-identical without it proves little (fresh pages read as zero anyway), so
   probe it with `-DEIGEN_INITIALIZE_MATRICES_BY_NAN` instead: any read-before-write
   then poisons the filter visibly.
2. **Whole-program optimization.** `-flto` across `libxivo` and the drivers, and
   `-fno-semantic-interposition` / `-fvisibility=hidden` for the shared library
   (every cross-TU call in a `-fPIC` shared object currently goes through the PLT
   and cannot be inlined). Both can change inlining and therefore FMA
   contraction, so each is validated the same way a numerical change is.
3. **Document the accidents rather than the flags.** `CMakeLists.txt:66` prepends
   `-O0` and Release appends `-O3 -DNDEBUG` after it, so `-O3` wins by flag order;
   line 77 hardcodes `set(CMAKE_BUILD_TYPE "Release")`, so the
   `-DCMAKE_BUILD_TYPE=Release` in every build recipe is a no-op. Neither is a
   bug, both are traps.

The instruction set is *not* an open question, contrary to what this entry said
before M6: `CMakeLists.txt:70-71` sets `-mtune=native -march=native` through the
`XIVO_ARCH_FLAGS` cache variable, so Eigen already vectorizes with the AVX-512
this EPYC 9R14 has. The variable exists to be overridden downwards, because
valgrind cannot execute that AVX-512. Whether *delivery* should default to a
portable `-march` is a packaging decision to record, not to measure.

*Delivered.* The flags were the smaller half of this milestone. Both `-flto` and
`-fvisibility=hidden`/`-fno-semantic-interposition` measured as **noise** (+0.4%
at best, negative in one of eight cells each) — the hot code is templated Eigen in
headers, so there is no cross-TU inlining left to win — and neither is adopted;
`XIVO_LTO` ships as an OFF knob. `-DEIGEN_INITIALIZE_MATRICES_BY_ZERO` becomes
`XIVO_EIGEN_INIT={none,zero,nan}` defaulting to **`none`**, worth **RSS 443 -> 133
MB (3.3x)** and ~0.25 s of start-up, i.e. **FPS 98.17 -> 99.12 mono and 43.76 ->
43.97 stereo** — a 1% that is startup amortization over 2821 frames, not a
per-frame gain, and the per-frame timers confirm it (all flat). The cost is not
per-frame either: the define also enables Eigen's refill in `resize()`, and
`OOSJacobian` resizes a 90x564 matrix per pooled `Feature`, so at
`max_features: 800` it faulted in and zeroed 310 MB one double at a time — which
matches the measured 317,456 kB delta exactly.

It was also **load-bearing, in five places**: `Mat3 Ka`, `Mat3 Kg`, `Vec3 Wsg`,
the six `curr_`/`last_`/`slope_` IMU members, and `Mat34 P1` in
`DirectLinearTransformSVD`, all read before being written and all fixed here. The
methodological result is that **valgrind memcheck reported 0 errors** and
`MALLOC_PERTURB_` gave a bit-identical trajectory while the bugs were live —
memcheck does not re-poison stack memory on frame pop, and `P1` is a stack object.
`XIVO_EIGEN_INIT=nan` plus an `FE_INVALID` `LD_PRELOAD` trap is what found it.
Accuracy: mono ATE **0.0797 -> 0.0786**, stereo **0.0549 -> 0.0549**, all four RPE
statistics unmoved, 93/96 runs bit-identical to M6 with the other three all on
room3 (the chaotic sequence); stereo is 48/48 identical to *M5*. Chaining the
within-batch ratios (x1.0097 mono, x1.0048 stereo) gives cumulative **4.76x** mono
and **3.56x** stereo — and M8 should replace that chain with a single batch
containing both the baseline and the final build, since seven chained ratios now
carry seven batches' worth of drift.

### M8 — validation and report
Full ensembles for both settings, the FPS table, `report-efficiency.md`.

## How each milestone is validated

**FPS.** `fps_batch.sh`, one core (`OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`), `setarch -R`, `XIVO_RANDOM_SEED=0`,
`-mode runOnly`. Runs strictly sequential; arms interleaved within each repeat so
load drift on this shared host hits all arms equally. Absolute FPS is
machine-dependent, so every claim is a within-batch ratio.

**Accuracy.** `XIVO_WT=xivo-efficiency ./run_ensemble_bugfix.sh <cfg> <out> 8`
over room1-room6. Single-run ATE differences below ~0.015 m are *not*
attributable on this system -- hard accept/reject gating makes the trajectory a
chaotic function of the last bits of its input -- so no milestone is accepted or
rejected on a single run. Report ATE@0.001 and ATE@0.02, stock RPE and
interpolated RPE (the stock evaluator scores a zero-error trajectory at 0.2847
deg).

**Exactness, where it applies.** M1 and the `H*P` de-duplication in M2 remove
multiplications by structural zeros; M3's batching is exact up to the order of
matrix products. For these, compare trajectory files directly: bit-identical is
the expected outcome, and a *large* divergence is a bug even if ATE looks fine.

## Risks

- **Chaos masquerading as regression.** Mitigated by ensembles, and by
  preferring changes that are provably exact.
- **Losing positive-semidefiniteness** in M2. Mitigated by keeping a Joseph
  variant available and by a covariance-health assertion.
- **Shared host.** 192 cores, load average has been observed between 10 and 141.
  Pinned single-thread repeats varied <5% even at load 141, and all claims are
  ratios within one interleaved batch.
