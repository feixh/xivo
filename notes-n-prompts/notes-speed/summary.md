# Efficiency milestone: summary

Branch `auto-speed` in `xivo-speed`, forked from `auto` @ `9e3ec06`. Five
commits, four of them milestones. Scored metric: end-to-end frames per wall
second on **one** cpu, `-mode runOnly`, ASLR off, all thread pools pinned to 1,
including PNG decode and the Python driver loop.

## Headline

Final paired timing: candidate on cpu 128, `auto` @ `9e3ec06` on cpu 129, in the
same time window, all six TUM-VI rooms, both modes
(`experiments/results/speed_fps_final` vs `speed_fpsbase_final`).

| 6-room mean, one core | baseline (paired) | **final** | ratio | OpenVINS | target | met |
|---|---|---|---|---|---|---|
| mono FPS end-to-end | 98.1 | **139.3** | **1.421x** | 114.4 | >= 114.4 | **yes, +22%** |
| stereo FPS end-to-end | 44.2 | **62.6** | **1.416x** | 71.2 | >= 71.2 | no, 88% of it |
| peak RSS mono | 131.6 MB | **80.7 MB** | **0.613x** | 89.2 MB | <= 89 | **yes** |
| peak RSS stereo | 137.0 MB | **85.3 MB** | **0.623x** | 95.7 MB | <= 96 | **yes** |

The brief's absolute baseline numbers (96.4 / 44.8 FPS, 131.7 / 137.1 MB) and the
paired baseline measured here agree to ~2%, so the ratios and the absolutes tell
the same story. Worst-case peak RSS over the twelve final runs is 81.2 MB mono and
85.8 MB stereo, so the memory targets are met per-sequence and not just on
average.

Per sequence, FPS:

| | room1 | room2 | room3 | room4 | room5 | room6 |
|---|---|---|---|---|---|---|
| mono base | 96.5 | 97.3 | 98.7 | 99.4 | 99.5 | 97.2 |
| **mono final** | **136.5** | **138.7** | **139.2** | **141.2** | **141.2** | **139.3** |
| ratio | 1.414 | 1.425 | 1.411 | 1.420 | 1.419 | 1.433 |
| stereo base | 42.4 | 43.4 | 43.6 | 45.4 | 46.2 | 44.1 |
| **stereo final** | **60.8** | **63.2** | **63.0** | **62.3** | **64.9** | **61.4** |
| ratio | 1.433 | 1.455 | 1.444 | 1.371 | 1.406 | 1.394 |

Every sequence in both modes improves by 37-46%; mono clears 114.4 on all six.

## Where the one-core wall clock goes (the required split)

Measured by attributing sampled instruction pointers to their **mapping** rather
than to a symbol, which removes the `addr2line` misattribution that made the
symbol profiles unreliable here (`harness/sampler.c` with `XIVO_SAMPLER_DEPTH=1`;
full method and tables in `split-estimator-vs-io.md`). Stereo room3, 45324
samples, 16.2 ms/frame:

| group | share | ms/frame |
|---|---|---|
| OpenCV vision (KLT + pyramids + FAST) | 40.8% | 6.60 |
| PNG decode (libz + libpng16) | 33.7% | 5.46 |
| **XIVO's own code** (`lib/pyxivo*.so`, Eigen inlined) | **21.0%** | **3.40** |
| libc / libm / libstdc++ | 4.1% | 0.67 |
| **the whole Python driver** | **0.34%** | **0.055** |

Mono is the same shape with one decode instead of two: PNG ~37%, OpenCV vision
~31%, XIVO ~30%. Compare OpenVINS: estimator ~60-67%, PNG 33% mono / 40% stereo.

Two things follow, and they set the whole plan.

1. **The Python driver is not a throughput target at all** -- 0.055 ms/frame, not
   the 0.4-0.5 ms a timer subtraction suggests. That is why M4 spent its effort on
   the driver's *memory* instead, where it owned 33 MB.
2. **Mono was never an estimator problem.** XIVO's mono estimator was already
   within 10% of OpenVINS' and is now 18% faster than it; the entire mono gap was
   the front end decoding three colour channels of a grayscale file.

## Why the stereo target is missed, arithmetically

71.2 FPS is 14.04 ms/frame. Third-party image code alone -- libz, libpng,
libopencv_video, libopencv_imgproc, all in the *shared*
`dependencies/opencv_install` that OpenVINS links too -- is 12.06 ms of that
budget, **86%**. Reducing XIVO's own code to *zero* would leave 12.8 ms/frame =
**78 FPS**, so the entire remaining headroom on this branch is 62.6 -> at most 78,
and every bit of it is in code XIVO does not own.

OpenVINS fits in 14.04 ms because its stereo front end does strictly less work,
not because its estimator is leaner:

- XIVO does **452 point-tracks per stereo frame**: 180 temporal, ~136 left->right,
  ~136 right->left for the circular consistency check.
- OpenVINS does 400 (two temporal passes over ~200 points) and **no per-frame
  left<->right matching at all** -- `perform_matching` is commented out at both
  call sites in `ov_core/src/track/TrackKLT.cpp` (lines 278 and 668, with a
  `// TODO: we should probably still do this to reject outliers`), so cross-camera
  KLT runs only for newly detected points.

Dropping XIVO's right->left back-track would save ~1.3 ms (~67 FPS) and dropping
the per-frame stereo match entirely would clear 71 -- but both delete a rejection
test the filter's measurements depend on, so both are accuracy changes the brief
forbids. Recorded as the reason the target is missed, not proposed as a fix.

Consistent with that, stereo is still 2.23x the wall-clock cost of mono (it was
2.22x): the front-end asymmetry is structural, since stereo pays two decodes, two
pyramids and three KLT passes against mono's one, one and one. The 1.6x that
OpenVINS achieves is the "no per-frame stereo matching" decision, not an
implementation advantage.

## Which milestone bought what

Paired ratios are room1/room3 for M1-M3 (fast iteration) and the 6-room mean for
the final state.

| commit | milestone | mono ratio | stereo ratio | peak RSS mono/stereo | arithmetic |
|---|---|---|---|---|---|
| `b5a9458` | M1 decode one channel, not three | 1.255 / 1.253 | 1.266 / 1.238 | 124.0 / 127.7 MB | changed (rounding) |
| `3dbd722` | M2 batch the two block-sparse EKF products | 1.363 / 1.356 | 1.387 / 1.352 | -- | changed (reassociation) |
| `5e7e822` | M3 three things nothing reads | 1.387 / 1.389 | 1.414 / 1.375 | 122.5 / 127.5 MB | exact |
| `03e5aaa` | M4 keep the driver's memory off the budget | 1.421 (6 rooms) | 1.416 (6 rooms) | **81.4 / 88.3 MB** | exact |
| `801d45e` | correct M2's record (comments only) | -- | -- | -- | -- |

- **M1 (`pybind11/pyxivo.cpp`, `src/app/vio.cpp`,
  `src/app/feature_tracker_only.cpp`) -- +25%, the single biggest win.**
  `cv::imread`'s default `IMREAD_COLOR` forced an 8UC3 result out of a 512x512
  **grayscale, 16-bit** PNG, so libpng decoded one plane, OpenCV broadcast it into
  three, and the pyramid, KLT, FAST and canvas all ran on three copies of the same
  data. Standalone per image: decode 3.69 -> 2.78 ms, pyramid 3.61 -> 1.17, KLT
  4.57 -> 2.81, FAST 0.31 -> 0.15. The numpy-buffer overloads are deliberately
  untouched, so a caller handing XIVO an HxWx3 array still gets the old behaviour.
- **M2 (`src/ekf_update.cpp`) -- +10pp, and the only estimator win.**
  `MeasurementTimesCov` and `CovTimesMeasurementT` cost 2.11 ms (stereo) for
  2.3 MFLOP, because they issued one gemm per (row block, column run, live run) --
  ~2538 calls per stereo update with M as small as 2 and K as small as 1, i.e.
  pure Eigen dispatch and packing. Now the four *fixed* shared runs (`Wsb Tsb`,
  `bg`, `Wbc Tbc`, `td` -- 16 of the 25 nonzero columns) are hoisted out and driven
  once over the whole span of consecutive sparse blocks; the group run is driven
  over merged spans (a feature's left/right pair always, consecutive features
  sharing a reference group usually); only the 3-column feature run is left per
  feature. ~2538 calls -> ~270. `actual-update` mono 1.72 -> 1.10 ms, stereo
  4.04 -> 2.53 ms (-37%).
- **M3 (`pybind11/pyxivo.cpp`, `src/tracker.cpp`, `src/options.cpp`) -- +3pp, all
  exact.** (a) `cv::imread`'s `FILE*` path made libpng issue ~75 `read(2)` calls
  per 300 kB PNG through a 4 kB stdio buffer; now one `read` into a reused
  `thread_local` buffer and `cv::imdecode`, which runs the same `PngDecoder` with a
  `memcpy` read callback. (b) Three `calcOpticalFlowPyrLK` calls asked for an `err`
  output nothing reads, and OpenCV runs a *second* full pass over every point's
  15x15 window at level 0 to compute it; `cv::noArray()` now. (c)
  `Criteria::CandidateComparison` -- a `std::sort` comparator -- was walking the
  JSON parameter tree and constructing a `std::string` on every comparison; the
  values are immutable for the life of a run and are now cached per ParameterServer
  instance.
- **M4 (`scripts/pyxivo.py`) -- the memory milestone, 0.60x / 0.63x peak RSS.**
  Both remaining items were in the driver, not the estimator. (a) `import savers`
  at module scope pulled in numpy + transforms3d + scipy_openblas + libgfortran +
  libcrypto = **15.6 MB** that `-mode runOnly` never calls; it moved into `main()`
  behind the same `args.mode != 'runOnly'` predicate the run loop already uses.
  (b) The driver built one list of boxed `(ts, (w, t))` tuples for ~28k IMU samples
  at ~460 B each = **17.7 MB live for the whole run**, bigger than the covariance;
  now `array.array('q')` plus a flat `array.array('d')`, 56 B per sample, and
  `readlines()` became a file iteration (removes a 3 MB transient). Throughput
  unchanged within scatter, as the 0.34% driver share predicts.

## Accuracy: unchanged

Two milestones deliberately change floating-point results, and neither changes the
algorithm or the capacity:

- **M1** is not bit-identical by construction. OpenCV's `LKTrackerInvoker`
  accumulates its window sums over channels, so with three identical channels
  `A11 A12 A22 b1 b2` are all exactly 3x larger; the flow increment
  `(A12*b2 - A22*b1)/D` is mathematically invariant but rounds differently, and the
  filter is chaotic under MH gating.
- **M2** is a reassociation. Per-element accumulation order and K are preserved,
  but merging blocks changes M, and **Eigen's gemm is not shape-invariant in the
  last bit** -- a different M means a different LHS packing and a different
  row-peeling path through `gebp_kernel`. Measured by compiling both forms into one
  binary: they differ on every update, in 1-50% of the elements, by at most 2e-13
  of the matrix's own max magnitude and typically one ulp.

The brief permits this with a full 6-member ensemble proof, and that is the
branch's accuracy evidence: `experiments/results/speed_acc_final` (`--jitter 6`,
`X.Vsb = [k*1e-6, 0, 0]`, six rooms x both modes x six members = 72 runs) against
`experiments/results/xivo_ref_jitter`. Per-member 6-room means, then mean and sd
over the six members:

| metric | mode | baseline (sd) | **final** | delta | in sd_base |
|---|---|---|---|---|---|
| ATE RMSE, 0.02 s window [m] | mono | 0.0928 (0.0067) | **0.0963** | +0.0034 | +0.51 |
| | stereo | 0.0636 (0.0045) | **0.0651** | +0.0015 | +0.33 |
| ATE RMSE, 0.001 s window [m] | mono | 0.0770 (0.0033) | **0.0795** | +0.0025 | +0.74 |
| | stereo | 0.0556 (0.0032) | **0.0560** | +0.0004 | +0.11 |
| ov_eval ATE position [m] | mono | 0.0968 (0.0065) | **0.1002** | +0.0034 | +0.52 |
| | stereo | 0.0688 (0.0042) | **0.0704** | +0.0016 | +0.39 |
| ov_eval ATE orientation [deg] | mono | 1.8243 (0.0395) | **1.7828** | **-0.0415** | -1.05 |
| | stereo | 1.7983 (0.0403) | **1.7586** | **-0.0396** | -0.98 |
| RPE 8 m, translation [m] | mono | 0.0480 (0.0017) | **0.0475** | **-0.0004** | -0.26 |
| | stereo | 0.0292 (0.0012) | **0.0288** | **-0.0004** | -0.36 |
| RPE 8 m, rotation [deg] | mono | 0.5153 (0.0066) | **0.5071** | **-0.0082** | -1.25 |
| | stereo | 0.5074 (0.0063) | **0.5069** | **-0.0005** | -0.08 |

**Every ATE mean moves by less than one ensemble sd in the worse direction; every
orientation and every RPE mean moves in the better direction.** Sign-mixed and
sub-sd on both sides is what "the same estimator, re-rolled" looks like. Per
sequence it is a reshuffle, not a trend: mono room1, room2 and room4 improve,
room3, room5 and room6 degrade. `speed_acc_final` is 69/72 trajectory-identical to
`speed_acc_m1`, and the three that differ (mono room2 member 5, mono room5 members
2 and 4) are M2's reassociation, isolated by bisection.

Supporting evidence:

- **No capacity was reduced.** `EKF_MAX_FEATURES=90`, `EKF_MAX_GROUPS=45`,
  `num_features_min/max=135/180`, `memory.max_features/max_groups=800/300`,
  `PrinceDormand.stepsize` and `tolerance` are all untouched, and
  `cfg/eff_mono.json` / `cfg/eff_stereo.json` are **unmodified** (no
  `config-delta.md` was needed). The per-run census is unchanged to three digits:
  room3 mono feature-slots 72.68 -> 72.66 / 90, group-slots 7.10 -> 7.28 / 45,
  `rows` 145.9 -> 145.7; stereo 72.13 -> 72.17, 7.18 -> 7.31, 282.4 -> 282.8.
  `print_timing` stays on.
- **Trajectory md5s** (`-mode eval`, `dump/tumvi_<seq>_cam0`) are identical across
  M1, M2, M3 and M4 on room1 and room3, mono and stereo:
  `b5185115fb76d44726cc6ee861ad6e73`, `ee2210f7a5093e5baf852cbb22ab09ca`,
  `6dc11ae2a241e8f5296690a708ba1dd0`, `129c2d3f7fd637389847346455ea67c3`.
  (`auto` @ 9e3ec06: `9670c4cbc5e359fa67b4adb250bc092a`,
  `f44de02c1ec8a75a5a04d9786a436dbc`, `8d06caeaacb00ea6cd29d673c2413393`,
  `6f3c60422206593ca8590b7d8f283f86`.)
- **All `bin/unitTests_*` pass**, including `unitTests_ekf_update` (which checks
  the fast update against the dense `EkfUpdateJoseph` reference),
  `unitTests_Jacobians`, `unitTests_jacobians_stereo`, `unitTests_determinism`,
  `unitTests_propagate_cov`, `unitTests_pyramid`, `unitTests_stereo`.

## Residual risks

1. **`md5sum dump/tumvi_<seq>_cam0` is not a bit-identity test, and I initially
   treated it as one.** That file is printed to six decimals
   (`XIVO_DUMP_PRECISE=1` makes it exact), so a matching md5 proves agreement to
   1e-6 m -- which a one-ulp perturbation usually satisfies. M2 was committed
   claiming bit-identity on that basis and the claim was false; `801d45e` corrects
   the code comments and `m2-batched-sparse-products.md` has the full postmortem.
   **Anyone re-verifying this branch should use `XIVO_DUMP_PRECISE=1`.** The
   consequence for the merge is only bookkeeping -- the branch always needed an
   ensemble proof because of M1, and that proof covers M2 -- but the record
   matters.
2. **M2's rounding change interacts with the filter's chaos.** 3 of 72 ensemble
   runs take a visibly different path. That is the expected behaviour of MH gating
   under a last-bit perturbation, and the ensemble shows no regression, but a
   future single-run comparison against `auto` on a sensitive sequence (mono room5
   is one) will not match and should not be read as a bug.
3. **`-march=native` was already on** in `auto` (`XIVO_ARCH_FLAGS`), so the binary
   is non-portable, unchanged from the baseline. `XIVO_LTO` was measured and is
   noise; it is left OFF. No build option was changed by this branch.
4. **M3's `imdecode` path adds an I/O fallback.** Any failure (open, fstat, short
   read, not a regular file) falls back to `cv::imread`, preserving OpenCV's error
   logging. The reused buffer is `thread_local`, so it stays safe if a caller ever
   drives `VisualMeas` from several threads.
5. **M3's `noArray()` removes a `status` test.** OpenCV clears `status` when a
   tracked point lands outside the image by more than *half a window* (`x < -8` or
   `x >= 519` at win 15). Every consumer in `tracker.cpp` already applies a
   strictly tighter test (`MaskValid`, the stereo bounds test, the 1 px circular
   threshold), so no point can survive without the check and be rejected with it.
   The only observable difference is which printed *counter* a >8 px-outside
   back-track lands in. Argued in full in `m3-cheap-bit-identical-wins.md`.
6. **M4 changes the driver's data structures, so `-mode eval` must keep working.**
   It does: the accuracy ensemble runs in `eval` mode, which constructs a saver and
   therefore exercises the lazy `import savers`; a broken import would show up as
   72 missing trajectories rather than as matching md5s. The two-pointer merge
   reproduces the old stable `data.sort` exactly, including the tie rule (image
   before IMU at an equal timestamp) and IMU file order; a non-monotonic IMU file
   is handled by a one-off stable permutation up front.
7. **Merge surface.** Two files in the diff are also being *functionally* modified
   by the sibling agents: `src/tracker.cpp` (position agent) and `src/options.cpp`.
   My edits there are small and local -- three `cv::noArray()` argument changes and
   one cached-struct rewrite of the `Criteria` predicates -- and no file was
   reformatted or restructured.

## Remaining headroom, and what was considered and not taken

XIVO's own code is 3.40 of stereo's 16.2 ms, of which `actual-update` is 2.51 ms of
dense algebra at a fixed capacity (`kFullSize` 564, `rows` 283, `live-dim` 331).
Per-update budget after M2, stereo: LLT 0.30, triangular solve 0.77, `rankUpdate` +
off-diagonal gemm 0.85, the two products 0.59, mirror 0.02 ms.

- **Transposed triangular solve** (`L^T \ M^T`, `OnTheRight`): 0.58 vs 0.77 ms in
  the bench, but it produces `W^T` in a layout where the downdate would read
  strided rows of a column-major matrix. 0.19 ms of a 16.2 ms frame; not taken.
- **Compacting `live` to the occupied set**: `live-dim` is 331 but `occupied-dim`
  285, so 14% of the solve and the downdate touch vacant slots -- but gathering and
  scattering `P` costs 2 x 0.65 MB of traffic to save 14% of ~2 MB. A wash.
- **`H_.setZero(total_size, kFullSize)`**, a 1.28 MB memset per stereo update:
  formally unnecessary in the visual-only case after M2, but a dense (OOS) block
  reads all `live` columns and `CheckLiveExtent` asserts on it. Not worth the
  invariant.
- **Filtering `MatchStereo`'s right->left batch** to the points that already
  passed: only 2.3% of 384260 room3 match attempts are rejected before the
  back-track, so it is 2.3% of 1.3 ms.
- **`Feature`'s `OOSJacobian::Hx`** (`2*kMaxGroup x kFullSize` = 397 kB x 800
  pooled features = a 312.6 MB mapping, 5.7 MB resident thanks to
  `XIVO_EIGEN_INIT=none`). Making it conditional on `use_OOS` would recover a few
  MB of RSS and all of that address space, but it is a layout change in
  `src/jac.h` / `src/feature.h` -- the accuracy agents' files -- and the memory
  targets are already met.
- **zlib-ng** would roughly halve inflate (stereo ~16.2 -> ~13.5 ms, ~74 FPS) but
  OpenVINS decodes the same PNGs through the same shared
  `dependencies/opencv_install`, so it moves both systems and closes no gap; it is
  also a shared-dependency change rather than a XIVO change.
- **Threading / pipelining the tracker and the filter**: the scored metric is one
  cpu, so overlapping cannot help, and it would make completion order
  nondeterministic. Not attempted. (`xivo-async` / branch `auto-async` in the
  workspace is **not mine** -- it did not exist at my first `ls` of the workspace
  and sits at `9e3ec06` with two untracked stereo configs.)

## Artifacts and how to reproduce

    experiments/results/speed_fps_final       final paired timing, candidate (cpu 128)
    experiments/results/speed_fpsbase_final   final paired timing, auto baseline (cpu 129)
    experiments/results/speed_acc_final       final --jitter 6 ensemble, 72 runs, scored
    experiments/results/xivo_ref_jitter       the baseline ensemble it is compared against

Timing (the two must run in the same window; quote the ratio, never the absolute):

    CPU_BASE=128 experiments/openvins/run_xivo_reference.sh --worktree xivo-speed \
      --out experiments/results/speed_fps_x --timing --no-score &
    CPU_BASE=129 experiments/openvins/run_xivo_reference.sh --worktree xivo \
      --out experiments/results/speed_fpsbase_x --timing --no-score &
    wait

Accuracy:

    CPU_BASE=128 CPU_SPAN=60 experiments/openvins/run_xivo_reference.sh \
      --worktree xivo-speed --out experiments/results/speed_acc_x --jitter 6
    experiments/openvins/score_openvins.py experiments/results/speed_acc_x

Build: `taskset -c 128-187 make -j24` in `xivo-speed/build`.

## Files touched

| file | milestone | what |
|---|---|---|
| `pybind11/pyxivo.cpp` | M1, M3 | `ReadImage` helper: `IMREAD_GRAYSCALE`, then one `read(2)` into a reused buffer + `cv::imdecode`, with an `imread` fallback |
| `src/app/vio.cpp` | M1 | same decode flag, so the C++ app does not diverge from the driver |
| `src/app/feature_tracker_only.cpp` | M1 | same |
| `src/ekf_update.cpp` | M2, `801d45e` | batched the two block-sparse products (helpers `SparseSpanEnd`, `MergeEnd`, `GroupRun`, `FeatureRun`, `ZeroOutsideRuns`, `kJacFixedRuns`); comments corrected |
| `src/tracker.cpp` | M3 | `cv::noArray()` for the three unused KLT `err` outputs (three small local edits) |
| `src/options.cpp` | M3 | `Criteria` predicates read their JSON once into a struct cached on the ParameterServer instance |
| `scripts/pyxivo.py` | M4 | lazy `import savers`; IMU in `array.array`; two-pointer merge replacing `data.sort` |

No config file, no CMake option and no capacity constant was changed.
