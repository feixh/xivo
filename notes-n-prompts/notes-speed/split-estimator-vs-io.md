# Where XIVO's one-core wall clock goes: estimator vs image I/O vs driver

Required deliverable: the equivalent of OpenVINS' "estimator 60%, PNG decode 33/40%"
split, for XIVO.

## How it is measured

`print_timing: true` in `cfg/eff_*.json` makes the estimator print cumulative
means every 50 frames. The last line of `run.log` is the whole-run mean per
frame. The harness's `stats.txt` gives `frames_processed` and `wall_total_s` for
the same run, so

    other = wall_total_s / frames - visual-meas

is everything outside the estimator's visual measurement: the PNG decode (which
happens in `pybind11/pyxivo.cpp`, *not* in Python -- the driver passes paths and
C++ calls `cv::imread`), the Python driver loop, the pybind11 boundary, and the
IMU calls. IMU propagation is separately timed at 0.032 ms/call.

`visual-meas` nests as

    visual-meas = track + process-tracks
    process-tracks >= update >= actual-update

`track` is `Tracker::Update`/`UpdateStereo` -- pyramid build, KLT, FAST detect,
and for stereo the right pyramid plus the two stereo KLT passes.
`actual-update` is `EkfUpdateDowndate`.

All numbers below: room3, 2821 frames, `taskset -c 128 setarch -R`,
`OMP/OPENCV/OPENBLAS/MKL_NUM_THREADS=1`, box under load from the two sibling
agents, so treat absolutes as ~5% soft and ratios as sound.

## Baseline (`auto` @ 9e3ec06), room3

| | mono | stereo |
|---|---|---|
| wall per frame | 10.11 ms (98.9 FPS) | 22.36 ms (44.7 FPS) |
| `visual-meas` | 6.39 ms (**63%**) | 15.30 ms (**68%**) |
| ... `track` | 3.79 | 10.25 |
| ... `process-tracks` | 2.51 | 4.95 |
| ... ... `update` | 1.78 | 4.08 |
| ... ... ... `actual-update` | 1.75 | 3.95 |
| ... ... `jacobian` | 0.074 | 0.125 |
| ... ... `MH-gating` | 0.090 | 0.099 |
| ... ... `stereo-gating` | 0 | 0.069 |
| decode + driver ("other") | 3.72 ms (**37%**) | 7.06 ms (**32%**) |

Of "other", `cv::imread` dominates: measured standalone at 3.69 ms per
512x512 16-bit PNG with the default `IMREAD_COLOR` flag
(`harness/bench_io.cpp`), which accounts for essentially all of mono's 3.72 ms
and 7.38/7.06 of stereo's two decodes. The Python loop itself is ~0.4-0.5 ms.

Independent confirmation from the sampling profiler (`harness/sampler.c` +
`resolve.py`, stereo room3, 31461 samples at 500 Hz): `cv::imread` 30.1%
inclusive, `calcOpticalFlowPyrLK` 37.7% inclusive, `UpdateStep` 21.9%.

## After M1 (grayscale decode), room3

| | mono | stereo |
|---|---|---|
| wall per frame | 8.06 ms (124.0 FPS) | 18.07 ms (55.3 FPS) |
| `visual-meas` | 4.83 ms (**60%**) | 11.96 ms (**66%**) |
| ... `track` | 2.32 | 6.79 |
| ... `process-tracks` | 2.43 | 5.07 |
| ... ... `actual-update` | 1.72 | 4.04 |
| decode + driver | 3.23 ms (**40%**) | 6.11 ms (**34%**) |

Decode is now 2.78 ms/image, so mono's "other" is 2.78 decode + 0.45 driver and
stereo's is 5.56 + 0.55.

## Comparison with OpenVINS, one core, 6-room mean

| ms/frame | XIVO base | XIVO M1 | OpenVINS |
|---|---|---|---|
| mono total | 10.4 | 8.3 | 8.74 |
| mono estimator | 6.4 | 4.8 | 5.87 |
| mono decode+driver | 4.0 | 3.5 | 2.87 |
| stereo total | 22.3 | 18.1 | 14.04 |
| stereo estimator | 15.3 | 12.0 | 8.33 |
| stereo decode+driver | 7.0 | 6.1 | 5.62 |

Two conclusions that set the whole plan:

1. **Mono was never an estimator problem.** XIVO's mono estimator was already
   within 10% of OpenVINS' and is now 18% *faster* than it. The entire mono gap
   was the front end reading three colour channels it never used. M1 alone puts
   mono past the target.

2. **Stereo is an estimator problem, and decode is a floor, not a lever.**
   Both systems pay ~5.6 ms to inflate two 512x512 16-bit PNGs -- that is
   libpng/zlib in the shared `dependencies/opencv_install`, identical for both,
   and not something a XIVO change can move. Hitting 71.2 FPS stereo means
   14.04 ms/frame, i.e. an estimator budget of ~8.5 ms against the 12.0 ms M1
   leaves. Every remaining stereo milestone has to come out of `track` (6.8 ms,
   which is three `calcOpticalFlowPyrLK` calls and two pyramids) or
   `actual-update` (4.0 ms).

## The EKF update, decomposed (`harness/bench_update.cpp`)

At the shipped capacity and the measured census (`rows` 283 stereo / 146 mono,
`live-dim` 331, two runs), the dense linear algebra inside
`EkfUpdateDowndate` costs, per update:

| | stereo (rows 283) | mono (rows 146) |
|---|---|---|
| `LLT(S)` | 0.30 ms | 0.06 ms |
| triangular solve, per-run (current) | 0.77 | 0.26 |
| triangular solve, whole 564 width | 1.39 | 0.54 |
| triangular solve, transposed/`OnTheRight` | 0.58 | 0.17 |
| `rankUpdate` + off-diagonal gemm | 0.85 | 0.53 |
| `MirrorLowerTriangle` | 0.02 | 0.02 |
| **sum** | **1.93** | **0.88** |
| measured `actual-update` | 4.04 | 1.72 |
| **unaccounted** | **2.11** | **0.84** |

The unaccounted half is `MeasurementTimesCov` + `CovTimesMeasurementT`, whose
*flop* count is trivial (~4 MFLOP together) but which issue
`nblocks x kJacRuns x live.nruns` = 72 x 6 x 2 = 864 and 72 x 6 = 432 Eigen
gemm calls per update, with M as small as 2-4 and K as small as 1. That is the
target of M2, not the triangular solve. (The sampling profile blamed 14.3% of
stereo CPU on `Eigen::internal::triangular_solve_vector`; the *cost* is real and
matches `actual-update`, but the symbol is a misattribution -- `addr2line`
folded the whole gemm/trsm kernel family onto one name, which is also why no
`gebp`/`rankUpdate` symbol appears anywhere in the profile.)

## Profiling on this box

`perf_event_paranoid` is 4 (no `perf_event_open`), `yama/ptrace_scope` is 1 (gdb
cannot attach to a non-descendant, so a poor-man's profiler is out), and
gperftools is not installed despite `-DUSE_GPERFTOOLS=ON` existing in
`CMakeLists.txt`. `harness/sampler.c` is a 100-line `LD_PRELOAD` replacement:
`ITIMER_PROF` + `backtrace()` into a preallocated buffer, `/proc/self/maps`
appended, resolved offline by `harness/resolve.py`. Usage:

    XIVO_SAMPLER_OUT=/tmp/prof.raw XIVO_SAMPLER_HZ=500 \
      LD_PRELOAD=.../harness/sampler.so taskset -c 187 python3 scripts/pyxivo.py ...
    ./resolve.py /tmp/prof.raw.<pid> --top 40
    ./resolve.py /tmp/prof.raw.<pid> --callers calcOpticalFlow

Read its symbol names with suspicion (see above); read its *fractions* as sound.

## The definitive split: attribute samples to shared objects, not to symbols

The timer-based split above has to guess how much of "other" is decode; the
symbol-based profile has an `addr2line` attribution problem. Both go away if the
RIP is attributed to its **mapping** instead of to a symbol name -- a mapping
cannot be misattributed, and every piece of third-party code lives in a different
`.so` than XIVO's. `harness/sampler.c` with `XIVO_SAMPLER_DEPTH=1` records the
raw RIP only (no `backtrace()`, so no unwinder cost and no inlining ambiguity),
and `/proc/self/maps` is appended for offline resolution.

Stereo room3, M3 build, 1000 Hz, **45324 samples** over a run whose measured wall
clock was 16.2 ms/frame:

| module | share | ms/frame | what it is |
|---|---|---|---|
| `libopencv_video` | 38.04% | 6.16 | `calcOpticalFlowPyrLK` + `buildOpticalFlowPyramid` |
| `libz` | 27.12% | 4.39 | inflate |
| `libpng16` | 6.58% | 1.07 | unfiltering, row assembly |
| `lib/pyxivo...so` | **21.01%** | **3.40** | **all** of XIVO's own code (Eigen is header-only, so it is inlined in here) |
| `libc` | 3.50% | 0.57 | memcpy/memset/malloc |
| `libopencv_imgproc` | 1.75% | 0.28 | FAST, cvtColor, resize |
| `libopencv_core` | 0.49% | 0.08 | Mat allocation, `fastMalloc` |
| `libopencv_features2d` | 0.49% | 0.08 | detector wrapper |
| `libm` | 0.40% | 0.06 | |
| **`/usr/bin/python3.14`** | **0.34%** | **0.055** | the whole Python driver loop |
| `libstdc++`, `libjsoncpp`, `ld.so` | 0.24% | 0.04 | |

Grouping: **PNG decode 33.71% = 5.46 ms**, **OpenCV vision 40.77% = 6.60 ms**,
**XIVO 21.01% = 3.40 ms**, **Python driver 0.34% = 0.055 ms**.

Three things this settles.

1. **The Python driver is not a target.** 0.055 ms/frame, not the 0.4-0.5 ms the
   timer subtraction suggested; that residual was decode all along. Anything
   framed as "make the driver faster" is capped at 0.3% -- which is why M4 spent
   its effort on the driver's *memory* instead.

2. **XIVO's own code is one fifth of the scored metric.** Of its 3.40 ms,
   `actual-update` is 2.51 ms, and that is dense linear algebra at a fixed
   capacity (`kFullSize` 564, `rows` 283, `live-dim` 331). M2 already took the
   easy 25% out of it. Even reducing XIVO's own code to **zero** leaves
   12.8 ms/frame, i.e. **78 FPS**, so the entire remaining headroom on the branch
   is 61.7 -> at most 78 FPS and all of it is in code XIVO does not own.

3. **The 71.2 FPS stereo target is arithmetically unreachable at a fixed
   algorithm.** 71.2 FPS is 14.04 ms/frame. Third-party image code alone is
   5.46 + 6.60 = 12.06 ms = **86% of the entire budget**. OpenVINS fits in
   14.04 ms because its front end does strictly less work, not because its
   estimator is leaner:

   - XIVO does **452 point-tracks per stereo frame** -- 180 temporal, ~136
     left->right, ~136 right->left for the circular consistency check.
   - OpenVINS does 400 (two temporal passes over ~200 points) and **no per-frame
     left<->right matching at all**: `perform_matching` is commented out in
     `ov_core/src/track/TrackKLT.cpp` at both call sites (line 278 and line 668,
     with a `// TODO: we should probably still do this to reject outliers`), so
     cross-camera KLT runs only for newly detected points.

   Dropping XIVO's right->left back-track would save ~1.3 ms/frame and get stereo
   to ~67 FPS, and dropping the whole per-frame stereo match would get past 71 --
   but both delete a rejection test that the filter's measurements depend on, so
   both are accuracy changes, which the brief forbids. Recorded as the reason the
   target is missed rather than proposed as a fix.

The same census for mono (M3 build, room3, wall 7.30 ms/frame): decode is one
image instead of two, so PNG is ~2.73 ms (37%), OpenCV vision ~2.28 ms (31%),
XIVO's own code ~2.16 ms (30%). Mono clears its 114.4 FPS target with 17% to
spare.
