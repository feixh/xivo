# M3 -- three bit-identical wins outside the linear algebra

After M2 the sampling profile (stereo room3, depth-1 RIP mode, 1000 Hz, 46314
samples = 16.4 ms/frame) looked like this:

```
LKTrackerInvoker                        36.81%   ~6.04 ms   three KLT calls
Eigen "triangular_solve_vector"         12.18%   ~2.00 ms   = the EKF update kernels (misattributed name)
PNG/zlib (inflate 7.07 + libz ~15.0 + png_read_png 5.36 + adler 1.98 + ...)
                                       ~31.3%   ~5.14 ms
__memset_avx512                          1.55%   ~0.25 ms
__introsort_loop<Feature**>              1.47%   ~0.24 ms
ScharrDerivInvoker                       1.19%   ~0.20 ms
__internal_syscall_cancel                1.15%   ~0.19 ms
MakePtrVectorUnique                      0.74%   ~0.12 ms
InnovationCov                            0.58%
Estimator::ComputeOOSMeasurements        0.54%   <- see below
PyrDownInvoker                           0.51%
EkfUpdateDowndate (self)                 0.50%
```

Decode (5.14 ms) and KLT (6.04 ms) are a floor at fixed capacity and a fixed
config, and the update kernels are M2's territory. M3 takes the three items in
between that can be removed without changing a single arithmetic result.

## 1. One `read(2)` per frame instead of ~75

`__internal_syscall_cancel` at 1.15% is glibc's syscall stub, and it is there
because `cv::imread` hands libpng a `FILE*`. libpng asks for a chunk header, then
a chunk, then the next header; glibc's stdio buffer is 4 kB; a 300 kB PNG
therefore becomes ~75 `read(2)` calls, on top of a separate
`fopen`/`fread`/`fclose` that `findDecoder` does just to sniff the signature.

`pybind11/pyxivo.cpp`'s `ReadImage` now does `open`/`fstat`/`read`/`close` into a
`thread_local std::vector<uchar>` that is reused across frames, and calls
`cv::imdecode` on a `cv::Mat` view of it. `imdecode` runs the *same*
`PngDecoder`; the only difference is that its read callback is a `memcpy`
(`grfmt_png.cpp:128`), so the decoded pixels are identical by construction. Any
failure (open, fstat, short read, not a regular file) falls back to
`cv::imread`, which also keeps OpenCV's error logging for a genuinely bad path.

## 2. `cv::noArray()` for the three KLT error outputs

`src/tracker.cpp` asked `calcOpticalFlowPyrLK` for `err` in all three places --
`UpdateLK` (temporal), and `MatchStereo`'s left->right and right->left passes --
and never read any of them. It is not free to ask: OpenCV's `LKTrackerInvoker`
runs a *second* full pass over the 15x15 window of every point at level 0 purely
to accumulate the photometric residual (`lkpyramid.cpp:684-721`).

That block is also the one place where `err` feeds back into `status`, which is
why this needed an argument rather than a glance. Lines 692-698 clear `status` if
the tracked point lands outside the image *by more than half a window*:
`inextPoint.x < -winSize.width || inextPoint.x >= J.cols`, i.e. (win 15, halfWin
7) `x < -8` or `x >= 519`. Every consumer of `status` in `tracker.cpp` already
applies a strictly tighter test:

| call | XIVO's own test on the result | tighter than OpenCV's? |
|---|---|---|
| `UpdateLK` | `MaskValid` rejects `x < 0 \|\| x >= cols` (`tracker.cpp:1015`) | yes, by 8 px |
| `MatchStereo` l->r | `pts_r` outside `[0, cols) x [0, rows)` rejected | yes, by 8 px |
| `MatchStereo` r->l | `\|pts_l_back - pts_l\| > circular_thresh` = 1 px, and `pts_l` is inside the image | yes, by 7 px |

So no point can survive without the check and be rejected with it. The only
observable difference is which *counter* a >8 px-outside back-track lands in
(`num_stereo_rejected_klt_` rather than `num_stereo_rejected_circular_`), and
those counters are printed, never fed back.

## 3. The `Criteria` predicates stopped re-reading the JSON tree

`src/options.cpp` held the worst ParameterServer access pattern in the codebase.
`Criteria::CandidateComparison` is a `std::sort` comparator, and it opened with

```cpp
ParameterServer& P{*ParameterServer::instance()};
std::string score_type = P.get("comparison_score_type", "DepthUncertainty").asString();
```

-- a walk of the JSON map plus a `std::string` construction, `O(n log n)` times
per frame. `Criteria::Candidate` / `CandidateStrict` are the predicates
`Graph::GetFeaturesIf` runs over every feature every frame, three JSON lookups
each. All four values are immutable for the life of a run:
`ParameterServer::Create` refuses to replace a live instance.

They are now read once into a small struct cached on the ParameterServer
*instance pointer* -- not once per process, so a test binary that installs its
own config still sees it (`unittest_determinism.cpp` does exactly that). The
string comparison became an enum resolved at cache-fill time, which also means an
invalid `comparison_score_type` logs once instead of once per comparison.

Bit-identical trivially: the same JSON node yields the same `double` every time.

## Also looked at, and not changed

- **`ComputeOOSMeasurements` at 0.54% despite `use_OOS: false`** is another
  `addr2line` misattribution: `src/manager.cpp:122` already reads
  `use_OOS_ ? ComputeOOSMeasurements() : 0`, and that is the only call site. The
  samples belong to neighbouring inlined code in `UpdateStep`. Same lesson as the
  `triangular_solve_vector` symbol in M2 -- read this profiler's fractions, not
  its names.
- **Filtering `MatchStereo`'s right->left batch.** `calcOpticalFlowPyrLK` is
  per-point independent, so running the back-track only on the points that
  already passed `status_lr`, the bounds test and the disparity test would be
  bit-identical for the survivors. But the shipped rejection statistics say it is
  not worth the restructuring: of 384260 stereo match attempts on room3, only
  8762 (klt) + 238 (disparity) = 2.3% are rejected before the back-track. The
  21401 circular rejections all *need* it. 2.3% of 1.3 ms is 0.03 ms.
- **`H_.setZero(total_size, kFullSize)`**, a 1.28 MB memset per stereo update and
  part of the 1.55% in `__memset_avx512`. After M2 the products only read the 25
  columns each block writes, so in the visual-only case the zeros are formally
  unnecessary -- but a dense (OOS) block reads all `live` columns of its own rows,
  and `CheckLiveExtent` asserts `H` is zero outside `live`, so removing it would
  make a debug build fail and would only be sound as long as no dense block is
  ever present. Not worth the invariant.

## Result

Estimator timers, room3, and the wall clock split (`other` = wall/frame minus
`visual-meas` = decode + Python driver + pybind11):

| ms/frame, room3 | mono M2 | mono M3 | stereo M2 | stereo M3 |
|---|---|---|---|---|
| `track` | 2.320 | 2.277 | 6.790 | 6.666 |
| `process-tracks` | 1.820 | 1.775 | 3.560 | 3.473 |
| `actual-update` | 1.103 | 1.100 | 2.530 | 2.509 |
| `visual-meas` | 4.140 | 4.140 | 10.350 | 10.237 |
| `other` (decode + driver) | 3.330 | 3.160 | 6.190 | 5.960 |
| **wall / frame** | **7.470** | **7.300** | **16.540** | **16.200** |

Attribution: `imdecode` -0.17 mono / -0.23 stereo (one and two decodes per
frame), the KLT `err` pass -0.04 / -0.12 (one and three calls), the `Criteria`
cache -0.05 / -0.09.

Paired wall clock, candidate on cpu 128 and `auto` @ 9e3ec06 on cpu 129 in the
same window, `--timing --no-score`, 2821 frames:

| | base | M2 | M3 | M3/base | FPS base -> M3 |
|---|---|---|---|---|---|
| mono room1 | 29.08 s | 21.53 s | 20.97 s | **1.387x** | 97.0 -> 134.5 |
| mono room3 | 28.61 s | 21.08 s | 20.60 s | **1.389x** | 98.6 -> 136.9 |
| stereo room1 | 67.17 s | 48.24 s | 47.50 s | **1.414x** | 42.0 -> 59.4 |
| stereo room3 | 62.86 s | 46.67 s | 45.70 s | **1.375x** | 44.9 -> 61.7 |

Peak RSS (pinned, so glibc allocates one malloc arena): 123.5 / 122.5 MB mono and
128.1 / 127.5 MB stereo, against the baseline's 131.7 / 133.6 and 137.1 / 137.8.

## Accuracy

All three changes are exact *by construction*, and each argument is above:
`imdecode` runs the same `PngDecoder` with a `memcpy` read callback, `noArray()`
removes an output nothing reads and a `status` test strictly weaker than XIVO's
own, and the `Criteria` cache returns the same `double` from the same JSON node.

Same md5 as M2 and M1 on all four checks:

    mono   room1 b5185115fb76d44726cc6ee861ad6e73
    mono   room3 ee2210f7a5093e5baf852cbb22ab09ca
    stereo room1 6dc11ae2a241e8f5296690a708ba1dd0
    stereo room3 129c2d3f7fd637389847346455ea67c3

**Caveat on what that md5 proves** (learned the hard way in M2, see the
CORRECTION in `m2-batched-sparse-products.md`): `dump/tumvi_<seq>_cam0` is printed
to six decimals, so a matching md5 means "agrees to 1e-6 m", not bit-identity.
The independent evidence that M3 does not perturb rounding is the M2 bisection:
a build with M3's *and* M4's changes in place and only `src/ekf_update.cpp`
reverted to M1 reproduced M1's ensemble trajectory md5 exactly on mono room5
member 2 -- one of the three of 72 ensemble runs that M2's reassociation *did*
visibly move, i.e. a configuration demonstrably sensitive to a one-ulp
perturbation. If M3 changed any arithmetic, that run would have diverged too.

`unitTests_determinism`, `unitTests_ekf_update`, `unitTests_Jacobians`,
`unitTests_jacobians_stereo`, `unitTests_pyramid`, `unitTests_stereo` all pass.
