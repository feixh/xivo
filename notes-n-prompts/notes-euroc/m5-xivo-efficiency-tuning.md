# M5 -- tuning XIVO's throughput on EuRoC MAV

Milestone M5 of `notes-n-prompts/plan-euroc.md`. Branch `auto-eurocfps`, worktree
`xivo-eurocfps`, forked from `auto-eurocacc` at `b1709d8`.

M4 ended with XIVO and OpenVINS statistically tied on accuracy across all eleven
sequences. It also ended with XIVO **31% slower**: one core, stereo+IMU, full 11,

| | ms/frame | FPS | peak RSS |
| --- | --- | --- | --- |
| XIVO (M4 config) | 14.764 +- 0.008 | 67.7 | 103.0 MB |
| OpenVINS | 10.739 +- 0.002 | 93.1 | 101.0 MB |

so M5's target is a **4.03 ms/frame** gap. Both figures are n=3 aggregates
(total wall / total frames) under the one-core protocol of §1.

**Outcome: 14.764 -> 11.588 ms/frame, 67.7 -> 86.3 FPS, for +0.003 m of ATE.**
XIVO ends M5 7.9% slower than OpenVINS instead of 37.5%, still winning 3 of the
5 accuracy metrics.

Getting there needed one idea that was not on the list of knobs. Instrumenting
both estimators (§2) shows the gap is not where a "XIVO is unoptimised" story
would put it: XIVO's PNG decode is *already faster* than OpenVINS' `imread`, and
the largest single front-end item is 2.06 ms/frame of CLAHE that OpenVINS does
not spend at all. CLAHE cannot be made cheaper by configuration (§3.1) and
removing it costs 0.007 m of ATE -- until you ask *what it buys*, which turns out
to be neither feature supply nor match quality but the spatial distribution of
corners, obtained by lifting dark regions over FAST's fixed threshold (§5).
Lowering the threshold on the raw image buys most of that back for 0.36 ms
instead of 2.06 (§8.2). That one substitution is 2.6 of the 3.2 ms.

What remains is structural and is quantified in §9.3: ~1.8 ms is a covariance
update over a deliberately larger state, which every front-end knob leaves
untouched. §4 gives the measured exchange rate for the dozen knobs that were
rejected -- they cluster between **0.11 and 0.46 ms/frame per 0.001 m of ATE**,
against the shipped substitution's 1.01.


## 1. Protocol

Everything in this milestone is measured with `run_xivo_reference.sh --timing`:

* `-mode runOnly` -- no visualisation, no trajectory scoring, no per-frame dump.
* `taskset -c $CPU_BASE setarch -R` -- one core, ASLR off.
* Every thread pool pinned to 1 (`cv::setNumThreads(0)`, OpenVINS'
  `num_opencv_threads=1`, XIVO's `ekf_update.chunks` left serial).
* Sequences run **serially**, never concurrently, because two pinned processes
  still contend for L3 and memory bandwidth.

Reproducibility under this protocol is sd **0.002 ms on 14.9 ms** (0.013%), so
any delta above ~0.01 ms/frame is real. That is what makes a 0.2 ms knob
measurable at all, and it is why the protocol is worth its wall-clock cost.

Two rules govern what may ship, both learned the expensive way in earlier
rounds:

1. **Screen on three sequences, confirm on eleven.** The timing screen uses
   `MH_01_easy V1_02_medium V2_03_difficult` -- one from each of the three
   EuRoC environments, chosen so that a knob whose cost depends on scene texture
   cannot hide. Screening is ~4x cheaper, but no screen number is ever quoted as
   a result.
2. **No timing arm ships without its accuracy measured on all eleven**, as a
   jitter ensemble (`--jitter 3`), against the M4 reference. A throughput knob
   is a *trade* until proven otherwise, and the two arms in §6 show that the
   trade can be catastrophic rather than merely bad.

Accuracy numbers below are ensemble means over 3 jitter members x 11 sequences,
reported on five metrics: `ate_002` (`evaluate_ate.py`, 0.02 s window; blind to
a global rotation) and `ov_eval error_singlerun posyaw`'s ATE pos / ATE ori /
RPE-8 pos / RPE-8 ori. The M4 reference and the OpenVINS target:

| | ate_002 | ov ATE pos | ov ATE ori | RPE8 pos | RPE8 ori |
| --- | --- | --- | --- | --- | --- |
| XIVO M4 (n=6) | 0.095 | 0.103 | 1.72 | 0.111 | 0.85 |
| OpenVINS (n=6) | 0.094 | 0.097 | 1.77 | 0.117 | 0.90 |


## 2. Where the time actually goes

Neither estimator's shipped output says where its frame budget goes at this
granularity, so the first work of M5 was instrumentation.

### 2.1 XIVO: per-stage front-end timers

`src/tracker.{h,cpp}` gains `Tick`/`Tock` pairs around every distinct piece of
front-end work -- `equalize-left`, `equalize-right`, `pyramid`, `klt`,
`stereo-pyramid`, `stereo-klt`, `stereo-match`, `detect-fast`, `detect-sort`,
`detect-subpix`, `detect-total` -- plus two counters, `mean_raw_detections()`
and `num_detect_frames()`, which turn out to carry the whole explanation in §7.3.
`src/manager.cpp` prints the tracker's timer from `ProcessTracks`.

One trap worth recording: `estimator.cpp:1631` also prints a tracker timer, but
that is the **tracker-only** code path, not the one a stereo VIO run takes. Timers
read from there are silently measuring nothing.

`pybind11/pyxivo.cpp` gains a `decode_timer_` and a `ReportDecode()` binding,
called from `scripts/pyxivo.py`'s `finally` block, because PNG decode happens in
the Python-facing wrapper and is invisible to the estimator's own timers. It is
22% of XIVO's frame, so leaving it out would have made every subsequent ratio
wrong.

### 2.2 A caveat in `common/timer.h`, and its fix

`Timer`'s stream operator did

```cpp
auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(e.duration);
os << ... << ms.count() / (float)e.occurrence << " ms\n";
```

-- it truncated the *accumulated total* to integer milliseconds before dividing
by the occurrence count. The reported per-call mean was therefore quantised to
`1/occurrence` ms and biased low by up to that much.

This was noticed because `detect-sort` and `detect-subpix` printed the
**bit-identical** `0.240924 ms`, which two unrelated code regions have no
business doing. The arithmetic confirms the mechanism exactly: detection ran on
303 frames, and `73 / 303 = 0.2409240...`. Both stages had accumulated into the
same integer-millisecond bin.

On the totals in this milestone (hundreds of ms over thousands of calls) the
error is under 0.4%, so **the numbers reported here stand**. The cast is fixed
anyway -- divide in nanoseconds -- and the occurrence count is now printed, since
every stage number below had to be hand-amortised by that count and having it in
the output removes a step where a reader can go wrong.

### 2.3 OpenVINS: parsing what it already reports

OpenVINS emits `[TIME]: %f seconds for <stage>` under `--verbosity DEBUG`. Three
things are needed to make it add up: strip the ANSI colour codes, collapse the
per-bin variants (`(N feats)`, `(N clones in state)`) with
`re.sub(r'\s*\(.*\)', '', name)` or every bin becomes its own stage, and divide
by the frame count rather than trusting `fps_mean` (which is a mean of
per-frame rates and so is not the reciprocal of the mean frame time).

MH_01_easy, stereo, one core, amortised over all 3678 frames
(`results/ov_dbg/run.log`):

| stage | ms/frame |
| --- | --- |
| feature tracking | 3.222 |
| SLAM update | 1.494 |
| re-triangulate & marginalise | 0.930 |
| MSCKF update | 0.411 |
| SLAM delayed init | 0.370 |
| IMU propagation | 0.078 |
| **sum of parts** | **6.504** |
| reported `total` | 6.082 |

The 0.42 ms the parts exceed `total` by is stages nested inside others; it is
small enough not to change any conclusion, and the comparison below uses the sum
so that nothing is dropped.

### 2.4 The head-to-head, MH_01_easy, one core

| stage | XIVO (M4) | OpenVINS | XIVO - OV |
| --- | --- | --- | --- |
| image read / decode | 3.296 | 3.475 | **-0.18** |
| feature tracking | 6.257 | 3.222 | **+3.03** |
| EKF propagation | 0.033 | 0.078 | -0.05 |
| EKF update | 3.813 | 2.275 | **+1.54** |
| marginalise / bookkeeping | 0.953 | 0.930 | +0.02 |
| **estimator total** | **11.00** | **6.50** | **+4.50** |

Two things fall out immediately, and both are the opposite of the obvious guess.

**XIVO's image path is already faster than OpenVINS'.** The `libdeflate` PNG
decode from the earlier `auto-frontfast` work beats `cv::imread` by 0.18 ms/frame.
There is nothing to win here.

**72% of the gap is the front end** -- and OpenVINS is doing *more* work in it,
carrying roughly 600 point-tracks against XIVO's 334. Per tracked point XIVO
spends ~10.0 us against OpenVINS' ~4.5 us. That is where M5 has to look, and
§3 says where inside it the time sits.


## 3. The front-end budget, and the timing screen

XIVO's 6.26 ms front end on MH_01_easy decomposes as (per-call means x calls per
frame):

| stage | ms/frame | note |
| --- | --- | --- |
| `equalize-left` + `equalize-right` | 2.06 | CLAHE, both cameras, every frame |
| `stereo-match` | 2.22 | of which `stereo-klt` 1.87, `stereo-pyramid` 0.25 |
| `klt` | 1.46 | temporal, 4 pyramid levels |
| `pyramid` | 0.27 | |
| `detect-total` | 0.16 | 1.92 ms, but only on 8.4% of frames |

`detect-total` being 0.16 ms/frame is the first useful surprise: detection is
loud in the profile per call and nearly free in aggregate, because XIVO only
re-detects when the tracked count falls below `num_features_min`. Any knob aimed
at the detector is aimed at 2.5% of the frame.

The screen then measured 18 single-knob arms against a 14.882 ms/frame control
(3 sequences, n=2, sd 0.001, RSS 104.2 MB):

| arm | patch | ms/frame | delta | RSS MB |
| --- | --- | --- | --- | --- |
| eqnone | `histogram_method=NONE` | 11.952 | **-2.930** | 103.1 |
| eqdethist | `equalize_for=DETECT` + `=HISTOGRAM` | 12.385 | -2.497 | 102.6 |
| eqhist | `histogram_method=HISTOGRAM` | 12.880 | -2.002 | 103.7 |
| eqdet | `equalize_for=DETECT` | 12.954 | -1.928 | 106.6 |
| noos | `use_OOS=false` | 13.268 | -1.614 | 94.7 |
| nf150 | `num_features_max=150 min=112` | 13.791 | -1.091 | 104.3 |
| klt2 | `KLT.max_level=2` | 14.318 | -0.564 | 99.5 |
| oosw10 | `OOS.pose_window=10` | 14.335 | -0.547 | 107.3 |
| nofast | `fast_png_decode=false` | 14.582 | -0.300 | 100.9 |
| klt3 | `KLT.max_level=3` | 14.647 | -0.235 | 111.4 |
| seed | `stereo_matching.seed_prev_disparity=true` | 14.670 | -0.212 | 106.6 |
| oosobs8 | `OOS.max_observations=8` | 14.751 | -0.131 | 103.8 |
| subpixoff | `subpix_refine=false` | 14.835 | -0.047 | 105.3 |
| chunk6 / chunk8 | `ekf_update.chunks=6/8` | 14.887 / 14.881 | inert | |
| chunk2 | `ekf_update.chunks=2` | 15.142 | +0.260 | |

`subpixoff` is a check on the new instrumentation rather than a candidate: the
timer says `detect-subpix` is 0.243 ms on 8.4% of frames, i.e. 0.020 ms/frame,
and turning it off saves 0.047. Same order, correct sign -- the amortisation
arithmetic is sound.

`nofast` deserves its own note because it is *negative* here. `fast_png_decode`
is the `libdeflate` path added in `auto-frontfast`; on TUM-VI it wins because it
fuses a 16-bit-to-8-bit strip conversion into the decode. EuRoC images are
already 8-bit, so there is no conversion to fuse and the path is a **0.300
ms/frame loss**. Turning it off on EuRoC is free in the strongest sense
available: `results/euroc_pngcheck/` confirms the trajectory files are
**byte-identical** with it on and off, on MH_01_easy and V1_02_medium.

A second round, on a control with `nofast` already applied (base3 = 14.569):

| arm | patch | ms/frame | delta |
| --- | --- | --- | --- |
| kltwin11 | `KLT.win_size=11` | 12.337 | -2.232 |
| kltwin13 | `KLT.win_size=13` | 13.601 | -0.968 |
| kltiter10 | `KLT.max_iter=10` | 13.852 | -0.717 |
| kltiter15 | `KLT.max_iter=15` | 14.112 | -0.457 |
| clahegrid2 | `clahe_grid_size=2` | 14.228 | -0.341 |
| stereolvl1 | `stereo_matching.max_level=1` | 14.297 | -0.272 |
| clahegrid4 | `clahe_grid_size=4` | 14.415 | -0.154 |
| clahegrid16 | `clahe_grid_size=16` | 14.892 | +0.323 |

### 3.1 CLAHE cannot be made cheaper by configuration

The `clahegrid` row settles a question worth asking, since equalization is the
single largest front-end item. Going from an 8x8 grid to 2x2 -- 64 tile
histograms down to 4 -- saves only 0.341 of CLAHE's 2.06 ms. The cost is
therefore **not** the per-tile histograms but the bilinear interpolation pass,
which touches all 360,960 pixels with four LUT lookups per pixel regardless of
grid size.

The obvious follow-up -- is the CLAHE object being rebuilt per frame? -- is no:
`clahe_` is constructed once at config-parse time (`src/tracker.cpp:196`). And
`equalize-left` reads 0.000 ms under `NONE`, which confirms that the `ToGray`
call inside that timer is a genuine no-op on EuRoC's already-8-bit images, so
the 1.031 ms is `clahe_->apply` and nothing else. At 360,960 px that is 2.86
ns/px, which is the expected range for OpenCV's non-vectorised interpolation
body.

So CLAHE is 2.06 ms/frame, it is all interpolation, and the only configuration
that removes it is removing it. What that costs is §5.


## 4. The accuracy price of every knob

Each surviving arm was then run on all eleven sequences, n=3, and scored against
the M4 reference (`results/euroc_fps_acc/`). The last column is the exchange
rate -- ms/frame saved per 0.001 m of `ate_002` given up -- so **higher is a
better deal**.

| arm | delta ms | ate_002 | delta ATE | ms per 0.001 | verdict |
| --- | --- | --- | --- | --- | --- |
| nofast | -0.300 | *byte-identical* | 0 | free | **ship** |
| klt2 | -0.564 | 0.095 | 0.000 | free | **ship** (also -4.7 MB RSS) |
| seed | -0.212 | 0.097 | +0.002 | 0.11 | see §7 |
| k_iter15 | -0.457 | 0.096 | +0.001 | 0.46 | see §8 |
| eqnone | -2.930 | 0.102 | +0.007 | 0.42 | Pareto |
| eqdethist | -2.497 | 0.101 | +0.006 | 0.42 | dominated by eqnone |
| oosw10 | -0.547 | 0.099 | +0.004 | 0.14 | no |
| k_iter10 | -0.717 | 0.100 | +0.005 | 0.14 | no |
| nf150 | -1.091 | 0.103 | +0.008 | 0.14 | no |
| k_win13 | -0.968 | 0.104 | +0.009 | 0.11 | no |
| k_win11 | -2.232 | 0.106 | +0.011 | 0.20 | no, dominated by eqnone |
| eqhist | -2.002 | 0.106 | +0.011 | 0.18 | no |
| k_slvl1 | -0.272 | 0.100 | +0.005 | -- | disqualified, §7.2 |
| eqdet | -1.928 | **0.222** | -- | -- | **disqualified, §6.1** |
| noos | -1.614 | **0.219** | -- | -- | **disqualified, §6.2** |

Two knobs are genuinely free -- `klt2` and `nofast`, together **-0.875 ms/frame
and -5.9 MB** at unchanged accuracy to three decimals on all five metrics. Every
other knob is a trade, and the trades cluster tightly between 0.11 and 0.46
ms per 0.001 m. That number is the real finding of §4: it does not vary by an
order of magnitude across a dozen unrelated knobs, which is what "XIVO is at a
point on a smooth accuracy/compute curve" looks like from the outside.

`k_iter15` at 0.46 is the single best deal found. Note that `k_iter10` is
three times worse at 0.14, so 30 -> 15 KLT iterations is nearly free while
15 -> 10 is not: 15 is the knee, not a point on a slope.


## 5. What CLAHE actually buys, since it is not what one would guess

`eqnone` costs +0.007 ATE, which makes CLAHE load-bearing. The interesting part
is the mechanism, because two plausible explanations are both wrong.

It is **not** keypoint supply. With CLAHE, FAST at threshold 20 returns **6873**
raw corners per detecting frame; without it, **1593** (the
`mean_raw_detections()` counter, MH_01_easy). 1593 is still an order of
magnitude more than the 180 features XIVO wants, so the detector is not starved.

It is **not** stereo match quality. Removing CLAHE *raises* the stereo match
rate, 79.2% -> 85.6%, and cuts epipolar rejections from 100k to 71k. CLAHE's
local, per-tile mapping makes the same physical patch look different in the two
cameras where a tile boundary falls differently; without it the two images agree
better.

What is left is **spatial distribution**. FAST at a fixed global threshold fires
only where local contrast already exceeds it, so on the raw image the corners
concentrate in the well-lit, well-textured parts of the frame. CLAHE lifts the
dark regions above the threshold, spreading features across the image, and a
better-spread feature set is better conditioned for pose. The per-sequence
pattern in `eqnone` fits: the Vicon-room sequences with the widest dynamic range
lose (V1_02 0.070 -> 0.091, V2_02 0.088 -> 0.104, MH_02 0.038 -> 0.063) while
the evenly-lit ones actually improve (MH_01 0.087 -> 0.074, V1_03 0.169 ->
0.154).

That diagnosis has a testable consequence, which §8.2 tests: if CLAHE is buying
distribution through the threshold, then lowering `FAST.threshold` on the raw
image should buy some of it back for a fraction of 2.06 ms.


## 6. Two disqualifications

Both of these look excellent on the timing screen and are unshippable. They are
the reason for rule 2 in §1.

### 6.1 `equalize_for=DETECT` -- ATE 0.222

`equalize_for=DETECT` defers equalization out of the per-frame path and into
`DetectionImage()`, so it is computed only on the 8.4% of frames that detect.
-1.928 ms/frame for what looks like pure bookkeeping.

It takes `ate_002` from 0.095 to **0.222**, with V2_03_difficult at
1.530 +- 2.213 -- i.e. intermittently diverging.

The mechanism is a real incompatibility, not a tuning artifact. Under `DETECT`,
FAST and `cornerSubPix` run on the equalized image while the KLT tracks the
**raw** one. Global histogram equalization is a monotonic point mapping: it moves
gradient magnitudes but not their locations, so a corner found on the equalized
image sits exactly where it sits on the raw image. CLAHE is *local* and not
monotonic across tile boundaries, so corners found on the CLAHE'd image do not
land on gradients of the image the KLT will actually track. Every new feature
starts with a sub-pixel bias, and on the sequence with the least margin that
compounds into divergence.

This predicts that `DETECT` should be safe with `HISTOGRAM`, and it is:
`eqdethist` is 0.101, a normal trade on the §4 curve. Both halves are cheap and
only the *combination* with CLAHE is broken -- which is exactly why the config
generator now documents `equalize_for=DETECT` as unsafe with CLAHE rather than
leaving it as a knob that appears to be worth -1.9 ms.

### 6.2 `use_OOS=false` -- ATE 0.219

Dropping the out-of-state (MSCKF) update entirely saves 1.614 ms and 9.5 MB and
takes ATE to **0.219** (V2_03 1.097, V1_03 0.363). The OOS path is load-bearing
on EuRoC, which is consistent with M4's finding that the difficult sequences
depend on it. `oosw10` -- shrinking the pose window from 20 to 10 rather than
removing the path -- is a normal trade at 0.14, and also not worth taking.


## 7. The stereo match

`stereo-match` is 2.22 ms, the second-largest front-end item, and unlike CLAHE it
looked like it might contain a real inefficiency.

### 7.1 The unseeded search

`MatchStereo` runs a KLT from the left pixel coordinate to find the right-image
correspondence, starting the search **at the left coordinate itself** -- i.e.
assuming zero disparity. Real disparities on EuRoC reach ~150 px. `stereo_matching.seed_prev_disparity` seeds the search from the previous frame's
disparity for that feature id with `cv::OPTFLOW_USE_INITIAL_FLOW`.

Measured on MH_01_easy (final `stereo-klt` value / match rate):

| arm | stereo-klt | match rate |
| --- | --- | --- |
| base (unseeded, 2 levels) | 1.875 | 79.2% |
| `seed_prev_disparity` | 1.641 | 82.0% |
| `stereo_max_level=1`, unseeded | 2.129 | 54.5% |
| `stereo_max_level=1`, seeded | 1.479 | 67.9% |

It is worth being explicit about what is *not* the problem, since both were
checked: `BuildOwnedPyramid` already passes `withDerivatives=true`, so the Scharr
gradients are precomputed and not recomputed per iteration; and `reuse_left`
correctly reuses the temporal pyramid whenever `stereo_max_level <= max_level`
and the window sizes match, so no second pyramid is being built.

Per matched point, the stereo KLT costs 12.2 us against the temporal KLT's
8.1 us. Both do about 1.3x their level-0 work, so the 1.5x ratio is
**entirely** the missing initial guess. A perfect seed's ceiling is therefore
~0.63 ms; `seed_prev_disparity` captures 0.234 of it, because it can only seed
features that were matched in the previous frame.

### 7.2 Why `stereo_max_level=1` is a trap

`stereolvl1` shows -0.272 ms on the screen and looks like a cheap win. It is the
one arm in this milestone whose wall-clock saving comes from **failing**.

Unseeded with one pyramid level, the coarse level cannot bracket a 150 px
disparity inside a 15 px window, so points iterate to the 30-iteration cap and
still fail: the match rate collapses to 54.5% and `stereo-klt` goes *up* to
2.129 ms. The apparent saving is downstream -- a quarter of the stereo
measurements no longer exist, so the update has less to do. Its ATE (0.100) even
looks like an ordinary trade. Disqualified on the mechanism, not the metric.

This is the clearest argument in the milestone for instrumenting stages rather
than trusting end-to-end wall clock: without `stereo-klt` and the match-rate
counter, `stereolvl1` reads as a 0.272 ms win with a modest accuracy cost.

### 7.3 The geometric seed, and why it was not built

The obvious next step is a *geometric* seed: predict the right-image location
from the feature's current 3-D estimate and the known stereo extrinsics, which
would approach the 0.63 ms ceiling instead of 0.23 of it.

It was not built, and the reason is worth recording because it is an
architectural fact rather than a judgement call. `Feature::Xc()`
(`src/feature.h:180`, `src/feature.cpp:154`) returns the point in its
**reference** camera frame, not the current one. The correct current-frame
prediction already exists -- `Xc1 = Rc1c0 * cache_.Xcn + Tc1c0` then
`cam1->Project(project(Xc1))`, at `src/feature.cpp:991-1006` -- but only inside
`ComputeJacobian`, which runs *after* `MatchStereo`. Getting it earlier means
threading the estimator's propagated pose into the `Tracker` singleton.

That is a real coupling between the front end and the filter, for ~0.4 ms of the
4.03 ms gap, and it would not change any conclusion in §9. Recorded as future
work instead.


## 8. Composition

All four candidate knobs touch the same KLT, so their savings cannot be assumed
to add. Two compositions were measured (3-sequence screen, n=2; `nofast` is baked
into the regenerated config and so into every control from here on):

| | patches | ms/frame | FPS | RSS |
| --- | --- | --- | --- | --- |
| base3 | `nofast` only | 14.569 | 68.6 | 102.2 |
| ship1 | + `klt2` | 14.007 | 71.4 | 97.1 |
| final_A | + `k_iter15` + `seed` | 13.405 | 74.6 | 100.9 |
| final_B | + `eqnone` | 10.686 | 93.6 | 105.3 |

Composition is close to additive: `final_A`'s naive sum is -1.233 and it
delivers -1.164 (94%); `final_B`'s is -4.163 and it delivers -3.883 (93%). The
lost few percent is the expected overlap -- `k_iter15` and `seed` both reduce the
same KLT's iteration count.

`final_B` at **10.686 ms/frame lands on OpenVINS' 10.664** on the same
three-sequence screen: a dead heat, within 4 sd.

### 8.1 The stage-level after-picture

The per-stage timers make it possible to see exactly what moved
(MH_01_easy, one core, per-call means):

| stage | base3 | ship1 | final_A | final_B |
| --- | --- | --- | --- | --- |
| `equalize-left` | 1.028 | 1.030 | 1.031 | **0.000** |
| `equalize-right` | 1.032 | 1.034 | 1.035 | **0.001** |
| `pyramid` | 0.273 | 0.255 | 0.254 | 0.253 |
| `klt` | 1.456 | 0.970 | 0.963 | 0.925 |
| `stereo-pyramid` | 0.250 | 0.250 | 0.249 | 0.247 |
| `stereo-klt` | 1.874 | 1.936 | 1.338 | 1.241 |
| `detect-fast` | 1.343 | 1.352 | 1.347 | **0.633** |
| `detect-sort` | 0.240 | 0.243 | 0.241 | **0.055** |
| `detect-subpix` | 0.240 | 0.243 | 0.241 | 0.222 |
| **`track`** (front end) | **6.251** | **5.816** | **5.223** | **2.934** |
| `process-tracks` | 4.764 | 4.671 | 4.727 | 4.724 |
| `actual-update` | 3.684 | 3.601 | 3.655 | 3.669 |
| `propagation` | 0.032 | 0.032 | 0.032 | 0.032 |
| `decode` | 3.290 | 3.288 | 3.290 | 3.289 |

The accounting closes. For `final_A`: `decode` 3.290 + `track` 5.223 +
`process-tracks` 4.727 + `propagation` 0.032 x 10 IMU/frame = 13.56 against a
measured 13.603 ms/frame, i.e. 99.7%. Same for base3 (14.63 vs 14.673) and
final_B (11.27 vs 11.340).

Three things are visible here that no end-to-end number shows:

* **`actual-update` is invariant** at 3.65 +- 0.04 across all four arms. Every
  front-end knob leaves the covariance update untouched, which is the reason
  §9's remaining gap is where it is.
* **`stereo-klt` rises** from 1.874 to 1.936 under `klt2` alone: with two
  temporal pyramid levels instead of four, features land slightly less
  accurately, so the stereo search starts slightly further away and iterates
  more. `k_iter15` + `seed` then take it to 1.338. A knob's cost can appear in a
  stage it does not touch.
* **`detect-fast` halves and `detect-sort` drops 4.4x** under `eqnone` -- because
  raw detections fall from 6873 to 1593 (§5), and both FAST's non-maximum
  suppression and the response sort scale with the candidate count. This is the
  one place where removing CLAHE pays a second time.

And the head-to-head from §2.4, re-run at `final_B`:

| stage | XIVO `final_B` | OpenVINS | XIVO - OV |
| --- | --- | --- | --- |
| image read / decode | 3.289 | 3.475 | **-0.19** |
| feature tracking | 2.934 | 3.222 | **-0.29** |
| EKF path (update + marg + init + prop) | 5.04 | 3.28 | **+1.76** |
| **total** | **11.34** | **9.99** | **+1.35** |

At `final_B` XIVO's decode *and* front end are both faster than OpenVINS', and
**the entire residual gap is the EKF path**. That is a much sharper statement
than the milestone started with, and it is the honest shape of the result: the
remaining difference is not implementation slack but state size -- XIVO carries
90 in-state features and a 20-pose OOS window against OpenVINS' 50 SLAM features
and 11 clones.

### 8.2 Recovering CLAHE's benefit with a cheaper detector threshold

§5 concluded that CLAHE's 0.007 ATE is bought by lifting dark regions above
FAST's fixed threshold, not by supply or by match quality. That predicts a much
cheaper substitute: **drop `FAST.threshold` on the raw image** instead of
spending 2.06 ms/frame raising the image's contrast to meet it. Detection runs on
8.4% of frames, so anything paid there is amortised 12x.

Three arms on a `klt2 + eqnone` base (full 11, n=3, and the one-core screen):

| arm | `FAST.threshold` | raw kps/frame | ms/frame | ate_002 | ov ATE pos | ov ATE ori | RPE8 pos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k_ship (CLAHE) | 20 | 6913 | 14.007 | 0.095 | 0.103 | 1.69 | 0.109 |
| B0 | 20 | 1615 | 11.194 | *0.307* | *0.315* | *3.25* | 0.113 |
| B10 | 10 | 4076 | 11.524 | 0.101 | 0.109 | 1.70 | 0.111 |
| **B7** | **7** | **6357** | **11.550** | **0.098** | **0.106** | **1.68** | **0.110** |

The prediction holds, and the supply match is almost exact: **threshold 7 on the
raw image yields 6357 candidates per detecting frame against CLAHE-at-20's
6913.** For 0.356 ms/frame the arm recovers 0.004 of `eqnone`'s 0.007 ATE
regression -- an exchange rate of 0.09 ms per 0.001 m *spent*, which is far
better than any of §4's rates for the same accuracy *bought*.

Measured against the M4 baseline, `B7` is **-3.019 ms/frame for +0.003 ATE:
a rate of 1.01**, two and a half times better than the best single knob in §4
(`k_iter15` at 0.46) and the only arm in the milestone that changes what §9 can
conclude.

Two details are worth recording.

**`B0` is a composition hazard that neither of its halves shows.** `eqnone`
alone is a well-behaved 0.102, and `klt2` alone is free at 0.095, but *together*
they produce a near-divergent V2_02_medium member at **2.344 +- 3.900 m** (which
is why B0's mean reads 0.307). Two independently safe economies -- one removing
contrast, the other removing two pyramid levels -- leave the front end with no
margin on that sequence. Both threshold arms fix it: V2_02 returns to 0.091
(B10) and 0.097 (B7). This is the concrete case for rule 1's "confirm on eleven":
the hazard is invisible on the three-sequence screen and invisible in both
single-knob arms.

**B0's apparent speed advantage over B7 is partly the speed of failing.** On
MH_01_easy the two are within 0.014 ms of each other (11.810 vs 11.824), and the
stage timers show detection accounting for only +0.083 ms/frame of the
difference (0.077 -> 0.160 amortised). The 0.356 ms aggregate gap comes almost
entirely from V2_03_difficult (10.039 -> 11.059) and V1_02_medium (11.167 ->
11.515) -- the sequences where B0 is dropping tracks and therefore has less
downstream work. Same failure signature as `stereolvl1` in §7.2, caught the same
way.

The 0.003 ATE that `B7` does not recover is presumably the part of CLAHE's effect
that candidate count cannot capture: a global threshold drop adds corners
wherever contrast is already near the threshold, i.e. mostly in regions that were
already textured, whereas CLAHE's local mapping specifically promotes the dark
ones. Matching the count is not the same as matching the distribution.


## 9. The operating point

### 9.1 Knob prices are not independent, so the composition had to be re-screened

Once `histogram_method=NONE` is in the configuration, three knobs that were
nearly free in §4 stop being so. Measured on the `B7` base, full 11, n=3:

| arm | patch on top of B7 | ate_002 with CLAHE | ate_002 without |
| --- | --- | --- | --- |
| B7i | `KLT.max_iter=15` | 0.096 (+0.001) | 0.104 (+0.006) |
| B7is | + `seed_prev_disparity` | 0.097 (+0.002) | 0.115 (+0.017) |
| C7 | `KLT.max_level=4` (i.e. no `klt2`) | 0.095 (=) | 0.099 (+0.001, and slower) |

The mechanism is the same one in each row: a KLT on a lower-contrast image has
weaker gradients, needs more iterations to converge, and has less margin to
absorb a worse initial guess. `k_iter15` was the best exchange rate in the whole
milestone at 0.46 -- and on the shipped front end it is 0.08, one of the worst.

This is the single most transferable lesson of M5, and it is why the §4 table is
presented as measurements rather than as a menu: **a per-knob price list is only
valid at the configuration where it was measured.** `C7` also confirms that
`klt2` stays: dropping it is both slower and slightly worse.

### 9.2 The Pareto pair

Two configurations survive everything above, and neither dominates the other:

| | patches vs M4 | ms/frame (screen) | ate_002 | ov ATE pos | ov ATE ori | RPE8 pos | RPE8 ori |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M4 reference | -- | 14.882 | 0.095 | 0.103 | 1.72 | 0.111 | 0.85 |
| **A** (`k_ship`) | `nofast`, `klt2` | 14.007 | 0.095 | 0.103 | 1.69 | 0.109 | 0.85 |
| **B** (`B7`) | + `eqnone`, `FAST.threshold=7` | **11.550** | 0.098 | 0.106 | **1.68** | **0.110** | 0.86 |
| OpenVINS | -- | 10.664 | 0.094 | 0.097 | 1.77 | 0.117 | 0.90 |

**A is free**: -0.875 ms/frame and -5.9 MB RSS at accuracy identical to M4's on
all five metrics to three decimals. It is not a trade and there is no argument
for not taking it.

**B costs 0.003 m of ATE for a further -2.457 ms/frame.** At a rate of 1.01 it is
the best trade available by a factor of two and a half, and it is the only thing
in this milestone that makes the throughput comparison close.

**B is the shipped configuration.** Against OpenVINS it wins 3 of the 5 accuracy
metrics (ATE ori 1.68 vs 1.77, RPE8 pos 0.110 vs 0.117, RPE8 ori 0.86 vs 0.90),
loses the two position ATEs (0.098 vs 0.094, 0.106 vs 0.097), and closes the
throughput gap from 31% to roughly 8%. A is kept documented, and reachable in one
flag (`--histogram_method CLAHE --fast_threshold 20`), for anyone who wants M4's
position accuracy back at 23% slower.

Both are regenerated by `make_euroc_cfg.py` from the dataset's own `sensor.yaml`,
and both are **one configuration for all eleven sequences** -- no per-sequence
tuning anywhere in this milestone.

### 9.2.1 Confirmation of the shipped configuration

Two things had to be checked before B could be called done, because neither
follows from the screen.

**The regenerated config reproduces the hand-patched arm.** Every accuracy number
above came from `sweep_xivo.sh` patching keys into a copy of the M4 config; the
shipped config comes from the generator's new defaults. They are not the same code
path. Run on all eleven, n=3 (`results/euroc_fps_ship`), they agree on every
metric:

| | ate_002 | ov ATE pos | ov ATE ori | RPE8 pos | RPE8 ori |
| --- | --- | --- | --- | --- | --- |
| `B7`, hand-patched | 0.098 | 0.106 | 1.68 | 0.110 | 0.86 |
| shipped `cfg/euroc_stereo.json` | 0.098 | 0.106 | 1.68 | 0.110 | 0.86 |

**The screen's 11.550 ms holds on the full eleven.** One core, n=3,
`results/euroc_fps_ship11/`:

| | ms/frame | FPS | peak RSS |
| --- | --- | --- | --- |
| XIVO M4 (baseline) | 14.764 +- 0.008 | 67.7 | 103.0 MB |
| **XIVO shipped (B)** | **11.588 +- 0.002** | **86.3** | **102.2 MB** |
| OpenVINS | 10.739 +- 0.002 | 93.1 | 101.0 MB |

**-3.176 ms/frame, +27.4% throughput**, and the three-sequence screen predicted
11.550 against the eleven-sequence 11.588 -- within 0.04 ms, which is the best
evidence available that the screen set was chosen well rather than luckily.

XIVO ends M5 **7.9% slower** than OpenVINS, down from 37.5%. Per sequence
(ms/frame, n=3):

| sequence | M4 | shipped | sequence | M4 | shipped |
| --- | --- | --- | --- | --- | --- |
| MH_01_easy | 15.215 | 11.823 | V1_02_medium | 14.942 | 11.507 |
| MH_02_easy | 15.438 | 12.090 | V1_03_difficult | 13.699 | 10.517 |
| MH_03_medium | 15.041 | 11.905 | V2_01_easy | 14.491 | 11.314 |
| MH_04_difficult | 14.634 | 11.895 | V2_02_medium | 14.463 | 11.367 |
| MH_05_difficult | 14.729 | 11.967 | V2_03_difficult | 14.136 | 11.032 |
| V1_01_easy | 14.828 | 11.561 | | | |

The gain is uniform -- between 2.7 and 3.4 ms on every sequence -- which is what a
front-end saving should look like, and is the last check that nothing here is a
single-sequence artifact.

### 9.3 What is left, and why it is not a configuration problem

The M5 target was 4.03 ms/frame. B closes about 3.3 of it. The residual is
accounted for, and none of it is reachable by tuning:

* **~1.8 ms is the EKF path** (§8.1): `actual-update` is 3.65 ms and provably
  invariant under every front-end knob screened here. XIVO carries 90 in-state
  features and a 20-pose OOS window against OpenVINS' 50 SLAM features and 11
  clones, and §6.2 shows the OOS path is load-bearing on EuRoC -- removing it
  takes ATE to 0.219. Shrinking it (`oosw10`) is a 0.14-rate trade. Reducing this
  means a cheaper update at the same state size, i.e. code, not config. The
  sequential-chunk work already merged at `b565b25` is the last thing that moved
  it; `ekf_update.chunks` is now inert or negative (§3).
* **~0.4 ms is the unseeded stereo search** (§7.3), which needs the estimator's
  propagated pose threaded into the `Tracker` singleton to fix properly.
* **The remaining 0.003 m of ATE** between B and M4 is CLAHE's spatial
  distribution of corners, which a global threshold cannot reproduce (§8.2). A
  spatially-adaptive detector threshold -- one bucket per tile, targeting a
  per-tile candidate count -- would plausibly recover it at a fraction of CLAHE's
  cost, and is the most promising single item left on this axis.

Stated plainly: after M5, XIVO's image decode and feature front end are both
*faster* than OpenVINS' (§8.1), and the entire remaining throughput difference is
a covariance update over a deliberately larger state. That is a design
difference, not unexploited slack, and the exchange rates in §4 are what buying
it back would cost.


## 10. Reproduction

All commands from `experiments/openvins`, with the venv on `PATH`:

```sh
export PATH="$PWD/../../dependencies/venv/bin:$PATH"
SEQS11="MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult \
V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult"
```

**One-core timing screen** (3 sequences, n=2), and its report:

```sh
WORKTREE=xivo-eurocfps REPEATS=2 ./sweep_fps_batch.sh \
  'base3' \
  'ship1 tracker_cfg.KLT.max_level=2'
python3 report_fps.py ../results/euroc_fps/{base3,ship1}
```

**The shipped configuration's two confirmations** (§9.2.1) -- accuracy on all
eleven from the regenerated config, and its one-core throughput:

```sh
cd ../../xivo-eurocfps
for m in stereo mono; do extra=""; [ $m = mono ] && extra="--mono"
  python3 scripts/make_euroc_cfg.py --base cfg/eff_$m.json \
    --seqdir ../data/euroc/MH_01_easy --out cfg/euroc_$m.json $extra
done
cd ../experiments/openvins

CPU_BASE=0 CPU_SPAN=60 ./run_xivo_reference.sh --profile euroc_mav \
  --mode stereo --jitter 3 --worktree xivo-eurocfps --cfg-prefix euroc \
  --out ../results/euroc_fps_ship
python3 agg_ensemble.py --mode stereo \
  --arm B7 ../results/euroc_fps_acc/B7 --arm shipped ../results/euroc_fps_ship

SEQS="$SEQS11" WORKTREE=xivo-eurocfps REPEATS=3 \
  OUT=../results/euroc_fps_ship11 ./sweep_fps_batch.sh 'ship_B'
python3 report_fps.py ../results/euroc_fps_ship11/ship_B \
  ../results/euroc_fps_base/xivo_r0
```

**Full-11 accuracy for a timing arm** (n=3 jitter ensemble):

```sh
OUT=../results/euroc_fps_acc WORKTREE=xivo-eurocfps MEMBERS=3 SEQS="$SEQS11" \
  ./sweep_batch.sh 'k_ship tracker_cfg.KLT.max_level=2'
python3 agg_ensemble.py --mode stereo \
  --arm base ../results/euroc_fps_acc/base \
  --arm k_ship ../results/euroc_fps_acc/k_ship
```

`--mode stereo` is not optional: the XIVO runner writes mono and stereo rows
into one `summary.csv`, and omitting it averages the two.

**PNG decode equivalence** (`nofast` is byte-identical, not merely
accuracy-neutral):

```sh
diff -r ../results/euroc_pngcheck/fast ../results/euroc_pngcheck/nofast
```

**OpenVINS per-stage breakdown:**

```sh
./run_openvins.sh --seqs MH_01_easy --mode stereo --timing \
  --extra '--verbosity DEBUG' --out ../results/ov_dbg
python3 - <<'EOF'
import re, collections
t = collections.Counter()
for l in open('../results/ov_dbg/stereo/MH_01_easy_r0/run.log', errors='replace'):
    l = re.sub(r'\x1b\[[0-9;]*m', '', l)
    m = re.search(r'\[TIME\]: ([\d.]+) seconds for (.+)', l)
    if m:
        t[re.sub(r'\s*\(.*\)', '', m.group(2)).strip()] += float(m.group(1))
for k, v in t.most_common():
    print(f'{v/3678*1000:8.3f} ms/frame  {k}')
EOF
```

Harness files live in `experiments/`, which is **not** version controlled; the
copies under `notes-n-prompts/notes-euroc/harness/` are the tracked ones. New in
M5: `sweep_fps.sh`, `sweep_fps_batch.sh`, `report_fps.py`. Modified: `sweep_batch.sh`
gains an `$OUT` passthrough (without it an M5 arm silently overwrites an M4 arm
of the same name in the shared results root), and `sweep_xivo.sh` gains a
`mkdir -p "$OUT"` before its log redirect (without it a fresh `--out` root fails
*at the redirect*, and the variant is reported as FAILED with no log to explain
why).
