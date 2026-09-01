# Front end and image path — branch `auto-frontfast`

Task 3 of the XIVO-vs-OpenVINS efficiency push, scoped to the tracker, the
equalization, the subpixel refinement, the stereo match and the image decode.
Worktree `xivo-frontfast`, branched from `auto` @ c0e7f62. `ctest` is **22/22**
(was 21/21; `PngFast` is new). Nothing under `src/estimator.cpp`,
`src/update.cpp`, `src/feature.cpp`, `src/jac.h` or the OOS path was touched.

## Result

Six rooms, one core (`taskset -c 64` + `setarch -R`), every thread pool at 1,
`-mode runOnly`, serial — the same `--timing` protocol the authoritative
baselines were measured with, run back to back on the same core so the two
columns are paired. `experiments/results/frontfast_TBASE` and
`frontfast_TFINAL`.

| | merged `auto` | **`auto-frontfast`** | ratio | OpenVINS |
|---|---|---|---|---|
| mono FPS | 83.0 | **94.9** | **1.143** | 114.6 |
| stereo FPS | 41.0 | **53.6** | **1.307** | 71.3 |
| mono peak RSS, mean / max [MB] | 94.0 / 101.5 | **92.2 / 98.9** | 0.98 / 0.97 | 88.1 |
| stereo peak RSS, mean / max [MB] | 127.3 / 154.1 | **115.8 / 129.1** | 0.91 / 0.84 | 95.5 |

The base column reproduces the lead's authoritative merged baseline (83.1 mono /
41.1 stereo) to within 0.1 FPS, so these ratios can be applied to it directly.

Accuracy, 6-member `--jitter` ensembles over all six rooms (each member is the
mean over the six rooms; ± is the sd across the six members).
`experiments/results/frontfast_D/summary.csv`.

| metric | merged mono | **final mono** | OV mono | merged stereo | **final stereo** | OV stereo |
|---|---|---|---|---|---|---|
| `ate_002` [m] | 0.0566 ±0.0015 | **0.0555 ±0.0025** | 0.0621 | 0.0472 ±0.0019 | **0.0491 ±0.0021** | 0.0677 |
| `ov_ate_ori_deg` | 0.9104 ±0.0414 | **0.8788 ±0.0304** | 1.5742 | 0.8844 ±0.0354 | **0.8920 ±0.0559** | 1.4440 |
| `ov_rpe8_pos_m` | 0.0263 ±0.0004 | **0.0266 ±0.0008** | 0.0308 | 0.0208 ±0.0004 | **0.0215 ±0.0008** | 0.0265 |
| `ov_rpe8_ori_deg` | 0.5138 ±0.0080 | **0.5126 ±0.0031** | 0.6445 | 0.5154 ±0.0068 | **0.5158 ±0.0083** | 0.5837 |

All eight are strictly better than the OpenVINS column, which is the contract.
Mono is neutral-to-better on three of four metrics and the fourth (`rpe8_pos`,
+0.0003) is inside its own noise; mono's ATE headroom against the floor actually
*grew*, 0.0055 -> 0.0066 m. Stereo spends 0.0019 m of ATE and 0.0007 m of RPE
against a 0.0186 m and 0.0050 m margin, for +30.7% throughput and -25 MB of peak
RSS. That is the one deliberate trade on this branch and it is recorded with both
numbers in `config-delta.md`.

## Where the win comes from

| change | mono | stereo |
|---|---|---|
| `fast_png_decode` (bit-identical) | +9.3% | +9.3% |
| `stereo_matching.back_track=false` + `max_level=2` | — | +27% |
| `KLT.max_level=4` | +3.5% | +2% |

Two of the three are new code; the third is a config value. Full attribution in
`m2-png-decode.md` and `m3-stereo-matching.md`; every measurement is a paired
same-window comparison, per the "single-run ATE is noise" rule, and the accuracy
claims all come from 6-member ensembles, never a single run.

## The two structural facts worth carrying forward

**1. Image decode was the largest single line item in either system, and it was
free to fix.** `cv::imread(IMREAD_GRAYSCALE)` on a TUM-VI 512_16 frame costs
2.81 ms — 21% of XIVO's mono frame. OpenVINS pays 2.82 ms/image, so decode is
5.64 ms of its 14.08 ms stereo frame, i.e. **40% of the 71.3 FPS target is PNG,
not VIO.** Replacing libpng+zlib with libdeflate plus a fused unfilter and 16->8
strip gives 1.42 ms with **zero** changed output bytes (checked by unit test
against `cv::imdecode` and by `cmp` on `XIVO_DUMP_PRECISE=1` trajectories, both
modes). XIVO now enters the comparison with a 2.8 ms/frame structural advantage
on stereo that OpenVINS does not have.

**2. Everything else in the image path is load-bearing for accuracy, and making
it cheaper loses.** Three separate attempts to spend less time per image —
a frozen radial gain map instead of CLAHE, equalizing only the detection image, a
lower FAST threshold on the raw image — each cut `track` by exactly the predicted
amount and each made the *run* slower, because a track that dies lands in the
OOS/MSCKF path, which costs 2-3x more per feature than the equalization that
would have kept it alive. The GAINMAP arm ran 21874 feature inits against
17743; the DETECT arm added 60 MB of stereo peak RSS. Numbers in `m1-*.md`. The
rule that came out of it, and the reason the decode work is the whole of the
accuracy-neutral win:

> Anything that degrades the image the *tracker* sees pays for itself twice over
> in `process-tracks`. `histogram_method=NONE` looks like an efficiency win
> (-1.22 ms) only because it degrades the detector too and the filter then
> carries far fewer features — it buys speed with accuracy.

## What is left, and whose it is

At 94.9 / 53.6 FPS the branch is 19.7 mono and 17.7 stereo FPS short of the
OpenVINS column on its own. In per-frame terms the final config is 10.54 ms mono
and 18.66 ms stereo, against 8.73 and 14.03 ms; of what remains, 1.42 / 2.84 ms
is decode and essentially all of the rest is `process-tracks` — 6.20 ms of the
mono frame and 9.94 ms of the stereo frame. The knob sweep prices the two items
that close it (`use_OOS` -3.10 ms, `consistent_init` -1.84 ms on mono), and both
are in the other agent's half of the tree. Peak RSS is the same story: 98.9 mono
/ 129.1 stereo against 88.1 / 95.5, with the OOS Jacobian buffers being the
large pooled allocation.

Files: `m1-where-the-front-end-time-goes.md` (budget, knob map, three negative
results), `m2-png-decode.md`, `m3-stereo-matching.md`, `config-delta.md`,
`sweep.sh` + `t1.sh` + `harness/` (the measurement tooling).
