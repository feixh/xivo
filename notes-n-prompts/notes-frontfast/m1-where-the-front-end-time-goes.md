# M1 — Where the front end's time actually goes, and the three things that don't work

Branch `auto-frontfast`, worktree `xivo-frontfast`, forked from `auto` @ c0e7f62.
All timings one core, `taskset -c <cpu> setarch -R`, every thread pool at 1,
`-mode runOnly`, room1, cpus in the 64–123 band. Every comparison below is
**paired**: the arms in a table were started together in the same window on
adjacent cores, because the box is shared and absolute FPS drifts 2–4% between
windows. Identical configs run in the same window land within 10 ms of each
other over a 75 s sequence, which is the noise floor of the paired form.

## The per-frame budget, merged `auto` config

| | mono | stereo |
|---|---|---|
| total (wall / frames) | 13.46 ms | 26.28 ms |
| `track` (mine) | 3.85 | 9.96 |
| `process-tracks` (other agent) | 6.09 | 10.69 |
| everything else = decode + IMU + driver | ~3.5 | ~5.6 |

`visual-meas ≈ track + process-tracks`, so the third row is what the estimator
never sees: PNG decode, the 10 `InertialMeas` calls per image frame, and the
Python loop. The Python loop itself is not in it — 11 pybind calls per frame is
~11 µs — so that row is essentially **image decode**.

## The single largest item in either system is PNG decode

`bench_decode` on room1/cam0, 300 images, one core:

```
A cv::imdecode GRAYSCALE             2.809 ms/img
B cv::imdecode UNCHANGED(16u)        2.566 ms/img
```

2.81 ms/image is 21% of XIVO's monocular frame and 21% of its stereo frame (two
images). It is not XIVO's fault and not a cost a live camera pays, but it is
inside the measured throughput — and OpenVINS pays it too, at the same rate:
`experiments/results/ov_fps_onecore/stereo/room1_r0/stats.txt` reports
`wall_imread_s=15.920` for 2821 pairs = **2.82 ms/image**, so the 71.3 FPS
target decomposes as

```
14.08 ms/frame end-to-end  =  5.64 decode  +  8.34 track+update  +  0.10 other
```

That reframes the whole task: 40% of the stereo target is a shared PNG cost, and
the estimator half of the target is 8.34 ms, not 14. M2 halves XIVO's half of
it. See `m2-png-decode.md`.

## The knob map for the rest (lead's 6-room sweep, mono, n=6)

| arm | Δms/frame | whose |
|---|---|---|
| `use_OOS=false` | −3.10 | other agent |
| `consistent_init.enable=false` | −1.84 | other agent |
| `histogram_method=NONE` | −1.22 | mine |
| `histogram_method=HISTOGRAM` | −0.97 | mine |
| `OOS.pose_window=10` | −0.45 | other agent |
| `subpix_refine=false` | −0.06 | mine |

So inside `track`, CLAHE is 1.22 ms of the mono frame and `subpix_refine` is
0.06 ms. `subpix_refine` is worth −0.0062 m of ATE for that 0.06 ms and stays
on; there is nothing to win in `cornerSubPix` window sizes or iteration counts
and I did not look further (confirmed by microbench: 45 points at win 5 / 20
iters is 0.286 ms, and the tracker refines far fewer than 45 per frame).

## Stage microbenchmark (`bench_front`, room1/cam0, 300 images, one core)

```
decode(GRAYSCALE)           2.759 ms/img
CLAHE(10,8x8)               0.784
equalizeHist                0.253
gainmap Q8 (1 pass)         0.154
tile-LUT 8x8 gather         0.432
CLAHE on half res           0.251
pyramid win15 L3/L4/L5      0.196 / 0.198 / 0.200
FAST(20)+sort, raw          0.130   (301 kps/img)
FAST(20)+sort, CLAHE        0.535  (3076 kps/img)
KLT 180 pts L5/30 iters     2.365
KLT 180 pts L5/15 iters     2.235
KLT 180 pts L4/30 iters     2.041
KLT 180 pts L3/30 iters     1.761
```

Two things to read off it. **Pyramid depth is free** — L3 and L5 differ by
0.004 ms — so a shallower pyramid only pays off through the KLT solve, not
through building it. And **CLAHE's 0.784 ms buys a 10x larger keypoint supply**,
which is the mechanism behind its −0.0158 m of ATE; it is not correcting a
gradient. That is what kills the next three ideas.

## Negative result 1 — a static radial gain map is not a CLAHE substitute

The idea (and the lead's first suggestion): the fisheye's brightness falloff is
a property of the lens, so estimate a radial gain once and apply it as one
fixed-point multiply, 0.154 ms instead of 0.784 ms. Implemented as
`histogram_method: "GAINMAP"` (still in the tree, config-gated, default off).

Measured, paired room1 mono: **GAINMAP 75.71 FPS vs base 80.55** — worse, not
better. `track` did fall, 3.579 → 3.205 ms, exactly as predicted; but
`process-tracks` rose 5.632 → 6.683 and peak RSS 101.9 → 112.2 MB, because the
run initialised 21874 features instead of 17743. More track turnover costs more
in the EKF than the image pass saves.

`bench_gain` explains why the gain map does not replace CLAHE:

```
raw            kps/img     278   saturated 1.66%
CLAHE                     2919
gain-multiply              363   saturated 1.83%
gain-affine                357
```

Saturation is not the problem, and the radial correction barely moves FAST's
supply. The measured radial mean profile falls 48.4 → 6.8 from centre to corner,
but that is the fisheye's **dark corners outside the ~190° image circle**
dominating the outer annuli, not vignetting inside the circle. A gain that flat
in the middle cannot manufacture local contrast, and local contrast is the whole
of what CLAHE gives FAST.

## Negative result 2 — lowering the FAST threshold cannot buy the supply either

Keypoints per image on the raw image, room1/cam0, as the FAST threshold drops:

```
20 -> 270    12 -> 422    10 -> 497    7 -> 711    5 -> 1002
```

against CLAHE at threshold 20 giving 2853. There is no threshold that reaches
CLAHE's supply, and the ones that come closest are dominated by low-contrast
corners that will not survive a KLT track.

## Negative result 3 — equalizing only the detection image makes things worse

The lead's second suggestion, and the more promising one: run the pyramid, the
temporal KLT and the stereo match on the raw image, and pay for CLAHE only on
the frames where `DetectLK` runs (which is only when the feature count falls
below `num_features_min`), and only for the left camera. The detector then sees
*byte-identical* input to today, so CLAHE's accuracy mechanism is preserved
exactly. Implemented as `tracker_cfg.equalize_for: "ALL" | "DETECT"` (still in
the tree, default `ALL` = current behaviour).

Paired room1, same window:

| arm | FPS | `track` | `process-tracks` | `oos-jacobian` | peak RSS |
|---|---|---|---|---|---|
| mono base | 75.29 | 3.85 | 6.09 | 0.374 | 101.7 MB |
| mono `DETECT` | 73.29 | **3.42** | **6.91** | 0.601 | 121.2 MB |
| stereo base | 38.72 | 9.79 | 9.98 | 0.639 | 136.4 MB |
| stereo `DETECT` | 38.53 | **8.70** | **11.23** | 1.001 | 195.7 MB |

The image pass got cheaper by exactly the predicted amount (−0.43 mono, −1.09
stereo — one CLAHE on non-detect frames, two on stereo) and the run got *slower*
both times, with 60 MB more peak RSS on stereo. Tracking on the un-equalized
image loses tracks; lost tracks land in the OOS/MSCKF path, which is 2–3x more
expensive per feature than the CLAHE that would have kept them.

This is the same failure as GAINMAP and it generalises:

> **Anything that degrades the image the *tracker* sees pays for itself twice
> over in `process-tracks`.** `histogram_method=NONE` is only faster because it
> degrades the *detector* too, so the filter carries far fewer features — it
> buys speed with accuracy, not with efficiency.

So the equalization stage is not reducible without giving up accuracy, and I
stopped trying. Both knobs stay in the tree, off, because the negative result is
worth being able to re-run.

## What that leaves

- Decode: 2.81 ms/image, provably reducible with **zero** change to any output
  byte. → M2.
- The stereo left→right matching structure, 9.96 ms of `track`, where XIVO does
  strictly more work per frame than OpenVINS does. → M3.
- The KLT solve's iteration budget and pyramid depth: small, and not free. → M3.
