# M3 — The stereo left→right match: 9.96 ms → 6.19 ms of `track`

Keys (all under `tracker_cfg.stereo_matching`, all defaulting to today's
behaviour): `back_track`, `max_level`, `seed_prev_disparity`,
`max_disparity_jump`. Plus `tracker_cfg.KLT.max_level`, which already existed.
Code: `src/tracker.cpp` `Tracker::MatchStereo`.

## Why stereo is 26.3 ms when mono is 13.5

Stereo does not merely decode a second image. Per frame it runs, on top of
everything mono does:

1. a second `buildOpticalFlowPyramid`, on the right image;
2. a left→right `calcOpticalFlowPyrLK` over every tracked feature, **unseeded**
   — the initial guess is the left pixel itself, so the search has to cross the
   entire disparity from the top pyramid level down;
3. a right→left `calcOpticalFlowPyrLK` over the *same* batch for the circular
   consistency check — a second full LK solve;
4. per-feature unprojection of both bearings and an epipolar residual.

Items 2 and 3 are two full KLT solves at `max_level=5` with `max_iter=30`, i.e.
about 2.3 ms each on 180 points, which is where the 9.96 ms of `track` goes.
OpenVINS's stereo path runs one left→right LK and rejects on RANSAC-fitted
epipolar geometry, not on a back-track — it never pays item 3.

## Three changes, and why each is safe

**(a) `max_level=2` for the stereo search only.** The temporal KLT needs 5
levels because a fast rotation moves a feature a long way. The *stereo* offset is
bounded by the baseline and the scene depth, so it does not need the same coarse
levels — and once (c) seeds the search, the coarse levels have nothing to find
that the seed does not already supply. The saving is not in the pyramid (M1's
microbench: L3 vs L5 differ by 0.004 ms) but in the LK solve, which iterates at
every level.

This needed a small fix to a guard I wrote earlier: `reuse_left` demanded
`stereo_max_level_ == max_level_`, so shortening only the stereo search made the
function build a *second, shallower* left pyramid and throw the temporal one
away. It is now `stereo_max_level_ <= max_level_`, which is correct because
`calcOpticalFlowPyrLK` clamps its `maxLevel` down to the shallower of the two
pyramids it is handed (`lkpyramid.cpp`, the `levels1 < maxLevel` / `levels2 <
maxLevel` tests) — so a shallow *right* pyramid plus an explicit
`stereo_max_level_` is what decides how many levels get searched, and the deep
left pyramid is free to reuse.

**(b) `back_track=false`.** Drops the right→left solve, i.e. one of the two LK
calls. The circular check is the best filter XIVO has against repeated texture,
so dropping it needs a replacement, and the epipolar gate is already sitting
right there: an aliased match is almost never epipolar-consistent to
`stereo_epipolar_thresh` in *normalized bearing* coordinates. The rejection
counts show the substitution happening cleanly — epipolar rejections go 12576 →
51617 while circular go 34942 → 7476, and the final matched fraction only moves
88.3% → 85.8%. The epipolar gate absorbs almost exactly the work back-tracking
used to do, for the price of two `UnProject` calls instead of a KLT solve.

**(c) `seed_prev_disparity=true` + `max_disparity_jump=6.0`.** Seeds the
left→right search with the disparity that worked for the same feature ID last
frame (`OPTFLOW_USE_INITIAL_FLOW`), because disparity is smooth in depth and
depth is smooth in time. This is what makes (a) safe: the search starts within a
pixel or two of the answer instead of at zero disparity. The table is rebuilt
from scratch every frame from that frame's accepted matches, so a seed can only
ever be one frame old and cannot outlive the feature it describes.

`max_disparity_jump` is the accuracy half of the trade: a seeded search can be
*captured* by the seed and converge to a stale location, so a match whose
disparity moved more than 6 px from last frame's is thrown out. It only has
meaning when the table exists, so the constructor warns and disables it if
`seed_prev_disparity` is off rather than silently reading an empty table. Its
rejections are counted in `num_stereo_rejected_circular_` on purpose: the two
gates do the same job (reject a left-right correspondence that is not
self-consistent), the shipped configs enable exactly one of them, and sharing the
counter keeps the printed diagnostic comparable across the two.

## Paired attribution, room1 stereo, same window

| arm | FPS | `track` ms |
|---|---|---|
| base | 38.00 | 9.96 |
| (a) `max_level=2` alone | 41.24 | 8.35 |
| (b)+(c) `back_track=false` + seed + jump gate | 41.75 | 7.53 |
| (a)+(b)+(c) | **43.65** | **6.54** |
| + `KLT.max_level=4` | 44.45 | 6.19 |

The two halves are close to additive (−1.61 and −2.43 alone, −3.42 together),
which is what you expect: (a) makes each solve cheaper, (b) removes one of the
two solves, and the overlap is only the part of (a) that applied to the solve (b)
deleted.

`KLT.max_level=4` shortens the *temporal* pyramid too, and is the only one of the
four that touches mono: paired mono 74.43 → 76.47, and with the decode 81.16 →
83.81 (+3.3%). It is also the only one I am not sure about on accuracy grounds,
since mono has only 0.0055 m of ATE headroom — hence the separate ensemble arm.

## Combined with M2, paired room1

| arm | mono FPS | ratio | stereo FPS | ratio |
|---|---|---|---|---|
| base | 74.28 | — | 38.05 | — |
| + `fast_png_decode` | 81.16 | 1.093 | 41.58 | 1.093 |
| + stereo (a)(b)(c) | — | — | 48.36 | **1.271** |
| + `KLT.max_level=4` | 83.81 | 1.128 | 49.13 | **1.291** |

## What I did not do

- **Half-resolution stereo search.** 0.25 ms of CLAHE and a shallower solve, but
  it changes the matched pixel coordinates, so it is an accuracy change, not an
  efficiency one, and M1's rule says those lose.
- **Fewer KLT iterations** (`max_iter` 30 → 15). Only 0.13 ms on the microbench
  (2.365 → 2.235) and it degrades the converged location of every point, mono
  included. Bad ratio.
- **Skipping the stereo match on some frames.** Cheap and effective, but a
  feature that misses its right observation on frame *k* is a different feature
  to the filter, so this is squarely in the other agent's territory.
