# Config delta — branch `auto-frontfast`

Every key below is read with a default equal to today's behaviour, so **merging
this branch's code without its config changes is bit-identical to `auto`**. The
four keys in the first table are the ones actually turned on in the shipped
configs; the four in the second table exist, are tested, and stay off.

## Turned on

| key | where | old | new | why |
|---|---|---|---|---|
| `fast_png_decode` | top level | `false` | `true` | The fast grayscale-PNG path. 2.81 → 1.42 ms/image, **bit-identical output** (verified by `cmp` on `XIVO_DUMP_PRECISE=1` trajectories, mono and stereo, and by `unitTests_pngfast` against `cv::imdecode`). +9.3% FPS in both modes for zero accuracy cost. `m2-png-decode.md`. |
| `tracker_cfg.stereo_matching.max_level` | stereo only | (`KLT.max_level`, 5) | `2` | The left→right disparity is bounded by the baseline and the scene depth, so the stereo search does not need the temporal search's coarse levels. −1.6 ms of `track`. `m3-stereo-matching.md`. |
| `tracker_cfg.stereo_matching.back_track` | stereo only | `true` | `false` | Drops the right→left LK solve, i.e. one of the two solves in the match. The epipolar gate on normalized bearings absorbs the rejections (epipolar 12576 → 51617, circular 34942 → 7476, matched 88.3% → 85.8%) at the price of two `UnProject` calls instead of a KLT solve. −2.4 ms of `track`. |
| `tracker_cfg.KLT.max_level` | both | `5` | `4` | The temporal search. Cheaper solve (microbench 2.365 → 2.041 ms on 180 points); the pyramid itself is free either way. +3.5% mono, +2% stereo. The only knob here that touches mono, hence the separate ensemble arm — mono ATE went 0.0566 → 0.0555, i.e. it does **not** spend mono's headroom. |

Applied to `cfg/eff_mono.json`, `cfg/eff_stereo.json` (the harness configs) and
the shipped `cfg/tumvi_mono_ctl.json`, `cfg/tumvi_mono_ctl_oos.json`,
`cfg/tumvi_stereo.json`, `cfg/tumvi_stereo_oos.json`. `stereo_matching.*` only
means anything in a stereo config, so the mono files get `fast_png_decode` and
`KLT.max_level` only. Every other config in `cfg/` is untouched and therefore
keeps today's behaviour exactly.

## In the tree, off, kept because the negative result is worth re-running

| key | default | what it does | why it is off |
|---|---|---|---|
| `tracker_cfg.histogram_method: "GAINMAP"` + `tracker_cfg.gainmap.*` | not selected (`CLAHE`) | one fixed-point radial gain multiply, 0.154 ms, instead of CLAHE's 0.784 | **Slower end to end**: 75.71 vs 80.55 FPS paired. `track` falls 0.37 ms but `process-tracks` rises 1.05 ms because the run initialises 21874 features instead of 17743. `m1`, negative result 1. |
| `tracker_cfg.equalize_for: "DETECT"` | `"ALL"` | equalize only the image FAST runs on; track/match on the raw image | **Slower end to end and +60 MB**: mono 73.29 vs 75.29, stereo 38.53 vs 38.72, stereo peak RSS 136 → 196 MB. `m1`, negative result 3. |
| `tracker_cfg.stereo_matching.seed_prev_disparity` | `false` | seeds the left→right search with last frame's disparity for the same feature id (`OPTFLOW_USE_INITIAL_FLOW`) | Works, but a wash: paired stereo 48.30 with vs 48.49 without. It moves 0.28 ms out of `track` and puts 0.36 ms back into `process-tracks` (it keeps more matches alive), costs 5 MB of peak RSS for the id→disparity table, and its ensemble is no better (ATE 0.0491 vs 0.0488). Two fewer keys for the same speed. |
| `tracker_cfg.stereo_matching.max_disparity_jump` | `inf` | rejects a match whose disparity moved more than N px since last frame | Only meaningful with `seed_prev_disparity` (it is the guard against a seeded search being captured by a stale seed), and warns + disables itself without it. Off with it. |

## Keys deliberately not changed

- `tracker_cfg.subpix_refine`: stays `true`. It is 0.06 ms/frame and worth
  −0.0062 m of ATE. Nothing to win in `cornerSubPix` window size or iteration
  count either — 45 points at win 5 / 20 iters is 0.286 ms and the tracker
  refines far fewer than 45 per frame.
- `tracker_cfg.histogram_method`: stays `CLAHE`. `NONE` is 1.22 ms/frame cheaper
  but buys that with accuracy (10x fewer keypoints), and `HISTOGRAM`
  (`equalizeHist`) is 0.97 ms cheaper for the same reason on a smaller scale.
  These are accuracy trades, not efficiency wins.
- `tracker_cfg.KLT.max_iter`: stays `30`. 30 → 15 is 0.13 ms and degrades every
  converged point.
- `tracker_cfg.stereo_matching.epipolar_thresh`: stays `0.005`. With
  `back_track=false` this gate now carries the load, so loosening it would be
  the wrong direction; tightening it costs matches. Left alone deliberately.
