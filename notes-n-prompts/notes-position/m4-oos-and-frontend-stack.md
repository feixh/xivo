# M4 — the stack that actually moved mono position: OOS supply + front end

Mono, 6 rooms, `--jitter 6`, `ate_002` / `ov_rpe8_pos_m`. Two reference points:

* **plain baseline** (`position_nochange`) 0.0928 / 0.0480
* **`oos_full`** = `use_OOS=true` + the tuned `OOS` block (see `m1`) 0.0875 / 0.0418

## One knob at a time, on the `oos_full` base

| arm | key | ate_002 | rpe8 | delta ATE |
|---|---|---|---|---|
| base | — | 0.0875 | 0.0418 | — |
| `oos_clahe` | `histogram_method=CLAHE` | **0.0717** | **0.0354** | **-0.0158** |
| `oos_eqhist` | `histogram_method=HISTOGRAM` | 0.0799 | 0.0410 | -0.0076 |
| `oos_subpix` | `subpix_refine=true` | 0.0813 | 0.0381 | -0.0062 |
| `oos_epi` | `epipolar_rejection.enable=true` | 0.0814 | 0.0409 | -0.0061 |
| `oos_mo2` | `OOS.min_observations=2` | 0.0840 | 0.0429 | -0.0035 |
| `oos_ready1` | `subfilter.ready_steps=1` | 0.0830 | 0.0404 | -0.0045 |
| `oos_r1` | `oos_meas_std=1.0` | 0.0856 | 0.0421 | -0.0019 |
| `oos_ae1` | `OOS.augment_every=1` | 0.0863 | 0.0393 | -0.0012 |
| `oos_gauge0` | `num_gauge_xy_features=0` | 0.0872 | 0.0420 | -0.0003 |
| `oos_gray` | `grayscale=true` alone | 0.0877 | 0.0410 | +0.0002 |

Only `oos_clahe` clears the 0.005 m noise floor on its own. The rest are
individually inconclusive -- which is exactly why they were then stacked and the
stack measured, rather than trusted one at a time.

**`oos_gray` is the control that makes the rest interpretable.** Converting the
frame to luminance changes nothing by itself (+0.0002). So the gains from
`subpix_refine` and the two histogram methods are the algorithms, not the
side-effect of finally running the front end on one channel instead of three.

## The stack

| arm | keys added to `oos_full` | ate_002 | rpe8 |
|---|---|---|---|
| `combo` | `augment_every=1`, `min_observations=2`, `oos_meas_std=1.0`, `grayscale`, `CLAHE` | 0.0643 | 0.0334 |
| `combo_epi` | `combo` + `epipolar_rejection` | 0.0673 | 0.0323 |
| `combo_subpix` | `combo` + `subpix_refine` | **0.0605** | **0.0326** |

Per sequence, `combo_subpix`: 0.0541 / 0.0658 / 0.0623 / 0.0428 / 0.0855 / 0.0524.

The knobs are close to additive: 0.0875 -> 0.0643 -> 0.0605 recovers most of the
sum of the individual deltas. `combo_epi` is +0.0030 on ATE and -0.0011 on RPE
against `combo`, i.e. no effect either way once CLAHE is already in -- the
epipolar test and the contrast normalization are removing the same bad
correspondences, so it was dropped from the stack.

## Why CLAHE is worth this much, and why it is not a tuning result

Measured on 40 frames spread over room1's cam0, after the 16-bit -> 8-bit
conversion `cv::imread` performs: **mean intensity 62.9 inside r < 80 px of the
principal point, 33.2 outside r > 270 px, a ratio of 0.53.** So the frames are
dark to begin with (a quarter of the 8-bit range) and the periphery is another
stop down on top of that -- the fisheye vignette plus a mocap room lit from above.

FAST uses a single global intensity threshold (`tracker_cfg.detector.threshold:
20`). A corner's FAST score scales with local contrast, which scales with
intensity, so a threshold that is reasonable at the centre is roughly twice as
strict at the edge of the image -- and the edge of the image is where the parallax
is. CLAHE equalizes in 8x8 tiles, which brings every tile to a comparable
effective threshold. That is the mechanism; it is a detection-supply fix, which is
why it shows up as a large ATE gain while raising `num_features_max` does not.

The parameters are OpenVINS's, not tuned here: `clip_limit = 10.0`,
`grid = 8x8`, copied from `ov_core/src/track/TrackKLT.cpp:61-63`. Note that
OpenVINS's *own* TUM-VI config selects `HISTOGRAM`, not `CLAHE`; on XIVO, CLAHE
beats plain equalization by 0.008 m, so this is not simply "use their setting".

## Corrections to earlier readings

* **`num_features_min/max` is not the supply limit.** `t240` (180/240) was
  +0.0047 on the plain baseline. The census says 76-85 of 90 feature slots are
  occupied; the filter is not short of tracked features, it is short of features
  whose measurements are *good*. Contrast normalization adds good ones in the
  places detection was failing; raising the cap adds weak ones everywhere.
* **`max_theta_deg` (radial FOV mask) is negative** and stays off. See `m2`.
* **`MH_max_strikes > 1` is negative** and stays at 1. See `m2`.
