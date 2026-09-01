# Config delta — position agent

Every key this agent changed in `cfg/eff_mono.json` **and** `cfg/eff_stereo.json`.
The two files are identical apart from `stereo` / `stereo_update.enable` /
`stereo_matcher.enable` and the header comment, and **every change below was
applied identically to both**. Nothing else in either file was touched.

Seven keys. Four of them are new blocks/keys that did not exist in the file; three
are existing values.

| key | old | new | why |
|---|---|---|---|
| `use_OOS` | `false` | `true` | Turns on the out-of-state (MSCKF) update, which uses the observations of tracks the EKF never promoted. Worth -0.0053 mono ATE by itself, and it is the precondition for `consistent_init` being positive at all. |
| `oos_meas_std` | `3.5` | `1.0` | The shipped 3.5 px was inherited from the depth sub-filter, not measured. An out-of-state row is an ordinary undistorted-pixel measurement; 1.0 is between the in-state `visual_meas_std` of 0.75 and the old value. -0.0019 alone, part of the stack. |
| `OOS` | *(absent)* | `{ "pose_window": 20, "min_observations": 2, "max_observations": 15 }` | See below. Only these three differ from the code defaults, so the block is deliberately three lines. |
| `tracker_cfg.grayscale` | *(absent, = false)* | `true` | `pybind11/pyxivo.cpp:120` reads frames with `cv::imread` (IMREAD_COLOR), so the front end ran on 3-channel BGR. `cv::cornerSubPix` and `cv::createCLAHE` both require one channel. Measured inert on its own (+0.0002) — this is a correctness prerequisite, not a tuning knob. |
| `tracker_cfg.histogram_method` | *(absent, = `NONE`)* | `"CLAHE"` | TUM-VI frames measure mean intensity 62.9 at r < 80 px and 33.2 at r > 270 px (ratio 0.53) while FAST uses one global threshold, so detection was starved in the periphery where the parallax is. **-0.0158 mono ATE, the single largest config effect.** Clip limit 10.0 / 8x8 grid are the code defaults, copied from OpenVINS `TrackKLT.cpp:61-63`; `clip_limit=4` measured worse (0.0623 vs 0.0605), so the default is left alone. |
| `tracker_cfg.subpix_refine` | *(absent, = false)* | `true` | `cv::cornerSubPix` on new detections. -0.0062 mono ATE. |
| `consistent_init` | *(absent)* | `{ "enable": true }` | Gives a promoted feature a covariance and cross-covariance consistent with the poses it was triangulated from, instead of the depth sub-filter's conditional 3x3 with all cross terms zeroed. -0.0042 mono ATE / -0.0055 RPE on the full stack, -0.0138 / -0.0101 on the OOS base. `min_views`, `meas_std` and `max_var` are left at their code defaults (2, `visual_meas_std`, 1e4) — all three were swept and none beat the default. |

The `OOS` sub-keys:

* `pose_window: 20` (code default `0`). **This is the load-bearing one.** With 0
  the OOS path is inert: no sliding window of past poses is kept, so a dropped
  track has nothing to constrain and `use_OOS: true` alone does nothing. 20 frames
  = 1 s at 20 Hz. `30` measured no better (0.0602 vs 0.0605).
* `min_observations: 2` (default: falls back to `OOS_update_min_observations`, 5).
  A 2-view track still contributes one marginalized row after the nullspace
  projection. -0.0035.
* `max_observations: 15` (default `kMaxGroup` = 45). Caps the rows one track can
  contribute. `30` measured worse (0.0634 vs 0.0605).

## Keys deliberately NOT changed

Recorded so the merge does not "helpfully" add them:

* `MH_thresh` stays `5.991`, `MH_adjust_factor` stays `1.15`, and the new
  `MH_max_strikes` stays at its default of 1. Loosening the gate is neutral at
  best (`m5`) and negative without `consistent_init`.
* `fej.mode` / `fej.oos` stay at their defaults of 0 / false. FEJ is implemented
  and correct but measured inside the noise floor on room1-6 (`m3`).
* `tracker_cfg.epipolar_rejection.enable` stays false: +0.0030 ATE once CLAHE is
  in — the two remove the same bad correspondences (`m4`).
* `tracker_cfg.max_theta_deg` stays absent (mask off): negative (`m2`).
* `subfilter.ready_steps` stays `2`. `1` is -0.0045 on the OOS base but +0.0005 on
  the final stack; it trades 0.0005 ATE for 0.0011 RPE, i.e. it is a coin flip.
* `tracker_cfg.num_features_min/max` stay `135`/`180`. Raising to 180/240 was
  +0.0047 — the filter is not short of tracked features, it is short of *good*
  ones (`m4`).
* `num_gauge_xy_features` stays `3`; `0` was inert.
* `visual_meas_std` (0.75), `subfilter.*`, `triangulation.*`, `max_depth`,
  `min_depth`, `stereo_update.*`, `feature_owner_change_cov_factor`: untouched.

## Merge notes

1. **The four "new" keys are all no-ops when absent.** Every code path added on
   this branch reads its config with a default that reproduces the old behaviour
   (`fej.mode` 0, `consistent_init.enable` false, `MH_max_strikes` 1,
   `tracker_cfg.grayscale`/`subpix_refine`/`epipolar_rejection` false,
   `histogram_method` NONE, `OOS.pose_window` 0). So merging the *code* without
   the config is safe and bit-identical; the numbers come from the config.
2. **`use_OOS` + `OOS.pose_window` + `consistent_init.enable` must move
   together.** `consistent_init` alone on a config without the OOS window is
   *+0.0057* mono ATE (`m5`). This is the one interaction in the delta.
3. Both files got the same edits. If the orientation agent also edits
   `cfg/eff_*.json`, its keys and mine do not overlap: it owns `Qimu`, `gravity`,
   `P.*`, `td`, and the initialization block; I own `OOS*`, `tracker_cfg.*`,
   `consistent_init`, `fej`, `MH_*`.
