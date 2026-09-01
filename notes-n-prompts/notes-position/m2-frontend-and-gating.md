# M2 — front-end quality and visual gating: what was tried and what it cost

All numbers are mono, 6 rooms, `--jitter 6`, `ate_002` / `ov_rpe8_pos_m`.
Baseline for the first table is `position_nochange` = 0.0928 / 0.0480.

## Code written (all default-off, all bit-identical at the defaults)

Verified: `sweep.sh nochange` with no keys reproduces 0.0762 / 0.1001 / 0.1343 /
0.0805 / 0.1065 / 0.0594 = 0.0928 and RPE 0.0480 exactly, i.e. every edit below
is a no-op unless its config key is set.

| key (under `tracker_cfg` unless noted) | what it does |
|---|---|
| `max_theta_deg` | radial field-of-view mask, folded into `valid_mask_` |
| `grayscale` | BGR -> luminance once, at the head of the tracker |
| `subpix_refine`, `subpix.{win_size,max_iter,eps}` | `cv::cornerSubPix` on each new detection |
| `histogram_method` (NONE/HISTOGRAM/CLAHE), `clahe_clip_limit`, `clahe_grid_size` | contrast normalization |
| `epipolar_rejection.{enable,thresh_px,confidence,min_points,max_bearing_norm}` | fundamental-matrix RANSAC on normalized bearings |
| `MH_max_strikes` (top level) | how many consecutive MH-gate failures destroy an in-state feature |

### The three bugs/limits these exposed

**1. XIVO tracks on 3-channel BGR.** `pybind11/pyxivo.cpp:120` reads frames with
`cv::imread(image_path)`, whose default flag is `IMREAD_COLOR`. So `Tracker::img_`
is CV_8UC3 for a monochrome dataset. It went unnoticed because both consumers
tolerate it — `FastFeatureDetector::detect` runs its own `cvtColor` internally and
throws the result away every frame, and `calcOpticalFlowPyrLK` tracks
multi-channel patches. It is not free and it is not the same problem as tracking
on luminance. It is also why the first attempt at `subpix_refine` and
`histogram_method=HISTOGRAM` aborted on frame 1 of every run:

```
cornersubpix.cpp:66: error: (-215:Assertion failed) src.channels() == 1
```

`Tracker::ToGray` fixes it; the constructor force-enables `grayscale` whenever
`subpix_refine` or `histogram_method` is set, so those configs cannot abort.

**2. `EquidistantCamera::UnProject` saturates instead of failing.**
`common/camera_equidist.h:155` clamps `th` to `kMaxTh = 1.5706`. On the TUM-VI
512x512 intrinsics theta = 90 deg is r = 296.9 px from the principal point while
the image corners are at r ~ 362 px, so **7.07%** of the image unprojects to a
bearing of norm ~6366, and **11.13%** is beyond 85 deg where tan(theta) > 11.
XIVO's detection mask is a rectangular `margin: 8` border, so all of it is
admitted. `max_theta_deg` adds the radial test (done through `UnProject` itself,
so it holds for any camera model, including ones that saturate).

**3. `Tracker::OutlierRejection` fits a homography to raw distorted pixels.**
Valid only for a planar scene or a pure rotation, and its residual is not a
metric distance under a fisheye. It is off by default in the shipped configs,
which is the right call. `OutlierRejectionEpipolar` is the OpenVINS approach
copied in (`ov_core/src/track/TrackKLT.cpp:873`): unproject both point sets to
normalized bearings, `cv::findFundamentalMat(FM_RANSAC, 2.0/f, 0.999)`, drop the
outliers. Two deviations from OpenVINS, both because XIVO's unprojection
saturates: bearings longer than `max_bearing_norm` (default 10, ~84 deg) are
excluded from the fit rather than rejected, and a frame with fewer than
`min_points` (10) usable correspondences is left alone.

## Batch 3 — measured on the plain baseline (0.0928 / 0.0480)

| arm | keys | ate_002 | rpe8 | verdict |
|---|---|---|---|---|
| `oos_full` | `use_OOS` + `OOS` block | **0.0875** | **0.0418** | keep, see m1 |
| `ready1` | `subfilter.ready_steps=1` | 0.0899 | 0.0451 | promising |
| `gauge0` | `num_gauge_xy_features=0` | 0.0914 | 0.0450 | ATE flat, RPE better |
| `loosecrit` | `strict_criteria_timesteps=100000` | 0.0928 | 0.0480 | **inert** (bit-identical) |
| `fov85` | `tracker_cfg.max_theta_deg=85` | 0.0952 | 0.0485 | negative |
| `t240` | `num_features_min/max = 180/240` | 0.0975 | 0.0499 | negative |
| `strikes3` | `MH_max_strikes=3` | 0.1031 | 0.0492 | negative |
| `subpix` | `subpix_refine=true` | crashed | — | rerun after `ToGray` |
| `eqhist` | `histogram_method=HISTOGRAM` | crashed | — | rerun after `ToGray` |

Notes on the negatives, since they are informative:

* **`fov85` is negative, so the saturating-unprojection finding is real but not
  the gap.** Masking the outer 11% of the image removes bearings that were
  numerically garbage, and the result is slightly *worse*. The reading is that
  those tracks were already being killed downstream (they fail the subfilter or
  the MH gate) and what the mask actually costs is the 11% of good peripheral
  image area that carries the most parallax. A saturating `UnProject` is still a
  latent bug for any code that trusts its output.
* **`t240` is negative**, which agrees with the census: mono occupies 76/90
  feature slots, so the filter is not short of *tracked* features, and adding 60
  more only adds weaker ones. The supply limit is on *candidates that survive
  long enough*, which is a different quantity.
* **`strikes3` is clearly negative.** Letting a feature that fails the 95% MH
  gate stay in the state for two more frames is worse than destroying it, even
  though it keeps a state slot occupied. The one-strike rule is right.
* **`loosecrit` is bit-identical to the baseline.** `strict_criteria_timesteps`
  is inert in this configuration; do not spend another arm on it.
* `MH_max_strikes=1` (the default) reproduces the original destroy-on-first-
  failure behaviour exactly, so the strike machinery stays in the tree at no cost.
