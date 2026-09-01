# M2 -- EuRoC MAV support in XIVO

Milestone M2 of `notes-n-prompts/plan-euroc.md`. Branch `auto-euroc`, worktree
`xivo-euroc`.

Goal: make XIVO able to *run* EuRoC at all, with one config shared by all 11
sequences, and with every calibration number traceable to the dataset rather
than transcribed. Tuning is M4; this milestone stops at "runs, and does not
diverge".

Most of the work turned out not to be plumbing. The plumbing took an afternoon
and the loader diff is 20 lines. What took the time was that with a correct
loader and a correct calibration, XIVO diverged on every EuRoC sequence by four
orders of magnitude, and the cause was one number in the config. That
investigation is the bulk of this note, because the conclusion generalises: the
TUM-VI config that M0--M7 of the previous round tuned so carefully encodes an
assumption about the *IMU hardware*, and EuRoC uses different hardware.


## 1. What was added

### `scripts/pyxivo.py` -- a `euroc` dataset branch

EuRoC's ASL folder sits directly under the sequence name, with no
`dataset-<seq>_512_16` wrapper:

    data/euroc/MH_01_easy/mav0/{cam0,cam1}/data/<ts>.png
    data/euroc/MH_01_easy/mav0/imu0/data.csv

Otherwise it is TUM-VI's layout exactly, including the useful part: cam0 and
cam1 are hardware-triggered and both files are named after the *shared*
timestamp, so the existing stereo pairing (build a `{ts: right_path}` map and
check membership) works unchanged and still reports dropped frames rather than
crashing on a partial download.

### `scripts/savers.py` -- groundtruth path per dataset, and an explicit slice

`TUMVISaver` hardcoded `mav0/mocap0/data.csv`. EuRoC's groundtruth is
`mav0/state_groundtruth_estimate0/data.csv`, so `EuRoCSaver` overrides the path
and the four mode savers (`Eval`, `Dump`, `CovDump`, `TrackerDump`) follow the
existing pattern.

One latent bug fixed on the way. The quaternion was read as `v[4:]`:

```python
q = [float(x) for x in v[4:]]   # before
q = [float(x) for x in v[4:8]]  # after
```

TUM-VI's `mocap0` rows are exactly 8 columns, so the open-ended slice happened
to give 4 elements. EuRoC's `state_groundtruth_estimate0` rows are **17**
columns -- position, quaternion, velocity, gyro bias, accel bias -- so the same
slice yields 13. The old code would still have produced the right rotation,
because only `q[0..3]` is read afterwards, but it is right by accident, and the
accident does not survive anyone later passing `q` to a normaliser or a length
check. Fixed to say what it means.

### `scripts/make_euroc_cfg.py` -- generate the config from `sensor.yaml`

EuRoC ships its calibration per sensor under
`<seq>/mav0/{cam0,cam1,imu0}/sensor.yaml`, and **those files are byte-identical
across all 11 sequences** (verified in M0 by `md5sum`). So one generated config
serving the whole dataset is not a compromise for the sake of the user's
"one configuration for all sequences" requirement -- it is what the dataset
itself says. The generator reads:

| config key | source |
| --- | --- |
| `camera_cfg`, `camera1_cfg` | `cam{0,1}/sensor.yaml` intrinsics + 4 radtan coefficients |
| `X.Wbc`, `X.Tbc` | cam0's `T_BS` (rotation, translation) |
| `stereo_cfg.T_c1c0` | `T_BS(cam1)^-1 T_BS(cam0)` |
| `Qimu` | imu0's four noise densities |

and takes the rest from flags, because it is scene geometry or tuning rather
than calibration: gravity magnitude, the depth window, the two bias priors, and
optional scale factors on the noise densities.

Four conventions were checked against source or against TUM-VI's own
calibration rather than assumed, since each has a plausible wrong reading that
fails silently:

* **`X.Wbc` is R_body_from_camera**, i.e. `X.Wbc`/`X.Tbc` together *are* EuRoC's
  `T_BS`. No inversion. (`estimator.cpp` reads a 3-vector as so(3) and falls
  back to a row-major 3x3; confirmed against the shipped TUM-VI config and
  TUM-VI's camchain.) Inverting here is the natural mistake and would put the
  camera 6.5 cm off in the wrong direction.
* **`stereo_cfg.T_c1c0` maps cam0 into cam1**, not the reverse.
* **`Qimu` holds standard deviations**, not variances -- `estimator.cpp` squares
  the whole block (`Qimu_ *= Qimu_`). So the dataset's continuous-time noise
  densities go in directly.
* **radtan maps term-for-term onto OpenCV's ordering.** XIVO's `RadTan` takes
  three radial coefficients in `k012` plus `p1`/`p2` separately; its expansion
  in `common/camera_radtan.h` is algebraically identical to OpenCV's
  (`p2*(3x^2+y^2) == p2*(r^2+2x^2)` and `p1*(x^2+3y^2) == p1*(r^2+2y^2)`), so
  Kalibr's `[k1,k2,p1,p2]` maps across with `k3 = 0`. This was checked
  algebraically rather than numerically because the venv has no `cv2`; the
  algebra is the stronger check anyway.

The generator also asserts that `imu0`'s own `T_BS` is identity. It is -- EuRoC
defines the body frame *as* imu0 -- which matters twice over: every extrinsic
above is relative to it, and it is also the frame `state_groundtruth_estimate0`
reports and the frame XIVO's `gsb` estimates, so scoring needs no re-framing
either. If a future EuRoC release moved the body frame, every number in the
generated config would be silently wrong by that transform, hence the assert
instead of a comment.

Generated: `cfg/euroc_stereo.json`, `cfg/euroc_mono.json`.


## 2. XIVO diverged on every sequence, by four orders of magnitude

With the loader and calibration in place, a straight port of the tuned TUM-VI
`eff_stereo.json` gave, on V1_01_easy:

    ATE = 22593 m

against a trajectory 58 m long. Not "inaccurate" -- no tracking at all.

### Ruling things out

Each of these was an A/B against the same build and sequence, and each is listed
because it was a plausible cause, not because it was a good guess:

| hypothesis | test | result |
| --- | --- | --- |
| `fast_png_decode` mis-decodes 752x480 mono8 | set `false` | identical failure |
| `tracker_cfg.use_prediction` unstable at 20 Hz | set `false` | worse: 4.7e6 |
| depth window alone (5 m is a TUM-VI room) | `max_depth = 30` | 21591 -- barely moved |
| stereo update is the problem | `stereo_update.enable = false` | 30969 |

Then the useful step: **establish a healthy control on the same build.** TUM-VI
room1 stereo, same binary, same session: ATE 0.037887, 84.6 of 90 feature slots
filled, 2799 of 2800 updates accepted. So the build, the stereo path, the PNG
decoder and the estimator were all fine, and the fault was in the EuRoC-specific
inputs -- data, calibration, or config.

Two more measurements narrowed it to config:

* **Onset.** Position error was 0.511 m at t = 2 s and 5.8 m at t = 4 s, while
  the groundtruth says the platform is *stationary* for the first few seconds.
  Error growing with a growing second derivative is a dead giveaway: a constant
  acceleration bias being integrated twice.
* **Initial attitude was correct.** `R_xivo @ a_body = [-0.004, 0.016, 9.810]`,
  i.e. gravity landed on +z to three decimals. So gravity alignment at init was
  right, and the attitude *drifted away* afterwards rather than starting wrong.

### The cause: EuRoC's IMU biases do not fit inside TUM-VI's priors

EuRoC's groundtruth reports the IMU biases it solved for, so this is not
inference. Across all 11 sequences:

| sequence | max \|b_w\| (rad/s) | max \|b_a\| (m/s^2) |
| --- | --- | --- |
| MH_01_easy | 0.0814 | 0.1595 |
| MH_02_easy | 0.0801 | 0.1615 |
| MH_03_medium | 0.0798 | 0.1406 |
| MH_04_difficult | 0.0795 | 0.1521 |
| MH_05_difficult | 0.0797 | 0.1417 |
| V1_01_easy | 0.0792 | **0.5563** |
| V1_02_medium | 0.0787 | 0.1411 |
| V1_03_difficult | 0.0797 | 0.2031 |
| V2_01_easy | 0.0854 | 0.1479 |
| V2_02_medium | 0.0830 | 0.1075 |
| V2_03_difficult | 0.0842 | 0.0954 |

The ported TUM-VI config's initial bias covariances are `P.bg = 1e-4` and
`P.ba = 1e-3` -- prior standard deviations of **0.010 rad/s** and
**0.032 m/s^2**. EuRoC's actual gyro bias is 0.079--0.085 rad/s, i.e. **8x the
prior sigma**, and its accel bias reaches 17x. XIVO starts both bias states at
zero, so the filter is told, with high confidence, that a bias it definitely has
is impossible.

The consequence chain, and it is all forced from there:

1. The gyro bias cannot be corrected, so attitude drifts at ~0.08 rad/s
   (4.6 deg/s).
2. A tilted attitude leaks gravity into the acceleration estimate -- at 4.6
   deg/s, after one second that is 0.8 m/s^2 of spurious acceleration, growing.
3. Position runs away quadratically from the first frame.
4. No feature ever survives long enough to be promoted to the state, because
   promotion requires the subfilter to converge and the predicted position to
   stay sane. The last `[census]` line showed **0.78 of 90 feature slots filled**.
5. So vision never gets the chance to fix the attitude that broke vision.

The `~0.08 rad/s` figure is worth pausing on: it is nearly identical on all 11
sequences, because it is the fixed turn-on bias of the ADIS16448 in the EuRoC
rig. It is a property of the hardware, not of any sequence -- which is exactly
why one shared config value is the right fix rather than a per-sequence patch.

### Confirmation

V1_01_easy, one knob at a time, on top of the ported TUM-VI config:

| config | ATE (m) | feature slots (of 90) | updates |
| --- | --- | --- | --- |
| ported TUM-VI as-is | 22593 | 0.78 | 546 |
| `P.ba = 0.25` only | 11528 | -- | -- |
| **`P.bg = 0.01` only** | **0.0712** | 86.3 | 2899 |
| `P.bg = 0.01`, `P.ba = 0.25` | 0.0688 | 86.3 | 2899 |
| ... `+ max_depth = 30` | 0.0635 | 86.2 | 2899 |

`P.bg` is the whole fix: one number, 22593 m -> 0.0712 m, and consistent
subfilter initialisations went from 835 to 11191. `P.ba` alone does nothing,
which fits the chain above -- the accel bias is not what is destroying the
attitude. `P.ba` is worth a further 3% once the gyro bias can move.

Chosen defaults: `P.bg = 0.01` (sigma 0.1 rad/s) and `P.ba = 0.25`
(sigma 0.5 m/s^2), each comfortably covering the observed range with margin.


## 3. The depth window is the second EuRoC-specific number

TUM-VI's `max_depth = 5.0` is a room. EuRoC's Machine Hall is tens of metres
deep. This is not a soft penalty: a feature whose depth falls outside
`[min_depth, max_depth]` is *refused* as an instate candidate
(`Criteria::Candidate`, `manager.cpp:800`) and as a stereo seed
(`manager.cpp:355`). Everything past 5 m is discarded outright.

With the bias fix applied, sweeping the shared window:

| `max_depth` (m) | MH_01_easy ATE | V1_01_easy ATE |
| --- | --- | --- |
| 5 | diverges (23937) | 0.0688 |
| 15 | 3.557 | -- |
| 30 | 3.674 | 0.0635 |
| **60** | **0.0993** | 0.0638 |
| 100 | 0.1014 | 0.0625 |
| 200 | 0.1453 | 0.0625 |

Machine Hall needs at least ~60 m or it does not survive; the Vicon room is flat
from 60 m up. 60 m is the shared default: best on the sequence that cares, and
within run-to-run noise on the sequence that does not. Note that 200 m is
*worse* on MH -- admitting arbitrarily distant features is not free, since a
badly-conditioned depth estimate still consumes a state slot.

The 60-vs-100 gap (0.0993 vs 0.1014, 0.0638 vs 0.0625) is smaller than XIVO's
own run-to-run spread of ~0.007 m, so these single runs cannot separate them.
Choosing between them properly needs the jitter ensemble, which is M4's job; for
M2 either is fine and 60 is taken as the default.

Also set: `initial_z = 5.0` (was 2.5) and `min_depth = 0.05`.
`adaptive_initial_depth` takes over within a few frames, so `initial_z` mostly
matters for the first features.


## 4. Status

* `ctest`: 22/22 passing in the worktree.
* Both configs generated and running on all 11 sequences without divergence.
* Accuracy is *not* tuned. On V1_01_easy XIVO is at ~0.063 m against OpenVINS'
  0.055 m, so it is in range but behind. That is M3 (baseline) and M4 (tuning).

## 5. Deliberately not done here

* No per-sequence configuration, at any point, for either system.
* No tuning of anything that is not needed to make the dataset run. The two
  bias priors and the depth window are changed because the shipped values make
  EuRoC structurally impossible, not because they scored better.
