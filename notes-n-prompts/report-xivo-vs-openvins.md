# XIVO against OpenVINS: orientation, position, efficiency

Six development efforts, each on an isolated `git worktree`, merged into `auto`.
Baseline is `auto` @ `9e3ec06` (tagged `pre-merge-auto`); the endpoint is `b565b25`.
A seventh was completed, measured and then deliberately not shipped; section 3d says
why.
Reference is OpenVINS v2.7 (`v2.7-20-g6948812`), ROS-free build, measured in
`notes-n-prompts/report-openvins-baseline.md`.

Dataset: TUM-VI `room1`-`room6`, 512x512 fisheye, mono+IMU and stereo+IMU.

**Status: tasks 1 and 2 are met on every accuracy metric in both modes -- 10 of 10
strictly better than OpenVINS. Task 3 is met for mono+IMU (1.08x throughput, 0.97x
peak RSS) and met on memory for stereo (0.99x); stereo throughput is 0.984x, i.e.
0.23 ms/frame short** -- see [Efficiency](#3-efficiency-met-on-mono-0984x-on-stereo-throughput).

The merge chain on `auto`: `235fb3f` (orientation) -> `6c4bb4d` (efficiency) ->
`c0e7f62` (position) -> `b0d7ec5` (OOS/update cost) -> `017c4a4` (front end) ->
`b565b25` (chunked covariance update).

---

## Headline

Accuracy, 6-member ensembles, 6-room means. Lower is better everywhere.
Bold = beats OpenVINS. `final` is `auto` @ `b565b25`,
`experiments/results/final_acc_final5`. Mono is identical to four decimals on all
five metrics across the last two merges; stereo moved by at most 0.0005.

| metric | pre-merge `auto` | 3-way merge | **final `auto`** | OpenVINS | margin |
|---|---|---|---|---|---|
| **mono** `ate_002` [m] | 0.0928 +- 0.0067 | 0.0566 +- 0.0015 | **0.0555 +- 0.0026** | 0.0621 | 11% |
| mono `ov_ate_pos_m` | 0.0968 | 0.0584 | **0.0575 +- 0.0028** | 0.0638 | 10% |
| mono `ov_ate_ori_deg` | 1.8243 | 0.9104 | **0.8788 +- 0.0303** | 1.5742 | 44% |
| mono `ov_rpe8_pos_m` | 0.0480 | 0.0263 | **0.0265 +- 0.0009** | 0.0308 | 14% |
| mono `ov_rpe8_ori_deg` | 0.5153 | 0.5138 | **0.5131 +- 0.0033** | 0.6445 | 20% |
| **stereo** `ate_002` [m] | 0.0636 +- 0.0045 | 0.0472 +- 0.0019 | **0.0490 +- 0.0022** | 0.0677 | 28% |
| stereo `ov_ate_pos_m` | 0.0688 | 0.0487 | **0.0507 +- 0.0022** | 0.0697 | 27% |
| stereo `ov_ate_ori_deg` | 1.7982 | 0.8844 | **0.8921 +- 0.0557** | 1.4440 | 38% |
| stereo `ov_rpe8_pos_m` | 0.0292 | 0.0208 | **0.0215 +- 0.0008** | 0.0265 | 19% |
| stereo `ov_rpe8_ori_deg` | 0.5074 | 0.5154 | **0.5161 +- 0.0080** | 0.5837 | 12% |

Throughput and memory, one core, `-mode runOnly`, all thread pools at 1, ASLR off,
serial, idle box. Higher FPS is better; lower RSS is better. **The last two rows are
means of three one-core passes each**, taken alternately in one session; the earlier
rows are single passes from earlier sessions.

| config | mono FPS | stereo FPS | mono peak RSS | stereo peak RSS |
|---|---|---|---|---|
| pre-merge `auto` | 101.8 | 45.0 | 134.1 MB | 139.3 MB |
| efficiency branch alone | 141.3 | 64.1 | 82.5 MB | 86.3 MB |
| 3-way merge (`c0e7f62`) | 83.1 | 41.1 | 101.6 MB | 161.8 MB |
| **final `auto` (`b565b25`)**, n=3 | **123.4** | 70.0 | **86.7 MB** | **94.6 MB** |
| OpenVINS, n=3 | 114.4 | 71.1 | 88.2 MB | 95.4 MB |
| final / OpenVINS | **1.079x** | 0.984x | **0.98x** | **0.99x** |

How reproducible each of those is, because two of the four margins are small enough
that it matters: ms/frame has a pass-to-pass sd of 0.002 (XIVO, both modes) and
0.012-0.021 (OpenVINS), so the throughput ratios are solid to better than 0.1%. Peak
RSS repeats to 0.1 MB on XIVO stereo and 0.2 MB on OpenVINS -- but XIVO's *mono* peak
RSS drifts **+-1.5 MB between sessions on a byte-identical binary** (85.4 then 86.7 MB
for this same commit), so read the mono memory margin as 1.5 +- 1.5 MB and the stereo
one as 0.8 +- 0.2 MB. Both are wins; the mono one is not a precise number.

Three things are worth stating plainly. **The final tree is faster than the pre-merge
baseline in both modes (1.21x mono, 1.56x stereo) while being 40% more accurate in
position and 51% more accurate in orientation** -- the accuracy work was not paid for
with throughput in the end. **Peak RSS beats OpenVINS in both modes**, narrowly. And
the ensemble sd of the 6-room mono mean fell from 0.0067 m to 0.0026 m: the pre-merge
filter's run-to-run scatter under a physically null 1e-6 m/s perturbation of the
initial velocity was 12% of its own error; the final filter's is 4.7%.

**Thirteen of the fourteen reported metrics clear the OpenVINS floor.** The one that
does not is stereo throughput, and it misses by 1.6% -- 14.285 ms/frame against the
14.057 that OpenVINS takes, i.e. **0.23 ms/frame**. Per sequence, XIVO/OpenVINS:
70.4/71.2, 69.2/73.4, 69.6/71.5, 69.5/70.4, 70.2/70.5, 71.2/69.9. It is a uniform
1-2% deficit, not one bad sequence, and XIVO already wins room6. Mono, by contrast,
wins all six: 123.3/113.1, 122.1/117.7, 122.1/116.5, 122.6/112.4, 122.8/112.3,
127.4/114.5.

---

## Measurement protocol

Both systems are near-deterministic, so error bars cannot come from a random seed
(at HEAD, mono XIVO is bit-identical across six seeds). They come from a
**physically null perturbation**:

* XIVO: `X.Vsb += k * 1e-6 m/s`, six orders of magnitude inside the filter's own
  0.7 m/s prior -- `run_xivo_reference.sh --jitter 6`.
* OpenVINS: `--gravity_mag` perturbed in the ninth significant digit.

Reported sd is the sd of the **6-room mean** across ensemble members. A single-run
comparison is not interpretable here: per-sequence single-run ATE moves by ~0.007 m
under that null perturbation, from chaotic MH-gating decisions.

Metrics:

* `ate_002` -- `evaluate_ate.py`, Horn SE(3) alignment, 0.02 s association window.
  **This is blind to a global rotation**, which is why it is quoted alongside the
  `ov_eval` numbers rather than alone.
* `ov_ate_*`, `ov_rpe8_*` -- `ov_eval error_singlerun posyaw`: position and *yaw*
  aligned only, because yaw and position are a VIO's unobservable directions while
  roll and pitch are observable. A roll/pitch frame offset therefore lands in the
  reported orientation error undiminished. RPE over 8 m segments.
* Throughput: `fps_wall = frames_processed / wall_total_s`, one core via `taskset`,
  `setarch -R`, `OMP_NUM_THREADS=OPENCV_FOR_THREADS_NUM=OPENBLAS_NUM_THREADS=1`.
  `nproc` honours `OMP_NUM_THREADS`, so cpu count must be read *before* the caps
  are exported.
* Timing deltas below ~0.05 ms/frame need **alternated** A/B passes in one session.
  Within a session an arm repeats to sd 0.002 ms/frame, but between sessions the same
  byte-identical binary moves by ~0.04 ms -- enough to invert the sign of a small
  effect, which it did once here (3d). Peak RSS has the same structure, with the extra
  wrinkle that XIVO mono `ru_maxrss` drifts +-1.5 MB across sessions; see Reproducing.

---

## 1. Orientation: 1.82 -> 0.88 deg mono, 1.80 -> 0.89 deg stereo (OpenVINS 1.57 / 1.44)

Worktree `xivo-orient`, branch `auto-orient`, merged as `235fb3f`. Notes:
`notes-n-prompts/notes-orient/`.

### The finding: it was a frame-convention bug, not attitude drift

XIVO's error state carries `Wsg`, a 2-DoF gravity direction: gravity in the spatial
frame `S` is `Rsg * g_`, so `Rsg` maps the gravity-aligned frame `W` into `S`. `S`
is *the body frame of the first IMU sample* (`X_.Rsb` starts at identity by
construction), so `S` is tilted by whatever the rig's attitude relative to gravity
happened to be at startup. **Nothing ever applied `Rsg` to the published pose.**
`Estimator::gsb()` returned `(Rsb, Tsb)` in `S`, and that is what the eval harness
read.

Per sequence, the startup tilt of `S` from level against the mocap, next to the
reported orientation ATE:

| seq | startup tilt of `S` [deg] | pre-merge mono ori ATE [deg] |
|---|---|---|
| room1 | 1.19 | 1.507 |
| room2 | **3.01** | **3.230** |
| room3 | 1.45 | 2.049 |
| room4 | 1.27 | 1.442 |
| room5 | 0.83 | 1.136 |
| room6 | 1.93 | 1.582 |

room2 -- the sequence where the gap lived -- reported almost exactly its own startup
tilt.

OpenVINS has no such term: its global frame is gravity-aligned by construction
(`Propagator.h:57`, `_gravity << 0, 0, gravity_mag`) and its static initializer sets
`R_GtoI` from the measured gravity direction (`ov_init/src/static/StaticInitializer.cpp:121-125`).
The benchmark had been comparing XIVO's tilted frame against OpenVINS' level one.

### M1 -- publish in the gravity-aligned frame

New output-only accessor `Estimator::gwb()` / `gwc()` in `src/estimator.h`:

```cpp
Rws = X_.Rsg.inverse();
return SE3{Rws * X_.Rsb, Rws * X_.Tsb};
```

Rotation *and* translation -- anything else is not a pose in any frame. Gated on
`gravity_align_output`, default `true`; set it `false` to recover the old convention
bit-for-bit. Wired into the pybind `gsb`/`gsc` bindings (what the harness reads) and
`src/app/vio.cpp`'s dump. `Estimator::gsb()` keeps its old meaning and is still what
`manager.cpp`, `update.cpp` and `estimator.cpp` read on the estimation path, so the
**filter is untouched and the estimate is bit-identical** -- confirmed by `ate_002`
holding at 0.0928 / 0.0636 to four decimals across the change.

`Rsg` is a state, so this is the filter's *current* estimate, causally available, no
post-hoc pass. It converges well: final `Rsg` against mocap gravity is 0.050 deg on
room2 and 0.399 deg on room4, from a 20-sample accelerometer average that starts
1.3 deg and 2.6 deg off.

### M2 -- fix the 4-DoF gauge to be about gravity, not the group's body z-axis

`Estimator`'s gauge fix zeroed rows and columns of `P_` corresponding to the gauge
group's rotation, using the group's own body z-axis as the fixed direction. The
unobservable rotational direction of a VIO is the **gravity** direction, not a body
axis. The correct fix projects out the gravity direction expressed in that group's
body frame (`State::operator+=` applies group updates on the right, `Rsb *= SO3::exp(dW)`,
so `dW` lives in the group's body frame):

```cpp
const Vec3 n_s = X_.Rsg * Vec3{0, 0, 1};
const Vec3 u   = (g->Rsb().inverse() * n_s).normalized();
const Mat3 Pi  = Mat3::Identity() - u * u.transpose();
P_.block(offset, 0, 3, N) = (Pi * P_.block(offset, 0, 3, N)).eval();
P_.block(0, offset, N, 3) = (P_.block(0, offset, N, 3) * Pi).eval();
P_.block(offset + 3, 0, 3, N).setZero();
P_.block(0, offset + 3, N, 3).setZero();
```

### Independent verification that this is not metric gaming

The concern with an output-frame change is that it could launder yaw error into the
alignment. It cannot, and this was checked directly rather than argued. For each
associated pose pair the world-side residual rotation `R_off = Rg * Re'` was split
into its **tilt** (the angle by which it moves the vertical -- the part a yaw-only
alignment cannot remove) and its **yaw**:

| mono, per sequence | pre-merge tilt | merged tilt | OpenVINS tilt |
|---|---|---|---|
| room1 | 0.891 (sd 0.195) | **0.261** | 0.275 |
| room2 | 3.103 (sd 0.293) | **0.324** | 0.322 |
| room3 | 1.476 (sd 0.310) | **0.336** | 0.374 |
| room4 | 1.197 (sd 0.266) | **0.317** | 0.298 |
| room5 | 0.690 (sd 0.218) | **0.283** | 0.318 |
| room6 | 1.428 (sd 0.294) | **0.321** | 0.287 |
| mean | 1.464 | **0.307** | 0.312 |

Two things follow. First, the pre-merge tilt has a **sd of only 0.2-0.3 deg about a
mean of 0.7-3.1 deg** -- a constant offset, not drift, exactly as a frame-convention
bug predicts. Second, the merged tilt lands at 0.307 deg against OpenVINS' 0.312 deg,
i.e. **this is the benchmark's own floor**, not XIVO's residual error; XIVO is better
on 4 of 6 sequences and the two means agree to 0.005 deg.

Meanwhile yaw scatter is untouched by the fix (6-room mean 0.698 deg pre-merge vs
0.704 deg merged), so no yaw error was absorbed. And on yaw XIVO is decisively
better than the reference: 0.704 deg against OpenVINS' 1.473 deg, better on **all
six** sequences.

### What did not work

Raising the IMU bias random walks to the kalibr values makes orientation *worse*.
`gravity: 9.80766` is worse than 9.8, because TUM-VI's accelerometer scale is ~0.5%
low -- the true gravity magnitude implied by the groundtruth is 9.73-9.77. First-estimates
Jacobians (FEJ) were implemented and correct but measured inside the noise floor;
they ship off by default. TUM-VI room sequences do **not** start stationary (6-18 deg/s
from the first IMU record), so a static-initialization window is not available.

room3 is the worst sequence in both modes afterwards and is the only one where stereo
is worse than mono. Its measured gyro bias is 2.7e-3 rad/s -- 22x what the filter can
represent and 4-5x every other sequence's.

---

## 2. Position: mono 0.0928 -> 0.0555 m, stereo 0.0636 -> 0.0492 m (OpenVINS 0.0621 / 0.0677)

Worktree `xivo-position`, branch `auto-position`, merged as `c0e7f62`. Notes:
`notes-n-prompts/notes-position/`, and `config-delta.md` there is the authoritative
key list.

Mono beats OpenVINS on `ate_002` (0.0566 vs 0.0621) and on RPE-8m position by 15%
(0.0263 vs 0.0308). Every sequence improves. The per-key contributions below were
measured on this branch, so they are quoted against its 0.0566; the two efficiency
branches that landed afterwards moved the endpoint to 0.0555 mono / 0.0492 stereo
without changing any of these keys.

Final per-sequence mono `ate_002`, 6-member means: room1 0.0578, room2 0.0444,
room3 0.0603, room4 0.0390, **room5 0.0941**, room6 0.0374. room5 is the outlier
that carries the mean; the other five average 0.0478.

### Seven config keys, with each one's measured contribution

| key | old -> new | mono `ate_002` effect |
|---|---|---|
| `tracker_cfg.histogram_method` | absent (`NONE`) -> `CLAHE` | **-0.0158** |
| `tracker_cfg.subpix_refine` | absent -> `true` | -0.0062 |
| `use_OOS` | `false` -> `true` | -0.0053 |
| `consistent_init.enable` | absent -> `true` | -0.0042 (on the full stack; **-0.0138** on the OOS base) |
| `OOS.min_observations` | 5 -> 2 | -0.0035 |
| `oos_meas_std` | 3.5 -> 1.0 | -0.0019 |
| `OOS.pose_window` | 0 -> 20 | load-bearing: with 0 the OOS path is inert |
| `tracker_cfg.grayscale` | absent -> `true` | +0.0002 (a prerequisite, not a tuning knob) |

**CLAHE is the single largest effect in the whole project.** The mechanism is
specific: TUM-VI frames measure mean intensity 62.9 at r < 80 px and 33.2 at
r > 270 px -- a 0.53 vignetting ratio -- while FAST uses one global threshold, so
detection was starved in exactly the periphery where the parallax is. Clip limit
10.0 and an 8x8 grid are the OpenCV defaults, matching OpenVINS
`TrackKLT.cpp:61-63`; `clip_limit=4` measured worse.

**`use_OOS` and `consistent_init` must ship together.** `consistent_init` on a config
without the OOS window is **+0.0057 m (worse)**; on the OOS base it is **-0.0138 m**.
This is the one interaction in the delta and the reason the merge could not cherry-pick.

Every one of the four new keys is a no-op when absent, so merging the *code* without
the config is bit-identical. The numbers come from the config.

### The leads that were wrong

Recorded because they are plausible and cost measurement time:

* `visual_meas_std: 0.75` is **not** a stereo leftover. The comment above it is a
  copy-paste, but the value is the shipped mono value, and 1.0 (+0.0053) and 1.5
  (+0.0095) both measure worse -- monotone in the wrong direction.
* "Switch to inverse depth" was already done: `Feature::x_` is `(X/Z, Y/Z, log Z)` in
  the reference group's camera frame. There is no XYZ representation to switch away
  from. Both adjacent knobs lose: `triangulation.method=l1_angular` is +0.0054, and
  `use_depth_opt=true` (`Feature::RefineDepth`) **diverges** -- room1 members score
  ATE 13913 / 2384 / 24863 m and two runs died on signal 6. That is a real HEAD bug;
  do not re-enable the flag as-is.
* `use_OOS` *alone* costs ATE (+0.0036) -- it only pays once the window and
  `consistent_init` are in.
* Robust cost is not the gap: neither system has one. XIVO's
  `Estimator::HuberOnInnovation` (`src/estimator.cpp:1610`) is dead code, never called.
* `epipolar_rejection` is +0.0030 once CLAHE is in -- the two remove the same bad
  correspondences. Raising `num_features_min/max` from 135/180 to 180/240 is +0.0047:
  the filter is not short of tracked features, it is short of *good* ones.

---

## 3. Efficiency: met on mono, 0.984x on stereo throughput

Four branches landed, in this order: `auto-speed` (`6c4bb4d`), `auto-oosfast`
(`b0d7ec5`), `auto-frontfast` (`017c4a4`), `auto-covrun` (`b565b25`). A fifth,
`auto-covscratch`, was built and measured and then not merged (3d). Notes:
`notes-n-prompts/notes-speed/`, `notes-oosfast/`, `notes-frontfast/`.

| stage | mono FPS | stereo FPS | mono peak RSS | stereo peak RSS |
|---|---|---|---|---|
| pre-merge `auto` | 101.8 | 45.0 | 134.1 | 139.3 |
| + `auto-speed`, alone | 141.3 | 64.1 | 82.5 | 86.3 |
| + the accuracy config (`c0e7f62`) | 83.1 | 41.1 | 101.6 | 161.8 |
| + `auto-oosfast` (`b0d7ec5`) | 101.8 | 46.2 | 96.1 | 142.8 |
| + `auto-frontfast` (`017c4a4`) | 120.3 | 64.0 | 88.1 | 105.2 |
| + `auto-covrun` (`b565b25`) | **123.4** | **70.0** | **86.7** | **94.6** |
| (`auto-covscratch`, not merged) | 123.3 | 70.1 | 85.3 | 96.2 |
| OpenVINS | 114.4 | 71.1 | 88.2 | 95.4 |

The three rows after the accuracy config are the recovery: **+2.25 ms/frame mono and
+2.71 ms stereo from the out-of-state and promotion products, then +9.3% from PNG
decode and +27% stereo from the match, then -1.38 ms/frame stereo from chunking the
covariance update** -- 12.03 -> 8.11 ms/frame mono, 24.36 -> 14.29 ms stereo. Mono
clears both targets; stereo clears memory and lands 0.23 ms/frame short on
throughput.

**Every one of the three came from a lead that turned out to be wrong, replaced by a
measurement.** The predicted QR measurement compression cannot pay because XIVO's
stacked residual is shorter than its state; the predicted vignette gain map made the
run *slower*; the predicted memory-bandwidth bound on `H P` was worth 0.023 ms, not
0.55. What actually paid was, in order: full-width products against a
`564 x 564` covariance for a handful of live rows, the PNG decoder, and the height of
the measurement.

### The efficiency branch met the target in isolation

Paired against the pre-merge baseline: mono 101.8 -> **141.3 FPS** (1.39x), stereo
45.0 -> **64.1 FPS** (1.42x); peak RSS mono 134.1 -> **82.5 MB**, stereo 139.3 ->
**86.3 MB**. That is 1.23x OpenVINS on mono throughput and 0.94x its memory, with
accuracy re-rolled and not regressed (every ATE mean within one ensemble sd, every
orientation and RPE mean improved).

Its four milestones: single-channel decode (`cv::imdecode(raw, IMREAD_GRAYSCALE)` at
the file-path entry points, so the front end stops converting 3-channel BGR every
frame), batched block-sparse products in the EKF update, removal of three computed
quantities nothing reads, and a memory pass that stopped giving all 800 pooled
features their own 399 kB OOS scratch Jacobian (-302 MB) and stopped `resize()`
zero-filling them (-310 MB, which had been 70% of RSS).

### Then the position config landed and cost 41%

The 3-way merge measured **0.73x / 0.58x** OpenVINS on throughput -- worse in both
modes than the pre-merge baseline it had beaten by 1.4x. Both individual targets were
real; they are in tension, and the accuracy work is by far the more expensive.

**A merged tree's throughput cannot be inferred from its branches'.** Every branch
here met its own target in isolation and the merge met one of three. Re-measuring the
merge from scratch, rather than composing ratios, is what turned "the merge is slow"
into an assignable defect.

### Where the 41% is

One-core mono, six rooms, 6 runs per arm, each arm one key flipped off the merged
config. Baseline 83.1 FPS = 12.03 ms/frame. Raw: `experiments/results/knobfps_*_mono/`.

| arm | FPS | ms/frame saved | peak RSS |
|---|---|---|---|
| `use_OOS=false` | 112.0 | **3.10** | 78.2 MB |
| `consistent_init.enable=false` | 98.1 | **1.84** | 98.1 MB |
| `histogram_method=NONE` | 92.5 | 1.22 | 115.2 MB |
| `histogram_method=HISTOGRAM` | 90.4 | 0.97 | 108.3 MB |
| `OOS.pose_window=20 -> 10` | 86.4 | 0.45 | 97.9 MB |
| `OOS.max_observations=15 -> 5` | 84.7 | 0.23 | 82.6 MB |
| `subpix_refine=false` | 83.5 | 0.06 | 96.5 MB |

Four readings:

1. `use_OOS` and `consistent_init` are **4.94 ms of the 12.03 ms/frame budget**, i.e.
   41% of all per-frame work.
2. **The OOS cost is per-frame overhead, not per-observation work.** Halving the pose
   window buys 0.45 ms and cutting max observations from 15 to 5 buys 0.23 ms --
   together 22% of the 3.10 ms, for a large accuracy sacrifice. So the cost is not in
   the number of rows processed. That is the signature of a naive full-size update,
   per-update allocation, or Eigen temporaries -- not of a mistuned window.
3. `consistent_init` at 1.84 ms/frame is out of all proportion to a covariance
   computed for a handful of promoted features per frame.
4. `subpix_refine` is **free** (0.06 ms) for -0.0062 m of ATE. CLAHE costs 1.22 ms,
   of which plain `equalizeHist` accounts for only 0.25 ms.

### 3a. `auto-oosfast` -- the update path, at zero accuracy cost

Merged as `b0d7ec5`. Mono 83.0 -> **101.8 FPS** (1.228x), stereo 41.1 -> **46.2 FPS**
(1.125x); peak RSS -7.7 MB mono, -8.0 MB stereo. **72 paired accuracy runs agree
run-for-run to the scorer's printed precision** (largest single-run difference 1e-6 m),
so this is the one part of the recovery that is free.

Estimator timers, ms/frame, mean over room1-6, `base/candidate`:

| timer | mono | stereo |
|---|---|---|
| `process-tracks` | 5.407 / **3.168** | 9.034 / **6.343** |
| untimed remainder (`MarginalizeOOSPoint` + `InitializeFeatureCovariance`) | 2.404 / **0.680** | 2.483 / **0.791** |
| `oos-jacobian` (incl. the OOS gate) | 0.327 / **0.044** | 0.563 / **0.083** |
| `update` | 2.504 / **2.270** | 5.757 / **5.232** |
| `track` (front end, untouched here) | 3.464 / 3.443 | 9.366 / 9.328 |

The mechanism, and it is not the one the lead predicted. **The QR measurement
compression was the wrong lead.** OpenVINS compresses because its stacked MSCKF
residual has more rows than states; XIVO's has ~181 mono / ~360 stereo rows against
564 columns, so `m < n` and the compression cannot pay. The real cost was that
`P_` is always `564 x 564` with vacated slots, and three code paths formed
full-width products against it for a handful of live rows:

* `OOSGating` read all 2.54 MB of `P_` to gate 9 rows -- hence the 7.4x on
  `oos-jacobian`.
* `MarginalizeOOSPoint` and `InitializeFeatureCovariance` formed `x * kFullSize`
  products. `consistent_init` fires **17511 times in one mono room1 run** and had no
  timer, which is why the knob sweep saw 1.84 ms/frame nobody could place. These two
  are the largest single part of the win.
* the OOS measurement blocks opted out of the block-sparse machinery with
  `gsind = -1`, so they paid for the live extent, 488.5 of 564 columns. A run-aware
  `RunSet` (`core.h`, +165 lines) fixed that.

Shipped behind `oos_fast.enable` (default `false`, `true` in both evaluated configs).

### 3b. `auto-frontfast` -- the image path

Merged as `017c4a4`. Mono 83.0 -> **94.9 FPS** (1.143x), stereo 41.0 -> **53.6 FPS**
(1.307x) measured on its own paired baseline; peak RSS -25 MB stereo. Three changes:

| change | mono | stereo |
|---|---|---|
| `fast_png_decode` (bit-identical) | +9.3% | +9.3% |
| `stereo_matching.back_track=false` + `max_level=2` | -- | +27% |
| `KLT.max_level=4` | +3.5% | +2% |

**Image decode was the largest single line item in either system, and it was free to
fix.** `cv::imread(IMREAD_GRAYSCALE)` on a TUM-VI 512_16 frame costs 2.81 ms -- 21% of
XIVO's mono frame. A new `src/pngfast.cpp` (libdeflate, fused unfilter, 16->8 strip,
zlib fallback if libdeflate is absent) does it in 1.42 ms with **zero changed output
bytes**: checked by unit test against `cv::imdecode`, and independently by `cmp` on
`XIVO_DUMP_PRECISE=1` trajectories with the key on and off, in **both** modes.

OpenVINS pays the same 2.82 ms/image, so decode is 5.64 ms of its 14.08 ms stereo
frame -- **40% of the 71.1 FPS target is PNG, not VIO.** XIVO now enters the
comparison with a 2.8 ms/frame structural advantage on stereo that OpenVINS does not
have.

**Everything else in the image path is load-bearing for accuracy, and making it
cheaper loses.** The lead's static-vignette-gain-map idea was refuted by measurement
(mono 75.7 vs 80.6 FPS -- *slower*), and so were two variants. The mechanism is worth
carrying forward: a track that dies lands in the OOS path, which costs 2-3x more per
feature than the equalization that would have kept it alive. The GAINMAP arm ran
21874 feature inits against 17743. The premise was wrong too -- the 62.9 -> 33.2
radial intensity falloff is mostly dark corners outside the ~190 deg image circle, not
vignetting.

The one deliberate accuracy trade in the whole project is here: stereo spends
0.0019 m of ATE against a 0.0186 m margin for +30.7% throughput and -25 MB.

### 3c. `auto-covrun` -- chunk the covariance update, and the memory follows

Merged as `b565b25`. Stereo 64.0 -> **70.1 FPS** and peak RSS 105.2 -> **94.6 MB**;
mono 120.3 -> **123.4 FPS**, 88.1 -> **86.7 MB** (the mono figure read 85.4 in the first
pass; see the +-1.5 MB caveat in the Headline). One change does both.

A phase probe inside `EkfUpdateDowndate` (`perf` is blocked by
`perf_event_paranoid` on this box) over room5's 2841 updates, ms/update:

| phase | stereo | mono |
|---|---|---|
| `M = H P` | 0.681 (14.5%) | 0.542 (24.8%) |
| `S = M H^T` | 0.269 | 0.088 |
| `LLT(S)` | 0.515 | 0.107 |
| `W = L^-1 M` (trsm) | **1.551 (32.9%)** | **0.536 (24.5%)** |
| `P -= W^T W` | **1.602 (34.0%)** | **0.836 (38.3%)** |
| total | 4.713 | 2.183 |

The trsm runs at 41 GFLOP/s and the downdate at 47 -- **both already at the
machine's single-core limit**, so rearranging them buys nothing. And the occupancy
census says `occupied-dim` 466 against `live-dim` 491, so describing the live set
better caps out at ~5% (measured: 2%). With `trsm = m^2 n/2`, `LLT = m^3/6`,
`downdate = n^2 m/2` and `n` fixed by the state, **`m` -- the measurement height --
is the only free variable, and the measurement splits.**

Apply the rows in `C` sequential chunks, each against the covariance the previous
chunks left and with its innovation re-predicted as `r_c -= H_c * err_so_far`. This
is the *same* update, not an approximation: the information form is additive in the
chunks,

    P+^-1     = P^-1 + sum_c H_c^T R_c^-1 H_c
    P+^-1 err = sum_c H_c^T R_c^-1 r_c

whenever `R` is block diagonal across them (it is diagonal) and `H` is held at one
linearization point (nothing is re-evaluated between chunks). It is a different
*factorization* -- `C` Choleskys of `m/C` rows for one of `m` -- so it is not
bit-identical, and it is better conditioned. Measured stereo sweep, ms/update:
`C=1` 4.618, `C=2` 3.728, `C=3` 3.478, **`C=4` 3.392**, `C=6` 3.388, `C=8` 3.446 --
trsm falls as `1/C`, `LLT` as `1/C^2`, the downdate is flat, the mirror grows
linearly, exactly the model. Shipped `chunks: 4` stereo, `3` mono.

**The same change is the memory fix.** A parallel probe attributed the stereo
peak-RSS overshoot by forcing `MALLOC_MMAP_THRESHOLD_=65536` and reconstructing the
live mapping set from an `mmap` strace. It is not the pyramids (forcing a second full
left pyramid every frame costs +0.2 MB, i.e. noise) and not the mapper. It is four
dense matrices coexisting in one update at room5's widest -- 473 rows mono, **860
stereo**, because a stereo view contributes 4 equations to a marginalized
out-of-state track instead of 2:

| buffer | mono | stereo |
|---|---|---|
| `H_`, `rows x 564` | 2.04 MB | 3.70 MB |
| `M = H P` | 2.04 MB | 3.70 MB |
| `S`, `rows^2` | 1.71 MB | **5.64 MB** |
| Eigen's copy of `S` inside `LLT` | 1.71 MB | **5.64 MB** |

The sizes are exactly `page_round(rows^2 * 8)`, which is how they were identified.
`S` is quadratic in `m`, so **chunking makes it quadratic in `m/C`**: a chunk only
ever forms the *diagonal* block of `S` it owns, and the off-diagonal blocks are the
inter-chunk cross-covariances, which sequential processing folds into `P` instead of
forming. 11.3 MB becomes 0.7 MB.

Accuracy: **415 of 432 values across a 72-run paired jitter ensemble are identical
run for run**; mono identical on all 216; stereo differs on 3 of 36 runs, all room6,
improving on four of five metrics. Correctness: `ChunkedEqualsTheBatchUpdate` checks
`P` and `err` at the real dimensions for `C` in {2,3,4,8,16} against **both** the
batch downdate and an independent dense Joseph form, to 1e-9 relative -- a subtle
error in a covariance update does not show up as a divergence, so it is checked
against a reference sharing no code with it.

One behaviour change, on a path that has never fired: with `chunks > 1`, a *later*
chunk whose `S_c` has no Cholesky factor is dropped with a warning rather than
triggering the whole-batch Joseph fallback, since `P` already carries the earlier
chunks. A first-chunk failure still returns false with `P` untouched. The alternative
was a 2.5 MB snapshot of `P` per update, i.e. a fifth of the memory saving spent on
an unreached path.

### 3d. `auto-covscratch` -- measured, and not merged

Notes: `notes-oosfast/m7-update-scratch.md`. The last round went after the leading
suspect in the remaining gap: `Eigen::LLT<MatX> llt(Sc)` copies the innovation
covariance before factorizing it, ~370 kB x 4 chunks per stereo update plus an
allocation each. `LLT<Ref<MatX>>` over a tight map removes the copy; two other pieces
of dead work went with it (per-update construction of the temporaries, and zeroing the
columns of `H P` outside the live extent, which the update never reads -- proved by a
NaN-poison test).

It is **bit-identical**, and that was checked rather than argued. The branch reports all
12 sequence-mode combinations reproducing `b565b25` byte for byte at 17 significant
digits (`XIVO_DUMP_PRECISE=1`); I re-ran 4 of them (room3, room5 x both modes) from one
worktree across the merge, so no build flag could differ, and got 8 of 8 files
identical -- with mono-vs-stereo as the control that the comparison is capable of
failing. `ctest` 22/22, `unitTests_ekf_update` 22 -> 23 cases.

**It was still not merged.** Three alternated A/B pairs on one core:

| | `b565b25` | `auto-covscratch` | delta |
|---|---|---|---|
| stereo ms/frame | 14.285 (sd 0.002) | **14.263** (sd 0.002) | -0.022 (pairs -0.026/-0.020/-0.018) |
| mono ms/frame | 8.107 (sd 0.003) | 8.110 (sd 0.002) | flat |
| stereo peak RSS | **94.6** (x3, one 94.7) | 96.2 (x4) | **+1.6, reproducible** |
| mono peak RSS | 86.7 | **85.3** | -1.4, same session |

The timing win is real, but **a single pass cannot see it**: one fresh pass compared
against a pass of the *same* commit taken two hours earlier gave +0.020 ms -- the wrong
sign, and larger than the effect. Only the interleaved pairs resolve it. That is what
the alternation rule in the protocol section is for.

The saving is nonetheless 0.15% of the stereo frame, and it does not flip stereo
throughput (0.984x -> 0.986x). The stereo peak RSS does flip: 0.992x -> 1.009x. So
merging it converts **13 of 14 reported metrics into 12 of 14**, and `auto` was reset
back to `b565b25`.

Three things about that regression are worth recording, because the first reading of it
was wrong:

* It is **not caused by any of the three new keys.** With all of them at their defaults
  -- behaviourally identical to `b565b25` -- room5 stereo still reads 96.0 MB. What
  moved arena top is the restructuring of the temporaries (four `VecX` members of a
  local struct in place of two `MatX` locals), not any change in behaviour. And the
  attribution runs both ways: turning *off* the dead-store elimination, which changes no
  arithmetic either, moves the same number by 1.1 MB (96.0 -> 94.9).
* The **live buffer set went the other way.** Forcing large blocks to `mmap` so peak RSS
  reflects live pages rather than arena top: 88.1 MB at `b565b25` against 86.6 for the
  new code -- 1.5 MB *smaller*. (That mode also costs 28% of stereo throughput, so it
  is a diagnostic, not an option.) `ru_maxrss` reverses the true ordering here.
* It is nonetheless a **real cost of the binary that would ship** -- the process does
  hold those pages, and `ru_maxrss` is the number this report quotes for both systems.
  Nor is it noise: OpenVINS repeats to 0.2 MB, `b565b25` to 0.1 MB, and the new code
  returned 96.2 on four passes out of four. The first reading of this leaned on the
  +-1.5 MB of cross-session drift, but that drift is a *mono* phenomenon; using it to
  excuse a reproducible stereo regression would be citing the wrong sequence's noise.

So it is better code that produces a worse scorecard, and it sits on `auto-covscratch`
@ `48884e5` waiting for either an allocator-policy fix or a stereo throughput win big
enough to make 0.022 ms worth having.

### What is still open on stereo

**0.23 ms/frame**: 14.285 ms against the 14.057 ms OpenVINS takes, 0.984x. Memory is
met (94.6 vs 95.4 MB). The remaining stereo frame is 9.33 ms of `track` (2.84 ms of it
decode, twice) and 3.39 ms of `update`.

The filter-side items have now all been costed, and they do not add up:

| item | worth | why it is not taken |
|---|---|---|
| the `LLT` copy and the dead stores | **0.022 ms**, measured | 3d: costs 1.6 MB of stereo peak RSS |
| per-chunk mirror, reading only `P`'s lower triangle | ~0.09 ms, modelled | transposes gemm operands, so not bit-identical; needs the full 72-run ensemble |
| `chunks: 5` instead of 4 | 0.017 ms, measured | a different factorization, so it needs its own ensemble |
| `ReserveOOSRows`' `2x` growth | unmeasurable as time | ~10 resizes per run; the 1.4-2.0 MB it leaves resident is the only real part |
| out-of-state rows as their own update | caps buffers at `max(360,500)^2` not `860^2` | moves the second block's linearization point, so it changes the estimate |

That totals ~0.13 ms of exact, in-scope headroom against a 0.23 ms gap, and two thirds
of it needs a fresh accuracy ensemble. **There is no route to 0.23 ms on the filter side
that is both exact and free.** Every other phase of the update is already at the core's
limit -- the downdate runs at 47 GFLOP/s and the triangular solve at 41.

What is deliberately *not* on that list: spending accuracy. Stereo's 0.0187 m of
`ate_002` margin could buy the throughput several times over (`use_OOS=false` alone
is worth 3.10 ms/frame), and it has not been spent.

---

## What was borrowed from OpenVINS, and the absence of a dependency

The brief allowed taking ideas or code from OpenVINS but forbade depending on it.
**XIVO's build and source reference nothing from OpenVINS.** `grep -rni` over `src`,
`pybind11`, `CMakeLists.txt` and `cfg` finds eleven matches and all eleven are
comments citing a file as the origin of a recipe; the link line is unchanged and no
`ov_*` target, header or library is used.

| what | where it landed | form |
|---|---|---|
| gravity-aligned output frame; `R_GtoI` from measured gravity | `estimator.h` `gwb()`/`gwc()` | idea (`ov_init/.../StaticInitializer.cpp:121-125`) |
| Joseph-form / invertible-init EKF update recipe | `ekf_update.h:77`, `feature.h:304` | algorithm, reimplemented (`StateHelper::EKFUpdate`, `initialize_invertible`) |
| first-estimates Jacobians, incl. refreshing the IMU `fej` every propagation | `feature.cpp:851`, `oos.cpp:519` | algorithm, reimplemented (`UpdaterHelper.cpp`) -- **ships off**, measured inside the noise floor |
| CLAHE clip limit 10.0, 8x8 grid | `cfg/eff_*.json:357` | two constants (`TrackKLT.cpp:61-63`) |
| two-view epipolar outlier rejection | `tracker.h:476` | algorithm, reimplemented -- **ships off**, +0.0030 m once CLAHE is in |
| no stereo re-verification of already-tracked points | `tracker.h:201` | idea, and the justification for `back_track=false` (`TrackKLT.cpp:278`) |

One borrowed idea was tried and rejected on the numbers:
`UpdaterHelper::measurement_compress_inplace`. See [3a](#3a-auto-oosfast----the-update-path-at-zero-accuracy-cost) --
it is only a win when the stacked residual has more rows than states, and XIVO's has
fewer.

---

## Reproducing

```bash
cd /home/ubuntu/workspace/auto-slam-engineer/experiments/openvins

# accuracy, final XIVO, both modes, 6-member null-perturbation ensemble (72 runs)
CPU_BASE=0 CPU_SPAN=180 ./run_xivo_reference.sh \
  --worktree xivo --mode both --jitter 6 --out ../results/final_acc_final5

# throughput, one core, serial, idle box (must not share the box); one call per mode
CPU_BASE=0 CPU_SPAN=1 ./run_xivo_reference.sh \
  --worktree xivo --mode mono --seeds 1 --timing --no-score \
  --out ../results/final_fps_final5_mono

# the OpenVINS reference, same protocol
./run_openvins.sh --onecore --repeats 1 --out ../results/final_fps_openvins
```

**Throughput needs at least three passes, and for small effects they must be
alternated.** A single pass carries ~0.04 ms/frame of session-level drift -- enough to
report the wrong *sign* on a 0.02 ms change, which is exactly what happened once here
before the A/B was run properly. Within one session each arm repeats to sd 0.002
ms/frame, so run `A B A B A B` back to back rather than two blocks:

```bash
for i in 1 2 3; do
  for w in xivo-oosbase xivo; do            # base, then candidate
    CPU_BASE=0 CPU_SPAN=1 ./run_xivo_reference.sh --worktree $w --mode stereo \
      --seeds 1 --timing --no-score --out ../results/ab6_${w}_s$i
  done
done
```

The same applies to peak RSS, with a caveat: OpenVINS repeats to 0.2 MB and XIVO
*stereo* to 0.1 MB, but XIVO *mono* `ru_maxrss` drifts **+-1.5 MB across sessions on a
byte-identical binary** (glibc arena top, not live pages). Quote a mono memory delta
only from arms measured in the same session, and if the number is load-bearing check it
under `MALLOC_MMAP_THRESHOLD_=65536`, which makes peak RSS track the live buffer set
(and costs 28% of stereo throughput, so it is a diagnostic only).

Scoring is `score_openvins.py`, one code path for both systems, one groundtruth, one
association window. It needs `dependencies/venv/bin/python` -- the system `python3`
(3.14) has no numpy. Aggregate a throughput dir with `/tmp/agg_fps.py <tag>`; it must
glob `<dir>/<mode>/`, **not** `<dir>/*/`, or a dir holding both modes reports the
alphabetically-first one twice.

The accuracy ensemble member column in `summary.csv` is `repeat`, not `seed`; group by
it and average the six rooms per member, then take the sd across members.

Result directories:

| dir | what |
|---|---|
| `experiments/results/final_acc_final5` | **final `auto` @ `b565b25`**, accuracy, both modes, jitter 6 |
| `experiments/results/final_fps_final5_{mono,stereo}` | **final** one-core throughput, pass 1 |
| `experiments/results/final_fps_final6_{mono,stereo}` | same binary, pass 2 -- the cross-session drift control |
| `experiments/results/ab6_xivo-oosbase_{s,m}{1,2,3}` | `b565b25` arm of the alternated A/B (3d) |
| `experiments/results/ab6_xivo_{s,m}{1,2,3}` | `839f45e` arm, interleaved with it |
| `experiments/results/fps_ov_p2`, `fps_ov_p3` | OpenVINS passes 2 and 3, for its own sd |
| `experiments/results/verify6_{base,new}` | 17-digit trajectory dumps, `b565b25` vs `48884e5`, room3+room5 x both modes |
| `experiments/results/final_fps_{merged,speedonly,premerge,openvins}_*` | earlier stages, same protocol |
| `experiments/results/knobfps_*_mono` | per-knob throughput attribution |
| `experiments/results/frontfast_{TBASE,TFINAL,D}` | `auto-frontfast` paired timing and accuracy |
| `experiments/results/xivo_ref_jitter` | pre-merge `auto` accuracy baseline |
| `experiments/results/ov_accuracy`, `ov_fps_onecore` | OpenVINS reference |

Branch-local dirs under `experiments/openvins/results/` (`covrun-*`, `oosfast-*`) are
paired against *their own* baselines. Correct as ratios; not quotable as absolutes on
the final tree.

Building a fresh worktree needs two things no script does: copy the built
`thirdparty` (`cp -a xivo/thirdparty/. <new>/thirdparty/` -- the trailing `/.` matters)
and configure with `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` plus
`-DOpenCV_DIR=.../dependencies/opencv_install/lib/cmake/opencv4`. Write the flags
literally; zsh does not word-split, so `cmake .. $CM` passes them as one argv.

For bit-identity checks set `XIVO_DUMP_PRECISE=1` -- `dump/tumvi_<seq>_cam0` otherwise
prints six decimals, so an `md5sum` match proves agreement only to 1e-6 m.
