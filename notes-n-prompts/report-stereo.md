# Stereo + IMU visual-inertial odometry in XIVO — final report

Branch: **`auto-stereo`** in `/home/ubuntu/workspace/auto-slam-engineer/xivo`, HEAD `abd0ede`.
Dataset: TUM-VI `room1`–`room6`, 512×512 fisheye stereo, 101.06 mm baseline.
Shipped config: `cfg/tumvi_stereo.json`. Detailed engineering notes: `notes-n-prompts/notes-stereo/`.

---

## 1. Result against the exit criteria

| criterion | target | achieved | verdict |
| --- | --- | --- | --- |
| 1. mean ATE, room1–room6 | < 0.06 m | **0.0575 m** (0.0476 m at the stricter association) | **met** |
| 2. mean ATE as small as possible | — | at the minimum of a swept, unimodal capacity curve; every other knob swept and rejected | **met, argued in §4** |
| 3. mean RPE_rot | < 0.5 deg | **0.6206 deg** | **not met — floor-limited, see §5** |

Headline ATE uses `evaluate_ate.py --max_difference 0.02`; the `0.001` column is
kept alongside because that is what the repo's `RESULTS.md` used, so the
comparison against the monocular baseline is like-for-like. Both are reported
everywhere in this report; the criterion is met under either.

### Final numbers (seed 0, ASLR off, single-threaded, default build)

| arm | ATE@0.001 | ATE@0.02 | RPE_tra (m) | RPE_rot (deg) |
| --- | --- | --- | --- | --- |
| **stereo + IMU (shipped)** | **0.0476** | **0.0575** | **0.0145** | 0.6206 |
| monocular, *same* capacity (control) | 0.0792 | 0.0953 | 0.0243 | 0.6204 |
| monocular, upstream capacity | 0.1144 | 0.1396 | 0.0299 | 0.6216 |
| workspace `README.md` monocular baseline | 0.121 | — | — | 0.622 |

Per-sequence, shipped stereo:

| seq | ATE@0.001 | ATE@0.02 | RPE_tra | RPE_rot | features seeded by stereo |
| --- | --- | --- | --- | --- | --- |
| room1 | 0.0551 | 0.0665 | 0.0142 | 0.529 | 75.4% |
| room2 | 0.0435 | 0.0491 | 0.0152 | 0.725 | 76.4% |
| room3 | 0.0549 | 0.0815 | 0.0144 | 0.731 | 69.5% |
| room4 | 0.0434 | 0.0538 | 0.0130 | 0.635 | 70.3% |
| room5 | 0.0612 | 0.0621 | 0.0198 | 0.573 | 75.8% |
| room6 | 0.0278 | 0.0320 | 0.0103 | 0.531 | 75.1% |
| **mean** | **0.0476** | **0.0575** | **0.0145** | **0.6206** | 73.8% |

The criterion is on the mean and the mean is met at both protocols. Being
explicit about the tail: at the loose protocol room3 (0.0815) and room1 (0.0665)
are individually above 0.06; at the strict protocol every room is below 0.062.

### "Much better than monocular" (requirement 2)

- **−61%** against the monocular number in the workspace `README.md` (0.121 → 0.048).
- **−40%** against a monocular control run at the *same* feature capacity and
  otherwise identical config (0.0792 → 0.0476). This is the honest figure for
  what stereo itself buys, and it is the one to quote: roughly half of the total
  gain came from stereo and half from the capacity increase that stereo made
  worth having.
- RPE_tra improves **−40%** on the same control (0.0243 → 0.0145), independently
  confirming better local metric consistency rather than a globally luckier
  alignment.

---

## 2. What was built

Eight commits on `auto-stereo`, each a milestone with its own tests:

| commit | milestone | what it does |
| --- | --- | --- |
| `0309993` | M0 | fixes a real Jacobian bug found while baselining (below) |
| `7afce70` | M1 | `CameraManager` singleton → indexed registry; `StereoRig` holding a fixed `T_c1c0` |
| `b399cf7` | M2 | right image from disk to tracker, both entry points; output byte-identical |
| `c38ad4b` | — | deterministic feature/group selection order, so sweeps are comparable |
| `5cb52e0` | M3 | left→right KLT matching with epipolar / circular / disparity gates |
| `30b03a2` | M4 | metric depth seeding by triangulation at first observation |
| `b595e55` | M5 | right-camera measurement rows in the EKF update |
| `abd0ede` | M6 | EKF capacity raised to 90/45, with startup guards for the caps |

31 files, ~3.6k lines under `src/`, `cfg/`, `scripts/`.

**Design choices worth knowing.** Extrinsics are held fixed outside the EKF
state (TUM-VI's calibration is good and the rig is rigid), so stereo adds
measurement rows without adding state. `dXc1_d(state) = R_c1c0 · dXc0_d(state)`,
so the entire monocular Jacobian chain is reused with one 3×3 multiply — that
identity is what kept M5 small. cam0/cam1 timestamps are bit-identical in this
dataset, so no interpolation is needed. Everything is off unless the config sets
`"stereo": true` and supplies `camera1_cfg` / `stereo_cfg`; with it absent the
binary is byte-for-byte the monocular one, which M1–M3 enforced as an explicit
regression gate.

**The M0 bug.** `src/feature.cpp:688-689` in `FillJacobianBlock` wrote the
column offset `goff` twice, so the reference-group *rotation* block was
overwritten by the translation block and `goff+3` was never written at all. The
filter still converged because the body-pose blocks were correct and because
`OnePointRANSAC` builds its own `H_`, bypassing the broken function. Fixing it
moved the monocular baseline 0.1209 → 0.1019 m. This is an upstream bug, not one
introduced here.

---

## 3. Where the gain came from

| step | mean ATE@0.001 | Δ |
| --- | --- | --- |
| monocular baseline (`RESULTS.md`) | 0.1209 | — |
| M0 Jacobian fix | 0.1019 | −16% |
| M4 stereo depth seeding | 0.0801 | −22% |
| M5 right-camera update rows | 0.0760 | −5% (inside the noise on its own) |
| M6 capacity 30/60 → 90/180 | **0.0476** | −37% |

`visual_meas_std` 1.5 → 0.75 also ships, and was measured before the capacity
change: mean ATE@0.02 0.1013 → 0.0841 at otherwise-fixed M5 settings. Its effect
is not separable from capacity in the final config, so it is not given a row.

Two things about this table are worth stating plainly.

**Depth seeding did the heavy lifting, not the extra measurement rows.**
Monocular XIVO initializes every feature at a guessed `initial_z: 2.5` m and
lets a depth subfilter converge. Stereo replaces that guess with a triangulated
metric depth and a real covariance at first observation — 74% of new features get
one. That removes a systematic scale error at birth, which is why RPE_tra fell
27% at M4. M5's second measurement row per feature is a smaller, better-conditioning
effect that mostly showed up by making `visual_meas_std: 0.75` pay off, which it
did not in the monocular configuration.

**The largest single win was two integers, and it is a stereo result even though
it looks like tuning.** `EKF_MAX_FEATURES` was 30 and `tracker_cfg.num_features_max`
was 60 in every published XIVO number. Raising both to 90/180 cut ATE 37%. It is
a stereo result because raising capacity is only worth it when the extra features
arrive with trustworthy metric depth; the same capacity increase applied to the
monocular arm gains much less (0.1144 → 0.0792, and that arm is still 66% worse
than stereo).

---

## 4. Why 0.0476 is the minimum, not an arbitrary stopping point

Capacity was swept as a curve, not to a target. `fN tM` = N features in the EKF
state (N/2 groups), M tracked:

| arm | mATE001 | mATE02 | RPE_rot |
| --- | --- | --- | --- |
| f30 t60 (upstream) | 0.0760 | 0.1013 | 0.6211 |
| f30 t120 | 0.0615 | 0.0802 | 0.6218 |
| f60 t120 | 0.0523 | 0.0629 | 0.6211 |
| **f90 t180 (shipped)** | **0.0476** | **0.0575** | 0.6206 |
| f120 t240 | 0.0485 | 0.0576 | 0.6217 |
| f150 t300 | 0.0568 | 0.0693 | 0.6220 |

f120 is inside the noise (+0.0009) and f150 is clearly worse (+0.0092), so this
is a genuine turnover — plausibly texture competition on a 512² fisheye image
(the tracker's `mask_size` is 15 px) plus more weakly-conditioned features whose
linearization error the filter must absorb.

**On "noise".** Error bars here come from perturbing a physically-neutral config
knob, which moves the six-room mean by ~0.006 m. RNG-seed replicates move it by
~0.001 m and therefore understate tuning uncertainty by ~6×; using them would
have made several rejected arms look like wins. The 0.0284 m capacity effect is
~5× the honest noise scale.

Everything else swept at the final capacity, and rejected:

| arm | mATE001 | verdict |
| --- | --- | --- |
| shipped | 0.0476 | — |
| `Qimu.gyro_bias` ×0.1 / ×10 | 0.0494 / 0.0503 | worse |
| `Qimu.gyro` over a 16× span | never better | worse |
| `visual_meas_std` over a 4× span | 0.75 is the minimum | shipped |
| `stereo_update.R_scale` ∈ {0.5,1,2,4,8} | flat within noise | 1.0 shipped |
| de-rotated 200-sample gravity init | 0.0520 | worse |
| `tracker_cfg.use_prediction: true` | 0.0476 | **bit-identical — the key is never read in `src/`** |
| `use_OOS: true` | — | unimplemented upstream: `LOG(FATAL)` at `estimator.cpp:126` |
| `use_depth_opt: true` | — | hangs / numerically unsound at this capacity |

---

## 5. Criterion 3: RPE_rot is limited by the reference, not the estimator

Mean RPE_rot is **0.6206 deg** against a 0.50 deg target. It did not move.
Across every knob swept it stayed in 0.6183–0.6575, including the capacity
change that cut ATE by 37% and moved RPE_rot by 0.0015 deg. A quantity that
ignores a change of that size is not primarily measuring the estimator.

The decisive observation: **per-sequence RPE_rot agrees to within 0.0008 deg
between the stereo arm and both monocular arms in all six rooms** (room1 0.5293 /
0.5289 / 0.5292; room3 0.7309 / 0.7304 / 0.7312; and so on). It is a property of
each sequence's ground truth, not of the trajectory being scored.

Measured decomposition (harness scripts under `notes-stereo/m6-artifacts/harness/`):

| term | deg | how it was measured |
| --- | --- | --- |
| GT association artifact | 0.31 | `evaluate_rpe.py` pairs each estimate with the *nearest* mocap sample, up to 4.2 ms away at 120 Hz. Slerping the GT to the estimate's stamps instead drops the score 0.6289 → 0.5439 |
| mocap's own attitude noise | 0.28 | local-cubic fit to GT attitude; residual 0.08–0.19 deg/axis, propagating to a per-room floor of 0.23–0.36 deg |
| real estimator attitude error | ~0.46 | remainder in quadrature |

Consistency: √(0.46² + 0.28² + 0.31²) = 0.626 vs 0.62 observed.

So **about 0.42 of the 0.50 deg budget is noise in the reference.** Reaching
0.50 deg as measured requires the estimator's own attitude error to fall from
~0.46 to below √(0.50² − 0.42²) = 0.27 deg — a 42% reduction in real attitude
error, not the 19% the raw numbers imply. Four leads were opened and closed:
gyro noise scaling, initial-attitude accuracy (improving it from 1.47 to 0.73 deg
changed nothing measurable — see `notes-stereo/m6-attitude-initialization.md`),
hand-eye calibration residual, and gyro scale/misalignment. None of them is the
bottleneck. Honest conclusion: this criterion is not reachable by tuning; it
needs either a fairer metric (interpolated GT association would give ~0.54 deg
for the *same* trajectories) or a structural change to attitude estimation.

---

## 6. What it costs

The accuracy is bought with compute, and the shipped default trades in favour of
accuracy. One core, `-mode runOnly`, two repeats per cell, arms interleaved;
sequences are 20 Hz.

| arm | EKF / tracker | room1 FPS | room6 FPS | slowdown | vs the camera | peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| mono, upstream | 30 / 60 | 88.5 | 90.4 | 1.0× | 4.5× real time | 153 MB |
| stereo, upstream capacity | 30 / 60 | 45.2 | 44.1 | 2.0× | 2.2× real time | 161 MB |
| mono, new capacity | 90 / 180 | 20.5 | 19.4 | 4.3–4.7× | 1.0× real time | 447 MB |
| **stereo, shipped** | 90 / 180 | **11.8** | **11.3** | **7.5–8.0×** | **0.6× real time** | 454 MB |

**~11.5 FPS shipped against ~89 FPS at upstream capacity: 7.5–8× slower, and below
real time on this dataset.** The 2×2 separates the causes — stereo alone costs 2.0×
(two images, two KLT passes, the left→right match and its gating), the capacity
increase alone 4.3–4.7×, and together 7.5×.

Per frame, room1: `track` 2.23 → 14.96 ms, MH gating 0.42 → 9.50, EKF update proper
1.26 → 36.18, total `visual-meas` 4.44 → 71.03 ms. So the cost is in the **covariance
update, not the front end**: tripling the in-state features made the update 29× more
expensive (an empirical exponent of ~2.7 on state size), while KLT over two images is
15 ms of the 71. The stereo Jacobians never exceeded 0.23 ms/frame, which is the
`dXc1_d = R_c1c0 · dXc0_d` reuse from M1/M5 paying off.

Threads do not rescue it: unpinned (≈255 OpenCV/OpenMP threads, 708% CPU) finished a
full eval in 165.7 s against 176.1 s pinned to one thread — 6% for 7 cores of CPU,
because the bottleneck is dense Eigen work on one matrix.

Pairing accuracy with speed across the capacity curve (FPS room1, one repeat):

| capacity | mean ATE@0.02 | room1 FPS |
| --- | --- | --- |
| 30 / 120 | 0.0802 | 39.4 |
| 60 / 120 | 0.0629 | 23.8 |
| **90 / 180 (shipped)** | **0.0575** | **11.8** |
| 120 / 240 | 0.0576 | 6.7 |

**60 / 120 is the configuration to recommend when real time matters**: 0.0629 m at
23.8 FPS, still 55% better than the upstream monocular baseline, and ahead of a 20 Hz
camera where the shipped default is not. Lower `EKF_MAX_FEATURES` and
`tracker_cfg.num_features_max` together to get it. The shipped default optimizes
accuracy because that is what criterion 2 asked for; this is the one place where
meeting the criteria as stated is not the same as the best engineering default, so it
is stated explicitly rather than buried.

Absolute FPS is machine-dependent (shared 192-core host, load average 10–141 during
the batch); repeat spread stayed under 5% and every ratio is within-machine. Full
method, component table and caveats: `notes-stereo/cost-and-throughput.md`.

---

## 7. Verification

- **84 unit tests, 82 pass.** 48 are new on this branch across 6 new test
  binaries: `unitTests_stereo` (13), `unitTests_jacobians_stereo` (15, numerical
  vs analytic Jacobians), `unitTests_stereo_loader` (3), `unitTests_determinism`
  (5), `unitTests_gravity_init` (5), `unitTests_memory_pools` (7, death tests).
- The 2 failures — `NumericalLinearAlgebra.SlowAndFastGivensMatch` and
  `Triangulation.Angular_Reprojection_Error` — **predate M0 and are untouched by
  this work.** Run from the repo root (`for t in bin/unitTests_*; do ./$t; done`),
  not `ctest`; `GLOG_minloglevel` must not be set, or the memory-pool death test
  that asserts on a logged advisory will fail spuriously.
- **Regression gate:** with `"stereo": false`, output is byte-identical to the M0
  monocular baseline. Enforced at M1, M2 and M3, when nothing consumed the right
  pixels yet — which is exactly when plumbing bugs are cheap to find.
- **Stereo match rate** on room1's first 600 frames: 97.8% of attempted
  left→right matches accepted (31347/32057), consistent with a correct rig.
- **End-to-end evaluation** was run on all six rooms at every milestone from M4
  on, and the final table in §1 was re-measured on the plain default build after
  M6 was committed.
- **Determinism:** `XIVO_RANDOM_SEED=0` plus `setarch -R`; runs are bit-identical
  across repeats. Batches are pinned to one thread per process
  (`OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1`) — proven
  bit-identical to the unpinned output, and necessary because each pyxivo process
  otherwise spawns ~255 threads and 90 concurrent runs drive load average past 5000.

---

## 8. Reproducing it

```bash
cd /home/ubuntu/workspace/auto-slam-engineer/xivo
git checkout auto-stereo
cmake -S . -B build && make -C build -j
OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 \
XIVO_RANDOM_SEED=0 setarch -R \
  ../dependencies/venv/bin/python3 scripts/pyxivo.py \
    -root ../data/tumvi -dataset tumvi -seq room1 \
    -cfg cfg/tumvi_stereo.json -mode eval -dump /tmp/out
```

The CMake defaults are already `-DEKF_MAX_FEATURES=90 -DEKF_MAX_GROUPS=45`.
Build with `30`/`15` to reproduce upstream-capacity numbers. Three capacity
numbers are easy to confuse and all three matter:

- `EKF_MAX_FEATURES` — compile-time filter capacity;
- `tracker_cfg.num_features_max` — how many the tracker supplies (raising only
  the first does nothing: 60 and 90 in-state were bit-identical while the tracker
  stayed at 60);
- `memory.max_features` — a fixed object pool, whose exhaustion is a `LOG(FATAL)`
  mid-run.

That last one silently killed 5 of 12 sweep runs ten minutes in. `CheckMemoryPools`
in `src/factory.cpp` now validates all three at startup, the config declares the
capacity it was tuned for via `require_ekf_max_features`, and the pools warn at
90% occupancy. A config tuned for 90 features run against a 30-feature binary now
refuses to start instead of quietly scoring 60% worse.

---

## 9. Summary

Stereo + IMU is implemented, tested, and delivered on `auto-stereo`. It reaches
**mean ATE 0.0575 m** (criterion 1, target < 0.06), a **40% improvement over a
like-for-like monocular control** and **61% over the published monocular
baseline** (criterion 2's "much better"), at what a swept curve shows to be the
accuracy minimum of the configuration space explored (criterion 2). Criterion 3,
mean RPE_rot < 0.5 deg, is **not met at 0.6206 deg**, and the measurements in §5
show ~0.42 deg of that 0.50 deg budget is noise in the mocap reference and its
association, leaving ~0.46 deg of real attitude error that no swept knob moved.
That criterion is not reachable by tuning this estimator.

The one thing a reader should carry away beyond the numbers: the accuracy costs
7.5× the compute and 2.9× the memory of upstream, which puts the shipped default
below real time at 20 Hz (§6). If throughput matters more than the last 0.005 m,
build at 60/120 and take 0.0629 m at 23.8 FPS.
