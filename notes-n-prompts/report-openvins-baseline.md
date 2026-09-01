# OpenVINS on TUM-VI room1–6: accuracy, throughput, and how it compares to XIVO

Task: set OpenVINS up in `experiments/open_vins`, run mono+IMU and stereo+IMU on
TUM-VI room1–room6, evaluate ATE, record FPS, and leave the harness reusable for
more datasets.

Detailed notes: [`notes-openvins-baseline/`](notes-openvins-baseline/) —
[setup](notes-openvins-baseline/01-setup-and-build.md),
[eval protocol](notes-openvins-baseline/02-eval-protocol.md),
[noise](notes-openvins-baseline/03-determinism-and-noise.md),
[efficiency](notes-openvins-baseline/04-efficiency.md),
[comparing against XIVO](notes-openvins-baseline/05-comparing-against-xivo.md).
Harness and how to add a dataset: [`experiments/openvins/HOWTO.md`](../experiments/openvins/HOWTO.md).
OpenVINS `v2.7-20-g6948812`; XIVO `auto` @ `0476a98` (M7).

## Headline

Every accuracy cell is the mean over a **6-member ensemble** of each system under
a physically null perturbation (see "Error bars" — single runs of either system
are not meaningful at this resolution). `±` is the sd of the 6-room mean across
members.

| 6-room mean, TUM-VI room1–6 | OpenVINS mono | OpenVINS stereo | XIVO mono | XIVO stereo |
|---|---|---|---|---|
| **ATE position RMSE [m]** | **0.061** ±0.001 | 0.068 ±0.001 | 0.093 ±0.007 | **0.064** ±0.005 |
| ATE position, `ov_eval` posyaw [m] | 0.063 | 0.070 | 0.097 | 0.069 |
| ATE orientation [deg] | **1.42** | 1.44 | 1.82 | 1.80 |
| RPE 8 m, translation [m] | 0.030 | **0.027** | 0.048 | 0.029 |
| RPE 8 m, rotation [deg] | 0.61 | 0.60 | 0.52 | **0.51** |
| **FPS, one core, end-to-end** | **114** | **71** | 96 | 45 |
| real-time margin @20 Hz | 5.7× | 3.6× | 4.8× | 2.2× |
| peak RSS, one core [MB] | 89 | 96 | 132 | 137 |
| divergences | 0/36 | 0/36 | 0/36 | 0/36 |

Four findings:

1. **In mono, OpenVINS wins clearly and cheaply.** 0.061 m vs XIVO's 0.093 m — a
   34% reduction, ~10 sem apart — while also running 19% faster on one core and
   in a third less memory. It is better on global orientation too (1.42° vs
   1.82°) and on 8 m translation drift (0.030 vs 0.048 m).
2. **In stereo the two are a wash on position.** 0.068 vs 0.064 m by
   `evaluate_ate.py`, 0.070 vs 0.069 m by `ov_eval`. XIVO's nominal 0.004 m edge
   on the first metric is only ~2 sem given its own ensemble spread, and shrinks
   to 0.001 m on the second. Call them equal — and note that OpenVINS reaches
   that tie at **1.6× the frame rate** (71 vs 45 FPS) and 0.70× the memory.
3. **The two systems make opposite orientation trade-offs.** OpenVINS has ~20%
   lower *global* orientation ATE in both modes; XIVO has ~15% lower *local*
   rotation drift (RPE over 8 m) in both modes. Consistent with XIVO's tighter
   short-horizon attitude and OpenVINS' better yaw observability over the whole
   trajectory.
4. **OpenVINS is far more stable.** Under equally null perturbations its 6-room
   mean moves by ±0.001 m and XIVO's by ±0.005–0.007 m; per-sequence
   peak-to-peak is up to 0.016 m for OpenVINS and up to **0.048 m** for XIVO.
   Whatever else the accuracy table says, OpenVINS' answer is much more
   reproducible.

Also worth flagging: **OpenVINS gets almost nothing from stereo on this
dataset**, and on global position it is slightly *worse* with it (0.068 vs
0.061). XIVO gains a lot (0.093 → 0.064). Discussed below.

## Setting it up

Docker was the requested route and is not available here — no `docker` binary,
and the six upstream `Dockerfile_ros{1,2}_*` are `osrf/ros:*` **build** recipes,
not published run-ready images. Used the upstream **ROS-free** build instead
(`-DENABLE_ROS=OFF`), which is a first-class configuration in OpenVINS' own CI.

OpenVINS v2.7 compiles on gcc-15 / OpenCV 4.10 / Eigen 3.4 / Ceres 2.2 with
**zero source changes** — worth noting next to the six porting fixes XIVO needed
for the same toolchain ([[xivo-modern-toolchain-build-fixes]]). Only the build
system needed four small patches: a cmake-4 policy flag, Boost 1.90 no longer
having a `system` component, `ov_eval` never looking for OpenCV on the ROS-free
path, and matplotlib off.

The one real gap: **ROS-free OpenVINS has no real-data entry point.** Both
`run_subscribe_msckf` and `ros1_serial_msckf` are behind `ENABLE_ROS`, and the
serial one reads a rosbag, which TUM-VI does not ship. So the harness adds
`ov_msckf/src/run_euroc_folder.cpp` (~390 lines), a port of the replay logic in
`ros1_serial_msckf` plus `ROS1Visualizer`'s callbacks: it reads ASL/EuRoC
folders, synchronizes the cameras, keeps the same ordering rule (feed a frame
only once an IMU sample past `t_cam + dt_CAMtoIMU` has been consumed), writes
TUM-format poses exactly as `publish_state()` does, and times each update. **No
estimator code was touched.** Configs are the authors' shipped
`config/tum_vi/estimator_config.yaml` plus a mono variant differing only in
`max_cameras: 1` / `use_stereo: false` — so these are OpenVINS' numbers, not
tuned-by-me numbers.

Wiring checks that mattered: online camera-intrinsic calibration converges to
the kalibr values (191.9/191.8/255.0/256.9 vs 190.98/190.97/254.93/256.90),
`frames_processed` equals the cam0 image count exactly on all six sequences, zero
unsynchronized and zero rate-dropped frames, and static init fires at the same
4.1–6.6 s in mono and stereo.

## ATE, per sequence

`evaluate_ate.py` (TUM RGB-D tool, Horn SE(3) alignment), 0.02 s association
window, groundtruth regenerated from `mav0/mocap0/data.csv` at full nanosecond
precision. 6-member ensemble mean per cell.

| system / mode | room1 | room2 | room3 | room4 | room5 | room6 | mean |
|---|---|---|---|---|---|---|---|
| OpenVINS mono | 0.0526 | 0.0768 | 0.0836 | 0.0304 | 0.0768 | 0.0462 | **0.0611** |
| OpenVINS stereo | 0.0749 | 0.0991 | 0.0750 | 0.0339 | 0.0947 | 0.0303 | 0.0680 |
| XIVO mono | 0.0762 | 0.1001 | 0.1343 | 0.0805 | 0.1065 | 0.0594 | 0.0928 |
| XIVO stereo | 0.0636 | 0.0684 | 0.0951 | 0.0472 | 0.0692 | 0.0379 | **0.0636** |

Cross-checked with OpenVINS' own scorer (`ov_eval error_singlerun posyaw`, which
interpolates onto groundtruth times and aligns position+yaw instead of full
SE(3)): 6-room means 0.063 mono / 0.070 stereo for OpenVINS, 0.097 / 0.069 for
XIVO. Two independent implementations with different alignment groups agreeing to
~0.003 m — the trajectories are being read and aligned correctly.

The mono gap is concentrated in **room3, room4 and room5** — XIVO mono reaches
0.134/0.081/0.107 there while OpenVINS mono holds 0.084/0.030/0.077, i.e. those
three sequences account for 0.022 of the 0.032 m mean gap. On room1/2/6 the two
mono arms are within 0.013–0.024 m. Counting best-arm-per-sequence,
an OpenVINS arm wins 4 of 6 (rooms 1, 3, 4, 6) and XIVO stereo wins 2 (rooms 2,
5). No arm diverged on any of the 144 runs.

### Mono vs stereo inside OpenVINS

Stereo improves OpenVINS' *local* accuracy and slightly degrades its *global*
position error: RPE over 8 m goes 0.030 → 0.027 m and 0.61 → 0.60°, orientation
ATE is flat at 1.42 → 1.44°, while position ATE goes 0.061 → 0.068 m. That last
gap survives the noise floor easily (0.0069 apart with sds of 0.0013 and 0.0006),
so it is a real property of this configuration, not a fluctuation.

That is a plausible outcome rather than a red flag. Stereo buys instant metric
depth for close features, which helps short-baseline geometry — exactly what RPE
measures — but the room sequences give a mono filter plenty of translation to
triangulate with anyway, and doubling the measurement count per frame gives the
chi-square gates twice as many chances to admit an outlier into a filter whose
state is already at capacity (11 clones, 50 SLAM features, 200 tracked). It is a
statement about *this config on these six sequences*, not about stereo VIO in
general — note that XIVO's stereo arm gains 0.029 m over its mono arm on exactly
the same data.

## Efficiency

`run_openvins.sh --onecore`: serial, `taskset -c 0`, ASLR off, every thread pool
forced to 1 — deliberately the same recipe XIVO's FPS harness
(`notes-efficiency/harness/fps_one.sh`) uses, so the columns belong in one table.
FPS is **end-to-end** (frames / wall clock, PNG decode included), because that is
what XIVO's published number is.

| one core, 6-room mean | OpenVINS mono | OpenVINS stereo | XIVO mono | XIVO stereo |
|---|---|---|---|---|
| end-to-end FPS | 114.4 | 71.2 | 96.4 | 44.8 |
| × real time @20 Hz | 5.72 | 3.56 | 4.82 | 2.24 |
| estimator-only FPS | 170.5 | 120.1 | — | — |
| per-frame track+update, mean | 5.87 ms | 8.33 ms | — | — |
| per-frame p95 / worst | 7.97 / 18.3 ms | 11.17 / 29.1 ms | — | — |
| peak RSS | 89.2 MB | 95.7 MB | 131.7 MB | 137.1 MB |
| init delay (no pose output) | 4.1–6.6 s | 4.1–6.6 s | none | none |

* **OpenVINS is 1.19× XIVO's throughput in mono and 1.59× in stereo, at ~0.7× the
  memory.** Both clear real time comfortably in both modes on a single core.
* The estimator alone is much faster than the pipeline: PNG decode is 33% of the
  mono and 40% of the stereo one-core wall clock at 512×512. Quote the
  end-to-end number unless you say otherwise; `fps_mean` is estimator-only and
  always higher.
* **The shipped config's `num_opencv_threads: 4` is a net loss** — 112.9 vs 119.9
  estimator-only FPS in stereo against 1 thread. Three extra cores to lose 6%.
  Same finding as XIVO ([[xivo-pin-threads-for-batches]]).
* Worst single frame in the whole one-core pass is 29 ms against a 50 ms budget,
  so there is no tail risk at this capacity either.
* XIVO mono's per-sequence numbers spread 82–104 FPS (room1 is a cold
  page-cache outlier); OpenVINS' spread 112–118. Means are used above.
* OpenVINS emits no pose for the first 4–7 s while its static initializer waits
  for `init_imu_thresh` of excitation; XIVO starts immediately. On a 90–150 s
  sequence that is 3–7% of the trajectory unavailable. It does not bias the ATE
  here (scoring is over associated pairs), but it is a real availability
  difference for a robot.
* Do **not** read FPS off an accuracy batch: the same binary reports 143.7 FPS
  estimator-only in stereo when 12 runs share 4 cpus each on this 192-core box,
  vs 120.1 pinned to one cpu.

## Error bars: why repeats measure nothing, in both systems

OpenVINS is **bit-for-bit deterministic** — one md5 per (mode, sequence) across
repeats, thread counts, cpu pinning and ASLR settings. A repeat-based error bar
would be exactly zero and would license claims no future change can reproduce.
At XIVO HEAD, `XIVO_RANDOM_SEED` turns out to do almost nothing either: mono is
bit-identical across 6 seeds and stereo varies only on room6.

Both ensembles above therefore perturb a **physically null** knob instead:

* OpenVINS: `--gravity_mag` jittered in its 9th significant digit
  (9.807660000000 … 9.807660049038, i.e. 1e-9 relative).
* XIVO: initial velocity `X.Vsb` shifted by k·1e-6 m/s — six orders of magnitude
  inside the filter's own declared 0.7 m/s prior. Same device as
  `run_ensemble_bugfix.sh` ([[xivo-tuning-noise-is-not-seed-noise]]).

Neither changes the problem being solved; both change which features the
chi-square gates admit, and the filters diverge from there. Resulting spread:

| ensemble sd of the 6-room mean | mono | stereo | worst per-sequence peak-to-peak |
|---|---|---|---|
| OpenVINS | 0.0013 | 0.0006 | 0.016 m |
| XIVO | 0.0067 | 0.0045 | 0.048 m |

**Rule: compare 6-room means, never single sequences, and require a cross-system
gap to clear both systems' spreads.** By that rule the mono gap (0.061 vs 0.093)
is real and the stereo gap (0.068 vs 0.064) is not.

## Comparability: what is controlled and what is not

Controlled: same host (AMD EPYC 9R14, otherwise idle), same six sequences, same
groundtruth files, the same `evaluate_ate.py` at the same 0.02 s window, an
independent second scorer agreeing, the same one-core timing recipe, both systems
driven through the same runner/scorer pair (`run_openvins.sh` /
`run_xivo_reference.sh` → `score_openvins.py`), and both re-run at HEAD rather
than quoted from stored results.

Not controlled, stated plainly:

* **Tuning effort.** OpenVINS runs its authors' shipped TUM-VI config untouched.
  XIVO runs `cfg/eff_{mono,stereo}.json` — the shipped config, but on a codebase
  that has had several rounds of bugfix, efficiency and accuracy work *in this
  workspace* against *these same six sequences*. If anything that favours XIVO.
* **Algorithmic scope.** Both are filtering VIO with no loop closure and no
  global bundle adjustment, which is what makes a bare ATE comparison fair at
  all. But state capacities differ (OpenVINS: 11 clones + 50 SLAM features + 200
  tracked; XIVO: 90 features / 45 groups) and neither was re-tuned to the other's
  compute budget. The FPS column is what it costs each system to reach *its own*
  ATE, not a like-for-like per-feature cost.
* **One dataset, six sequences, one motion profile.** All six rooms are the same
  handheld indoor mocap setting. Nothing here says anything about outdoor,
  aerial, or long-corridor behaviour; `HOWTO.md` explains how to add EuRoC or
  UZH-FPV, for which OpenVINS already ships configs.
* **The stale-reference trap.** `results/final/triangulation_configs/sweep_dlt_nodesc`
  (what `RESULTS.md` publishes) scores 0.152 m mean ATE@0.02. That is XIVO from
  before the M-series work, not XIVO at HEAD, and using it would have overstated
  OpenVINS' win by 2.4×. `score_xivo_reference.sh` still scores those stored
  files — it reproduces `RESULTS.md`'s 0.1209 @0.001 exactly, which is how the
  wiring was validated — but it must not be used as the comparator.

## A second trap: do not score OpenVINS at a 0.001 s window

XIVO's own pipeline, and hence `RESULTS.md`, uses `--max_difference 0.001`.
OpenVINS stamps poses at camera time plus its *online-estimated* camera–IMU time
offset, so a 1 ms window associates between **3 and 1138** of ~2700 poses
depending on the sequence. On room1 mono it reports "0.0040 m" — the RMSE of
**three** of 2689 poses. The column is kept in `summary.csv` for provenance and labelled
unusable; every number in this report is at 0.02 s (~98% coverage, including the
initialization phase where the largest errors live). See
[eval protocol](notes-openvins-baseline/02-eval-protocol.md), and note this is a
*different* problem from the one in [[xivo-ate-eval-protocol]], which is about
the same window scoring only blocks of XIVO's own frames.

## Reproducing / extending

```bash
# accuracy, both modes, six sequences (concurrent, ~4 min)
experiments/openvins/run_openvins.sh --out experiments/results/ov_accuracy --repeats 3
# throughput, the cross-system protocol (serial, one core, ~7 min)
experiments/openvins/run_openvins.sh --out experiments/results/ov_fps_onecore --onecore
# noise floor: 6 members, gravity jittered in its 9th digit
for k in 0 1 2 3 4 5; do
  g=$(python3 -c "print('%.12f' % (9.80766 * (1 + $k * 1e-9)))")
  experiments/openvins/run_openvins.sh --out experiments/results/ov_jitter/m$k \
    --extra "--gravity_mag $g"
done
# the XIVO comparator, same layout, same scorer
experiments/openvins/run_xivo_reference.sh --out experiments/results/xivo_ref_jitter --jitter 6
experiments/openvins/run_xivo_reference.sh --out experiments/results/xivo_ref_fps --timing
```

A new dataset needs exactly one new file: a profile in
`experiments/openvins/profiles/` saying where its `mav0/` folders, groundtruth
csv and estimator configs live. `HOWTO.md` walks through it, including which
configs OpenVINS already ships (`euroc_mav`, `kaist`, `uzhfpv`, `rs_*`, `rpng_*`)
and how to derive a mono variant of one.

Result directories behind this report:

| directory | what |
|---|---|
| `experiments/results/ov_accuracy` | 3 repeats × 6 seq × 2 modes (deterministic) |
| `experiments/results/ov_jitter/m{0..5}` | **headline OpenVINS ATE** + noise floor |
| `experiments/results/ov_fps_onecore` | **headline OpenVINS FPS** |
| `experiments/results/ov_timing_t4`, `ov_timing_t1` | `num_opencv_threads` comparison |
| `experiments/results/xivo_ref_accuracy` | XIVO, 6 seeds — evidence seeds do nothing |
| `experiments/results/xivo_ref_jitter` | **headline XIVO ATE** + noise floor |
| `experiments/results/xivo_ref_fps` | **headline XIVO FPS** |

Note on naming: this file is `report-openvins-baseline.md`, deliberately not
`report-openvins.md`, which `requirements-openvins.md` reserves for a separate
task (porting OpenVINS *ideas* into XIVO's mono pipeline). The numbers here are a
useful target for that task: its exit criteria are mono ATE < 0.06 m and mono RPE
rotation < 0.5°, and OpenVINS mono itself sits at **0.061 m and 0.61°** under this
protocol. So the ATE criterion is roughly "match OpenVINS mono" (XIVO mono is at
0.093 today), while the RPE criterion is something OpenVINS does *not* meet and
XIVO mono already nearly does (0.515 ±0.007) — i.e. borrowing wholesale would
move the wrong metric backwards.

Each result directory holds `summary.csv` (one row per run, every metric) and `summary.md`
(tables), plus per-run `traj.txt`, `timing.csv`, `stats.txt`, `cmd.txt` and
`run.log`, and a `run_info.txt` recording the full invocation and the
`git describe` of the system that produced it.
