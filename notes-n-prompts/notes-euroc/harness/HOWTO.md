# OpenVINS benchmark harness — how to run more experiments

Everything here drives the OpenVINS checkout in `experiments/open_vins`
(upstream `rpng/open_vins`, v2.7-20-g6948812) through datasets in ASL/EuRoC
folder format and scores the result. Written for a future agent: start at
"Run something" and read the rest only when it disagrees with you.

## Layout

```
experiments/open_vins/            upstream checkout (4 small local patches, see below)
  ov_msckf/src/run_euroc_folder.cpp   ROS-free dataset player  (added here)
  config/tum_vi/estimator_config_mono.yaml  mono variant of the shipped config (added here)
experiments/ov_build/             cmake build dir -> run_euroc_folder
experiments/ov_build_eval/        cmake build dir -> ov_eval binaries (error_singlerun ...)
experiments/openvins/
  run_openvins.sh                 runner: profile x modes x sequences x repeats
  score_openvins.py               scorer: ATE / RPE / timing -> summary.csv + summary.md
  run_xivo_reference.sh           same, but runs XIVO -> scored by the same scorer
  score_xivo_reference.sh         scores *stored* XIVO trajectories (see caveat below)
  asl_gt_to_tum.awk               ASL groundtruth csv -> TUM txt
  profiles/tumvi_room.sh          dataset profile (root, sequences, config, per-seq args)
experiments/results/              run outputs (ov_accuracy, ov_fps_onecore,
                                  ov_timing_t*, ov_jitter/*, xivo_ref_*)
```

## Run something

```bash
cd /home/ubuntu/workspace/auto-slam-engineer
# accuracy: 6 sequences x {mono,stereo}, concurrent, 4 cpus each
experiments/openvins/run_openvins.sh --out experiments/results/my_run

# timing/FPS numbers: one core, one thread per pool, ASLR off, serial
experiments/openvins/run_openvins.sh --out experiments/results/my_timing --onecore

# the XIVO comparator, same layout, same scorer (6-seed ensemble, then FPS)
experiments/openvins/run_xivo_reference.sh --out experiments/results/xivo_ref_accuracy --seeds 6
experiments/openvins/run_xivo_reference.sh --out experiments/results/xivo_ref_fps --timing

# one sequence, mono, 30 s of data (quick smoke test)
experiments/openvins/run_openvins.sh --out /tmp/smoke --mode mono --seqs room1 \
    --extra "--duration 30"

# re-score an existing directory without re-running anything
experiments/openvins/score_openvins.py experiments/results/my_run
```

`--help` lists every flag. Anything after `--extra` goes straight to the player,
whose own `--help` lists its overrides (`--max_cameras`, `--use_stereo`,
`--init_imu_thresh`, `--gravity_mag`, `--num_opencv_threads`, `--start`,
`--duration`).

## Rebuilding

```bash
cd experiments/ov_build      && cmake . && make -j32 run_euroc_folder
cd experiments/ov_build_eval && cmake . && make -j32
```

From scratch (both dirs), the two flags no default invocation sets:

```bash
sudo apt-get install -y libboost-all-dev libceres-dev libgoogle-glog-dev libgflags-dev libsuitesparse-dev
mkdir -p experiments/ov_build && cd experiments/ov_build
cmake ../open_vins/ov_msckf -DENABLE_ROS=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -Wno-dev && make -j32
mkdir -p ../ov_build_eval && cd ../ov_build_eval
cmake ../open_vins/ov_eval -DENABLE_ROS=OFF -DDISABLE_MATPLOTLIB=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -Wno-dev && make -j32
```

`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is required because cmake 4.x rejects the
`cmake_minimum_required(VERSION 3.3)` in every OpenVINS package. There is no
ROS and no docker on this box, so the ROS-free path is the only one available;
the upstream Dockerfiles are unusable (no docker binary, and they are all
`osrf/ros:*` images anyway).

## Adding a dataset

Copy `profiles/tumvi_room.sh` and define, for the new dataset:

* `PROFILE_ROOT`, `PROFILE_SEQS`
* `seq_folder <seq>` — directory containing `mav0/` (needs `imu0/data.csv`,
  `cam0/data.csv` + `cam0/data/`, and `cam1/*` for stereo)
* `seq_gt_csv <seq>` — ASL groundtruth csv; `asl_gt_to_tum.awk` handles both
  TUM-VI `mav0/mocap0/data.csv` and EuRoC `mav0/state_groundtruth_estimate0/data.csv`
* `seq_config <seq> <mode>` — estimator config. OpenVINS ships configs for
  euroc_mav, kaist, uzhfpv, rs_*, rpng_* under `experiments/open_vins/config/`;
  each one is stereo, so for a mono arm make a `*_mono.yaml` next to it with
  `max_cameras: 1` and `use_stereo: false` (see `config/tum_vi/estimator_config_mono.yaml`)
* `seq_extra <seq> <mode>` — per-sequence overrides. Read the comments in the
  shipped config: TUM-VI needs `init_imu_thresh 0.25` on room6, and the EuRoC
  config has similar per-sequence notes.

Then `run_openvins.sh --profile <name> --out ...`. Nothing else needs touching:
if the dataset has no `cam1`, pass `--mode mono`.

When editing a config **do not put a colon inside an end-of-line comment**
(`# PATCH: foo`): OpenVINS parses configs with `cv::FileStorage`, which reads
that as a nested mapping and fails with "invalid boolean type of []".

## Reading the output

`DIR/<mode>/<seq>_r<k>/` holds `traj.txt` (TUM format, IMU pose in the
gravity-aligned world frame, timestamped in the IMU clock), `timing.csv`
(per-frame track+update seconds), `stats.txt` (key=value summary incl. FPS and
peak RSS), `cmd.txt`, `run.log`. `DIR/run_info.txt` records the invocation and
the OpenVINS git describe. `DIR/summary.csv` / `summary.md` come from the scorer.

Metric choices, and the traps behind them, are in
`notes-n-prompts/notes-openvins-baseline/02-eval-protocol.md`. The short version:

* Headline ATE = `evaluate_ate.py` with `--max_difference 0.02`, matching how
  XIVO is scored in this workspace, plus `ov_eval error_singlerun posyaw` as an
  independent check (they agree to ~0.001 m here).
* **Do not** use the 0.001 s association window on OpenVINS output. Its poses are
  stamped at camera time + the *online-estimated* camera-IMU offset, so a 1 ms
  window associates a phase-dependent 3–1138 of ~2700 poses. That column exists
  in `summary.csv` only because RESULTS.md quotes XIVO at 0.001.
* Runs are bit-for-bit reproducible, but a *physically null* perturbation
  (`--gravity_mag` in the 9th digit) moves per-sequence ATE by ±0.003–0.007 m.
  Compare 6-room means (sd 0.001), not single sequences.
* Quote `fps_wall` (end-to-end) across systems, not `fps_mean` (estimator only) —
  PNG decode is a third of the wall clock here. Both are in `summary.csv`.
* **Never quote `fps_wall` or `peak_rss_mb` from an accuracy pass.** The accuracy
  pass launches every run at once (220 processes for an n=10 × 11-sequence ×
  2-mode sweep); `CPU_SPAN` chooses which cpu each run is *pinned* to, it does not
  cap concurrency. The same config read 27.9 and 65.7 FPS in two such passes.
  Timing comes only from `--timing`, and is aggregated by `report_onecore.py`.

## Aggregating results

Three scripts, and they are not interchangeable:

* `agg_ensemble.py --mode {mono,stereo} --arm NAME DIR [--arm ...]` — the accuracy
  tables. `--mode` is **mandatory** (the XIVO runner writes mono and stereo rows
  into one `summary.csv`), and member globs must be **unquoted** so the shell
  expands them — a quoted `'…/stereo_m*'` reports "(not all arms have all
  sequences)" for every metric rather than failing.
* `report_fps.py DIR [DIR ...]` — `sweep_fps.sh` variants, frame-weighted. Reads
  the `time.txt` that only the XIVO path writes, so it prints a nonzero-exit
  warning for every OpenVINS run.
* `report_onecore.py NAME=GLOB [NAME=GLOB ...]` — the cross-system timing table.
  Reads `stats.txt`, which both paths write, and accepts both directory layouts
  (`<arm>/r0/stereo/<SEQ>_r0` from `sweep_fps.sh`, and
  `<arm>/stereo/<SEQ>_r{0,1,2}` from `run_xivo_reference.sh --jitter`). Use this
  one whenever XIVO and OpenVINS appear in the same table.

## Comparing against XIVO

Use `run_xivo_reference.sh`, which runs the XIVO worktree with
`cfg/eff_{mono,stereo}.json` and writes `DIR/<mode>/<seq>_r<k>/traj.txt` in the
same layout, so `score_openvins.py` scores both systems through one code path.

`score_xivo_reference.sh` instead scores the *stored* trajectories under
`results/final/triangulation_configs/sweep_dlt_nodesc/`. Those are **stale** —
that config predates the bugfix/efficiency work and scores 0.152 mean ATE@0.02
where XIVO HEAD scores 0.064 stereo / 0.093 mono. Keep it for reproducing
RESULTS.md (it recovers 0.1209 at the 0.001 window, exactly the published
number), not for comparisons.

Two flags matter for a real comparison:

* `--jitter 6`, not `--seeds 6`. At HEAD `XIVO_RANDOM_SEED` changes nothing in
  mono and almost nothing in stereo, so a seed ensemble is one sample dressed up
  as six (and lands ~0.004 low). `--jitter` perturbs `X.Vsb` by k·1e-6 m/s
  instead and gives the real ±0.005–0.007 spread.
* `--timing` for throughput; it is XIVO's `fps_one.sh` recipe (one cpu, ASLR off,
  all thread pools at 1, `-mode runOnly`) and pairs with
  `run_openvins.sh --onecore`. It writes no trajectory — the scorer emits
  FPS/RSS-only rows for such runs.

Full protocol, including what it does *not* control:
`notes-n-prompts/notes-openvins-baseline/05-comparing-against-xivo.md`.

## Local patches to the upstream checkout

`git -C experiments/open_vins diff` shows all of them:

1. `ov_msckf/CMakeLists.txt`, `ov_eval/CMakeLists.txt` — drop the `system`
   component from `find_package(Boost ...)`; Boost.System is header-only since
   1.69 and ships no cmake component in Ubuntu's boost 1.90.
2. `ov_eval/cmake/ROS1.cmake` — `find_package(OpenCV 4)` + include/link it: the
   ROS-free build compiles ov_core's sources into `ov_eval_lib`, and upstream
   never looks for OpenCV in that package.
3. `ov_msckf/cmake/ROS1.cmake` — register the `run_euroc_folder` target.
4. `ov_msckf/src/run_euroc_folder.cpp`, `config/tum_vi/estimator_config_mono.yaml` — new files.

No estimator source was touched: gcc-15 / OpenCV 4.10 / Eigen 3.4 / Ceres 2.2
compile OpenVINS v2.7 unmodified.
