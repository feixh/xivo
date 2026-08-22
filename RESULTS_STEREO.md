# Stereo + IMU results on TUM-VI

Branch `auto-stereo`. Config `cfg/tumvi_stereo.json`, built at the CMake defaults
(`EKF_MAX_FEATURES=90`, `EKF_MAX_GROUPS=45`). Seed 0, ASLR off, one thread per
process. Full write-up and the engineering notes behind every number are in the
workspace's `notes-n-prompts/report-stereo.md` and `notes-n-prompts/notes-stereo/`.

ATE is `evaluate_ate.py`; the two columns are `--max_difference 0.02` (headline)
and `0.001` (the association used by the monocular numbers this is compared
against — it scores only ~26% of frames, so both are reported). RPE is
`--fixed_delta --delta_unit s --delta 1`.

## Means over room1–room6

| arm | ATE@0.001 | ATE@0.02 | RPE_tra (m) | RPE_rot (deg) |
| --- | --- | --- | --- | --- |
| **stereo + IMU** | **0.0476** | **0.0575** | **0.0145** | 0.6206 |
| monocular, same capacity (control) | 0.0792 | 0.0953 | 0.0243 | 0.6204 |
| monocular, upstream capacity 30/60 | 0.1144 | 0.1396 | 0.0299 | 0.6216 |

Stereo is 40% better than the like-for-like monocular control and 61% better than
the monocular result at upstream capacity.

## Per sequence, stereo + IMU

| seq | ATE@0.001 | ATE@0.02 | RPE_tra | RPE_rot | new features seeded by stereo |
| --- | --- | --- | --- | --- | --- |
| room1 | 0.0551 | 0.0665 | 0.0142 | 0.529 | 75.4% |
| room2 | 0.0435 | 0.0491 | 0.0152 | 0.725 | 76.4% |
| room3 | 0.0549 | 0.0815 | 0.0144 | 0.731 | 69.5% |
| room4 | 0.0434 | 0.0538 | 0.0130 | 0.635 | 70.3% |
| room5 | 0.0612 | 0.0621 | 0.0198 | 0.573 | 75.8% |
| room6 | 0.0278 | 0.0320 | 0.0103 | 0.531 | 75.1% |
| **mean** | **0.0476** | **0.0575** | **0.0145** | **0.6206** | 73.8% |

## Notes on interpreting these

- **Roughly half the gain is stereo and half is capacity.** The upstream builds
  hold 30 features and track 60; this config holds 90 and tracks 180, which is
  worth having only because stereo seeds those features with metric depth. See
  the capacity discussion in `README.md`.
- **RPE_rot is dominated by the reference, not the estimator.** Per-sequence
  RPE_rot agrees to within 0.0008 deg between the stereo and monocular arms in
  all six rooms. Of the 0.62 deg, ~0.31 deg is `evaluate_rpe.py`'s
  nearest-neighbour ground-truth association (interpolating the mocap to the
  estimate's stamps scores the same trajectories at ~0.54 deg) and ~0.28 deg is
  the mocap's own attitude noise, leaving ~0.46 deg of real attitude error.
- **Noise scale.** Perturbing a physically neutral config knob moves the
  six-room mean ATE by ~0.006 m. Reseeding the RNG moves it by ~0.001 m and so
  understates tuning uncertainty by about 6×; don't use seed replicates as error
  bars here.

## Reproducing

    cmake -S . -B build && make -C build -j
    OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 \
    XIVO_RANDOM_SEED=0 setarch -R \
      python scripts/pyxivo.py -root /path/to/tumvi -dataset tumvi \
        -seq room1 -cfg cfg/tumvi_stereo.json -mode eval -dump /tmp/out

For the monocular controls, set `stereo`, `stereo_init.enable` and
`stereo_update.enable` to `false` (all three: a stereo-init flag with no rig
configured is a startup error). For the upstream-capacity arm also build with
`-DEKF_MAX_FEATURES=30 -DEKF_MAX_GROUPS=15` and set
`tracker_cfg.num_features_min/max` to 45/60.
