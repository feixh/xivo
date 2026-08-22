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

## Speed and memory

The accuracy above is bought with compute. Measured on one core (`OMP_NUM_THREADS=1
OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1`), `-mode runOnly`, two repeats per
cell, arms interleaved across repeats; wall clock includes PNG decode and the Python
feed loop. room1 is 2821 frames, room6 is 2636.

| arm | EKF / tracker | room1 FPS | room6 FPS | slowdown | vs the 20 Hz camera | peak RSS | mean ATE@0.02 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| monocular, upstream | 30 / 60 | 88.5 | 90.4 | 1.0× | 4.5× real time | 153 MB | 0.1396 |
| stereo, upstream capacity | 30 / 60 | 45.2 | 44.1 | 2.0× | 2.2× real time | 161 MB | not scored |
| monocular, new capacity | 90 / 180 | 20.5 | 19.4 | 4.3–4.7× | 1.0× real time | 447 MB | 0.0953 |
| **stereo, shipped** | 90 / 180 | **11.8** | **11.3** | **7.5–8.0×** | **0.6× real time** | 454 MB | 0.0575 |

**The shipped configuration runs at about 11.5 FPS on one core, against ~89 FPS at
upstream capacity — 7.5–8× slower, and no longer real time on this dataset.** The
2×2 separates the two causes: stereo alone costs 2.0× (two images, two KLT passes,
the left→right match and its gating), the capacity increase alone costs 4.3–4.7×,
and together 7.5× — slightly less than the product, because some per-frame cost is
fixed.

### Where the time goes

Per frame, room1, mean of two repeats. `visual-meas` is XIVO's own timer and
excludes image decode; `propagation` is per IMU sample, of which there are 9.97 per
frame.

| component | mono 30/60 | stereo 90/180 | factor |
| --- | --- | --- | --- |
| track (KLT, + stereo match) | 2.23 | 14.96 | 6.7× |
| MH gating | 0.42 | 9.50 | 23× |
| stereo geometric gating | — | 8.84 | new |
| EKF update proper | 1.26 | 36.18 | 29× |
| **`visual-meas` total** | **4.44** | **71.03** | **16×** |
| IMU propagation, per sample | 0.330 | 0.672 | 2.0× |
| wall clock per frame | 11.30 | 84.92 | 7.5× |

The cost is in the **covariance update, not the front end**. Tripling the in-state
feature count made the EKF update 29× more expensive — an empirical exponent of
about 2.7 on state size, consistent with the cubic-ish cost of a dense EKF update —
while KLT on two images is only 15 ms of the 71 ms. This also means threads do not
help: an unpinned run (≈255 OpenCV/OpenMP threads, 708% CPU) finished a full eval in
165.7 s against 176.1 s pinned to one thread, a 6% gain for 7 cores, because the
bottleneck is dense Eigen work on a single matrix.

### Accuracy against speed

Capacity is the knob that trades one for the other. Accuracy is the six-room mean
from the M6 sweep; FPS is room1, one repeat, same protocol as above.

| capacity (EKF / tracker) | mean ATE@0.001 | mean ATE@0.02 | room1 FPS |
| --- | --- | --- | --- |
| 30 / 120 | 0.0615 | 0.0802 | 39.4 |
| 60 / 120 | 0.0523 | 0.0629 | 23.8 |
| **90 / 180 (shipped)** | **0.0476** | **0.0575** | **11.8** |
| 120 / 240 | 0.0485 | 0.0576 | 6.7 |

**60 / 120 is the point to take if real time matters**: 0.0629 m at 23.8 FPS is
still 55% better than the upstream monocular baseline while keeping ahead of a 20 Hz
camera, where the shipped configuration does not. The shipped default optimizes
accuracy because that is what the exit criteria asked for; `EKF_MAX_FEATURES` and
`tracker_cfg.num_features_max` are the two numbers to lower, together, if the
trade should go the other way.

Caveats: single-core numbers on a shared 192-core host, so absolute FPS depends on
the machine — but the repeat spread was under 5% even with load average varying
between 10 and 141, and every ratio above is within-machine. `-mode eval` adds a few
percent for state saving.

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
