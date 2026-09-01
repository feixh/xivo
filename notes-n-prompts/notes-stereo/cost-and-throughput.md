# Cost of the shipped stereo config — FPS, where it goes, and the trade curve

Measured after M7, in answer to "what is the FPS now, and how much slower is it than
upstream capacity?". Harness: `m6-artifacts/harness/fps_one.sh`, `fps_batch.sh`,
`fps_curve.sh`; raw output `m6-artifacts/sweeps/fps.log` and `fps_curve.log`.

## Protocol

- One core: `OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1`. Not an artificial handicap — see §Threads.
- `-mode runOnly`, so the Python saver is out of the loop. `XIVO_RANDOM_SEED=0`,
  `setarch -R`, so each cell does the same arithmetic on every repeat.
- Wall clock from `/usr/bin/time`; per-component times from XIVO's own `Timer`
  (`print_timing: true`, which prints the cumulative mean every 50 frames — the last
  block is the run average).
- **Runs are strictly sequential.** Running arms concurrently would measure memory
  bandwidth, not the estimator.
- Two repeats per cell, arms interleaved across repeats so load drift hits all of them.
  room1 = 2821 frames, room6 = 2636, both 20 Hz.
- Four builds are involved because EKF capacity is compile-time: `lib_f30`, `lib_f60`,
  `lib` (=90/45), `lib_f120`. Sanity-checked that `lib_f30` really is 30 by pointing
  `cfg/tumvi_stereo.json` at it and watching `CheckMemoryPools` refuse to start.

Configs are the shipped `cfg/tumvi_stereo.json` with only the capacity numbers and
the stereo flags changed, so the mono arms here are exactly the control arms in
`xivo/RESULTS_STEREO.md`. Pools were held at ≥2× the tracker cap in every arm, so
pool size is not a hidden variable.

## The 2x2

| arm | EKF / tracker | room1 FPS | room6 FPS | slowdown | peak RSS |
| --- | --- | --- | --- | --- | --- |
| mono, upstream | 30 / 60 | 88.5 | 90.4 | 1.0× | 153 MB |
| stereo, upstream capacity | 30 / 60 | 45.2 | 44.1 | 2.0× | 161 MB |
| mono, new capacity | 90 / 180 | 20.5 | 19.4 | 4.3–4.7× | 447 MB |
| **stereo, shipped** | 90 / 180 | 11.8 | 11.3 | 7.5–8.0× | 454 MB |

**~11.5 FPS shipped, ~89 FPS at upstream capacity, so 7.5–8× slower and 0.6× real
time against a 20 Hz camera.** Stereo alone is 2.0×; capacity alone is 4.3–4.7×;
together 7.5×, slightly under the product because some per-frame cost is fixed.

Both single factors are worth keeping in mind separately: someone who wants stereo's
accuracy at real time can have the 2.0× and decline the 4.3×.

## Where the time goes (per frame, room1)

| component | mono 30/60 | stereo 90/180 | factor |
| --- | --- | --- | --- |
| track | 2.23 | 14.96 | 6.7× |
| MH gating | 0.42 | 9.50 | 23× |
| stereo geometric gating | — | 8.84 | new |
| EKF update proper (`actual-update`) | 1.26 | 36.18 | 29× |
| `visual-meas` total | 4.44 | 71.03 | 16× |
| IMU propagation, per sample (9.97/frame) | 0.330 | 0.672 | 2.0× |
| wall clock per frame | 11.30 | 84.92 | 7.5× |

Reading this:

- **The covariance update dominates, not the front end.** 3× the in-state features
  made `actual-update` 29× more expensive: an empirical exponent of ~2.7 on state
  size, which is what a dense EKF update should look like. KLT over two images is
  15 ms of 71.
- **MH gating scaled worse than the update** (23× for 3× the features) and is now 9.5
  ms/frame. It is a per-feature Mahalanobis test against a growing covariance, so it
  inherits the same scaling. If anyone optimizes this system for speed, `MH-gating`
  and `actual-update` are the only two places worth looking.
- **Stereo's own gating is 8.8 ms/frame** at this capacity, comparable to MH gating,
  and 0.39 ms at 30 features — it also scales with the number of features carried.
- `jacobian` never exceeded 0.23 ms. The `dXc1_d = Rc1c0 · dXcn_d` reuse (M1/M5) kept
  the stereo Jacobian essentially free, as intended.
- Wall clock minus `visual-meas` minus propagation is ~7 ms/frame for stereo and
  ~3.6 mono: PNG decode (two images vs one) plus the Python feed loop. Real deployment
  would not pay the Python part.
- `use_canvas: true` in the shipped config costs ≤0.5 ms/frame (the residual inside
  `visual-meas` after `track` and `process-tracks`), so it is not worth turning off.

## Threads do not rescue it

An unpinned process spawns ~255 OpenCV/OpenMP threads and runs at 708% CPU. Measured
on a full eval run: **165.7 s unpinned against 176.1 s pinned to one thread — 6%
faster for 7 cores of extra CPU**, with bit-identical output. The bottleneck is dense
Eigen work on a single covariance, which OpenCV's pool cannot touch. See
`[[xivo-pin-threads-for-batches]]`.

So 11.5 FPS is not a measurement artifact of the pinning; it is roughly what this
configuration does on a core, and ~12.2 FPS is the best it does with 7 more.

## Accuracy against speed

Accuracy is the six-room mean from the M6 sweep; FPS is room1, one repeat.

| capacity (EKF / tracker) | mean ATE@0.001 | mean ATE@0.02 | room1 FPS | peak RSS |
| --- | --- | --- | --- | --- |
| 30 / 120 | 0.0615 | 0.0802 | 39.4 | 173 MB |
| 60 / 120 | 0.0523 | 0.0629 | 23.8 | 314 MB |
| **90 / 180 (shipped)** | 0.0476 | 0.0575 | 11.8 | 454 MB |
| 120 / 240 | 0.0485 | 0.0576 | 6.7 | 855 MB |

Cost roughly doubles for each step up the curve while ATE improves by 0.005–0.017 m,
and the last step buys nothing at all: **120/240 is 1.8× slower and 1.9× larger than
90/180 for an ATE difference of 0.0001 m**, which is a second, independent reason not
to ship it — the accuracy argument alone (inside the noise, see `[[m6-capacity]]`)
made it a coin flip, and the cost breaks the tie.

**60/120 is the configuration to recommend when real time matters**: 0.0629 m at
23.8 FPS, still 55% better than the upstream monocular baseline, and ahead of a 20 Hz
camera where the shipped default is not. The shipped default optimizes accuracy
because that is what exit criterion 2 asked for.

## Caveat

Absolute FPS is machine-dependent (shared 192-core host; load average varied between
10 and 141 during the batch). Repeat spread stayed under 5% regardless, and every
ratio quoted here is within-machine, so the ratios are the durable part.
