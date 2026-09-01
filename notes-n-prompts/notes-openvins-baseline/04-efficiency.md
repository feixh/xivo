# Measuring OpenVINS throughput (2026-08-27)

## What is being timed

`run_euroc_folder` wraps a clock around each `feed_measurement_camera()` call —
that is KLT tracking + MSCKF/SLAM update + marginalization, i.e. everything the
estimator does with an image. `stats.txt` reports mean/median/p95/max ms and
`fps_mean = 1/mean`. Separately, `wall_total_s` covers the whole replay loop
(PNG decode + IMU feeding + all of the above), giving
`fps_wall = frames_processed / wall_total_s`.

**`fps_wall` is the number to quote across systems.** XIVO's FPS in this
workspace is wall-clock over frames including PNG decode and the Python feed loop
(`notes-n-prompts/notes-efficiency/harness/fps_one.sh`), so `fps_mean` would be
comparing an estimator against a pipeline. On TUM-VI 512×512 the gap is large:
mono 170 FPS estimator-only vs 114 FPS end-to-end. `wall_imread_s / wall_total_s`
says why — PNG decode alone is **33% of the mono and 40% of the stereo wall
clock** on one core.

## The protocol, and why each knob is there

```bash
experiments/openvins/run_openvins.sh --out experiments/results/ov_fps_onecore --onecore
```

`--onecore` = serial + `taskset -c 0` + `setarch -R` +
`OMP_NUM_THREADS=OPENCV_FOR_THREADS_NUM=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`
+ `--num_opencv_threads 1`. This is deliberately the same recipe XIVO's FPS
harness uses, so the two systems' numbers can sit in one table.

Measured on this box (AMD EPYC 9R14, 192 cpus, load ~1.6), 6 rooms, mean:

| setting | stereo | mono |
|---|---|---|
| 1 cpu, all pools = 1 — **the comparable number** | **71.2** FPS end-to-end, 120.1 estimator-only | **114.4** FPS end-to-end, 170.5 estimator-only |
| unpinned, `num_opencv_threads: 1` | 119.9 estimator-only | 169.9 |
| unpinned, `num_opencv_threads: 4` (config default) | 112.9 estimator-only | 164.6 |
| 4 cpus/run, 12 runs concurrent | 143.7 estimator-only | 206.7 |

Three things worth noticing:

* **The config's own `num_opencv_threads: 4` is slower than 1** (112.9 → 119.9
  FPS stereo, a 6% loss). Same finding as XIVO
  ([[xivo-pin-threads-for-batches]]): at 512×512 with ~200 features the parallel
  regions are too small to pay for the barriers. It costs 3 extra cores to lose
  6% throughput.
* **Concurrent runs look *faster* than the serial one-core run** (143.7 vs 120.1
  estimator-only) because each got 4 cpus and the box is 192-core, so nothing
  contended. Never read FPS off an accuracy batch.
* **Peak RSS depends on the cpu mask**: 95.7 MB stereo pinned to one cpu, 110.3 MB
  unpinned on the same binary and the same data. glibc sizes its malloc arenas
  from the visible cpu count, so an unpinned process on a 192-core box carries
  ~15 MB of extra arena. Quote the pinned figure and say so
  ([[xivo-memory-measurement-caveats]]).

## Trap in the harness itself: `nproc` obeys `OMP_NUM_THREADS`

Both runners pin concurrent runs with `taskset -c $((slot % NCPU))`. Setting the
thread caps *before* reading `NCPU="$(nproc)"` makes `nproc` answer **1** —
coreutils' `nproc` deliberately honours `OMP_NUM_THREADS`/`OMP_THREAD_LIMIT` — so
`slot % 1 == 0` and all 72 runs of an ensemble land on cpu 0. The symptom is
subtle: everything runs, produces correct output, and takes 17× longer;
`ps -o psr=` is what makes it visible. Read the cpu count before exporting the
caps (both scripts now do, with a comment).

## Real-time margin

At 20 Hz images, one core: **3.56× real time stereo, 5.72× mono** (mean over the
six rooms; worst sequence 3.50× / 5.58×). p95 per-frame is 11.2 ms stereo /
8.0 ms mono against a 50 ms budget, and the worst single frame in the whole
one-core pass is 29 ms — so there is no tail risk at this capacity either.

`init_time_s` (4.05–6.60 s, identical mono/stereo) is not a cost of the
estimator: it is the static-initialization detector waiting for
`init_imu_thresh` of excitation, which is a property of how each sequence starts.
It does mean the first 4–7 s of every sequence has no pose output, which is why
`traj.txt` is ~130 poses shorter than `frames_processed`.
