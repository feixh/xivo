# What `async_run` actually buys

This started as a correctness check on my own reporting. Throughout the
efficiency work I described XIVO as running "on one core", and the objection was
reasonable: the code *does* have a second thread, and the feature tracker and the
filter *are* separate stages. So the first question was whether the measurements
in `full-tumvi.md` were single-threaded at all. They were. The second question --
what the second thread would buy if it were switched on -- turned out to need
three bug fixes before it could be answered, and the answer is a clean 1.33x.

Worktree `xivo-async`, branch `auto-async` off `auto` at `9e3ec06`.

## 0. Summary

* The earlier measurements were single-threaded, confirmed three ways (§1). But
  the reason I had been giving for that -- thread-pinning environment variables --
  was the wrong reason: those cap library pools and cannot constrain a
  `std::thread`. The correct reason is that `async_run` was `false` in every
  config evaluated.
* `async_run` does **not** run the tracker in parallel with the filter. It is a
  producer/consumer split: the caller's thread queues measurements, one worker
  thread runs the entire estimator. So the ceiling is set by how much work the
  *caller* does, which here is PNG decode -- **27.1%** of a frame, hence a
  predicted **1.37x** (§2).
* Measured, room4: **40.0 -> 55.1 FPS, 1.38x**, CPU 1.00 -> 1.38 cores (§4).
  Across eight sequences spanning 62 240 stereo pairs: **41.2 -> 59.4 FPS
  aggregate**, per-sequence **1.32x-1.45x** (plus slides1, whose apparent 1.72x is
  partly a cheaper trajectory, not overlap). On seven of the eight the speedup
  equals the CPU-utilisation ratio to within 0.02, which is what "the gain is all
  overlap and none of it is an artifact" looks like as a number.
* As shipped it was unmeasurable, not merely slow: the naive A/B looked like a
  **3.1x** speedup and was really a **63% frame drop**, at 2.4 GB peak RSS,
  ending in `abort`. Three separate defects, all fixed (§3).
* Queue depth is a non-knob: the minimum legal depth (11) already gives the full
  overlap, because the producer decodes the next image pair *before* it blocks on
  the push. Deeper only costs RSS -- +70 MB at 512 for zero FPS (§5).
* Accuracy: with the online temporal calibration frozen, async is **byte-identical**
  to sync. With it on, async cannot be identical by construction, but an 8-member
  ensemble over room1-6 puts every metric within half a standard deviation of the
  synchronous ensemble (§6).
* Two measurement traps found on the way, both of which produced a wrong answer
  first: unsynchronized polling made async look 11.9% worse on RPE and made ATE
  *fail outright* on two sequences (§6.1), and the stock `--max_difference 0.001`
  ATE is decided by a single constant timestamp-phase offset per run (§6.2).

## 1. Were the earlier runs single-threaded?

Three independent lines, because the claim had been justified badly the first
time and I did not want to re-justify it badly:

1. **Measured.** The 28-sequence throughput pass records user and wall time per
   sequence. `(user+sys)/wall = 0.99` on all 28. A second thread doing real work
   cannot hide from that ratio.
2. **Configured.** `async_run` is explicitly `false` in all four evaluated
   configs, and the default in `Estimator::Estimator` is `false`. `worker_` is
   never created.
3. **Structural.** The only other threading in the tree is `Process<>`, and
   neither subclass is in this code path: `EstimatorProcess` is instantiated only
   by `src/app/legacy/vio.cpp`, and `ViewPublisher` is the viewer, which
   `-mode runOnly` and the eval harness both leave off.

What I had said instead was that `OMP_NUM_THREADS` and friends pinned it to one
core. Those are set, and they matter -- OpenCV's pool otherwise burns seven cores
for 6% of wall clock -- but they cap *library* pools. They have no effect on a
`std::thread` the estimator starts itself. Right conclusion, wrong reason.

## 2. What the ceiling is, before measuring anything

`async_run` starts one worker (`Estimator::Run`) that drains a
timestamp-ordered heap buffer; every `VisualMeas*`/`InertialMeas` call becomes a
push. So the split is **caller vs estimator**, not tracker vs filter. Everything
inside the estimator -- tracking, stereo matching, propagation, the EKF update --
stays on the one worker thread and stays strictly sequential.

That makes the ceiling measurable in advance. From a `print_timing` run on room4
(`cfg/timing_sync.json`), per stereo frame:

| | ms | share |
| --- | --- | --- |
| wall | 24.86 | 100% |
| `visual-meas` (tracker + EKF; the timer nests `track` and `process-tracks`) | 17.80 | 71.6% |
| `propagation` x 10.0 IMU messages per frame | 0.32 | 1.3% |
| **unaccounted: PNG decode + the Python feed loop** | **6.74** | **27.1%** |

The 6.74 ms is the producer's work, and it is almost all `cv::imread`: the pybind
wrapper decodes both PNGs on the calling thread (`pybind11/pyxivo.cpp`, in
`VisualMeasStereo`), so `data` in `scripts/pyxivo.py` holds paths, not images.

Perfect overlap therefore gives `max(17.80 + 0.32, 6.74) = 18.12` ms, i.e.
**1.37x**, or 39.8 -> ~55 FPS. That number is the whole story of this note, and
it comes with a corollary worth stating plainly: **the gain is the decode share,
so on a live camera that hands over decoded frames, `async_run` buys essentially
nothing.** It is a benchmark-harness win, not an algorithmic one.

## 3. Why it could not be measured as shipped

The first A/B, unmodified code, room4:

```
ARM sync  rc=0   wall=55.9  user=55.20  cpu=1.00  rss_mb=424   stereo: 2226 frames
ARM async rc=134 wall=18.0  user=31.20  cpu=1.85  rss_mb=2435  stereo:  831 frames
```

18.0 s against 55.9 s is a 3.1x speedup and is entirely fictional. Three defects,
independent of each other:

**No backpressure.** `MaintainBuffer` executes a message only in the
`!async_run_` branch, so in synchronous mode the queue is self-limiting -- one
execute per push -- and in async mode the worker is the only consumer with
nothing to stop the producer. The producer is 2.7x faster than the consumer
(§2), so it queued the whole sequence: 2.4 GB of RSS, because every `Visual`
message owns its `cv::Mat`.

**No way to wait for the estimator.** The feed loop returned when the *queue*
accepted the last measurement. `~Estimator` then set `stop_` and joined,
discarding everything still queued -- 1395 of 2226 frames. So the 18.0 s was the
producer's runtime, and the trajectory was truncated to whatever the worker had
reached.

**Shutdown ordered last.** The destructor printed calibration and OOS statistics
*before* stopping the worker, i.e. read counters the worker was still writing.
The abort (`signal 6`) was `LOG(FATAL) << "unknown camera model"` at
`src/camera_manager.h:86` -- the worker touching the camera singleton during
teardown.

The fixes are two commits on `auto-async`:

* `async_queue_limit` (default 16) with a `not_full` condition variable the
  producer waits on; the worker waits on `not_empty` instead of polling with a
  200 us sleep; `Estimator::Wait()` to drain, called from `scripts/pyxivo.py`
  after the feed loop; worker shutdown moved to the top of the destructor. The
  twelve duplicated `buf_.push_back` sites -- an `if (async_run_)`/`else` pair in
  each of the six measurement entry points, differing only in whether they took
  the lock -- collapse into one `Push()`.
* `Estimator::WaitWindow()`, for callers that read state between pushes. See §6.1;
  this one exists because of a measurement error, not a crash.

## 4. The throughput result

`-mode runOnly` in both arms, single sequence at a time, threads pinned,
`XIVO_EIGEN_INIT=none`. `frames` is what the *estimator* processed
(`print_stereo_stats`), not what was fed -- that column is there because of §3.

| sequence | frames | FPS sync | FPS async | speedup | CPU async | user time | RSS sync | RSS async |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| room1 | 2 818 | 38.5 | 51.7 | 1.34x | 1.36 | +1.0% | 448 | 454 |
| room4 | 2 226 | 39.9 | 54.4 | 1.36x | 1.37 | +0.6% | 424 | 424 |
| room6 | 2 633 | 38.1 | 50.3 | 1.32x | 1.34 | +1.7% | 425 | 423 |
| corridor4 | 1 925 | 43.7 | 62.9 | 1.44x | 1.43 | -0.9% | 364 | 370 |
| corridor1 | 5 987 | 45.7 | 65.4 | 1.43x | 1.45 | +1.3% | 452 | 446 |
| slides1 | 5 579 | 39.8 | 68.6 | *1.72x* | 1.48 | **-14.4%** | 474 | 457 |
| magistrale1 | 15 444 | 38.5 | 54.1 | 1.41x | 1.39 | -0.9% | 582 | 693 |
| outdoors1 | 25 628 | 43.0 | 62.5 | 1.45x | 1.45 | -0.4% | 936 | 702 |
| **total** | **62 240** | **41.2** | **59.4** | **1.44x** | | | | |

Read the `speedup` and `CPU async` columns together. Total CPU work divided by
wall clock *is* the number of cores kept busy, so if the work is unchanged the
speedup must equal the CPU ratio -- and on seven of the eight it does, to within
0.02. That is the cleanest available statement that the entire gain is overlap
and none of it is a measurement artifact.

**slides1 is the exception and is not a 1.72x.** Its `user` time *fell* 14.4%,
and its OOS features used fell from 3 974 to 2 313: the chaotic td divergence of
§6 sent that run down a genuinely cheaper trajectory. The overlap there is the
CPU ratio, 1.48x; the rest is a different amount of work. This is the one place
where the divergence mattered for anything, and it mattered for the *timing*
rather than the accuracy.

Two smaller things in the table:

* **The async arm processes one more frame** on room1, room6, corridor1,
  magistrale1 and outdoors1 (e.g. 2819 vs 2818). That is `Wait()` draining the
  10-message reordering window at the end of the run. The synchronous path has no
  equivalent, so it has always discarded the tail of the window -- about one
  frame. Pre-existing, harmless, and now visible.
* **RSS is not the queue.** magistrale1 +111 MB and outdoors1 **-234 MB** swamp
  the ~5 MB the 16-deep queue can hold. Those two are the sequences with the
  known ~350 MB / 1.5 s transient inside the OOS update, where peak RSS is set by
  whether a burst happens to land; see `full-tumvi.md` §5. On the six sequences
  without it, the async cost is between -6 and +6 MB.

Room4 is the sequence §2 predicted a ceiling for, so it is the one to check. Two
independent A/B repeats:

```
ARM sync  room4 rc=0 frames=2226 wall=55.6 fps=40.0 cpu=1.00 rss_mb=426
ARM async room4 rc=0 frames=2226 wall=40.4 fps=55.1 cpu=1.38 rss_mb=430   -> 1.38x
ARM sync  room4 rc=0 frames=2226 wall=55.8 fps=39.9 cpu=1.00 rss_mb=424
ARM async room4 rc=0 frames=2226 wall=40.9 fps=54.4 cpu=1.37 rss_mb=424   -> 1.36x
```

**1.36-1.38x against a predicted 1.37x.** The overlap is essentially perfect;
there is nothing left on the table. The CPU ratio of 1.37-1.38 against the sync
arm's 1.00 is the direct evidence that two threads ran concurrently.

Memory cost is **+4 MB** at the default depth of 16. §5.

The gain is not uniform, and the pattern is the one §2 predicts. The corridors
gain most (1.43-1.44x) and the rooms least (1.32-1.36x), because PNG decode costs
the same everywhere while the filter does not: corridor1 runs at 45.7 FPS
synchronous against room6's 38.1, so decode is a *larger share* of a corridor
frame and Amdahl gives back more. The sequence where async helps least is the
sequence where the estimator is busiest -- which is the opposite of what you would
want from a throughput knob.

## 5. Queue depth is not a tuning knob

The limit must exceed the reordering window (`InternalBuffer::MAX_SIZE` = 10) or
producer and worker deadlock: the producer would block at a size the worker
refuses to pop from. That is asserted at construction. Swept above the floor on
room4 (frozen-td config, so the arms are numerically identical and only speed
varies):

| depth | FPS | vs sync | RSS (MB) | stereo matches |
| --- | --- | --- | --- | --- |
| sync | 40.1 | 1.00x | 433 | 262 489 |
| 11 | 53.0 | 1.32x | 435 | 262 489 |
| 16 | 54.3 | 1.35x | 434 | 262 489 |
| 24 | 54.7 | 1.36x | 437 | 262 489 |
| 48 | 53.2 | 1.33x | 435 | 262 489 |
| 128 | 54.6 | 1.36x | 451 | 262 489 |
| 512 | 54.1 | 1.35x | 504 | 262 489 |

The 53.0-54.7 spread is run-to-run noise. **The floor already captures the whole
gain**, which is not obvious until you notice where the producer blocks: it
decodes both PNGs and *then* calls the push, so one free slot is enough to have
already overlapped a full frame of decode with the worker. Queue depth buys
jitter absorption, and TUM-VI played from disk has no jitter to absorb.

Depth is therefore a pure cost above the floor: +70 MB at 512, each queued image
pair being two 512x512 `cv::Mat`s. Default set to 16 -- a little slack over the
floor, ~1 frame of queue.

## 6. Does it change the answer?

Two separate questions, and conflating them is what made this section take three
attempts: does the *filter* execute differently, and does the *harness observe*
it differently.

First, the precondition: collapsing the twelve push sites into one `Push()`
touched the synchronous path, so that path has to be shown unchanged before
anything else means anything. room4 eval, `tumvi_stereo_oos`, `xivo` at `auto`
against `xivo-async` at `auto-async`: trajectories **byte-identical**. All the
synchronous baselines quoted here are therefore comparable to `full-tumvi.md`.

The filter does not, and this is provable rather than arguable. Freeze the online
temporal calibration (`cfg/tumvi_stereo_oos_frozentd.json`, `P.td: 0`, and there
is no td process noise, so td stays exactly 0) and the two arms agree exactly:

| | room4, frozen td | sync | async |
| --- | --- | --- | --- |
| stereo matches | | 262 489 | 262 489 |
| OOS features used | | 2 340 | 2 340 |
| FPS | | 40.1 | 53.3 |

and in eval mode the dumped trajectories are **byte-identical** on room1, room4
and room6, with ATE and RPE agreeing to all six printed digits. Same messages,
same order, same arithmetic; only the wall clock differs.

With online td on -- the shipped configuration -- they cannot be identical, and
this is structural rather than a bug to be fixed. Under
`USE_ONLINE_TEMPORAL_CALIB` the *producer* reads `X_.td` to place a message in
the timestamp-ordered buffer, because that is what the ordering is *for*. But the
producer pushes ten IMU messages per frame before the worker has reached the
previous image, so the td it reads has not absorbed that image's update. Removing
the divergence would mean synchronizing before every push, which is synchronous
mode. Given XIVO's known sensitivity -- a 1e-11 relative perturbation moves the
6-room mean ATE by 0.013 -- a microsecond-level td difference is enough to select
a different set of surviving features.

So the question becomes statistical, and the answer is an 8-member ensemble
(`run_ensemble_bugfix.sh`, perturbing `X.Vsb` by k*1e-6 m/s), room1-6, 48 runs
per arm:

| metric | sync | async | delta |
| --- | --- | --- | --- |
| ATE | 0.0455 +/- 0.0025 | 0.0459 +/- 0.0015 | +0.2 sd |
| RPE_rot | 0.6215 +/- 0.0006 | 0.6213 +/- 0.0005 | -0.3 sd |
| RPE_tra | 0.0133 +/- 0.0002 | 0.0133 +/- 0.0001 | 0.0 sd |
| RPE_rot_i | 0.5131 +/- 0.0008 | 0.5130 +/- 0.0004 | -0.1 sd |
| RPE_tra_i | 0.0126 +/- 0.0002 | 0.0126 +/- 0.0001 | 0.0 sd |

Per-sequence ATE agrees within one sd on all six (sync / async): room1 0.0571 /
0.0595, room2 0.0428 / 0.0442, room3 0.0484 / 0.0446, room4 0.0338 / 0.0342,
room5 0.0635 / 0.0649, room6 0.0273 / 0.0275.

**`async_run` is accuracy-neutral.** Not "no measurable difference on one run" --
the difference is smaller than the ensemble spread of either arm.

### 6.1 The first async ensemble said otherwise, and was wrong

Before `WaitWindow` existed, that same comparison came out:

| metric | sync | async (broken) |
| --- | --- | --- |
| ATE | 0.0455, 6 seqs | 0.0305, **4 seqs** -- room1 and room6 FAIL |
| RPE_rot | 0.6215 +/- 0.0006 | 0.6952 +/- 0.0007 (**+11.9%**, 100+ sd) |
| RPE_tra_i | 0.0126 +/- 0.0002 | 0.0129 +/- 0.0001 (+2.4%) |

None of it was the filter. `EvalModeSaver.onVisionUpdate` reads the estimator
through two separate calls, `estimator.now()` then `estimator.gsb()`
(`scripts/savers.py`). Under `async_run` both race the worker, so the pose and
the timestamp written to the trajectory can come from *different instants*, and
which instant gets sampled at all depends on how far the worker happened to have
got. `--max_difference 0.001` then found **zero** associable pairs on room1 and
room6 and the ATE tool reported failure.

`Estimator::WaitWindow()` fixes it: block until nothing is executing and the
buffer is back down to the reordering window. Deliberately not `Wait()`, which
drains the window too and thereby changes the order messages execute in -- it
would trade one artifact for a worse one. `scripts/pyxivo.py` calls it before
each poll, in every mode but `runOnly`.

The cost is real and is the right trade: an eval-mode async run gives up the
overlap and runs at synchronous speed. A trajectory that is going to be scored
should not be sampled by a race. It is also why §4 and §5 use `-mode runOnly`,
where nothing polls.

The general lesson: **adding a thread invalidated the measurement harness before
it invalidated anything about the algorithm**, and the harness failed loudly on
two sequences and quietly on four. Had I only had the four, I would have reported
a 2.4% RPE regression that does not exist.

### 6.2 The 0.001-window ATE is decided by one phase offset

Chasing those two FAILs turned up a sharper version of something already known
about this evaluator. Distance from each estimate timestamp to the nearest mocap
sample (the grid is 8.4 ms):

| sequence | arm | median offset | pairs within 1 ms |
| --- | --- | --- | --- |
| room1 | sync | 1.97 ms | 720 of 2818 |
| room1 | async (racing) | 3.21 ms | **0** |
| room4 | sync | 2.23 ms | 476 of 2225 |
| room4 | async (racing) | 0.90 ms | 1174 of 2225 |
| room6 | sync | 2.13 ms | 644 of 2633 |
| room6 | async (racing) | 3.38 ms | **0** |

The estimate timestamps are quasi-periodic at the 20 ms image rate against an
8.4 ms mocap grid, so association at a 1 ms window is settled by a single phase
offset that is *constant for the whole run*. Any change that shifts that phase
flips the metric between "scores a quarter of the frames" and "scores none" --
and it moves the reported ATE by more than the effect under test, as the fake
0.0305 above shows. Use `ATE_02` and the interpolated RPE.

A related detail, useful when reading any XIVO trajectory dump: in synchronous
mode `estimator.now()` at poll time sits a median 2.2-2.5 ms *before* the image
just fed, not on it. That is the reordering window -- the message that executes
during the image push is the oldest of the 11 buffered, which is the first IMU
sample after the *previous* image.

## 7. What this means for the efficiency work

* Nothing in `full-tumvi.md` needs restating. Those runs were single-threaded and
  the FPS numbers there are single-core numbers.
* `async_run: true` is a legitimate 1.33-1.38x on a file-fed benchmark, at +4 MB
  and zero accuracy cost, now that it drains and applies backpressure. It is not
  a milestone in the FPS-at-fixed-capacity task, because it buys wall clock
  without making the estimator cheaper: the 17.8 ms of `visual-meas` is untouched,
  and on a live camera the win goes away with the decode.
* The honest headline: **XIVO's estimator is 40 FPS on one core; the harness can
  be made to feed it at 55 by decoding on a second.**
* Left undone: the same overlap would come from decoding PNGs in a Python thread
  pool with no C++ change at all, which is worth knowing before anyone treats
  `async_run` as the fix for a slow benchmark.

## 8. Reproduce

From the workspace root, in the `xivo-async` worktree (`auto-async`).

```bash
H=notes-n-prompts/notes-efficiency/harness

# throughput A/B, one sequence, runOnly both arms
OUT_DIR=/tmp/async_ab $H/async_ab.sh room4 tumvi_stereo_oos xivo-async

# the eight-sequence sweep of §4
for s in room1 room4 room6 corridor4 corridor1 slides1 magistrale1 outdoors1; do
  OUT_DIR=/tmp/async_sweep $H/async_ab.sh $s tumvi_stereo_oos xivo-async
done

# queue-depth sweep of §5 (frozen td, so only speed varies)
$H/async_depth_sweep.sh room4 tumvi_stereo_oos_frozentd_async 11 16 24 48 128 512

# byte-identity of §6
export OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 XIVO_EIGEN_INIT=none XIVO_WT=xivo-async
./run_eval_bugfix.sh tumvi_stereo_oos_frozentd       /tmp/frz_sync  room1 room4 room6
./run_eval_bugfix.sh tumvi_stereo_oos_frozentd_async /tmp/frz_async room1 room4 room6
for s in room1 room4 room6; do
  cmp /tmp/frz_sync/tumvi_${s}_cam0 /tmp/frz_async/tumvi_${s}_cam0 && echo "$s identical"
done

# the ensembles of §6
./run_ensemble_bugfix.sh tumvi_stereo_oos       /tmp/ens_sync  8 room1 room2 room3 room4 room5 room6
./run_ensemble_bugfix.sh tumvi_stereo_oos_async /tmp/ens_async 8 room1 room2 room3 room4 room5 room6
```

Results in `results/efficiency-async/`.
