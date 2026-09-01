# M0 — harness, baseline, and the occupancy census

Commit: `M0: FPS harness configs and an EKF occupancy census`.

## What is being measured

FPS here is **frames of the sequence divided by wall-clock seconds of
`pyxivo.py -mode runOnly`**, i.e. PNG decode + the Python feed loop + the whole
estimator, with nothing written out. That is the number a user of the package
experiences, and it is the number the requirements ask to improve.

Two configs, held fixed for the whole of this work:

| | file | EKF state | tracker | differs from shipped |
| --- | --- | --- | --- | --- |
| monocular + IMU | `cfg/eff_mono.json` | 90 feat / 45 grp | 135–180 | `print_timing: true` |
| stereo + IMU | `cfg/eff_stereo.json` | 90 feat / 45 grp | 135–180 | `print_timing: true` |

They are otherwise byte-identical to `cfg/tumvi_mono_ctl.json` and
`cfg/tumvi_stereo.json`, and differ from *each other* in exactly three keys
(`stereo`, `stereo_init.enable`, `stereo_update.enable`), so they are a
controlled pair rather than two independently tuned setups.

### Why pinned to one thread

`harness/fps_one.sh` exports `OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` and `XIVO_RANDOM_SEED=0`, and runs
under `setarch -R`. Unpinned, one `pyxivo.py` spawns ~255 OpenCV/OpenMP threads
to buy 6% of wall clock (`notes-stereo/cost-and-throughput.md`), so a *batch* of
unpinned runs measures memory bandwidth contention, not the estimator. The
pinned output is bit-identical to the unpinned output, so nothing about the
computation is being changed by the pinning — only the noise.

`harness/fps_batch.sh` runs everything strictly sequentially and **interleaves
the arms inside each repeat**, so load drift on this shared host (load average
has been seen between 0.8 and 141) hits every arm equally. Absolute FPS is
machine- and load-dependent; every claim in this work is a ratio *within one
batch*.

`harness/tab.py` medians the repeats and reports FPS and a speedup column.
Its frame counts are `room1 2821, room2 2571, room3 2593, room4 2521,
room5 2439, room6 2636`.

## Baseline

`sweeps/m0_baseline.log`, one repeat, room1 and room6, loadavg 0.84:

| arm | seq | wall (s) | FPS | ms/frame |
| --- | --- | --- | --- | --- |
| mono | room1 | 133.73 | 21.10 | 47.4 |
| mono | room6 | 125.69 | 20.97 | 47.7 |
| stereo | room1 | 224.28 | 12.58 | 79.5 |
| stereo | room6 | 216.75 | 12.16 | 82.2 |

The stereo number agrees with the independent measurement in
`notes-stereo/cost-and-throughput.md` (11.3 FPS on room6, different session,
different load), so the baseline is reproducible.

Where the time goes, from the estimator's own cumulative timers (room6, ms per
frame; `propagation` is per IMU sample × 9.97 samples/frame):

```
                     mono          stereo
  actual-update      23.05  48%     35.40  43%    dense EKF covariance update
  MH-gating           9.32  20%      9.49  12%    J P J^T per in-state feature
  stereo-gating          -            8.84  11%    the same thing for the right rows
  track               3.80   8%     13.49  16%    KLT (two images in stereo)
  propagation         6.59  14%      6.64   8%    Prince-Dormand, 24x24 + 24x540
  jacobian            0.10   0%      0.18   0%
  decode + Python     4.0    8%      6.8    8%    wall - visual-meas - propagation
  ---------------------------------------------------------------
  total              47.7          82.2
```

So **68% (mono) / 66% (stereo) of the frame is dense linear algebra over `P_`**,
and the largest single item in both settings is the covariance update.

## The census

New instrumentation, printed beside the timing block every 50 frames
(`Estimator::Census`, `PrintCensus`):

```
mono   [census] frames:2800 updates:2791 feature-slots:76.27/90 group-slots:7.271/45
                occupied-dim:296.4/564 update-features:76.52 rows:153.0 (right:0   oos:0)
stereo [census] frames:2250 updates:2240 feature-slots:76.59/90 group-slots:7.284/45
                occupied-dim:297.5/564 update-features:76.94 rows:303.1 (right:149.2 oos:0)
```

Three things fall out of it, and all three are load-bearing for the rest of the
plan:

1. **The feature slots are nearly full (76 of 90) but the group slots are
   nearly empty: 7.3 of 45.** So the 564-dimensional error state carries
   `24 + 6·7.3 + 3·76.3 ≈ 296` live dimensions — **47% of the state is a
   permanently-zero block that every dense product still touches**. This is what
   justifies M5 (active-set compaction): it is worth ~3.6x on the O(m N²) terms
   and ~6.9x on the O(N³) ones, and it is *exact*, because vacated and
   never-used slots are exactly zero rows and columns of `P_` — the state
   decouples, it is not approximately decoupled.

   45 groups is not over-provisioning by accident: `kMaxGroup` bounds how far
   back an out-of-state track can reach, and the shipped config asks for 45. The
   census says the *in-state* graph never needs more than ~7-8 of them at once.

2. **Every measurement is 2 rows and there are ~76 of them per update** (153
   rows mono, 303 stereo). With 25 structurally nonzero columns per row pair
   (see M1), `H` is 153×564 with 3.5% nonzeros held densely.

3. `updates ≈ frames` (2791 of 2800): there is no "cheap frame" to exploit; the
   update runs essentially every frame, so per-frame cost *is* per-update cost.

## Cost of the instrumentation

Two `std::count` calls over 90 + 45 `bool`s per frame and eight `long`
increments per update, plus a print every 50 frames under `print_timing`. Below
the noise floor of the harness; it stays in the shipped code because these are
exactly the numbers any future capacity decision needs.

## Files

```
notes-efficiency/harness/fps_one.sh     one timed run -> a RESULT line
notes-efficiency/harness/fps_batch.sh   interleaved arms x seqs x repeats
notes-efficiency/harness/tab.py         RESULT lines -> FPS table with speedups
notes-efficiency/sweeps/m0_baseline.log the numbers above
```
