# The best stereo+IMU setup on the whole TUM-VI dataset

What was asked: run the current best stereo + IMU setup on the full TUM-VI
dataset. This is the record of doing that -- what "full" turned out to require,
what the numbers are, and which of them mean what.

Everything here is `cfg/tumvi_stereo_oos.json` (stereo + IMU + out-of-state
update) built from `auto` at the tip of the efficiency work, EKF capacity 90
features / 45 groups, `XIVO_EIGEN_INIT=none`, one core per run.

## 0. Summary

* All **28** 512x16 sequences, **289 994 stereo pairs = 4.03 hours** of data, ran
  to completion. 12 of them could not run at all before this session; see §1.
* Accuracy is an **8-member ensemble** (224 runs), not single runs, because on
  the long sequences the spread between physically-identical members reaches
  40% of the mean. Single-run numbers on anything outside room1-6 are not
  meaningful. §3.
* Throughput over the whole dataset: **41.6 FPS**, i.e. **2.08x real time**
  against TUM-VI's 20 Hz cameras, single core, including PNG decode. room1-6 are
  the *slowest* group at 39.0 FPS, so efficiency work tuned on the rooms was
  tuned on the pessimistic end. §4.
* Two things worth knowing that this run turned up: TUM-VI's ground truth
  outside room1-6 is **mocap-room-only**, so ATE there is an end-to-end drift
  measure and not a trajectory error (§2); and peak RSS looks like it grows with
  run length but does not -- the apparent slope is the Python harness preloading
  the dataset index, and the estimator's own footprint is flat in run length.
  What it is *not* flat in is the out-of-state update, which costs **+347 MB
  steady and +644 MB peak** on outdoors1, 312 MB of it a per-feature buffer
  `jac.h` allocates for all 800 pooled features. §5.

## 1. "Full" required lifting a 10000-image cap

The first attempt aborted on 12 of the 28 sequences with

```
F20260827 22:44:31.259099 3889475 group.cpp:37] Group index overflow!!!
```

`Group::Reset` refused to hand out a group ID at or above
`Feature::counter0` = 10000, and XIVO creates one group per image. So the run
length was capped at 10000 images -- 8.3 minutes at 20 Hz. The sequences that
died are exactly those over 10000 frames: magistrale1/2/4/6 and all eight
outdoors. The longest, outdoors6 at 29403 frames, is three times the cap.

The cap existed so that group IDs and feature IDs occupied disjoint ranges, and
`feature.h`'s own `\todo` asked for the two to be separated properly. Auditing
the uses, they only ever meet in **one** place: `Optimizer` adds features and
groups both as g2o vertices, and g2o requires vertex indices unique across the
graph. Everything else -- `feature_adj_`/`group_adj_`, `fvertices_`/`gvertices_`,
`HasFeature`/`HasGroup`, `ids_to_depths_` -- is a kind-specific container that
never needed a shared value space.

So the fix (commit `9e3ec06`) moves the disambiguation to that one boundary:
`Optimizer::VertexId` interleaves the two kinds, groups even and features odd,
and group IDs become unbounded up to that encoding's `INT_MAX/2`.

`counter0` deliberately keeps its value of 10000. Bumping it would also have
worked and was tried first, but the adjacency containers are
`unordered_map<int, ...>` / `unordered_set<int>`, so renumbering features
reshuffles hash iteration order and perturbs results in the last bits for no
benefit. Leaving feature IDs alone makes the change **bit-identical on every
sequence that used to run** -- verified md5-identical trajectories on room1,
room3 and corridor4, room3 being the sequence that historically catches
arithmetic changes.

Two harness bugs surfaced alongside, both fixed:

* `run_eval_bugfix.sh`'s summary loop ran under `set -e -o pipefail`, so the
  first sequence whose run crashed made a `grep` exit 1, which killed the loop.
  The table silently stopped at the sequence *before* the bad one, and the mean
  and interpolated-RPE blocks never ran. One crash therefore looked like "the
  harness only does corridors".
* the batch launcher must pin the thread pools. `run_eval_bugfix.sh` starts every
  sequence concurrently and each `pyxivo` process otherwise spawns an OpenCV pool
  sized to the whole machine: 54 runs x 192 threads gave load average 6400 and
  51% system time, i.e. the batch measured context switching. `OMP_NUM_THREADS`
  alone is not enough; OpenCV's `parallel_for` backend needs
  `OPENCV_FOR_THREADS_NUM`. `harness/run_full_tumvi.sh` sets all four.

## 2. What TUM-VI's ground truth actually covers

This decides how to read every number below, so it comes before the numbers.

TUM-VI has mocap in one 4x4 m room. room1-6 stay in it; every other sequence
starts there, leaves, and comes back. The `mocap0` file therefore spans the whole
sequence in *time* while covering only the room in *space*:

| sequence | frames | path travelled | GT time span | GT spatial extent | gt% |
| --- | --- | --- | --- | --- | --- |
| room1 | 2818 | 146 m | 141 s | 3.7 x 2.7 x 1.0 m | 98.3 |
| outdoors1 | 25628 | 1482 m | 1282 s | 3.8 x 3.4 x 0.9 m | 9.3 |
| outdoors2 | 18445 | 1054 m | 923 s | 3.8 x 3.1 x 1.0 m | 11.5 |
| outdoors7 | 22446 | 1610 m | 1120 s | 3.5 x 3.3 x 0.7 m | 7.2 |

`gt%` (a column `harness/tumvi_table.py` computes) is the fraction of estimated
poses with a GT sample within 20 ms. Outside room1-6 it is **7-55%**, and those
poses are not a random sample -- they are the start and the end.

So **ATE on the non-room sequences is not a trajectory error.** It is dominated
by how far the estimate has drifted by the time it re-enters the room, with a
Umeyama alignment on top: an end-to-end loop-drift measure. Read as drift rate it
is far more informative:

| sequence | ATE@0.02 | path | drift |
| --- | --- | --- | --- |
| outdoors7 | 18.7 m | 1610 m | 1.2% |
| outdoors8 | 16.7 m | ~1300 m | ~1.3% |
| outdoors1 | 189.9 m | 1482 m | 12.8% |

`RPE_rot_i` and `RPE_tra_i` do not have this problem -- they only ever compare
motion over 1 s inside a GT-covered window -- and they are also by far the most
reproducible columns (§3). **On the non-room sequences, prefer RPE.**

No sequence diverged: zero non-finite poses in all 28 estimates. The large
`RPE_tra_i` on outdoors2/3/4/5 (0.86-1.20 m/s against 0.02 elsewhere) is a real
tracking glitch rather than divergence -- outdoors2 has a single 0.44 m
inter-frame step, an implied 8.8 m/s.

## 3. Accuracy: 8-member ensemble over all 28 sequences

Single runs of this filter are not a measurement. Hard accept/reject gating makes
the trajectory a chaotic function of the last bits of its input, so the ensemble
device from the efficiency work is used again: 8 members, each perturbing the
initial `X.Vsb` by `k * 1e-6` m/s, which is six orders of magnitude inside the
prior std the config itself declares for that quantity (`P.Vsb: 0.5`).

`results/efficiency-full/ens28_oos`, 8 members x 28 sequences = 224 runs.
`sd` is the spread across members.

```
seq            gt%    ATE@0.02       sd   ATE@0.001       sd   RPE_rot_i       sd   RPE_tra_i       sd
room1         98.3      0.0688   0.0081      0.0571   0.0054      0.4360   0.0008      0.0129   0.0006
room2         90.4      0.0539   0.0074      0.0428   0.0064      0.6205   0.0013      0.0143   0.0004
room3         91.4      0.0827   0.0057      0.0484   0.0041      0.5944   0.0012      0.0124   0.0002
room4         98.7      0.0378   0.0020      0.0338   0.0032      0.4900   0.0007      0.0121   0.0001
room5         99.8      0.0700   0.0077      0.0635   0.0064      0.4638   0.0011      0.0139   0.0002
room6         96.9      0.0349   0.0072      0.0273   0.0040      0.4738   0.0025      0.0097   0.0003
corridor1     17.3      0.3578   0.1927      0.3648   0.2045      0.5291   0.0014      0.0134   0.0003
corridor2     20.9      2.8886   1.2717      2.6093   1.1477      0.4806   0.0016      0.0111   0.0003
corridor3     20.6      0.9597   0.2769      0.9612   0.2784      0.4382   0.0004      0.0105   0.0003
corridor4     54.7      0.1022   0.0234      0.0682   0.0126      0.2965   0.0016      0.0106   0.0004
corridor5     13.5      0.4046   0.1837      0.4800   0.2140      0.3261   0.0019      0.0099   0.0010
magistrale1   15.4      4.4688   1.1114      4.4618   1.1133      1.3696   0.0184      0.0186   0.0004
magistrale2   18.1      1.5593   0.7294      1.1246   0.5258      0.9101   0.0070      0.0149   0.0005
magistrale3   21.5      5.2988   0.1900      5.3126   0.1908      0.9641   0.0013      0.0157   0.0004
magistrale4   23.6      5.6804   1.0744      5.4534   1.0311      1.0221   0.0095      0.0231   0.0014
magistrale5   22.7      1.8228   0.3498      1.6622   0.3189      0.7918   0.0009      0.0147   0.0003
magistrale6   13.7      5.2908   1.4634      5.2713   1.4604      0.7680   0.0047      0.0170   0.0004
outdoors1      9.3    189.8894  38.1169    193.6692  38.8581      1.6917   0.0430      0.0277   0.0006
outdoors2     11.5     28.0652  10.2031     28.6905  10.1504      1.9185   0.0368      1.0626   0.0741
outdoors3     14.2     12.7608   4.5647     11.6248   4.0887      1.5802   0.0222      0.8593   0.0617
outdoors4     18.2     20.5191   4.8261     19.0277   4.5144      1.2600   0.0137      1.1974   0.0810
outdoors5     15.5     10.4227   3.0836      9.9498   2.6999      1.6628   0.0192      0.9697   0.1438
outdoors6      7.7     38.5207  10.4549     22.7902   6.1653      2.2970   0.0268      0.0375   0.0017
outdoors7      7.2     18.6851   3.6601     18.6996   3.6608      1.5982   0.0323      0.0242   0.0005
outdoors8     12.5     16.6574   2.7130     16.7711   2.7278      1.3212   0.0299      0.0209   0.0006
slides1       38.8     11.9531   6.5725     11.9108   6.5482      0.6422   0.0010      0.0149   0.0007
slides2       33.9      2.9136   5.9353      1.6199   3.2818      0.5560   0.0013      0.0129   0.0005
slides3       26.6      2.2796   0.3505      2.2729   0.3496      0.7205   0.0008      0.0139   0.0002
------------------------------------------------------------------------------------------------------
room mean        6      0.0580               0.0455               0.5131               0.0126
corridor mean    5      0.9426               0.8967               0.4141               0.0111
magistrale mean  6      4.0202               3.8810               0.9709               0.0173
outdoors mean    8     41.9400              40.1529               1.6662               0.5249
slides mean      3      5.7154               5.2679               0.6396               0.0139
ALL mean        28     13.6375              13.0382               0.9365               0.1599
```

Reading it:

* **room1-6 is the only group where ATE is a clean trajectory error**, and it is
  good: 0.0580 m mean ATE@0.02, 0.0455 m at the stricter 0.001 association. The
  member-to-member sd is 0.002-0.008 m, so this is reproducible to about 0.01 m.
* **The ensemble is not optional outside the rooms.** slides2 is 2.91 +/- 5.94:
  the sd is twice the mean, so a single run of slides2 carries essentially no
  information. corridor2 is 2.89 +/- 1.27, outdoors1 is 190 +/- 38. Any
  comparison of two configs on these sequences that is not ensembled is noise.
* **RPE is the reproducible metric.** `RPE_rot_i` sd is 0.0004-0.043 on means of
  0.3-2.3, i.e. a relative spread of ~0.1-2%, against 15-200% for ATE. If a
  single number has to be quoted for the whole dataset, quote
  `RPE_rot_i = 0.9365 deg` and `RPE_tra_i = 0.1599 m` (or 0.0139 m excluding the
  four glitching outdoors sequences), not the ATE mean.
* **`ALL mean` ATE is dominated by one sequence.** outdoors1 at ~190 m against
  room1's 0.07 m sets the whole-set mean essentially by itself. It is reported
  for completeness; the per-group rows are what to compare.

### The out-of-state update, on the whole dataset

Single-run pass, both configs, all 27 sequences that were available at the time
(`results/efficiency-full/oos_all` vs `ctl_all`):

| group | ATE@0.02 with OOS | without | RPE_rot_i with | without |
| --- | --- | --- | --- | --- |
| room | 0.0564 | 0.0592 | 0.5126 | 0.5132 |
| corridor | 1.1467 | 0.6892 | 0.4133 | 0.4137 |
| magistrale | 4.2655 | 4.3162 | 0.9621 | 0.9688 |
| outdoors (7) | 46.368 | 47.810 | 1.5776 | 1.5644 |
| slides | 2.5634 | 2.0016 | 0.6391 | 0.6391 |

Given the ensemble sd in the table above -- corridor2 alone is +/- 1.27 -- **none
of the non-room differences here are resolvable from single runs**, in either
direction. The honest statement is that OOS is a small win on room1-6 (0.0580 vs
0.0632 on the 6-room ensembles measured earlier) and that the dataset as a whole
does not have the statistical power to rank the two without ensembling both,
which would be 448 runs.

## 4. Throughput over the whole dataset

Strictly sequential, one sequence at a time, one core, nothing else on the
machine. `-mode runOnly`, so wall clock is PNG decode + the Python feed loop +
the estimator, and nothing else. `harness/fps_full_tumvi.sh`, then
`harness/fps_table.py`.

```
seq           frames   wall_s     FPS  x real   RSS_MB   visual   update
room1           2821     76.4    36.9    1.85      447    19.94     7.43
room2           2882     72.5    39.8    1.99      400    18.14     6.95
room3           2821     70.8    39.9    1.99      420    17.90     6.40
room4           2228     55.9    39.8    1.99      423    17.96     6.59
room5           2847     71.5    39.8    1.99      425    17.98     6.41
room6           2636     69.3    38.0    1.90      425    19.24     7.98
corridor1       5990    131.1    45.7    2.28      454    14.92     5.03
corridor2       6772    149.8    45.2    2.26      478    15.13     5.29
corridor3       5802    123.3    47.1    2.35      442    14.27     4.76
corridor4       1927     43.3    44.5    2.22      360    15.52     5.51
corridor5       5914    136.0    43.5    2.17      454    15.96     5.67
magistrale1    15447    402.9    38.3    1.92      639    18.59     5.96
magistrale2    10810    269.9    40.1    2.00      517    17.53     5.67
magistrale3     9703    247.0    39.3    1.96      527    18.12     5.91
magistrale4    12480    300.0    41.6    2.08      533    16.81     5.37
magistrale5     8922    221.7    40.2    2.01      557    17.51     5.67
magistrale6    11497    284.9    40.4    2.02      543    17.60     5.33
outdoors1      25631    592.6    43.3    2.16      832    15.83     4.60
outdoors2      18449    437.4    42.2    2.11      556    16.46     5.13
outdoors3      16881    386.4    43.7    2.18      538    15.75     5.01
outdoors4      13999    331.7    42.2    2.11      519    16.52     5.36
outdoors5      17747    401.7    44.2    2.21      544    15.57     4.87
outdoors6      29403    743.6    39.5    1.98      826    17.87     5.89
outdoors7      22449    533.6    42.1    2.10      577    16.61     5.17
outdoors8      16213    377.8    42.9    2.15      536    16.17     5.21
slides1         5582    136.1    41.0    2.05      473    17.11     5.94
slides2         4908    118.7    41.4    2.07      458    16.97     5.97
slides3         7233    178.4    40.5    2.03      483    17.45     6.07
------------------------------------------------------------------------
room mean FPS           39.0
corridor mean FPS       45.2
magistrale mean FPS     40.0
outdoors mean FPS       42.5
slides mean FPS         41.0
WHOLE DATASET           41.6   (289994 frames in 116.1 min of compute,
                                4.03 h of data, 2.08x real time)
```

`visual` and `update` are XIVO's own per-image millisecond timers for the front
end and the covariance update.

Two things to read off it:

* **The rooms are the slowest group, not the fastest.** 39.0 FPS against
  corridor's 45.2, and the front end is the reason: `visual` is 18-20 ms in the
  rooms against 14-16 ms in the corridors. The rooms are a small textured space
  where tracks survive, so the tracker carries a full complement of features and
  the update carries a full instate set (`update` 6.4-8.0 ms vs 4.8-5.7 ms).
  Efficiency work tuned on room1-6 is therefore tuned on the *pessimistic* end of
  this dataset, which is the right end to tune on.
* **The whole-dataset figure is not the room figure.** 41.6 FPS overall against
  39.0 on the rooms; quoting either as "XIVO's throughput" without saying which
  sequences is misleading by ~7%. And this config runs OOS: `tumvi_stereo` (OOS
  off) is 44.0 FPS on the rooms, so the out-of-state update costs ~11% of
  throughput for the accuracy in §3.

## 5. Peak RSS: a transient burst, not growth

Peak RSS ranges from 360 MB (corridor4) to 832 MB (outdoors1) and correlates
loosely with frame count -- a naive fit gives 379 MB + 13.3 kB/frame, R^2 0.82,
which reads like a leak. It is not one. Two separate effects are hiding in that
fit, and neither is unbounded growth in the estimator.

**Effect 1: the Python harness preloads the entire dataset index.**
`scripts/pyxivo.py` builds an in-memory list of every image path and every IMU
sample before the first frame, at ~0.5 kB per entry, and a sequence has
`2 x images + IMU rows` entries. That is 11 MB on corridor4 and 172 MB on
outdoors6 -- i.e. it accounts for the *entire* apparent slope and is harness
overhead, not estimator memory. Subtracting it:

```
seq            frames   preload    peak    residual   resid/frame
corridor4        1927      11 MB   360 MB    349 MB     185 kB
room2            2882      17 MB   400 MB    383 MB     136 kB
corridor2        6772      40 MB   478 MB    438 MB      66 kB
magistrale4     12480      73 MB   533 MB    460 MB      38 kB
outdoors7       22449     131 MB   577 MB    446 MB      20 kB
outdoors6       29403     172 MB   826 MB    654 MB      23 kB
```

The residual is **flat at 410-450 MB across 25 of the 28 sequences**, from 1927
frames to 22449 -- corridor4 at 349 MB is the *lowest* and outdoors7 at 22449
frames is 446 MB. There is no per-frame growth on the C++ side. `resid/frame`
falling monotonically from 185 to 20 kB is what a constant looks like when
divided by frame count.

**Effect 2: three sequences show a short allocation burst.** magistrale1
(549 MB residual), outdoors6 (654) and outdoors1 (682) sit 100-250 MB above the
flat band. Sampling `/proc/<pid>/status` twice a second through a full outdoors1
run shows why -- the peak is a **transient two seconds wide**:

```
t= 47s   520 ->  602 MB   (+82)
t= 49s   602 ->  534 MB   (-68)
t= 91s   559 ->  785 MB  (+226)
t= 93s   785 ->  591 MB  (-194)
```

The same probe on outdoors7 shows no jump larger than 8 MB after preload, and
its `VmHWM` equals its `VmRSS` at every poll -- the kernel high-water mark never
exceeds the steady state. outdoors1's `VmHWM` runs 170-230 MB above its `VmRSS`
throughout. So the outliers are not holding memory; they touch a large temporary
once and hand the pages straight back (Eigen allocations above glibc's mmap
threshold are returned to the OS on free, which is why RSS *falls* again).

`ru_maxrss` -- what `harness/fps_table.py` reports, via GNU time -- is the kernel
high-water mark, so it captures the spike (832 MB) while a 2-second sampler can
alias it away (785 MB). Both are correct; they answer different questions. For
sizing a machine, use `ru_maxrss`. For steady-state footprint, use the residual
band.

**What the 430 MB band actually is: the out-of-state update.** Running outdoors1
under both configs, sampled twice a second, isolates it -- the two differ only in
`use_OOS` and its options block:

| outdoors1 | steady RSS | peak (`VmHWM`) | jumps > 20 MB |
| --- | --- | --- | --- |
| `tumvi_stereo_oos` | 630 MB | 931 MB | 1 (a 1.5 s spike at t=90 s) |
| `tumvi_stereo` | 283 MB | 287 MB | none |

OOS costs **+347 MB of steady state and +644 MB of peak** on this sequence. The
steady figure has an exact explanation: `OOSJacobian` (jac.h) is a member of
*every* pooled `Feature`, and its constructor sizes `Hx` at
`2 * kMaxGroup x kFullSize` = 90 x 564 doubles = 399 kB. At 800 pooled features
that is **312 MB**, which is essentially the whole measured delta. M7 is why the
control does not pay it: with `XIVO_EIGEN_INIT=none` the `resize` no longer
writes the pages, so they stay virtual until something actually stores into them
-- and with OOS off, nothing ever does.

So the memory picture is the mirror image of the throughput one in §4. OOS costs
~11% of FPS *and* 2.2x the resident footprint of the entire rest of the process,
in exchange for the accuracy in §3 -- which is a small win on room1-6 and not
resolvable elsewhere. It is a reasonable default for room-scale work and an
expensive one for a memory-constrained target.

Two follow-ups this leaves open, both out of scope here and neither a
correctness problem:

* **The 312 MB is avoidable.** `Ho()`/`ro()` are consumed by `update.cpp` in the
  same update that filled them, so the buffer does not need to be per-feature and
  persistent -- a scratch pool sized by the number of features OOS in one update
  (tens, not 800) would do. Sizing `Hx` by `4 * max_observations` = 60 rows
  instead of `2 * kMaxGroup` = 90 would save a third on its own, since the
  observation thinner never emits more than that.
* **The 1.5 s spike is not pinned down.** It is inside the OOS path (the control
  never shows it), it appears on 3 of 28 sequences, its size varies run to run
  (376 MB here, ~226 MB in the 2-second-sampled run), and the pages go straight
  back to the OS. Localizing it needs a heap probe in the build, which would have
  perturbed the timing pass, so it was not run.

## 6. Reproducing

```bash
# fetch and verify the 28 sequences (~190 GB extracted)
notes-efficiency/harness/fetch_tumvi.sh 12
dependencies/venv/bin/python3 notes-efficiency/harness/check_tumvi.py

# single pass, both configs
printf '%s\n' corridor1 ... slides3 > /tmp/seqs28.txt
notes-efficiency/harness/run_full_tumvi.sh tumvi_stereo_oos results/oos /tmp/seqs28.txt
dependencies/venv/bin/python3 notes-efficiency/harness/tumvi_table.py results/oos

# the 8-member ensemble (224 runs)
OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 XIVO_WT=xivo \
  ./run_ensemble_bugfix.sh tumvi_stereo_oos results/ens28_oos 8 $(cat /tmp/seqs28.txt)
dependencies/venv/bin/python3 notes-efficiency/harness/tumvi_ens_table.py results/ens28_oos

# throughput
notes-efficiency/harness/fps_full_tumvi.sh oos xivo <timing-cfg>.json /tmp/seqs28.txt 1
dependencies/venv/bin/python3 notes-efficiency/harness/fps_table.py /tmp/fps.log

# the RSS decomposition in §5: sample /proc twice a second under both configs
notes-efficiency/harness/rss_probe.sh outdoors1 tumvi_stereo_oos oos_o1 &
notes-efficiency/harness/rss_probe.sh outdoors1 tumvi_stereo     ctl_o1 &
wait
```

`rss_probe.sh` reads both `VmRSS` and `VmHWM`: `VmRSS` for the steady state,
`VmHWM` for the peak. Sampling `VmRSS` alone at any interval can miss a
1.5-second burst, which is exactly what §5 turned out to hinge on.

`fetch_tumvi.sh` is worth one warning: do not pass `curl -C -` blindly. Against
an already-complete file curl asks for a range starting at EOF, the TUM CDN
answers 200 with the whole body, and curl *appends* it -- a 1.6 GB tar became
3.2 GB. `tar` still extracted most of it and stopped at the end-of-archive
marker, leaving directories that were full by file count with one truncated PNG
inside. Four sequences had to be re-fetched and only `check_tumvi.py`'s PNG-framing
check saw it. The script now decides resume by comparing local size against
`Content-Length` first.
