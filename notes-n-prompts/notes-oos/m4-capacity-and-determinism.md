# M4 — tuning, and two bugs that made tuning meaningless

Two things had to be fixed before any sweep could be believed. Both were found by
noticing that runs which *should* have been identical weren't.

## Bug 1: the Python bindings disagreed with the library about the state layout

M3 raised `EKF_MAX_GROUPS` from 15 to 40 to make room for the pose window, and
put the define in `src/CMakeLists.txt`. That is wrong: `add_definitions` applies
to the current directory and the subdirectories added *after* it, and `pyxivo` is
built from the **top-level** `CMakeLists.txt`, which is not a subdirectory of
`src/`. So `pybind11/pyxivo.cpp` compiled against `kMaxGroup = 15` while
`libxest` compiled against 40.

This is not benign. `Estimator` has

```cpp
std::array<bool, kMaxGroup> gsel_;
std::array<bool, kMaxFeature> fsel_;
```

as *members*, so `sizeof(Estimator)` and the offset of every member declared
after them differ between the two translation units. Any accessor that is inlined
from `estimator.h` into `pyxivo.cpp` — which includes the ones the evaluation
scripts use to read out the trajectory — then reads at the wrong offset. Every
number measured between the M3 sweeps and this fix is unreliable.

Fixed by moving both `EKF_MAX_*` defines to the top-level `CMakeLists.txt`, with
a comment saying why they must live there.

Note also that `EKF_MAX_GROUPS` is **not** result-neutral even with the pose
window off, which I initially assumed and was wrong about: the count of free
group slots decides which branch of `SelectAndAddNewFeatures` runs and how many
groups `AddGroupOfFeatures` admits. So the OOS-off baseline has to be re-measured
with the same binary rather than quoted from an earlier build.

## Bug 2: the estimate was not reproducible

Even after the ODR fix, two runs of a byte-identical config on the same binary
gave room2 ATE 0.0755 and 0.0933. Nothing is threaded (`async_run: false`) and
`XIVO_RANDOM_SEED=0` seeds the RNG, so the remaining candidate was iteration
order over pointer-keyed containers — which depends on the heap addresses of the
memory-manager slots, and therefore on ASLR. Two sites mattered:

* `DiscardAffectedGroups` iterates `affected_groups_`
  (`unordered_set<GroupPtr>`). Discarding a group re-homes its features and
  decides which become `NULLREFED`, so the visit order feeds back into the
  estimate. Now visited by group id.
* the collinearity check in `FindNewGaugeFeatures` builds its point list by
  iterating a pointer-keyed gauge set, and `PointsAreCollinear` evaluates cross
  products in that order, so a decision at the `1e-3` threshold could flip. Now
  sorted by feature id.

Verified with six replicates in separate processes under heavy load
(byte-identical) plus a seventh in a later batch (also identical). Re-verified at
the end of the work: a fresh six-room run of the delivered `cfg/oos.json`
reproduces `results/oos/m4-final/oos/summary.txt` to all six printed digits on
every sequence, half an hour later and under a different load.

### Seed sensitivity

`XIVO_RANDOM_SEED` still feeds one real decision — the `std::shuffle` in the
collinearity retry loop of `FindNewGaugeFeatures`. Seeds 0/1/2 on the M4 config
(`results/oos/m4-seeds`; the same check on the final M5 config is in
[m5-depth-cap-and-dead-knobs.md](m5-depth-cap-and-dead-knobs.md)):

| seed | room1 | room2 | room3 | room4 | room5 | room6 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.100703 | 0.079665 | 0.071957 | 0.050367 | 0.076299 | 0.027452 | 0.0677 |
| 1 | 0.100703 | 0.079665 | 0.070612 | 0.050367 | 0.076299 | 0.027452 | 0.0675 |
| 2 | 0.100703 | 0.079665 | 0.072597 | 0.050367 | 0.076299 | 0.027452 | 0.0678 |

Only room3 moves at all — the retry loop only runs when the first pick comes out
collinear, which happens on that sequence and not the others. Spread of the mean
is 0.0003, so the reported figure is not a lucky seed.

Why this mattered so much: the nondeterminism was worth up to **0.018 m of ATE on
a single sequence and ~0.003 m on the six-room mean** — the same magnitude as the
differences the sweeps were trying to resolve. Several "improvements" measured
before the fix were noise.

## A note on which binary each table below was measured with

The sweeps in this file were run before `USE_ONLINE_TEMPORAL_CALIB` was turned on,
which was the last change to the delivered configuration. That define changes
`kMotionSize` and therefore the whole state layout, so it shifts every absolute
number by roughly 0.002–0.004 m of ATE (in our favour). Deltas *within* each table
are on one binary and are comparable; absolute numbers here sit slightly above the
final ones in `results/oos/m4-final/table.txt`, which is the authoritative table
and the one quoted in `../report-oos.md`. For reference, the same three
configurations on the two binaries:

| config | pre-calib (these sweeps) | final binary |
| --- | --- | --- |
| shipped `sweep_dlt_nodesc` | 0.0967 | 0.0923 |
| tuned, `use_OOS: false` | 0.0799 | 0.0781 |
| tuned, OOS + pose window | 0.0701 | **0.0677** |

## What actually moves the accuracy: in-state capacity

The stock configuration allows 30 in-state features (`kMaxFeature`) and detects
45–60 per frame. Raising both, with the memory pool raised to match:

| in-state slots | tracker features | mean ATE (OOS on) |
| --- | --- | --- |
| 30 | 45–60 | 0.0933 |
| 60 | 60–80 | 0.0875 |
| 100 | 80–110 | 0.0736 |
| 150 | 80–110 | 0.0702 |
| 200 | 100–130 | **0.0695–0.0701** |
| 200 | 110–140 | 0.0746 |
| 200 | 130–160 | 0.0787 |

`EKF_MAX_FEATURES=200`, `EKF_MAX_GROUPS=60`, tracker 100–130, memory pool
600/300 is the operating point. Beyond that the tracker starts returning features
that don't survive, and the extra state costs more than it buys.

Note the memory pool has to grow with the tracker count or the run dies with
`Out of feature slots in the memory manager` (`mm.cpp:94`, `LOG(FATAL)`).

## Everything else I swept, and what it did

All on the operating point above, six-room mean ATE, deterministic build. Nothing
here beat the base config (0.0701 on that binary):

| knob | values tried | best | verdict |
| --- | --- | --- | --- |
| `oos_meas_std` | 1.5 / 2.0 / 2.5 | 2.0 | keep |
| `OOS.min_observations` | 3 / 5 | 3 | keep |
| `OOS.max_observations` | 15 / 25 | 15 | keep |
| `OOS.max_mean_reproj_err` | 0.75 / 1.0 / 1.5 / 2.5 | 1.5 | keep |
| `OOS.MH_thresh` | 1.0 / 2.0 / 3.0 / 5.991 | any | inert, see M3 note |
| `augment_every` | 1 / 2 / 3 | 2 | keep |
| `pose_window` | 12 / 20 / 30 | 20 | keep |
| `visual_meas_std` | 1.2 / 1.5 / 1.8 / 2.0 | 1.5 | keep |
| `use_depth_opt` | on / off | off | on cost 0.04 |
| `max_group_lifetime` | 30 / 60 / 120 | 60 | keep |
| `triangulate_pre_subfilter` | on / off | on | noise |
| `use_1pt_RANSAC` | on / off | off | keep |
| `remove_outlier_counter` | 5 / 10 | 10 | keep |
| `group_degrees_fixed` | 4 / 6 | 4 | keep |
| `num_gauge_xy_features` | 1 / 2 / 3 | 3 | keep |
| `sort_gauge_features_by_cov` | on / off | off | see below |
| `mask_size` | 15 / 20 | 15 | keep |
| `Qimu` gyro scale | 1× / 2× / 3× / 6× | 3× | the shipped 3× inflation is already right |
| `Qimu` accel scale | 1× / 3× | 3× | keep |

### The gauge-covariance sort

`Graph::FindNewGaugeFeatures` picks the features whose XY it will fix out of an
*unsorted* candidate vector — the sort by covariance is commented out upstream,
and the fallback branch even says "defaulting to using those with smallest
covariance" while handing it the unsorted backup. Fixing a feature's XY asserts
that those two directions are known exactly, so it ought to be spent on the
best-known features. Implemented behind `sort_gauge_features_by_cov`, with the
feature id as a tiebreak so the comparator is a strict weak ordering. It measured
*worse* (0.0984 vs 0.0933), so it ships off by default. The reasoning still seems
right to me; I suspect the filter is benefiting from the arbitrary choice
spreading the gauge over a wider spatial extent, which the collinearity retry
loop only partially enforces.

### Dead code found on the way

`Criteria::CandidateComparison` computes `score1`/`score2` from
`comparison_score_type` and then never uses them — it compares `f1->score()`
against `f2->score()`, which is hardcoded to `-P_(2,2)`. So
`comparison_score_type` has no effect whatsoever. Left alone: changing it changes
feature-admission order, which is a tuning question of its own, and the default
`"DepthUncertainty"` happens to agree with what `score()` returns.

## Where the accuracy comes from

Final binary (`results/oos/m4-final/table.txt`), six-room mean:

| config | mean ATE | mean RPE rot |
| --- | --- | --- |
| shipped `sweep_dlt_nodesc` (stock capacity) | 0.0923 | 0.6202 |
| tuned capacity, `use_OOS: false` (`cfg/oos_off.json`) | 0.0781 | 0.6196 |
| tuned capacity, OOS + pose window (`cfg/oos.json`) | **0.0677** | 0.6194 |

M5 then found one more in-state lever (the `max_depth` promotion gate) and took
these to 0.0923 / 0.0733 / **0.0648** — see
[m5-depth-cap-and-dead-knobs.md](m5-depth-cap-and-dead-knobs.md). `cfg/oos.json` in
the tree is the M5 version; the 0.0677 row above is what it looked like at this
commit.

An attribution run (pre-calib binary, `results/oos/m4-attribution`) separating the OOS residuals from the pose window's side
effects (`min_observations: 99` keeps the window but never forms a measurement;
`pose_window: 0` keeps the residuals but not the window):

| | mean ATE |
| --- | --- |
| OOS off entirely | 0.0815 |
| OOS residuals, no pose window | 0.0776 |
| pose window, no OOS residuals | 0.0729 |
| both | 0.0695 |

So both halves contribute and they are not redundant, but a good part of the
window's benefit is *indirect*: holding recent poses in the state changes group
slot pressure and feature promotion, independently of the OOS constraints.

## The honest limitation

At the tuned operating point the OOS update is close to inert again, for the
opposite reason to M2. With 200 in-state slots, essentially every track worth
having is promoted, and what's left for OOS is short-lived junk:

```
candidates=7094  used=30  too_short=6891  bad_triangulation=97  gated=0
views/candidate: all=1.66  instate=0.91
```

1.66 observations per candidate — these are tracks that died after one or two
frames. The OOS update needs long tracks that *didn't* make it into the state,
and at high in-state capacity there aren't any. That is the fundamental tension
of the hybrid: an in-state (EKF-SLAM) feature is more informative per track than
a marginalized MSCKF constraint, so whenever you can afford the slot you should
spend it, and OOS is left with the residue.

Which means the honest reading of the table above is that OOS earns its 0.010 m
at the tuned operating point mostly through the pose window, and earns a clean
0.017 m (0.1101 → 0.0933) at the *stock* capacity where it has real tracks to
work with. Both are real; the second is the one that is actually about the OOS
update.
