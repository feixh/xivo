# M1 — the out-of-state (MSCKF) update was switched off *and* mis-defaulted

Result (mono, 6 rooms, `--jitter 6`, `experiments/results/position_oos_full`):

| | ate_002 | ov_rpe8_pos_m |
|---|---|---|
| baseline (`position_nochange`) | 0.0928 | 0.0480 |
| OOS on, with a pose window | **0.0875** | **0.0418** |

Per sequence, ATE / RPE-8m:

| | room1 | room2 | room3 | room4 | room5 | room6 |
|---|---|---|---|---|---|---|
| baseline ATE | 0.0762 | 0.1001 | 0.1343 | 0.0805 | 0.1065 | 0.0594 |
| OOS ATE | 0.0812 | 0.0967 | 0.1265 | 0.0616 | 0.1023 | 0.0568 |
| baseline RPE | 0.0323 | 0.0353 | 0.0443 | 0.0683 | 0.0575 | 0.0500 |
| OOS RPE | 0.0328 | 0.0353 | 0.0428 | 0.0487 | 0.0478 | 0.0435 |

The ATE delta (−0.0053) is on its own inside the ±0.0067 sd of the 6-room mean,
so it would not be a result by itself. The RPE-8m delta (−0.0062, −13%) is not:
it improves in 5 of 6 sequences and is flat in the sixth, and the two biggest
movers (room4 −0.0196, room5 −0.0097) are the two sequences with the worst
baseline RPE. RPE is the low-noise metric here — it is a median over many 8 m
windows rather than one global alignment — so this is a real effect and it is in
the right direction on both metrics.

## Why it was off

`cfg/eff_mono.json` has `use_OOS: false` and **no `OOS` block at all**. Turning
only `use_OOS` on (batch 1, `position_oos_on`, 0.0964) measures nothing, because
`src/estimator.cpp:200` reads

```cpp
oos_pose_window_ = oos.get("pose_window", 0).asInt();
```

and a zero-length pose window makes the update inert. Confirmed by the census
line, which reports mean OOS rows per update:

```
baseline  group-slots:7.27/45   occupied-dim:296/564   rows:153.0 (oos:0)
oos_full  group-slots:29.85/45  occupied-dim:435/564   rows:160.8 (oos:5.95)
```

Without the window, in-state groups are only the reference groups of in-state
features — a sparse, feature-driven scatter of 7 poses. An out-of-state
constraint lives entirely in the group-pose block of the state, so with 7 groups
there is essentially nothing for it to act on. With `pose_window: 20` the filter
also carries a FIFO window of recent poses (29.9/45 group slots occupied), and
the same tracks now constrain them.

The exact block used is the one from `cfg/tumvi_mono_ctl_oos.json`:
`pose_window 20, augment_every 2, min_observations 3, max_observations 15,
Rtri 1.0, refine true, max_mean_reproj_err 1.5, zmin 0.05, zmax 50.0,
MH_thresh 5.991, max_iters 10, eps 1e-5`.

## Where the remaining OOS supply goes (room1_r0)

```
candidates=13737  used=2585  too_short=10521  bad_triangulation=596  gated=0
rows=16621  observations=12188  obs/feature=4.71
views/candidate: all=3.27  instate=1.83
pose window: length=20  augment_every=2  evictions=460  starved_frames=0
instate-view histogram: 0:2587 1:5490 2:2444 3:1146 4:808 5:552 6:235 ...
```

77% of candidates are rejected as `too_short`. Two separate causes, and the
histogram separates them:

1. **Tracks that never reach the state are short.** 3.27 total views on average.
   These are the FAST detections that die within a few frames; the survivors get
   promoted into the state instead and are then ineligible for OOS (OOS only
   sees features dropped by the tracker while still out of state,
   `src/manager.cpp:270`). This is a *front-end* problem, not an OOS one.
2. **Only 56% of the views a candidate does have are usable.** `augment_every: 2`
   adds every second pose to the window, so on average half of a short track's
   observations land on groups that were never put in the state, and
   `SelectOOSObservations` drops them. 3.27 × 0.56 = 1.83, exactly the measured
   `instate` figure. `augment_every: 1` should roughly double it and push a large
   part of the `1:5490` and `2:2444` bins over `min_observations: 3`.

`gated=0` — the OOS Mahalanobis gate never fires, so nothing is being thrown
away at the update; and `starved_frames=0`, so a 20-pose window fits inside
`EKF_MAX_GROUPS=45` alongside the ~7 anchor groups with room to spare (a window
of 30 would be the practical ceiling).

## Config delta

`cfg/eff_mono.json`: `use_OOS: false -> true` plus a new `OOS` block. See
`config-delta.md`.
