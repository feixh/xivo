# M3 — giving the OOS update something to update

## The problem M2 left behind

M2 wired a correct out-of-state measurement into the filter and it did almost
nothing: on room1, **1 of 4322 candidate features** produced a measurement. The
instrumentation added in M2 explained why:

```
views/candidate: all=4.53  instate=0.42
instate-view histogram: 0:2691 1:1318 2:277 3:29 4:6 ...
```

A dropped track had 4.5 observations on average but only 0.42 of them came from a
group that was in the EKF state at the time, and the update needs at least 3.
62 % of candidates had *zero* usable views.

The cause is structural, and it is the single most important thing to understand
about XIVO's state layout: **XIVO has no pose window.** Groups (camera poses)
enter the state only as a side effect of feature promotion —
`AddFeaturesWithInGroups` calls `AddGroupToState(f->ref())` for the reference
group of a feature being promoted, and `AddGroupOfFeatures` does the same for a
group that owns enough candidates. So the in-state group set is a sparse,
feature-driven scatter of ~12 *anchor* poses spread over the whole trajectory,
not the last N frames.

MSCKF assumes the opposite: the state holds a sliding window of recent poses, and
a track dropped now is observed by many of them. In XIVO, a track dropped now is
observed by the last few frames — almost none of which are in the state.

## What M3 does

`Estimator::MaintainOOSPoseWindow(int slots_needed)` (src/manager.cpp) plus an
augmentation hook in `UpdateStep`:

* after `graph.AddGroup(g)` for the current frame, and every `augment_every`-th
  vision frame, call `MaintainOOSPoseWindow(1)` and then `AddGroupToState(g)`;
* eviction only considers groups that are in-state, are not the gauge group, and
  that **no in-state feature refers to**. Anything else is load-bearing: a group
  that owns an in-state feature is that feature's parametrisation anchor.
* evictable groups are visited oldest-id-first and dropped by plain FIFO, using
  the same order the rest of the code uses for this
  (`FindNewOwnersForFeaturesOf` → `DiscardFeatures(nullrefs)` → mark
  `NULLREFED` → `DiscardGroup`).
* `EnforceMaxGroupLifetime` now skips in-state groups. A pose-window group owns
  no reference features, so the lifetime sweep would have removed it from the
  graph while leaving its state slot allocated — a slot leak.

Two config knobs (under `"OOS"` in the config):

| key | meaning |
| --- | --- |
| `pose_window` | number of recent poses to keep in the state; `0` disables the window entirely (M2 behaviour) |
| `augment_every` | add every k-th vision frame to the window; larger k spans more time with the same number of slots |

`EKF_MAX_GROUPS` has to grow to make room for the window on top of the anchor
groups. It is set in the **top-level** `CMakeLists.txt` — see the ODR note in
[m4-capacity-and-determinism.md](m4-capacity-and-determinism.md), this matters.

### A knob that looked good and wasn't

The first version of the eviction pass *thinned* the window (dropping every other
pose) instead of dropping the oldest, exposed as `window_stride`. Strides of 2, 3
and 4 gave byte-identical results, which is what gave it away: in steady state
only one slot needs freeing per frame, so the thinning pass always evicted
`evictable[1]` and pinned the oldest pose forever. That stale anchor cost about
0.010 m of mean ATE. Plain FIFO plus `augment_every` gives the same
"window spans more time" control and actually works.

## Result

room1, `pose_window: 20`, `min_observations: 3`:

| | window off (M2) | window on (M3) |
| --- | --- | --- |
| candidates | 4322 | 4074 |
| used | 1 | 1649 |
| views/candidate, all | 4.53 | 4.31 |
| views/candidate, in-state | 0.42 | 4.13 |
| room1 ATE | 0.1355 | 0.1048 |

So the update went from inert to carrying ~1650 marginalized constraints per
sequence, and the accuracy followed.

At this point (30 in-state feature slots, the stock capacity) the six-room mean
ATE was **0.0933** with OOS + window against **0.1101** for the same binary with
`use_OOS: false` — a 15 % improvement, and better than the 0.1209 in the
workspace README.

## The Mahalanobis gate is inert, and that is fine

`MH_thresh` values of 1.0, 2.0, 3.0 and 5.991 (per degree of freedom) all give
byte-identical results, and `gated=0` in every run. The gate is reached — it runs
on every accepted candidate — it just never fires, because the pose-window
covariance is large compared with `Roos_`, so `S = Ho P Hoᵗ + Roos I` is
dominated by the prior and `rᵗS⁻¹r/n` stays well under 1. The measurements that
survive the reprojection-error gate in `RefineOOSDepth` are already consistent
with the state; the residual gate in front of it (`max_mean_reproj_err`) is doing
the actual outlier rejection, and that one is not inert — every value other than
1.5 px was worse:

| `max_mean_reproj_err` | 0.75 | 1.0 | 1.5 | 2.5 |
| --- | --- | --- | --- | --- |
| mean ATE | 0.1001 | 0.1030 | **0.0908** | 0.0962 |

(Those numbers are from the pre-M4 build; see the M4 note for why the absolute
values moved afterwards. The ordering is what matters here.)
