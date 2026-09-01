# M4 — where the estimator's resident memory actually goes

No config key. Two changes, both pure lifetime: `Feature::ReleaseOOS()` called from
`Estimator::CleanupOOSFeatures` (M4), and then the per-feature buffer deleted
outright (M5, its own note).

## The probe

`use_OOS` is worth ~15 MB of RSS on its own, and before writing any code I wanted to
know *which part* of it. Peak RSS from `/usr/bin/time %M`, TUM-VI room5, mono,
candidate build, one core, everything else held fixed:

| variant | peak RSS (kB) | delta vs plain |
| --- | --- | --- |
| plain (`use_OOS: true`) | 94836 | — |
| `use_OOS: false` | 79956 | **−14880** |
| `OOS.min_observations: 99` (window intact, no Jacobian ever computed) | 81748 | −13088 |
| `OOS.max_observations: 5` (window intact, shorter Jacobians) | 82832 | −12004 |
| `OOS.pose_window: 5` | 84340 | −10496 |
| plain + `MALLOC_MMAP_THRESHOLD_=16384 MALLOC_TRIM_THRESHOLD_=32768` | 90452 | −4384 (and 32% slower) |

Read the second and third rows together: keeping the 20-pose window but never
computing a marginalized Jacobian recovers 13.1 of the 14.9 MB. **The pose window is
not the cost. The per-feature marginalized Jacobian is.** The glibc row says another
~4.4 MB is allocator retention rather than live data, and that buying it back with
`MALLOC_*` tunables costs a third of the throughput, so that is not the route.

Why 13 MB: `Feature::oos_` was an `OOSJacobian` *per pooled feature*, and its `Hx`
was sized `(2n-3) x kFullSize` — `(4n-3) x 564 x 8` = **257 kB** for a 15-view stereo
track, 100 kB for a typical mono one. `Feature` is pooled through
`CircBufWithHash`, whose free-slot search is circular, so a released feature keeps
its Eigen allocations until the ring wraps all the way round: with 90 in-state slots
and hundreds of pooled ones, nearly every slot ends up holding a full-width
Jacobian. `src/jac.h` already carried a comment warning that this buffer "is a member
of *every* pooled feature, so it is not something to grow lightly" — the buffer was
the problem, not its growth.

## M4, the small version

`Estimator::CleanupOOSFeatures` is the last point at which an out-of-state feature is
reachable, and its marginalized rows have been consumed by the update by then. Adding
`f->ReleaseOOS()` there returns the buffer at that point instead of at the pooled
slot's next `Reset`. Nothing reads the rows again, so it changes no arithmetic;
throughput-neutral; and worth:

| | peak RSS max over room1-6 | mean |
| --- | --- | --- |
| mono base | 102.0 MB | 93.1 MB |
| mono +M4 | 91.6 MB | 85.2 MB |
| stereo base | 153.9 MB | 128.0 MB |
| stereo +M4 | 137.8 MB | 104.9 MB |

The md5 of a room1 mono dump at 17 significant figures did not move for this change on
its own, and at the time I wrote that down as "bit-identical". It is weaker evidence
than it looks: freeing a buffer earlier changes what the next allocation gets, and M5
went on to show that this tree's output depends on its heap layout at the 4e-12 m level
(HEAD moves that far under a `MALLOC_MMAP_THRESHOLD_` change alone). So the match here
is consistent with exactness rather than proof of it, and the argument that carries the
claim is the code one: `oos_features_` is not read after this loop.

M5 then removes the per-feature buffer altogether, which subsumes this: after M5
`ReleaseOOS()` has nothing to free and only clears the row counter, which is still
needed so that a recycled slot cannot claim the previous occupant's rows.

## Two things the probe changed my mind about

* **Tuning is not the lever, and was never on the table.** `max_observations: 5` and
  `pose_window: 5` each give back less than the storage fix does, and both cost
  accuracy. The fix is to stop storing `kMaxGroup`-sized Jacobians per feature.
* **`ru_maxrss` alone is not enough to attribute anything.** Every row above is a
  separate process; within one process the peak is dominated by whichever feature
  happens to hold the longest track. The mean column in the M4 table moves much
  more than the max, which is what a lifetime fix should look like — the max is set
  by the single worst frame and only the storage fix (M5) moves it.
