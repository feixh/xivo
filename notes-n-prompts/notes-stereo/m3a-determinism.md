# M3a — a 1-in-20 nondeterminism, and why it had to be fixed before tuning

Not a planned milestone. Found by the M3 regression gate and worth its own note,
because it changes what every ATE number in this project means.

## Symptom

The first 6-room stereo run disagreed with the monocular baseline on exactly one
sequence:

```
          M1 baseline   first M3 stereo run
room3     0.154850      0.170322
```

The other five matched byte-for-byte. Since M3 writes right observations that
nothing reads, a stereo run *cannot* legitimately differ from mono, so this
looked like a bug in the left-tracking path.

## It was not stereo, and it was not deterministic

Running room3 stereo in isolation reproduced the mono trajectory exactly. So the
difference only appeared under the 6-way parallel eval — which pointed at
something environmental rather than algorithmic.

Running room3 stereo 8× concurrently:

```
7 × 13e3579f7a   (matches the mono baseline)
1 × 456fc323fe   (the alternate trajectory)
```

Same binary, same config, same input, same `XIVO_RANDOM_SEED=0` — two different
answers. And notably *only* two: the alternate hash was bit-identical to the one
seen in the failed 6-room run, which says one discrete decision is flipping
somewhere rather than floating-point noise accumulating.

A single-threaded deterministic program cannot do this. XIVO contains no threads
and no clock reads (`grep` for `std::thread`, `steady_clock`, `getTickCount`,
`parallel_for` finds nothing outside OpenCV), so the candidates were
uninitialized memory or something address-dependent.

Address dependence, tested by disabling ASLR with `setarch -R`:

```
ASLR off:  40 runs, 40 identical
ASLR on:   40 runs, 38 identical + 2 flakes   (~5%)
```

## Mechanism

Three places order things by pointer, and pointers are randomized per run.

1. **`GraphBase::features_` / `groups_` are keyed by `int`** — these are *fine*.
   libstdc++ hashes integers with the identity function, so iteration order is
   determined by the insert/erase history, which is reproducible. `GetFeaturesIf`
   is therefore not a source of nondeterminism. Worth stating explicitly because
   it is the container one suspects first.

2. **`Criteria::CandidateComparison` left ties unresolved.** Candidate lists
   reach `std::sort` via `MakePtrVectorUnique`, which is
   `std::sort(v.begin(), v.end())` on a `vector<T*>` — i.e. *sorted by address*.
   The comparator returned `(s1 > s2) || (s1 == s2 && f1->score() > f2->score())`,
   so tied features were mutually incomparable and `std::sort` (not stable) was
   free to return either order. Ties are the common case, not a rare one: every
   freshly initialized candidate carries the same initial depth variance, and
   `score()` is exactly `-P_(2,2)`. The sorted list is then truncated at
   `kMaxFeature`, so *which* of several equally-uncertain features got promoted
   into the EKF state was decided by the input order.

3. **Two `unordered_set` containers keyed by pointer are iterated**, and hashing
   a pointer means hashing its value, so bucket order shifts with the ASLR base:
   - `Graph::FindNewGaugeFeatures` → `collinear_check` iterates
     `gauge_features_[g]` (an `unordered_set<FeaturePtr>`) into
     `PointsAreCollinear`, whose verdict is a thresholded cross product and is
     *not* invariant to which point plays the role of the base.
   - `Estimator::DiscardAffectedGroups` iterates `affected_groups_` while
     **mutating the graph**: `FindNewOwnersForFeaturesOf` reassigns feature
     ownership, so discarding one group changes whether the next group still
     meets the instate-feature threshold.

On the relative weight of (2) versus (3): the `MemoryManager` hands out slots
from 512 individually `new`-ed objects allocated once at startup in a fixed
order, so their *relative* addresses are stable within and across runs — ASLR
shifts the whole layout but preserves ordering. That makes (2) a latent fragility
rather than the trigger, and points at (3), where an address is *hashed* rather
than compared, as the actual source of the run-to-run variation. **I did not
isolate which of the two set iterations was responsible** — all three were fixed
together and the flake disappeared. Ablating them individually would have cost
two more 32-run batches for information that would not have changed any of the
three fixes.

## Fixes

- `options.cpp`: `CandidateComparison` now falls through to `f1->id() < f2->id()`,
  making it a *total* order — the sort result no longer depends on input order at
  all. Ascending id prefers the older feature, which has the longer track.
- `manager.cpp`: the group comparator in `SelectAndAddNewFeatures` gained the
  same `g->id()` tie-break; small integer feature counts tie constantly.
- `manager.cpp`: `DiscardAffectedGroups` iterates a vector sorted by group id.
- `graph.cpp`: `collinear_check` orders its points by feature id.

Also fixed in passing: `CandidateComparison` computed `score1`/`score2` from the
configured `comparison_score_type` and then **discarded them**, comparing
`f->score()` unconditionally. `comparison_score_type` was a silently ignored
knob. Every config in `cfg/` sets it to `DepthUncertainty`, which is exactly what
`score()` returns, so no past result was affected — but the knob would have
looked inert if M6 tried to tune it.

## Verification

`unitTests_determinism` (5 tests). The load-bearing one is
`SortResultDoesNotDependOnInputOrder`: it sorts 200 random permutations of the
same tied feature set and asserts the id sequence is always identical.
`TruncationKeepsTheSameFeaturesRegardlessOfInputOrder` asserts the same for the
*truncated* subset, which is where a partial order actually changes behavior.
`ScoreOutranksIdTieBreak` guards against the fix quietly becoming an
"oldest-first" selection policy, and `TiedFeaturesHaveIdenticalScores` guards the
premise — without it the antisymmetry test could pass trivially if `Initialize`
ever stopped producing identical covariances.

The tests were confirmed to have teeth by ablation: replacing the tie-break with
`return false` makes 3 of the 5 fail and leaves the 2 premise-guards passing.

End-to-end: 32 concurrent room3 stereo runs after the fix → 32/32 identical.

## Consequences — this is the part that matters for M6

**The baseline moved.** The fix changes which features get promoted, so all
earlier numbers are superseded. Mono and stereo, all 6 rooms, byte-identical to
each other:

```
seq      before (M0/M1)   after (M3a)
room1    0.133641         0.107525
room2    0.068441         0.080113
room3    0.154850         0.143678
room4    0.091062         0.096501
room5    0.099227         0.109758
room6    0.063883         0.077238
mean     0.1019           0.1025
```

Three rooms improved, three got worse, the mean barely moved (0.1019 → 0.1025).
That is the signature of an arbitrary choice being made differently, not of a
better or worse algorithm — which is precisely why the next point matters.

**The noise floor.** Varying only `XIVO_RANDOM_SEED` (which drives the gauge
feature shuffle) over 4 seeds:

```
seq        seed0     seed1     seed2     seed3    spread     std
room1     0.1075    0.1075    0.1075    0.1075   0.0000   0.0000
room2     0.0801    0.0801    0.0801    0.0801   0.0000   0.0000
room3     0.1437    0.1368    0.1470    0.1358   0.0112   0.0054
room4     0.0965    0.1060    0.1005    0.0964   0.0096   0.0045
room5     0.1098    0.1094    0.1098    0.0963   0.0134   0.0067
room6     0.0772    0.0772    0.0772    0.0772   0.0000   0.0000

mean of 6 rooms:  0.1025 / 0.1029 / 0.1037 / 0.0989
spread of the mean across seeds: 0.0048 m   (std 0.0021)
```

So, for the rest of this project:

- A **per-room** ATE change smaller than ~0.013 m is noise.
- A change in the **6-room mean** smaller than ~0.005 m is noise.
- room1, room2 and room6 are seed-insensitive; room3, room4 and room5 carry all
  of the variance, so a config that "improves room5 by 0.01" has improved
  nothing.
- Single-run comparisons are only meaningful for differences well above these
  bounds. Anything marginal needs multiple seeds.

The M6 target (mean ATE < 0.06 from 0.1025) requires roughly a 0.04 m
improvement, which is ~8× the mean noise. The target is comfortably above the
floor, so single-seed runs are adequate for tracking progress toward it — but
not for choosing between two configs that land within 0.005 m of each other.

## Reusable technique

`setarch -R <cmd>` is the cheapest possible test for "is this nondeterminism
address-dependent?". It took one 8-run batch to convert "room3 is mysteriously
flaky" into "something orders by pointer", which narrowed the search from the
whole estimator to four `grep` hits.
