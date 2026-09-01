# M0 — baseline on `auto`

Worktree `xivo-bugfix`, branch `auto-bugfix`, branched from `auto` (888511d).
Built with the same flags as `build_all.sh` step 5; `thirdparty/` was rsynced
from the `xivo` tree (unmodified vendored source in both, so this only reuses
the compiled deps and saves a ~20 min rebuild).

Eval harness: `run_eval_bugfix.sh` (a copy of `run_eval.sh` pointing at the
`xivo-bugfix` worktree; `XIVO_WT=` overrides the tree). Config
`sweep_dlt_nodesc`, `XIVO_RANDOM_SEED=0`.

## Unit tests

| binary | result |
|---|---|
| `unitTests_Jacobians` | 13/13 pass |
| `unitTests_atan` / `equi` / `pinhole` / `radtan` | all pass |
| `unitTests_NumericalAlgorithms` | **1 fail**: `NumericalLinearAlgebra.SlowAndFastGivensMatch` |
| `unitTests_triangulation` | **1 fail**: `Triangulation.Angular_Reprojection_Error` |

Two shipped tests fail on a clean build of the base branch. Both are in
numerical code the filter uses; treated as bug reports the authors already
wrote, and assigned to M4.

## End-to-end baseline (`results/bugfix/m0_baseline/`)

```
seq      ATE      RPE_rot   RPE_tra
room1    0.121896 0.530619  0.023907
room2    0.119918 0.725508  0.042083
room3    0.216530 0.736194  0.062585
room4    0.084557 0.636780  0.022895
room5    0.123986 0.575496  0.033299
room6    0.089733 0.531867  0.033810
---------------------------------------
mean     0.1261   0.6227    0.0364
```

**This is the number all later milestones are compared against**, not the
0.1209 in `RESULTS.md`.

## Three things established before measuring any fix

**1. Which known bugs the base branch already carries fixes for.** `auto`
already fixes three of the four defects found during earlier sibling-branch
work: `anynan()` now uses runtime `rows()/cols()`; `Feature::ClampLogDepth()`
saturates log-depth at ±80 (called from `UpdateState`, `SubfilterUpdate`,
`RefineDepth`); `MHGating` bounds the threshold relaxation and gives up instead
of spinning. `RefineDepth` also rejects a non-finite Gauss-Newton delta.
So the only *known* defect still live is `FillJacobianBlock` — confirmed still
present at `src/feature.cpp:688-689`, both reference-group blocks written to
`goff`.

**2. Runs are bit-reproducible — per binary.** room5 run three times (twice
serial, once concurrent with the other five) gives byte-identical trajectories
and ATE 0.123986 every time. Concurrency in `run_eval_bugfix.sh` is therefore
safe.

**3. But they are *not* stable across binaries, and that is a real bug.**
The stored `results/final/triangulation_configs/sweep_dlt_nodesc/` run of the
*same config* gives mean ATE 0.1209, and comparing trajectories pose-by-pose:

| seq | vs stored run |
|---|---|
| room2, room4 | byte-identical |
| room1 | identical for 1743 poses, then diverges |
| room3 | identical for 2154 poses, then diverges |
| room5 | identical for 2132 poses, then diverges |
| room6 | identical for 1470 poses, then diverges |

A byte-identical prefix followed by a clean bifurcation is not floating-point
noise — it is a single discrete decision flipping. The cause is in
`Graph::FindNewGaugeFeatures` (`src/graph.cpp:299-306`): the collinearity check
iterates a `std::unordered_set<FeaturePtr>`, i.e. a hash set **keyed by
pointer**, so the order in which the three candidate gauge features are handed
to `PointsAreCollinear` depends on heap addresses. `PointsAreCollinear` uses
`pts[0]`/`pts[1]` as its base direction, so its verdict is order-dependent — and
that verdict decides which features fix the filter's gauge.

Consequence for this task: an unrelated code change that shifts allocation
layout can move mean ATE by ~0.005 m on its own. That is **5x the seed noise**
(the existing 8-seed study in `results/seeds/` moves only room3, 0.221→0.229,
mean 0.1382→0.1394 — everything else is seed-invariant). So ATE deltas smaller
than ~0.005 m mean cannot be attributed to a fix, and this order-dependence is
itself on the fix list.
