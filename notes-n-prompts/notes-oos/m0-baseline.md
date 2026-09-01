# M0 — worktree, harness, baseline

## Setup

* worktree: `git worktree add ../xivo-oos -b auto-oos auto` (as required).
  `thirdparty/` build artefacts are gitignored, so a fresh worktree has the
  vendored *sources* but none of the installed libs/headers. `rsync -a
  xivo/thirdparty/ xivo-oos/thirdparty/` copies them over (253 MB, 1 s) and
  leaves `git status` clean — no need to re-run the dependency half of
  `build_all.sh`.
* build: same cmake line as `build_all.sh` step 5, in `xivo-oos/build`
  (~60 s with `-j32`).
* `run_eval_oos.sh` at the workspace root = `run_eval.sh` with `XIVO` pointed at
  the worktree (override with `XIVO_DIR`).

## Pre-existing test status (unchanged by anything below)

```
unitTests_Jacobians            13/13 pass   (incl. 6 OOS Jacobian tests)
unitTests_NumericalAlgorithms   2/3 pass    FAIL: SlowAndFastGivensMatch
unitTests_triangulation         4/5 pass    FAIL: Angular_Reprojection_Error
```

`SlowAndFastGivensMatch` failing is worth flagging: both functions in that test
are the OOS marginalisation primitives. M1 replaces the call site with an
explicit orthonormal projection, so neither is on the critical path afterwards.

## The `FillJacobianBlock` bug

`src/feature.cpp`, in the copy of the per-feature Jacobian into the stacked `H`:

```cpp
H.block<2, 3>(offset, goff) = J_.block<2, 3>(0, goff);
H.block<2, 3>(offset, goff) = J_.block<2, 3>(0, goff + 3);   // dest: goff + 3
```

Both writes target `goff`, so the reference group's **rotation** block ends up
holding the **translation** Jacobian, and the translation slot at `goff + 3`
stays zero (`H_.setZero` runs once per update in `FilterUpdate`). `J_` itself is
right — `ComputeJacobian` fills `goff` and `goff + 3` correctly — so
`OnePointRANSAC`, which reads `mh_inliers[i]->J()` directly, was never affected.
That is also why the filter still converged: the body-pose blocks (`Wsb`/`Tsb`)
were always correct and the group-pose block is only part of the model.

Fixed here because it is upstream of everything else: tuning OOS against a
corrupted in-state measurement model would be meaningless.

## Baseline (cfg `sweep_dlt_nodesc`, seed 0, all six rooms)

```
seq      ATE      RPE_rot   RPE_tra
room1    0.1304   0.5287    0.0239
room2    0.0684   0.7235    0.0259
room3    0.1703   0.7320    0.0349
room4    0.0911   0.6367    0.0228
room5    0.0992   0.5754    0.0315
room6    0.0389   0.5272    0.0197
------------------------------------
mean     0.0997   0.6206    0.0264
```

Reference points: `RESULTS.md` reports mean ATE **0.1209** for this config
(pre-fix), the authors' own shipped run is 0.101, the (stale) wiki table 0.093.
So the fix alone takes mono XIVO from 0.1209 to 0.0997 (-18%), i.e. slightly
better than the authors' shipped numbers, with rotational RPE untouched
(0.6222 -> 0.6206).

**All OOS comparisons from here on are against ATE 0.0997 / RPE_rot 0.6206**,
not against the 0.1209 in `RESULTS.md`. The exit criteria then need a further
-40% on ATE and -19% on rotational RPE.

Raw output: `results/oos/m0-baseline/`.

Side note: room3 (0.170) and room1 (0.130) dominate the mean; room6 (0.039) and
room2 (0.068) are already well inside the target. Whatever OOS buys has to show
up on room1/room3 to move the mean much.
