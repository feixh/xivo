# Merging auto-stereo, auto-bugfix, auto-memory and auto-oos into `auto`

Four branches merged sequentially, each merge evaluated in stereo and monocular
mode against the performance level recorded in [RESULTS_STEREO.md](RESULTS_STEREO.md).

| stage | commit    | merge          | tests                        | verdict |
|-------|-----------|----------------|------------------------------|---------|
| 1     | `5d091f3` | `auto-stereo`  | 13/14 binaries (2 pre-existing failures) | reproduces the reference exactly |
| 2     | `b19d737` | `auto-bugfix`  | 14/14, 116 tests             | no regression; RPE_tra improves |
| 3     | `d0d335d` | `auto-memory`  | 16/16, 121 tests             | trajectories byte-identical to stage 2 |
| 4     | `fdcf9b6` | `auto-oos`     | 17/17, 134 tests             | OOS off: byte-identical to stage 3 |
| 4b    | `HEAD`    | stereo OOS rows | 18/18, 143 tests            | OOS improves both modes; right rows neutral |

## Protocol

Each stage: `room1..room6` x {stereo, monocular}, 12 runs.

* Stereo is `cfg/tumvi_stereo.json`. Monocular is `cfg/tumvi_mono_ctl.json`, the
  same file with `stereo`, `stereo_init.enable` and `stereo_update.enable` set
  false -- so it is a control at identical EKF capacity and identical tracker
  settings, not the shipped mono config.
* `XIVO_RANDOM_SEED=0`, `setarch -R` (ASLR off), and one thread per process
  (`OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1`). Pinning
  is bit-identical to letting OpenCV size its own pool and stops a batch of 12
  from thrashing.
* ATE from `scripts/tum_rgbd_benchmark_tools/evaluate_ate.py` at
  `--max_difference` 0.001 (the window RESULTS_STEREO.md quotes; it scores ~26%
  of frames, in blocks, and skips the initialization phase) and at 0.02 (~98% of
  frames). RPE from `evaluate_rpe.py --fixed_delta --delta_unit s --delta 1`.
* Harness: `merge/{one,batch,ens,tab,enstab,tests}.sh|py` in the workspace root.

## Stage 1 -- auto-stereo

Content-wise a fast-forward (`auto^{tree} == auto-stereo^{tree}`), recorded as a
`--no-ff` commit for a symmetric, revertible history. Its evaluation doubles as
the reproduction of the reference: all 48 numbers match RESULTS_STEREO.md to
four decimals, including the per-sequence stereo-seeded rates.

## Stage 2 -- auto-bugfix

Five textual conflicts, plus two interactions the merge had to decide rather
than resolve (see the commit message for both). The load-bearing one:
`tracker_cfg.use_prediction` was dead code before this branch -- the tracker
always seeded KLT from the EKF's predicted measurement -- so the value that
*preserves* the behaviour `cfg/tumvi_stereo.json` was tuned against is `true`,
not the `false` the file said. Honoring `false` diverges room4 (ATE 0.10 ->
1.04). `auto-bugfix` migrated its own 23 configs in `1c9e5a8`; the stereo config
was written later and missed that pass.

Single runs, six-room means:

| metric | stage 1 | stage 2 | reference |
|--------|---------|---------|-----------|
| stereo ATE@0.001 | 0.0476 | 0.0540 | 0.0476 |
| stereo ATE@0.02  | 0.0575 | 0.0599 | 0.0575 |
| stereo RPE_tra   | 0.0145 | 0.0137 | 0.0145 |
| mono ATE@0.001   | 0.0792 | 0.0758 | 0.0792 |
| mono ATE@0.02    | 0.0953 | 0.0888 | 0.0953 |
| mono RPE_tra     | 0.0243 | 0.0222 | 0.0243 |

The stereo ATE@0.001 delta of +0.0064 is not interpretable from single runs: the
six-room mean has an intrinsic sd of ~0.007 at that window, because hard
accept/reject gating inside the tracker<->filter loop makes the trajectory a
deterministic but chaotic function of the input. So both trees were run as
6-member ensembles, each member perturbing the initial velocity `X.Vsb` by
`k * 1e-6` m/s -- six orders of magnitude inside the config's own prior
(`P.Vsb: 0.5`), i.e. the same physical problem in every member:

| metric | stage 1 (mean +- sd) | stage 2 (mean +- sd) | delta | Welch t |
|--------|----------------------|----------------------|-------|---------|
| stereo ATE@0.001 | 0.0524 +- 0.0025 | 0.0556 +- 0.0032 | +0.0032 | +1.96 |
| stereo ATE@0.02  | 0.0643 +- 0.0048 | 0.0637 +- 0.0044 | -0.0006 | -0.24 |
| stereo RPE_tra   | 0.0146 +- 0.0001 | 0.0138 +- 0.0002 | **-0.0007** | -10.5 |
| stereo RPE_rot   | 0.6213 +- 0.0004 | 0.6207 +- 0.0005 | -0.0006 | -2.16 |
| mono ATE@0.001   | 0.0760 +- 0.0057 | 0.0784 +- 0.0061 | +0.0024 | +0.74 |
| mono ATE@0.02    | 0.0951 +- 0.0069 | 0.0945 +- 0.0083 | -0.0006 | -0.15 |
| mono RPE_tra     | 0.0241 +- 0.0006 | 0.0228 +- 0.0004 | **-0.0013** | -2.90 |
| mono RPE_rot     | 0.6206 +- 0.0004 | 0.6204 +- 0.0004 | -0.0002 | -0.84 |

Conclusions:

* **RPE_tra improves in both modes**, by 5x its own spread in stereo. RPE_rot is
  flat to slightly better. These are the two metrics readable from a small
  sample.
* **ATE is unchanged.** At the 98%-coverage window both modes move -0.0006. At
  0.001 both move up by less than one sd (t = 1.96 stereo, 0.74 mono), and
  per-room the shifts scatter in both directions with |t| up to 4.8 -- room5
  mono +0.0261, room1 mono -0.0163, room3 stereo +0.0090, room5 stereo -0.0115.
  That is reshuffled gating decisions, not a systematic loss.
* RESULTS_STEREO.md's headline 0.0476 is the **minimum** of its own tree's
  6-member ensemble (mean 0.0524), so part of the apparent stereo gap is that the
  reference is a favourable draw. Future comparisons should quote the ensemble.

## Stage 3 -- auto-memory

Five conflicts, three of them the *same defect fixed twice* (invalid `std::sort`
predicates, the dangling `gauge_group_ptr_`, nine `std::max` -> `std::min`
output sizings). See the commit message.

All 12 estimated trajectories are **byte-identical** to stage 2:

```
byte-identical trajectories: 12/12, differing: 0
```

so every metric is identical by construction. This is the strongest form the
no-regression check can take, and it is what the branch claimed: its changes are
leak fixes, pooled-object hygiene, buffer-safety fixes on the pybind11 boundary
and instrumentation, none of which touch the filter's arithmetic.

`ctest` also works from the build directory again (16/16). It had been reporting
12 of 13 binaries as failing, entirely because the test fixtures are opened by
paths relative to the source root; auto-memory's `WORKING_DIRECTORY` on
`add_test` is now applied to all sixteen targets.

## Stage 4 -- auto-oos, and its stereo rows

Two steps, evaluated separately because they answer different questions.

**4 (`fdcf9b6`) is the merge.** `auto-oos` adds the out-of-state (MSCKF) update:
when the tracker drops a track that never entered the state, its observations are
triangulated and the 3-D point marginalized out through the left nullspace of its
Jacobian, so the track still constrains the pose window. The update is off unless
`use_OOS` is set, and with it off all 12 trajectories are **byte-identical** to
stage 3 -- the merge itself changes nothing that runs.

**4b is making that update use the right camera.** The measurement `auto-oos`
was developed against is monocular; on the stereo config it was discarding the
right observation of every dropped track. A right match adds two rows per view
(`4n - 3` rows after marginalization instead of `2n - 3`), reusing the left
camera's Jacobian chain because the rig is fixed and outside the error state.
The design and the two constraints that shaped it (isotropic noise, and a
Jacobian buffer that lives on every pooled feature) are in the workspace's
`notes-n-prompts/notes-oos/m6-stereo-oos.md`. On room1, 89.7% of the views
of a dropped track carry a right observation; the estimator's exit summary reports
that as `with_right=`.

Four arms x 6 members x 6 rooms = 144 runs
(`merge/logs/e_{mo,st}_{off,on}.log`, runs in `merge/runs/e_*`):

| arm | ATE@0.001 | ATE@0.02 | RPE_tra | RPE_rot |
|-----|-----------|----------|---------|---------|
| mono, OOS off   | 0.0784 +- 0.0061 | 0.0945 +- 0.0083 | 0.0228 +- 0.0004 | 0.6204 |
| mono, OOS on    | 0.0686 +- 0.0034 | 0.0852 +- 0.0051 | 0.0213 +- 0.0010 | 0.6203 |
| stereo, OOS off | 0.0556 +- 0.0032 | 0.0637 +- 0.0044 | 0.0138 +- 0.0002 | 0.6207 |
| stereo, OOS on  | **0.0453 +- 0.0024** | **0.0591 +- 0.0029** | **0.0132 +- 0.0000** | 0.6215 |

Off -> on, with the Welch t of each delta:

| metric | mono delta | t | stereo delta | t |
|--------|-----------|---|--------------|---|
| ATE@0.001 | **-0.0098** | -3.46 | **-0.0103** | -6.29 |
| ATE@0.02  | **-0.0093** | -2.34 | **-0.0046** | -2.15 |
| RPE_tra   | **-0.0015** | -3.42 | **-0.0006** | -9.54 |
| RPE_rot   | -0.0001 | -0.52 | +0.0008 | +2.99 |

Conclusions:

* **No regression in either mode, and nothing measurably worse.** All six ATE and
  RPE_tra deltas improve, every one of them by more than two sd; the two RPE_rot
  deltas are flat (next bullet). Stereo with OOS on is the best arm on every
  metric and the tightest of the four (sd 0.0024), and it beats mono with OOS on
  by -0.0232 at t = -13.6.
* **The +0.0008 deg of stereo RPE_rot is an evaluator artifact, not a
  regression.** `evaluate_rpe.py` matches timestamps by nearest neighbour, which
  scores a *perfect* trajectory at ~0.30 deg on these sequences -- half of the
  0.62 deg it reports. Re-scoring all 144 runs with `evaluate_rpe_interp.py`
  (`scripts/rpe_interp_dir.py`, aggregated by `merge/rpeitab.py`, raw in
  `merge/logs/e_rpei.log`) gives stereo 0.5126 +- 0.0006 -> 0.5131 +- 0.0006,
  **+0.0005 at t = +1.34**, i.e. nothing; mono goes -0.0006 at t = -1.79. On the
  same evaluator RPE_tra improves -0.0007 at t = -10.3 (stereo) and -0.0016 at
  t = -3.4 (mono). Rotational claims on TUM-VI have to come from the interpolated
  evaluator.
* **The gain is the update, not the right rows.** Running the stereo OOS config
  with `OOS.use_stereo: false` -- the monocular measurement on the stereo
  pipeline -- gives 0.0449 +- 0.0015 / 0.0565 +- 0.0027 / 0.0133 / 0.6216, so
  turning the right rows on moves ATE@0.001 by +0.0005 (t = +0.40), ATE@0.02 by
  +0.0025 (t = +1.55) and RPE_tra by -0.0001 (t = -1.15): nothing clears its own
  noise (`merge/logs/e_st_on_monomeas.log`). The rows are not inert -- on room1
  they take the update from 19727 to 44992 rows, drop `bad_triangulation` from
  795 to 778 and raise accepted tracks from 2695 to 2762, at no measurable wall
  clock -- but `too_short=10538` of `candidates=14115` and a mean of 2.0 in-state
  views per candidate say what actually limits the update on these sequences.
  Better depth on an already well-determined depth buys nothing; the rows are the
  better-posed measurement for the motions this dataset does not contain (slow or
  forward, where the view-to-view baseline collapses). Hence `use_stereo`
  defaults on, with a knob to turn it off.
* **Nothing outside the OOS path moved.** 18 trajectories are byte-identical
  across the change: stereo OOS-off and mono OOS-off against their stage-4
  counterparts (12), and mono OOS-on against the pre-stereo-rows tuning run
  `e1__mo_oos_r35` (6). The last six are the load-bearing ones -- they say the
  stereo work cannot have perturbed the monocular measurement, which is otherwise
  hard to establish since both paths share `ComputeOOSJacobianInternal`.

Configs: `cfg/tumvi_stereo_oos.json` and `cfg/tumvi_mono_ctl_oos.json`, each
differing from its OOS-off twin only by `use_OOS: true` and a commented `OOS`
block. `ctest` is 18/18 (143 cases); the new binary is `unitTests_oos_stereo`
(9 cases, `ctest -R OOSStereo`).

## Caveats

* Wall-clock numbers in `merge/logs/` are not comparable to the timing table in
  RESULTS_STEREO.md: part of these runs shared the machine with an unrelated
  141-process job. ATE/RPE are unaffected (the runs are deterministic).
* `cfg/tumvi_cam0_faithful.json`, `cfg/tumvi_cam0_ref.json` and
  `cfg/tumvi_cam0_ref2.json` are untracked files in the working tree that still
  say `use_prediction: false`. Since stage 2 that key is live and that value
  diverges room4; anything reusing them needs the same migration.
