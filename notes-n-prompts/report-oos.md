# Out-of-state (MSCKF) update for XIVO — report

Delivered on branch **`auto-oos`** of the xivo package, in the worktree
`xivo-oos/` (created from `auto`).

## Summary

XIVO shipped a partial, unreachable out-of-state (OOS) code path. It is now a
working MSCKF-style update running alongside the in-state (EKF-SLAM) update, in
monocular + IMU configuration, and the combination is **46 % better than the
workspace README baseline**.

| configuration | mean ATE | mean RPE rot | mean RPE tra |
| --- | --- | --- | --- |
| workspace README reference (`dlt+nodesc`) | 0.1209 | 0.6206 | — |
| shipped config, this binary (`cfg/sweep_dlt_nodesc.json`) | 0.0923 | 0.6202 | 0.0246 |
| tuned, OOS **off** (`cfg/oos_off.json`) | 0.0733 | 0.6192 | 0.0243 |
| **tuned, OOS + pose window** (`cfg/oos.json`) | **0.0648** | 0.6192 | 0.0212 |

Per sequence, the delivered configuration:

| | room1 | room2 | room3 | room4 | room5 | room6 |
| --- | --- | --- | --- | --- | --- | --- |
| ATE (m) | 0.0878 | 0.0684 | 0.0749 | 0.0498 | 0.0646 | 0.0435 |
| RPE rot (deg) | 0.528 | 0.722 | 0.731 | 0.634 | 0.573 | 0.528 |

`results/oos/m5-final/table.txt`. With the same tuning and the same binary,
turning the OOS update off costs 0.0085 m of mean ATE (0.0648 → 0.0733).

### Exit criteria

| criterion | target | achieved | met |
| --- | --- | --- | --- |
| mean ATE over room1–6 | < 0.06 m | **0.0648 m** | no |
| mean ATE as small as possible | — | 0.0648 m, from 0.1209 | — |
| mean rotational RPE | < 0.5° | **0.6192°** | no |

I did not reach either numeric threshold. The ATE is 8 % above the target. The
rotational RPE is not reachable by configuration at all, and the evidence for that
is strong enough to state plainly — see "Why the rotation criterion is out of
reach" below.

## What was wrong, and what I changed

Milestones, each a commit on `auto-oos`:

| | commit | what |
| --- | --- | --- |
| M0 | `3ae9e37` | baseline, harness, and a real bug: `FillJacobianBlock` copied only one of the reference-group Jacobian blocks |
| M1 | `4cc1665` | correct OOS measurement model, unit-tested (13 new tests) |
| M2 | `1934ea0` | wire the update into the filter |
| M3 | `36702f7` | a pose window, without which the update is inert |
| M4 | `0b8e76c` | make the estimate reproducible run to run |
| M4 | `0d6ec6c` | tuning; ship the tuned config |
| M5 | `95378f3` | the depth cap, a silently-ignored config key, final config |

Detailed notes for each are in [`notes-oos/`](notes-oos/).

The branch has since been merged into `auto` (`fdcf9b6`), and the update taught to
use the right camera of the stereo rig (`fe696f6`) -- see
[`notes-oos/m6-stereo-oos.md`](notes-oos/m6-stereo-oos.md) and stage 4 of
`xivo/RESULTS_MERGE.md`. Nothing below changed: with `use_OOS` off the merge is
byte-identical to its parent, and the monocular measurement is byte-identical to
the runs this report quotes.

### The measurement (M1)

A feature dropped by the tracker without ever entering the state is triangulated
over its observations; the 3-D point is then marginalized out of the stacked `2n`
reprojection residuals by projecting onto the left nullspace of `Hf`, leaving a
`2n-3` row constraint on the in-state group poses. The nullspace basis comes from a
Householder QR, which matters: the basis has to be *orthonormal* because the update
feeds a diagonal `Roos_`, and a non-orthonormal basis would silently correlate the
whitened noise.

Bugs fixed on the way: the reference-group Jacobian copy (M0), and the residual
sign and camera-frame convention in the marginalization (M1).

### The pose window (M3) — the thing that actually made it work

This is the part worth reading. **XIVO has no pose window.** A group enters the EKF
state only as a side effect of feature promotion, so the in-state groups are a
sparse, feature-driven scatter of ~12 anchor poses spread over the whole
trajectory, not the last N frames. MSCKF assumes the opposite.

The consequence, measured with purpose-built instrumentation rather than guessed
at: 62 % of dropped tracks had **zero** observations from an in-state group, and on
room1 exactly **1 of 4322** candidates produced a measurement. The update was
correct and completely inert.

Adding a FIFO window of recent poses (`pose_window`, `augment_every`), evicting
only poses that no in-state feature depends on, took room1 from `used=1/4322` to
`used=1649/4074` and its ATE from 0.1355 to 0.1048.

### Two bugs that made the tuning meaningless (M4)

Both were found by noticing that runs which should have been identical weren't.

1. **An ODR violation between the library and the Python bindings.** M3's
   `EKF_MAX_GROUPS` bump was placed in `src/CMakeLists.txt`. `add_definitions`
   doesn't reach the parent directory, and `pyxivo` is built from the *top-level*
   list — so the bindings compiled against `kMaxGroup = 15` while the library used
   40. `Estimator` holds `std::array<bool, kMaxGroup> gsel_` as a *member*, so
   every member declared after it sat at a different offset in the two translation
   units, and the evaluation scripts read the trajectory through header-inlined
   accessors. Every number measured in that window was unreliable. Fixed by moving
   the defines to the top level, with a comment saying why they must live there.

2. **The estimate was not reproducible.** Byte-identical config and binary gave
   room2 ATE 0.0755 and 0.0933 on different runs. Nothing is threaded and the RNG
   is seeded; the cause was iteration order over pointer-keyed containers, which
   depends on heap addresses and therefore on ASLR — `DiscardAffectedGroups`
   iterating `affected_groups_`, and the collinearity check in
   `FindNewGaugeFeatures`. Both now iterate by id. This was worth up to 0.018 m of
   ATE on one sequence and ~0.003 m on the six-room mean, i.e. the same size as the
   differences the sweeps were trying to resolve.

If you take one thing from this work, take the second one: several apparent
improvements measured before that fix were noise, and I only caught it because I
re-ran a configuration I thought I had already measured.

### Tuning (M4, M5)

The two largest levers turned out not to be the OOS update at all.

**In-state capacity.** The stock 30 in-state feature slots are the binding
constraint. `EKF_MAX_FEATURES=200`, `EKF_MAX_GROUPS=60`, tracker at 100–130
features (and the memory pool raised to match, or the run dies in `mm.cpp` with
"Out of feature slots") take the OOS-off configuration from 0.0923 to 0.0781 on
their own.

**The depth cap.** `max_depth: 5.0` gates feature promotion on estimated depth
(`Criteria::Candidate`), so in a room that is bigger than 5 m, every distant
feature — the ones that constrain attitude and scale best — was barred from the
state. `max_depth: 10.0` plus `strict_criteria_timesteps: 10` is worth another
0.0029. The optimum is a genuine peak (0.0669 at 8, 0.0657 at 10, 0.0684 at 15),
and raising the *triangulation* depth gate alongside it is worse, not better: those
two caps do different jobs.

About 50 other knob settings were swept one-at-a-time on six sequences each; the
tables are in [`notes-oos/m4-capacity-and-determinism.md`](notes-oos/m4-capacity-and-determinism.md)
and [`notes-oos/m5-depth-cap-and-dead-knobs.md`](notes-oos/m5-depth-cap-and-dead-knobs.md).
Notable outcomes: the shipped 3× inflation of the TUM-VI IMU noise densities is a
local optimum in both directions on both channels, at two different operating points
— which suggests the inflation stands in for something structural rather than being
a lucky fudge factor; online temporal calibration
is worth ~0.002 and is enabled; online IMU calibration measured worse and is not;
and the OOS Mahalanobis gate is inert at every threshold from 1.0 to 5.991, because
the reprojection-error gate in front of it is doing the actual outlier rejection.

Four knobs turned out to be dead code rather than untuned: `comparison_score_type`
(computed, then not used), `use_compression` (read into a member with no consumer),
`tracker_cfg.use_prediction` (the string does not appear in `src/`), and
`outlier_thresh` (its only consumer, `HuberOnInnovation`, is never called). A fifth,
`feature_owner_change_cov_factor`, was silently ignored because `estimator.cpp` read
the key as `filter_owner_change_cov_factor` — that one is a bug and is fixed.

**On reading the sweep tables.** Each row is one deterministic run, not a
statistical estimate, and the filter is chaotic with respect to tiny config
changes: a knob that shifts one admission decision reshuffles a whole trajectory
downstream. Per-room ATE moves by 0.01–0.03 between configurations that differ in
one threshold. So a 0.001–0.002 difference in the six-room mean is not evidence of
anything on its own; the conclusions above rest on either a monotone trend across
several values, or a delta of 0.005 and up.

## How much of the gain is the OOS update?

At the delivered operating point, turning the OOS update off (same binary, same
tuning, `use_OOS: false` and `pose_window: 0`) costs 0.0085 m: **0.0648 → 0.0733**.

But that number bundles the marginalized constraints together with the side effects
of holding recent poses in the state. An attribution run separates them
(`min_observations: 99` keeps the window but never forms a measurement,
`pose_window: 0` keeps the residuals but not the window). These four were measured
on the binary from just before online temporal calibration was enabled, so they sit
~0.002 above the final table; they are on one binary and the deltas between them
are what matters:

| | mean ATE |
| --- | --- |
| OOS off entirely | 0.0815 |
| OOS residuals, no pose window | 0.0776 |
| pose window, no OOS residuals | 0.0729 |
| both | 0.0695 |

Both halves contribute and they aren't redundant, but a good part of the window's
benefit is *indirect*: holding recent poses in the state changes group slot pressure
and feature promotion, independently of any OOS constraint.

And there is a real tension in the hybrid that the tuning exposed. At 200 in-state
slots the OOS update is nearly inert again — for the opposite reason to M2. Every
track worth having gets promoted, so what's left for OOS averages 1.63 observations
per candidate (`candidates=7034 used=22` on room1). An in-state feature is more
informative per track than a marginalized MSCKF constraint, so whenever you can
afford the slot you should spend it, and OOS gets the residue. The clean
demonstration of OOS's value is therefore at the *stock* capacity, where long
unpromoted tracks actually exist: **0.1101 → 0.0933, a 15 % improvement**, same
binary, `use_OOS` the only difference.

That the delivered arm still gains 0.0085 m from 22 measurements per sequence is
not a contradiction — those 22 are long, well-triangulated tracks whose constraints
tie together a whole window of poses at once — but it does mean the honest headline
is "the hybrid is better than either alone", not "MSCKF carried this".

## Why the rotation criterion is out of reach

Mean rotational RPE is 0.6192° and it does not move. Across roughly 60
configurations it stayed in 0.618–0.653: IMU noise densities from 1× to 6×, in-state
capacity from 30 to 200 features, integration stepsize halved, adaptive stepping on,
visual measurement noise, gauge handling, depth caps, OOS on and off, online
temporal calibration, online IMU calibration.

Per sequence it sits within 0.005° of the numbers published on the TUM-VI wiki for
XIVO, and of the authors' own shipped results:

| | room1 | room2 | room3 | room4 | room5 | room6 |
| --- | --- | --- | --- | --- | --- | --- |
| mine | 0.528 | 0.722 | 0.731 | 0.634 | 0.573 | 0.528 |
| TUM-VI wiki | 0.53 | 0.72 | 0.74 | 0.64 | 0.57 | 0.53 |

The evaluation protocol matches the one behind the README baseline
(`evaluate_rpe.py --fixed_delta --delta_unit s --delta 1`, `--max_difference 0.001`
for ATE, `XIVO_RANDOM_SEED=0`).

So a mean below 0.5° is below what this estimator achieves on these sequences by the
authors' own published figures, and the OOS update cannot change it: it constrains
the same visual geometry the in-state update already does. Reaching it would take a
change to how rotation is observed — a different attitude parameterization, or a
working IMU intrinsics calibration — not a change to the update model or its tuning.
I flagged this rather than tuning further, because another 40 configurations of the
same knobs would not have moved it.

## Reproducing

```bash
cd /home/ubuntu/workspace/auto-slam-engineer
./run_eval_oos.sh oos      <outdir>   # 0.0648, all six rooms
./run_eval_oos.sh oos_off  <outdir>   # 0.0733, the control
```

Results are archived under `results/oos/`: **`m5-final`** (the table above, with the
config each row used), `m5-sweeps` (the four M5 sweep tables), `m4-final`,
`m4-attribution`, `m4-determinism` (six replicates, byte-identical), `m4-seeds`, and
the earlier `m0-baseline` and `m2`.

Reproducibility: fresh runs of both the M4 and the M5 config reproduced their
archived summaries to all six printed digits on every sequence, under a different
load, and the shipped-config row is identical between the M4 and M5 batches. Seeds
1 and 2 give mean ATE 0.0655 and 0.0655 against 0.0648 for seed 0 — four of the six
sequences are seed-independent, and the spread of the mean is 0.0007, so the
headline number is not a lucky draw (`results/oos/m5-seeds`).

Unit tests, from `xivo-oos/` — **run them from the repo root**, they load a config by
relative path:

```bash
bin/unitTests_OOSUpdate    # 13/13, new in M1
bin/unitTests_Jacobians    # 14/14, includes the M0 FillJacobianBlock regression
```

`unitTests_NumericalAlgorithms.SlowAndFastGivensMatch` and
`unitTests_triangulation.Angular_Reprojection_Error` fail; both are pre-existing on
`auto` and documented in [`notes-oos/m0-baseline.md`](notes-oos/m0-baseline.md).

## Plan items I deliberately did not do

The plan ([`plan-oos.md`](plan-oos.md)) listed two M4 items that turned out not to be
worth doing, and I want that visible rather than silently dropped:

* **Measurement compression.** `use_compression` and `compression_trigger_ratio` are
  read into `Estimator` members and then referenced nowhere — the compression call
  site went with the rest of the MSCKF logic in `48ffac8`, so there is no no-op to
  "fix", only a feature to write from scratch. It is also a *cost* optimization
  (mathematically equivalent to the uncompressed update) and `helpers.cpp:QR`
  requires `rows > cols`, i.e. more than 983 stacked rows at the delivered state
  size. It would never trigger. Left as the dead knob it is.
* **Interaction with 1-pt RANSAC.** Swept (`use_1pt_RANSAC` on/off) and left off, as
  shipped; the details are in the M4 sweep table.

## If someone picks this up

In rough order of expected value:

1. **A real sliding window, not a bolted-on one.** The window competes with
   in-state features for `EKF_MAX_GROUPS` slots and for the update's attention.
   Separating pose slots from anchor slots in the state layout would let the window
   be longer without starving feature promotion.
2. **Promotion policy.** The depth-cap finding says feature admission is
   under-tuned in ways that dominate the update model. `Feature::score()` is
   hardcoded to `-P_(2,2)` and `comparison_score_type` is dead; a real score
   (parallax × track length × depth uncertainty) is a contained experiment with a
   plausible 0.005–0.01 in it.
3. **Attitude.** Everything about the rotation error points at the IMU model rather
   than the visual update — `USE_ONLINE_IMU_CALIB` exists and measures *worse*,
   which is itself suspicious and worth a look before concluding the 0.62 floor is
   fundamental.
