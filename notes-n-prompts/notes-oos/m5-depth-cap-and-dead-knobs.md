# M5 — the depth cap, and three knobs that do nothing

A last tuning pass over the config keys the M4 sweeps had not touched. It found
one real lever and three dead knobs, one of which is a bug.

Everything here is on the deterministic build, six-room mean ATE, single-knob
deltas from the M4 config (`cfg/oos.json` at 0.0677).

## The depth cap was the lever

```json
"max_depth": 5.0
```

`Criteria::Candidate` and `Criteria::CandidateStrict` (`src/options.cpp`) both
gate feature promotion on `zmin < f->z() < zmax`, with `zmax = max_depth`, and
`max_depth` also clamps the adaptive initial-depth median (`manager.cpp:315`). At
5.0 m every feature further away than that was barred from the state — in a TUM-VI
*room*, that is a large share of the useful structure, and exactly the structure
that constrains attitude and scale best.

| `max_depth` | 5.0 (M4) | 8.0 | 10.0 | 12.0 | 15.0 | 20.0 |
| --- | --- | --- | --- | --- | --- | --- |
| mean ATE | 0.0677 | 0.0669 | **0.0657** | 0.0660 | 0.0684 | 0.0680 |

Raising `triangulation.zmax` (5.0) alongside it is *worse* (0.0715 at
`max_depth: 10`): that gate decides which two-view triangulations are trusted at
initialization, and a distant point triangulated from a short baseline is exactly
the case you want to reject. The two caps do different jobs and should not be
moved together.

`strict_criteria_timesteps` composes with it. It controls how long after
initialization the strict admission criteria apply; raising it 5 → 10 buys another
0.0009 on top of `max_depth: 10`, and saturates there (10, 15 and 20 give
byte-identical results).

| config | mean ATE |
| --- | --- |
| M4 config | 0.0677 |
| `max_depth: 10` | 0.0657 |
| `max_depth: 10` + `strict_criteria_timesteps: 10` | **0.0648** |
| `max_depth: 15` + `strict_criteria_timesteps: 10` | 0.0661 |

Also swept and rejected in the same round: `initial_z` 1.5 (0.0700) and 3.5
(0.0674) against 2.5; `min_depth` 0.2 (0.0760) against 0.05;
`max_subfilter_outlier` 1.5 (0.0678) against 0.01.

## Three knobs that do nothing

Three single-knob configs came back *byte-identical* to the base, which is a
stronger signal than "no improvement" — it means the value never reached a
decision. Each had a code reason:

1. **`tracker_cfg.use_prediction`** — the string `use_prediction` does not appear
   anywhere in `src/`. A dead config key.
2. **`outlier_thresh`** — read into `outlier_thresh_`, whose only consumer is
   `Estimator::HuberOnInnovation`, which is declared, defined, and never called.
   (The source even carries a `FIXME (xfei)` next to it saying it "kinda overlaps
   with MH gating".) Dead in practice.
3. **`feature_owner_change_cov_factor`** — a **bug**: `estimator.cpp` read the key
   as `filter_owner_change_cov_factor`, so the configured value was silently
   ignored and the 1.5 default always applied. Fixed to read the documented name.
   Result-neutral for every config in `cfg/` except one that sets 1.0
   (`f_nodesc_cmp.json`), and verified neutral by re-running the base config after
   the fix: byte-identical to before it.

This is the third instance in this work of a knob that looked tuned and wasn't
(after `comparison_score_type` and `use_compression` in M4). Worth checking the
consumer before believing a sweep that shows no effect.

## The M4 operating point survives the new depth cap

Everything tuned in M4 was re-checked against the new base, in case admitting
distant features changed what the rest of the filter wants. Nothing did — the M4
values are still the best ones:

| knob | base | also tried | mean ATE |
| --- | --- | --- | --- |
| tracker features | 100–130 (0.0648) | 110–140 / 130–160 | 0.0727 / 0.0701 |
| `oos_meas_std` | 2.0 (0.0648) | 1.5 / 2.5 | 0.0709 / 0.0679 |
| `OOS.pose_window` | 20 (0.0648) | 30 | 0.0684 |
| `OOS.augment_every` | 2 (0.0648) | 1 | 0.0727 |
| `visual_meas_std` | 1.5 (0.0648) | 1.2 / 1.8 | 0.0739 / 0.0747 |
| `Qimu.gyro` | 3× TUM-VI (0.0648) | 2× / 4× | 0.0680 / 0.0682 |
| `Qimu.accel` | 3× TUM-VI (0.0648) | 2× / 4× | 0.0705 / 0.0713 |
| `max_group_lifetime` | 60 (0.0648) | 120 | 0.0687 |
| `OOS.max_observations` | 15 (0.0648) | 25 | 0.0655 |

The IMU rows are worth a second look: 3× inflation of the published TUM-VI noise
densities is a local optimum in *both* directions on *both* channels, at two
different operating points (M4's and this one). That is a strong hint that the
inflation is standing in for something structural — an unmodelled bias dynamic or
IMU intrinsic — rather than being a fudge factor that happens to fit.

## Final numbers and robustness

`results/oos/m5-final/table.txt`, all six rooms, same binary:

| config | mean ATE | mean RPE rot | mean RPE tra |
| --- | --- | --- | --- |
| shipped `sweep_dlt_nodesc` | 0.0923 | 0.6202 | 0.0246 |
| tuned, `use_OOS: false` (`cfg/oos_off.json`) | 0.0733 | 0.6192 | 0.0243 |
| tuned, OOS + pose window (`cfg/oos.json`) | **0.0648** | 0.6192 | 0.0212 |

The `sweep_dlt_nodesc` row is identical to the M4 batch's, which is a free
cross-batch check that the typo fix really is result-neutral and that the build is
still deterministic.

Seed sensitivity of the delivered config (`results/oos/m5-seeds`; the only
consumer of the seed is the `std::shuffle` in the gauge-feature collinearity retry
loop):

| seed | room1 | room2 | room3 | room4 | room5 | room6 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.087821 | 0.068426 | 0.074867 | 0.049779 | 0.064649 | 0.043450 | 0.0648 |
| 1 | 0.087821 | 0.071668 | 0.075676 | 0.049779 | 0.064649 | 0.043450 | 0.0655 |
| 2 | 0.087821 | 0.072697 | 0.074903 | 0.049779 | 0.064649 | 0.043450 | 0.0655 |

Four of six sequences never reach the retry loop and are seed-independent; the
spread of the mean is 0.0007. Seed 0 reproduced `results/oos/m5-final` to all six
printed digits.

## How to read these tables

Each row is one deterministic run, not a statistical estimate. The filter is
chaotic with respect to small config changes — flipping one admission decision
reshuffles the rest of the trajectory — so per-room ATE moves by 0.01–0.03 between
configurations that differ in a single threshold, and a 0.001–0.002 difference in
the six-room mean means nothing on its own. The conclusions here rest on either a
monotone trend across several values of a knob (the depth cap) or a delta of 0.005
and up (the depth cap again, and OOS on/off).
