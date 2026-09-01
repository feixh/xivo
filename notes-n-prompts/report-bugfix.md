# Report — scan the code and fix bugs (XIVO, monocular + IMU)

Branch **`auto-bugfix`** in the xivo package, worktree
`/home/ubuntu/workspace/auto-slam-engineer/xivo-bugfix`, branched from `auto`.
Seven commits. Detailed notes in `notes-n-prompts/notes-bugfix/`.

---

## Summary

**67 defects found and catalogued** (`notes-bugfix/m1-bug-register.md`), of which
**27 are reachable** on a monocular TUM-VI run with the shipped config and the
rest are wrong code behind a disabled flag. 62 are fixed; 5 are documented and
deliberately not fixed, each for a stated reason. The unit suite went from
**39 tests with 2 failures to 69 tests, all passing** -- 67 in the default
build, the other 2 ifdef'd behind `USE_ONLINE_IMU_CALIB`.

**The most important result is not a fix — it is that the project's own
evaluation was unsound, in three independent ways.** Correcting all three
changed the verdict on everything that came before:

1. A single six-sequence run does not measure a code change. The run-to-run
   spread is ±0.007 (1 sd) on mean ATE, driven by genuine chaotic sensitivity in
   the filter (defect #64). The baseline I had been comparing every milestone
   against was a **+3.3 sd outlier**.
2. **17% of the reported RPE rotation error was the evaluation script measuring
   itself** (defect #65). `evaluate_rpe.py` scores a *zero-error trajectory* at
   0.2847 deg.
3. **ATE was scored on 26% of frames, with every run's initialization phase
   excluded** (defect #67), understating the true error by ~25%.

With both corrected, the honest end-to-end result is:

| | ATE@0.001 (m) | ATE@0.02 (m) | RPE_rot (deg) | RPE_tra (m) |
|---|---|---|---|---|
| **M0 baseline** (`auto`, 8-member ensemble) | 0.1042 ± 0.0067 | 0.1290 ± 0.0114 | 0.6202 ± 0.0005 | 0.0286 ± 0.0008 |
| **M8 delivered** (`auto-bugfix`, 8-member ensemble) | 0.1071 ± **0.0020** | 0.1377 ± **0.0035** | 0.6200 ± 0.0006 | **0.0265 ± 0.0009** |
| authors' own shipped results | 0.1014 | — | 0.6224 | 0.0325 |
| *same, corrected metric (#65)* | — | — | *0.5120 ± 0.0006* | *0.0261 ± 0.0010* |

Two ATE columns because the harness's default association window is itself a
defect (#67, §3.3): `0.001` scores only ~26% of frames and skips every run's
initialization phase, `0.02` covers ~98%. **Both are reported everywhere.**

- **RPE_tra improves 7%** vs the baseline and **19% vs the authors' own published
  numbers** (0.0325 → 0.0265). At ~7 sem this is the one unambiguous accuracy
  gain.
- **Run-to-run variance drops 3.3×**, and this is the same factor under *both*
  ATE protocols (sd 0.0067 → 0.0020 and 0.0114 → 0.0035). A protocol-independent
  3× stability improvement is the second real result, and arguably the more
  useful one for a filter that has to run unattended.
- **Mean ATE is flat to slightly worse** under both protocols (+0.003 at 0.001,
  +0.009 at 0.02; ~2 sem either way). §6 explains why, and §3.3 records that I
  tested and *rejected* the obvious excuse.
- **RPE_rot is flat.**

**Neither stretch goal is reached, and both sit below what the original code
ever achieved.** ATE < 0.06 m needs a ~40% error reduction that nothing in this
project moved by more than 5%, against the authors' own 0.1014. RPE_rot < 0.5
deg is missed by 0.012 (20 sd) on the corrected metric, and is *unreachable as
literally stated* on the stock metric, whose floor for a perfect trajectory is
0.2847 deg but whose value here cannot be pushed below ~0.618 by any config.
Quantified in `m7-measurement-and-calibration.md` §7.

I want to be direct about the shape of this: **the fixes are demonstrably
correct and almost entirely invisible in ATE.** That is not a hedge, it is the
finding. §4 explains why, and it is a property of the benchmark, not an excuse.

---

## 1. What was asked, and what was done

| Requirement | Status |
|---|---|
| Assume monocular camera + IMU | Scope held; stereo/mapper/loop-closure touched only where a defect also corrupts the mono path |
| Plan first, split into milestones, written to notes dir | `notes-n-prompts/plan-bugfix.md` |
| Test each milestone; e2e evaluation when appropriate | 69 unit tests, 30 new; 8-member × 6-sequence ensembles at each milestone |
| One git commit per milestone | 7 commits on `auto-bugfix`; M0/M1/M8 produced notes rather than code |
| `report-bugfix.md` in the notes dir | this file |
| Detailed notes under `notes-bugfix/` | 8 documents, one per milestone |
| Sub-agents allowed; deliver on `auto-bugfix` | 4 sub-agents for the parallel audit (M1); everything delivered on `auto-bugfix` |
| Worktree `xivo-bugfix` from `auto` | done |

**Deviation from the plan, stated plainly.** The plan's M2–M6 split (by defect
class, report at M6) did not survive contact: the audit returned far more than
expected in `common/` and the accessors, so numerics became its own milestone,
and the evaluation problems of §3 forced an entire milestone that the plan did
not anticipate. Delivered milestones:

| | commit | content |
|---|---|---|
| M0 | — | baseline; worktree built; reproducibility established |
| M1 | — | bug register: 4 parallel sub-agent audits + hand re-verification |
| M2 | `431da19` | `FillJacobianBlock` reference-group blocks |
| M3 | `732f052`, `1c9e5a8` | `AdaptInitialDepth` median; tracker front-end and dead config keys |
| M4 | `69f1486` | triangulation NaN guards, gravity Jacobian, badtri units |
| M5 | `1f0047e` | feature/group lifecycle: re-anchoring, covariance transforms |
| M6 | `559532b` | camera models, nullspace projection, filter plumbing |
| M7 | `4a73ff7` | the RPE metric defect; temporal calibration |
| M8 | this file | report |

The plan's own risk register was partly wrong, which is worth recording. It
said *"`XIVO_RANDOM_SEED=0` is required or every run differs; seed-sensitivity
is checked before claiming a win."* Both halves are false: the seed is **inert**
for this config (nothing draws from the RNG with RANSAC and the sim paths off,
and seeds 0/1/2 produce byte-identical output), and seed-sensitivity is
therefore no protection at all against the run-to-run variation that actually
exists. The mitigation the plan relied on would have let every false claim
through. §3.

## 2. How the bugs were found

Five passes, each catching a class the others miss.

**Static audit, subsystem by subsystem.** ~14 kLOC is small enough to read
exhaustively. Split five ways and audited in parallel by sub-agents, each
required to report `file:line`, the reason it is wrong, the mono+IMU impact, and
a minimal fix. **Every finding was then re-verified by hand before any code
changed**, and five claims did not survive — they are listed at the bottom of
the register with the reasoning, because a plausible-but-wrong finding is worse
than no finding. This yielded the bulk of the 67.

**The repo's two failing tests.** A shipped test that fails is a bug report the
authors already wrote. `Triangulation.Angular_Reprojection_Error` was **the code
being wrong** (#19: unclamped `acos` → NaN silently disables the reprojection
gate; the in-tree comment "fails in RELEASE but passes in DEBUG" is the
fingerprint of `cos` landing on 1.0000000000000002 under FMA contraction).
`NumericalLinearAlgebra.SlowAndFastGivensMatch` was **three stacked bugs**
(#33, #34) plus a test that cannot be satisfied as written (#54): any two
orthonormal bases of the same nullspace differ by an arbitrary orthogonal
factor, so element-wise equality is not an invariant. It now asserts the
invariants that are.

**Differential and invariant testing.** 30 new tests, aimed one layer *above*
where the shipped tests aimed — which is the whole reason the worst bug survived
(§3). Analytic vs numerical Jacobians including the composite chains,
`Aᵗ·Hf = 0` for the marginalizer, project/unproject round-trips over the full
field of view including the singular lines.

**Runtime instrumentation on real data.** Only possible after fixing
`anynan()`, whose NaN guards were all no-ops because it iterated
`Derived::RowsAtCompileTime`, which is `-1` for dynamic matrices. Note that
Release builds define `NDEBUG`, which compiles out **every `CHECK(...)` and all
Eigen bounds assertions**, and the build also sets
`-DEIGEN_INITIALIZE_MATRICES_BY_ZERO`, which masks missing `setZero()`. Both of
the out-of-bounds heap writes below (#60, #61) are silent for exactly this
reason.

**End-to-end regression.** Which is where the trouble started.

## 3. The three evaluation defects

These matter more than any single fix, so they come before the fixes.

### 3.1 The filter is chaotic, and single runs measure nothing (#64)

XIVO makes hard accept/reject decisions — `MHGating`'s chi-square test,
depth-validity ranges, group lifetimes. An arbitrarily small numerical
perturbation flips one, which changes the set of surviving features, which
changes the trajectory macroscopically.

Measured: perturbing **one expression in `camera_equidist.h::UnProject` by 1e-11
relative** (~5e-9 pixels, far below any physical scale) moves mean ATE over six
sequences by 0.013. Across twelve such physically-identical pipelines: mean
0.1104, **sd 0.0073**, range 0.0983–0.1219.

I found this the honest way: an M6 "regression" that I could not attribute.
Reverting `camera_equidist.h` alone reproduced M5 byte-for-byte, so every other
M6 change was empirically inert, and the entire "regression" was a 1-ulp change
in one intermediate.

The mechanism is localised. `run_ensemble_bugfix.sh` (workspace root) draws an
ensemble by perturbing initial velocity by `k·1e-6` m/s — six orders of
magnitude inside the `P.Vsb = 0.5` prior the config itself declares, so no
member is a physically distinguishable scenario. Two independent perturbation
mechanisms agree on the spread: 0.1104 ± 0.0073 (source epsilon) and 0.1102 ±
0.0047 (initial condition). Setting `use_prediction: false` collapses it to
**sd 0.0004**, which pins the cause on the tracker↔filter feedback loop: filter
state → KLT initial guess → which pixel KLT converges to → measurement → filter
state.

Consequences, all of which I had to accept about my own earlier work:

- The M0 baseline of 0.1261 that every milestone was measured against is
  **+3.3 sd above the M0 ensemble mean of 0.1042**, outside the range of eight
  equivalent runs. The "M0 → M4 improvement" was an unlucky baseline draw
  compared against a lucky one.
- **Mean-ATE differences below ~0.015 are not attributable** to a code change.
- Fixing #3 (`use_prediction` declared in 23 configs, read by nothing)
  *switched the feedback loop on* at commit `1c9e5a8`, so every evaluation from
  M3 onward conflated code with config. A dead config key is not just a missing
  feature; it is a silent change of regime the moment someone fixes it.

### 3.2 `evaluate_rpe.py` reports 0.2847 deg on a perfect trajectory (#65)

`evaluate_rpe.py` pairs each estimate timestamp with the **nearest** ground-truth
sample and does not interpolate. TUM-VI ground truth is ~120 Hz, so each
endpoint of each 1-second interval carries ±4.17 ms of independent
quantization — on a metric whose sensitivity is **0.11 deg/ms**.

Two causal demonstrations:

*Decimate the ground truth*, leaving the estimate byte-identical:

| GT spacing (ms) | 8.33 | 16.67 | 25.00 | 33.33 | 50.00 |
|---|---|---|---|---|---|
| reported RPE_rot (deg) | 0.6205 | 0.8614 | 0.9951 | 1.2735 | 1.9733 |

An estimator's error cannot depend on how finely the *reference* was sampled.

*Feed it a zero-error trajectory* — the ground truth interpolated onto the
estimate's own timestamps. Stock `evaluate_rpe.py`: **0.2847 deg / 0.0038 m**.
That is the metric's floor. `evaluate_rpe_interp.py` on the same input:
0.000001 deg, so the interpolating implementation has no floor of its own.

`scripts/tum_rgbd_benchmark_tools/evaluate_rpe_interp.py` SLERP-interpolates the
ground truth to each estimate timestamp. Everything else is identical — it
*imports* `ominus`, `compute_angle`, `compute_distance` and `transform44` from
`evaluate_rpe.py` rather than reimplementing them, so no second opinion about
the metric definition can creep in.

I ruled out the obvious alternative explanation — that the gain comes from
evaluating a different set of pairs, e.g. mishandling TUM-VI's multi-second
mocap dropouts. The interpolated version uses 1–3% **fewer** pairs than stock
(room1 2669 vs 2698), so it is not adding easy ones, and tightening its gap
filter twofold changes the answer by 0.0007. Removing the dropout filter
entirely inflates the number to 1.5–2.5 deg, confirming both implementations
exclude the same dropouts. The change is interpolation, not pair selection.

**This changes the measurement, not the estimator.** A lower number here is a
more accurate measurement of the same trajectory, not a better trajectory. Both
are reported side by side everywhere, the stock number remains the one
comparable to published results, and the script's docstring says so — because
this is precisely the kind of change that becomes an accidental overclaim later.
Corrected: RPE_rot 0.6200 → **0.5120**, RPE_tra 0.0265 → 0.0261.

A side benefit: sub-millisecond offset scans became meaningful once the ±4 ms
quantization was gone, showing the residual time offset is a *small* term and
the ~0.505 floor is genuine rotation error.

### 3.3 ATE was scored on 26% of frames, with initialization excluded (#67)

`run_and_eval_pyxivo.py` invokes `evaluate_ate.py --max_difference 0.001`. The
tool's own default is **0.02**. Images are 20 Hz (period 50.158 ms) and ground
truth ~120 Hz (8.333 ms), a ratio of 6.019, so the image-to-GT phase offset
drifts *slowly* and a 1 ms association window matches only ~26% of frames —
2589 pose pairs per sequence at 0.02 versus ~720 at 0.001.

The subset is not a random subsample. Because the phase drift is slow, the
matched frames fall in contiguous blocks, and on room1 they begin 23 s into a
141 s run: **the entire initialization phase is excluded.** That is where the
largest errors live, so `0.001` systematically flatters runs with poor
initialization — and initialization is exactly what the triangulation (#17/#18)
and lifecycle (#25–#30) fixes target.

Re-scoring the *same* trajectories at 0.02 (`rescore_ate.sh`, no re-running):

| | ATE@0.001 | ATE@0.02 |
|---|---|---|
| M0 baseline | 0.1042 ± 0.0067 | 0.1290 ± 0.0114 |
| M8 delivered | 0.1071 ± 0.0020 | 0.1377 ± 0.0035 |

**I expected this to be the explanation for §6, and it is not.** The honest
protocol shows the same flat-to-slightly-worse mean (+0.009, ~2 sem). Reporting
that plainly: a promising hypothesis, tested, rejected. The 0.001 window was
hiding roughly 25% of the true error, but it was not hiding an improvement.

What the re-scoring *did* establish is that the **3.3× variance reduction is
protocol-independent** (sd 0.0114 → 0.0035 at 0.02, 0.0067 → 0.0020 at 0.001).
Two association windows, two different frame subsets, the same factor — that is
not an artifact of either metric.

`-ate_max_difference` was parsed and then ignored on `auto` (#53); it is now
honoured, with the default left at 0.001 so previously published numbers stay
comparable. Both are reported: `run_eval_bugfix.sh` now prints an `ATE_02`
column beside the 0.001 one on every run (re-scoring the trajectory files the
run already wrote, so it costs nothing), symmetrically with how it already
prints stock and interpolated RPE. Making the honest number the *default* view
is the remedy here — the flag is a knob, and the reason this defect survived is
that nobody had a reason to turn it.

## 4. The fixes that matter

Full list with `file:line` and severity in `m1-bug-register.md`; per-milestone
reasoning in the seven `m*-*.md` notes. The ones a reader should know about:

**#1 — `FillJacobianBlock` wrote both reference-group blocks to the same offset**
(`feature.cpp:688`). The rotation block held the translation Jacobian and the
translation block stayed zero. Every stacked measurement in every update saw a
wrong reference-group Jacobian. *This is the one bug the shipped test suite could
not have caught*: all 13 pre-existing Jacobian tests inspect `J_`, which was
always correct — the defect was in the **copy** of `J_` into the stacked `H`, and
nothing called `FillJacobianBlock`. Test coverage was aimed one layer below the
bug, which is why the new tests are aimed one layer above.

**#17/#18 — DLT-SVD divided by a near-zero singular value and returned `true`**
(`helpers.cpp:126`), and `Triangulate`'s range test let the resulting NaN
through into the success branch (`feature.cpp:755`), so a NaN depth was reported
as a successful triangulation and entered the filter.

**#26/#27 — covariance transforms written to a dead local.** `ChangeOwner`
transformed only the function-local `P_` copy, never the filter block
(`feature.cpp:235`); `inflate_cov` scaled the same dead local *and* ran on the
failure path (`graph.cpp:191`). Re-anchoring a feature to a new group therefore
left its covariance describing its relationship to the *old* group.

**#25 — in-state features were re-anchored after their Jacobians were computed**
(`manager.cpp:72,86,104`), so the Jacobians referred to a group the feature was
no longer anchored to.

**#20 — `dV_dWsg` used `Rsb` where the right-perturbation convention needs
`Rsg`** (`estimator.cpp:647`). The two agree only at `t = 0`, which is
presumably how it survived; from then on the gravity column was rotated by
`Rsb·Rsgᵗ` relative to the truth.

**#35 — `rodrigues.h` flattened derivative outputs column-major** while every
consumer assumes row-major, and `dA_dAu()` returned uninitialised memory. Latent
at first (the calibration flags were off), then **promoted to demonstrated**:
installing the original file with `USE_ONLINE_IMU_CALIB` on fails exactly three
tests, one of which checks `estimator.cpp:726`'s `dV_dCa` chain against a
numerical derivative.

**Two out-of-bounds heap writes, both silent under `NDEBUG`.** #60: `SlowGivens`
resized its working copy of the fixed 30-row `oos_.Hx`. #61: `InstateGroupCovs`
declared `int cnt;` uninitialised and wrote at an arbitrary index on the first
iteration.

**Systematic classes, not one-offs.** Three config keys were declared, plumbed
into the sweep infrastructure, and read by nothing (`use_prediction`,
`comparison_score_type`, and a misspelled `feature_owner_change_cov_factor`
lookup) — anyone who swept them measured noise. The `Qmodel` loader reads 3 of
its 8 keys (#56). Nine accessors size their output with `std::max` where the
fill loop caps at `n_output` (#42). Two `std::sort` comparators are not strict
weak orderings (#10, #32), which is undefined behaviour, and one of them made
results depend on heap pointer order.

**Five defects were left unfixed, each deliberately.** #59 (the adaptive-step
loop never rejects a step) is a design gap in an integrator no config selects,
and fixing it would change the numerics of a dormant path with nothing to
validate against. #64 is characterised rather than fixed — see §6. #66 is a
wrong comment. The remaining two are in the register with reasons.

## 5. Did the exit criterion get met?

The stated criterion is *"the code is free of bugs."*

**Not literally, and it cannot be claimed.** What I can state precisely:

- Every subsystem of the mono+IMU path was read line by line, and every finding
  was hand-verified.
- 62 of 67 catalogued defects are fixed; the 5 exceptions are documented with
  reasons.
- All 69 unit tests pass (67 in the default build), including 30 new ones that assert invariants the
  shipped suite did not reach; both shipped failures are resolved at the correct
  layer.
- All six sequences run to completion with no NaN, no state blow-up, no hang,
  and no non-PSD covariance, with the NaN guards actually working for the first
  time.

**What I know remains.** Defect #64 — the chaotic sensitivity — is real,
reachable, and *not fixed*. It is a design property of hard gating inside a
feedback loop, not a coding error, and removing it means changing the
estimator's architecture (soft/robust weighting instead of hard accept-reject),
which is outside "fix bugs". It is measured and documented rather than papered
over. Anyone who reports a single-run number for this code is reporting noise,
and that is the single most useful thing in this report.

Beyond that: the dormant subsystems (stereo, mapper, OOS-as-a-feature, loop
closure) were audited only where they touch the mono path, so their defect
density is unknown; and an exhaustive audit finding 67 defects in 14 kLOC is
weak evidence that a *second* exhaustive audit would find zero.

## 6. Why the fixes barely move ATE

This deserves a straight answer rather than a caveat.

The chaotic term dominates the signal. Mean ATE has an intrinsic sd of ~0.005
on this benchmark. A correct Jacobian and an incorrect one produce trajectory
*distributions* whose means differ by less than those distributions' widths —
not because the Jacobian does not matter, but because six 2-minute room
sequences do not contain enough independent information to resolve it through a
chaotic map.

Two things support that reading rather than "the fixes were pointless":

1. **RPE_tra, which averages over ~1500 short intervals per sequence instead of
   one global alignment, does move** — 19% better than the authors' own results,
   at ~7 sem. A metric with more independent samples resolves the improvement
   that ATE cannot.
2. **The variance halved** (sd 0.0047 → 0.0020) once temporal calibration was
   enabled, which is only possible because #46 fixed the `td` plumbing.
   Removing systematic per-frame reprojection error makes the gating decisions
   less marginal, which damps the chaos. A more correct model is a *more stable*
   one even where it is not a more accurate one.

There is also a specific reason not to expect ATE gains: **the shipped config
was tuned against the buggy code.** The clearest evidence is that it inflates
the IMU noise densities ~3× above datasheet; un-inflating them
(`cfg/m7_qimu_exact.json`) makes ATE **2.3× worse** (0.2399). That inflation is
load-bearing, absorbing unmodelled error. Correcting the model without
re-deriving the tuning gives back some of what the tuning was compensating for.

I swept for that. Seven single-knob configs spread deliberately wide
(`m7_*.json`) move ATE by 0.14 and RPE_tra by 0.02, while **RPE_rot stays within
0.6184–0.6206** — four ensemble sd. The best ATE found, 0.1035 ± 0.0034
(`m7_feat_more`), is 1 sem from the delivered number and comes with worse
RPE_tra. There is no config in the neighbourhood that turns these fixes into an
ATE win, and I would rather report that than sweep until a lucky draw appears.

## 7. Stretch goals

**Mono+IMU mean ATE < 0.06 m over room1–room6: not met, under either
protocol.** Delivered 0.1071 ± 0.0020 at the harness's 0.001 window and
**0.1377 ± 0.0035** at the honest 0.02 window (§3.3); best measured
configuration 0.1035 ± 0.0034; best single-sequence draw anywhere 0.0459
(room6). The target needs a 44% error reduction on the quoted protocol and 56%
on the honest one. The authors' own published figure for this code is 0.1014,
and nothing in this project — 62 fixes and an eight-config sweep — moved the
mean by more than 5% outside the noise band. This is below what the estimator
achieves, not below what correct code should achieve; reaching it plausibly
needs loop closure or the mapper, i.e. a different algorithm, not fewer bugs.

**Mean ATE as small as possible: 0.1071 ± 0.0020 delivered** (0.1377 ± 0.0035
honest), with **3.3× less run-to-run variance** than the baseline under both
protocols. I did not chase the 0.1035 config because it is within 1 sem and
costs RPE_tra, and picking the best of eight draws is the exact error §3.1 is
about.

**Mean RPE < 0.5 deg: not met, by 0.012.** On the corrected metric,
0.5120 ± 0.0006 — a **20 sd** miss, so this is a real gap and not a measurement
question. On the stock metric the goal is unreachable as stated: its floor for a
perfect trajectory is 0.2847 deg (§3.2) while no configuration pushes its value
below ~0.6184. Everything that could plausibly move it was tried and did not:
seven config knobs (all within 0.002), online temporal calibration (0.0000),
online IMU calibration (0.0003, half a sd). A sub-ms offset scan bottoms out at
0.5054, so residual time offset is a small term and ~0.505 is genuine rotation
error in the estimator.

## 8. If this continued

In the order I would actually do it:

1. **Attack #64 directly** — replace hard MH accept/reject with robust
   weighting (Huber/Cauchy on the innovation) so a marginal feature degrades
   smoothly instead of vanishing. This is the root cause of the unmeasurability
   of everything else, and until it is fixed no amount of careful work on this
   codebase can be validated on ATE.
2. **Re-derive the tuning against the fixed model.** The 3× IMU inflation is
   compensating for errors that are now fixed; a joint sweep (not the one-knob-
   at-a-time of §6) is the only honest way to find out how much of it is still
   needed, and it is the most likely source of a real ATE gain.
3. **Extend the ensemble harness to per-sequence significance.** room3 and
   room1 carry most of the instability (sd 0.015 and 0.011 vs 0.010 elsewhere);
   understanding why would probably explain #64's magnitude.
4. **Audit the dormant subsystems.** Stereo, mapper and OOS were only read
   where they touch the mono path, and the defect density in what *was* read
   (27 live, 40 dormant) suggests they are not clean.

## 9. Artifacts

| | |
|---|---|
| `notes-bugfix/m0-baseline.md` | baseline, reproducibility (note: its "seed noise ≈ 0.001" figure is superseded by §3.1 and should not be cited) |
| `notes-bugfix/m1-bug-register.md` | all 67 defects, severity, liveness, and the 5 rejected sub-agent claims |
| `notes-bugfix/m2…m6-*.md` | per-milestone reasoning; `m6` §7 is the chaos measurement |
| `notes-bugfix/m7-measurement-and-calibration.md` | the ensemble method, the metric defect, the config sweep, the calibration flags, and **§8: things audited and found sound** |
| `run_ensemble_bugfix.sh` (workspace root) | ensemble harness. **Do not compare single runs.** |
| `rescore_ate.sh` (workspace root) | re-score a finished ensemble's ATE at another `--max_difference` without re-running XIVO (§3.3) |
| `run_eval_bugfix.sh` (workspace root) | reports stock and interpolated RPE, and ATE at both 0.001 and 0.02, side by side |
| `scripts/tum_rgbd_benchmark_tools/evaluate_rpe_interp.py` | the corrected metric |
| `scripts/rpe_interp_dir.py` | per-sequence corrected RPE + the stock metric's artifact floor |
| `cfg/m7_*.json` | the config sweep record |
