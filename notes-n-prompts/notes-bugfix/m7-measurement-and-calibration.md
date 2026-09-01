# M7 — what the numbers actually say

This milestone contains almost no estimator code. It is the milestone where I
stopped trusting the evaluation and went and checked it, and found that **all
three numbers I had been reporting were partly measuring themselves**, not
XIVO. Everything downstream of that — including the honest verdict on M2–M6 —
is in here.

Four results, in decreasing order of how much they change the story:

1. A single run of six sequences does not measure a code change. The run-to-run
   spread is ±0.007 (1 sd) on mean ATE, and the M0 baseline I had been
   comparing everything against was a +3.3 sd outlier. §1–§3.
2. 17% of the reported RPE rotation error is an artifact of the *evaluation
   script*, not of the estimator. Proven by running the evaluator on a
   zero-error trajectory, where it reports 0.2847 deg. §4.
3. ATE was scored on 26% of frames, in contiguous blocks that exclude every
   run's initialization phase, understating the error by ~25%. §9.
4. Neither stretch goal is reachable, and both sit below what the original
   code ever achieved. §7.

§9 was written last and is the one place where a hypothesis I expected to
vindicate the earlier milestones was tested and came back negative.

---

## §1 — XIVO's output is chaotic, and the ensemble that measures it

Established in M6 §7 and used throughout here. Restated because everything
below depends on it:

The filter contains hard accept/reject decisions — `MHGating`'s chi-square
test, depth-validity range checks, group lifetime limits. A perturbation far
below any physical scale flips one of those decisions, which changes the set of
surviving features, which changes the trajectory macroscopically. Measured:
perturbing one expression in `camera_equidist.h::UnProject` by 1e-11 relative
(~5e-9 pixels) moves mean ATE over the six rooms by 0.013.

`run_ensemble_bugfix.sh` turns that from a hazard into an instrument. Each
member perturbs the initial velocity `X.Vsb` by `k * 1e-6` m/s. The
justification is in the script header; the short version is that the config
itself declares `P.Vsb = 0.5`, a prior sigma of 0.7 m/s, so 1e-6 m/s is six
orders of magnitude inside the uncertainty the filter claims for this quantity.
No member is a physically distinguishable scenario. A filter whose answer
depended on this would be inconsistent by its own stated covariance.

**Two independent mechanisms agree on the spread.** Source-level epsilon
(12 pipelines differing by 1 ulp in `rth`): 0.1104 ± 0.0073. Config-level IC
ensemble (8 members): 0.1102 ± 0.0047. Same distribution.

So: **mean-ATE differences below ~0.015 are not attributable to a code
change.** Not "probably noise" — not measurable with a single run.

`XIVO_RANDOM_SEED` is inert for this config (seeds 0/1/2 byte-identical);
nothing draws from the RNG with RANSAC and the sim paths off. The
"seed noise ≈ 0.001" figure in `m0-baseline.md` does not transfer and should
not be cited.

## §2 — the M0 baseline was an outlier, and the M0→M4 story was an artifact

| | mean ATE | sd | min | max |
|---|---|---|---|---|
| M0 single run (what I had been quoting) | 0.1261 | — | — | — |
| M0 ensemble, 8 members | **0.1042** | 0.0067 | 0.0947 | 0.1149 |

The single number I had been treating as "the baseline" is 3.3 sd above the
ensemble mean and lies outside the observed range of eight equivalent runs.
Every per-milestone "improvement" quoted before M7 was measured against it.

Corrected, on an ensemble basis:

| | ATE | RPE_rot | RPE_tra |
|---|---|---|---|
| M0 (`auto`, 8 members) | 0.1042 ± 0.0067 | 0.6202 ± 0.0005 | 0.0286 ± 0.0008 |
| M6 (`559532b`, 8 members) | 0.1102 ± 0.0047 | 0.6198 ± 0.0005 | 0.0273 ± 0.0008 |

**The bug fixes are ATE-neutral, drifting slightly worse** (+0.006, about
2 sem — marginal). They improve RPE_tra by 0.0013 (1.6 sem) and leave RPE_rot
flat. This is the honest headline and it replaces every earlier claim.

That is not an argument that the fixes were wrong. Thirty-odd of them are
provably-wrong code with unit tests to prove it (#1's Jacobian block, #17's
division by a zero singular value, #26's covariance transform written to a dead
local). It is an argument that **ATE on six 2-minute room sequences cannot
resolve them**, because the chaotic term dominates. A correct Jacobian and an
incorrect one draw from distributions whose means differ by less than their
widths.

## §3 — where the chaos comes from, and the config confound

Setting `use_prediction: false` collapses the ensemble spread from sd 0.0047 to
**sd 0.0004**, with per-sequence sd ≈ 0.

That localises the mechanism precisely. `use_prediction` makes the KLT tracker
seed its initial guess from the *filter's* predicted feature position. That
closes a loop: filter state → tracker initial guess → which pixel KLT converges
to → measurement → filter state. Without the loop the tracker is a pure
function of the images and the chaos has nowhere to circulate.

**This also uncovered a confound in my own milestone comparisons.** Bug #3 was
"`use_prediction` is declared in 23 configs and read by nothing". M3 (`1c9e5a8`)
fixed the plumbing — which means M3 *simultaneously* switched the feedback loop
on, because the config said `true` all along. Every evaluation from M3 onward
conflated code changes with that config change. M0 was chaotic despite its
config saying `false`, which is exactly the symptom of the dead key.

For the record, `use_prediction: false` is also much worse: ATE 0.1833 ±
0.0004, with room1 at 0.5304. The feedback loop earns its instability.

## §4 — `evaluate_rpe.py` reports 0.2847 deg of error on a perfect trajectory

`evaluate_rpe.py` pairs each estimate timestamp with the **nearest** ground
truth sample (`find_closest_index`) and does not interpolate. TUM-VI ground
truth is logged at ~120 Hz, so each endpoint of each evaluated 1-second
interval carries up to ±4.17 ms of independent timestamp quantization.

RPE over a 1 s window is brutally sensitive to that. Two measurements:

*Constant timestamp shift* of the estimate changes mean RPE_rot by ~0.11 deg
per ms.

*Decimating the ground truth*, which multiplies the quantization while leaving
the estimate byte-identical:

| GT spacing (ms) | 8.33 | 16.67 | 25.00 | 33.33 | 50.00 |
|---|---|---|---|---|---|
| reported RPE_rot (deg) | 0.6205 | 0.8614 | 0.9951 | 1.2735 | 1.9733 |

An estimator's error cannot depend on how finely the reference was sampled. The
trend is pure measurement artifact.

**The decisive test.** Take the ground truth, interpolate it onto the
estimate's timestamps, and hand *that* to the evaluator as the estimate. This
trajectory has, by construction, exactly zero error. Stock `evaluate_rpe.py`
reports:

```
rotational_error.rmse     0.2847 deg
translational_error.rmse  0.0038 m
```

That is the floor of the metric. `evaluate_rpe_interp.py` on the same input
reports 0.000001 deg, confirming the interpolating implementation has no floor
of its own.

`scripts/tum_rgbd_benchmark_tools/evaluate_rpe_interp.py` removes the term by
SLERP-interpolating the ground truth to each estimate timestamp. The pairing
rule, `ominus`, and the reported statistics are identical — its helpers are
imported from `evaluate_rpe.py` rather than reimplemented, so no second
opinion about the metric definition can creep in.
`scripts/rpe_interp_dir.py` runs it over a run directory and also reports the
stock evaluator's artifact floor per sequence.

**Ruling out the alternative explanation.** A lower number could also come from
evaluating a *different set of pairs* rather than from interpolating — in
particular from silently including or excluding TUM-VI's multi-second mocap
dropouts. Stock `evaluate_rpe.py` drops a pair when either endpoint's nearest GT
sample is more than `2 * gt_interval` away (`evaluate_rpe.py:275`); this script
instead rejects an interpolation spanning a GT gap wider than
`max_gap_factor` median intervals, which for a gap of width W puts the nearest
endpoint at most W/2 away, so `max_gap_factor = 4` is the matching criterion.

Verified rather than assumed. Pair counts, stock vs interpolated: room1
2698/2669, room3 2370/2302, room6 2503/2484 — the interpolated version uses
1–3% *fewer* pairs, so it cannot be inflating the improvement by adding easy
ones. And the result is insensitive to the threshold: tightening
`max_gap_factor` from 4 to 2 changes room1 by 0.000000 and room3 by 0.0007.
Removing the filter altogether (factor 1000) blows the number up to 2.48 / 1.46
/ 1.48 deg, which confirms the dropout filter is load-bearing and that both
implementations correctly exclude the same dropouts.

So the 0.62 → 0.51 change is interpolation, not pair selection.

Corrected M6 numbers:

| | stock | interpolated | artifact |
|---|---|---|---|
| RPE_rot | 0.6198 ± 0.0005 | **0.5120 ± 0.0005** | 0.108 (17%) |
| RPE_tra | 0.0273 ± 0.0008 | 0.0269 ± 0.0008 | 0.0004 (1.5%) |

**This changes the metric, not the estimator.** A lower number here is a more
accurate measurement of the same trajectory, not a better trajectory. Both are
reported everywhere, and the stock number remains the one that is comparable to
published results. The docstring says so explicitly, because this is exactly
the kind of change that turns into an accidental overclaim two months later.

Sub-millisecond offset scans only became possible once the ±4 ms quantization
was gone. The minimum is at +0.5 to +1.0 ms, giving 0.5054 against 0.5124 at
zero — so a residual time offset is a *small* term and the ~0.505 floor is
genuine rotation error.

## §5 — RPE_rot is immovable by tuning

RPE_rot has sd 0.0005 across the ensemble, so unlike ATE it is measurable in a
single run to four digits. Seven single-knob configs, deliberately spread wide:

| config | ATE | RPE_rot |
|---|---|---|
| M6 baseline | 0.1102 ± 0.0047 | 0.6198 |
| `m7_mh_high` (looser gate) | 0.1032 ± 0.0073 | 0.6206 |
| `m7_mh_low` (tighter gate) | 0.1183 ± 0.0087 | 0.6195 |
| `m7_feat_more` | 0.1035 ± 0.0034 | 0.6198 |
| `m7_collinear_05` | 0.1084 ± 0.0098 | 0.6199 |
| `m7_vms_low` | 0.1096 ± 0.0090 | 0.6204 |
| `m7_qimu_gyroexact` | 0.1188 ± 0.0058 | 0.6184 |
| `m7_qimu_exact` | 0.2399 ± 0.0100 | 0.6185 |

Every one lands in **0.6184–0.6206**, a range of 0.002 — four ensemble sd,
while ATE moves by 0.14 and RPE_tra by 0.02. RPE_rot is pinned by something the
config cannot reach.

`m7_qimu_exact` is worth calling out: the shipped config inflates the IMU noise
densities ~3× above the datasheet. Un-inflating them makes ATE **2.3× worse**
(0.2399). The inflation is load-bearing, presumably absorbing unmodelled error;
it is not a mistake to be corrected.

## §6 — the two calibration paths, and what #35 was worth

The M6 fixes to `rodrigues.h` (#35) and the `td` plumbing (#46) exist to make
`USE_ONLINE_IMU_CALIB` and `USE_ONLINE_TEMPORAL_CALIB` correct. Both were
compiled out in the shipped build, which is why those bugs survived. With them
repaired, both flags now build clean and the suite passes — and
`unitTests_Jacobians` grows from 21 to 23 tests, because two Jacobian tests are
themselves ifdef'd behind `USE_ONLINE_IMU_CALIB`.

**#35 promoted from latent to demonstrated.** Installing the original
`common/rodrigues.h` (`git show auto:common/rodrigues.h`) with
`USE_ONLINE_IMU_CALIB` on fails exactly three tests:

```
Rodrigues.dABdAUsesRowMajorFlatteningOnBothSides
Rodrigues.dABdBUsesRowMajorFlatteningOnBothSides
Rodrigues.AccelCalibrationChainMatchesNumericalDerivative
```

The third reproduces `estimator.cpp:726`'s `dV_dCa` chain against a numerical
derivative. So it is a real bug on a real path, not a theoretical one.

*(Method note: the first attempt at this mutation test was void. `git stash push
common/rodrigues.h` reverts to HEAD, and HEAD already contains the fix, so the
"revert" was a no-op and all tests passed. Reverting to `auto:` is the correct
mutation.)*

**End-to-end, both flags are worth much less than the unit tests suggest.**

| build | ATE | RPE_rot_i | RPE_tra_i |
|---|---|---|---|
| M6, neither flag | 0.1102 ± 0.0047 | 0.5120 ± 0.0005 | 0.0269 ± 0.0008 |
| + temporal calib | **0.1071 ± 0.0020** | 0.5120 ± 0.0006 | 0.0261 ± 0.0010 |
| + temporal + IMU calib | 0.1090 ± 0.0051 | 0.5117 ± 0.0006 | 0.0264 ± 0.0009 |

Temporal calibration is a genuine if modest win, and its most interesting
effect is on the *variance*: sd 0.0047 → **0.0020**, less than half. Estimating
the time offset removes one source of systematic per-frame reprojection error,
which makes the gating decisions less marginal, which damps the chaos of §3.
It is enabled in the delivered build. `print_calibration` shows `td` converging
to −7.70e-05 s on room1 — small but real.

Online IMU calibration is **inert and slightly harmful**: it moves nothing
(ATE 0.1090 vs 0.1102 and RPE_rot_i 0.5117 vs 0.5120 are both well inside the
noise) and it undoes temporal calibration's variance reduction, sd 0.0020 →
0.0051, by adding 15 weakly observable states. It is left off, with the
reasoning recorded in `src/CMakeLists.txt` so the next person does not have to
rediscover it.

The single run with the *buggy* `rodrigues.h` and IMU calib on scores
ATE 0.1090 / RPE_rot_i 0.5123 — inside the fixed build's ensemble. So #35 is
fixed because it is provably wrong, not because it improves the benchmark.

**Why IMU calibration cannot help here**, measured rather than assumed. With
`print_calibration` on, after a full sequence:

```
Ca = [[0.999999, -7.2e-07, -1.8e-06], [0, 1, 1.5e-06], [0, 0, 1.00001]]
Cg = [[0.999999, -1.6e-06, 7.9e-06], [-3.5e-06, 1, 7.0e-06], [2.4e-06, 1.5e-06, 1.00001]]
```

`P.Cg = P.Ca = 1e-5` is a prior sigma of 0.32%; the estimates moved ~1e-6, i.e.
**0.03% of one sigma**. The states are effectively frozen. I checked this is not
a broken Jacobian: `estimator.cpp:721,726` populate both `F_` blocks, `dWsb_dCg`
correctly uses the *raw* gyro rather than the calibrated one, and `dV_dCa` goes
through the row-major-consistent chain #35 repaired. The path is live and
correct. TUM-VI's IMU is simply already factory-calibrated, so there is no
scale or misalignment error left for the filter to find.

(`Ca` printing with zeros below the diagonal is correct, not a bug — it is
parameterised as upper-triangular, 6 free parameters. This is what the M6
correction to #35's diagnosis was about.)

## §7 — the stretch goals are below what the original code ever achieved

Reproducing the authors' own shipped results (`misc/results.tar.gz`, not the
stale wiki table) against the delivered build:

| | ATE | RPE_rot | RPE_tra |
|---|---|---|---|
| authors' shipped | 0.1014 | 0.6224 | 0.0325 |
| M7 delivered | 0.1071 ± 0.0020 | 0.6200 ± 0.0006 | 0.0265 ± 0.0009 |
| stretch goal | < 0.06 | < 0.5 | — |

The shipped ATE of 0.1014 is one draw from a distribution of width ~0.007; it
is not distinguishable from either M0's 0.1042 or M7's 0.1071. RPE_rot matches.
**RPE_tra is 19% better** (0.0325 → 0.0265) and that difference is ~7 sem, the
one clearly real end-to-end improvement in the project.

On the goals:

- **ATE < 0.06 m.** Not reached; not close. Best measured mean is 0.1035
  (`m7_feat_more`), and the *best single sequence draw* in any ensemble is
  room6 at 0.0459. Reaching 0.06 as a six-sequence mean needs roughly a 40%
  error reduction, which no bug fix or config in this project moved by more
  than 5% outside the noise band — while the authors' own published number for
  this code is 0.1014. The goal is below what the estimator achieves, not
  below what it *should* achieve given correct code.
- **RPE_rot < 0.5 deg.** Not reached, and the gap is now precisely bounded.
  With the artifact removed the honest figure is 0.5120 ± 0.0006 — 0.012 short,
  which is **20 sd**, so this is a real miss and not a measurement question. §5
  shows config tuning cannot touch it; §6 shows neither calibration path can.
  Note the stock metric's 0.2847 deg floor means the goal is *unreachable as
  literally stated*: no trajectory, including a perfect one, scores below 0.5 on
  `evaluate_rpe.py` and above ~0.28. Under the corrected metric it is at least
  a well-posed target, and we are 2.4% away from it.

## §8 — things I checked and did not report as bugs

Recorded because "audited and sound" is a result, and because each of these
looks wrong at first glance.

**`AddGroupToState`'s four-step covariance augmentation**
(`estimator.cpp:852-863`) appears to alias: step 1 copies the `Wsb` row block
across the full width *including* the destination's own `Tsb` columns, which at
that moment hold a discarded group's stale values. Traced element by element,
it is self-repairing — step 4 overwrites `P_(offset+i, offset+3+j)` from
`P_(offset+i, Tsb+j)`, which step 1 had already set correctly, and the
symmetric argument covers the other three sub-blocks. Correct as written.

**`P_.setIdentity(kFullSize, kFullSize)`** leaves variance 1.0 on every
not-yet-allocated feature, group, and (with camera calib off) camera-intrinsics
slot, and discarded slots retain their old covariance until reused. Inert:
`J_` is zeroed per call (an M6 fix) and `H` per update, and both are only
written at live offsets, so stale columns are always multiplied by zero.

**The gyro-bias measurement Jacobian** `J_.block<2,3>(0, Index::bg)`
(`feature.cpp:740`) is guarded by `USE_ONLINE_TEMPORAL_CALIB`, which looks like
a core motion state accidentally hidden behind a calibration flag. It is
correct: the vision measurement depends on `bg` only through the `gyro * td`
forward-rotation of the state to the image time, so the block is proportional
to `td` and vanishes identically when `td ≡ 0`. Same argument for the `Cg`
block. Gyro bias remains observable through the propagation Jacobian either
way.

**`td` converges to −0.077 ms while the offset scan of §4 prefers +0.75 ms.**
Not a contradiction: the scan optimum also absorbs any camera-to-ground-truth
clock offset in the dataset itself, which is not the camera-to-IMU offset `td`
models and which the estimator cannot observe.

## §9 — the third measurement defect: ATE on 26% of frames (#67, found in M8)

Found after this milestone was written, while cross-checking against notes from
the sibling stereo branch. `run_and_eval_pyxivo.py:30` defaults
`-ate_max_difference` to **0.001**; `evaluate_ate.py`'s own default is 0.02.

Images are 20 Hz (period 50.158 ms), ground truth ~120 Hz (8.333 ms) — ratio
6.019 — so the image-to-GT phase offset drifts slowly and a 1 ms association
window matches only ~26% of frames (2589 pose pairs per sequence at 0.02 vs
~720 at 0.001). Because the drift is slow the matched frames fall in contiguous
blocks, and on room1 they start 23 s into a 141 s run: **initialization is
entirely excluded**, which is where the largest errors are.

Re-scoring the same trajectories with `rescore_ate.sh` (no re-running):

| | ATE@0.001 | ATE@0.02 |
|---|---|---|
| M0 baseline | 0.1042 ± 0.0067 | 0.1290 ± 0.0114 |
| M8 delivered | 0.1071 ± 0.0020 | 0.1377 ± 0.0035 |

I expected this to explain why the fixes look ATE-neutral — the triangulation
and lifecycle fixes target exactly the initialization phase that 0.001 throws
away. **It does not.** The honest protocol shows the same flat-to-slightly-worse
mean. Hypothesis tested, rejected, recorded.

The useful part: the **3.3× variance reduction is protocol-independent**
(0.0114 → 0.0035 at 0.02; 0.0067 → 0.0020 at 0.001). Two association windows
selecting two different frame subsets give the same factor, so it is a property
of the estimator, not of either metric. This is the strongest end-to-end result
in the project alongside RPE_tra.

`-ate_max_difference` was parsed and ignored on `auto` (#53) and is now
honoured; the default stays 0.001 so earlier numbers remain comparable, and both
are reported. `run_eval_bugfix.sh` prints an `ATE_02` column next to the 0.001
one by default — it re-scores the trajectory files the run already wrote, so it
adds no XIVO time (`NO_ATE02=1` skips it). Verified on room1: 0.1328 at 0.001
vs 0.1676 at 0.02, and the existing four columns are byte-identical, because the
new column is appended *after* the `grep … | tail -1` parsing of the run log.
Leaving it behind a flag would have reproduced the original failure — the flag
existed the whole time; what was missing was a reason to set it.

## §10 — what the delivered build is

- `USE_ONLINE_TEMPORAL_CALIB` on, `USE_ONLINE_IMU_CALIB` off (§6), both
  documented in place.
- `evaluate_rpe_interp.py` + `rpe_interp_dir.py` added; `run_eval_bugfix.sh`
  reports interpolated RPE alongside the stock numbers by default
  (`NO_INTERP=1` to skip) and ATE at both association windows (`NO_ATE02=1`).
- `run_ensemble_bugfix.sh` for any future comparison. **Do not compare single
  runs.**
- `cfg/m7_*.json` kept as the sweep record for §5; `cfg/_ensemble/` is
  generated and gitignored.
