# M6 -- is the bundle adjustment's own covariance a better prior than the config's?

M5 shipped the feature. The handoff writes four quantities into the filter --
`Vsb`, the gravity-implied attitude, `bg`, `ba` -- and their **covariance** still
comes from three fixed numbers in the config, `P.Vsb`, `P.bg`, `P.ba`, identical on
every sequence. The bundle adjustment has just solved a 41-frame window and can
report its own marginal over the same three quantities, so the obvious next step is
to hand *that* to the filter: a filter given an easy window starts confident, one
given a hard window starts humble, and the config prior stops having to cover both.

The plan committed in advance: **ships only if it helps**; a negative result gets
written up and reverted, like the seventh branch in the TUM-VI round. It does not
help. This note is the measurement, and it is worth more than the feature would
have been, because it says *why* -- the matrix is confident exactly where it is
right and wildly overconfident where it is wrong, which is the one failure mode a
seeded covariance cannot survive.

| file | what it is |
|---|---|
| `src/init_ba.cpp` `ReducedMarginal` | the marginal, by Schur elimination of the track blocks |
| `src/init_ba.h` `BAOptions::want_covariance`, `BAResult::cov` | the switch and the result; off by default |
| `bin/linear_probe -ba -cov` | prints the 9x9, rotated into the last frame's body axes |
| `harness/m6_cov.py` | scores it against groundtruth: scale, rank, and the incumbent by the same statistic |
| `src/test/unittest_init_ba.cpp` `MarginalCovarianceMatchesTheDenseHessian` | the elimination against a dense inverse |

Everything in that table survives. What was written and then reverted is the
consumer: an `InitDecision::cov`, a `dynamic_init.seed_covariance` flag, and an
`Estimator::SeedCovarianceFromDynamicInit` that overwrote the `(Vsb, bg, ba)` block
of `P_`. It was implemented in full and measured rather than argued away, because
the argument below would otherwise have been a prediction.

## What the covariance is, and one trap in reading it

The BA's normal equations are `H = J'J` over 9 columns per frame plus 3 for `bg`,
3 for `ba`, and 3 per participating track. `ReducedMarginal` eliminates the track
blocks -- each is 3x3 and independent, so this is exact and cheap -- and inverts the
reduced system, giving the joint covariance of the frames and the biases. Two
things about it are easy to get wrong:

- **The gauge.** Translation is pinned by fixing `p_0` (its rows and columns are
  replaced by an identity), and world yaw is pinned by a prior row whose Jacobian
  is a function of `R_0 R_ref,0'` -- the *seed's* attitude, not the solution's.
  Referencing it anywhere else is a different Hessian. This is not pedantry: the
  unit test first disagreed with the dense inverse by 1e-3 relative for exactly
  this reason, which is precisely the size of disagreement that would otherwise be
  blamed on the elimination.
- **The frame.** The velocity block is only gauge-invariant in body coordinates,
  so both `linear_probe -cov` and the reverted filter arm report `R_k' Sigma_v R_k`
  for the handoff frame `k`. `bg` and `ba` are body quantities already.

`MarginalCovarianceMatchesTheDenseHessian` builds a K=9, 0.8 s synthetic window with
the robustifier off, solves it, re-linearizes at the solution with the *seed* as the
gauge reference, forms `(J'J)^-1` densely, and compares all 81 entries of the
9x9 selection correlation-scaled. Worst disagreement **6e-6**, gate at 1e-4. So the
negative result below is about the cost function's self-assessment, not a bug in
computing it.

## Result 1 -- it is not calibrated, and the miscalibration is not a constant

`m6_cov.py` runs the shipped 41-frame window on all eleven EuRoC sequences and
scores each block against `state_groundtruth_estimate0` at the window's last frame
-- the same source and the same instant M5 scored the seeds at. The statistic is the
normalized Mahalanobis distance `z = sqrt(e' C^-1 e / 3)`: **1 is calibrated,
greater than 1 is overconfident**, and it is scale-free, so the three blocks are
comparable.

Mid-flight (`--start 55`, the eleven windows M5's evaluation actually uses):

| block | median z | mean z | max z | rms e/s | spearman(s, e) |
|---|---|---|---|---|---|
| v | 12.7 | 98.7 | 475.6 | 251.6 | +0.01 |
| bg | 13.2 | 23.6 | 84.4 | 52.8 | -0.22 |
| ba | 11.8 | 16.4 | 39.7 | 29.6 | -0.47 |

A median z of 12 means the reported variance is ~150x too small. That alone is
survivable -- one inflation constant per block fixes a constant bias, and `rms e/s`
is exactly that constant. What is not survivable is the *spread*: `z_v` runs from
3.8 to 475.6. On MH_04 the window's true velocity error is 1.33 m/s and the matrix
reports a sigma of 0.0017 m/s.

At a table start (`--start 0`, ten scorable windows -- MH_03 has no groundtruth at
its handoff) the same screen looks completely different:

| block | median z | mean z | max z | rms e/s | spearman(s, e) |
|---|---|---|---|---|---|
| v | 1.8 | 4.6 | 26.9 | 15.4 | +0.59 |
| bg | 6.2 | 7.5 | 18.8 | 13.7 | -0.24 |
| ba | 8.9 | 11.4 | 31.9 | 23.3 | +0.31 |

Velocity is nearly calibrated there (median z 1.8) -- and that is the problem in one
line. **The covariance is honest on the easy windows and wrong by two orders of
magnitude on the hard ones.** A single inflation constant tuned to fix MH_04 makes
the ten easy windows uselessly humble; one tuned for the easy windows leaves MH_04
at z=475.

The `ba` block behaves as `init_ba.h` predicts and for the documented reason: `ba`
is priored at 0.01 m/s^2 over a window this short rather than estimated, so its
block reports the prior back (sigma 5.9e-3 to 8.3e-03 across every window, a 1.4x
spread) while the true errors span 0.05 to 0.55 m/s^2, an 11x spread. That block is
not an estimate's uncertainty at all.

## Result 2 -- it does not rank the windows; the residuals do

Scale is fixable with a constant. What a seeded covariance offers that a config
prior *cannot* is per-window ranking: even at the wrong scale, a matrix that knows
which window was hard lets one constant serve every sequence. That is the whole
case for M6, and it fails outright -- Spearman(sigma, |e|) mid-flight is **+0.01,
-0.22, -0.47**. No information for velocity, and `ba` actively anti-correlated: the
windows where it reports the largest sigma are the ones where it is most nearly
right. A seeded covariance that ranks at zero is a config prior with extra steps.

The secondary finding is where a future attempt would have to start. The solve's
own *residuals* do rank difficulty, and they rank it well:

| predictor | e_v | e_bg | e_ba |
|---|---|---|---|
| sigma (block) | 0.01 | -0.22 | -0.47 |
| pixel median | **0.75** | 0.45 | -0.12 |
| imu rms | 0.69 | 0.35 | 0.00 |
| sigma * imu rms | **0.75** | 0.41 | 0.00 |
| sigma * pixel median | 0.58 | **0.55** | -0.18 |

(mid-flight; at `--start 0` `sigma * imu rms` reaches 0.89 for `e_v`.) Both
quantities are already computed and already logged -- `stage_b.pixel_median` and
`stage_b.imu_rms` -- and `max_pixel_median` already gates on the first. So a
*scalar* per-window inflation of the config prior, driven by the reprojection
median, is a live idea that this note does not test. What it does establish is that
the covariance matrix is not the vehicle for it.

## Result 3 -- the incumbent is already right-scaled

The comparison M6 owed the config priors, by the identical statistic. `P.*` entries
are **standard deviations**, not variances -- `estimator.cpp` does `P_ *= P_;` after
filling the diagonal -- so euroc ships sigma_v = 0.5 m/s, sigma_bg = 0.01 rad/s,
sigma_ba = 0.25 m/s^2. Scored as isotropic 3x3 blocks over the same eleven
mid-flight windows:

| block | median z | mean z | max z | rms e/s |
|---|---|---|---|---|
| v (prior 0.5 m/s) | 0.06 | 0.32 | 1.53 | 0.97 |
| bg (prior 0.01 rad/s) | 0.34 | 0.62 | 1.89 | 1.42 |
| ba (prior 0.25 m/s^2) | 0.37 | 0.43 | 1.24 | 0.88 |

`rms e/s` within a factor of 1.5 of unity on all three blocks, and max z below 1.9
-- these numbers are close to what a well-tuned prior looks like, covering the worst
window without being absurd on the best. They cannot rank anything, by construction,
and they do not need to: an unranked prior at the right scale beats a ranked-at-zero
one at 1/150th of it.

## Result 4 -- the filter, measured

The consumer was still built and run, because "ships only if it helps" deserves an
empirical answer. `SeedCovarianceFromDynamicInit` inflated each block's *sigma* by
the measured `rms e/s` (250, 50, 30) as a congruence `K C K'` -- preserving the
cross-correlations the solve reported -- clamped any block whose inflated variance
exceeded its config prior back to that prior, so the seeded matrix could only ever
be *more* confident than the config, never less, and refused to touch `P_` at all
if the result was not positive definite. That is the most favourable form of the
idea available: right-scaled on average, and fenced against the worst case.

Two arms, `dynamic_init.enabled=true` against the same plus
`seed_covariance=true`, n=3 ensembles jittered by `ptsb`, all eleven sequences
started 55 s in, both modes:

| metric | on | on_cov | delta |
|---|---|---|---|
| stereo ATE (m) | 0.078 | 0.083 | **+0.005** |
| stereo ov ATE pos (m) | 0.085 | 0.090 | +0.005 |
| stereo ov ATE ori (deg) | 1.78 | 1.80 | +0.02 |
| stereo RPE8 pos (m) | 0.111 | 0.115 | +0.004 |
| stereo RPE8 ori (deg) | 1.00 | 1.03 | +0.03 |
| mono ATE (m) | 0.204 | 0.208 | **+0.004** |
| mono ov ATE pos (m) | 0.213 | 0.217 | +0.004 |
| mono ov ATE ori (deg) | 2.19 | 2.26 | +0.07 |

Ten of ten metrics worse or flat, none better. Per sequence the mean delta is
+0.0054 +- 0.0048 m stereo and +0.0045 +- 0.0056 m mono, which taken alone is
inside the ensemble's own noise -- but the aggregate is not the argument, the tails
are:

| sequence | mode | on | on_cov | delta |
|---|---|---|---|---|
| V2_03_difficult | stereo | 0.1151 +- 0.0013 | 0.1675 +- 0.0500 | +0.0524 |
| V2_03_difficult | mono | 0.1046 +- 0.0015 | 0.1463 +- 0.0020 | **+0.0418 +- 0.0025** |
| MH_01_easy | mono | 0.1074 +- 0.0037 | 0.1451 +- 0.0014 | **+0.0377 +- 0.0039** |
| MH_01_easy | stereo | 0.0453 +- 0.0016 | 0.0522 +- 0.0029 | +0.0069 |

Those two mono cells are 10-15 standard errors wide. They are not reshuffling, they
are the feature. Best case anywhere in the table is -0.0155 m (V1_03 mono), so the
distribution is a small symmetric wash plus a hard left tail.

`MH_04_difficult` and `MH_05_difficult` are **bit-identical** in both arms, on every
metric, in both modes. That is not a null result, it is the dispatcher working: 55 s
into those two sequences the window's reprojection median is 9.8 px and 1.6 px
against a `max_pixel_median` of 1.5, so both fall back to the static path and no
covariance is ever seeded. So the delta above is carried by **nine** sequences, not
eleven, and over just those nine it is +0.0066 m stereo and +0.0056 m mono.

The mechanism, from a diagnostic-build run of the two worst cells:

- MH_01 at t=55 s: seeded sigma_v = (0.245, 0.293, 0.500) m/s after inflation and
  clamping, against a true velocity error of **0.41 m/s**. The config prior of 0.5
  covers it; the seeded matrix is 1.7x too tight on two of three axes, on the
  sequence with the largest mono regression.
- V2_03 at t=58.9 s: seeded sigma_bg = (0.010, 0.0058, 0.0069) rad/s against a true
  gyro-bias error of **0.0159 rad/s** -- 2.3x too tight -- on the sequence with the
  largest regression in both modes.

In both cases the clamp is what kept it from being worse: it bites on one axis of
every block in every run observed. An unclamped seed is the version that turns a
degradation into a divergence, and it was never worth running.

## Verdict

Reverted. `P.Vsb`, `P.bg` and `P.ba` remain the filter's initial covariance on the
dynamic path, and there is no `seed_covariance` knob to find.

What ships is the diagnostic and the record: `ReducedMarginal`, `BAResult::cov`
behind a `want_covariance` that only `linear_probe -cov` ever sets, the unit test
that pins it to a dense inverse, `m6_cov.py`, and a `comment6` in both euroc
configs stating the outcome with its numbers so this is not re-attempted blind.
Nothing in the estimator or the dispatcher reads any of it, so the shipped code
path is unchanged -- verified, not assumed: after the revert the tree reproduces
the `on` arm's trajectory **byte for byte** on MH_01 (dynamic), V2_03 (dynamic,
the worst regression) and MH_04 (static fallback), and `ctest` is 26/26.

The one idea M6 leaves alive is Result 2's: a scalar inflation of the config prior
driven by the reprojection median, which ranks the true velocity error at 0.75
where the covariance ranks it at 0.01. That is a different milestone, and it does
not need a 9x9.

## Repro

```sh
# from the worktree root, venv on PATH
python3 notes-n-prompts/notes-dyninit/harness/m6_cov.py --start 55   # Results 1-3
python3 notes-n-prompts/notes-dyninit/harness/m6_cov.py --start 0
./bin/unitTests_init_ba --gtest_filter='*MarginalCovariance*'        # the elimination
./bin/linear_probe -cfg cfg/euroc_stereo.json -root ../data/euroc -dataset euroc \
    -seq MH_04_difficult -start 55 -frames 41 -ba -cov -at_frame -1
```

Result 4 is `results/dyninit/m6-n3/` and cannot be re-run against this tree -- the
`on_cov` arm needs the reverted commit's parent. Its command was

```sh
./notes-n-prompts/notes-dyninit/harness/m5_ensemble.sh --members 3 --mode both \
    --start-sec 55 --out .../results/dyninit/m6-n3 \
    --base-name on --base-patch 'dynamic_init.enabled=true' \
    --tag cov --patch 'dynamic_init.seed_covariance=true'
```

which is also why `m5_ensemble.sh` grew `--base-name` / `--base-patch`: a milestone
asking whether one more thing on top of the shipped feature helps has to compare
against the shipped feature, not against its absence, or every number it prints is
the M5 effect again.

Screens saved: `results/dyninit/m6-cov/cov_t55_w41.txt`,
`results/dyninit/m6-cov/cov_t0_w41.txt`.
