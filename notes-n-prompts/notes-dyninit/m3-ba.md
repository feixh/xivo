# M3 -- the initialization bundle adjustment

Stage B is the joint optimization the task called for: given Stage A's velocity
and gravity as a seed, refine the whole window -- every frame's attitude,
position and velocity, both IMU biases, and every feature -- against the pixel
error and the preintegrated IMU together.

| file | what it is |
|---|---|
| `src/init_ba.{h,cpp}` | the state, the residuals, the Jacobians, and the LM/Schur solver |
| `src/test/unittest_init_ba.cpp` | 8 tests |
| `src/test/init_test_fixture.h` | the synthetic fixture, shared with M2 |

`bin/linear_probe -ba` runs Stage A then Stage B on one real window;
`harness/linear_check.py --ba` scores both against EuRoC ground truth.

`ctest`: **26/26 green.**

## Headline result

On all 11 EuRoC sequences, at the shipped defaults, with no per-sequence tuning:

| | Stage A | Stage B | change |
|---|---|---|---|
| velocity error, mean | 0.1727 m/s | **0.0164 m/s** | **-90.5%** |
| velocity error, max | 0.2257 m/s | 0.0459 m/s | |
| gravity tilt, mean | 4.148 deg | **0.892 deg** | **-78.5%** |
| gravity tilt, max | 5.682 deg | 2.606 deg | |
| gyro bias error, mean | (not estimated) | **0.00292 rad/s** | vs a true \|bg\| ~ 0.08 |

Every one of the 11 improves on both metrics. The gyro bias figure is an
**external check**: EuRoC solves for the biases in its own batch estimator and
publishes them in `state_groundtruth_estimate0` columns 11-13, so agreement is
not self-consistency. 0.0029 against 0.08 is a 3.6% recovery from a zero seed.

The two sequences that actually matter are the two the M1 detector routes to the
dynamic branch:

| sequence | \|v\|gt | v: A -> B | tilt: A -> B | bg error |
|---|---|---|---|---|
| MH_01_easy | 0.802 | 0.1443 -> **0.0459** | 4.166 -> **0.829** deg | 0.00146 |
| MH_02_easy | 0.473 | 0.1087 -> **0.0184** | 4.139 -> **0.973** deg | 0.00083 |

The other nine are a generalization check rather than a requirement -- the
detector sends them to the static path -- but Stage B improves all nine too,
which is the evidence that a later re-tuning of the detector thresholds cannot
quietly make things worse.

Repro, from the worktree root with the venv on `PATH`:

```
python3 notes-n-prompts/notes-dyninit/harness/linear_check.py \
    --root ../data/euroc --cfg cfg/euroc_stereo.json --auto-start --ba
```

## The formulation, and the two things it gets right by construction

Unknowns live in a **gravity-aligned world frame `W`** (z up): per frame
`R_{W<-Ik}`, `p`, `v`; global `bg`, `ba`; per track `p_F^W`.

**Gravity is not a variable.** It is `[0,0,-9.81]` by the definition of `W`, and
its two free directions are frame 0's roll and pitch, which are ordinary
unknowns. So the direction is still refined, but `|g|` cannot drift and there is
no norm constraint to maintain -- which was Stage A's main piece of machinery.

**The remaining 4-dof gauge is removed exactly.** Global translation is held by
pinning `p_0 = 0` (zeroing its rows and columns). World yaw is an exact nullspace
of every data term -- a yaw `Q` satisfies `Q g_W = g_W`, so every residual is
invariant under `p,v,f -> Qp,Qv,Qf` with `R -> QR` -- and is pinned by one prior
row at `sigma_yaw = 1e-3`. Because the direction is an exact nullspace, that
prior changes the answer not at all, which is why a modest weight is right and
the usual 1e6-weight pin (a nullspace traded for a bad condition number) is not.
`WorldYawIsAGaugeFreedomAndTiltIsNot` checks both halves: yaw leaves the cost
invariant to 1e-6, a 0.05 rad tilt about world x raises it by more than 1.

Residuals: `r_alpha`, `r_beta`, `r_theta` per consecutive frame pair, and a
Cauchy-robustified reprojection per observation. The preintegrals are corrected
to the current bias to first order through the Jacobians `init_preint` carries --
that correction is what makes the biases *observable* rather than merely present.

Solver: LM with the Schur complement on the track block, Marquardt's
`lambda*diag(H)` rather than `lambda*I` (these parameters differ in scale by six
orders of magnitude), hand-rolled in Eigen.

## The accelerometer bias is deliberately not estimated

This is the substantive design decision of M3, and it was made by measurement,
against the intention I started with.

Left free, `ba` does not stay near zero -- over a 1.5 s window it is nearly
indistinguishable from a gravity tilt, and it takes that interpretation. Measured
unpriored at the shipped `cauchy_c = 3`: `|ba|` reaches **1.98 m/s^2 on MH_01 and
2.75 on MH_02**, against true values of 0.158 and 0.161 -- 12x to 17x too large,
and it drags the recovered gravity with it. (With the robust loss also off it
reaches ~9 m/s^2, i.e. gravity-sized; the two defects compound.) Sweeping the
prior:

| `sigma_ba_prior` | velocity mean | tilt mean | tilt max | `ba` error |
|---|---|---|---|---|
| 0 (free) | 0.0251 | 6.95 deg (**+68% vs seed**) | 20.4 deg | 1.180 |
| 0.2 | 0.0357 | 3.12 deg (-25%) | 16.2 deg | 0.539 |
| 0.1 | 0.0325 | 2.22 deg (-46%) | 11.2 deg | 0.386 |
| 0.05 | 0.0255 | 1.48 deg (-64%) | 5.20 deg | 0.266 |
| **0.01** | **0.0190** | **0.90 deg (-78%)** | **2.61 deg** | 0.180 |
| 0.003 | 0.0303 | 0.90 deg (-78%) | 2.59 deg | 0.178 |
| 0.001 | 0.0367 | 0.91 deg (-78%) | 2.60 deg | 0.179 |

(at `-ba_iters 200` so the comparison is not about budget; `0.01` is the optimum
-- tighter starts costing velocity as `ba` can no longer absorb anything.)

So `sigma_ba_prior = 0.01` ships, and the honest reading of it is **"`ba` is not
estimated"**. That is the same conclusion VINS-Mono reaches and states outright.
The `ba` error column is essentially `|ba|gt` throughout: we are reporting the
true bias as the error because we hold the estimate at zero.

**It costs exactly what theory says it should, which is how we know the 0.90 deg
is a floor and not solver slack.** With `ba` held at zero, a bias `b`
perpendicular to gravity *must* appear as `|b|/9.81` radians of tilt. Across the
11 windows:

| | \|ba\|gt | predicted tilt | measured tilt |
|---|---|---|---|
| MH_01 | 0.151 | 0.881 | 0.829 |
| MH_02 | 0.153 | 0.895 | 0.973 |
| MH_03 | 0.136 | 0.795 | 0.714 |
| MH_04 | 0.198 | 1.155 | 0.625 |
| MH_05 | 0.135 | 0.788 | 0.600 |
| V1_01 | **0.551** | **3.219** | **2.606** |
| V1_02 | 0.136 | 0.794 | 0.472 |
| V1_03 | 0.201 | 1.173 | 0.768 |
| V2_01 | 0.139 | 0.811 | 0.916 |
| V2_02 | 0.094 | 0.548 | 0.501 |
| V2_03 | 0.090 | 0.526 | 0.903 |

Pearson r = **0.93**, least-squares slope through the origin **0.82** -- below 1
because `ba`'s component *along* gravity tilts nothing. V1_01, whose true `|ba|`
is 0.55 where every other sequence sits near 0.15, is duly the worst window.
Beating 0.9 deg needs a longer window, not a better optimizer.

The gyro bias is the opposite case and gets the opposite treatment: it enters
`r_theta` directly, is recovered to 3.6%, and sweeping `sigma_bg_prior` over two
decades (0.02 / 0.1 / 1.0) moves the velocity by 0.003 m/s and the bias error not
at all. **It ships with no prior** -- adding one would have been cargo cult.

## The robust loss earns its place on real data, not on synthetic data

`cauchy_c` sweep, real windows, shipped otherwise:

| `cauchy_c` | velocity mean | velocity max | tilt mean | tilt max | bg error |
|---|---|---|---|---|---|
| 0 (off) | 0.0527 | 0.2133 | 1.605 deg | 6.140 deg | 0.00843 |
| 1.5 | 0.0192 | 0.0457 | 0.904 deg | 2.591 deg | 0.00243 |
| **3** | **0.0170** | 0.0459 | 0.896 deg | 2.606 deg | 0.00283 |
| 5 | 0.0159 | 0.0464 | 0.901 deg | 2.608 deg | 0.00358 |

3x on velocity and on the gyro bias, and the max cases are where it shows.
1.5-5 are all comparable; 3 is the middle and ships. (10 was also tried and is
clearly worse: 0.0271 and +37% tilt.)

On the *synthetic* fixture the loss looks useless, and understanding why was
worth the detour. My first outlier model applied a constant offset to every
observation of a track -- but that is almost consistent with the track's 3D point
having moved, so the solve absorbs it into the feature position and there is
little for the loss to do. A real KLT failure jumps *partway* through a track,
which no single 3D point can explain. With that model, and measuring the right
thing, the effect is clear -- not a better number, but **bounded influence**:

```
mode 0 (whole track, 10/25/60 px):  cauchy 0.0812 0.0816 0.0822 | plain 0.0704 0.0718 0.0690
mode 1 (mid-track,   10/25/60 px):  cauchy 0.0829 0.0806 0.0806 | plain 0.0682 0.0456 0.1375
```

The robust spread across all six is 0.0023 m/s; the non-robust spread is 0.0919,
**40x larger**. The non-robust answer is *better* in four of the six -- which is
exactly why a test asserting "Cauchy improves the velocity" would have been
measuring the seed and not the mechanism. `CauchyBoundsTheInfluenceOfOutliers`
asserts stability, and the monotone version of the same story in the median
reprojection error (pinned at 0.35 px with the loss; degrading 0.374 -> 0.443 ->
0.682 without).

Worth noting: that stability needs **both** the loss and the bias prior. Freeing
`ba` in the same test takes the robust spread from 0.0023 to 0.0368.

## The tests

8 tests, all passing. Two exist to catch the failure modes a converging solver
hides completely -- a wrong derivative (it still descends, just elsewhere) and a
residual that does not actually depend on the biases (it reports the seed's).

1. **`AnalyticJacobiansMatchCentralDifferences`** -- every column, including both
   prior families and the yaw gauge row, to **1e-6 relative**. The dense rows are
   emitted by the *same* accumulation call as the sparse ones, so this tests the
   assembler the solver runs rather than a parallel implementation of it. Run at
   `cauchy_c = 0`: IRLS reweighting makes the residual a different function of the
   state than the one the Jacobian differentiates, which is the standard
   approximation and not a bug, but it does mean a finite-difference check has to
   be run on the underlying least-squares residual. Evaluated away from the
   optimum and at a *wrong* bias prior, where the bias Jacobians are load-bearing.
2. **`WorldYawIsAGaugeFreedomAndTiltIsNot`** -- above.
3. **`RecoversTheExactStateFromATiltedSeed`** -- see the correction below.
4. **`CostNeverIncreasesWithAnExtraIteration`** -- re-solves at budgets 0..12 and
   checks the cost never rises. Tested from outside, by re-running, so that the
   iteration count is the only thing that varies; an internal per-iteration log
   could not make that claim.
5. **`RecoversAPlantedBiasUnderPixelNoise`** -- 6 seeds, bias planted at EuRoC's
   magnitude, preintegrals built at a zero prior so the bias must come out of the
   optimization. Measured: mean `|bg|` error **8.4e-5** rad/s (max 1.3e-4) against
   a planted 0.0814, `|ba|` 1.1e-2 against 0.1475, velocity 3.0e-2. Dropping the
   rotation's bias Jacobian alone takes the gyro error to 0.08 -- i.e. no recovery
   at all -- so the tolerance is tight enough to be load-bearing.
6. **`ImprovesOnStageAEndToEnd`** -- Stage A through `SeedBAState` into Stage B.
7. **`CauchyBoundsTheInfluenceOfOutliers`** -- above.
8. **`DefaultPriorHoldsTheAccelBiasNearZero`** -- pins the shipped prior so that a
   later "let's just relax that" cannot pass silently. Stated as a comparison
   (0.091 priored vs 0.146 unpriored on the same window) rather than as an
   absolute, because how far the prior drags `ba` back depends on how well the
   window determines it, and a synthetic window determines it far better than a
   real one.

### Correction to the plan's gate (b)

The plan asked for `v` within 1e-6 m/s on noiseless data. **That gate is not
attainable and should not have been written.** The solver reaches a cost of
1.41e-22 -- *below* the 1.76e-22 the exact truth scores -- with a reprojection
RMS of 1.6e-13 px, and still sits 0.0094 m/s from the true velocity.

Both facts hold at once because the window has a nearly flat direction: a global
scale on `(p, v, f)` trades against an accelerometer bias along the mean
acceleration. Both signatures are present -- the recovered scale is **1.00686**,
and dividing it out leaves **5.2e-4 m/s**, while the `ba` error lies mostly along
`a_w` (`dba . a_hat = -0.75`). So to the last bit of double precision the state
found fits the data at least as well as the state that generated it, and this is
not something a better solver fixes: 4x the angular rate makes it worse (0.034),
8x the acceleration makes it worse (0.036), and only a longer span helps (0.0056
at 2 s).

The test now asserts the identifiable parts -- `bg` at **1.4e-16**, gravity tilt
at 4.9e-4 rad, the scale within 2% of 1, the scale-corrected velocity at 5.2e-4,
and the cost no worse than the truth's. This is the same flat direction that
`sigma_ba_prior` exists to close, met in its cleanest possible form.

## Two traps worth recording

**`pixel_rms` alone is a misleading diagnostic.** It is unrobustified by design
(so it stays comparable across configurations), which means a handful of outlier
tracks among ~4000 observations makes a perfectly converged MH_01 window read
9.4 px. The same window's median is 0.335 px. `BAResult` now carries both, and
they answer different questions: the median says whether the bulk of the window
fits, the RMS says how much gross outlier energy the robust loss is carrying.

**A probe must not shadow the default it is validating.** `linear_probe`
originally declared `-sigma_ba_prior` with a default of 0 to match the older
library default, and assigned it unconditionally. After the library default
changed to 0.01, an unflagged run silently measured the *unpriored* solve and
reported 4.5 deg of gravity tilt as the shipping configuration. Every Stage B
flag is now applied only if `gflags::GetCommandLineFlagInfoOrDie(name).is_default`
is false, so an unflagged run measures `BAOptions{}` and cannot drift from it.

## One real bug this milestone found

**LM was buying cost reductions by deleting rows.** The min-depth guard was
`if (Xc(2) < opt_.min_depth) continue;`, which makes the *number of residual rows*
a function of the state. LM accepts any step that lowers "the cost", and it duly
lowered it by pushing features behind the camera and deleting their rows: from a
Stage A seed 1.2 m/s off, the cost fell to 1.28e-10 with a 1.37 m/s velocity
error and a pixel RMS of 2.4e-7 on data carrying 0.3 px of noise, because most of
the observations had quietly left the problem. Fixed by *clamping* the depth
instead of skipping, with the matching exact derivative for the clamped branch, so
the residual stays a continuous function of the whole state and descent means what
it says. `AnalyticJacobiansMatchCentralDifferences` asserts the row count is
unchanged by every perturbation, which is what would have caught it.

## Known and left alone

* **MH_04's window is the visually hardest of the 11** -- median reprojection
  error 0.76 px where the others sit at 0.03-0.34, whitened IMU RMS 2.4 where the
  others are 0.3-1.0, and the worst gyro bias error (0.012). Its IMU stream is not
  at fault: dt is exactly 0.005 s with zero gaps, and the motion is only mildly
  more aggressive than MH_01's. It is also **routed to the static branch** by the
  M1 detector (`|v|gt = 0.0056` -- the platform is nearly stationary there), so
  this window will not reach Stage B in the shipped system. Recorded, not chased.
* **The LM accepts roughly one step per rejection on real data** (e.g. 30 accepted
  / 28 rejected). That is a naive damping policy, not a derivative bug -- the
  Jacobian test rules that out, and the answer at 25 iterations already matches
  the answer at 200 (0.0162 vs 0.0176 mean velocity). A better policy would save
  time in M5's throughput accounting, not accuracy.
* `max_iterations = 30` ships. 25 is already converged; 60 and 200 buy nothing.
