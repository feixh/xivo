# M6 — numerical algorithms, camera models, and plumbing

Covers bug-register entries #32–#54. Branch `auto-bugfix` of the `xivo-bugfix`
worktree. All 66 unit tests green for the first time in the project's history
(the two shipped failing tests, `SlowAndFastGivensMatch` and the atan
round-trip, are both fixed here).

The headline result of this milestone is not a fix, it is a *measurement*: see
§7. The end-to-end ATE numbers I have been quoting per milestone are mostly
inside this pipeline's chaotic noise band, and the band is ~3x wider than the
figure recorded in `m0-baseline.md`.

---

## 1. Givens / nullspace projection (#33, #34, #50, #54)

`SlowGivens` is the MSCKF nullspace projection: given the 2M x 3 feature
Jacobian `Hf`, it builds a basis for the left nullspace of `Hf` and applies it
to `Hx` and the innovation. Two independent defects:

* **#34 — the projection was not a projection.** The hand-rolled sequence of
  rotations did not actually annihilate `Hf`, so `Hx` retained a component
  along the feature's own error direction. The MSCKF update then treated
  feature-position error as pose error.
* **Heap corruption.** `oos_.Hx` is a fixed `2*kMaxGroup = 30`-row buffer.
  `SlowGivens` resized its working copy and wrote past the buffer for tracks
  with many observations. Under `NDEBUG` (Release) the Eigen bounds assertions
  are compiled out, so this was a silent out-of-bounds write into whatever the
  allocator had placed next.

Rewritten as a Householder QR of `Hf`, taking the trailing `2M-3` columns of
`Q` as the nullspace basis, and returning an explicit `effective_rows` so the
caller never reads rows the projection did not fill.

`Givens` (the fast path) had `hx_cols` mis-derived and no guard that the row
count is even; both fixed, with a `CHECK(rows % 2 == 0)`.

**#54 / the shipped `SlowAndFastGivensMatch` failure was two bugs, not one.**
The test compared `Givens` against `SlowGivens`; both were wrong, in different
ways, so the test failure did not localise either. `src/test/unittest_givens.cpp`
is now 6 tests that check each routine against its *specification*
(`basis^T * Hf == 0`, orthonormality, dimension bookkeeping) rather than
against each other.

**Liveness: dead on the benchmark.** `cfg/sweep_dlt_nodesc.json:15` has
`"use_OOS": false`, so no OOS update ever runs. Proven empirically in §7 —
reverting everything except `camera_equidist.h` reproduces M5 byte-for-byte.

## 2. Feature covariance comparators (#32)

`FeatureCovComparison` / `FeatureCovXYComparison` in `estimator.cpp` read the
feature's covariance from `Feature::P_`. For an in-state feature that member is
dead storage — the authoritative block is
`P_.block(kFeatureBegin + 3*sind, ...)`. The comparators therefore sorted
in-state features by a stale subfilter covariance from before the feature
entered the state. They feed gauge-feature selection
(`num_gauge_xy_features: 3`) and the `n_output` accessors.

## 3. `Feature::Merge` and `RefineDepth` (#36, #37, #38)

* `Merge` (loop closure) did not renormalise after combining observation sets.
* `RefineDepth`'s Gauss-Newton used the wrong residual sign on one branch and
  did not restore state on a failed refinement.
* `cfg/sweep_dlt_nodesc.json:166` carries a comment that independently
  corroborates #38 — whoever wrote the config had noticed the same thing and
  worked around it with a threshold rather than fixing it.

## 4. Camera models (#39, #40)

**The live TUM-VI config uses `"model": "equidistant"`** (`:184`), not pinhole.
This is the only M6 change that is live on the benchmark.

`CameraEquidistant::UnProject`, three defects:

1. `rth = xn / (fx_ * cos_phi)`. Analytically this equals
   `sqrt((xn/fx)^2 + (yn/fy)^2)` — the `xn` cancels against
   `cos_phi = fy*xn/sqrt(a^2+b^2)` — but as written it is `0/0` on the entire
   line `xp[0] == cx_`. In floating point with `xn` exactly zero it yields
   `0 / (fx * 6.1e-17) == 0`, i.e. it unprojects every pixel on the image's
   vertical centre line to the optical axis. Replaced with the radius form.
2. The Newton iteration for `th` is unconstrained, but `Project` builds `th` as
   `atan2(|xy|, 1)`, so `th` is confined to `[0, pi/2)`. For a pixel outside the
   model's valid radius the iteration overshoots past `pi/2`, where `tan(th)`
   turns negative and the function returns a ray pointing *backwards* through
   the principal point — a mirrored measurement reported as valid. Clamped.
3. The analytic Jacobian used `x1`, the loop variable holding `d rth / d th` at
   the *second-to-last* iterate, and read it **uninitialised** when
   `max_iter_ == 0`. Recomputed at the final `th`. The principal point (where
   `phi` is undefined and every expression is `0/0`) now returns the correct
   limit `diag(1/fx, 1/fy)`.

`CameraAtan` had the analogous uninitialised-derivative and domain problems.

New tests: `unittest_camera_equi.cpp` +4 (including
`EquiUnprojectionJac`, a central-difference check over a grid, and
`EquiUnprojectOnTheVerticalCentreLine`), `unittest_camera_atan.cpp` +3. All
mutation-verified: reverting each fix fails the corresponding test.

## 5. Accessors (#42, #43)

* `std::max((int)size, n_output)` -> `std::min(...)` in nine accessors. The
  fill loop stops at `min`, so `max` returned arrays with unwritten rows.
  Line 318 of the same file already used `min` for the identical computation.
* `InstateGroupCovs` declared `int cnt;` **uninitialised** and reset it inside
  the outer loop. Two consequences: an out-of-bounds heap write at an arbitrary
  index on the first iteration, and — had the index been valid — columns 6..20
  of the 21-entry packed upper triangle never written while columns 0..5 were
  overwritten by each successive row.

These are Python-facing and not called in `mode eval`, hence inert in §7.

## 6. Filter plumbing (#41, #44–#49, #51–#53)

| # | Fix | Live on TUM-VI? |
|---|---|---|
| #35 | `rodrigues.h` flattening convention + missing `setZero` | no — behind undefined `USE_ONLINE_IMU_CALIB` |
| #41 | `PrinceDormandStep` returned a hardcoded `0` instead of the embedded error estimate | no — `control_stepsize: false`, return value discarded |
| #44 | `DiscardGroup` left `gauge_group_ptr_` dangling | no — `use_1pt_RANSAC: false` |
| #45 | `Qmodel` loader | no — all TUM-VI configs have Qmodel identically zero |
| #46 | `State::td` uninitialised | no — see below |
| #47 | `GoodTimestamp` compared truncated milliseconds | no — proven byte-identical, never fires on TUM-VI |
| #48 | `Track::Reset` did not clear `descriptors_`/`keypoint_` | no — `extract_descriptor: false` |
| #49 | worker thread had no stop flag; also spun at 100% CPU | no — `async_run: false` |
| #51 | group with zero contributed features still added to state | no |
| #52 | `InitializeJustCreatedTracks` branch order | no — `sim_initialize_depths_: false` |
| #53 | `--max_difference` hardcoded, ignoring `-ate_max_difference` | eval script only |

### Corrections to the bug register

* **#35 is latent, not live, and the register's diagnosis was wrong.** The
  register claimed `dA_dAu` "should include mirrored entries" for the lower
  triangle. It should not: `Ca` is upper-triangular by construction
  (`IMU::IMU` CHECKs `Ca(1,0)==Ca(2,0)==Ca(2,1)==0`, and `IMUState::operator+=`
  only touches `j >= i`), so the zero rows are *structurally* correct. The real
  defects are (a) a missing `setZero()`, masked by
  `-DEIGEN_INITIALIZE_MATRICES_BY_ZERO` in the top-level `CMakeLists.txt`, and
  (b) a genuine convention mismatch: `dAB_dA`/`dAB_dB` flattened their **output
  rows** column-major (`p*N+n`) while their input columns, and `dA_dAu`'s rows,
  are row-major. The two are multiplied together at `estimator.cpp:644-647`,
  silently transposing `Rsb*Ca`.

  Corroborated three ways: every other derivative in `rodrigues.h` is row-major
  (`dhat`, `dvee`, `dAt_dA`, `rodrigues`' `dR_dw`, `invrodrigues`' `dw_dR`); the
  hand-built `dWsb_dCg.block<1,3>(i, 3*i) = gyro` at `estimator.cpp:641` is
  row-major; and both *commented-out* reference implementations in the same file
  (lines 202, 264) already use `n*P+p`. `unittest_rodrigues.cpp` reproduces the
  `estimator.cpp` accel-calibration chain against a central difference; reverting
  the two index changes fails exactly 3 of its 6 tests.

  All three calib flags are commented out at `src/CMakeLists.txt:13-15`, so both
  call sites are compile-time dead. Fixed anyway — the next person to enable
  online IMU calibration would have had no chance.

* **#46: the register says `td` is "live (UB)". The UB is real; the
  "affects results" is not.** All four timestamp-shift blocks that read `X_.td`
  (`estimator.cpp:983/1002/1026/1052`) are themselves inside
  `#ifdef USE_ONLINE_TEMPORAL_CALIB`. `td` is nonetheless copied by value at
  `update.cpp:29/369/406` and `manager.cpp:105`, returned by `Estimator::td()`,
  and printed by `~Estimator` — a genuine indeterminate read that happened to
  come out zero only because a freshly mapped heap page is zero. I deliberately
  did *not* move the config read out of the ifdef; loading a value nothing
  consumes would be more misleading than not loading it.

### New defects found beyond the register

* **`Qmodel` read 3 of its 8 keys.** `Tsb`, `Vsb`, `wb`, `ab`, `Tbc` were never
  loaded. `cfg/pcw.json` sets `"Vsb": 0.01` and silently got zero. Also, the
  original squared the block by matrix self-multiplication (`B *= B`) rather
  than squaring the std devs, and injected `Qmodel_` once per call rather than
  scaled by `dt`, making the effective model noise a function of the IMU rate.
* **`PrinceDormand`'s `attempts` is read from config and never used**, and the
  adaptive-step loop never rejects or retries a step — `total_step += h` is
  unconditional, so the computed error estimate only ever resizes the *next*
  step. Documented with a one-shot `LOG(WARNING)` rather than half-implemented:
  real rejection needs an `X_`/`P_` snapshot per trial step, which is a design
  change, not a bug fix.
* The unconditional `std::cout << "err=..."` per integration step is now
  `VLOG(1)`.

---

## 7. The end-to-end result, and why the milestone ATE table is mostly noise

`cfg/sweep_dlt_nodesc`, mono cam0 + IMU, room1–room6:

| variant | mean ATE | RPE_rot | RPE_tra |
|---|---|---|---|
| M0 baseline (`auto`) | 0.1261 | 0.6227 | 0.0364 |
| M2 | 0.1041 | 0.6207 | 0.0282 |
| M3 | 0.1041 | 0.6207 | 0.0282 |
| M4 | 0.0929 | 0.6204 | 0.0252 |
| M5 | 0.1051 | 0.6205 | 0.0270 |
| **M6** | **0.1180** | 0.6205 | 0.0267 |

M6 looks like a +0.013 regression. It is not.

### Attribution

Reverting `common/camera_equidist.h` alone, keeping every other M6 change,
reproduces M5 **byte-for-byte** across all six sequences. So (a) every other
M6 change is confirmed inert on this config, exactly as the liveness table
claims, and (b) the whole delta comes from the undistortion.

`XIVO_RANDOM_SEED` is also inert for this config — seeds 0, 1, 2 give identical
output, because nothing draws from the RNG when `use_1pt_RANSAC` and the
simulation paths are off. The `m0-baseline.md` "seed noise ~= 0.001" figure was
measured on a config that does use RANSAC and does not transfer here.

### Measuring the chaotic band

To find out whether +0.013 means anything, I perturbed `rth` by a *physically
meaningless* relative epsilon and re-ran all six sequences. `rth` is a radius in
normalised units; `1e-11` relative is about `5e-9` pixels, nine orders of
magnitude below the tracker's own precision.

| perturbation | mean ATE |
|---|---|
| none (= M5) | 0.1051 |
| `1e-13` | 0.1051 |
| `(xn/fx)/cos_phi` (pure reassociation) | 0.1051 |
| `-1e-12` | 0.1057 |
| `+1e-12` | 0.1076 |
| `+2e-12` | 0.1106 |
| `+5e-12` | 0.1060 |
| `-5e-12` | 0.1219 |
| `-1e-11` | 0.1168 |
| `+1e-11` | 0.1178 |
| `+2e-11` | 0.1123 |
| `-2e-11` | 0.1178 |
| `+1e-10` | 0.0983 |
| **M6 (real fix)** | **0.1180** |

n = 12 physically-identical pipelines: **mean 0.1104, sd 0.0073, range
0.0983–0.1219.**

M6's 0.1180 is +1.0 sd. It is an ordinary draw from the same distribution.
A sub-nanopixel change to one undistortion expression reproduces the entire
"regression", and one sequence (room5) swings by 0.017 on its own.

### Consequences for the whole project

1. **The honest noise floor for a mean-of-6 ATE with this config is ~+/-0.007
   (1 sd), ~+/-0.015 peak-to-peak — not the 0.005 recorded in
   `m0-baseline.md`.** Any change that perturbs measurements at all re-rolls
   which features survive Mahalanobis gating, and that is a discrete choice
   with macroscopic consequences.
2. **Milestone deltas of <= ~0.015 are not attributable.** That covers M2
   (-0.022, marginal), M3 (0.000), M5 (+0.012) and M6 (+0.013). Only M0 -> M4
   (-0.033, -2.4 sd from the band centre) is clearly outside it, and even that
   should be re-measured as an ensemble.
3. **Single-run comparisons must stop.** From M7 on, every configuration is
   evaluated as an ensemble of >= 8 perturbed runs and reported as mean +/- sd.
   The machine has 192 cores and a 6-room run takes minutes, so this is
   affordable.
4. **`RPE_rot` is the stable metric and it has not moved.** 0.6227 -> 0.6205
   across every milestone, sd across the 12 probes is ~0.0005 — 15x tighter than
   ATE, because it is a differential measure over 1 s windows and does not
   accumulate. The `< 0.5 deg` stretch goal is therefore a *real* target with a
   *real* gap, and it is a rotation-estimation problem, not a tuning problem.
   That is where M7 effort belongs.
5. The stretch goal `mean ATE < 0.06` needs roughly a factor of two, which is
   far outside the noise band. No amount of threshold tuning inside the band
   will get there.

### Not fixed here

The chaotic sensitivity is itself arguably the deepest defect in the codebase:
the filter's output depends discontinuously on the last bits of its input.
The mechanism is hard gating decisions (`MH_thresh`, depth validity, group
lifetime) applied to a state whose covariance is not well calibrated. Fixing it
properly means soft/robust weighting instead of hard accept-reject, which is a
redesign of `OutlierRejection`. Recorded here as the top item for any follow-on
work rather than attempted.
