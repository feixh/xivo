# M2 -- preintegration and the linear initializer

Stage A is the closed-form part of dynamic initialization: given a window of
tracked features and the IMU between them, recover the initial velocity, the
gravity direction, and the feature depths, holding the biases at a prior so the
problem stays linear. Its only job is to land inside the basin Stage B converges
from.

Four modules, all outside `Estimator` and touching no filter state:

| file | what it is |
|---|---|
| `src/init_preint.{h,cpp}` | IMU preintegration with bias Jacobians |
| `src/init_problem.h` | the problem data types, no OpenCV and no dataset |
| `src/init_linear.{h,cpp}` | the `3N+6` solve, Schur complement, sphere-constrained gravity |
| `src/init_window.{h,cpp}` | KLT tracking over the window, problem assembly |

Plus `bin/linear_probe` (Stage A on one real window) and
`harness/linear_check.py` (scores its output against EuRoC ground truth).

## What was verified, and how

`unitTests_init_linear`, 17 tests, all passing; `ctest` 25/25 green.

**Preintegration** (7 tests). The convention is pinned to `ComposeMotion`
(`src/estimator.cpp:1005`) rather than to a paper, because a preintegral that
disagrees with the propagation it seeds is worse than none. Midpoint integration
in both the IMU reading and the rotation, which makes it second order:
`IsSecondOrderAccurateWhileSpinning` measures the error ratio across 200/400/800/
1600 Hz and gets 4.0 +/- 0.35. `ExactWhenTheSpecificForceIsConstant` runs at
omega = 0 where the scheme is exact and holds it to 1e-13, which pins the
algebra including the gravity sign and the alpha-before-beta ordering.
`ReproducesTheFilterPropagation` closes the loop against `ComposeMotion` itself
to <1e-9. Bias Jacobians match central differences.

Measured discretization error at EuRoC's 200 Hz: **4e-5 m/s of velocity,
independent of window span** -- roughly 1500x below the pixel-noise term at the
shipped window length. The preintegrator is not the thing to improve.

**The sphere-constrained solve** (3 tests). `||g||` has to be enforced, not
hoped for. Formulated as the equality-constrained trust-region subproblem and
solved through its secular equation, which is provably the global minimiser and
needs no rank precondition -- unlike OpenVINS' degree-6 polynomial plus 6x6
companion eigensolve, which rank-checks first and can bail. Checked against an
independent brute-force search on 40 random problems (at least 5 asserted
indefinite), plus the degenerate "hard" case where the linear term has no
component along the smallest eigenvector.

**Stage A on synthetic data with exact ground truth** (7 tests). Recovers `v`
and `g` **exactly** (<1e-9, residual <1e-10) from consistent data at every span
tested. So anything below is data, not arithmetic.

## The three measurements that changed the design

### 1. The window has to be ~1.5 s, not the 0.5 s first written down

Pixel noise is the error a longer window actually fixes. At 0.3 px (what survives
the tracker's forward-backward gate), 150 features, 20 frames/s:

| span | v_err | g_cond |
|---|---|---|
| 0.50 s | **1.144 m/s** | 1.20e-06 |
| 0.75 s | 0.441 | 2.94e-06 |
| 1.00 s | 0.199 | 6.05e-06 |
| 1.50 s | **0.060** | 1.82e-05 |
| 2.00 s | 0.036 | 4.19e-05 |

A 0.5 s window -- `InitWindowTracker`'s first-cut default of 11 frames -- costs
more than half the true speed (|v_true| = 2.1 m/s).
1.5 s costs 7% of it: **19x from one knob.** Real data agrees, and adds the other
half of the trade-off -- with the bias held at zero the curve is U-shaped,
because a longer window averages noise down but integrates the unmodelled bias
longer:

| span | MH_01 v_err | MH_02 v_err | tilt |
|---|---|---|---|
| 0.5 s | 0.418 | 0.257 | 4.1 deg |
| 1.0 s | 0.356 | **0.029** | 4.8 |
| 1.5 s | **0.144** | 0.109 | 4.2 |
| 2.0 s | 0.276 | 0.201 | 5.3 |
| 3.0 s | 0.564 | 0.592 | 7.0 |

The plateau is 1.0-1.5 s and both ends are clearly bad. **1.5 s** for now; these
are single windows, so M5 tunes it properly.

### 2. The depth-scaled cost is bimodal, and that is not a solver bug

The consequential finding. With an accelerometer bias held at zero, the objective
acquires a second minimum about **40 degrees away in gravity direction whose cost
is lower than the one near the truth by one part in 1e4**. The sphere solve
correctly returns it, and the velocity is then wrong by >10 m/s at every span:

| |ba| | omega | v_err | v_err/|ba| |
|---|---|---|---|
| 0.01 | 0.76 | 0.012 | 1.2 |
| 0.03 | 0.76 | 0.037 | 1.2 |
| 0.06 | 0.76 | 0.076 | 1.3 |
| 0.159 | 0.76 | **13.718** | 86.3 |
| 0.01 | 0.10 | **12.923** | 1292 |

An error that large and that insensitive to the size of the perturbation is not
amplification. Confirmed by evaluating the objective at both points: at
`|ba|=0.010, omega=0.10` the solver's answer scores **-2531.78748** against the
truth's **-2531.78534**, and an independent 20k-direction sphere scan lands in the
same basin. The solver is right; the cost prefers the wrong answer.

No conditioning fix applies, because nothing in this cost separates the branches.
The accelerometer does: averaged over the window and rotated into `I0` it reads
`mean(a_world) - g`, so its direction is good to a few degrees whenever the mean
specific force is small next to 9.81. Hence
`LinearInitOptions::PriorMode::Check` -- the prior is a **discriminator, not an
estimate**, which matters because the two roles have opposite rankings:

| case | constrained solve | forced to prior | Check |
|---|---|---|---|
| clean | 0.000 | 1.475 | **0.000** |
| 0.3 px noise | 0.060 | 1.477 | **0.060** |
| accel bias | 13.731 | 1.422 | **1.422** |

Check fired on exactly the 4 flipped rows and none of the other 20. On real EuRoC
it never fires (0 flips / 11 sequences), which is precisely why the guard needs a
unit test rather than a dataset run.

Two things were tried and rejected. **Dropping the `||g||` constraint**: the
hypothesis was that an accel bias is a gauge direction of gravity (it perturbs
`alpha_k` by `0.5 R ba dt^2`, which the row equation absorbs with `dg = -R ba`).
Measured false -- the free solve gives 7.78 m/s under pure `ba`, not ~0, because
`alpha_k`'s bias term is a double integral of `R(t) ba` and the rig sweeps 65
degrees across the window, so no single constant offset absorbs it. Under pixel
noise the free solve is **35x worse** (2.11 vs 0.060). The constraint stays.
**Depth-reweighting the rows**: the rows are depth-scaled, so a 2-9 m scene is
heteroscedastic by 20x in variance; estimated payoff ~1.4x against ~19x from
window length, so it is scope creep. M3 can revisit if Stage A is ever the
bottleneck.

Also worth recording: **`g_cond` is not a health check.** A planted gyro bias
*raises* it by five orders of magnitude (1.2e-6 -> 0.165), because a wrong
rotation chain injects signal outside the span of the `{dt, dt^2/2}` families
whose near-collinearity is what makes a clean short window ill-conditioned.
Gating on it would admit biased windows first.

### 3. Real data: the bias is the whole error, and it is mostly the *gyro*

All 11 EuRoC sequences, 1.5 s window starting just after each sequence's ground
truth begins:

| bias prior | v_err mean | v_err max | tilt mean | tilt max | flips |
|---|---|---|---|---|---|
| zero | 0.173 m/s | 0.226 | 4.15 deg | 5.68 | 0 |
| GT's solved bias | **0.027** | 0.091 | **0.42** | 0.84 | 0 |

Per-sequence, the two dynamic starts:

| seq | \|v\|gt | v_err (b=0) | tilt (b=0) | v_err (b=GT) | tilt (b=GT) |
|---|---|---|---|---|---|
| MH_01 | 0.802 | 0.144 | 4.17 deg | 0.091 | 0.62 deg |
| MH_02 | 0.473 | 0.109 | 4.14 | 0.020 | 0.26 |

The residual falls 5x alongside (4.3e-2 -> 1.5e-2), so the bias correction
improves the *fit*, not just the comparison against truth -- which rules out a
non-gravity-aligned ground-truth world frame as the explanation for the 4 degrees.

The mechanism is the gyro bias, not the accel bias, and the magnitudes say so:
`|ba| = 0.158 m/s^2` is only `atan(0.158/9.81) = 0.92 deg` of tilt, but
`bg_z = 0.0785 rad/s` over 1.5 s corrupts the rotation chain by 6.7 degrees,
which smears the accelerometer readings as they are rotated into `I0`. The
synthetic sweep agrees: gyro-only costs 1.83 m/s where gyro-and-accel costs 1.80.
This is good news for M3 -- the gyro bias is the one component a static prefix
can measure directly, and the one Stage B estimates most reliably.

**Two conclusions for M4.** First, 4.15 degrees of tilt is *worse* than the static
initializer achieves on a static rig, so Stage A must never ship alone -- Stage B
is mandatory, not a refinement. Second, on the nine static sequences Stage A
reports 0.15-0.23 m/s of velocity where the truth is ~0.01, so the M1 detector
routing them to the static path is load-bearing rather than cosmetic.

## A premise of the task, confirmed from the data

`|v|` at the first ground-truth sample:

| MH_01 | MH_02 | the other nine |
|---|---|---|
| 0.801 | 0.437 | 0.006 - 0.034 |

MH_01 and MH_02 are the only EuRoC sequences that start in motion. That is
exactly the partition M4's bit-identical prediction rests on.

## Corrections to the plan

* M2's "**predicted ~0.1-0.2 m/s**" for Stage A on real data is **confirmed**:
  measured 0.11-0.23, mean 0.17. An intermediate synthetic figure of 1.8-2.1 m/s
  was the outlier, not the plan -- that fixture holds `|a_w| = 1.06 m/s^2` and
  `|omega| = 0.76 rad/s` constant across the whole window, whereas real
  hand-carried motion averages both down.
* `Options::max_frames` changes from **11 to 31** (0.5 s to 1.5 s at 20 Hz). The
  plan left the window length open; this is the number M5 will tune around.
* Stage A gains one option the plan did not anticipate,
  `PriorMode::Check`, for the bimodality above.

## Reproducing

```sh
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DOpenCV_DIR=<opencv_install>/lib/cmake/opencv4 && make -j64
cd .. && ./bin/unitTests_init_linear          # 17 tests
cd build && ctest --output-on-failure          # 25/25

# real data, from the worktree root
export PATH="<venv>/bin:$PATH"
python3 notes-n-prompts/notes-dyninit/harness/linear_check.py \
    --root ../data/euroc --auto-start                 # zero bias prior
python3 notes-n-prompts/notes-dyninit/harness/linear_check.py \
    --root ../data/euroc --auto-start --gt-bias       # bias error isolated
python3 notes-n-prompts/notes-dyninit/harness/linear_check.py \
    --root ../data/euroc --seqs MH_01_easy MH_02_easy --auto-start --span-sweep

# one window, verbose
./bin/linear_probe -cfg cfg/euroc_stereo.json -root ../data/euroc \
    -seq MH_01_easy -start 1.1 -frames 31 -header
```
