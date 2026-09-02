# Dynamic initialization for XIVO -- plan and milestones

Goal: XIVO must initialize correctly when the sensor platform is **already
moving** at the first sample. That means (a) *detecting* whether the platform is
static or dynamic, (b) keeping today's static path exactly as it is when it is
static, and (c) when it is not, solving a small **bundle adjustment over a short
window** for the initial velocity, the gyro and accelerometer biases, the gravity
direction, and the window's feature geometry, then handing that to the filter.

Starting point: `auto` @ `8d2d052`. Development on branch `auto-dyninit`, worktree
`xivo-dyninit` (created; builds clean, `ctest` 23/23). Reference for the
algorithm: OpenVINS v2.7 `ov_init` (read, not linked -- see "Dependencies").

## 1. What is actually broken

`Estimator::InitializeGravity()` (`src/estimator.cpp:740`) is the whole
initializer. It averages `gravity_init_counter` (20) accelerometer samples,
optionally de-rotating them, and sets the 2-DoF gravity state from
`FromTwoVectors(-g_, accel_calib)`. **`X_.Vsb`, `X_.bg` and `X_.ba` are never
estimated -- they stay at their configured values, which is zero.** That is
correct if and only if the rig is at rest.

The one guard against motion is M4's `gravity_init_max_accel_dev` gate
(`estimator.cpp:840`): reject a sample when
`| ||a|| - ||g|| | > dev` (0.1 shipped), give up after
`gravity_init_max_skip` (2000) rejections. It can only *wait for a quiet
stretch*; it cannot initialize during motion. And its statistic is blind in
exactly the case that matters:

> **An accelerometer cannot distinguish constant-velocity motion from rest.**
> Specific force is `R'(a - g)`; at constant velocity `a = 0` and the
> magnitude reads exactly `||g||`, so `| ||a|| - ||g|| |` is zero. Accelerometer
> statistics detect *acceleration*, never *velocity*.

Simulating the shipped gate sample-by-sample on all eleven EuRoC sequences, and
reading ground-truth velocity **at the instant the gate actually fires**:

| sequence | fires at | samples skipped | true \|v\| there |
|---|---|---|---|
| MH_01_easy | 2.100 s | 401 | **0.671 m/s** |
| MH_02_easy | 1.020 s | 185 | **0.481 m/s** |
| the other nine | 0.095-0.505 s | 0-81 | <= 0.034 m/s |

So two of eleven EuRoC sequences initialize with a velocity error of 0.5-0.7 m/s
asserted as zero, and nine are genuinely static. (The measurement window must be
the **IMU** window: EuRoC ground truth starts 0.9-2.4 s after IMU t0, so "GT at
t=0" describes a time XIVO has already initialized past.)

This is not academic. MH_01 is the only Machine Hall sequence XIVO loses to
OpenVINS on (`ate_002` 0.080 `fast` / 0.087 `acc` vs OpenVINS 0.073), and it is
the sequence where mono `acc` diverges 10 runs out of 10.

## 2. Why the detector needs vision, and why OpenVINS' detector is also wrong

OpenVINS decides static-vs-dynamic from **accelerometer sample standard
deviation** over two half-windows against `init_imu_thresh`
(`ov_init/src/static/StaticInitializer.cpp`), plus a **raw pixel disparity**
threshold of 10 px (`InertialInitializer.cpp`). Both cues are individually
unsound, and measurement says so:

Accel sample sd (m/s^2) over 0.5 s windows in the first 5 s of each EuRoC
sequence, against the truth from the table above:

| sequence | sd @0.5s | sd @1.0s | sd @2.0s | **min over windows** | truth |
|---|---|---|---|---|---|
| MH_01_easy | 0.897 | 1.115 | 1.358 | **0.668** | moving 0.67 m/s |
| MH_02_easy | 0.499 | 0.527 | 1.753 | **0.499** | moving 0.48 m/s |
| MH_03_medium | 0.481 | 0.417 | 0.269 | 0.190 | static |
| MH_04_difficult | 0.219 | 0.162 | 0.450 | 0.136 | static |
| MH_05_difficult | 0.064 | 0.058 | 0.067 | 0.057 | static |
| V1_01_easy | **0.831** | **1.366** | 0.284 | 0.192 | static |
| V1_02_medium | 0.050 | 0.427 | 0.711 | 0.047 | static |
| V1_03_difficult | 0.042 | 0.040 | 0.046 | 0.039 | static |
| V2_01_easy | 0.462 | 0.290 | 0.662 | 0.222 | static |
| V2_02_medium | 0.346 | 0.225 | 0.559 | 0.221 | static |
| V2_03_difficult | **1.093** | **1.087** | 0.268 | 0.260 | static |

Two readings, both of which shape the design:

* **A variance test at a fixed instant misclassifies.** V1_01 reads 0.831 and
  V2_03 reads 1.093 while genuinely static -- somebody is picking the rig up or
  bumping the table. Any single-window threshold that passes MH_02 (0.499) as
  moving also flags those two.
* **The minimum over candidate windows does separate**, 0.499 against 0.260 --
  under 2x margin, which is thin but real. So the IMU test must *search for a
  quiet window* and conclude "dynamic" only when no window is quiet enough.

Even that is a proxy. Since the IMU is blind to constant velocity in principle,
the detector needs a second, independent cue, and it must be **vision** --
vision is the only sensor here that sees translation at constant velocity.

Raw pixel disparity (OpenVINS' choice) conflates two things it cannot separate:
disparity grows with translation/depth, so a fixed 10 px threshold calls a slow
near-field motion static and a stationary-but-rotating rig dynamic. XIVO
already knows the rig rotates 0.11-0.32 rad/s at init on TUM-VI. The fix is to
**remove the rotational component using the gyro** before measuring flow: a
stationary camera has *zero* residual flow after de-rotation, at any depth, at
any rotation rate. Residual flow is then a direct translation signal.

Design: **static iff (min-window accel sd < `dyninit_imu_thresh`) AND
(gyro-de-rotated residual flow < `dyninit_flow_thresh`)**, requiring both cues to
agree, with dynamic as the fallback. This is strictly stronger than either
system's current test, and M1 has to demonstrate that on data.

## 3. The algorithm

Two stages, following OpenVINS' structure because the structure is right: a
closed-form linear solve to get into the basin, then a nonlinear joint
optimization that adds the biases.

### Stage A -- linear, closed form, with `||g||` enforced

Unknowns `x = [p_F1^{I0} ... p_FN^{I0}, v_{I0}^{I0}, g^{I0}]`, size `3N + 6`.
Body position at window frame `k` follows from preintegration between frame 0 and
frame `k` at the *prior* bias:

    p_{Ik}^{I0} = v_{I0} * dt_k + 0.5 * g^{I0} * dt_k^2 + alpha_k

Each observation of feature `f` in frame `k`, camera `c`, at normalized
coordinates `(u,v)`, contributes 2 rows. With `H_proj = [[1,0,-u],[0,1,-v]]`
(which enforces `p_F^{Ck} || [u,v,1]`) and `Y = H_proj * R_{b->c}' * R_{I0->Ik}`:

    Y * p_F^{I0}  -  dt_k * Y * v  -  0.5 dt_k^2 * Y * g  =  Y * alpha_k - H_proj * p_{b}^{c}

Biases are **not** recovered here -- they are held at the prior. Stage A only has
to be good enough to seed Stage B.

`||g|| = 9.81` must be enforced or the solve absorbs bias and scale error into
gravity's magnitude. Eliminate the features and velocity by Schur complement to
get a 3-variable problem `min g' D g + 2 d' g s.t. ||g|| = r`, then solve it as a
**secular equation**: eigendecompose `D = V L V'`, set `e = -V' d`, and root-find

    sum_i e_i^2 / (l_i - lambda)^2  =  r^2

for `lambda < l_min`, where the left side is monotone increasing -- so a
safeguarded Newton iteration converges to the unique global minimizer.
`g = V (L - lambda I)^{-1} e`. This is the standard trust-region-subproblem
solve; it is ~25 lines, has no rank or conditioning preconditions, and it
replaces OpenVINS' route (a symbolically-expanded degree-6 polynomial fed to a
6x6 companion-matrix eigensolve, with an explicit rank check that can and does
bail out). Deriving it here rather than transcribing their coefficients is both
cleaner and keeps the "nothing from OpenVINS is linked or copied" property.

### Stage B -- the bundle adjustment

Joint nonlinear least squares over the whole window, in a **gravity-aligned world
frame `W`** whose `z` is `-g_` (so gravity direction is refined implicitly, via
the free roll/pitch of frame 0, rather than carried as a separate variable):

* per window frame `k = 0..K-1`: `R_{W<-Ik}` (3, local `SO3::exp` increments),
  `p_{Ik}^W` (3), `v_{Ik}^W` (3)
* global: `bg` (3), `ba` (3)
* per feature: `p_F^W` (3)

Residuals:

* **IMU preintegration**, consecutive frames, 9-dim (`alpha`, `beta`, rotation),
  with the `d/dbg`, `d/dba` Jacobians carried through the preintegration -- that
  is what makes the biases observable rather than merely present.
* **Reprojection**, 2 per observation, weighted by `1/sigma_pix`, Cauchy-robustified.
* **Gauge**: `p_0 = 0` held fixed (3 DoF) and a tight prior on the yaw of
  `R_{W<-I0}` (1 DoF). Scale is observable from the IMU, so nothing else is fixed.

Solver: **Levenberg-Marquardt with Schur complement** on the feature block
(features are the large, block-diagonal part), Cholesky on the reduced camera
system. Hand-rolled in Eigen -- see "Dependencies".

### Handing the result to the filter

XIVO's state at init: `X_.Rsb = I` by construction (the spatial frame `S` *is*
the body frame at init), `X_.Tsb = 0`, plus `X_.Vsb`, `X_.bg`, `X_.ba`, and the
2-DoF `X_.Rsg` which maps the gravity-aligned frame into `S`.

Take the **last** window frame as the init instant. Then

    Vsb = R_{W<-Ilast}' * v_{Ilast}^W
    bg  = bg,   ba = ba
    Rsg  from FromTwoVectors(-g_, R_{W<-Ilast}' * (-g_W)), with Wsg(2) = 0

Every input on the right is **gauge-invariant** -- each is a world quantity
rotated into body coordinates -- so the arbitrary yaw the BA settled on cannot
leak into the filter. Reusing the existing `FromTwoVectors`/`Wsg(2)=0` code for
`Rsg` also guarantees the 2-DoF constraint is respected exactly, and the whole
handoff degenerates to today's static path when `v = 0` and the biases are zero.

Features tracked during the window are **discarded** by default and the filter
starts clean at the last window frame; seeding them is a separate, measured
milestone (M6), not an assumption.

## 4. Dependencies

The BA is **hand-rolled in Eigen**: `find_package(Ceres)` is deliberately *not*
added. XIVO's dependency set (OpenCV, Eigen, Pangolin, glog, gflags, jsoncpp)
does not change, so nothing downstream has to be rebuilt against a new library,
and the repo's standing property that nothing from OpenVINS' stack is linked or
copied survives. The cost is implementation risk on the Jacobians and the LM
loop; M2 and M3 pay that down with numerical-Jacobian checks and synthetic
problems that have exact answers, and M4 cross-checks the end-to-end result
against OpenVINS' dynamic initializer on the same two sequences.

The init-window feature tracker is **self-contained** (`goodFeaturesToTrack` +
`calcOpticalFlowPyrLK` inside the initializer), not XIVO's `Tracker`. Reason: the
tracker is entangled with the feature memory pool, the `MemoryManager` lifecycle
and `Predict`, and running it before the filter exists -- then throwing its
features away -- risks perturbing pool state and feature IDs. Keeping the init
window out of that machinery is what makes "the static path is bit-identical" a
structural guarantee instead of a hope. Cost: ~80 lines of duplicated tracking,
and the init window does not share work with the tracker.

## 5. Milestones

Each milestone is a commit on `auto-dyninit`, and each commit is preceded by an
experiment that could have failed. `ctest` stays green throughout.

### M0 -- quantify the problem, reproducibly
* A script that, for any sequence, replays the shipped gate sample by sample,
  reports where it fires, and reports ground-truth velocity **at that instant**.
* Run it on all 11 EuRoC + 6 TUM-VI room sequences; the tables in sections 1-2
  become checked-in output rather than transcript.
* **Confirms before commit:** the two-sequence diagnosis is reproducible from a
  clean checkout, and TUM-VI room1-6 are all static (so the TUM-VI results are
  not at risk).
* **Commit:** the diagnosis and the script.

### M1 -- the static/dynamic detector, observation only
* `src/initializer.{h,cpp}`: min-over-windows accel sd, and gyro-de-rotated
  residual optical flow.
* Wired to *log* its classification and nothing else. Config key
  `dynamic_init` absent => entire feature off => not one instruction changes.
* Unit test on synthetic IMU + camera: a stationary-but-rotating rig classifies
  static (this is the case raw disparity gets wrong); a constant-velocity rig
  classifies dynamic (this is the case accel magnitude *and* accel variance both
  get wrong).
* **Confirms before commit:** 17 of 17 sequences classified correctly
  (2 dynamic, 15 static), and an n=3 EuRoC stereo pass is **bit-identical** to
  `euroc_m6_final/fast` members 0-2 in all 165 metric cells.
* **Commit:** the detector.

### M2 -- preintegration and the linear initializer
* IMU preintegration between window frames with bias Jacobians.
* Stage A: build the `3N+6` system, Schur-eliminate, secular-equation solve for
  `g`, back-substitute.
* `unittest_init_linear`: synthetic scene, exact ground truth, no noise --
  recovers `v` and `g` to ~1e-9; `||g||` exact by construction; verify the
  secular solve finds the *global* minimizer by brute-force comparison on a
  sphere grid.
* **Confirms before commit:** Stage A alone recovers MH_01's and MH_02's true
  initial velocity from real data to within a tolerance set by the bias error it
  cannot see (predicted ~0.1-0.2 m/s -- Stage B's job is to remove that).
* **Commit:** the linear initializer.

### M3 -- the bundle adjustment
* Analytic Jacobians for both residual families; LM with Schur complement.
* `unittest_init_ba`: (a) every analytic Jacobian matches central differences to
  1e-6; (b) noiseless synthetic problem seeded away from the answer converges to
  `v` within 1e-6 m/s and `bg`/`ba` within 1e-5; (c) with realistic noise and a
  planted bias of EuRoC's magnitude (0.08 rad/s gyro), the recovered bias is
  within a stated tolerance of the planted one; (d) cost decreases monotonically
  and the solver terminates.
* **Confirms before commit:** on MH_01/MH_02 real data, Stage B improves on
  Stage A's velocity, and the recovered gyro bias agrees with EuRoC's own
  solved-for bias (`state_groundtruth_estimate0` columns 11-13) -- an external
  check, not a self-consistency check.
* **Commit:** the BA.

### M4 -- wire it into the estimator
* Init-window buffering (IMU triples + incremental tracking), dispatch on the M1
  detector, BA on the dynamic branch, the gauge-invariant handoff of section 3,
  and a fallback to the static path if the BA fails or is ill-conditioned.
* **The falsifiable prediction this milestone lives or dies by:** with
  `dynamic_init` on, **the nine static EuRoC sequences must be bit-identical to
  the current shipped result**, because the detector selects the static path and
  the static path is untouched; only MH_01 and MH_02 may change. This is the same
  evidence structure that validated OpenVINS' own `--init_dyn_use 1`, where all
  six Vicon Room sequences came out byte-identical.
* Plus: TUM-VI room1-6 bit-identical (all static), and `ctest` green.
* **Commit:** dynamic initialization, live.

### M5 -- evaluate and tune
* n=10 jitter ensembles, all 11 EuRoC, stereo and mono, `dynamic_init` on vs off.
  Metrics: `ate_002` and `ov_eval error_singlerun posyaw` (ATE pos/ori, RPE 8 m
  pos/ori), per the established protocol. Screen at n=3, ship at n=10.
* Tune only: window length, the two detector thresholds, `sigma_pix`, the LM
  budget. One configuration for all sequences -- no per-sequence overrides.
* One-core throughput and peak RSS: init runs once, so the cost must land as a
  one-off latency, not a per-frame regression. Report both.
* Divergence census, and the mono MH_01 case specifically.
* **Commit:** the shipped configuration and the evaluation.

### M6 -- covariance seeding (conditional)
* The BA's marginal covariance over `(v, bg, ba)` is available; today the filter
  starts those blocks from a config prior that has no idea whether init was easy
  or hard.
* Measured, n=10. **Ships only if it helps**; a negative result gets written up
  and reverted, like the seventh branch in the TUM-VI round.

### M7 -- report, README, merge
* `notes-n-prompts/report-dynamic-init.md`: the diagnosis, the algorithm, every
  measurement, the negative results, the protocol, repro commands.
* `README.md`: a concise section on what was added and the numbers behind it.
* Merge `auto-dyninit` into `auto`, then **re-verify on the merged tree** rather
  than assume -- branch numbers do not compose, which was the lesson from both
  previous rounds. Build, `ctest`, and an n=3 pass matched cell by cell.

## 6. Protocol carried over (settled, not up for re-litigation)

* Never compare single runs; jitter ensembles (`X.Vsb += k*1e-6`), report the sd
  of the sequence-mean across members. Screen at n=3, **ship at n=10**.
* Score with both `evaluate_ate.py --max_difference 0.02` (blind to a global
  rotation) and `ov_eval ... posyaw` (charges roll/pitch in full).
* Throughput: one pinned core, ASLR off, all pools at 1, `-mode runOnly`,
  `sweep_fps.sh` + `report_onecore.py`. Never quote FPS or RSS from an accuracy
  pass.
* `agg_ensemble.py --mode {mono,stereo} --arm NAME DIR`, globs **unquoted**.
* Harness files under `experiments/` are not version controlled; mirror anything
  worth keeping into `notes-n-prompts/notes-dyninit/harness/`.
