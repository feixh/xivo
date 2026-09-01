# Things that did not work

All screened with `notes-n-prompts/notes-orient/screen.sh <tag>`: the standard
harness, mono, room1-room6, `--jitter 6` (36 runs). References:

| tag | base | ori [deg] | rpe_ori [deg] | ate002 [m] |
|---|---|---|---|---|
| **m1** | — | **1.013** | **0.5185** | **0.0928** |
| **s_gauge** (= M2) | m1 | **0.949** | **0.5149** | **0.0957** |
| s_qbias | m1 | 1.095 | 0.5360 | 0.0963 |
| s_grav | m1 | 0.958 | — | 0.1047 |
| s_pwsg | m1 | 0.955 | 0.5178 | 0.0967 |
| s_ginit | m1 | 1.009 | — | 0.0986 |
| s_pbias | m2 | 1.091 | 0.5412 | 0.1037 |
| s_pba | m2 | 1.041 | 0.5342 | 0.1002 |
| s_pbg | m2 | 1.010 | 0.5122 | 0.0965 |
| s_g975 | m2 | 1.088 | 0.5118 | 0.1048 |
| s_qgyro | m2 | 1.058 | 0.5140 | 0.1014 |
| s_cal | m2 | 1.079 | 0.5047 | 0.1046 |
| s_cal2 | m2 | 1.065 | 0.5136 | 0.0997 |

What each tag is: `s_qbias` = kalibr bias random walks (lead 2); `s_grav` =
`gravity` 9.80766 (lead 3); `s_pwsg` = `P.Wsg` 0.1 (lead 4); `s_ginit` = 200-sample
derotated gravity init (lead 5); `s_pbias`/`s_pba`/`s_pbg` = `P.ba` 0.05 and/or
`P.bg` 0.002; `s_g975` = `gravity` 9.75, the data-consistent value; `s_qgyro` =
`Qimu.gyro` 1.6e-4, the kalibr white-noise density; `s_cal` = measured `X.ba` with
`gravity` 9.75; `s_cal2` = measured `X.ba` with `gravity` 9.8.

**Nine of the eleven break the `ate_002` budget (baseline + 0.005 = 0.0978) or the
`rpe_ori` limit (0.53).** The only two that improve orientation ATE against their own
base, `s_grav` and `s_pwsg`, do so by 1.3 sigma each -- indistinguishable from noise
-- while both move `ate_002` the wrong way. The shipped config is a tightly tuned
local optimum: see the gravity result below, where moving in *either* direction from
9.8 costs the same +0.012 m of position ATE, which is the signature of a value that
was fitted rather than derived.

Ensemble sd of a 6-room mean is 0.061 deg for `ori`, so its standard error is
0.061/sqrt(6) = 0.025 deg and a difference of two means needs about 0.10 deg
before it is worth believing. `ate_002` has sd 0.0067 m, sem 0.0027 m.
Non-regression limits: `rpe_ori` <= 0.53, `ate002` <= baseline + 0.005.

## The one measurement that explains most of this page

Before screening the IMU-side leads I measured the *actual* biases of the TUM-VI
room sequences from the mocap groundtruth, and compared them with what XIVO's
filter is able to represent.

`P` entries are standard deviations (`Estimator::Estimator` does `P_ *= P_`) and
`Qimu` entries are continuous-time densities added straight into `Pdot`, so the
marginal bias variance the filter carries after `t` seconds of a run is
`P0^2 + q^2 t`:

| | `P.bg` = 1e-4, `q` = 6.6e-6 | `P.ba` = 1e-3, `q` = 2.58e-4 |
|---|---|---|
| std at t=0 | 1.0e-4 rad/s = 0.0057 deg/s | 0.0010 m/s^2 |
| std at t=100 s | 1.2e-4 rad/s = 0.0069 deg/s | 0.0028 m/s^2 |
| std at t=200 s | 1.4e-4 rad/s = 0.0078 deg/s | 0.0038 m/s^2 |

The bias states are, for practical purposes, **pinned at zero for the whole run**.

Measured truth (windowed least squares against the mocap; for `ba` the model is
`dv = int R (a_meas - ba) dt + g_w T` solved for `ba` *and* `g_w` over 1 s windows,
for `bg` it is the SO(3) version, `log((R_gt(s) P)' R_gt(e)) = M bg` with
`M = -sum_k P_k' dt_k`, over 2 s windows):

| seq | `bg` [rad/s] | \|bg\| [deg/s] | `ba` [m/s^2] | \|ba\| | tilt that the horizontal part of `ba` implies |
|---|---|---|---|---|---|
| room1 | ( 0.00004, -0.00031,  0.00052) | 0.035 | (-0.014, 0.020, 0.020) | 0.031 | 0.14 deg |
| room2 | (-0.00025, -0.00048, -0.00038) | 0.038 | (-0.020, 0.018, 0.053) | 0.059 | 0.16 deg |
| room3 | ( 0.00037,  **0.00273**,  0.00016) | **0.158** | (-0.025, 0.038, 0.045) | 0.064 | 0.26 deg |
| room4 | (-0.00003,  0.00039,  0.00040) | 0.032 | (-0.028, 0.028, 0.027) | 0.047 | 0.23 deg |
| room5 | (-0.00045,  0.00023, -0.00007) | 0.029 | (-0.020, 0.021, 0.005) | 0.030 | 0.17 deg |
| room6 | (-0.00081,  0.00034, -0.00009) | 0.050 | (-0.043, 0.025, 0.036) | 0.061 | 0.29 deg |

So the true accel bias is 8-15x the filter's *entire* uncertainty budget and the
true gyro bias is 2.5-22x it. Two things follow, and they are the two halves of
`oridecomp.py`'s split of the orientation ATE:

* The **tilt floor** -- or so I thought, and this half is **wrong**; see the
  subsection two below. The argument was: a constant *body-frame* accel bias is
  indistinguishable from a tilt of `|ba_horiz|/g`, and because it is fixed in the
  body the world-frame tilt error it produces rotates with the rig, which is exactly
  what is observed (`oridecomp.py` says the tilt part is 0.42 deg RMS of which only
  0.088 deg is a constant offset). The predicted 0.14-0.29 deg is the right size and
  roughly the right per-sequence ordering. It is still a false explanation.
* **room3**, the worst sequence in both modes (mono 1.42, stereo 2.02), has a gyro
  bias 22x the filter's budget and 4-5x every other sequence's. Its yaw error is
  1.33 deg mono / 1.95 deg stereo against 0.43-1.18 elsewhere. It is not a visual
  problem; it is an IMU-calibration problem.

**And loosening the priors so the filter can actually estimate those biases makes
the result worse, every time.** `P.ba` = 0.05 -> ori 1.041, `rpe_ori` 0.5342
(breaks the limit), `ate002` 0.1002 (breaks the budget). `P.bg` = 0.002 -> ori
1.010, `ate002` 0.0965. Both together -> ori 1.091, `rpe_ori` 0.5412, `ate002`
0.1037. Plus lead 2's random walks, which fail identically. **Four independent
knobs, four failures, and every one of them fails on `rpe_ori`** -- the local
attitude metric, which is the giveaway that the bias state is chasing per-frame
noise rather than settling on the DC value.

The bias states are not the binding constraint; the filter's *consistency* is.
Without FEJ the covariance is over-confident, so any extra state freedom gets spent
absorbing visual residuals and linearization error rather than the DC bias.

### The half of this that turned out to be wrong

The bias table above explains the tilt floor beautifully and **the explanation is
false**. Setting `X.ba` to the measured bias directly -- which sidesteps the prior
entirely, since `P.ba` then keeps the filter *at* the right value instead of at zero
-- does not move the tilt error at all:

| | tilt | yaw | total |
|---|---|---|---|
| M2 (`X.ba` = 0, `gravity` 9.8) | 0.420 | 0.832 | 0.949 |
| `s_cal` (measured `X.ba`, `gravity` 9.75) | 0.416 | 0.974 | 1.079 |
| `s_cal2` (measured `X.ba`, `gravity` 9.8) | 0.418 | 0.968 | 1.065 |

0.420 -> 0.416 is nothing, and yaw and position both get clearly worse. (One
consolation: `s_cal` has the best `rpe_ori` of anything screened, 0.5047, so the
correct bias *does* improve the local attitude increment -- it just cannot touch the
floor.)

The floor is not the accelerometer bias, and it is not anything on the estimator's
side of the interface: OpenVINS reproduces it sequence by sequence. See
`tilt-floor-is-the-benchmark.md`. This is the single most useful thing on this page,
and I only found it by testing a theory I was confident in.

## The prompt's leads, one at a time

### Lead 1 -- no FEJ. Not attempted, deliberately.

XIVO recomputes every Jacobian at the current estimate. First-estimate Jacobians
are the standard consistency fix and, per the paragraph above, the *reason* the
three bias knobs fail: they would be the enabling change, not an alternative to it.

Implementing FEJ means freezing each group's pose at the value it had when the
group entered the state and evaluating the *feature* measurement Jacobian at the
frozen anchor. `Feature::ComputeJacobian` reads `ref_->Rsb()` / `ref_->Tsb()`
directly, so this cannot be done without editing `src/feature.cpp`, which the task
fences off as the position agent's file. With the 1.42 deg target already met by
M1 + M2 and with three sibling agents merging into one tree by hand, spending a
guaranteed conflict in someone else's file on an unmeasured change was the wrong
trade. It is the clear next step for whoever owns the whole tree, and the payoff
is probably larger than everything on this page: it is the precondition for the
bias states being usable at all.

### Lead 2 -- IMU bias random walk is 3.33x too small vs kalibr. REJECTED.

The premise checks out: TUM-VI's kalibr report gives gyro bias random walk
2.2e-5 rad/s^1.5 and accel bias random walk 8.6e-4 m/s^2.5, and the config carries
6.6e-6 / 2.58e-4 -- a factor of 3.33 in each, suspiciously exactly.

Raising both to the kalibr values: ori 1.013 -> 1.095, and `rpe_ori`
0.5185 -> 0.5360, which breaks the 0.53 deg limit on its own. `ate002` 0.0963.
See the consistency argument above.

### Lead 3 -- gravity is 9.8, not 9.80766. REJECTED, and 9.8 is the better number.

Setting 9.80766: ori 1.013 -> 0.958 (1.3 sigma, not significant) but `ate002`
0.0928 -> 0.1047, i.e. +0.012 m, more than twice the constraint-2 budget.

The `ba`/`g_w` least squares above says why, and it is the nicest result on this
page: solving for the gravity vector *in the groundtruth frame* from the
accelerometer gives **9.727 to 9.773, mean 9.75**, tilted 0.04-0.18 deg from the
mocap vertical. The TUM-VI accelerometer's effective scale is about 0.5% low, so
the specific force it reports at rest is ~9.75 m/s^2, not 9.807. XIVO's 9.8 is
*closer to what the sensor actually reports* than the textbook value for Munich
is. Lead 3 asks to move away from the data, and the position metric duly gets
worse.

Screening `gravity = -9.75`, the data-consistent value, as `s_g975`: ori 1.088,
`ate002` 0.1048. So **moving gravity in either direction from 9.8 costs the same
+0.012 m of position ATE** (9.80766 -> 0.1047, 9.75 -> 0.1048). 9.8 is not a
physical constant here, it is a fitted parameter sitting at the bottom of its own
valley, and it is doing double duty: the accelerometer's vertical bias is +0.02 to
+0.05 m/s^2 (table above) and its effective gravity is 9.75, and 9.75 + 0.05 = 9.8.
The config's "wrong" gravity is compensating the accel bias the filter is not
allowed to estimate. That also predicts the `s_cal` result: correct the bias *and*
the gravity and you have removed both halves of a working cancellation, which is
why `s_cal` is worse than either.

### Lead 4 -- `P.Wsg` initial std 3.01 is enormous. REJECTED.

3.01 rad of prior std on a 2-DoF direction is meaningless (larger than the
diameter of the space). Tightened to 0.1 rad, about 6 deg, which is a fair
reflection of how well a 20-sample accel average can level the rig.

ori 1.013 -> 0.955 (1.3 sigma; and the ensemble sd rose 0.061 -> 0.085, so if
anything the filter got *less* repeatable), `ate002` 0.0928 -> 0.0967. Below the
0.005 m limit but adverse, in exchange for nothing measurable. Rejected under
"keep config edits minimal": there is no point carrying a config delta into
someone else's hand-merge for a 1.3-sigma effect.

Why it does so little: after M1 the published attitude is levelled by the filter's
*current* `Rsg` at every timestep, so the quality of the *initial* `Rsg` no longer
leaks into the reported error. M1 already collected this lead's payoff.

### Lead 5 -- "the rooms start stationary for 4-6 s; average accel over that". REJECTED; the premise is false.

TUM-VI room1-room6 do **not** start stationary. Mean |gyro| over the first samples
is 6-18 deg/s from the very first IMU record -- the rig is already hand-held and
moving when recording starts. There is no static window to average.

The existing 20-sample average is therefore *better* than a longer one, which
integrates more centripetal and linear acceleration into the "gravity" estimate.
Screened a 200-sample window with each sample derotated by the integrated gyro
before averaging (`s_ginit`, the honest version of the lead -- a plain 200-sample
mean is worse still): ori 1.013 -> 1.009, a wash, `ate002` 0.0928 -> 0.0986.
Same reason as lead 4: M1 made the reported attitude depend on the converged
`Rsg`, not the initial one.

### Lead 6 -- integration-scheme consistency. Checked; nothing wrong.

`integration_method: "PrinceDormand"`, an embedded RK4(5). Read
`src/princedormand.cpp`: every stage calls `ComputeMotionJacobianAt` at *that
stage's* state and feeds the result to both `MotionCovSlope` and the transition
accumulation, so the nominal state, the transition Jacobian and the covariance are
integrated with the same stages and the same coefficients -- there is no
first-order-P / fourth-order-X mismatch. `ComposeMotion` integrates attitude on
SO(3) (`X.Rsb *= SO3::exp(gyro_calib * dt)`, right multiplication, matching the
body-frame error state) rather than as a normalized quaternion. The gyro sign
convention and the `td` shift are the same in the propagation and in
`Feature::ComputeJacobian`.

Independent evidence that the local increment is fine: `rpe_ori` over 8 m segments
is 0.515 deg, *better* than OpenVINS' 0.614. The residual is a slow global effect,
which is not what an integration bug looks like.

### Lead 7 -- gauge fixing, `feature_owner_change_cov_factor`, `td`.

The gauge half of this lead is the one that paid: see `m2-gauge-axis.md`.

`feature_owner_change_cov_factor` (1.5) inflates a feature's covariance when its
anchor group retires. That is a depth/position concern and it is applied to
feature blocks, so it was left to the position agent.

`td`: the online temporal calibration is a single scalar with prior std 1e-5 s. At
the sequences' mean angular rate of 32 deg/s, a `td` error of 1 ms costs 0.032 deg
of attitude, so to explain even the 0.42 deg tilt floor `td` would have to be wrong
by 13 ms -- two IMU samples and half a frame interval, which the visual updates
would not tolerate. And the floor is not XIVO's at all (OpenVINS has the same one),
so `td` is not the mechanism for it either. Left alone.

## Not one of the prompt's leads: gyro white noise. REJECTED.

`Qimu.gyro` is 2.4e-4 rad/s/sqrt(Hz); TUM-VI's kalibr report says 1.6e-4. Unlike the
*bias* random walk (lead 2), lowering this makes the filter trust the gyro *more*,
which is the opposite direction and worth a separate test -- if attitude were being
dragged around by inconsistent visual updates, tightening the gyro should help.

It does not: ori 1.058 (against M2's 0.949), `ate002` 0.1014 (breaks the budget).
`rpe_ori` improves marginally, 0.5149 -> 0.5140. The attitude increment was never
the problem; `rpe_ori` was already better than OpenVINS' before any of this work.
