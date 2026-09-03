# Dynamic initialization for XIVO

**The brief.** Some EuRoC sequences are not static at the beginning. XIVO must
detect whether the sensor platform is moving, keep today's path when it is not,
and when it is, solve a small bundle adjustment for the initial velocity and the
bias terms and hand that to the filter. OpenVINS' `ov_init` as a reference if
needed. Plan first, milestones, an experiment that could have failed before each
commit, then a report and a README section, then merge to `auto`.

**What shipped.** A two-cue motion detector, a closed-form linear initializer, a
hand-rolled Levenberg-Marquardt bundle adjustment over a 2.0 s window solving for
41 frame poses, 41 velocities, both IMU biases and every feature, and a
divert-and-replay dispatcher that holds the estimator's messages while the
decision is made. `dynamic_init.enabled` is **true** in both euroc configs. From
`auto` @ `8d2d052`, seven commits on `auto-dyninit`, +10603 lines, `ctest` 26/26
(23 before).

**The headline.** On EuRoC started mid-flight, **69 of 220 runs diverge without
dynamic initialization and 0 with it**, and no run diverges that did not diverge
before. Absolute orientation error falls **0.54 deg (stereo) / 3.82 deg (mono)**
on the sequences that finish either way. At a table start the feature is neutral:
+0.0065 m against a null control whose own scatter is 0.006-0.009 m. It costs
~0.9 s of one-off compute on a dynamic start, 0.8-2.1% of end-to-end throughput,
and +8 MB of peak RSS that belongs to the pre-init tracker rather than the solver.

---

## 1. What was broken

`Estimator::InitializeGravity()` was the whole initializer: average 20
accelerometer samples, set the 2-DoF gravity state from `FromTwoVectors`.
**`X_.Vsb`, `X_.bg` and `X_.ba` were never estimated** -- they stayed at zero.
Correct if and only if the rig is at rest.

The one guard was M4-era `gravity_init_max_accel_dev`, which rejects a sample when
`| |a| - |g| | > dev`. It can wait for a quiet stretch; it cannot initialize during
motion, and its statistic is blind in the case that matters:

> **An accelerometer cannot distinguish constant-velocity motion from rest.**
> Specific force is `R'(a - g)`; at constant velocity `a = 0` and the magnitude
> reads exactly `|g|`. Accelerometer statistics detect *acceleration*, never
> *velocity*.

Replaying the shipped gate sample by sample and reading groundtruth velocity at the
instant it fires (`harness/init_diag.py`, M0):

| sequence | fires at | samples skipped | true \|v\| there |
|---|---|---|---|
| MH_01_easy | 2.100 s | 401 | **0.671 m/s** |
| MH_02_easy | 1.020 s | 185 | **0.481 m/s** |
| the other nine | 0.095-0.505 s | 0-82 | <= 0.034 m/s |

Two of eleven sequences assert `Vsb = 0` while doing 0.5-0.7 m/s. MH_01 was also
the only Machine Hall sequence XIVO lost to OpenVINS on, and the one where mono
`acc` diverged 10 runs of 10.

TUM-VI room1-6, checked at the same time: **zero of six** moving. So the TUM-VI
round is a static-initialization result and is not at risk from this work -- and
room1-6 became a six-sequence negative control that M4's regression could demand
bit-identical output on.

## 2. The design, and where it departs from OpenVINS

Nothing from OpenVINS is linked or copied; `ov_init` was read for structure. Three
places the design is deliberately different, each because a measurement said so.

### The detector needs vision, and needs it to be rotation-free

OpenVINS decides static-vs-dynamic from accelerometer sample standard deviation
plus a raw pixel disparity threshold of 10 px. Both cues are individually unsound
and measurement says so. Accel sd over 0.5 s windows in the first 5 s: V1_01 reads
0.831 and V2_03 reads 1.093 m/s^2 while **genuinely static** (somebody bumping the
table), against MH_02's 0.499 while genuinely moving -- so no single-window
threshold separates them. The minimum over candidate windows does separate, 0.499
against 0.260, under 2x. Raw disparity conflates translation with rotation: a fixed
threshold calls slow near-field motion static and a stationary-but-rotating rig
dynamic.

The plan's fix was to de-rotate the flow with the gyro. **That was measured and
rejected**: it assumes an unbiased gyro, and EuRoC's turn-on gyro bias is 1.8
px/frame of predicted motion that never happened -- larger than the signal, and
circular, since the bias is one of the unknowns. What ships instead fits the
rotation **from the images** and uses its residual. Margins over all 17 sequences:

| cue | margin |
|---|---|
| image-fitted rotation residual (**ships**) | **6.8x** |
| raw disparity (OpenVINS' cue) | 5.0x |
| accel sd, min over windows | 1.93x |
| gyro de-rotation (planned, rejected) | 1.37x |

Shipped rule: **static iff (min-window accel sd < `imu_thresh`) AND (image-fitted
flow residual < `flow_thresh`)**, dynamic as the fallback. Two independent cues
that must agree, one of which is the only sensor here that sees translation at
constant velocity.

### Stage A: `|g|` enforced by a secular equation, not a degree-6 polynomial

Unknowns `[p_F1..p_FN, v_I0, g_I0]`, size `3N + 6`, built from preintegration at
the prior bias and one `H_proj = [[1,0,-u],[0,1,-v]]` pair of rows per observation.
Biases are held, not recovered -- Stage A only has to reach Stage B's basin.

`|g| = 9.81` must be enforced or the solve absorbs bias and scale error into
gravity's magnitude. Schur-eliminate features and velocity to a 3-variable problem
`min g'Dg + 2d'g s.t. |g| = r`, eigendecompose `D`, and root-find the secular
equation `sum_i e_i^2/(l_i - lambda)^2 = r^2` for `lambda < l_min`, where the left
side is monotone -- a safeguarded Newton iteration reaches the unique global
minimizer in ~25 lines with no rank or conditioning preconditions. OpenVINS instead
feeds a symbolically-expanded degree-6 polynomial to a 6x6 companion-matrix
eigensolve with an explicit rank check that can bail out. The unit test verifies
globality by brute force against a sphere grid.

### Stage B: the bundle adjustment, hand-rolled

Gravity-aligned world frame `W` with `z = -g`, so the gravity *direction* is
refined implicitly through the free roll/pitch of frame 0 rather than carried as a
variable. Per frame: `R_W<-Ik` (local `SO3::exp` increments), `p_Ik^W`, `v_Ik^W`.
Global: `bg`, `ba`. Per feature: `p_F^W`. Residuals are 9-dim IMU preintegration
between consecutive frames **with the `d/dbg`, `d/dba` Jacobians carried through**
-- that is what makes the biases observable rather than merely present -- plus
Cauchy-robustified reprojection. Gauge: `p_0 = 0` fixed and a tight prior on the
yaw of `R_W<-I0`; scale is observable from the IMU so nothing else is fixed.
Levenberg-Marquardt with Schur complement on the feature block, Cholesky on the
reduced camera system, analytic Jacobians throughout, all verified against central
differences to 1e-6.

`find_package(Ceres)` was deliberately not added even though Ceres is vendored:
XIVO's dependency set does not change, so nothing downstream rebuilds, and the
repo's standing "nothing from OpenVINS' stack is linked" property survives.

### The handoff, gauge-invariant by construction

Take the **last** window frame as the init instant:

    Vsb = R_W<-Ilast' * v_Ilast^W
    bg = bg,  ba = ba
    Rsg from FromTwoVectors(-g_, R_W<-Ilast' * (-g_W)), with Wsg(2) = 0

Every input is a world quantity rotated into body coordinates, so the arbitrary yaw
the BA settled on cannot leak into the filter, and reusing the existing
`FromTwoVectors`/`Wsg(2)=0` code guarantees the 2-DoF constraint exactly. The whole
handoff degenerates to today's static path when `v = 0` and the biases are zero.
Window features are discarded; the filter starts clean.

The init-window tracker is **self-contained** (`goodFeaturesToTrack` +
`calcOpticalFlowPyrLK` inside the initializer), not XIVO's `Tracker`, because the
tracker is entangled with the feature memory pool, the `MemoryManager` lifecycle
and `Predict`. Keeping the init window out of that machinery is what makes "the
static path is bit-identical" a structural guarantee rather than a hope.

## 3. Results

### R1 -- the detector is 22 for 22, with a factor of 21 to spare

`bin/init_probe -start` asks the shipped `MotionDetector` directly. Both start
conditions, all eleven sequences, one threshold pair:

| | t=0 | t=55 s |
|---|---|---|
| called dynamic | MH_01, MH_02 | all 11 |
| true speed of those | 0.28-0.48 m/s | 0.14-1.42 m/s |
| called static | the other 9 | none |
| true speed of those | <= 0.006 m/s | -- |

Every one of the 22 calls is correct. At t=0 the static class peaks at **0.097 px**
of flow residual and the dynamic class floors at **2.03 px**, a factor of 21 with
`flow_thresh` at 0.25 in between. The tightest call anywhere is V2_01 at t=55,
genuinely creeping at 0.14 m/s, and it still reads 0.445 px and 1.21 m/s^2.

An independent corroboration fell out of it: on the static sequences the detector's
`gyro_bias_hint` reads 0.072-0.086 rad/s and EuRoC's groundtruth gyro bias is
0.076-0.082 rad/s on all eleven. Two unrelated estimators agreeing on a 4.5 deg/s
bias that the static path seeds as zero.

### R2 -- the two stages, against groundtruth and against EuRoC's own biases

All 11 sequences, shipped defaults, no per-sequence tuning
(`harness/linear_check.py --ba`):

| | Stage A | Stage B | change |
|---|---|---|---|
| velocity error, mean | 0.1727 m/s | **0.0164 m/s** | **-90.5%** |
| velocity error, max | 0.2257 m/s | 0.0459 m/s | |
| gravity tilt, mean | 4.148 deg | **0.892 deg** | **-78.5%** |
| gravity tilt, max | 5.682 deg | 2.606 deg | |
| gyro bias error, mean | (not estimated) | **0.00292 rad/s** | against a true \|bg\| ~ 0.08 |

Every one of the 11 improves on both metrics. The bias figure is an **external
check**, not self-consistency: EuRoC solves for the biases in its own batch
estimator and publishes them in `state_groundtruth_estimate0` columns 11-13.
0.0029 against 0.08 is a 3.6% recovery from a zero seed. On the two sequences the
detector actually routes: MH_01 velocity 0.1443 -> **0.0459** m/s and tilt 4.166 ->
**0.829** deg; MH_02 0.1087 -> **0.0184** and 4.139 -> **0.973**.

Stage A alone is *worse* than the static initializer on a static rig (4.15 deg of
tilt), which is why Stage B is mandatory rather than a refinement, and it claims
0.15-0.23 m/s on nine sequences whose true speed is ~0.01 -- which is why the
detector's routing is load-bearing rather than a convenience.

### R3 -- the static path is bit-identical, proved rather than argued

The prediction M4 lived or died by: with the feature on, the nine static EuRoC
sequences must be **bit-identical** to the shipped result, because the detector
selects the static path and the static path is untouched.

Divert-and-replay makes that non-trivial. The dispatcher holds every message while
it decides, then replays them. Freezing the temporal calibration (`P.td = 0`, so
the EKF can never move `td` and every frame enqueues with the same offset) makes
the claim exact, and on V1_01: `off 2909 poses, on 2891 | offset 18 | aligned tail
2891 vs 2891 | mismatches: 0`. **Every shared pose byte-identical.**

- **18 of 18** static rows exact (9 sequences x 2 modes) on EuRoC.
- **12 of 12** exact on TUM-VI room1-6 -- the dispatcher does not fire on a dataset
  it was never tuned for.
- The two rows that change are the two M1 predicted before any of this was wired
  up. MH_01 starts reporting **0.55 s earlier** with the dynamic branch, in both
  modes: the static path has to wait out a wrong initial velocity before
  `VisionInitialized()` turns true.

It also confirms the message-count argument: `MaintainBuffer` pops exactly one
message per message pushed, so the replay is *in step* with the unbuffered filter
from the handoff onward, not merely close. Had the burst put the estimator ahead or
behind, the aligned tails would have had different lengths.

With the shipped (unfrozen) `td` the arms are not bit-identical, and that is
unavoidable: buffered frames are enqueued with `td_0`, no EKF update having run yet
to move `td`, so an image/IMU tie in the estimator's heap flips and gating
decisions follow. The nine static sequences still move by 0.009 m of ATE on average
and 0.071 m at worst on **provably identical arithmetic**. That number is not a
regression, it is XIVO's single-run ATE noise floor on EuRoC, and it is why every
accuracy number below is an n=10 ensemble mean.

### R4 -- at a table start the feature is free and does nothing

n=10 per arm, all eleven sequences, `ate_002` in metres. `off` is the shipped
config with one key flipped, not "the base config left alone".

| | stereo off | stereo on | delta | mono off | mono on | delta |
|---|---|---|---|---|---|---|
| MH_01 (dynamic) | 0.0746 | 0.0789 | +0.0043 | 0.1051 | 0.1134 | +0.0083 |
| MH_02 (dynamic) | 0.0555 | 0.0642 | +0.0087 | 0.1058 | 0.1105 | +0.0047 |
| 9 static (null control) | | | **+0.0003 +- 0.0032** | | | **+0.0030 +- 0.0042** |
| 2 dynamic | | | **+0.0065** | | | **+0.0065** |

The honest reading: **at a table start the dynamic path is not measurably better or
worse than the static one**, on any of the five metrics, in either mode. It is not
supposed to be -- MH_01 and MH_02 leave the table at 0.28-0.48 m/s, slow enough
that averaging twenty accelerometer samples is only 0.4-1.2 deg wrong about
gravity. This is the no-regression check and it passes.

### R5 -- mid-flight it is the difference between working and not

Every public VIO dataset begins with the rig on a table, which is why only 2 of 11
EuRoC sequences reach the dynamic branch and why 9 of 11 can only measure the
feature's *cost*. `pyxivo.py -start_sec N` drops the first N seconds of both
streams and turns the estimator on mid-flight -- still EuRoC, still with
groundtruth, still the same evaluator, only the initial condition is hard. At
**N = 55 s** ten of eleven sequences are moving at 0.16-1.42 m/s over the window
and every one has >= 29 s of trajectory left to score.

What the static initializer does there is not a degradation, it is a failure: the
accelerometer average is 0.83-**23.98** deg from gravity (against 0.19-2.54 deg at
t=0), and it then asserts the platform is at rest while it is doing 1.4 m/s.

n=10 per arm, both arms started 55 s in, `ate_002` in metres. `DIVERGED` counts
members above 100 m -- runs that did not degrade, they failed:

| | stereo off | stereo on | mono off | mono on |
|---|---|---|---|---|
| MH_01 | 0.0449 | 0.0478 | **10/10 diverged** | 0.1070 |
| MH_02 | 0.0347 | 0.0337 | 0.0704 | 0.0628 |
| MH_03 | 0.2065 | **0.0934** | 0.4470 | **0.1543** |
| MH_04 | 0.0745 | 0.0789 | 0.8423 | **0.6143** |
| MH_05 | 0.1318 | 0.1318 | 0.5738 | 0.5738 |
| V1_01 | 0.0554 | 0.0558 | 0.1249 | 0.1175 |
| V1_02 | **9/10 diverged** | 0.0530 | 1.2766 | **0.0937** |
| V1_03 | **10/10 diverged** | 0.1444 | **10/10 diverged** | 0.3103 |
| V2_01 | 0.0323 | 0.0286 | 0.0815 | 0.0900 |
| V2_02 | 0.0519 | 0.0518 | **10/10 diverged** | 0.1656 |
| V2_03 | **10/10 diverged** | 0.1117 | **10/10 diverged** | 0.1071 |
| **divergence census** | **29 of 110** | **0 of 110** | **40 of 110** | **0 of 110** |
| mean delta, comparable seqs | | -0.0138 +- 0.0142 | | -0.2443 +- 0.1632 |

**The census is the result; the deltas are the footnote.** On the sequences that
finish either way the worst regression anywhere is +0.0044 m stereo / +0.0085 m
mono -- inside the t=0 null's own scatter -- against best cases of -0.113 m and
-1.183 m.

MH_05's exact `+0.0000` in both modes is the acceptance gate working, not a
rounding artifact: its window's reprojection median is 1.649 px, above the 1.5 px
gate, so the solve is discarded and `cmp` shows the `on` run is **byte-identical**
to `off`. A rejected solve costs accuracy exactly nothing. MH_04 is rejected far
more decisively (median 9.8 px) and that is right too: its mono `off` run is 0.84 m,
so the static path is already struggling, and a window the BA cannot fit to better
than 10 px would not have helped -- yet `on` still improves it to 0.61 m, because
the divert alone shifts where the filter starts.

### R6 -- orientation is where the effect lives, and that is not an accident

`ate_002` Horn-aligns and is therefore **blind to a global rotation**; `ov_eval
... posyaw` fixes only position and yaw and charges roll and pitch in full. What
the static initializer gets wrong at a moving start *is* a global tilt, so
`ate_002` is the least sensitive metric available for this feature.

| t=55 s | stereo, 8 comparable | mono, 7 comparable |
|---|---|---|
| `ate_002` m | -0.0138 +- 0.0142 | -0.2443 +- 0.1632 |
| `ov_ate_pos_m` | -0.0152 +- 0.0175 | -0.2454 +- 0.1603 |
| `ov_ate_ori_deg` | **-0.5399 +- 0.3295** | **-3.8151 +- 3.2158** |
| `ov_rpe8_pos_m` | -0.0083 +- 0.0062 | -0.2066 +- 0.1655 |
| `ov_rpe8_ori_deg` | -0.1043 +- 0.0649 | -1.9046 +- 1.7322 |

Against a t=0 null of 0.03 deg. The clearest single number is mono V1_02, **25.31
deg -> 2.28 deg**, and `harness/seed_error.py` measures the same thing at the other
end of the chain: at t=55 V1_02's accelerometer average is **13.09 deg** from
gravity and the BA's gravity is **0.78 deg** from it. A tilt seeded at
initialization is not something a VIO filter works off.

The counterexample is worth as much. V1_01 is the one sequence where the solved
gravity is *worse* than the accelerometer average (3.14 vs 1.90 deg) and its
orientation error does not move at all: 5.85 -> 5.80 deg mono. Its 5.8 deg is
therefore not the seed's, a worse seed did not make it worse, and "better gravity"
is not a universal explanation for anything here.

### R7 -- cost: a one-off, and 8 MB that is not the solver's

Two instruments, because the questions differ. `bin/init_probe -dispatch` runs the
shipped dispatcher over real data and times it (n=5, one core); the timing pass
runs the whole filter on one pinned core, serial, `-mode runOnly`.

| ms, one core | images | buffer (2 KLTs) | solve (window + A + B) | total |
|---|---|---|---|---|
| static verdict (9 seqs, t=0) | 20-21 | 231-308 | 0 | **231-308** |
| dynamic (MH_01/02, t=0) | 41 | 359-389 | 499-590 | **888-965** |
| dynamic (11 seqs, t=55) | 32-41 | 301-429 | 296-1298 | **724-1634** |

So **~0.9 s of one-core compute on a dynamic start and ~0.26 s on a static one**.
It is one-off (`decided_` short-circuits every entry point afterwards) and it is
paid while messages are buffered rather than dropped, so on a 20 Hz stream it fits
inside real time -- but the handoff is a latency spike, and first-pose latency is
~1.0 s of data on the static path against ~2.0 s plus the solve on the dynamic one.

| one core, serial, mean over 11 | off | on | delta |
|---|---|---|---|
| stereo `fps_wall` | 86.7 | 86.0 | **-0.8%** |
| mono `fps_wall` | 157.8 | 154.5 | **-2.1%** |
| stereo `peak_rss_mb` | 96.9 | 106.2 | **+9.3 MB** |
| mono `peak_rss_mb` | 85.2 | 93.0 | **+7.8 MB** |

The per-frame cost is **zero**: subtract the probe's one-off from each sequence's
wall-clock delta and the residual is -0.12 s (stereo) and -0.02 s (mono) over 11
sequences, with per-sequence scatter of +-0.3 s -- the same +-0.3 s two runs of the
*identical* config differ by. Mono's -2.1% exceeds stereo's -0.8% because the
constant is the same and mono's runs are half as long.

The memory is the interesting half, because it is **not the bundle adjustment's**.
MH_03 at t=0 takes the static path, never builds a problem, and still pays +7.1 MB.
A third arm settles it: `max_wait_sec = 0` constructs the dispatcher, diverts
messages, and gives up on the first one, so the divert path is live and zero images
are tracked -- and mono MH_01 peak RSS goes 87.8 (off) / 96.4 (on) / **88.1**, wall
23.33 / 24.31 / **23.31**. Both costs vanish. So the 8 MB is the **pre-init KLT**
(two trackers, `goodFeaturesToTrack` and its pyramids, over 20-41 frames), not the
dispatcher's existence, not the buffered messages, not the solver. XIVO's own
retained structures are bounded by construction and far too small to be it: 41
frames of at most 160 observations (0.26 MB), one 752x480 gray clone (0.36 MB), at
most 600 IMU samples in each of two buffers (0.07 MB) -- under 1 MB total, so
releasing the dispatcher after handoff could not return what is missing.

## 4. Negative results and rejected alternatives

Each of these was implemented or measured, not argued away.

| what | why it was rejected |
|---|---|
| **Gyro de-rotated flow** (the planned detector cue) | assumes an unbiased gyro; EuRoC's turn-on bias is 1.8 px/frame of motion that never happened, larger than the signal, and circular since the bias is an unknown. Margin 1.37x against the image-fitted 6.8x. |
| **Seeding the filter's covariance from the BA marginal** (M6) | 10-500x too tight and *worst calibrated where the window is hard*; ranks windows at Spearman +0.01 / -0.22 / -0.47 while the config priors already score rms \|e\|/sigma 0.97 / 1.42 / 0.88. Built, measured (ten of ten metrics worse or flat, stereo ATE 0.078 -> 0.083 m), reverted. `notes-dyninit/m6-covariance.md`. |
| **`sigma_pix` below 1.0** | improves every seed metric nearly monotonically (MH_01's velocity error 0.410 -> 0.136 m/s) and end-to-end does the opposite: mono MH_05 goes 0.574 -> 0.942 (0.25) -> 1.144 (0.125), because pulling its reprojection median under the acceptance gate makes the filter *accept* a seed worse than the static fallback it displaces. The tuning signal and the gate are the same quantity. |
| **A 31-frame (1.5 s) window** | a wash on the mid-flight mean and identical on the census, but its worst surviving stereo run is V2_03 at **5.56 m** against w41's 0.16 m. 2.0 s buys a tail. |
| **Dropping the `\|g\|` constraint in Stage A** | the solve absorbs bias and scale error into gravity's magnitude. |
| **Depth-reweighting Stage A's rows, gating on `g_cond`** | measured, no benefit. |
| **A 0.5 s / 11-frame window** (the tracker's first-cut default) | under 0.3 px of tracking noise it costs 1.14 m/s against 0.060, a 19x penalty from one knob; real data puts the plateau at 1.0-1.5 s. |
| **Forcing the accel-mean gravity prior** in Stage A rather than using it as a discriminator | 25x worse when the solve has not flipped. `PriorMode::Check` fires on all 4 synthetic flips, none of the other 20 rows, and never on real EuRoC. |
| **Ceres for the BA** | vendored, and deliberately unused: `find_package(Ceres)` would change XIVO's dependency set and break the "nothing from OpenVINS' stack is linked" property. Cost paid down with numerical-Jacobian checks and exact-answer synthetic problems. |

Two plan targets were corrected against measurement rather than quietly loosened:

- **M3's noiseless velocity gate (1e-6 m/s) is not attainable and should not have
  been written.** The solver reaches a cost of 1.41e-22 -- *below* the 1.76e-22 the
  exact truth scores -- with 1.6e-13 px reprojection RMS, and still sits 0.0094 m/s
  from the true velocity. A 1 s window has a nearly flat direction: a global scale
  on `(p, v, f)` trades against an accelerometer bias along the mean acceleration.
  The recovered scale is 1.00686 and dividing it out leaves 5.2e-4 m/s. More
  excitation makes it worse (4x angular rate: 0.034; 8x acceleration: 0.036); only
  a longer span helps (0.0056 at 2 s). The gate is now: the identifiable parts to
  machine precision (`bg` at 1.4e-16), the scale-corrected velocity to 2e-3, the
  scale within 2% of 1, and the cost no worse than the truth's. This flat direction
  is a real property of short-window visual-inertial initialization and is the
  reason `sigma_ba_prior` exists at all.
- **An n=3 null control that looked like a mechanism.** At n=3 the nine static
  sequences moved +0.0043 m (sem 0.0012, 8 of 9 the same sign) and the same offset
  came back to the digit under two different `on` configurations, which reads as
  deterministic. At n=10 it is gone: +0.0003 +- 0.0032. The reproducibility was
  reproducibility of the *same three jitter values*. Recorded because it is exactly
  the trap the harness exists to avoid and it still caught me once.

## 5. Two bugs the instruments found

**A coin-flip demotion of moving platforms to the static initializer.**
`bin/init_probe -dispatch`, written to measure cost, reported `window build failed`
on MH_01 -- a sequence the filter had been initializing dynamically for two
milestones. Neither was wrong. `InitWindow::Build` refuses a window whose last
frame the IMU does not reach; on EuRoC **every** image timestamp coincides exactly
with an IMU sample (3682 of 3682 on MH_01); and which of the two a caller is handed
first is decided by a non-stable `std::sort` in `DataLoader` and a timestamp-only
heap in `MaintainBuffer` -- unspecified in both. The estimator was on the lucky side
of that tie, the probe on the unlucky one, where a moving platform is quietly
demoted. `Decide()` now waits for IMU coverage before solving (one more message,
<= 5 ms at 200 Hz).
`InitDetectTest.DispatchDoesNotDependOnCoincidentMessageOrder` runs the dispatcher
over both orders and fails with exactly `window build failed` if the check is
removed. The fix is bit-identical on EuRoC -- three stored members re-run
afterwards reproduce byte for byte -- and probe and filter now agree to the digit.

**A gauge reference in the wrong place.** The M6 covariance test first disagreed
with a dense `(J'J)^-1` by 1e-3 relative. The yaw prior's Jacobian is a function of
`R_0 R_ref,0'` -- the *seed's* attitude, not the solution's -- so referencing it
anywhere else is a different Hessian. With `seed` as the reference the worst of 81
entries disagrees by **6e-6**. Recorded in the test, because 1e-3 is exactly the
size of disagreement that would otherwise have been blamed on the Schur
elimination.

## 6. Protocol

Carried over from the previous rounds and not re-litigated:

- **Never compare single runs.** Jitter ensembles, screen at n=3, ship at n=10,
  report the sd of the sequence-mean across members. The jitter knob here is
  `P.Tsb *= 1 + k*1e-6`, **not** the established `X.Vsb += k*1e-6`: a dynamic
  initializer *solves for* the initial velocity and overwrites the perturbation, so
  with the usual knob every member of the `on` arm is bit-identical on precisely the
  two sequences the experiment exists to measure and the ensemble reports a
  confident `+-0.0000`. Validated on the `off` arm, where both knobs work and give
  the same means to within 0.005 m.
- **Two metrics, because they disagree by design.** `evaluate_ate.py
  --max_difference 0.02` (blind to a global rotation) and `ov_eval ... posyaw`
  (charges roll/pitch in full). For this feature the second is the sensitive one.
- **Divergence is a census, never an average.** Members above 100 m are counted,
  not averaged; a mean that mixes 0.05 m and 9000 m describes neither.
- **Never quote FPS or RSS from an accuracy pass.** One pinned core, ASLR off, all
  pools at 1, `-mode runOnly`.
- **Diverted-arm alignment was ruled out, not assumed.** `on`'s dump begins 18
  poses (0.9 s) later, so the two arms could have been Horn-aligned over different
  pose sets. `trunc_control.py` re-scores `off` truncated to `on`'s first timestamp:
  the difference is under 0.0001 m on every sequence in both modes.
- One configuration for all eleven sequences, both modes, no per-sequence overrides.

## 7. What ships

`dynamic_init.enabled: true` in `cfg/euroc_stereo.json` and `cfg/euroc_mono.json`,
with `window_frames: 41`, `sigma_pix: 1.0`, `max_pixel_median: 1.5`,
`max_wait_sec: 3.0`, `flow_thresh: 0.25` px, `imu_thresh: 0.35 m/s^2`, `ba_iters:
30`, and the M2/M3 defaults elsewhere.

The case for turning it on is the census, not the mean. What it does **not** do: it
does not improve a table start, and it declines to try on two of eleven mid-flight
windows (MH_04 at 9.8 px of reprojection median, MH_05 at 1.65) where it falls back
to the static path bit-identically. Both are the gate working. And the mid-flight
numbers come from a start-time device, not a second dataset -- `-start_sec 55` is
EuRoC with a hard initial condition, which is the closest thing to dynamic-start
data that a public benchmark with groundtruth provides.

## 8. Milestones, and where each measurement lives

| | commit | what | the experiment that could have failed |
|---|---|---|---|
| M0 | `c8fe390` | the plan, and `harness/init_diag.py` | 2 of 11 EuRoC moving at init, 0 of 6 TUM-VI rooms |
| M1 | `2436bb8` | `init_detect.{h,cpp}`, `bin/init_probe` | 17 of 17 classified, 20.9x margin at the decision horizon |
| M2 | `cd5503a` | preintegration + Stage A, `bin/linear_probe` | MH_01 0.144 m/s, MH_02 0.109 -- within the bias error Stage A cannot see |
| M3 | `944ff0f` | `init_ba.{h,cpp}`, LM + Schur | -90.5% velocity, -78.5% tilt, `bg` to 3.6% against EuRoC's own solved bias |
| M4 | `e9c9599` | `init_window`, `init_dispatch`, the handoff | 18 of 18 EuRoC + 12 of 12 TUM-VI rows byte-identical on the static path |
| M5 | `39e2d63` | the shipped config and the evaluation | 69 of 220 mid-flight runs diverge without it, 0 with it |
| M6 | `78f05d4` | the BA marginal, measured and **not** seeded | ten of ten metrics worse or flat; reverted |

Per-milestone notes: `notes-n-prompts/notes-dyninit/m{0..6}-*.md`. Harness mirrored
into `notes-n-prompts/notes-dyninit/harness/`.

## 9. Repro

From the worktree root with the venv on `PATH`. Results land outside the tree, in
`../results/dyninit/`.

```sh
# the diagnosis (section 1)
python3 notes-n-prompts/notes-dyninit/harness/init_diag.py \
    --dataset euroc --root ../data/euroc --cfg cfg/euroc_stereo.json

# the detector's verdict and statistics at any instant (R1)
./bin/init_probe -cfg cfg/euroc_stereo.json -root ../data/euroc -dataset euroc \
    -seq MH_01_easy -start 0
# what the shipped dispatcher decides, and what the decision cost (R7)
./bin/init_probe -cfg cfg/euroc_stereo.json -root ../data/euroc -dataset euroc \
    -seq MH_01_easy -dispatch

# Stage A then Stage B on one real window, scored against groundtruth (R2)
python3 notes-n-prompts/notes-dyninit/harness/linear_check.py --ba
python3 notes-n-prompts/notes-dyninit/harness/seed_error.py --start 55

# the static path is bit-identical (R3)
./notes-n-prompts/notes-dyninit/harness/m4_bitident.sh --profile euroc_mav
./notes-n-prompts/notes-dyninit/harness/m4_bitident.sh --profile tumvi_room \
    --dynamic "" --no-ship

# the evaluation: off vs on, n=10, both modes (R4, R6)
./notes-n-prompts/notes-dyninit/harness/m5_ensemble.sh --members 10 --mode both
# the same, all eleven started mid-flight (R5, R6)
./notes-n-prompts/notes-dyninit/harness/m5_ensemble.sh --members 10 --mode both \
    --start-sec 55

# cost: the probe's one-off, then one-core throughput and peak RSS (R7)
./notes-n-prompts/notes-dyninit/harness/m5_cost.sh

# the covariance screen, and the elimination against a dense inverse (section 4)
python3 notes-n-prompts/notes-dyninit/harness/m6_cov.py --start 55
./bin/unitTests_init_ba --gtest_filter='*MarginalCovariance*'
```
