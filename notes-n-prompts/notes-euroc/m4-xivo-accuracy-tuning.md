# M4 -- tuning XIVO's accuracy on EuRoC MAV

Milestone M4 of `notes-n-prompts/plan-euroc.md`. Branch `auto-eurocacc`, worktree
`xivo-eurocacc`, forked from `auto-euroc` at `721b417`.

M3 left two things on the table: **15 of 66 stereo runs diverged**, and a
diagnosis of why. M4 removes the divergences, then confronts the harder half of
the brief -- *one* configuration for all eleven sequences -- and finds that on
EuRoC one number for the visual measurement noise is not merely suboptimal, it is
provably unable to serve both halves of the dataset. The fix is to stop choosing
it: estimate it online from the filter's own consistency statistic.

Every number below is a mean over an ensemble (jitter members, `--jitter N`), on
**all eleven sequences**. That matters -- see §3.


## 1. The two divergence mechanisms, confirmed

### 1.1 `P.Wsg` -- a 1.73 rad prior on which way is down

M3's suspicion was `P.Wsg = 3.01`, the shipped TUM-VI prior variance on the
two-DOF gravity direction state, i.e. a prior sigma of **1.73 rad**. With a prior
that loose the filter can absorb an early vision residual by rotating its
estimate of gravity instead of correcting the pose -- and a wrong gravity
direction feeds straight back into the acceleration estimate. Positive feedback,
which is the shape required to turn a 1e-6 m/s velocity perturbation into 71 m of
error.

Confirmed by bracketing it (full 11, stereo, n=3):

| `P.Wsg` | prior sigma | mean ATE (m) |
| --- | --- | --- |
| 1e-6 | 0.001 rad | 0.109 |
| 1e-4 | 0.01 rad | 0.116 |
| **2e-3** | **0.045 rad** | **0.105** |
| 1e-2 | 0.1 rad | 0.112 |
| 3.01 (shipped) | 1.73 rad | diverges intermittently |

The response is flat across four orders of magnitude below the shipped value, so
the load-bearing claim is "not 3.01", not "0.002 specifically". 0.002 is chosen
because it is the flat region's interior rather than its edge. **This alone
removes every divergence: 0/66 on stereo, down from 15/66.**

### 1.2 MH_01_easy -- a gravity initialiser with no stationarity gate

`Estimator::InitializeGravity` averages `gravity_init_counter` accelerometer
samples with no check that the platform is actually still. M3 measured that
window on all eleven sequences: ten are within 0.30 m/s^2 of gravity, and
**MH_01_easy is off by -1.463 m/s^2** -- its 20-sample window averages
8.347 m/s^2 against 9.810. MH_01 is already being carried when its first IMU
sample lands.

Fix: `gravity_init_max_accel_dev`, which rejects a sample whose `| |a| - |g| |`
exceeds a threshold, in m/s^2, with 0 disabling the gate (so the default
behaviour of every shipped config is unchanged). At 0.1 it moves MH_01 from
0.145 to 0.118 and leaves the other ten identical to four decimals -- exactly the
signature a correct gate should have. Committed in `5493562` with four unit tests
(`src/test/unittest_gravity_init.cpp`).

Note what this is *not*: M3 hypothesised the same defect explained MH_04's
deterministic 6/6 failure, and measurement said no -- MH_04's init window is
within 0.050 m/s^2 and its max |w| is 0.11 rad/s. MH_04 is genuinely still when
XIVO initialises; the take-off that broke OpenVINS' initialiser starts 1.5 s
later. MH_04 was fixed by §1.1, not by the gate.

### 1.3 Where §1 leaves us

`results/euroc_xivo_m4`, n=6, both modes, the config committed in `0622cc5`
(`P.Wsg = 0.002`, `P.bg = 0.01`, `P.ba = 0.25`, 60 m depth window,
`visual_meas_std = 2.4`, `gravity_init_max_accel_dev = 0.1`):

| metric | XIVO | OpenVINS |
| --- | --- | --- |
| ATE pos (m) | 0.138 | **0.094** |
| ov ATE pos (m) | 0.141 | **0.097** |
| ov ATE ori (deg) | 1.77 | 1.77 |
| RPE8 pos (m) | 0.124 | **0.117** |
| RPE8 ori (deg) | **0.89** | 0.90 |

Zero divergences. And the useful reading is the **shape**, not the mean: the
orientation ATE is a dead tie and XIVO wins the 8-frame relative pose error.
XIVO's *local* accuracy already matches OpenVINS. The entire remaining gap is
global, absolute drift.


## 2. One number for the pixel noise cannot work

`visual_meas_std` was where the M4 config had ended up at 2.4 px -- and that
choice came from a five-sequence screen (MH_04, V1_03, V2_01, V1_01, MH_01) which
was **mis-specified**: four of the five were M3 divergence cases, so the screen
could only ever see the upside of loosening the pixel noise and never charged for
the Machine Hall regression it caused. Re-running the bracket on all eleven makes
that immediately visible (stereo, n=3; the best value per row in bold):

| sequence | 0.75 | 1.2 | 1.8 | 2.4 | OpenVINS |
| --- | --- | --- | --- | --- | --- |
| MH_01_easy | **0.094** | 0.103 | 0.107 | 0.126 | 0.073 |
| MH_02_easy | **0.054** | 0.095 | 0.095 | 0.121 | 0.090 |
| MH_03_medium | **0.094** | 0.117 | 0.148 | 0.135 | 0.116 |
| MH_04_difficult | **0.109** | 0.123 | 0.180 | 0.157 | 0.207 |
| MH_05_difficult | **0.113** | 0.159 | 0.200 | 0.267 | 0.190 |
| V1_01_easy | 0.077 | 0.082 | 0.082 | **0.076** | 0.055 |
| V1_02_medium | 0.135 | **0.093** | 0.146 | 0.181 | 0.047 |
| V1_03_difficult | 1.830 | 0.233 | 0.160 | **0.140** | 0.059 |
| V2_01_easy | 0.241 | 0.081 | 0.046 | **0.044** | 0.054 |
| V2_02_medium | 0.252 | 0.075 | **0.072** | 0.086 | 0.049 |
| V2_03_difficult | 0.546 | 0.580 | 0.294 | **0.181** | 0.096 |
| **MEAN** | 0.322 | 0.158 | 0.139 | 0.138 | **0.094** |

The five Machine Hall sequences all want **0.75**. Five of the six Vicon room
sequences want **1.8 or 2.4**. The split is perfect and it is not subtle: at 0.75
V1_03 is 1.830 m, at 2.4 MH_05 is 0.267 m. A per-sequence oracle -- the
minimum of each row, which the brief explicitly forbids -- averages **0.098**,
level with OpenVINS. **One fixed value costs ~40% of the achievable ATE**, and
that is the whole gap.

This is not a tuning accident. Machine Hall is slow, well-lit and sharp; the
Vicon room sequences are fast and visibly motion-blurred. The *true* KLT
tracking noise really does differ by about 3x between them, so no single number
describes both, and picking one is picking which half of the dataset to be wrong
about.

## 3. Why the gate and the weight are the same knob

Before adapting anything, it is worth being precise about what `visual_meas_std`
actually controls, because it turns out to be two things.

`R_` (the square of `visual_meas_std`) is a scalar variance. `MHGating` computes,
per in-state feature,

```
dist = r' (H P H' + R)^-1 r
```

and destroys the feature if `dist > MH_thresh` (after `MH_max_strikes`
consecutive failures). So `R_` sets **both** the Kalman update weight **and the
radius of the Mahalanobis gate**. Loosening it to survive motion blur does not
merely down-weight the blurred measurements, it also *widens the gate*, and the
bracket in §2 cannot tell those two effects apart.

Decoupling them, by holding `visual_meas_std = 0.75` (the Machine Hall optimum)
and opening `MH_thresh` instead (full 11, n=3):

| `MH_thresh` | chi2(2) quantile | mean ATE (m) |
| --- | --- | --- |
| 5.991 | 95% | 0.322 |
| 12 | 99.75% | 1.568 |
| **30** | -- | **0.125** |
| 80 | -- | 0.154 |

(The 12 row is a bimodal draw, not a real regression: V2_03 alone reports
16.3 +- 27.4. Two of three members track and one partially loses the trajectory
without crossing the 100 m divergence threshold. Reporting 1.568 as a mean is
exactly the failure mode M3 §1 warns about; it is quoted here only so the ladder
is not silently pruned.)

`MH_thresh = 30` at `visual_meas_std = 0.75` is **0.125** -- better than any fixed
`visual_meas_std` at the textbook gate, and it beats OpenVINS on MH_05 (0.188 vs
0.190) and V2_01 (0.051 vs 0.054). But the same Machine-Hall-vs-Vicon tension
reappears inside the gate ladder that appeared inside the weight bracket, which is
the informative part: **both knobs are proxies for one latent quantity** -- how
noisy the tracks actually are, right now -- and neither is independently
tunable. That is the argument for measuring it rather than choosing it.

## 4. Estimating the pixel noise online

The filter already computes the statistic needed. `dist` above is the normalised
innovation squared, and for a 2-vector residual it is **chi-square(2) distributed
exactly when the assumed noise matches the real noise**. If the median of the
`dist` values over a frame is above the chi-square(2) median (2 ln 2 =
1.3862943611), the filter is over-confident and `R_` is too small; below it, too
large. The ratio says by how much.

`Estimator::AdaptVisualMeasNoise` (`src/update.cpp`) applies a geometric EMA in
log space:

```
log R  <-  log R + alpha * log(median(dist) / 2 ln 2)
```

clamped to `[min_std^2, max_std^2]`. Design points worth stating:

* **Log space, not linear.** `R_` is a scale parameter; the statistic is a ratio.
  An additive EMA on a scale parameter has a step size whose meaning depends on
  where you are, and can be driven negative.
* **The median, not the mean.** A single badly mistracked feature has unbounded
  `dist`; the median of a chi-square(2) sample is a bounded-influence estimator of
  its scale, which is the entire point on a dataset whose problem *is*
  intermittent mistracking.
* **One frame's residuals never choose the covariance that whitens them.** The
  estimate formed in update *k* is written to `R_pending_` and adopted at the top
  of update *k+1*, before anything reads `R_`. Within a frame the gate, `diagR_`
  and the right-camera variance (`R_ * stereo_update_R_scale_`) therefore all see
  one value. Feeding the estimate back inside the same update would make the
  innovation covariance a function of the innovation, which is not a Kalman
  update any more and self-confirms.
* **The ceiling is load-bearing, the floor is not.** Because the gate radius grows
  with `R_` (§3), an unbounded upward walk admits progressively worse
  measurements and has no fixed point. `max_std` bounds that. `min_std` mostly
  just avoids a degenerate `R_ -> 0`.
* **Attributing the whole discrepancy to `R_` is a choice.** `dist` is inflated by
  an under-estimated `P` just as much as by an under-estimated `R`, and the median
  cannot tell them apart. Charging it all to `R_` is a *consistency correction*,
  not a noise identification -- it makes the filter's own uncertainty statement
  self-consistent, which is what the gate needs to be well-calibrated, and is
  defensible precisely because §2 established that the real pixel noise does vary
  by 3x here.

Off by default (`visual_meas_adapt.enable`, absent from every shipped config), and
`LOG(FATAL)` if enabled without `use_MH_gating` since that is where `dist` comes
from. Verified inert when off: TUM-VI room1 stereo trajectory md5
`500d4fa6b8cd1593a1e33af8121f0cef`, **bit-identical** before and after the change.

Eight unit tests in `src/test/unittest_adapt_meas_noise.cpp`, driving the real
`Estimator` through `#define private public`. They use deterministic chi-square(2)
quantiles -- `-2 log1p(-p)` at `p = (k+0.5)/n` with n = 61 odd, so the middle
order statistic *is* the exact median -- rather than random draws, which makes
every expectation an equality rather than a tolerance. They pin that a consistent
filter is a fixed point, that the estimate converges to the truth monotonically
and without overshoot from either side, that both clamps bind, that warm-up and
the minimum sample count hold it still, that non-finite `dist` values are dropped
before ranking rather than sorted to an end, and that the even-n case averages the
two middle values. The header says explicitly what they do *not* show: that the
loop lands on a *useful* value on real data. That is what §5 is for.

## 5. The time constant is the load-bearing parameter

The first adaptive run used `alpha = 0.05`, chosen as "a 1 s time constant at
EuRoC's 20 Hz" and nothing more. It was a **large win on Machine Hall and a clear
loss in the Vicon room** -- mean 0.144, worse than the 0.125 the wide fixed gate
already reached. The obvious reading is that the loop finds a fixed point that is
not the ATE-optimal one. That reading is wrong, and the census line
(`visual-std:<now> (<min>..<max>)`, printed only when adaptation is on) says why.

At `alpha = 0.05`, final estimate and range over the run:

| sequence | final sigma | range | ATE |
| --- | --- | --- | --- |
| MH_01..MH_05 | 0.60 -- 0.76 | 0.6 .. 1.2 | 0.040 -- 0.106 |
| V1_01_easy | 0.60 | 0.6 .. 1.37 | 0.072 |
| V2_01_easy | 0.60 | 0.6 .. 2.11 | 0.115 |
| V2_03_difficult | 1.73 | 0.66 .. 3.88 | 0.602 |

The Machine Hall sequences settle and stay settled -- they are quasi-static, one
value describes them, and the loop finds it. The Vicon room ones *excurse and fall
back*: V2_01 reaches 2.11 at some point and ends at the floor. The excursions are
the fast segments, and with a 20-update time constant the estimate is still
climbing when the segment ends and still falling when the next one starts. It
never gets where it is going. The problem was never the fixed point; it was the
lag.

Shortening the time constant (full 11, stereo, n=3, `MH_thresh` at the textbook
5.991 throughout):

| `alpha` | time constant | mean ATE (m) |
| --- | --- | --- |
| 0.02 | 50 updates, 2.5 s | 0.163 |
| 0.05 | 20 updates, 1.0 s | 0.144 |
| **0.15** | **6.7 updates, 0.33 s** | **0.095** |
| 0.30 | 3.3 updates, 0.17 s | 0.097 |
| 0.50 | 2 updates, 0.10 s | 0.099 |

A knee between 0.05 and 0.15, then flat out to 0.5. So `alpha` is load-bearing --
a 1.7x swing in the full-11 mean, larger than anything else measured in M4 -- but
only in the sense that it must be *fast enough*; above the knee its exact value is
second-order, which is the good kind of parameter to have. And at 0.15 the
estimate now actually reaches the `max_std = 4` ceiling on V2_01/V2_02/V2_03,
where at 0.05 it never got past 2.1: V2_01 goes **0.115 -> 0.048**, i.e. from
losing to OpenVINS' 0.054 to beating it.

Raising the ceiling to 6.0 at `alpha = 0.15` is inert (0.097 vs 0.095), so 4.0 is
kept.

## 6. The two mechanisms do not compose, which confirms §3

§3 and §5 arrive at parity by different routes:

| config | mean ATE | Machine Hall mean | Vicon room mean |
| --- | --- | --- | --- |
| adapt, `alpha = 0.15`, gate 5.991 | 0.095 | **0.086** | 0.103 |
| adapt, `alpha = 0.05`, gate 12 | 0.093 | 0.102 | **0.086** |
| both: `alpha = 0.15`, gate 12 | 0.099 | 0.105 | 0.094 |
| OpenVINS | 0.094 | 0.135 | 0.060 |

Each mechanism wins the half of the dataset the other loses, so the natural move
is to combine them -- and the combination is **worse than either alone**. It keeps
the wide gate's Vicon room result and gives up the fast adaptation's Machine Hall
result (MH_02 0.069 vs 0.038, MH_04 0.128 vs 0.095, MH_05 0.106 vs 0.088).

That is not a disappointment, it is the confirmation of §3. Both knobs widen the
same effective gate: `alpha` widens it dynamically during the blurred bursts,
`MH_thresh` widens it statically all the time. Applying both over-widens it, and
Machine Hall -- where the tracks are sharp and the gate *should* be tight -- pays
for it. They were never two independent improvements to be stacked; they are two
parameterisations of one correction, and only one of them should be used.

## 7. Does the adaptation earn its complexity? Two controls say yes

An adaptive loop that merely reproduces what a wider gate already achieves would
not be worth ~90 lines and a new config block. So each finalist was run against
its *own* control -- the identical config with `visual_meas_adapt.enable = false`,
leaving `visual_meas_std` pinned at the 1.2 it would have started from:

| gate | adaptation off | adaptation on | gain |
| --- | --- | --- | --- |
| 5.991 (textbook) | 0.158 | **0.095** (`alpha = 0.15`) | **40%** |
| 12 | 0.125 | 0.093 (`alpha = 0.05`) | 26% |

Both are far outside ensemble noise, so the loop is doing real work in both, and
the effect is not a relabelled gate widening.

## 8. The configuration, and why this one

Two candidates reach parity with OpenVINS. Choosing between them on the mean would
be choosing on noise -- 0.093 vs 0.095 is 2 mm on a 95 mm number whose
per-sequence standard deviations run to 0.05 -- so the tiebreak is on properties
that are not noise:

| | `alpha = 0.15`, gate 5.991 | `alpha = 0.05`, gate 12 |
| --- | --- | --- |
| mean ATE | 0.095 | 0.093 |
| metrics won vs OpenVINS | 3 of 5 | 4 of 5 |
| knobs off their default | 1 | 2 |
| `MH_thresh` | textbook chi2(2) 95% | 12, unexplained |
| gain over its own control | **40%** | 26% |
| parameter response nearby | 0.095 / 0.097 / 0.099 at alpha 0.15 / 0.30 / 0.50 | gate 9 / 12 / 20 = 0.286 / 0.093 / 0.108 |

The last row decides it. The alpha ladder is smooth and flat above the knee, so
`alpha = 0.15` sits in a flat interior. The gate ladder is not: `MH_thresh = 9`,
immediately next door, reports **0.286**, and that is entirely V2_03 at
2.320 +- 3.739 -- one member of three partially losing the trajectory. (The other
ten sequences at gate 9 average 0.083, so this is a bimodality, not a broad
regression -- which is exactly what makes it disqualifying rather than merely bad.
The other ten being *good* is what a knife edge looks like.) A configuration whose
neighbour is bimodal is a configuration that got a good draw, and shipping the
better mean here would mean shipping the worse config.

The textbook gate also has an argument behind it that the number 12 does not: §3's
whole point is that `R_` sets the gate radius, so an estimate that tracks the real
noise should keep a correctly calibrated gate at its nominal quantile. `alpha =
0.15` at 5.991 is that claim coming out true; `alpha = 0.05` at 12 is that claim
being patched with a second tuned constant.

Final configuration, generated by `scripts/make_euroc_cfg.py` and committed as
`cfg/euroc_stereo.json` / `cfg/euroc_mono.json`:

| key | value | from |
| --- | --- | --- |
| `P.bg` | 0.01 | M2 -- gyro bias prior, sigma 0.1 rad/s |
| `P.ba` | 0.25 | M2 -- accel bias prior, sigma 0.5 m/s^2 |
| `P.Wsg` | 0.002 | §1.1 -- sigma 0.045 rad |
| `min_depth` / `max_depth` | 0.05 / 60.0 | M2 |
| `initial_z` | 5.0 | M2 |
| `gravity` | 9.81 | matches OpenVINS' EuRoC config |
| `gravity_init_max_accel_dev` | 0.1 | §1.2 |
| `visual_meas_std` | 1.2 | §5 -- only the starting point now |
| `visual_meas_adapt.enable` | true | §4 |
| `visual_meas_adapt.alpha` | **0.15** | §5 |
| `visual_meas_adapt.min_std` / `max_std` | 0.6 / 4.0 | §5 |
| `MH_thresh` | 5.991 | §8 -- the textbook value, unchanged |
| `MH_max_strikes` | 1 | unchanged from HEAD |

Everything else -- calibration, `Qimu`, extrinsics, the stereo baseline -- comes
from the dataset's own `sensor.yaml` files, not from tuning. One configuration,
all eleven sequences, both modes.

## 9. Confirmation at n=6, both modes

`results/euroc_xivo_m4b`, 132 runs: 11 sequences x 2 modes x 6 jitter members,
using the generated `cfg/euroc_*.json` rather than a sweep-patched copy, so this
is the configuration as it will actually ship.

### 9.1 Stereo -- **0 of 66 diverged**

| sequence | XIVO | OpenVINS |
| --- | --- | --- |
| MH_01_easy | 0.082+-0.008 | **0.073+-0.000** |
| MH_02_easy | **0.039+-0.010** | 0.090+-0.011 |
| MH_03_medium | **0.113+-0.034** | 0.116+-0.003 |
| MH_04_difficult | **0.099+-0.009** | 0.207+-0.000 |
| MH_05_difficult | **0.089+-0.006** | 0.190+-0.000 |
| V1_01_easy | 0.078+-0.004 | **0.055+-0.003** |
| V1_02_medium | 0.070+-0.004 | **0.047+-0.002** |
| V1_03_difficult | 0.169+-0.016 | **0.059+-0.002** |
| V2_01_easy | **0.049+-0.002** | 0.054+-0.004 |
| V2_02_medium | 0.090+-0.003 | **0.049+-0.002** |
| V2_03_difficult | 0.169+-0.013 | **0.096+-0.010** |
| **MEAN** | **0.095** | 0.094 |

n=6 reproduces n=3 to the third decimal on every aggregate, which is the point of
running it. Across all five metrics:

| metric | XIVO | OpenVINS | |
| --- | --- | --- | --- |
| ATE pos (m) | 0.095 | 0.094 | tie (0.001, inside noise) |
| ov ATE pos (m) | 0.103 | **0.097** | OpenVINS |
| ov ATE ori (deg) | **1.72** | 1.77 | XIVO |
| RPE8 pos (m) | **0.111** | 0.117 | XIVO |
| RPE8 ori (deg) | **0.85** | 0.90 | XIVO |

**XIVO wins three of five, ties the headline metric, and loses one.** The split
is the same one M3 and §1.3 found and it has not moved: XIVO is better locally
(both relative-pose metrics, both orientation metrics) and worse in absolute
position. What *has* moved is the magnitude -- ATE pos 0.138 -> 0.095, a 31%
improvement over §1.3, entirely from §4/§5.

Per sequence it is cleanly split by *scene*, not by difficulty: XIVO wins four of
five Machine Hall sequences, three of them by 2x or more (MH_02 0.039 vs 0.090,
MH_04 0.099 vs 0.207, MH_05 0.089 vs 0.190), and loses five of six Vicon room
ones. Machine Hall mean 0.084 vs OpenVINS' 0.135; Vicon room 0.104 vs 0.060. The
adaptation turned Machine Hall from a loss into a rout and left the Vicon room a
loss -- so the remaining gap is one specific thing, in one specific place.

### 9.2 Mono -- **0 of 66 diverged**

Not the deliverable (the brief specifies stereo + IMU), so this is measured and
reported but not tuned. Reported because the M3 baseline had **6/6 divergences on
MH_01 and 4/6 on MH_02**, and that is now zero.

| sequence | XIVO mono | OpenVINS mono |
| --- | --- | --- |
| MH_01_easy | 1.357+-0.548 | **0.132+-0.008** |
| MH_02_easy | **0.102+-0.020** | 0.124+-0.004 |
| MH_03_medium | 0.180+-0.021 | **0.141+-0.025** |
| MH_04_difficult | 0.258+-0.041 | **0.197+-0.000** |
| MH_05_difficult | **0.284+-0.030** | 0.426+-0.011 |
| V1_01_easy | 0.112+-0.008 | **0.058+-0.003** |
| V1_02_medium | 0.155+-0.020 | **0.067+-0.000** |
| V1_03_difficult | 0.284+-0.030 | **0.068+-0.003** |
| V2_01_easy | **0.079+-0.005** | 0.124+-0.014 |
| V2_02_medium | 0.155+-0.009 | **0.062+-0.001** |
| V2_03_difficult | 0.283+-0.028 | **0.195+-0.000** |
| **MEAN** | 0.295 | **0.145** |

MH_01 at 1.357 +- 0.548 is the one bad cell, and an sd 40% of the mean says it is
bimodal rather than uniformly poor -- the same signature §1.2 diagnosed. Without
stereo there is no metric depth at initialisation, so MH_01's already-moving start
has to be absorbed by `initial_z` and the subfilter, and sometimes is not. Tuning
mono is out of scope; the honest statement is that XIVO mono is 2x worse than
OpenVINS mono overall while winning three sequences.

### 9.3 Throughput during this pass -- **not** the efficiency result

Many sequences ran concurrently, so these only show nothing pathological:

| mode | FPS (wall) | peak RSS (MB) |
| --- | --- | --- |
| stereo | 27.9 | 130.8 |
| mono | 37.0 | 120.1 |

The real comparison -- one core, `taskset -c 0`, pools pinned to 1, serial, idle
box, n>=3 -- is M5/M6.

## 10. What was tried and did not work

Negative results, kept because the next person to look at the Vicon room gap will
otherwise try them again. All on top of the §8 configuration, full 11, n=3, against
its 0.095.

| change | mean ATE | reading |
| --- | --- | --- |
| `MH_max_strikes = 2` | 0.102 | worse |
| `MH_max_strikes = 3` | 0.104 | worse (V2_03 0.169 -> 0.258) |
| `max_group_lifetime = 30` | 0.095 | **bit-identical**, i.e. never binds |
| `stereo_update.R_scale = 2.0` | 0.109 | worse |
| `visual_meas_adapt.max_std = 6.0` | 0.097 | inert |
| `visual_meas_adapt.min_std = 0.3` | 0.149 (at alpha 0.05) | worse |
| `alpha = 0.15` + `MH_thresh = 12` | 0.099 | worse -- see §6 |

`MH_max_strikes` is worth singling out because it tests a specific hypothesis and
refutes it. The concern was survivor bias: a feature whose Mahalanobis distance is
large gets destroyed, so the population whose median §4 takes is biased toward
agreement, which would drive the estimate down exactly where a blur burst should
drive it up. Letting a feature sit out an update instead of dying (`strikes > 1`)
would break that. It makes things worse, monotonically in the number of strikes,
and worst on V2_03 -- the fastest sequence. So on this dataset a feature that fails
the gate once is genuinely gone rather than momentarily occluded, and keeping it in
the state costs more than the biased median does. (The bias is also smaller than it
looks: `dist` is computed for *every* in-state feature before the gate is applied,
so within a frame nothing is missing. Only the cross-frame population is filtered.)

`max_group_lifetime` producing a byte-identical trajectory is worth recording as a
fact about the system rather than a tuning result: groups are retired by other
mechanisms long before 60 frames, so the knob is dead code on EuRoC.

### The gap that remains

Five of six Vicon room sequences still lose, V1_03 (0.169 vs 0.059) and V2_03
(0.169 vs 0.096) worst. §2 shows this is not a `visual_meas_std` problem any more
-- the per-sequence oracle for V1_03 was 0.140, and the adaptive config is at
0.169, so even a per-sequence fixed value would not close it. Combined with §9.1's
finding that the relative-pose metrics are *won* while absolute position is lost,
the remaining gap looks like slow global drift rather than local tracking error,
which points at the group/keyframe geometry rather than at the measurement model
-- OpenVINS carries 11 clones plus 50 SLAM features with FEJ, and XIVO's
sliding-window structure is different in kind. That is a design difference, not a
parameter, and out of scope for M4.

## 11. Reproduce

```bash
cd experiments/openvins

# The shipped configuration, regenerated from the dataset's own sensor.yaml:
cd ../../xivo-eurocacc
export PATH="$PWD/../dependencies/venv/bin:$PATH"
for m in stereo mono; do extra=""; [ $m = mono ] && extra="--mono"
  python3 scripts/make_euroc_cfg.py --base cfg/eff_$m.json \
    --seqdir ../data/euroc/MH_01_easy --out cfg/euroc_$m.json $extra
done
cd ../experiments/openvins

# The n=6 confirmation of §9 (132 runs):
CPU_BASE=0 CPU_SPAN=60 ./run_xivo_reference.sh --profile euroc_mav \
  --mode both --jitter 6 --worktree xivo-eurocacc --cfg-prefix euroc \
  --out ../results/euroc_xivo_m4b

python3 agg_ensemble.py --mode stereo \
  --arm xivo ../results/euroc_xivo_m4b \
  --arm openvins ../results/euroc_ov_acc_dyn/stereo_m*
python3 agg_ensemble.py --mode mono \
  --arm xivo ../results/euroc_xivo_m4b \
  --arm openvins ../results/euroc_ov_acc_dyn/mono_m*

# Any single tuning arm (writes cfg/tune_<name>_*.json, which .gitignore covers):
SEQS="MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult \
V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult" \
WORKTREE=xivo-eurocacc MEMBERS=3 \
  ./sweep_batch.sh 'h_a30 visual_meas_adapt.alpha=0.30'
```

`--mode stereo` is not optional in the aggregation: the XIVO runner writes mono and
stereo rows into one `summary.csv`, and averaging across sensor modes produces a
number that means nothing.

Unit tests for the new code:

```bash
cd xivo-eurocacc/build && make -j20 unitTests_adapt_meas_noise unitTests_gravity_init
cd .. && ./bin/unitTests_adapt_meas_noise && ./bin/unitTests_gravity_init
```

**Every full-11 sweep in this milestone is quotable; the earlier five-sequence
screens are not** -- see §2 for why. Result directories under
`results/euroc_tune/` prefixed `f_`, `g_`, `h_`, `i_` and `j_` are full-11; those
prefixed `a_`, `b_`, `c_` and `w_` are the mis-specified screen and are kept only
for provenance.
