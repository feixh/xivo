# XIVO against OpenVINS on EuRoC MAV: stereo + IMU, one configuration, eleven sequences

The brief for this round: run the same head-to-head we ran on TUM-VI on the
**EuRoC MAV** dataset, in **stereo + IMU**, with **one XIVO configuration shared
by all eleven sequences** rather than one per sequence, benchmark **both accuracy
and runtime efficiency**, and tune XIVO to match or outperform
[OpenVINS](https://github.com/rpng/open_vins). Feature work was isolated on
branches in separate `git worktree`s, committed per milestone, and merged back
into `auto` at the end.

Reference: OpenVINS v2.7 (`v2.7-20-g6948812`), built ROS-free and run **on this
machine** rather than quoted from a paper, so both systems see the same images,
the same ground truth, one evaluation code path and one core.

Three branches, seven milestones, all merged: `auto-euroc` (dataset support and
the two baselines), `auto-eurocacc` (accuracy), `auto-eurocfps` (throughput and
the final evaluation). `ctest` is **23/23** on the merged tree.


## Headline

Eleven sequences, stereo + IMU, one configuration each. XIVO n=10 (110 runs per
arm), OpenVINS n=6 (66 runs). `±` is the standard error of the eleven-sequence
mean. Bold is the best of the three.

| metric | XIVO `acc` | **XIVO `fast`** (shipped) | OpenVINS |
| --- | --- | --- | --- |
| `ate_002` [m] | 0.0950 ± 0.0009 | 0.1028 ± 0.0016 | **0.0941** ± 0.0006 |
| ATE position, `posyaw` [m] | 0.1035 ± 0.0009 | 0.1102 ± 0.0016 | **0.0972** ± 0.0006 |
| ATE orientation [deg] | 1.709 ± 0.009 | **1.706** ± 0.010 | 1.773 ± 0.010 |
| RPE 8 m, position [m] | **0.1093** ± 0.0003 | 0.1109 ± 0.0005 | 0.1168 ± 0.0007 |
| RPE 8 m, orientation [deg] | **0.852** ± 0.002 | 0.867 ± 0.003 | 0.902 ± 0.005 |

**XIVO wins three of the five accuracy metrics** — both orientation metrics and
the position drift rate — each by 4.6–9.9 combined standard errors. It **ties
`ate_002`** at the accurate operating point (0.8 σ) and **loses absolute ATE
position**. That combination is coherent rather than confusing: XIVO drifts less
per 8 m travelled and holds attitude better, and accumulates more absolute
position error over a whole sequence.

Efficiency, one core (`taskset`, `setarch -R`, every thread pool at 1, sequences
serial, idle box, `-mode runOnly`), whole-process wall clock including PNG
decode, peak RSS from `/usr/bin/time`:

| | ms/frame | FPS | peak RSS | vs OpenVINS |
| --- | --- | --- | --- | --- |
| XIVO, M4 accuracy config | 14.756 | 67.8 | 95.3 MB | +37.4% |
| XIVO `acc` | 13.921 | 71.8 | 96.2 MB | +29.7% |
| **XIVO `fast`** (shipped) | **11.593** | **86.3** | **97.1 MB** | **+8.0%** |
| OpenVINS | 10.737 | 93.1 | 99.2 MB | — |

At the shipped operating point XIVO is **8.0% slower end-to-end** and uses
**3.0 MB less** peak RSS, at 4.3× real time on one core (EuRoC is 20 Hz).

**The most substantive result is not in either table.** The eleven-sequence
accuracy near-tie is two large opposite effects cancelling:

| `ate_002` | XIVO `acc` | XIVO `fast` | OpenVINS |
| --- | --- | --- | --- |
| Machine Hall (5 sequences) | **0.0848** | 0.0883 | 0.1351 |
| Vicon Room (6 sequences) | 0.1036 | 0.1148 | **0.0599** |
| all 11 | 0.0950 | 0.1028 | 0.0941 |

**XIVO is 37% better on Machine Hall; OpenVINS is 42% better on Vicon Room.**
Reporting only the eleven-sequence mean would be true and useless. Where that
gap comes from, and the RPE-8 diagnostic that narrows it, is
[§5.3](#53-machine-hall-vs-vicon-room-and-what-rpe-8-says-about-it).

Zero divergences: 0 of 110 `acc`, 0 of 110 `fast`, 0 of 66 OpenVINS, stereo. The
XIVO baseline this round started from diverged on **15 of 66**.


## 1. Measurement protocol

Settled before any tuning and not revisited, except for the one revision in §1.4.

### 1.1 Metrics

Five per run, all lower-is-better, both systems scored by the same code:

* **`ate_002`** — `evaluate_ate.py` with `--max_difference 0.02`, i.e. a 20 ms
  association window. Aligns with a full similarity transform and is therefore
  **blind to a global rotation**: a constant tilt costs nothing.
* **ATE position / ATE orientation** — `ov_eval error_singlerun posyaw`. Aligns
  position and yaw only, so roll and pitch error is charged in full. About
  0.31 deg of every orientation number is the benchmark's own floor (measured in
  the TUM-VI round).
* **RPE 8 m, position / orientation** — the same tool's 8 m relative pose error.
  This is the local drift-rate metric, and it is insensitive to a single early
  excursion in a way ATE is not. That asymmetry turns out to carry the main
  diagnostic of the whole comparison (§5.3).

Both ATE metrics are reported because they disagree in a specific, informative
way: `ov_ate_pos` ≥ `ate_002` throughout, which is the visible signature of the
similarity alignment absorbing a global rotation.

The 1 ms association window used elsewhere in this repo is **never** used on
OpenVINS output: its poses are stamped at camera time plus the *online-estimated*
camera-IMU offset, so a 1 ms window associates a phase-dependent 3–1138 of ~2700
poses.

### 1.2 Ensembles, because single runs are noise

XIVO's per-run output is deterministic, but its gating decisions are chaotic: a
velocity perturbation of k·1e-6 m/s — six orders of magnitude inside the filter's
own prior — changes the ATE by ±0.007 m on TUM-VI and by up to ±0.040 m on
EuRoC's `V2_03_difficult`. Every XIVO number in this report is a **jitter
ensemble** mean over members that perturb `X.Vsb` by k·1e-6 m/s.

OpenVINS is deterministic in the stronger sense that rerunning reproduces the
trajectory bit for bit, so `--repeats` measures nothing. Its ensemble perturbs
`--gravity_mag` in the **ninth significant digit** (9.81 → 9.810000005). Member 0
is the unperturbed shipped value, so the canonical run is inside the ensemble
rather than replaced by it. OpenVINS is genuinely less sensitive to this than
XIVO is to velocity jitter — several sequences report ±0.000 — which is a real
property of the estimator on this data, and the reason n=6 is enough for it while
XIVO needs n=10.

### 1.3 Throughput and memory

`run_xivo_reference.sh --timing` / `run_openvins.sh --onecore`, identical
protocol for both systems:

* `-mode runOnly` — no visualisation, no trajectory scoring, no per-frame dump.
* `taskset -c $CPU_BASE setarch -R` — one core, ASLR off.
* Every thread pool pinned to 1 (`cv::setNumThreads(0)`, OpenVINS'
  `num_opencv_threads=1`, XIVO's `ekf_update.chunks` left serial).
* Sequences run **serially**, never concurrently, because two pinned processes
  still contend for L3 and memory bandwidth.
* `ms/frame` = total wall / total frames, frame-weighted over all eleven.

Reproducibility is **sd 0.002 ms on 14.9 ms** (0.013%), which is what makes a
0.2 ms knob measurable at all.

Three traps, each of which produced a wrong number before it was understood:

* **Never quote FPS or RSS from an accuracy pass.** The accuracy pass launches
  every run at once — 220 processes for an n=10 × 11-sequence × 2-mode sweep —
  because `CPU_SPAN` chooses which cpu each run is *pinned* to and does not cap
  concurrency. The same configuration read 27.9 FPS in one such pass and 65.7 in
  another. ATE is unaffected by load (proved by bit-identical members across
  passes run at very different load), so the accuracy pass is still valid for
  accuracy.
* **Peak RSS is mode-dependent, and only `runOnly` is the deployment figure.**
  Under `-mode eval`, which writes a per-frame dump for scoring, XIVO's peak RSS
  is 131.3 MB against OpenVINS' 105.1 MB. That +34 MB is the dump buffer;
  OpenVINS' equivalent overhead is +6 MB. Quoting the eval-mode number would
  report a 25% memory *deficit* where the deployment configuration has a 2%
  advantage.
* **`fps_mean` is not the reciprocal of the mean frame time** (it is a mean of
  per-frame rates), and it excludes image decode, which is 22% of XIVO's frame.
  `fps_wall` is the cross-system number.

### 1.4 The one revision: screen at n=3, ship at n=10

M5's accuracy screens were n=3 ensembles. That is not enough on EuRoC. Re-running
*the identical shipped configuration* at n=6 and then n=10 — with members 0–2
bit-identical to M5's, verified value by value — moved the eleven-sequence
`ate_002` mean **0.098 → 0.102 → 0.103**:

| | members | `ate_002`, all 11 |
| --- | --- | --- |
| `results/euroc_fps_ship` | 0,1,2 | 0.098 |
| `results/euroc_m6_xivo` | 0–5 | 0.102 |
| `results/euroc_m6_final/fast` | 0–9 | **0.103** |

Same configuration, same runs; members 3–5 simply happened to be worse, mostly on
`V2_03_difficult` (0.228 / 0.316 / 0.213 against the first three members'
0.187 / 0.202 / 0.168). Nothing about the estimator changed, only the size of the
sample.

Consequences, all of them applied: every accuracy number in M5's notes was
corrected in place; the shipped configuration's honest accuracy is 0.103 rather
than 0.098; the front-end substitution of §4.4 costs **+0.008 m**, not +0.003, so
its exchange rate is 0.31 ms per 0.001 m rather than 1.01, and the claim that it
beat every other equalization arm "by a factor of two and a half" is withdrawn.
What survives untouched is everything whose margin is an order of magnitude
outside this noise — the two disqualifications in §9, the `B0` composition hazard,
and every timing and keypoint-count measurement (reproducibility 0.013%).

**A caveat these error bars do not cover.** They are the reproducibility of *one
configuration's* ensemble mean. They are not the uncertainty in "XIVO's accuracy"
as a design: perturbing a neutral configuration knob moves the mean roughly six
times as far as reseeding does. Treat any two-*configuration* difference under
~0.005 m as unresolved, even when both sides carry ±0.001 error bars.


## 2. Fairness: what OpenVINS was given

A baseline that is quietly handicapped makes the whole comparison worthless, and
the handicap here was not hypothetical.

With its own shipped `euroc_mav` configuration, unmodified, **OpenVINS diverges on
`MH_04_difficult` in 6 of 6 members** (ATE ~9349 m) while tracking the other ten
sequences to 0.05–0.36 m. It would have been easy to report that as "OpenVINS
fails on MH_04" and move on.

Reading `ov_init/src/init/InertialInitializer.cpp:79-159` explains it, and the
mechanism is a genuine blind spot rather than a bug: the initializer chooses the
static path when measured feature disparity over the init window is below
`init_max_disparity` (10 px shipped). MH_04 opens with a take-off — ground truth
`|v|` reaches 0.47 m/s — but the Machine Hall scene is tens of metres deep, and
disparity is translation over depth, so 0.47 m/s produces under 10 px. The static
initializer then asserts **zero velocity** on a platform moving at 0.47 m/s. A
disparity threshold in pixels is being used as a proxy for a velocity threshold
in m/s, and the conversion factor is scene depth, which the initializer does not
know yet.

The fix given to OpenVINS is `--init_dyn_use 1`: its **own** dynamic initializer,
already in the codebase and simply off by default, applied **uniformly to all
eleven sequences and all six members** so the one-configuration property holds.
The evidence that this is a fix and not a tuning knob is that all five Machine
Hall sequences improve and **all six Vicon Room sequences are bit-identical** —
they begin genuinely stationary, so the static path is still selected and the
dynamic code never runs:

| sequence | shipped | `init_dyn_use 1` |
| --- | --- | --- |
| MH_01_easy | 0.113 ± 0.014 | **0.073** ± 0.000 |
| MH_02_easy | 0.122 ± 0.003 | **0.090** ± 0.011 |
| MH_03_medium | 0.121 ± 0.003 | **0.116** ± 0.003 |
| MH_04_difficult | **diverged 6/6** | **0.207** ± 0.000 |
| MH_05_difficult | 0.358 ± 0.000 | **0.190** ± 0.000 |
| V1_01…V2_03 (6 seqs) | 0.055 / 0.047 / 0.059 / 0.054 / 0.049 / 0.096 | *bit-identical* |

The obvious alternative was rejected on measurement: `--init_max_disparity 3.0`
fixes MH_04 *better* (0.161 against the dynamic initializer's 0.207) and breaks
two sequences that previously worked (MH_02 and V2_03 both diverge, MH_05 goes
0.358 → 0.574). Trading one divergence for two is not progress — and it is why
the screen has to include sequences that already work. Had only MH_04 been
measured, `disp3` would have looked like the better fix by 0.046 m.

Two further fairness points:

* OpenVINS needs **no per-sequence override** on EuRoC (unlike TUM-VI, where
  room6 needs its own `init_imu_thresh`), so both systems really are running one
  configuration for all eleven.
* Nothing is per-sequence on the XIVO side at any point in this round, including
  during tuning: every screen and every reported arm is all eleven sequences with
  one config, or an explicitly-labelled three-sequence timing screen whose
  numbers are never quoted as results.


## 3. The dataset, and why one configuration is honest here

EuRoC is not "TUM-VI with different files":

| | TUM-VI room1–6 | EuRoC MAV |
| --- | --- | --- |
| camera model | equidistant fisheye, 512×512 | **pinhole + radtan**, 752×480 |
| sequences | 6, all in the mocap room | **11**, three environments (MH, V1, V2) |
| ground truth | `mav0/mocap0/data.csv`, room-only | `state_groundtruth_estimate0`, whole sequence |
| motion | handheld, slow-ish | **MAV flight**, incl. fast/dark V1_03, V2_03 |
| IMU / stereo rate | 200 / 20 Hz | 200 / 20 Hz |

Two consequences shape everything below. First, EuRoC's ground truth covers the
whole trajectory rather than a mocap volume, so unlike TUM-VI outside room1–6
every pose is scorable and ATE is meaningful on all eleven. Second, the difficulty
spread across eleven sequences in three environments is far wider than across
room1–6, so "one shared configuration" is a genuine constraint rather than a
formality — §4.3 shows it costs ~40% of the achievable ATE if handled naively.

**One configuration is what the dataset itself says.**
`mav0/{cam0,cam1,imu0}/sensor.yaml` is **byte-identical across all eleven
sequences** (md5 `84411ac5` / `dec090ef` / `ec43620a`, 11 of 11 each), and those
values match OpenVINS' shipped `kalibr_imucam_chain.yaml` exactly. So the shared
configuration is not a compromise imposed for fairness.

Data notes worth keeping: three sequences have unequal left/right image counts,
and `V2_03_difficult` is **missing 414 left images** — a known property of the
release, not a bad download. Both systems pair on timestamp and drop unmatched
frames, so this costs V2_03 ~18% of its stereo frames on both sides equally. The
canonical host `robotics.ethz.ch` is unreachable from this box (TCP connect hangs
on 80 and 443 while every other host answers), so the data came from the
HuggingFace mirror `GlowBond/EuRoC_MAV_Dataset`.


## 4. What was done, milestone by milestone

### 4.1 M2 — making EuRoC run at all: one config number, four orders of magnitude

The plumbing was small: a `euroc` dataset branch in `scripts/pyxivo.py` (20
lines), a per-dataset ground-truth path in `scripts/savers.py` — which hardcoded
TUM-VI's `mav0/mocap0/data.csv` — and `scripts/make_euroc_cfg.py`, which reads
each sequence's own `sensor.yaml` and emits the shared mono and stereo configs.

Four conventions were checked against source rather than assumed, because each
has a plausible wrong reading that fails silently: `X.Wbc` is `R_body_from_camera`
(so `X.Wbc`/`X.Tbc` *are* EuRoC's `T_BS`, no inversion — inverting would put the
camera 6.5 cm off in the wrong direction); `stereo_cfg.T_c1c0` maps cam0 into
cam1; `Qimu` holds standard deviations, not variances (`estimator.cpp` squares the
block); and XIVO's `RadTan` expansion is algebraically identical to OpenCV's, so
Kalibr's `[k1,k2,p1,p2]` maps across term-for-term with `k3 = 0`. The generator
also *asserts* that `imu0`'s own `T_BS` is identity — EuRoC defines the body frame
as imu0, which is simultaneously the frame every extrinsic is relative to, the
frame the ground truth reports, and the frame XIVO's `gsb` estimates, so a future
release moving it would silently invalidate every generated number.

One latent bug was fixed on the way: the ground-truth quaternion was read as
`v[4:]`, which happens to yield 4 elements on TUM-VI's 8-column `mocap0` rows and
yields 13 on EuRoC's 17-column rows. Only `q[0..3]` is read downstream, so the old
code was right by accident; it is now `v[4:8]`.

**Then XIVO diverged on every EuRoC sequence by four orders of magnitude** — ATE
22593 m on a 58 m trajectory. Four plausible causes were A/B'd and refuted
(`fast_png_decode`, `use_prediction`, the depth window alone, the stereo update).
The step that made progress was establishing a healthy control on the same binary
in the same session — TUM-VI room1 stereo, ATE 0.0379, 84.6 of 90 feature slots
filled — which localised the fault to EuRoC-specific inputs. Two measurements then
localised it to the config: position error grew with a *growing second derivative*
(a constant acceleration bias being integrated twice), while the initial attitude
was correct to three decimals (`R_xivo @ a_body = [-0.004, 0.016, 9.810]`), so
gravity alignment at init was right and attitude *drifted away* afterwards.

The cause is one number. EuRoC's ground truth reports the IMU biases it solved
for: the gyro bias is **0.079–0.085 rad/s on all eleven sequences** — the fixed
turn-on bias of the rig's ADIS16448, a property of the hardware — while the ported
TUM-VI config's `P.bg = 1e-4` is a prior sigma of 0.010 rad/s. The filter is told
with high confidence that a bias it definitely has is impossible, so attitude
drifts at 4.6 deg/s, a tilted attitude leaks gravity into the acceleration
estimate, position runs away quadratically, and no feature ever survives long
enough to be promoted (0.78 of 90 slots filled). Vision never gets the chance to
fix the attitude that broke vision.

| config | ATE [m] | feature slots (of 90) | updates |
| --- | --- | --- | --- |
| ported TUM-VI as-is | 22593 | 0.78 | 546 |
| `P.ba = 0.25` only | 11528 | — | — |
| **`P.bg = 0.01` only** | **0.0712** | 86.3 | 2899 |
| `P.bg = 0.01`, `P.ba = 0.25` | 0.0688 | 86.3 | 2899 |
| … `+ max_depth = 30` | 0.0635 | 86.2 | 2899 |

`P.bg` is the whole fix: 22593 m → 0.0712 m, one number, and consistent subfilter
initialisations go from 835 to 11191. The generalisable form of this: **the
carefully tuned TUM-VI config encodes an assumption about the IMU hardware**, and
EuRoC uses different hardware.

The second EuRoC-specific number is the depth window. TUM-VI's `max_depth = 5.0`
is a room; Machine Hall is tens of metres deep, and a feature outside
`[min_depth, max_depth]` is *refused* as an instate candidate and as a stereo
seed, not merely down-weighted. `max_depth = 60` is the shared value: Machine Hall
does not survive below ~60 m, the Vicon room is flat from 60 up, and 200 m is
*worse* on MH because a badly-conditioned depth estimate still consumes a state
slot.

### 4.2 M3 — the baseline: 15 of 66 stereo runs diverge, and why that is the useful number

With a correct loader, correct calibration and nothing tuned for accuracy, XIVO
already **won three sequences outright** (MH_02 0.057 vs 0.090, MH_03 0.105 vs
0.116, MH_05 0.090 vs 0.190) and diverged on four. The divergences were the
signal, and specifically their *intermittency*: V1_03 failed 5 of 6 members, V2_01
3 of 6, MH_01 1 of 6. A sequence that fails half the time under a 1e-6 m/s
velocity perturbation is not failing because it is hard; it is sitting on a knife
edge, and something is amplifying a perturbation that should be irrelevant.

Two sequences with **zero** divergences were equally informative: `V1_02_medium`
at 9.24 ± 14.14 m and `V2_02_medium` at 1.52 ± 2.11 m. An sd larger than the mean
is not a measurement of accuracy, it is a bimodal distribution — some members
track and some partially lose the trajectory without crossing the 100 m
divergence threshold. Reporting the mean alone there would have been actively
misleading, which is why an sd travels with every number in this project.

Diagnosis, from dumping the estimated poses: MH_04's reported attitude jumps
**40 degrees between poses 1 and 3** while the same dump on healthy sequences
stays flat — an *initialisation* failure, broken before vision can contribute. The
obvious hypothesis, that XIVO's gravity initialiser shares OpenVINS' blind spot,
was **refuted by measurement**: over MH_04's actual 0.1 s init window the mean
specific force is 9.760 m/s² against 9.810, with max |ω| of 0.11 rad/s. MH_04 is
genuinely still when XIVO initialises; the take-off that defeated OpenVINS starts
1.5 s later, after XIVO's window has closed. Checking the same window on all
eleven sequences did turn up a real defect, just on a different sequence —
`MH_01_easy`, off by **−1.463 m/s²**, 30× the next worst.

### 4.3 M4 — accuracy: two divergence mechanisms, then the one that needed code

**`P.Wsg` — a 1.73 rad prior on which way is down.** The shipped TUM-VI prior
variance on the 2-DoF gravity direction state is 3.01, i.e. σ = 1.73 rad. With a
prior that loose the filter absorbs an early vision residual by rotating its
estimate of gravity instead of correcting the pose, and a wrong gravity direction
feeds straight back into the acceleration estimate — positive feedback, which is
the shape required to turn 1e-6 m/s into 71 m. Bracketing it:

| `P.Wsg` | prior σ | mean ATE [m] |
| --- | --- | --- |
| 1e-6 | 0.001 rad | 0.109 |
| 1e-4 | 0.01 rad | 0.116 |
| **2e-3** | **0.045 rad** | **0.105** |
| 1e-2 | 0.1 rad | 0.112 |
| 3.01 (shipped) | 1.73 rad | diverges intermittently |

The response is flat across four orders of magnitude below the shipped value, so
the load-bearing claim is "not 3.01", not "0.002 specifically"; 0.002 is the flat
region's interior rather than its edge. **This alone removes every divergence:
0/66 stereo, down from 15/66.**

**A stationarity gate on gravity init.** `gravity_init_max_accel_dev` rejects a
sample whose `||a| − |g||` exceeds a threshold in m/s², with 0 disabling the gate
so every existing config is unchanged. At 0.1 it moves MH_01 from 0.145 to 0.118
and leaves the other ten identical to four decimals — exactly the signature a
correct gate should have. Four unit tests.

**Then the hard half of the brief.** `visual_meas_std` had been set to 2.4 px by a
five-sequence screen that was **mis-specified**: four of the five were M3
divergence cases, so the screen could only see the upside of loosening the pixel
noise and never charged for the Machine Hall regression it caused. Re-running the
bracket on all eleven makes the problem structural rather than suboptimal:

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
| **mean** | 0.322 | 0.158 | 0.139 | 0.138 | **0.094** |

All five Machine Hall sequences want **0.75**; five of six Vicon Room sequences
want **1.8 or 2.4**. The split is perfect and not subtle — at 0.75 V1_03 is
1.830 m, at 2.4 MH_05 is 0.267 m. A per-sequence oracle (the row minimum, which
the brief forbids) averages **0.098**, level with OpenVINS. **One fixed value
costs ~40% of the achievable ATE, and that was the entire gap.** This is not a
tuning accident: Machine Hall is slow, well-lit and sharp, the Vicon rooms are
fast and visibly motion-blurred, so the true KLT tracking noise really does differ
by ~3× and picking one number is picking which half of the dataset to be wrong
about.

**`R_` is two knobs wearing one name.** `MHGating` computes
`dist = rᵀ (H P Hᵀ + R)⁻¹ r` and destroys the feature if `dist > MH_thresh`, so
`R_` sets **both** the Kalman update weight **and** the radius of the Mahalanobis
gate. Loosening it to survive motion blur also *widens the gate*, and the bracket
above cannot tell those apart. Holding `visual_meas_std` at the Machine Hall
optimum and opening the gate instead reaches 0.125 — better than any fixed
`visual_meas_std` at the textbook gate — but reproduces the same MH-vs-Vicon
tension *inside* the gate ladder. Both knobs are proxies for one latent quantity:
how noisy the tracks actually are, right now. That is the argument for measuring
it rather than choosing it.

**The code: `Estimator::AdaptVisualMeasNoise`.** `dist` is the normalised
innovation squared, and for a 2-vector residual it is χ²(2)-distributed *exactly
when the assumed noise matches the real noise*. So compare the per-frame median of
`dist` against the χ²(2) median (2 ln 2) and apply a geometric EMA in log space:

```
log R  ←  log R + α · log( median(dist) / 2 ln 2 )
```

clamped to `[min_std², max_std²]`. Design points, each chosen against a specific
failure: **log space** because `R_` is a scale parameter and the statistic is a
ratio (an additive EMA on a scale parameter has a step whose meaning depends on
where you are, and can go negative); the **median** because one badly mistracked
feature has unbounded `dist` and the median is a bounded-influence estimator of
scale, which is the whole point on a dataset whose problem *is* intermittent
mistracking; and **one frame's residuals never choose the covariance that whitens
them** — the estimate formed in update *k* goes to `R_pending_` and is adopted at
the top of update *k+1*, because feeding it back inside the same update would make
the innovation covariance a function of the innovation, which is not a Kalman
update any more and self-confirms. The ceiling is load-bearing (since the gate
radius grows with `R_`, an unbounded upward walk admits progressively worse
measurements and has no fixed point); the floor is not.

Stated honestly: `dist` is inflated by an under-estimated `P` just as much as by
an under-estimated `R`, and the median cannot tell them apart. Charging it all to
`R_` is a **consistency correction, not a noise identification** — it makes the
filter's own uncertainty statement self-consistent, which is what a
well-calibrated gate needs, and it is defensible precisely because the bracket
above established that the real pixel noise varies by 3× here.

Off by default (`visual_meas_adapt.enable`, absent from every shipped config),
`LOG(FATAL)` if enabled without `use_MH_gating`, and **verified inert when off**:
TUM-VI room1 stereo trajectory md5 `500d4fa6b8cd1593a1e33af8121f0cef`,
bit-identical before and after. Eight unit tests drive the real `Estimator`
through `#define private public` using *deterministic* χ²(2) quantiles —
`-2 log1p(-p)` at `p = (k+0.5)/n` with n = 61 odd, so the middle order statistic
*is* the exact median — which makes every expectation an equality rather than a
tolerance. They pin that a consistent filter is a fixed point, that the estimate
converges monotonically from either side without overshoot, that both clamps bind,
that warm-up and the minimum sample count hold it still, that non-finite `dist`
values are dropped before ranking rather than sorted to an end, and that even n
averages the two middle values. The test header says explicitly what they do not
show — that the loop lands on a *useful* value on real data.

**The time constant is the load-bearing parameter, and the first attempt at it was
wrong for an instructive reason.** `α = 0.05` ("a 1 s time constant at 20 Hz")
gave 0.144 — worse than the wide fixed gate. The obvious reading is that the loop
finds a fixed point that is not ATE-optimal. That reading is wrong: the census line
shows the Machine Hall sequences settling and staying settled, while the Vicon Room
ones *excurse and fall back* (V2_01 reaches σ = 2.11 mid-run and ends at the 0.60
floor). The excursions are the fast segments, and with a 20-update time constant
the estimate is still climbing when the segment ends. The problem was never the
fixed point; it was the lag.

| `α` | time constant | mean ATE [m] |
| --- | --- | --- |
| 0.02 | 50 updates, 2.5 s | 0.163 |
| 0.05 | 20 updates, 1.0 s | 0.144 |
| **0.15** | **6.7 updates, 0.33 s** | **0.095** |
| 0.30 | 3.3 updates, 0.17 s | 0.097 |
| 0.50 | 2 updates, 0.10 s | 0.099 |

A knee between 0.05 and 0.15, then flat to 0.5 — so `α` must be *fast enough* and
its exact value above the knee is second-order, which is the good kind of
parameter. At 0.15 the estimate actually reaches the `max_std = 4` ceiling on
V2_01/02/03, and V2_01 goes 0.115 → 0.048, from losing to OpenVINS' 0.054 to
beating it.

**Two mechanisms that each win half the dataset do not compose**, and that is the
confirmation of the gate/weight identity above:

| config | mean | Machine Hall | Vicon Room |
| --- | --- | --- | --- |
| adapt α=0.15, gate 5.991 | 0.095 | **0.086** | 0.103 |
| adapt α=0.05, gate 12 | 0.093 | 0.102 | **0.086** |
| both | 0.099 | 0.105 | 0.094 |
| OpenVINS | 0.094 | 0.135 | 0.060 |

Applying both over-widens the same effective gate — α widens it dynamically during
blurred bursts, `MH_thresh` widens it statically all the time — and Machine Hall,
where the tracks are sharp and the gate *should* be tight, pays for it.

**Does the adaptation earn its complexity?** Each finalist was run against its own
control, the identical config with `visual_meas_adapt.enable = false`: 0.158 →
0.095 at the textbook gate (**40%**) and 0.125 → 0.093 at gate 12 (26%). Both far
outside ensemble noise, so the loop is doing real work and is not a relabelled
gate widening.

**Choosing between the two finalists on the mean would be choosing on noise**
(0.093 vs 0.095 is 2 mm on a 95 mm number). The tiebreak was the *neighbourhood*:
the α ladder is smooth and flat above the knee, while `MH_thresh = 9` — immediately
next door to 12 — reports **0.286**, entirely V2_03 at 2.320 ± 3.739, one member
of three partially losing the trajectory. The other ten sequences at gate 9 average
0.083, and *that* is what makes it disqualifying rather than merely bad: the other
ten being good is what a knife edge looks like. A configuration whose neighbour is
bimodal is a configuration that got a good draw. The textbook gate also has an
argument behind it that the number 12 does not — if the estimate tracks the real
noise, a correctly calibrated gate should stay at its nominal quantile.

M4's endpoint, n=6, both modes, **0 of 66 divergences in each**: stereo `ate_002`
0.138 → **0.095**, a 31% improvement, with the shape already fixed — XIVO wins
both orientation metrics and RPE-8, loses absolute position, and splits per
sequence by *scene* rather than by difficulty.

### 4.4 M5 — throughput: 14.76 → 11.59 ms/frame

M4 ended tied on accuracy and **37% slower**. The gap to close was 4.03 ms/frame.

**Instrumentation first, because the gap is not where the obvious story puts it.**
`src/tracker.{h,cpp}` gained `Tick`/`Tock` pairs around every distinct piece of
front-end work plus two counters (`mean_raw_detections()`, `num_detect_frames()`);
`pybind11/pyxivo.cpp` gained a `decode_timer_` and `ReportDecode()`, because PNG
decode happens in the Python-facing wrapper and is invisible to the estimator's own
timers — and it is 22% of the frame, so leaving it out would have made every ratio
wrong. OpenVINS was instrumented by parsing what it already reports under
`--verbosity DEBUG`, which needs three fixes to add up: strip ANSI colour codes,
collapse the per-bin stage variants, and divide by frame count rather than trusting
`fps_mean`.

Two traps found while instrumenting, both recorded because they silently produce
wrong numbers: `estimator.cpp:1631` also prints a tracker timer, but that is the
**tracker-only** code path, not the one a stereo VIO run takes; and `Timer`'s
stream operator truncated the accumulated total to integer milliseconds *before*
dividing by the occurrence count, quantising every per-call mean to `1/occurrence`
ms and biasing it low. That was noticed because `detect-sort` and `detect-subpix`
printed the bit-identical `0.240924 ms` — and the arithmetic confirms the mechanism
exactly: detection ran on 303 frames and `73/303 = 0.2409240…`. Fixed to divide in
nanoseconds, with the occurrence count now printed.

MH_01_easy, stereo, one core:

| stage | XIVO (M4) | OpenVINS | XIVO − OV |
| --- | --- | --- | --- |
| image read / decode | 3.296 | 3.475 | **−0.18** |
| feature tracking | 6.257 | 3.222 | **+3.03** |
| EKF propagation | 0.033 | 0.078 | −0.05 |
| EKF update | 3.813 | 2.275 | **+1.54** |
| marginalise / bookkeeping | 0.953 | 0.930 | +0.02 |
| **estimator total** | **11.00** | **6.50** | **+4.50** |

Both readings are the opposite of the obvious guess: **XIVO's image path is
already faster** than OpenVINS' `imread` (the `libdeflate` decode from the previous
round), and **72% of the gap is the front end** — while OpenVINS is doing *more*
work in it, carrying ~600 point-tracks against XIVO's 334.

Inside XIVO's 6.26 ms front end: equalization 2.06 ms (CLAHE, both cameras, every
frame), `stereo-match` 2.22, `klt` 1.46, `pyramid` 0.27, and `detect-total` **0.16**
— which is the first useful surprise, since detection costs 1.92 ms per call but
runs on only 8.4% of frames. Any knob aimed at the detector is aimed at 2.5% of the
frame; anything *paid* there is amortised 12×.

**CLAHE cannot be made cheaper by configuration.** Going from an 8×8 grid to 2×2 —
64 tile histograms down to 4 — saves only 0.341 of the 2.06 ms, so the cost is the
bilinear interpolation pass that touches all 360,960 pixels regardless of grid
size. The CLAHE object is not being rebuilt per frame (constructed once at
config-parse time), and `equalize-left` reads 0.000 ms under `NONE`, which confirms
the `ToGray` inside that timer is a genuine no-op on already-8-bit images, so
1.031 ms is `clahe_->apply` and nothing else — 2.86 ns/px, the expected range for
OpenCV's non-vectorised interpolation body.

**What CLAHE buys is not what one would guess, and that is what made the win
possible.** It is *not* keypoint supply: FAST at threshold 20 returns 6873 raw
corners per detecting frame with CLAHE and 1593 without, and 1593 is still an order
of magnitude more than the 180 features XIVO wants. It is *not* stereo match
quality: removing CLAHE **raises** the match rate 79.2% → 85.6% and cuts epipolar
rejections from 100k to 71k, because CLAHE's local per-tile mapping makes the same
physical patch look different in the two cameras where a tile boundary falls
differently. What is left is **spatial distribution**: FAST at a fixed global
threshold fires only where local contrast already exceeds it, so on the raw image
corners concentrate in the well-lit parts of the frame; CLAHE lifts the dark
regions over the threshold and spreads the features, and a better-spread feature
set is better conditioned for pose. The per-sequence pattern fits — removing CLAHE
loses on the wide-dynamic-range sequences and *improves* the evenly-lit ones.

That diagnosis has a testable consequence: if CLAHE buys distribution *through the
threshold*, then lowering `FAST.threshold` on the raw image should buy much of it
back for a fraction of 2.06 ms. It does, and the supply match is nearly exact —
**threshold 7 on the raw image yields 6357 candidates per detecting frame against
CLAHE-at-20's 6913**, for 0.356 ms/frame:

| arm | `FAST.threshold` | raw kps/frame | ms/frame | `ate_002` (n=3) |
| --- | --- | --- | --- | --- |
| CLAHE at 20 | 20 | 6913 | 14.007 | 0.095 |
| `B0` — no CLAHE | 20 | 1615 | 11.194 | *0.307* |
| `B10` | 10 | 4076 | 11.524 | 0.101 |
| **`B7`** (shipped) | **7** | **6357** | **11.550** | **0.098** |

**That one substitution is 2.5 of the 3.2 ms.** The 0.008 m it does not recover is
presumably the part of CLAHE's effect that a candidate *count* cannot capture: a
global threshold drop adds corners wherever contrast is already near the threshold,
i.e. mostly where it was already textured, whereas CLAHE's local mapping
specifically promotes the dark regions. Matching the count is not matching the
distribution.

**Composition is 93–94% additive** and the stage timers show exactly what moved.
Three things visible there that no end-to-end number shows: `actual-update` is
**invariant** at 3.65 ± 0.04 ms across every arm (every front-end knob leaves the
covariance update untouched — which is why the residual gap is where it is);
`stereo-klt` *rises* under `klt2` alone, because with two temporal pyramid levels
instead of four features land slightly less accurately and the stereo search starts
further away (a knob's cost can appear in a stage it does not touch); and
`detect-fast` halves while `detect-sort` drops 4.4× under `eqnone`, because raw
detections fall 6873 → 1593 and both non-maximum suppression and the response sort
scale with candidate count — the one place where removing CLAHE pays a second time.

The after-picture, and the honest shape of the result:

| stage | XIVO shipped | OpenVINS | XIVO − OV |
| --- | --- | --- | --- |
| image read / decode | 3.289 | 3.475 | **−0.19** |
| feature tracking | 2.934 | 3.222 | **−0.29** |
| EKF path (update + marg + init + prop) | 5.04 | 3.28 | **+1.76** |
| **total** | **11.34** | **9.99** | **+1.35** |

**XIVO's decode and front end are now both faster than OpenVINS', and the entire
residual gap is the EKF path** — state size, not implementation slack: 90 in-state
features and a 20-pose OOS window against OpenVINS' 50 SLAM features and 11 clones,
where the OOS path is load-bearing on EuRoC (removing it takes ATE to 0.219).

### 4.5 M6 — the final evaluation

No tuning. Both operating points measured at n=10 on all eleven sequences in both
modes (440 runs), plus a one-core timing pass for both in the same session, against
the M1 OpenVINS reference. This is where the n=3 → n=10 correction of §1.4 was
found and applied. Results in §5.


## 5. Results

### 5.1 Accuracy, and how significant each difference is

The headline table is at the top. In units of the combined standard error:

| metric | `acc` vs OpenVINS | `fast` vs OpenVINS |
| --- | --- | --- |
| `ate_002` | −0.0009, **0.8 σ — a tie** | −0.0087, 5.1 σ worse |
| ATE position | −0.0063, 5.8 σ worse | −0.0130, 7.6 σ worse |
| ATE orientation | +0.064, 4.6 σ **better** | +0.067, 4.7 σ **better** |
| RPE 8 m position | +0.0075, 9.9 σ **better** | +0.0059, 6.9 σ **better** |
| RPE 8 m orientation | +0.050, 8.9 σ **better** | +0.035, 6.1 σ **better** |

Counting the 55 per-sequence-per-metric cells three ways, XIVO takes **32**
(`acc` 17, `fast` 15) against OpenVINS' 23.

### 5.2 Per sequence

`ate_002`, stereo:

| sequence | XIVO `acc` | XIVO `fast` | OpenVINS | winner |
| --- | --- | --- | --- | --- |
| MH_01_easy | 0.087 ± 0.007 | 0.080 ± 0.018 | **0.073** ± 0.000 | OV |
| MH_02_easy | 0.052 ± 0.012 | **0.047** ± 0.010 | 0.090 ± 0.011 | XIVO, 1.9× |
| MH_03_medium | **0.099** ± 0.007 | 0.101 ± 0.006 | 0.116 ± 0.003 | XIVO |
| MH_04_difficult | 0.100 ± 0.008 | **0.098** ± 0.009 | 0.207 ± 0.000 | XIVO, 2.1× |
| MH_05_difficult | **0.086** ± 0.012 | 0.115 ± 0.013 | 0.190 ± 0.000 | XIVO, 2.2× |
| V1_01_easy | 0.069 ± 0.004 | 0.074 ± 0.006 | **0.055** ± 0.003 | OV |
| V1_02_medium | 0.071 ± 0.003 | 0.091 ± 0.004 | **0.047** ± 0.002 | OV, 1.5× |
| V1_03_difficult | 0.163 ± 0.009 | 0.157 ± 0.023 | **0.059** ± 0.002 | OV, 2.7× |
| V2_01_easy | **0.050** ± 0.002 | 0.057 ± 0.004 | 0.054 ± 0.004 | XIVO |
| V2_02_medium | 0.091 ± 0.003 | 0.095 ± 0.008 | **0.049** ± 0.002 | OV, 1.9× |
| V2_03_difficult | 0.176 ± 0.018 | 0.215 ± 0.040 | **0.096** ± 0.010 | OV, 1.8× |

Orientation and RPE-8 position, same ensembles:

| sequence | ori `acc` | ori `fast` | ori OV | RPE8 `acc` | RPE8 `fast` | RPE8 OV |
| --- | --- | --- | --- | --- | --- | --- |
| MH_01_easy | **1.42** | 1.49 | 1.57 | 0.091 | **0.078** | 0.083 |
| MH_02_easy | **0.73** | 0.87 | 1.10 | 0.047 | **0.041** | 0.082 |
| MH_03_medium | 1.49 | 1.57 | **1.29** | 0.112 | **0.108** | 0.158 |
| MH_04_difficult | 1.46 | 1.36 | **1.23** | 0.155 | **0.151** | 0.182 |
| MH_05_difficult | 0.76 | **0.70** | 0.73 | **0.075** | 0.089 | 0.133 |
| V1_01_easy | 5.48 | 5.58 | **5.41** | 0.262 | 0.259 | **0.246** |
| V1_02_medium | 1.94 | 1.99 | **1.88** | 0.123 | 0.122 | **0.111** |
| V1_03_difficult | 1.94 | **1.63** | 2.38 | 0.123 | 0.133 | **0.083** |
| V2_01_easy | 1.24 | 1.15 | **1.12** | **0.059** | 0.063 | 0.084 |
| V2_02_medium | 1.06 | **1.02** | 1.24 | 0.080 | 0.074 | **0.047** |
| V2_03_difficult | **1.28** | 1.41 | 1.53 | **0.075** | 0.102 | 0.075 |

`V1_01_easy`'s 5.4–5.6 deg is not a XIVO problem: all three arms are within 3% of
each other on it, and XIVO's *baseline* gave 5.53 against OpenVINS' 5.41. Two
independent estimators do not share a defect that specific, so it is a property of
`posyaw` alignment on that sequence's ground truth. Reported, not chased.

### 5.3 Machine Hall vs Vicon Room, and what RPE-8 says about it

The environment split is in the headline. What separates the two halves:

* **Machine Hall** — large, dimly and unevenly lit, flown slowly, distant
  structure, and two of the five sequences open with a manoeuvre rather than a
  static hold. OpenVINS' two worst sequences on the entire dataset are MH_04
  (0.207) and MH_05 (0.190), and both are initialisation-limited. XIVO's delayed
  feature initialisation and its 20-pose OOS window do real work here.
* **Vicon Room** — small, bright, flown fast, close-range features, heavy
  rotational motion, high feature churn, short tracks. XIVO loses 5 of 6, by
  1.5–2.7×. `V2_03_difficult` is also where all of XIVO's member spread lives
  (±0.040 m against OpenVINS' ±0.010 on the same sequence), which says the losses
  are gating decisions going different ways rather than a bias.

**The diagnostic is that RPE-8 disagrees with ATE on exactly these sequences.** On
`V2_03_difficult` XIVO `acc`'s ATE is 1.8× worse than OpenVINS' (0.176 vs 0.096)
while its RPE-8 position is *equal* (0.075 vs 0.075); on `V1_03_difficult` ATE is
2.7× worse while RPE-8 is only 1.5× worse. A competitive local drift rate
alongside an uncompetitive absolute error means the Vicon Room losses are dominated
by **a small number of transient excursions** that ATE integrates and RPE-8
averages out — not by a uniformly worse motion estimate. Finding those excursions
is the highest-value item left on the accuracy axis, and it is an estimator
question rather than a configuration one: M4 swept the configuration space around
this, and even a per-sequence oracle on `visual_meas_std` leaves V1_03 at 0.140.

### 5.4 Throughput and memory

The one-core table is in the headline. Two refinements:

**End-to-end flatters XIVO relative to the estimator alone**, because its PNG
decode is faster (2.972 vs 3.197 ms/frame over all eleven):

| | end-to-end | decode | estimator only |
| --- | --- | --- | --- |
| XIVO `acc` | 13.921 | 2.972 | 10.949 (+45.2%) |
| XIVO `fast` | 11.593 | 2.972 | 8.621 (**+14.3%**) |
| OpenVINS | 10.737 | 3.197 | 7.542 |
| `fast` vs OV | +8.0% | −7.0% | **+14.3%** |

The honest framing is the estimator column: **XIVO's estimator is 14% more
expensive**, and essentially all of it is the covariance update over a
deliberately larger state (§4.4), not unoptimised code.

**Per sequence the M5 gain is uniform** — between 2.7 and 3.4 ms on every one of
the eleven — which is the last check that nothing here is a single-sequence
artifact. `acc` was measured twice and reported 13.9207 and 13.9210 ms/frame.

### 5.5 Divergence census

A run counts as diverged at `ate_002` > 100 m.

| | stereo | mono |
| --- | --- | --- |
| XIVO `acc` | 0 / 110 | **10 / 110** (`MH_01_easy`, all members) |
| **XIVO `fast`** (shipped) | **0 / 110** | **0 / 110** |
| OpenVINS | 0 / 66 | 0 / 66 |

Both shipped configurations complete every sequence in both modes. Worth recording
because it was not true earlier on either side: XIVO's M3 baseline diverged on 4 of
11 sequences (15 of 66 runs), and OpenVINS diverges on MH_04 in 6 of 6 without
`--init_dyn_use 1`.

### 5.6 Mono, as a secondary mode

The brief specifies stereo + IMU for the final evaluation. Mono is reported because
the generator produces both and because it is where the `acc` failure above shows
up. All eleven, XIVO `fast` n=10 against OpenVINS n=6:

| metric | XIVO `fast` | OpenVINS |
| --- | --- | --- |
| `ate_002` [m] | 0.185 | **0.145** |
| ATE position [m] | 0.190 | **0.149** |
| ATE orientation [deg] | **1.77** | 1.87 |
| RPE 8 m position [m] | 0.165 | **0.125** |
| RPE 8 m orientation [deg] | 0.93 | **0.91** |

**OpenVINS wins mono 4 metrics to 1.** The environment split is the same shape but
more lopsided — XIVO wins MH_01, MH_02, MH_04, MH_05 and V2_01, and loses V1_01
(0.134 vs 0.058), V1_02 (0.161 vs 0.067) and V1_03 (0.325 vs 0.068) by 2–5×.
Losing the stereo baseline's scale observability amplifies exactly the Vicon Room
weakness §5.3 identifies, which is consistent with the excursion hypothesis:
without a metric baseline, a transient depth error has nothing to pull it back.
Mono was measured throughout and **never used to choose a configuration value**,
only to reject one (§6). It is not tuned.


## 6. Which configuration ships, and why

The M5 front-end work produced a genuine Pareto pair, both reachable from the same
generator:

| | | ms/frame | what it is |
| --- | --- | --- | --- |
| **`fast`** | shipped | 11.593 | `histogram_method NONE`, `FAST.threshold 7`, `KLT.max_level 2`, `fast_png_decode false` |
| **`acc`** | alternative | 13.921 | the same with CLAHE restored and `FAST.threshold` back to 20 |

`acc` is better than or equal to `fast` on all five accuracy metrics and costs
2.328 ms/frame more — a clean trade with no dominance either way, so the choice
needs stated grounds. **`fast` ships**, on three:

1. **It is the only one of the three configurations that keeps mono `MH_01_easy`
   alive.** M4's config already scored a bad 1.357 ± 0.548 m there; `acc`, which
   adds `KLT.max_level 2` on top of CLAHE, **diverges in 10 of 10 members** (270 m
   to 28 km); `fast`, which has `KLT.max_level 2` but no CLAHE, scores 0.115 m.
   This is a composition effect of exactly the kind M5 found in stereo with `B0`,
   and it was invisible during M5 because M5 screened accuracy in stereo only.
   Shipping a configuration that turns a working sequence into a 28 km divergence
   in a supported mode is not defensible.
2. **It makes the throughput comparison close** — 8.0% behind OpenVINS against
   `acc`'s 29.7%. Since both operating points win and lose the same 3-of-5 split of
   accuracy metrics, throughput is the axis where they actually differ.
3. **It is what the generator produces**, so the shipped config, the committed
   `cfg/euroc_*.json` and the documented defaults are one thing rather than three.

What `fast` gives up is the *size* of the position-ATE loss, and that is a real
cost worth stating plainly: `acc` ties `ate_002` with OpenVINS at 0.8 σ where
`fast` loses it at 5.1 σ. Anyone who wants that tie back, and does not need mono,
has it in two flags:

```sh
python3 scripts/make_euroc_cfg.py --base cfg/eff_stereo.json \
  --seqdir /path/to/euroc/MH_01_easy --out cfg/euroc_stereo.json \
  --histogram_method CLAHE --fast_threshold 20
```


## 7. What was added to XIVO

Three branches, `486837a..auto-eurocfps`: 2181 insertions, 50 deletions across 16
files. New files: `scripts/make_euroc_cfg.py`, `cfg/euroc_{stereo,mono}.json`,
`src/test/unittest_adapt_meas_noise.cpp`.

| area | what | default |
| --- | --- | --- |
| `scripts/pyxivo.py` | `euroc` dataset branch (`<root>/<seq>/mav0/...`) | new dataset |
| `scripts/savers.py` | ground-truth path resolved per dataset; `v[4:8]` quaternion slice | fixes TUM-VI hardcoding |
| `scripts/make_euroc_cfg.py` | generate one shared config from the dataset's own `sensor.yaml`, with the `T_BS`-identity assert and the two operating-point flags | new |
| `src/estimator.{h,cpp}` | `gravity_init_max_accel_dev` stationarity gate | 0 = off, i.e. inert |
| `src/update.cpp`, `src/estimator.h` | `Estimator::AdaptVisualMeasNoise`, the χ²(2)-median consistency loop, `R_pending_` one-frame delay | `visual_meas_adapt.enable` absent from every other config |
| `src/tracker.{h,cpp}`, `src/manager.cpp` | per-stage front-end timers and two detection counters | instrumentation |
| `common/timer.h` | integer-millisecond truncation fixed; occurrence count printed | bug fix |
| `pybind11/pyxivo.cpp` | `decode_timer_` + `ReportDecode()` | instrumentation |
| tests | `unittest_adapt_meas_noise` (8 cases, deterministic χ² quantiles), `unittest_gravity_init` (+4 cases) | `ctest` 23/23 |

Every new estimator key is a no-op when absent, so merging the code without the
EuRoC configs is behaviourally identical to before — verified by md5 on a TUM-VI
room1 stereo trajectory.

Nothing from OpenVINS is linked or included. What was learned from reading it —
that a disparity threshold is a bad proxy for a velocity threshold — informed
XIVO's own gate; the implementation shares no code.


## 8. Methodological findings that generalise

Recorded separately because each cost real time and each will cost it again.

1. **Screen at n=3, ship at n=10** (§1.4). Three members under-sample this dataset
   by ~0.005 m, which is the same order as the differences the tuning had to
   resolve. The failure is invisible without re-running the *same* members.
2. **A per-knob price list is only valid at the configuration where it was
   measured.** Once `histogram_method=NONE` is in the config, `KLT.max_iter=15` —
   the best exchange rate in the entire throughput milestone at 0.46 ms per
   0.001 m — becomes one of the worst at 0.08, because a KLT on a lower-contrast
   image has weaker gradients and less margin. Three knobs that were nearly free
   stopped being free.
3. **Screens must include the sequences that already work.** OpenVINS'
   `disp3` fix, and XIVO's original five-sequence `visual_meas_std` screen, both
   looked good precisely because they were measured only where things were already
   broken.
4. **Instrument stages, not end-to-end wall clock.** Two arms in this round save
   wall clock *by failing*: `stereo_max_level=1` collapses the stereo match rate to
   54.5% (and `stereo-klt` goes *up*) so the update has less to do, and `B0` drops
   tracks on the two hardest sequences. Both read as ordinary trades on ATE alone.
   Disqualify on the mechanism, not on the metric.
5. **An sd larger than the mean is a bimodal distribution, not an accuracy
   measurement.** Several arms in M3 and M4 report exactly that, and the correct
   reading is "some members partially lost the trajectory without crossing the
   divergence threshold".
6. **A configuration whose immediate neighbour is bimodal got a good draw.** That,
   not the mean, decided M4's final configuration.
7. **An equivalence check cannot double as a measurement.** The hand-patched arm
   and the regenerated config agreed on all five metrics — and both were the same
   under-sampled n=3 draw.


## 9. Negative results, complete

Everything measured and rejected, so the next person does not re-run it.

**Disqualified on accuracy despite excellent timing** (against a 0.095 baseline):

| arm | ms/frame | `ate_002` | why |
| --- | --- | --- | --- |
| `equalize_for=DETECT` | −1.928 | **0.222** | Real incompatibility with CLAHE, not a tuning artifact: FAST and `cornerSubPix` run on the equalized image while the KLT tracks the raw one. Global histogram equalization is a monotonic point map, so a corner keeps its location; CLAHE is local and not monotonic across tile boundaries, so every new feature starts with a sub-pixel bias. The prediction that `DETECT` is safe with `HISTOGRAM` holds (0.101), which is the confirmation. Now documented as unsafe with CLAHE in the generator. |
| `use_OOS=false` | −1.614 | **0.219** | The OOS/MSCKF path is load-bearing on EuRoC (V2_03 1.097, V1_03 0.363). Shrinking the window instead (`OOS.pose_window=10`) is a normal 0.14-rate trade and also not worth taking. |
| `stereo_matching.max_level=1` | −0.272 | 0.100 | Saves wall clock *by failing*: unseeded with one pyramid level the coarse level cannot bracket a 150 px disparity in a 15 px window, so points iterate to the 30-iteration cap, the match rate collapses 79.2% → 54.5%, and `stereo-klt` goes **up** to 2.129 ms. The apparent saving is a quarter of the stereo measurements no longer existing. |
| `MH_thresh = 9` | — | 0.286 | Bimodal: V2_03 at 2.320 ± 3.739 while the other ten average 0.083. |

**Rejected as bad trades** (exchange rate, ms/frame per 0.001 m of `ate_002` given
up — higher is better; read as an ordering, since all were screened at n=3):
`OOS.pose_window=10` 0.14, `KLT.max_iter=10` 0.14, `num_features_max=150` 0.14,
`KLT.win_size=13` 0.11, `KLT.win_size=11` 0.20, `histogram_method=HISTOGRAM` 0.18,
`equalize_for=DETECT`+`HISTOGRAM` 0.42 (dominated by `NONE`). They cluster between
0.11 and 0.46, and that clustering across a dozen unrelated knobs is what "XIVO
sits on a smooth accuracy/compute curve" looks like from outside.

**Inert or negative:** `ekf_update.chunks=6/8` inert, `chunks=2` +0.26 ms/frame,
`max_group_lifetime=30` bit-identical (groups are retired by other mechanisms long
before 60 frames, so the knob is dead code on EuRoC),
`visual_meas_adapt.max_std=6.0` inert (0.097 vs 0.095), `clahe_grid_size=16`
+0.32 ms. `fast_png_decode` is a **0.300 ms/frame loss** on EuRoC — the
`libdeflate` path wins on TUM-VI by fusing a 16→8-bit strip conversion into the
decode, and EuRoC images are already 8-bit, so there is nothing to fuse; turning it
off is byte-identical, confirmed by `diff -r` on two sequences' trajectories.

**Refuted hypotheses:** that XIVO's gravity initialiser explained MH_04 (measured:
its init window is within 0.050 m/s² and max |ω| is 0.11 rad/s — MH_04 is genuinely
still when XIVO initialises); that `MH_max_strikes > 1` would fix survivor bias in
the adaptive loop's median (worse, monotonically in strikes, and worst on the
fastest sequence — a feature that fails the gate once is genuinely gone here); that
CLAHE buys keypoint supply or stereo match quality (it buys neither — §4.4); that
`num_features0 < 15` was OpenVINS' initialisation loophole (that branch *refuses*
to initialise rather than falling through).

**Not built, deliberately.** A *geometric* stereo seed — predicting the right-image
location from the feature's 3-D estimate and the known extrinsics — would approach
the 0.63 ms ceiling instead of the 0.23 that `seed_prev_disparity` captures. It was
not built because `Feature::Xc()` returns the point in its **reference** camera
frame; the correct current-frame prediction exists only inside `ComputeJacobian`,
which runs *after* `MatchStereo`, so getting it earlier means threading the
estimator's propagated pose into the `Tracker` singleton. That is a real front-end/
filter coupling for ~0.4 ms, and it would not change any conclusion here. Also not
attempted: `GAINMAP` (a frozen radial gain fitted to TUM-VI vignetting, documented
as losing to CLAHE on keypoint supply) and `FAST.threshold` below 7 (the 20→10→7
trend was already flattening while detect cost rose).


## 10. What this does not show

* **Eleven sequences and one dataset.** The environment split is a strong effect
  but it is two samples of "environment", not a law. Nothing here predicts which
  system wins on a dataset that is neither a machine hall nor a mocap room.
* **The error bars are member spread, not configuration uncertainty** (§1.4). Treat
  any two-configuration difference under ~0.005 m as unresolved.
* **Accuracy and throughput are measured under different protocols** by necessity
  — 220 concurrent processes versus one pinned process at a time. No number is ever
  taken from the other pass's columns.
* **The two ensembles are not equivalent.** OpenVINS' `gravity_mag` perturbation
  produces genuinely distinct trajectories, but OpenVINS is much less sensitive to
  it than XIVO is to velocity jitter (SE 0.0006 vs 0.0016). Its narrower error bars
  are a real property of the estimator on this data, not a measurement artefact —
  but they are narrow for a different reason than "more samples".
* **Timing is single-core by construction.** The two systems parallelise
  differently; multi-core throughput is a separate question this protocol
  deliberately excludes.
* **Mono is not tuned**, and XIVO loses it 4–1.

### What is left

1. **The Vicon Room excursions** (§5.3) — the highest-value accuracy item, and an
   estimator question. RPE-8 says the local motion estimate is competitive there,
   so the target is specific: find the transient excursions that ATE integrates.
2. **A spatially-adaptive detector threshold** — one bucket per tile targeting a
   per-tile candidate count — would plausibly recover the 0.008 m that `fast` gives
   up, at a fraction of CLAHE's 2.06 ms, since §4.4 shows what CLAHE buys is
   distribution rather than count.
3. **The geometric stereo seed** (§9), ~0.4 ms.
4. **A cheaper covariance update at the same state size**, which is the whole
   remaining 14% estimator gap and is code rather than configuration.


## 11. Reproducing

All commands from `experiments/openvins` unless noted, with the venv on `PATH`:

```sh
export PATH="/home/ubuntu/workspace/auto-slam-engineer/dependencies/venv/bin:$PATH"
```

**The shipped configs, regenerated from the dataset's own calibration** (this is
the definition of the configuration, not a copy of it):

```sh
cd ../../xivo
for m in stereo mono; do extra=""; [ $m = mono ] && extra="--mono"
  python3 scripts/make_euroc_cfg.py --base cfg/eff_$m.json \
    --seqdir ../data/euroc/MH_01_easy --out cfg/euroc_$m.json $extra
done
```

**Run one sequence** (the same entry point as any other dataset):

```sh
python3 scripts/pyxivo.py -root ../data/euroc -dataset euroc \
  -seq MH_01_easy -cfg cfg/euroc_stereo.json -mode eval -dump /tmp/out
```

**Accuracy, both XIVO operating points, n=10, all eleven, both modes** (440 runs,
~50 min on 176 cores):

```sh
CPU_BASE=0 CPU_SPAN=176 ./m6_final_batch.sh      # -> ../results/euroc_m6_final/{fast,acc}
```

**The OpenVINS reference** (6 members, both modes, ~2 h):

```sh
./ov_euroc_ens.sh                                # -> ../results/euroc_ov_acc_dyn
```

**The head-to-head tables** (§5.1, §5.2, §5.3):

```sh
python3 agg_ensemble.py --mode stereo \
  --arm xivo_acc  ../results/euroc_m6_final/acc \
  --arm xivo_fast ../results/euroc_m6_final/fast \
  --arm openvins  ../results/euroc_ov_acc_dyn/stereo_m*
```

`--mode` is mandatory (the XIVO runner writes mono and stereo rows into one
`summary.csv`), and the member glob must be **unquoted** — `agg_ensemble.py` takes
`--arm NAME DIR` pairs and expects the shell to have expanded them; quoting yields
"(not all arms have all sequences)" for every metric rather than an error. Swap
`--mode mono` for §5.6.

**Throughput, one core, all eleven** (~40 min per repeat per arm):

```sh
SEQS="MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult \
V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult"
CPU_BASE=0 ./sweep_fps.sh --name fast --seqs "$SEQS" --repeats 1 \
  --out ../results/euroc_m6_fps
CPU_BASE=0 ./sweep_fps.sh --name acc  --seqs "$SEQS" --repeats 2 \
  --out ../results/euroc_m6_fps \
  --patch 'tracker_cfg.histogram_method="CLAHE"' --patch 'tracker_cfg.FAST.threshold=20'
python3 report_onecore.py \
  xivo_M4=../results/euroc_fps_base/xivo_r0 \
  xivo_acc=../results/euroc_m6_fps/acc \
  xivo_fast=../results/euroc_m6_fps/fast \
  openvins=../results/euroc_fps_base/ov_r0
```

`report_onecore.py` is the cross-system aggregator: `report_fps.py` reads the
`time.txt` that only the XIVO path writes and warns on every OpenVINS run, while
both paths write `stats.txt`.

**Tests, and the merged-tree re-verification** (§13):

```sh
cd ../../xivo/build && make -j32 && ctest        # 23/23
cd ../../experiments/openvins
CPU_BASE=0 CPU_SPAN=176 ./run_xivo_reference.sh --profile euroc_mav \
  --mode stereo --jitter 3 --worktree xivo --cfg-prefix euroc \
  --out ../results/euroc_m7_merged
```

Then compare `euroc_m7_merged/summary.csv` against the `mode=stereo`,
`repeat<=2` rows of `euroc_m6_final/fast/summary.csv` — the two files carry the
same schema, so the check is a straight cell-by-cell string comparison over the
five metric columns.

Harness files live in `experiments/`, which is **not** version controlled; the
tracked copies are under `notes-n-prompts/notes-euroc/harness/`, and
`harness/HOWTO.md` is the entry point for running more experiments.


## 12. Result directories

All under `experiments/results/`, which is not version controlled. Quotable means
"a number from here appears in this report"; the rest are kept because a rejected
arm's numbers are the evidence for §9.

| directory | what it is | quotable |
| --- | --- | --- |
| `euroc_m6_final/{fast,acc}` | n=10, all 11, both modes, both operating points | **yes** — §5.1–§5.3, §5.5, §5.6 |
| `euroc_m6_fps/{fast,acc}` | one-core timing, all 11 | **yes** — §5.4 |
| `euroc_fps_base/{ov_r0..2,xivo_r0..2}` | one-core timing, OpenVINS and XIVO M4 | **yes** — §5.4 |
| `euroc_ov_acc_dyn/{stereo,mono}_m*` | the OpenVINS n=6 reference | **yes** |
| `euroc_m7_merged` | merged-tree re-verification, n=3 stereo | **yes** — §13 |
| `euroc_xivo_base` | the M3 baseline, n=6 both modes | **yes** — §4.2 |
| `euroc_xivo_m4b` | M4's n=6 reference | superseded by `euroc_m6_final/acc` |
| `euroc_m6_xivo` | the n=6 intermediate | only for §1.4 |
| `euroc_fps_ship`, `euroc_fps_ship11` | M5's n=3 confirmation | superseded; see M5's correction |
| `euroc_xivo_m4` | an earlier M4 intermediate, mean 0.138 | **no** |
| `euroc_fps_acc/*`, `euroc_fps/*`, `euroc_tune/{f,g,h,i,j}_*` | the tuning screens | ordering only, not values |
| `euroc_tune/{a,b,c,w}_*` | the mis-specified five-sequence screen | **no** — §4.3 |

Per-milestone notes are in `notes-n-prompts/notes-euroc/m0`–`m6`, the plan and its
one revision in `notes-n-prompts/plan-euroc.md`.


## 13. The merged tree

`auto-euroc`, `auto-eurocacc` and `auto-eurocfps` form a linear chain, and each was
merged into `auto` with `--no-ff` so the milestone structure survives in the
history. The merged tree is byte-identical to `auto-eurocfps`
(`git diff --stat auto-eurocfps HEAD` is empty), and both build directories are
configured identically (`Release`, `-march=native`, `EKF_MAX_FEATURES=90`,
`EKF_MAX_GROUPS=45`, `XIVO_EIGEN_INIT=none`).

Re-verified on the merged tree rather than assumed, because branch numbers not
composing was the lesson of the previous round:

* `make -j32` clean, `ctest` **23/23**.
* All eleven sequences, stereo, n=3 from the merged worktree
  (`results/euroc_m7_merged`) against members 0–2 of `euroc_m6_final/fast`: all
  **165 metric cells** (11 sequences × 3 members × 5 metrics) are **identical to
  the last printed digit**, and the 33-run `ate_002` mean matches at 0.097856 on
  both sides. So the merged tree is the tree the results in §5 were measured on.

That 0.0979 against the shipped n=10 mean of 0.1028 is §1.4's point restated: the
same configuration, the same first three members, 0.005 m of pure under-sampling.
Members 0–2 are the right thing to compare for *equivalence* and the wrong thing
to quote as a *result*.
