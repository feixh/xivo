# M5 -- what dynamic initialization is worth, and where

M4 wired the dispatcher into the filter and shipped it **off**. M5 is the
measurement that decides whether it goes on, and it turned out to hinge on a
question the plan did not anticipate: *on what data*.

| file | what it is |
|---|---|
| `harness/m5_ensemble.sh` | the two-arm ensemble, the null control, the divergence census |
| `harness/m5_cost.sh` | what it costs: the probe's one-off, then one-core throughput and peak RSS |
| `harness/trunc_control.py` | re-scores `off` over `on`'s pose set, to rule out an alignment artifact |
| `harness/seed_error.py` | what each initializer gets wrong, against groundtruth, at its own handoff |
| `bin/init_probe -start` | what the detector thinks, at any instant in a sequence |
| `bin/init_probe -dispatch` | what the shipped dispatcher decides, and what the decision cost |
| `scripts/pyxivo.py -start_sec` | turn the estimator on mid-flight |

To see what the *filter* decided rather than what the probe decides, configure a
second build tree with `-DXIVO_STRIP_LOG=0` -- `XIVO_STRIP_LOG` defaults to 1 and
compiles every `LOG(INFO)`, including the dispatch summary, out of the binary. Add
`-DXIVO_OUTPUT_SUFFIX=_log` with it, or the variant build overwrites the shared
`bin/` and `lib/` and every later measurement silently runs against the diagnostic
binary. With the suffix, a diagnostic filter run is `XIVO_LIB=lib_log python3
scripts/pyxivo.py ...` and the dispatch summary lands in `/tmp/pyxivo.*.INFO.*`.

## The measurement device, and one trap in it

Single-run ATE on EuRoC is not a measurement. M4 proved the nine static
sequences run **provably identical arithmetic** with the feature on and still move
by 0.009 m on average and 0.071 m at worst, because holding messages back changes
an image's enqueue timestamp by a few hundred nanoseconds and that flips a gating
decision. So every number here is a mean over an n-member ensemble whose members
differ only by a physically neutral perturbation, per the workspace's tuning
protocol.

The established knob perturbs the initial velocity, `X.Vsb += k * 1e-6 m/s`. **It
cannot be used here**, and this is the kind of trap that quietly invalidates an
evaluation: a dynamic initializer *solves for* the initial velocity and overwrites
the perturbation, so every member of the `on` arm comes out bit-identical on
precisely the two sequences the experiment exists to measure, and the ensemble
reports a confident `+-0.0000`. Measured, not assumed.

The replacement is `P.Tsb *= 1 + k * 1e-6` -- the prior *variance* of a quantity
that is zero by definition, which no initializer writes. Validated on the `off`
arm, where both knobs work: they give the same means to within 0.005 m and
comparable spread, i.e. they are the same sampling device. At 1e-9 relative it is
too small to flip a gating comparison and every member is bit-identical; 1e-6 is
the working point.

## Result 1: the two sequences EuRoC actually gives us

MH_01_easy and MH_02_easy are the only EuRoC sequences already moving at turn-on.
The other nine take the static path in both arms, which makes them a **built-in
null control**: whatever they say is the noise floor of this very comparison.

At n=3 that control did not look centred on zero: the nine static sequences moved
by **+0.0043 m** (sem 0.0012, 8 of 9 the same sign) in stereo and **-0.0041 m** in
mono, and the same offsets came back to the digit under two different `on`
configurations -- which reads as deterministic rather than noisy.

**At n=10 the offset is gone**: `+0.0003 +- 0.0032` (stereo) and `+0.0030 +-
0.0042` (mono), both indistinguishable from zero. The reproducibility across
configurations was reproducibility of the *same three jitter values*, not evidence
of a mechanism, and three members cannot resolve a per-sequence delta of this
size. Recorded because it is the trap the whole harness exists to avoid, and it
still caught me once: an n=3 offset with a plausible story attached.

What survives at n=10 is scatter, not bias: per-sequence `|delta|` averages
**0.006 m** in stereo and **0.009 m** in mono, worst 0.024 / 0.027. That is the
noise floor of this comparison, and it is what the two dynamic sequences have to
beat.

The scatter is real arithmetic, not evaluation bookkeeping. The two arms are *not*
bit-identical on a static verdict (`cmp` on the dumps differs from the first pose),
and `on`'s dump begins 18 poses -- 0.9 s -- later, because the diverted images
produce no pose for `pyxivo.py` to write. The obvious suspicion is therefore that
the two arms are Horn-aligned over different pose sets and so scored in different
frames; `trunc_control.py` re-scores `off` truncated to `on`'s first timestamp and
**rules it out**: truncation moves `off`'s ATE by less than 0.0001 m on every
sequence in both modes, those poses being a handful out of ~2000. What is left is
the M4 mechanism -- buffered frames are enqueued with `td_0`, no EKF update having
run yet to move `td`, so an image/IMU tie in the estimator's heap flips and gating
decisions follow. Nothing here can remove it: to be bit-identical on a static
verdict the filter would have to not be held back, and it cannot know the verdict
before it computes it.

## Result 2: the experiment the feature is actually for

Every public VIO dataset begins with the rig sitting on a table. That is why only
2 of 11 EuRoC sequences reach the dynamic branch at all, and why 9 of 11 can only
ever measure the *cost* of the feature. Testing an initializer on data that barely
needs it answers the wrong question.

So `pyxivo.py -start_sec N` drops the first N seconds of both streams and turns
the estimator on mid-flight. This is still EuRoC, still with groundtruth, still
scored by the same evaluator -- only the initial condition is hard. At **N = 55 s**
ten of the eleven sequences are moving at 0.16-1.42 m/s over the window the BA
would solve, and every sequence still has at least 29 s of trajectory left to
score.

What the static initializer does there is not a degradation, it is a failure. It
averages 20 accelerometer samples of a *moving* rig and calls the result gravity:

| | tilt error of the accel-average, deg | worst |
|---|---|---|
| at t=0 | 0.19 - 2.54 | V1_01 2.54 |
| at t=55 s | 0.83 - **23.98** | V1_03 23.98 |

and then asserts the platform is at rest while it is doing 1.4 m/s.

## Verdict quality: the detector is 22 for 22

`bin/init_probe -start` asks the shipped `MotionDetector` directly. Both start
conditions, all eleven sequences, one threshold pair:

| | t=0 | t=55 s |
|---|---|---|
| called dynamic | MH_01, MH_02 | all 11 |
| true speed of those | 0.28 - 0.48 m/s | 0.14 - 1.42 m/s |
| called static | the other 9 | none |
| true speed of those | <= 0.006 m/s | -- |

Every one of the 22 calls is correct, and the margin is not thin. At t=0 the
static class peaks at **0.097 px** of flow residual and the dynamic class floors at
**2.03 px** -- a factor of 21 with `flow_thresh` at 0.25 in between. The tightest
call anywhere is V2_01 at t=55, genuinely creeping at 0.14 m/s, and it still reads
0.445 px and 1.21 m/s^2, several times either threshold.

Worth noting because it is an independent check on something else: on the static
sequences the detector's `gyro_bias_hint` reads 0.072-0.086 rad/s, and EuRoC's
groundtruth gyro bias is 0.076-0.082 rad/s on all eleven. Two unrelated
estimators agreeing on a 4.5 deg/s bias that the static path seeds as zero.

## Result 3: at a table start, the feature is free and does nothing

n=10 per arm, all eleven sequences, `ate_002` in metres. `off` is the shipped
config with `dynamic_init.enabled=false` -- one key apart from `on`, not "the base
config left alone".

| | stereo off | stereo on | delta | mono off | mono on | delta |
|---|---|---|---|---|---|---|
| MH_01 (dynamic) | 0.0746 | 0.0789 | +0.0043 | 0.1051 | 0.1134 | +0.0083 |
| MH_02 (dynamic) | 0.0555 | 0.0642 | +0.0087 | 0.1058 | 0.1105 | +0.0047 |
| 9 static (null) | | | **+0.0003 +- 0.0032** | | | **+0.0030 +- 0.0042** |
| 2 dynamic | | | **+0.0065** | | | **+0.0065** |

The two sequences that reach the bundle adjustment move by +0.0065 m in both
modes, against a null whose own per-sequence spread is 0.006 (stereo) / 0.009
(mono). The honest reading: **at a table start the dynamic path is not measurably
better or worse than the static one.** It is not supposed to be -- MH_01 and MH_02
leave the table at 0.28-0.48 m/s, which is slow enough that averaging twenty
accelerometer samples is only 0.4-1.2 deg wrong about gravity. This table is the
no-regression check, and it passes.

## Result 4: mid-flight, it is the difference between working and not

The same n=10 device with both arms started 55 s in, so all eleven sequences are
dynamic starts. `DIVERGED` counts members whose ATE exceeds 100 m -- runs that did
not degrade, they failed, and averaging them with runs that worked would describe
neither.

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

The census is the result; the deltas are the footnote. **69 of 220 runs fail
without dynamic initialization and none fail with it**, and no run diverges that
did not diverge before. On the sequences that finish either way the mean moves
-0.014 m (stereo) and -0.244 m (mono), with the worst regression anywhere being
+0.0044 m stereo / +0.0085 m mono -- inside the t=0 null's own scatter -- against
best cases of -0.113 m and -1.183 m.

MH_05's exact `+0.0000` in both modes is not a rounding artifact. Its window's
reprojection median is 1.649 px, above the 1.5 px gate, so the solve is thrown away
and the static path runs instead: `cmp` on the dumps shows the `on` run is
**byte-identical** to the `off` run, in both modes and on every member checked. A
rejected solve costs accuracy exactly nothing.

MH_04 is rejected for the same reason and far more decisively (median 9.8 px), and
that is the right answer too: its mono `off` run is 0.84 m, i.e. the static path is
already struggling, and a window the BA cannot fit to better than 10 px would not
have helped. `on` still improves it to 0.61 m, because the divert alone shifts
where the filter starts.

## The metric that shows it is orientation, and that is not an accident

Everything above is `ate_002`, which Horn-aligns the two trajectories and is
therefore **blind to a global rotation**; `ov_eval ... posyaw` fixes only position
and yaw and charges roll and pitch in full. What the static initializer gets wrong
at a moving start *is* a global tilt. So `ate_002` is the least sensitive metric
available for this feature, and the four `ov_eval` metrics the protocol asks for are
not a formality. Same n=10 arms, same members; `+-` is the sem over the sequences in
the group.

| t=0 | 9 static (null) | 2 dynamic |
|---|---|---|
| `ate_002` m | +0.0003 +- 0.0032 | +0.0065 |
| `ov_ate_pos_m` | +0.0003 +- 0.0032 | +0.0110 |
| `ov_ate_ori_deg` | -0.0298 +- 0.0215 | +0.0549 |
| `ov_rpe8_pos_m` | -0.0006 +- 0.0005 | +0.0023 |
| `ov_rpe8_ori_deg` | +0.0104 +- 0.0050 | +0.0031 |

Neutral on every metric, in both modes (mono: null -0.0230 deg of orientation, the
two dynamic sequences +0.0276).

| t=55 s | stereo, 8 comparable | mono, 7 comparable |
|---|---|---|
| `ate_002` m | -0.0138 +- 0.0142 | -0.2443 +- 0.1632 |
| `ov_ate_pos_m` | -0.0152 +- 0.0175 | -0.2454 +- 0.1603 |
| `ov_ate_ori_deg` | **-0.5399 +- 0.3295** | **-3.8151 +- 3.2158** |
| `ov_rpe8_pos_m` | -0.0083 +- 0.0062 | -0.2066 +- 0.1655 |
| `ov_rpe8_ori_deg` | -0.1043 +- 0.0649 | -1.9046 +- 1.7322 |

Orientation is where the effect lives: **-0.54 deg (stereo) and -3.82 deg (mono)**
of absolute orientation error, on sequences that finish either way, against a t=0
null of 0.03 deg. The single clearest number in this evaluation is mono V1_02,
**25.31 deg -> 2.28 deg**, and `harness/seed_error.py` measures the same thing at
the other end of the chain: at t=55 V1_02's accelerometer average is **13.09 deg**
from gravity and the BA's gravity is **0.78 deg** from it. A tilt seeded at
initialization is not something a VIO filter works off.

The counterexample is worth as much. V1_01 is the one sequence where the solved
gravity is *worse* than the accelerometer average (3.14 deg against 1.90), and its
orientation error does not move at all: 5.85 -> 5.80 deg mono, 5.71 -> 5.67 stereo.
Its 5.8 deg is therefore not the seed's, a worse seed did not make it worse, and
"better gravity at t=0" is not a universal explanation for anything here.

## Result 5: the bug this milestone's own instrument found

`bin/init_probe -dispatch` was written to measure cost, and the first thing it
reported was `window build failed` on MH_01 -- a sequence the filter had been
initializing dynamically for two milestones. Neither was wrong. `InitWindow::Build`
refuses a window whose last frame the IMU does not reach, and:

* on EuRoC **every** image timestamp coincides exactly with an IMU sample (3682 of
  3682 on MH_01, and the estimator prints `measurement timestamps coincide?` for
  each one);
* which of the two the dispatcher is handed first is decided by a non-stable
  `std::sort` in `DataLoader` and by a timestamp-only heap in
  `Estimator::MaintainBuffer` -- unspecified in both;
* so when the window fills on an image, whether it can be preintegrated to its own
  last frame is a coin flip that the two callers happen to call differently.

The estimator was on the lucky side of that tie, which is why the M2-M4 results are
what they are; the probe was on the unlucky one, where a moving platform is quietly
demoted to the static initializer. `Decide()` now waits for IMU coverage before
solving -- one more message, <=5 ms at 200 Hz -- so the decision no longer depends
on the order at all. `InitDetectTest.DispatchDoesNotDependOnCoincidentMessageOrder`
runs the dispatcher over both orders and fails with exactly `window build failed`
if the check is removed.

Two things follow. First, the fix is **bit-identical on EuRoC**: three stored
members re-run afterwards -- V1_01 stereo t=0 (static), MH_01 stereo t=0 (dynamic),
V2_03 mono t=55 (dynamic where `off` diverges) -- reproduce their trajectories
byte-for-byte, so every number above stands. Second, the probe and the filter now
agree to the digit (MH_01: median 0.329 px, `|v|` 0.6667 m/s from both), which is
what makes the probe usable as a cost instrument at all.

## Tuning: two knobs mattered, and one of them mattered the other way

The plan allowed five: window length, the two detector thresholds, `sigma_pix`, the
LM budget. M3 had already settled the budget (30 iterations; 25 matches 200) and the
detector's margin is a factor of 21, so neither was touched. One configuration for
all eleven sequences, both modes.

**Window length: 41 frames, chosen on the worst case.** w31 and w41 are a wash on
the mid-flight mean (stereo -0.0209 vs -0.0156; mono -0.2418 vs -0.2575) and
identical on the census. They are not a wash on the tail: w31's worst surviving
stereo run is V2_03 at **5.56 m** against w41's 0.16 m, and its worst mono run is
0.87 m against 0.58 m. 2.0 s of window instead of 1.5 s buys a tail.

**`sigma_pix`: left at 1.0, and the reason is the interesting part.** It is really
the vision-to-IMU weight ratio, and the seed sweep said smaller is better
mid-flight, nearly monotonically -- MH_01's velocity error falls 0.410 -> 0.136 m/s
and MH_02's gyro-bias error 24x as it goes 1.0 -> 0.125. It also drops MH_05's
reprojection median from 1.649 px to 1.247 (0.5) and 1.090 (0.25), i.e. back under
the acceptance gate, which looked like it fixed the one sequence the gate rejects.

End-to-end it does the opposite. n=3 `on` arms against the same n=10 `off` control:

| | stereo mean | stereo worst | mono mean | mono worst |
|---|---|---|---|---|
| sigma_pix 1.0 (ships) | -0.0138 | +0.0044 | **-0.2443** | **+0.0085** |
| sigma_pix 0.25 | -0.0161 | +0.0119 | -0.1964 | **+0.3682** |
| sigma_pix 0.125 | -0.0175 | +0.0047 | -0.1669 | **+0.5700** |

Mono MH_05 goes 0.5738 -> 0.9420 (0.25) and 1.1438 (0.125): pulling its median
under the gate makes the filter *accept* a seed that is worse than the static
fallback it displaces, and two independent screens agree on it. The gate was doing
its job at 1.0. At t=0 the knob is irrelevant either way -- the null control is
+0.0000 (stereo) / -0.0049 (mono) at 0.25, the same as at 1.0.

The general lesson, and it was flagged as a caveat before it became a result: the
reprojection median is both the tuning signal and the acceptance gate, so a knob
that improves the signal can defeat the gate. Decisions from the seed sweep have to
be confirmed on end-to-end ATE, which is what this table is.

## Cost: one-off compute, and 8 MB that is not the solver's

`harness/m5_cost.sh` measures cost two ways, because the two questions are
different. `bin/init_probe -dispatch` runs the shipped dispatcher over real data
and times it directly (n=5, one core); `run_xivo_reference.sh --timing` runs the
whole filter on one pinned core, serial, `-mode runOnly`, and reports end-to-end
throughput and peak RSS.

**What it costs, from the probe** (ms, one core; `buffer` is the two KLTs inside
`AddImage`, `solve` is the window build plus Stage A plus Stage B):

| | images | buffer | solve | total |
|---|---|---|---|---|
| static verdict (9 seqs, t=0) | 20-21 | 231-308 | 0 | **231-308** |
| dynamic (MH_01/02, t=0) | 41 | 359-389 | 499-590 | **888-965** |
| dynamic (11 seqs, t=55) | 32-41 | 301-429 | 296-1298 | **724-1634** |

So the honest headline is **~0.9 s of one-core compute on a dynamic start and
~0.26 s on a static one**, not the ~50 ms an earlier note guessed. It is one-off:
`decided_` short-circuits every entry point afterwards. It is also paid while the
filter is held back and the messages are buffered rather than dropped, so on a
20 Hz stream it fits inside real time -- but the handoff is a latency spike, and
first-pose latency is ~1.0 s of data on the static path (the detector's 20 frames)
against ~2.0 s plus the solve on the dynamic one.

**What it costs the run, from the timing pass** (one core, serial, mean over 11
sequences):

| | off | on | delta |
|---|---|---|---|
| stereo `fps_wall` | 86.7 | 86.0 | **-0.8%** |
| mono `fps_wall` | 157.8 | 154.5 | **-2.1%** |
| stereo `peak_rss_mb` | 96.9 | 106.2 | **+9.3 MB (+9.6%)** |
| mono `peak_rss_mb` | 85.2 | 93.0 | **+7.8 MB (+9.2%)** |

`fps_mean` and the `update_*_ms` columns are not produced for XIVO by this harness
at all -- they come from OpenVINS' own per-update timers -- so "one-off or per
frame?" cannot be read off a second throughput column. It can be read off the
residual: subtract the probe's one-off from each sequence's wall-clock delta and
what is left is the per-frame cost. It is **zero**, mean residual -0.12 s (stereo)
and -0.02 s (mono) over 11 sequences, with per-sequence scatter of +-0.3 s -- the
same +-0.3 s two runs of the *identical* config differ by (stereo MH_03: 32.57 s
and 32.26 s). Mono's -2.1% is larger than stereo's -0.8% for the same reason: the
constant is the same and mono's runs are half as long.

The **memory** is the interesting half, because it is not the bundle adjustment's.
MH_03 at t=0 takes the static path and never builds a problem, and it still pays
+7.1 MB (stereo) / +6.1 MB (mono). A third arm settles it: `max_wait_sec = 0.0`
constructs the dispatcher, diverts messages, and gives up on the first one, so the
divert path is live and **zero images are tracked**.

| mono | off | on | max_wait_sec=0 |
|---|---|---|---|
| MH_01 peak RSS | 87.8 | 96.4 | **88.1** |
| MH_03 peak RSS | 84.2 | 91.8 | **84.4** |
| MH_01 wall s | 23.33 | 24.31 | **23.31** |

Both costs go away entirely. So the 8 MB is the **pre-init KLT** -- two trackers,
`goodFeaturesToTrack` and its pyramids, over 20-41 frames -- and neither the
dispatcher's existence, nor the buffered messages, nor the solver. XIVO's own
retained structures are bounded by construction and far too small to be it: 41
frames of at most 160 observations (0.26 MB), one 752x480 gray clone (0.36 MB), and
at most 600 IMU samples in each of two buffers (0.07 MB), under 1 MB in total, so
releasing the dispatcher after handoff could not return what is missing. Two
caveats stated rather than buried: peak RSS reproduces to only +-2.6 MB run to run
here (stereo MH_03 `off`: 100.4 and 97.8 MB), so the mean shift is real but the
per-sequence spread of +4 to +13 MB is mostly noise; and this is peak, so a
platform sized by high-water mark pays it even though the filter's steady state
does not.

## What ships

`dynamic_init.enabled` goes to **true** in `cfg/euroc_stereo.json` and
`cfg/euroc_mono.json`, with `window_frames: 41`, `sigma_pix: 1.0`,
`max_pixel_median: 1.5`, `max_wait_sec: 3.0` and the M2/M3 defaults elsewhere. One
configuration, both modes, all eleven sequences, no per-sequence tuning.

The case for turning it on is the census, not the mean: **69 of 220 mid-flight runs
diverge without it and 0 with it**, no run diverges that did not diverge before, and
at a table start the change is +0.0065 m against a null control whose own scatter is
0.006-0.009 m. The price is ~0.9 s of one-off compute on a dynamic start, ~0.26 s on
a static one, -0.8%/-2.1% of end-to-end throughput (all of it that one-off), and
+8 MB of peak RSS that belongs to the pre-init tracker.

What it does **not** do: it does not improve a table start, and it declines to try on
two of eleven mid-flight windows (MH_04 at 9.8 px of reprojection median, MH_05 at
1.65) where it falls back to the static path bit-identically. Both are the gate
working. And the mid-flight numbers come from a start-time device, not from a second
dataset -- `-start_sec 55` is EuRoC with a hard initial condition, which is the
closest thing to dynamic-start data that a public benchmark with groundtruth
provides.
