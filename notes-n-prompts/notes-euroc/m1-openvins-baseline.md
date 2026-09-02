# M1 -- the OpenVINS baseline on EuRoC MAV

Milestone M1 of `notes-n-prompts/plan-euroc.md`.

The point of this milestone is to produce the number XIVO has to beat, and to
produce it *honestly*. That means giving OpenVINS the same courtesy the user's
brief demands of XIVO: **one configuration for all eleven sequences**, chosen to
be the best single configuration OpenVINS has, not the first one that ran. A
baseline that is quietly handicapped makes the whole comparison worthless, and
the handicap here was not hypothetical -- the shipped configuration diverges on
one of the eleven sequences, and it would have been easy to report that as
"OpenVINS fails on MH_04" and move on.

Harness: `run_openvins.sh --profile euroc_mav`, player
`experiments/ov_build/run_euroc_folder` (OpenVINS v2.7), evaluator
`experiments/ov_build_eval`. Results in `../results/euroc_ov_acc_dyn/`.


## 1. Getting an error bar out of a deterministic system

OpenVINS is deterministic: rerunning a sequence reproduces the trajectory bit
for bit, so `--repeats` measures nothing at all. XIVO, by contrast, has a
run-to-run spread of ~0.007 m from chaotic gating. Comparing a single OpenVINS
run against a single XIVO run would therefore compare a point against a
distribution, and most of the differences worth arguing about on EuRoC are
smaller than that spread.

So the ensemble is provoked deliberately: six members with
`--gravity_mag` perturbed in its **ninth significant digit**
(9.81 -> 9.810000005). At 1e-9 relative this is far below any physical
uncertainty in the gravity constant -- it changes no physics -- but it does
reshuffle the order in which floating-point gating decisions land, which is the
thing that actually varies. Member 0 is the unperturbed shipped value, so the
canonical run is always inside the ensemble rather than replaced by it.

That the perturbation works is visible in the results: `MH_02_easy` spreads
+-0.011 m across members while `MH_04_difficult` and `MH_05_difficult` come out
+-0.000. Some sequences are genuinely insensitive; the ensemble tells you which,
which is itself worth knowing before quoting a two-decimal difference.


## 2. MH_04_difficult diverged in 6 of 6 members, and it was not the harness

With the shipped EuRoC configuration, stereo:

| sequence | ATE (m), shipped config |
| --- | --- |
| MH_04_difficult | **diverged, 6/6** (ATE ~9349 m) |

The first thing to rule out was my own harness -- a wrong topic, a bad
timestamp offset, a truncated download. It was none of those: the same player,
the same folder and the same calibration track the other ten sequences to
0.05--0.36 m.

Reading `ov_init/src/init/InertialInitializer.cpp:79-159` settled it. The
initializer picks between a static and a dynamic path, and the static path is
chosen when the measured feature disparity over the init window is below
`init_max_disparity` (10.0 px shipped). I first suspected the
`num_features0 < 15` early-return was bypassing the disparity check; reading it
showed the opposite -- that branch *refuses* to initialise rather than falling
through, so it cannot be the loophole.

The actual mechanism is a genuine blind spot in the static/dynamic decision:

1. MH_04 opens with a take-off, a hover and a landing. Groundtruth `|v|`
   reaches **0.47 m/s** between 1.5 s and 11 s -- the platform is moving.
2. The Machine Hall scene is *far away*. Disparity is a function of translation
   over depth, so 0.47 m/s against a scene tens of metres deep produces less
   than 10 px of disparity.
3. So `is_still` comes out true, and the static initializer asserts **zero
   velocity** on a platform moving at 0.47 m/s.
4. The filter starts with a large velocity error it has no way to attribute, and
   never recovers.

The failure is not "MH_04 is hard". It is that a disparity threshold in pixels
is being used as a proxy for a velocity threshold in m/s, and the conversion
factor between them is scene depth, which the initializer does not know yet.
This is worth stating plainly because the same trap is waiting in XIVO: its own
gravity initialiser averages the first `gravity_init_counter` accelerometer
samples with **no stationarity gate at all** (see `notes-euroc/m4`).

### The fix, and why it is legitimate

`init_dyn_use 1` enables OpenVINS' own dynamic initializer -- a feature already
in the codebase, simply off by default. Two overrides were added to
`run_euroc_folder.cpp` so it could be set from the command line without editing
a config per sequence (`--init_dyn_use`, `--init_max_disparity`).

It is applied **uniformly to all eleven sequences**, which is what keeps it
inside the one-config-for-all rule. And the results show it really is uniform
rather than an MH_04 special case:

| sequence | shipped | `init_dyn_use 1` |
| --- | --- | --- |
| MH_01_easy | 0.113+-0.014 | **0.073+-0.000** |
| MH_02_easy | 0.122+-0.003 | **0.090+-0.011** |
| MH_03_medium | 0.121+-0.003 | **0.116+-0.003** |
| MH_04_difficult | **diverged 6/6** | **0.207+-0.000** |
| MH_05_difficult | 0.358+-0.000 | **0.190+-0.000** |
| V1_01_easy | 0.055+-0.003 | 0.055+-0.003 |
| V1_02_medium | 0.047+-0.002 | 0.047+-0.002 |
| V1_03_difficult | 0.059+-0.002 | 0.059+-0.002 |
| V2_01_easy | 0.054+-0.004 | 0.054+-0.004 |
| V2_02_medium | 0.049+-0.002 | 0.049+-0.002 |
| V2_03_difficult | 0.096+-0.010 | 0.096+-0.010 |

All five Machine Hall sequences improve; **all six Vicon sequences are
bit-identical**. That is not luck, and it is the check that the change is not
secretly a tuning knob: the V sequences begin genuinely stationary, so the
static path is still selected and the dynamic code never runs. The change only
touches the cases the shipped heuristic was getting wrong.

### The rejected alternative

The other obvious fix is to lower the disparity threshold so a hover no longer
reads as still. `--init_max_disparity 3.0` does fix MH_04 (0.161, better than
the dynamic initialiser's 0.207) -- and breaks two sequences that previously
worked:

| sequence | shipped | `disp 3.0` | `init_dyn_use 1` |
| --- | --- | --- | --- |
| MH_04_difficult | diverged | **0.161** | 0.207 |
| MH_02_easy | 0.124 | **diverged** | 0.096 |
| V2_03_difficult | 0.083 | **diverged** | 0.083 |
| MH_05_difficult | 0.358 | 0.574 | **0.190** |

Rejected. Trading one divergence for two is not progress, and it illustrates why
the screen has to include sequences that already work: had I only measured
MH_04, `disp3` would have looked like the better fix by 0.046 m.


## 3. The baseline

Six members, `--init_dyn_use 1`, one configuration for all eleven sequences,
**0 diverged runs** in either mode.

### ATE RMSE [m], `evaluate_ate.py`, 0.02 s association window

| sequence | stereo | mono |
| --- | --- | --- |
| MH_01_easy | 0.073+-0.000 | 0.132+-0.008 |
| MH_02_easy | 0.090+-0.011 | 0.124+-0.004 |
| MH_03_medium | 0.116+-0.003 | 0.141+-0.025 |
| MH_04_difficult | 0.207+-0.000 | 0.197+-0.000 |
| MH_05_difficult | 0.190+-0.000 | 0.426+-0.011 |
| V1_01_easy | 0.055+-0.003 | 0.058+-0.003 |
| V1_02_medium | 0.047+-0.002 | 0.067+-0.000 |
| V1_03_difficult | 0.059+-0.002 | 0.068+-0.003 |
| V2_01_easy | 0.054+-0.004 | 0.124+-0.014 |
| V2_02_medium | 0.049+-0.002 | 0.062+-0.001 |
| V2_03_difficult | 0.096+-0.010 | 0.195+-0.000 |
| **mean** | **0.094** | **0.145** |

### `ov_eval error_singlerun posyaw`, stereo

| sequence | ATE pos (m) | ATE ori (deg) | RPE8 pos (m) | RPE8 ori (deg) |
| --- | --- | --- | --- | --- |
| MH_01_easy | 0.073+-0.000 | 1.57+-0.00 | 0.083+-0.000 | 0.55+-0.00 |
| MH_02_easy | 0.098+-0.012 | 1.10+-0.19 | 0.082+-0.008 | 0.41+-0.08 |
| MH_03_medium | 0.117+-0.003 | 1.29+-0.08 | 0.158+-0.008 | 0.32+-0.02 |
| MH_04_difficult | 0.209+-0.000 | 1.23+-0.00 | 0.182+-0.000 | 0.37+-0.00 |
| MH_05_difficult | 0.193+-0.000 | 0.73+-0.00 | 0.133+-0.000 | 0.43+-0.00 |
| V1_01_easy | 0.055+-0.003 | **5.41+-0.01** | 0.246+-0.000 | **3.66+-0.02** |
| V1_02_medium | 0.047+-0.002 | 1.88+-0.00 | 0.111+-0.000 | 0.51+-0.02 |
| V1_03_difficult | 0.059+-0.002 | 2.38+-0.13 | 0.083+-0.001 | 0.82+-0.03 |
| V2_01_easy | 0.065+-0.004 | 1.12+-0.10 | 0.084+-0.011 | 0.64+-0.06 |
| V2_02_medium | 0.055+-0.002 | 1.24+-0.05 | 0.047+-0.001 | 1.21+-0.04 |
| V2_03_difficult | 0.097+-0.010 | 1.53+-0.08 | 0.075+-0.006 | 0.99+-0.06 |
| **mean** | **0.097** | **1.77** | **0.117** | **0.90** |

`V1_01_easy`'s orientation error is 3--5x every other sequence in *both*
systems (XIVO's stereo baseline gives 5.53 deg against OpenVINS' 5.41). Two
independent estimators do not share a defect that specific, so this is a
property of the sequence or of its groundtruth alignment, not of either filter.
It is reported and not chased.

### Two notes on what these numbers are not

* **`ate_002` is blind to a global rotation.** It aligns with a full similarity
  transform, so a constant tilt costs nothing; `ov_eval posyaw` aligns yaw only
  and charges roll/pitch in full. Both are reported for exactly that reason, and
  the `ov_ate_pos_m` column being consistently >= `ate_002` is the visible
  signature of it.
* **The FPS and RSS columns in `euroc_ov_acc_dyn` are not the efficiency
  result.** That ensemble ran 8 cores per run with several runs in parallel, to
  get the accuracy numbers quickly; it reports ~102 FPS stereo, which measures a
  loaded box, not the system. The quotable efficiency numbers come from the
  dedicated one-core pass (`--onecore`, `taskset -c 0`, all thread pools 1,
  serial, idle box, n=3), reported in M5/M6.


## 4. Reproduce

```bash
cd experiments/openvins
./ov_euroc_ens.sh                       # 12 runs x 11 sequences, ~2 h
python3 agg_ensemble.py \
  --arm ov_stereo ../results/euroc_ov_acc_dyn/stereo_m* \
  --arm ov_mono   ../results/euroc_ov_acc_dyn/mono_m*
```

The `run_euroc_folder.cpp` override patch is mirrored in
`notes-euroc/harness/` -- `experiments/` is not version controlled, so anything
needed to reproduce this lives in the repo under that directory.
