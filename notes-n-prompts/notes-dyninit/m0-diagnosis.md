# M0 -- what XIVO believes at the instant it initializes

Reproduce with (from the repository root, venv on `PATH`):

```
python3 notes-n-prompts/notes-dyninit/harness/init_diag.py \
    --dataset euroc --root ../data/euroc --cfg cfg/euroc_stereo.json
python3 notes-n-prompts/notes-dyninit/harness/init_diag.py \
    --dataset tumvi --root ../data/tumvi --cfg cfg/tumvi_stereo.json
```

The script replays `InertialMeasInternal`'s gravity-init path sample by sample --
the gyro integrated across every sample accepted or not, the
`| |a| - |g| | > gravity_init_max_accel_dev` rejection with its
`gravity_init_max_skip` safety valve, and the `gravity_init_counter` trigger --
then asks the ground truth what was true at the instant it fired.

## EuRoC MAV, `cfg/euroc_stereo.json` (counter 20, max_accel_dev 0.1)

| sequence | t_init | skipped | seen | \|v\|@init | tilt_deg | a_sd_min | w_mag | verdict |
|---|---|---|---|---|---|---|---|---|
| MH_01_easy | 2.100 | 401 | 421 | **0.671** | 0.576 | 0.697 | 0.2372 | **MOVING** |
| MH_02_easy | 1.020 | 185 | 205 | **0.481** | 1.941 | 0.497 | 0.1518 | **MOVING** |
| MH_03_medium | 0.195 | 20 | 40 | 0.006 | 0.932 | 0.202 | 0.0835 | static |
| MH_04_difficult | 0.170 | 15 | 35 | 0.007 | 0.923 | 0.138 | 0.0782 | static |
| MH_05_difficult | 0.105 | 2 | 22 | 0.008 | 0.657 | 0.058 | 0.0803 | static |
| V1_01_easy | 0.095 | 0 | 20 | 0.017 | 2.702 | 0.188 | 0.0803 | static |
| V1_02_medium | 0.095 | 0 | 20 | 0.011 | 0.473 | 0.048 | 0.0801 | static |
| V1_03_difficult | 0.095 | 0 | 20 | 0.015 | 0.801 | 0.039 | 0.0786 | static |
| V2_01_easy | 0.265 | 34 | 54 | 0.034 | 1.681 | 0.223 | 0.0933 | static |
| V2_02_medium | 0.240 | 29 | 49 | 0.031 | 0.584 | 0.215 | 0.0870 | static |
| V2_03_difficult | 0.505 | 82 | 102 | 0.022 | 1.020 | 0.258 | 0.1039 | static |

**Two of eleven sequences are moving when XIVO asserts `Vsb = 0`**, at 0.671 and
0.481 m/s. Both are Machine Hall; MH_01 is also the only MH sequence XIVO loses to
OpenVINS on (`ate_002` 0.080 `fast` vs 0.073) and the one where mono `acc`
diverges 10 runs of 10.

## TUM-VI room1-6, `cfg/tumvi_stereo.json` (counter 20, no gate)

| sequence | t_init | \|v\|@init | tilt_deg | a_sd_min | w_mag | verdict |
|---|---|---|---|---|---|---|
| room1 | 0.095 | 0.012 | 0.236 | 0.105 | 0.1088 | static |
| room2 | 0.095 | 0.007 | 1.191 | 0.146 | 0.1732 | static |
| room3 | 0.095 | 0.027 | 1.394 | 0.093 | 0.2140 | static |
| room4 | 0.095 | 0.039 | 2.894 | 0.138 | 0.3188 | static |
| room5 | 0.095 | 0.015 | 0.489 | 0.211 | 0.2249 | static |
| room6 | 0.095 | 0.035 | 1.749 | 0.137 | 0.1987 | static |

**Zero of six.** So the entire TUM-VI round -- the one XIVO wins 13 of 14 metrics
on -- is a static-initialization result and is not at risk from this work. That
also makes room1-6 a 6-sequence negative control for M1's detector: any
classifier that calls one of them dynamic is wrong, and M4's regression can
demand bit-identical output there.

## Four things this measurement settles

**1. The gate cannot fix this, by construction.** `gravity_init_max_accel_dev`
compares `| |a| - |g| |` against a tolerance. Specific force is `R'(a - g)`, so at
constant velocity `a = 0` and the magnitude reads *exactly* `|g|`. The statistic
is a proxy for stillness that is blind to constant-velocity motion. What the gate
does on MH_01 is not detect motion -- it skips 401 samples waiting for a quiet
stretch, accepts 20 that happen to have `|a| ~ |g|`, and initializes anyway,
2.1 s in and at 0.67 m/s. Waiting made it *worse*: it delayed init by 2 s and
still got the velocity wrong.

**2. The window must be measured against IMU t0, not ground-truth t0.** EuRoC's
`state_groundtruth_estimate0` starts 0.9-2.4 s after the first IMU sample (the
cameras start within +-40 ms of it). An earlier version of this measurement read
GT velocity over "the first second of the ground-truth file" and produced a table
describing a time XIVO had already initialized past. Every row above is indexed
by seconds since the first IMU sample.

**3. Accelerometer variance at a fixed instant does not separate the classes.**
Measured at t=0.5 s, V1_01 reads 0.831 and V2_03 reads 1.093 -- both genuinely
static, both noisier than MH_02 is while moving at 0.48 m/s (0.499). Somebody is
picking the rig up. The **minimum over candidate windows** does separate, and it
is the only IMU statistic here that does:

| | moving (2) | static (15) | margin |
|---|---|---|---|
| `a_sd_min`, 0.5 s window, 5 s horizon | >= 0.497 | <= 0.258 | 1.93x |

A threshold at 0.35 sits 1.36x above the static maximum (V2_03, 0.258) and 1.42x
below the moving minimum (MH_02, 0.497). Usable, but thin -- 15 static samples
is not many, and nothing here rules out a static sequence noisier than V2_03. So
the detector must not rest on this cue alone.

**4. Tilt error is real but is not a motion signal.** The `tilt_deg` column is the
angle between the gravity direction XIVO derives and the truth. It reaches 2.70
deg on V1_01 and 2.89 deg on room4 -- both *static*. Initial attitude error is
mostly caused by rotation during the averaging window, which is what M4's
`gravity_init_derotate` addressed, and it is largely independent of whether the
rig is translating. Dynamic initialization is a fix for velocity and bias; it
should not be sold as a fix for tilt, and M5 will report tilt separately so the
two claims stay apart.

## What M1 has to do that M0 could not

No IMU statistic can see constant velocity, so the 1.93x margin above is the
best the accelerometer will ever offer and it is a margin on a proxy, not on the
quantity of interest. The detector needs an independent cue that responds to
*translation*, which means vision. M1 measures gyro-de-rotated residual optical
flow -- zero for a stationary camera at any depth and any rotation rate, unlike
raw pixel disparity, which is what OpenVINS thresholds and which conflates
translation with unknown scene depth. The rig is already turning at 0.08-0.32
rad/s (`w_mag` above) at init on all 17 sequences, so de-rotation is not a
refinement here; it is the difference between a signal and noise.
