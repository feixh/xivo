# M1 -- publish the pose in the gravity-aligned frame

`experiments/results/orient_m1` vs `experiments/results/xivo_ref_jitter` (baseline `auto` @ 9e3ec06).
Both are 6-member `X.Vsb` jitter ensembles over TUM-VI room1-room6, mono and stereo.

## What was wrong

XIVO's error state carries `Wsg`, a two-degree-of-freedom gravity direction: gravity
in the spatial frame `S` is `Rsg * g_`, so `Rsg` maps the gravity-aligned frame `W`
(z along `-g_`) into `S`. `S` itself is *the body frame of the first IMU sample* --
`X_.Rsb` starts at identity by construction -- so `S` is tilted by whatever the rig's
attitude relative to gravity happened to be at startup.

Nothing ever applied `Rsg` to the published pose. `Estimator::gsb()` returned
`(Rsb, Tsb)` in `S`, and that is what `EvalModeSaver` wrote and what the viewer drew.
So every pose XIVO published was rolled/pitched by the rig's initial tilt.

That offset is not something an evaluator can remove. The standard VIO evaluation
aligns **yaw and position only** (`ov_eval error_singlerun posyaw`, and every other
benchmark, because yaw and position are the unobservable directions of a VIO while
roll and pitch are *observable* and therefore must not be aligned away). A roll/pitch
frame offset therefore lands in the reported orientation error undiminished.

Measured on the baseline runs, initial tilt of `S` from level against the mocap, and
the reported orientation ATE next to it:

| seq | initial tilt of `S` [deg] | baseline ori ATE, mono [deg] |
|---|---|---|
| room1 | 1.19 | 1.507 |
| room2 | **3.01** | **3.230** |
| room3 | 1.45 | 2.049 |
| room4 | 1.27 | 1.442 |
| room5 | 0.83 | 1.136 |
| room6 | 1.93 | 1.582 |

room2, the sequence the prompt flagged as "where the gap lives", is almost exactly its
own initial tilt. This was never attitude *drift*; it was a frame-convention bug.

OpenVINS has no such term: its global frame is gravity-aligned by construction
(`Propagator.h:57`, `_gravity << 0, 0, gravity_mag`), and its static initializer sets
`R_GtoI` from the measured gravity direction
(`ov_init/src/static/StaticInitializer.cpp:121-125`). The benchmark was comparing
XIVO's tilted frame against OpenVINS' level one.

## The fix

`Estimator::gwb()` (new, `src/estimator.h`): `p_w = Rsg' p_s`, i.e.

    Rws = X_.Rsg.inverse();
    return SE3{Rws * X_.Rsb, Rws * X_.Tsb};

Rotation and translation both -- anything else is not a pose in any frame. Gated on
`gravity_align_output`, which defaults **true** (no config edit needed; set it false to
get the old convention back bit-for-bit). Used by the pybind `gsb`/`gsc` bindings,
which is what the eval harness reads, and by `src/app/vio.cpp`'s text dump.

Two properties that make this safe:

* **The filter is untouched.** `Estimator::gsb()` is read by `manager.cpp`,
  `update.cpp` and `estimator.cpp` on the estimation path and keeps its old meaning;
  `gwb` is a new accessor used only on output paths. The estimate is bit-identical --
  confirmed by `ate_002`, which does a full SE(3) Horn alignment and so is blind to a
  global rotation: 0.0928 mono / 0.0636 stereo before and after, to four decimals.
* **`Rsg` carries no yaw.** `State::operator+=` zeroes the third component of the
  `Wsg` tangent update and reprojects every `kEnforceSO3Freq` steps, so `Rsg` is a pure
  levelling. It cannot rotate the trajectory about the vertical and so cannot launder
  yaw error into the alignment.

`Rsg` is a state, so this uses the filter's *current* estimate -- the causally
available one, no post-hoc pass. It converges well. Final `Rsg` against the mocap
gravity direction: room2 0.050 deg, room4 0.399 deg, where the 20-sample accel average
it starts from is off by 1.3 deg and 2.6 deg respectively.

## Numbers, 6-room means

| metric | mono before | mono after | stereo before | stereo after |
|---|---|---|---|---|
| **ov_ate_ori_deg** | 1.824 | **1.013** | 1.798 | **0.959** |
| ov_rpe8_ori_deg | 0.515 | 0.518 | 0.507 | 0.509 |
| ate_002 [m] | 0.0928 | 0.0928 | 0.0636 | 0.0636 |
| ov_ate_pos_m | 0.0968 | 0.0936 | 0.0688 | 0.0640 |
| ov_rpe8_pos_m | 0.0480 | 0.0480 | 0.0292 | 0.0292 |

Per-sequence orientation ATE:

| mode | room1 | room2 | room3 | room4 | room5 | room6 | mean |
|---|---|---|---|---|---|---|---|
| mono before | 1.507 | 3.230 | 2.049 | 1.442 | 1.136 | 1.582 | 1.824 |
| mono after | 1.225 | 0.920 | 1.423 | 0.841 | 0.955 | 0.713 | **1.013** |
| stereo before | 1.212 | 3.177 | 2.438 | 1.431 | 0.977 | 1.555 | 1.798 |
| stereo after | 0.872 | 0.658 | 2.015 | 0.824 | 0.778 | 0.606 | **0.959** |

Per-sequence ensemble sd of the orientation ATE is 0.06-0.24 deg, so the sd of a
6-room mean is about 0.07 deg. The -0.81 deg move is ~11 sigma. No run diverged
(`summary.csv`: 0 rows with pos ATE > 0.5 m or ori ATE > 10 deg).

## Costs and caveats

* `ov_rpe8_ori_deg` rises by 0.003 deg (mono) / 0.002 deg (stereo). `Rsg` is estimated,
  so applying `Rsg(t)` rather than a single constant warps the trajectory very slightly,
  and RPE over 8 m segments is exactly the statistic that sees a warp. It is 4% of the
  0.53 deg budget and both modes stay well inside it (0.518 / 0.509).
* `ov_ate_pos_m` and `ov_rpe8_pos_m` are the position agent's metrics and this moves
  `ov_ate_pos_m` (favourably, -0.003 / -0.005). Flagged for the merge.
* `src/estimator_process.cpp`'s publisher path still emits `gsb()` in `S`, deliberately:
  it publishes a 6x6 pose covariance alongside, which would have to be rotated with it,
  and nothing in this benchmark exercises it.
* After this, room3 is the worst sequence in both modes (mono 1.423, stereo 2.015) and
  is the only one where stereo is much worse than mono. That is a real estimator
  problem and it survived everything tried afterwards -- M2 left it at 1.402 / 2.012.
  Its measured gyro bias is 2.7e-3 rad/s, 22x what the filter can represent and 4-5x
  every other sequence's; see `negative-results.md`.
