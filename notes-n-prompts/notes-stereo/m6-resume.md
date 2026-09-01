# M6 — resume state (written 2026-08-22, mid-milestone)

> **Superseded 2026-08-22.** M6 and M7 are both done: HEAD is now `abd0ede` (M6)
> plus the M7 report commit, and the final numbers are in
> `notes-n-prompts/report-stereo.md`. The mid-milestone details below are kept as
> the record of how the capacity result was found; §Open and §Next steps are
> resolved and annotated at the bottom. `[[m6-capacity]]` has the final table.

Resume the conversation with, from `/home/ubuntu/workspace/auto-slam-engineer`:

    claude --resume 041e1899-ae18-4dc8-b258-ef1a268fab97

Transcript:
`/home/ubuntu/.claude/projects/-home-ubuntu-workspace-auto-slam-engineer/041e1899-ae18-4dc8-b258-ef1a268fab97.jsonl`

Everything below is reconstructable without the transcript.

## Where the work is

Branch `auto-stereo`, HEAD `b595e55` (M5). M6 is **uncommitted** in the work tree:

| file | change |
| --- | --- |
| `src/estimator.{h,cpp}` | gyro de-rotation in `InitializeGravity`, behind `gravity_init_derotate` (**default false**) |
| `src/test/unittest_gravity_init.cpp` | 5 tests, all passing |
| `src/CMakeLists.txt` | `unitTests_gravity_init` target |
| `CMakeLists.txt` | `EKF_MAX_FEATURES`, `EKF_MAX_GROUPS`, `XIVO_OUTPUT_SUFFIX` as cache variables |
| `scripts/pyxivo.py` | reads `XIVO_LIB` (must be env, not argparse — it is needed before `import pyxivo`) |
| `cfg/m6_*.json` | sweep configs, all deleted before the M6 commit |

`cfg/_ensemble/`, `cfg/tumvi_cam0_faithful.json`, `cfg/tumvi_cam0_ref*.json` belong to a
**different concurrent job** (`eed7d63b`) working in this same tree. Do not commit them.
That job also runs against the shared `lib/pyxivo*.so`, which is why the state-size
variants build into `lib_f60/`, `lib_f90/`, `lib_f120/`, `lib_f150/` instead:

    cmake -S . -B build_f90 -DEKF_MAX_FEATURES=90 -DEKF_MAX_GROUPS=45 \
      -DXIVO_OUTPUT_SUFFIX=_f90 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
      -DOpenCV_DIR=$W/dependencies/opencv_install/lib/cmake/opencv4 \
      -DPython3_EXECUTABLE=$W/dependencies/venv/bin/python \
      -DPYTHON_EXECUTABLE=$W/dependencies/venv/bin/python \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    make -C build_f90 -j40 pyxivo
    XIVO_LIB=lib_f90 ... scripts/pyxivo.py ...

`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` and the three explicit paths are all required; a
bare `cmake -S . -B dir` fails on thirdparty policies and then on `xfeatures2d.hpp`.

## The result that mattered: capacity, not tuning

`tracker_cfg.num_features_max` was **60** and `core.h`'s `kMaxFeature` was **30**. Every
number in `RESULTS.md` was produced with a filter holding 30 features. `memory.max_features`
is only the object pool and does *not* raise the EKF cap — that is why an early 200→400
arm was bit-identical on 5 of 6 rooms.

Headline ATE is `--max_difference 0.02`; `0.001` is reported alongside for README
comparability (see `[[xivo-ate-eval-protocol]]`).

```
arm         room1   room2   room3   room4   room5   room6   mATE001  mATE02  RPEtra  RPErot
M5 shipped  0.0793  0.0578  0.0976  0.0541  0.1144  0.0530   0.0760  0.1013  0.0201  0.6211
f30t120     0.0846  0.0630  0.0687  0.0547  0.0655  0.0325   0.0615  0.0802  0.0177  0.6218
f60t120     0.0448  0.0524  0.0664  0.0431  0.0682  0.0392   0.0523  0.0629  0.0155  0.6211
f90t180     0.0551  0.0435  0.0549  0.0434  0.0612  0.0278   0.0476  0.0575  0.0145  0.6206
```

`fNtM` = N EKF feature slots (N/2 groups) and M tracker features. **Exit criterion 1
(mean ATE < 0.06 m) is met at both protocols by f90t180.** README monocular baseline is
0.1209. f60 and f90 were bit-identical until the *tracker* cap was raised too — raise
both or neither.

Also carried into these arms, from earlier sweeps: `visual_meas_std` 1.5 → 0.75
(mATE02 0.1013 → 0.0841), `Qimu.gyro` at nominal, `memory` 400/150.

## Criterion 3 (RPE_rot < 0.5 deg) is floor-limited — decomposition of the 0.62

RPE_rot sat in 0.6206–0.6575 across **everything**: 16x `Qimu.gyro`, 4x `visual_meas_std`,
6x feature capacity (which cut ATE 37%), and a halved initial attitude error. Measured
breakdown, mean over the six rooms:

| term | deg | how measured |
| --- | --- | --- |
| GT association artifact | 0.31 | slerp GT to the estimate stamps instead of nearest-neighbour: 0.6289 → 0.5439 (`harness/rpe_assoc.py`) |
| mocap's own attitude noise | 0.28 | local-cubic residual, 0.08–0.19 deg/axis (`harness/mocap_noise.py`) |
| real estimator attitude error | ~0.46 | remainder in quadrature |

Check: sqrt(0.46^2 + 0.28^2 + 0.31^2) = 0.626 vs 0.62 observed. **~0.42 deg of the 0.50 deg
budget is noise in the reference**, leaving < 0.27 deg for the estimator — a 42% reduction
in real attitude error. Treat criterion 3 as probably unattainable and report the
decomposition rather than chasing it further.

Eliminated by measurement, not argument — do not re-litigate:

- **Gyro scale/misalignment (`Cg`).** Fitted against mocap on all six rooms: deviation from
  identity <= 0.3%, and the cross-room std (0.06–0.16%) equals the mean, so it is fit noise.
  Contributes ~0.06 deg. (`harness/gyro_calib.py`)
- **Propagation integrator.** RK4 / Prince-Dormand with proper `SO3::exp` and
  `X.Rsb.normalize()` in `ComposeMotion`. Not a coarse first-order scheme.
- **Initial attitude.** The rig is *moving* at gravity-init in all six rooms
  (|w| = 0.11–0.32 rad/s over the 20 samples `InitializeGravity` averages), and the shipped
  initializer's mean tilt error is 1.47 deg. De-rotating over 200 samples halves it to
  0.73 deg and changes ATE and RPE by nothing measurable — `X.Rsg`'s prior variance is 3.01,
  so the filter absorbs a 1.5 deg error easily. The code is kept, **off by default**, because
  the negative result is only credible with the code in the tree. (`harness/grav_init.py`)
- **Out-of-state / MSCKF update.** `use_OOS: true` hits
  `LOG(FATAL) << "MSCKF not implemented"` at `src/estimator.cpp:126`. Closed upstream.

## Open at the moment of writing — all resolved, see §Outcome

A detached 24-run batch (`harness/batch_f.sh`, log `sweeps/m6f.log`) was still running:
`f120t240`, `f150t300`, and two like-for-like **monocular controls** at f90 (`t180` and
`t60`). The monocular controls matter: raising capacity helps monocular too, so
`RESULTS.md`'s 0.1209 is *not* the right control for the f90t180 number. If `m6f.log` is
missing or short, re-run that script — it is idempotent.

## Next steps

1. Collect `m6f.log`; pick the final config (f90t180 unless 120/150 wins by more than the
   ~0.006 config-perturbation spread — size error bars from a neutral knob, never the RNG
   seed, see `[[xivo-tuning-noise-is-not-seed-noise]]`).
2. One last attitude-focused sweep: `Qimu.gyro_bias`, `tracker_cfg.use_prediction`.
3. Regenerate `cfg/tumvi_stereo.json` at the chosen capacity via
   `scripts/make_stereo_cfg.py`; decide whether to ship a non-default `EKF_MAX_FEATURES`
   (it is a build argument, so the README/`build_all.sh` need to say so).
4. Delete `cfg/m6_*.json`, run the full test suite from the repo root
   (`for t in bin/unitTests_*; do ./$t; done` — *not* `ctest` from `build/`; two failures,
   `NumericalLinearAlgebra.SlowAndFastGivensMatch` and
   `Triangulation.Angular_Reprojection_Error`, predate M0), commit M6, write
   `notes-n-prompts/report-stereo.md` (M7).

## Outcome (appended after M6/M7 landed)

Every numbered next step above was carried out:

1. The batch finished. `f120t240` scored 0.0485 and `f150t300` 0.0568, so **f90t180
   ships** — the curve is unimodal with its minimum there. The monocular controls
   at the same capacity scored 0.0792 (t180) and 0.1144 (t60), which is what the
   40%/61% claims in the report are measured against.
2. `Qimu.gyro_bias` ×0.1 and ×10 were both worse (0.0494 / 0.0503).
   `tracker_cfg.use_prediction` came out **bit-identical** — the key is never read
   anywhere in `src/`. It is a dead knob; do not sweep it again.
3. `cfg/tumvi_stereo.json` was edited in place rather than regenerated, so its
   comments survive; it now declares `require_ekf_max_features: 90` /
   `require_ekf_max_groups: 45`, which `CheckMemoryPools` enforces at startup.
   `EKF_MAX_FEATURES`/`EKF_MAX_GROUPS` default to 90/45 in the top-level
   `CMakeLists.txt`, so a plain `cmake -S . -B build` is correct and the README
   documents the two-places rule.
4. `cfg/m6_*.json` deleted; suite is 84 tests / 82 pass with only the two pre-M0
   failures. M6 is `abd0ede`; M7 added `notes-n-prompts/report-stereo.md` plus
   `xivo/RESULTS_STEREO.md`.

One thing found the hard way during the sweeps and worth carrying forward:
`memory.max_features` at 400 against a tracker cap of 240 **aborted mid-run**
(`Out of feature slots`) and the harness swallowed the message by piping stderr
into a grep. `CheckMemoryPools` and the 90%-occupancy warning exist because of
that; see `[[m6-memory-pools]]`. Also pin threads for batches
(`OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1`) — output is
bit-identical and each unpinned process otherwise spawns ~255 threads.

## Ephemeral vs durable

Harness and sweep logs are saved beside this file under `m6-artifacts/`. The scripts in
`m6-artifacts/harness/` **hard-code `/home/ubuntu/.claude/jobs/041e1899/tmp`**
(`one.sh`, `one_lib.sh`, `score_all.sh`, `batch_f.sh`) — repoint them before reuse. The
670 MB of trajectory dumps under that path are *not* saved; re-run to regenerate.
