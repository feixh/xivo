# M0 -- baseline and instruments

Everything below is at the fixed capacity this branch works at,
`EKF_MAX_FEATURES=90 / EKF_MAX_GROUPS=45`, i.e. an EKF state of
n = 24 + 6*45 + 3*90 = **564** (`kFullSize`). FPS is only comparable at equal
capacity, so no arm in this branch changes it.

Baseline is the `xivo` worktree at `d13ec97` (this branch's merge-base), already
built, `number_t = double`.

## Accuracy baseline

6 members x room1..room6 per mode, `merge/ens.sh` (members perturb `X.Vsb` by
`k*1e-6` m/s, k = 0..5), aggregated by `merge/enstab.py`. Raw logs
`merge/logs/bf16_base_{mono,stereo}.log`, table `logs/m0_ate_base.txt`.

| mode | ATE@0.001 | ATE@0.02 | RPE_tra | RPE_rot |
|---|---|---|---|---|
| mono (`cfg/tumvi_mono_ctl_oos.json`) | 0.0686 +- 0.0034 | 0.0852 +- 0.0051 | 0.0213 +- 0.0010 | 0.6203 +- 0.0003 |
| stereo (`cfg/tumvi_stereo_oos.json`) | 0.0453 +- 0.0024 | 0.0591 +- 0.0029 | 0.0132 +- 0.0000 | 0.6215 +- 0.0004 |

The `+-` is the sd across the six members, i.e. across runs of *the same
physical problem* differing by 1e-6 m/s in one initial-velocity component --
six orders of magnitude inside the config's own prior (`P.Vsb = 0.5`). It is the
floor below which a delta is not attributable to a code change, and it is large:
0.0034 on a 0.0686 mean is 5%. Any claim in this branch is read against it.

## FPS baseline

`notes-efficiency/harness/fps_batch.sh`, `-mode runOnly`, one thread
(`OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1`),
`setarch -R`, `XIVO_RANDOM_SEED=0`. Configs are the shipped headline ones with
`print_timing` on (`cfg/tumvi_mono_ctl_oos_timing.json`,
`cfg/tumvi_stereo_oos_timing.json`). Raw log `logs/m0_fps_base.log`.

| mode | seq | wall (s) | frames | ms/frame | FPS |
|---|---|---|---|---|---|
| mono | room1 | 135.37 | 2818 | 48.0 | 20.8 |
| mono | room6 | 130.86 | 2733 | 47.9 | 20.9 |
| stereo | room1 | 225.60 | 2818 | 80.1 | 12.5 |
| stereo | room6 | 218.65 | 2733 | 80.0 | 12.5 |

Estimator's own per-component means, ms/frame (room1):

| component | mono | stereo | what it is |
|---|---|---|---|
| `actual-update` | 23.1 | 35.2 | Joseph form: two dense n x n x n products |
| `MH-gating` | 9.0 | 9.1 | 90 x `J_i P J_i^T`, P streamed once per feature |
| `stereo-gating` | 0.0 | 8.5 | the same sweep on the right-camera rows |
| `propagation` | 0.66 | 0.66 | ~10 IMU samples per frame |
| `jacobian` | 0.10 | 0.18 | |
| `oos-jacobian` | 0.20 | 0.35 | |
| `track` | 4.1 | 11.8 | KLT front end |

**Dense algebra on the one 564 x 564 covariance is 32.1 of mono's 48.0 ms and
52.8 of stereo's 80.1 ms.** That is where this branch has to win, and it is why
the target is the covariance kernels rather than `number_t`.

## Hardware ceiling

AMD EPYC 9R14 (Zen 4): `avx512_bf16` (`vdpbf16ps`, 32 bf16 MACs into 16 fp32
lanes) and `avx512_vnni`, **no AMX**. Register-resident issue-throughput probe,
8 independent accumulator chains, one core (`bench/peak.cpp`):

| arithmetic | GFLOP/s | vs fp64 |
|---|---|---|
| fp64 `vfmadd*pd` | 117.0 | 1.00x |
| fp32 `vfmadd*ps` | 234.2 | 2.00x |
| bf16 `vdpbf16ps` | 312.1 | 2.67x |

Two thirds of the arithmetic ceiling is plain fp32; bf16 adds 1.33x on top of
it. The other half of the win is width -- a bf16 element is a quarter of an fp64
one, and P is streamed once per feature in the gating sweep, so packing P as
bf16 (0.63 MB) puts it inside this core's 1 MB L2 where the fp64 original
(2.54 MB) does not fit.

## Kernel-level measurement

`bench/gemm_shapes.cpp`, n = 563, m = 96 measurement rows, 90 features -- one
row short of the filter's 564 and at half its real measurement count, which
`bench/kernel_api.cpp` in M3 corrects. The bf16
column is the hand-written `vdpbf16ps` microkernel in `bench/bf16_gemm.h`
(inputs rounded to bf16, accumulation in fp32); error is relative Frobenius
against the fp64 result.

| kernel | fp64 | fp32 | bf16 | bf16 vs fp64 | bf16 rel.err |
|---|---|---|---|---|---|
| `A P A^T` (Joseph) | 13.69 ms | 6.73 ms | 4.00 ms | 3.4x | 4.1e-3 |
| `H P H^T` (innovation) | 1.37 ms | 0.71 ms | 0.50 ms | 2.7x | 2.5e-3 |
| 90 x `J P J^T` (gating) | 10.27 ms | 6.67 ms | 1.87 ms | 5.5x | 1.3e-4 |

The gating sweep gains most (5.5x) because it is bandwidth bound, not compute
bound -- consistent with the L2 argument above.

## Deviation from the plan

The plan's M0 also listed a *precision-sensitivity harness* -- round one
subsystem's quantities to bf16 while computing in fp64, to attribute the
accuracy cost per subsystem before any kernel is written. That step was dropped
in favour of going straight to the two end-to-end arms (M1's literal bf16, M2's
fp32) because those turned out to answer the same question faster and less
ambiguously: see `m1-m2-precision-arms.md`. The per-kernel form of the
instrument survives as M4's per-kernel precision knob, which is the version that
can actually pick the shipped configuration.
