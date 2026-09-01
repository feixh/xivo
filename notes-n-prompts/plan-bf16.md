# Plan: bfloat16 as XIVO's working numerical type

Target branch `auto-bf16`, worktree `/home/ubuntu/workspace/auto-slam-engineer/xivo-bf16`
(created from `auto` @ `d13ec97`). Notes in `notes-n-prompts/notes-bf16/`,
final report in `notes-n-prompts/report-bf16.md`.

## What the requirement asks, and what the hardware allows

The goal is bf16 as `number_t` *without losing accuracy* and *with better FPS*.
Two measurements taken before writing this plan decide the shape of the work.

**1. What bf16 buys on this box.** AMD EPYC 9R14 (Zen 4) has `avx512_bf16`
(`vdpbf16ps`: 32 bf16 multiply-accumulates into 16 fp32 lanes) but no AMX.
Register-resident issue-throughput probe (`notes-bf16/bench/peak.cpp`):

| arithmetic | GFLOP/s (1 core) | vs fp64 |
|---|---|---|
| fp64 `vfmadd*pd` | 117.0 | 1.00x |
| fp32 `vfmadd*ps` | 234.2 | 2.00x |
| bf16 `vdpbf16ps` | 312.1 | **2.67x** |

So the compute ceiling is 2.67x over today's fp64, of which 2x is just fp32;
bf16 adds 1.33x on top. The other half of the win is width: a bf16 element is a
quarter of an fp64 one, and XIVO's covariance is streamed repeatedly per frame.

**2. Where XIVO spends a frame.** Shipped monocular config, capacity fixed at
`EKF_MAX_FEATURES=90 / EKF_MAX_GROUPS=45` (state size n = 23 + 6*45 + 3*90 =
**563**), TUM-VI room1, one core, `-O3 -march=native`: 135.4 s / 2818 frames =
48.0 ms per frame, of which

| component | ms/frame | what it is |
|---|---|---|
| `actual-update` | 23.1 | Joseph form: two dense n x n x n products |
| `MH-gating` | 9.0 | 90 x `J_i P J_i^T`, i.e. P streamed once per feature |
| `propagation` | 6.6 | 10 IMU samples x 0.66 ms |
| `track` | 4.1 | KLT front end |
| rest | ~5 | PNG decode, Python feed, canvas |

**Two thirds of the frame is dense linear algebra on one 563 x 563 covariance in
fp64.** That is the object of this work.

**3. What the three kernels do in each precision** (`notes-bf16/bench/gemm_shapes.cpp`,
n=563, m=96, 90 features; bf16 = own `vdpbf16ps` microkernel, fp32 accumulation;
error is relative Frobenius against fp64):

| kernel | fp64 | fp32 | bf16 | bf16 vs fp64 | bf16 rel.err |
|---|---|---|---|---|---|
| `A P A^T` (Joseph) | 13.69 ms | 6.73 ms | **4.00 ms** | 3.4x | 4.1e-3 |
| `H P H^T` (innovation) | 1.37 ms | 0.71 ms | **0.50 ms** | 2.7x | 2.5e-3 |
| 90 x `J P J^T` (gating) | 10.27 ms | 6.67 ms | **1.87 ms** | 5.5x | 1.3e-4 |

The gating sweep is bandwidth bound, so it gains most: P packed once per frame as
bf16 is 0.63 MB and fits this core's 1 MB L2, where the fp64 original (2.54 MB)
does not.

**Consequence for the design.** bf16 has an 8-bit significand: 0.4% relative
error per element. A covariance is a weighting matrix and tolerates that; a
*state* does not -- position integrated at 200 Hz with 0.4% per-step rounding, or
a timestamp difference held in bf16, destroys the estimate. And Eigen's own
`bfloat16` scalar type has no fast path on x86: it converts to fp32 per operation,
so a literal `using number_t = bfloat16` would be *slower* than fp64 as well as
much less accurate. The delivered design is therefore mixed precision with an
explicit contract:

* **bf16 is the arithmetic type of the covariance algebra** -- every product
  involving P is rounded to bf16 on input and accumulated in fp32 by
  `vdpbf16ps`. This is where the frame time is.
* **fp32 is the storage type** of P, the Jacobians and the state; **fp64 stays**
  where the quantity is integrated over time or differenced (timestamps, IMU
  pre-integration, SO3 renormalization) -- selected by measurement, not by
  assumption.
* `number_t` becomes a build-time choice (`double` | `float` | `bf16`) so the
  literal reading of the requirement is buildable and measurable, and so the
  intermediate fp32 rung is a separately evaluated arm rather than an assumption.

Anything that raises FPS by changing the *algorithm* (exploiting that `I - KH` is
a rank-m perturbation, or that `J_i` is structurally sparse) is deliberately out
of scope here: it would confound the precision measurement this branch is for.
Both are noted in the report as the next lever.

## Measurement protocol (fixed for every milestone)

* Capacity fixed at 90 features / 45 groups for every arm, mono and stereo. FPS
  is only comparable at equal capacity.
* Accuracy: 6-member ensembles x room1..room6, `merge/ens.sh` + `merge/enstab.py`,
  members perturbing `X.Vsb` by `k*1e-6` m/s. A single 6-room mean has sd ~0.007
  at ATE@0.001, so **no single-run comparison is admissible**. Report ATE@0.001
  and ATE@0.02, RPE_tra, RPE_rot, and RPE from `evaluate_rpe_interp.py` for any
  rotational claim.
* Efficiency: `notes-efficiency/harness/fps_one.sh` + `fps_batch.sh`
  (`-mode runOnly`, one thread, `setarch -R`, seed pinned), arms interleaved
  inside each repeat, on an otherwise quiet host. Report wall-clock FPS as the
  headline plus the estimator's own per-component means.
* Arms: `mono` = `cfg/tumvi_mono_ctl_oos.json`, `stereo` = `cfg/tumvi_stereo_oos.json`
  (the two shipped headline configs, OOS on).
* Baseline for every comparison is the `xivo` worktree at `d13ec97`, already
  built, i.e. this branch's own merge-base.
* `ctest` must stay at 18/18 binaries, and bit-identity is the preferred
  no-regression evidence wherever a change is supposed to be inert.

## Milestones

**M0 -- baseline and instruments.** Worktree + build; the three benchmarks above;
baseline FPS (mono, stereo) and baseline accuracy ensembles; a *precision
sensitivity harness* that rounds a chosen subsystem's quantities to bf16 while
computing in fp64, so the accuracy cost of bf16 in P / H / state / IMU can be
attributed separately before any kernel is written.
Exit: baseline table for both configs, and a per-subsystem precision budget.

**M1 -- precision plumbing.** `XIVO_NUMBER_T` build option; `common/alias.h`
split into storage-precision and accumulate-precision aliases; the two broken
`using number_t = number_t;` lines in `common/utils.h` fixed; `bf16.h` scalar
type. Build and run all three settings. Exit: fp32 and (literal) bf16 builds run
end to end; the literal-bf16 arm is measured and documented -- it is expected to
fail both criteria and that is the evidence for the mixed design.

**M2 -- fp32 storage.** `number_t = float` as an evaluated arm: full ensembles
mono+stereo, FPS, `ctest`. Any place fp32 is not enough gets a typed exception
(fp64) justified by a measured diff against the fp64 reference. Exit: FPS up
~1.5-1.7x end to end with ATE/RPE inside noise, or a documented reason a
quantity had to stay fp64.

**M3 -- bf16 kernels.** `common/bf16_gemm.h` (the `vdpbf16ps` microkernel,
already prototyped) wired into `UpdateJosephForm`, `MHGating`,
`GateStereoMeasurements`, `OOSGating`, and the OOS/loop-closure updates: P packed
to bf16 once per frame and reused by every gating sweep. Unit tests comparing
each kernel against an fp64 reference on covariances captured from a real run,
plus a determinism test. Exit: per-kernel timings matching the microbenchmark
within ~20%, and the accuracy arms run.

**M4 -- numerics and tuning.** Symmetrization, exact diagonal, PSD guard, and
error-compensated split bf16 (hi + lo) available per kernel; a config knob
(`bf16.enable`, per-kernel) so the precision assignment is data-driven. Choose
the final assignment from measured ATE/RPE, not from the error norms. Exit: the
chosen configuration has ATE/RPE statistically indistinguishable from baseline in
both modes.

**M5 -- final evaluation and report.** Baseline vs final, both modes, 6x6
ensembles, interleaved FPS batch on a quiet host, ctest, RESULTS update,
`report-bf16.md`. Exit criteria: FPS materially up (target >= 1.8x mono,
>= 1.6x stereo), every accuracy delta within its own noise, and each milestone a
commit on `auto-bf16`.

## Risks

* **bf16 P loses positive definiteness.** Mitigation: fp32 accumulation, explicit
  symmetrization, exact diagonal, and the split-bf16 fallback in M4; the Joseph
  form is already the numerically defensive arrangement.
* **Chaotic gating makes accuracy hard to read.** Mitigation: ensembles only,
  Welch t on every delta, and the interpolated RPE evaluator for rotation.
* **The win is smaller end to end than in the microbenchmark**, because packing
  and the non-covariance parts do not shrink. Mitigation: FPS is the headline
  metric and is measured every milestone, so this is visible early rather than at
  the end.
* **fp32 is where most of the compute win is** (2.00x of the 2.67x). This is
  reported honestly per rung -- fp64 -> fp32 -> bf16 -- rather than attributing
  the whole gain to bf16.
