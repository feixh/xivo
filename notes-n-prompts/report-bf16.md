# bfloat16 in XIVO: report

Requirement: [`requirements-bf16.md`](requirements-bf16.md) -- support bf16 as the
main numerical type without degrading ATE on the six TUM-VI sequences, while
improving FPS, at fixed state capacity, in mono and stereo.
Plan: [`plan-bf16.md`](plan-bf16.md). Detailed notes: [`notes-bf16/`](notes-bf16/).
Branch: `auto-bf16` in `xivo-bf16/`.

## Outcome

**2.10-2.12x mono and 1.84-1.85x stereo end-to-end FPS, with ATE and RPE inside
the measurement's own noise in both modes.** Stereo crosses from 0.63x to 1.15x
real time on one core at unchanged capacity.

**bf16 is not the main numerical type, and cannot be.** It carries the gating
sweeps and nothing else. The speedup is almost entirely a reformulation of the
covariance update that the bf16 investigation surfaced; bf16 itself contributes
the last 1.06x. This is the honest headline and it is the opposite of what the
plan assumed, so the reasoning is set out in full below.

| | mono | stereo |
| --- | --- | --- |
| FPS, one core (room1 / room6) | 21.0 / 20.6 -> **44.2 / 43.6** | 12.6 / 12.3 -> **23.1 / 22.7** |
| vs the 20 Hz camera | 1.05x -> **2.21x** real time | 0.63x -> **1.15x** real time |
| ATE@0.001, 6x6 ensemble | 0.0686 -> 0.0704 (Welch t **+1.07**) | 0.0453 -> **0.0437** (t **-1.36**) |
| ATE@0.02 | 0.0852 -> 0.0866 (t +0.50) | 0.0591 -> 0.0539 (t -3.91) |
| RPE_tra / RPE_rot | +0.0002 (t +0.37) / +0.0001 (t +0.75) | +0.0000 (t +0.57) / +0.0000 (t +0.05) |
| diverged runs | 0 / 36 | 0 / 36 |

Both accuracy deltas are under 0.6 sd of the ensemble spread, and they have
opposite signs -- which is what "no degradation" looks like when the measurement
is honest about its own noise. Exit criteria met: FPS materially up (targets were
1.8x / 1.6x), no accuracy metric degraded.

## Why bf16 cannot be the main type

Two independent findings, both measured rather than argued.

**1. On this hardware bf16 is only 1.33x fp32.** AMD EPYC 9R14 (Zen 4) has
AVX512-BF16 `vdpbf16ps` but no AMX. Measured issue throughput on one core: fp64
117.0, fp32 234.2, bf16 312.1 GFLOP/s. So bf16's *ceiling* over the fp64
baseline is 2.67x on a pure-GEMM workload, and any error-compensation scheme that
costs two bf16 products (split hi+lo, iterative refinement) is strictly dominated
by one fp32 product -- fewer significand bits at higher cost. On hardware where
bf16 is 8x fp32, that conclusion flips. ([`notes-bf16/m0-baseline.md`](notes-bf16/m0-baseline.md))

**2. An EKF covariance update is a small difference of large numbers.** This is
the load-bearing finding. Instrumenting the running filter
(`XIVO_DIAG_UPDATE=1`):

* One update moves the covariance by `|dP|/|P| =` 6.5e-5 .. 2.2e-1, typically
  ~5e-3.
* A bf16 matrix product at the filter's shapes is wrong by ~3e-3 in norm.

**The arithmetic error is the size of the information.** Writing `P <- A P A^T`
in bf16 does not make the filter less accurate; it makes the filter never hold a
feature -- 113420 frames with zero in-state features against 2530 at fp64.

Narrowing the *correction* instead of the covariance reduces the injected error
~200x and is still not enough, for a reason the norm error hides. Recomputing
`K (H P)` at every precision inside the filter:

| | Frobenius relative error | **worst elementwise** relative error |
| --- | --- | --- |
| f32 | 3e-8 .. 4e-7 | 1e-5 .. 4e-2 |
| bf16 | 8e-4 .. 3e-3 | 1.8 .. **660** |

A product's error scales to the *matrix norm*, while the covariance's small
entries are its well-converged states -- precisely the ones the filter relies on
being small. Individual entries of the bf16 correction are wrong by factors of
100 to 600. No diagonal or block scaling repairs this: scaling the inner index
leaves every partial product unchanged, and scaling a row divides its terms
equally, so neither touches the cancellation that causes it. `mindiag(P)` is
exactly 0 at every frame (gauge-fixed states), so a "floor the diagonal" guard is
also unavailable.

**fp32 is below the floor too, in mono.** fp32 on the correction product passes
every single run and the entire stereo ensemble, then loses mono room3 in 2 of 6
ensemble members -- blowing up 0.8 to 3.3 s in, during initialization. Two
isolations pin it down: narrowing *only* `K (H P)` reproduces it in the same two
members, and perturbing the initial velocity by up to 5e-2 m/s at fp64 -- five
orders of magnitude more than the ensemble's own perturbation -- gives 12/12
clean runs. So it is the arithmetic, not chaos, and it is not salvageable without
an fp64 warmup whose threshold would be fitted to one sequence.
([`notes-bf16/m3-kernels.md`](notes-bf16/m3-kernels.md),
[`notes-bf16/m4-short-form.md`](notes-bf16/m4-short-form.md))

**Where bf16 does belong:** the MH, stereo and OOS gating sweeps. Their output is
a 2x2 (or small) matrix feeding a chi-square threshold -- a statistic that is
discarded at the end of the frame, with no dynamic range to lose. The rule the
measurements support is: **narrow what is discarded at the end of the frame; keep
what is integrated.**

## Where the speedup came from

Cumulative, room1, one core, minimum over five interleaved sweeps (90 runs):

| | mono | stereo | mechanism |
| --- | --- | --- | --- |
| shipped baseline | 1.00x | 1.00x | |
| + fp64 refactor | 1.05x | 1.06x | `H P` computed once and reused for `S`, the gain and the correction |
| + batched gating | 1.22x | 1.28x | 180 Jacobian rows as one `J P` product instead of 90 separate ones -- **bit-identical** |
| + short covariance form | 1.99x | 1.74x | `P -= sym(K (H P))` |
| + **bf16 gating** | **2.10x** | **1.84x** | the only precision change that ships |

The two large wins are algebraic and carry no numerical risk:

**Blocking the gating sweep** takes it from 9.88 to 2.36 ms at fp64. Ninety
separate 2xn products repack P ninety times; one 180xn product packs it once.
Eigen's k-panel blocking depends on k and the cache, not the row count, so each
output element accumulates in exactly the same order -- the trajectory is bit-identical.

**The short covariance form.** Given the optimal gain, the Joseph form's three
correction terms collapse to one: expanding
`(I-KH) P (I-KH)^T + K R K^T` gives `P - K(HP) - (HP)^T K^T + K S K^T` with
`S = H P H^T + R`, and `K = P H^T S^-1` makes `K S K^T = K (H P)`. XIVO's gain
comes from `S.ldlt().solve(HP)`, so the identity applies. That deletes `K H`,
both `n^3` products and `K R K^T` and adds one `n x m x n`: **1.33e8 multiply-adds
against 5.49e8** at n = 564, m = 180, a 4.1x flop reduction on the largest kernel
of the frame.

The Joseph form's extra robustness is what it costs 4.1x for, and here it buys
nothing measurable: `|joseph - short| / |dP|` is 5.5e-16 .. 2.6e-11, and end to
end the trajectory moves by **1.4e-11 m** (mono) / **3.3e-11 m** (stereo) over
2771 poses with no gating decision flipped. Its ensembles are t = -0.25 (mono)
and t = -0.30 (stereo). ([`notes-bf16/m4-short-form.md`](notes-bf16/m4-short-form.md))

Per-component, room1, ms/frame: `MH-gating` 8.66 -> 1.12, `stereo-gating`
8.33 -> 0.95, `actual-update` 23.13 -> 5.85 (mono) and 35.53 -> 14.24 (stereo),
with `track` unchanged at ~4.1 -- the check that nothing outside the update moved.

## What shipped

`auto-bf16` in `xivo-bf16/`, three commits:

| commit | milestone | content |
| --- | --- | --- |
| `60cd045` | M1-M2 | `number_t` as a build option; measurement of what narrowing it does |
| `58022c2` | M3 | `common/bf16_gemm.h` (AVX512-BF16 GEMM behind an Eigen-facing API), 13 unit tests, four filter call sites behind a per-kernel precision knob |
| `f8422f2` | M4 | the short covariance form, the `DIAGK` diagnostic, `XIVO_DUMP_PRECISE` |

Enabled in `cfg/tumvi_mono_ctl_oos.json` and `cfg/tumvi_stereo_oos.json`:

```json
"kernel_precision": {
  "joseph": "f64", "innovation": "f64", "gating": "bf16",
  "batch_gating": true, "covariance_form": "short"
}
```

Every knob defaults to the historical behaviour, so an unmodified config gets an
unmodified filter, and `f32`/`bf16` fall back to `f64` on a host without
AVX512-BF16 -- a config stays portable. The two configs above, stripped of
comments, parse identically to the arm configs the ensembles were run with, so
the shipped configuration *is* the measured configuration.

`ctest` is 19/19 with nothing disabled or relaxed.

## Limitations and what to do next

* **The result is hardware-specific in one direction only.** The algebraic wins
  (blocking, the short form) are flop reductions and hold everywhere. The bf16
  contribution is 1.05x because Zen 4 gives bf16 1.33x over fp32; on AMX or a
  tensor core it would be larger, and split-bf16 for the covariance -- ruled out
  here on cost -- would become worth re-testing.
* **The short form is not the code default.** It is only equivalent for the
  optimal gain. Any future change making the gain suboptimal (damping, clipping,
  robust reweighting) needs the Joseph form back, and it is one config key away.
* **The remaining `K (H P)` is the next target and it is m-bound.** Stereo's
  update improves 2.5x against mono's 4.0x precisely because it doubles m. The
  next real win is not precision but exploiting the structure of `H` -- it is
  block-sparse, two rows per feature touching one group and one feature block --
  which would cut the product itself rather than its arithmetic.
* **The measurement protocol is the reusable part.** Single runs on room1 ranked
  the fp32 arms as the best in the table, and 6x6 ensembles then rejected all of
  them. A single-sequence, single-run A/B on this system cannot distinguish a 2x
  speedup that is safe from one that loses a sequence.
