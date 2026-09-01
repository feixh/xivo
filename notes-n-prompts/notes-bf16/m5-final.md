# M5 -- final evaluation

The deliverable arm, measured against the shipped baseline in both modes.

```json
"kernel_precision": {
  "joseph": "f64", "innovation": "f64", "gating": "bf16",
  "batch_gating": true, "covariance_form": "short"
}
```

Now the content of `cfg/tumvi_mono_ctl_oos.json` and `cfg/tumvi_stereo_oos.json`.
Those two files, stripped of comments, parse to dicts *identical* to the
`short_gbf16` arm configs the ensembles below were run with, so the shipped
configuration is the measured configuration. Confirmed end to end -- the shipped
configs run straight out of `cfg/` reproduce ensemble member 0 (the unperturbed
member) on room1 to every digit, in both modes:

| | ATE@0.001 | ATE@0.02 | RPE_tra | RPE_rot | poses |
| --- | --- | --- | --- | --- | --- |
| mono, `_ens` member 0 | 0.069831 | 0.084723 | 0.016564 | 0.527733 | 2771 |
| mono, `cfg/tumvi_mono_ctl_oos.json` | 0.069831 | 0.084723 | 0.016564 | 0.527733 | 2771 |
| stereo, `_ens` member 0 | 0.056459 | 0.066036 | 0.013754 | 0.526466 | 2771 |
| stereo, `cfg/tumvi_stereo_oos.json` | 0.056459 | 0.066036 | 0.013754 | 0.526466 | 2771 |

The stereo run's own counters match too (29351 seeded / 8270 no-match / 201
rejected), so this is the same trajectory and not just the same score.

## Accuracy -- 6 members x 6 rooms x 2 modes, 144 runs

`merge/ens.sh`, aggregated by `merge/enstab.py`. Members differ only by a
`k * 1e-6 m/s` perturbation of the initial velocity, six orders of magnitude
inside the config's own prior on it, so the sd is the harness's own noise.

| mode | arm | ATE@0.001 | ATE@0.02 | RPE_tra | RPE_rot | diverged |
| --- | --- | --- | --- | --- | --- | --- |
| mono | baseline | 0.0686 +- 0.0034 | 0.0852 | 0.0213 | 0.6203 | 0/36 |
| mono | `short` (fp64) | 0.0681 | 0.0863 | 0.0215 | 0.6207 | 0/36 |
| mono | **final** | 0.0704 | 0.0866 | 0.0215 | 0.6204 | 0/36 |
| | *Welch t vs baseline* | **+1.07** | +0.50 | +0.37 | +0.75 | |
| stereo | baseline | 0.0453 +- 0.0031 | 0.0591 | 0.0132 | 0.6215 | 0/36 |
| stereo | `short` (fp64) | 0.0448 | 0.0579 | 0.0132 | 0.6215 | 0/36 |
| stereo | **final** | 0.0437 | 0.0539 | 0.0133 | 0.6215 | 0/36 |
| | *Welch t vs baseline* | **-1.36** | -3.91 | +0.57 | +0.05 | |

Mono is +0.0018 m (0.5 sd), stereo -0.0017 m; opposite signs, neither
resolvable. The stereo ATE@0.02 t of -3.91 is nominally an improvement and is
not claimed as one -- it is the same chaotic reshuffling of accepted features,
sampled favourably. Rotational deltas are quoted from the stock
`evaluate_rpe.py` only because they are flat; anything non-flat would need
`evaluate_rpe_interp.py` (see `xivo-bf16/RESULTS_MERGE.md`).

Logs: `merge/logs/bf16_{base,mono_short,mono_short_gbf16,stereo_short,stereo_short_gbf16}*.log`
plus `bf16_base_{mono,stereo}.log`.

## FPS -- one core, interleaved arms, min over repeats

`notes-efficiency/harness/fps_batch.sh`, `-mode runOnly`, `setarch -R`,
`XIVO_RANDOM_SEED=0`, one thread per process. Wall clock includes PNG decode and
the Python feed loop, so it is the end-to-end rate a single-threaded consumer
would see, not an estimator-only figure. room1 is 2821 frames, room6 is 2636.

**The reported statistic is the minimum over repeats, not the mean.** This host
is shared and its load average went from 2 to 7346 during the sweep; interleaving
spreads a drift across the arms but does not remove it, so `fps_batch.sh` now
records the per-run load average and the sweep was repeated behind a
wait-for-quiet gate. Contaminated runs are visibly slower on every arm and the
minimum is the only estimator that is not a function of who else was on the box.

90 runs: 50 mono, 40 stereo, five arms x two sequences x five sweeps.

| mode | arm | what changed | room1 FPS | room6 FPS | speedup |
| --- | --- | --- | --- | --- | --- |
| mono | base | shipped `auto` | 21.0 | 20.6 | 1.00x |
| mono | bf16base | + the M3 refactor at fp64 (`H P` computed once) | 22.2 | 21.7 | 1.05-1.06x |
| mono | batch | + gating sweep as one `J P` product | 25.8 | 25.5 | 1.22-1.24x |
| mono | short | + `P -= sym(K (H P))` | 41.8 | 41.5 | 1.99-2.02x |
| mono | **final** | + bf16 on the gating sweep | **44.2** | **43.6** | **2.10-2.12x** |
| stereo | base | shipped `auto` | 12.6 | 12.3 | 1.00x |
| stereo | bf16base | + the M3 refactor at fp64 | 13.3 | 13.0 | 1.06x |
| stereo | batch | + gating sweep as one `J P` product | 16.0 | 15.7 | 1.28x |
| stereo | short | + `P -= sym(K (H P))` | 21.8 | 21.4 | 1.74-1.75x |
| stereo | **final** | + bf16 on the gating sweep | **23.1** | **22.7** | **1.84-1.85x** |

Both exceed the plan's targets (1.8x mono, 1.6x stereo). The rows are cumulative,
so each one's contribution is the ratio to the row above it: the refactor
1.05-1.06x, blocking the gating sweep 1.16x/1.20x, the short form 1.62x/1.36x,
bf16 1.06x.

**Stereo crosses the camera rate.** RESULTS_STEREO.md's headline was 11.8 FPS
against a 20 Hz camera -- 0.6x real time on one core. The final arm is 23.1 FPS,
**1.15x real time**, at the same 90-feature / 45-group capacity and the same ATE.

XIVO's own per-component timers, room1, ms per frame:

| | mono base | mono final | stereo base | stereo final |
| --- | --- | --- | --- | --- |
| `MH-gating` | 8.66 | **1.12** | 8.70 | **1.18** |
| `stereo-gating` | -- | -- | 8.33 | **0.95** |
| `actual-update` | 23.13 | **5.85** | 35.53 | **14.24** |
| `track` | 4.08 | 4.12 | ~4.4 | ~4.1 |

The three dense-algebra timers fall by 7.7x, 8.8x and 2.5-4.0x; `track` is
untouched, which is the check that nothing outside the update moved. Stereo's
`actual-update` falls less than mono's (2.5x vs 4.0x) because the surviving
`K (H P)` product is `n x m x n` and stereo roughly doubles m -- the short form's
win shrinks as the measurement count grows, while the Joseph form's two `n^3`
products, which it deletes, do not depend on m at all.

Raw: `sweeps/fps_{mono,stereo}.log` (first sweep, R=2, contaminated in places)
and `sweeps/fps_{mono,stereo}2.log` (quiet-gated, R=3, per-run load recorded --
every mono run and 25 of 30 stereo runs at load < 7).

## Tests

`ctest` 19/19 (`unitTests_bf16_gemm` is the 13-case addition, `ctest -R
BF16Gemm`). No test was disabled or relaxed. The strongest correctness evidence
is not in ctest, though: the `f64` and `batch` arms are *exactly bit-identical*
to the shipped trajectory in both modes at `XIVO_DUMP_PRECISE=1`, which
exercises the riskiest new code -- the stereo gather predicate and its row cursor
in `GateStereoMeasurements` -- against the code it replaced.

## What is left off

* fp32 anywhere in the covariance or gain path. It survives every single run and
  the whole stereo ensemble, and loses mono room3 during initialization in 2 of 6
  members. See `m4-short-form.md`.
* bf16 in the covariance or gain path. Diverges by orders of magnitude in every
  arrangement tried, for a measured reason (`m3-kernels.md`, `m4-short-form.md`).
* `covariance_form: short` is not the code default -- `joseph` is. The short form
  is only equivalent for the *optimal* gain, and the Joseph form is what keeps P
  positive semi-definite if a future change makes the gain suboptimal (a damped
  or clipped update, a robust reweighting). Turning it on is a per-config
  decision, and it is on in the two configs that were measured.
