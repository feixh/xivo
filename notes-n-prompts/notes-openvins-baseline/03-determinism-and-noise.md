# How much of an OpenVINS ATE difference is real? (2026-08-27)

## Repeats are byte-identical, so repeats measure nothing

`md5sum` over `traj.txt` for 3 repeats × 6 sequences × {mono, stereo}: **one
distinct hash per (mode, seq)**. Also identical across `--num_opencv_threads 4`
vs `1`, and across serial vs 4-cpu-pinned concurrent execution. So OpenVINS as
run here is fully deterministic — no RANSAC seed drift, no thread-order effects
in the estimator.

That is convenient but it is a trap: it means an error bar built from repeats is
exactly 0.000, which would license "0.0700 beats 0.0710" claims that will not
survive any change in the pipeline. XIVO has the opposite problem — its repeats
*do* differ (chaotic MH-gating candidate ordering, [[xivo-single-run-ate-is-noise]])
so its spread is visible for free.

## Use a physically null perturbation instead

Same trick as [[xivo-tuning-noise-is-not-seed-noise]]: perturb a knob that cannot
change the physics, and read off how much ATE moves anyway. Here the knob is
`--gravity_mag`, jittered in the **9th significant digit** (9.807660000000 →
9.807660049038, i.e. k × 1e-9 *relative* for k = 0..5 — orders of magnitude below
TUM-VI's own gravity uncertainty, and below fp32 resolution). 6 members, in
`experiments/results/ov_jitter/m{0..5}`:

```bash
for k in 0 1 2 3 4 5; do
  g=$(python3 -c "print('%.12f' % (9.80766 * (1 + $k * 1e-9)))")
  experiments/openvins/run_openvins.sh --out experiments/results/ov_jitter/m$k \
      --extra "--gravity_mag $g"
done
```

ATE RMSE @0.02 s, mean ± sd over the 6 members:

| mode | room1 | room2 | room3 | room4 | room5 | room6 | 6-room mean |
|---|---|---|---|---|---|---|---|
| stereo | 0.0749 ±0.0050 | 0.0991 ±0.0034 | 0.0750 ±0.0039 | 0.0339 ±0.0028 | 0.0947 ±0.0032 | 0.0303 ±0.0006 | **0.0680 ±0.0006** |
| mono | 0.0526 ±0.0028 | 0.0768 ±0.0025 | 0.0836 ±0.0067 | 0.0304 ±0.0057 | 0.0768 ±0.0011 | 0.0462 ±0.0049 | **0.0611 ±0.0013** |

Peak-to-peak on a single sequence reaches **0.016 m** (mono room3, mono room6),
i.e. ~20% of the value. The 6-room mean is 20–50× tighter (sd 0.0006–0.0013)
because the per-sequence wobbles are independent.

Mechanism: a 1e-9 relative change in `gravity_mag` changes the initialization solve in
the last bits, which changes which features pass chi-square gating a few frames
later, which changes the SLAM feature set, and from there the trajectories
diverge macroscopically. Deterministic, but not *stable*.

## The same measurement for XIVO, so the comparison is symmetric

XIVO at HEAD turns out to be *nearly* deterministic too, which was a surprise:
`XIVO_RANDOM_SEED` 0..5 gives **bit-identical mono** results on all six rooms, and
stereo differs only on room6 (sd 0.0106) and room4 (sd 0.0001). So a seed
ensemble understates its spread just as badly as OpenVINS repeats do
([[xivo-tuning-noise-is-not-seed-noise]]). `run_xivo_reference.sh --seeds 6`
(→ `experiments/results/xivo_ref_accuracy`) exists only as the evidence for that.

The neutral knob for XIVO is the initial velocity: `X.Vsb += k * 1e-6 m/s`, six
orders of magnitude inside the config's own declared prior of sqrt(0.5) ≈ 0.7 m/s
— the same device `run_ensemble_bugfix.sh` uses. `--jitter 6` generates the
member configs itself (`<out>/cfg/eff_<mode>_m<k>.json`):

```bash
experiments/openvins/run_xivo_reference.sh --out experiments/results/xivo_ref_jitter --jitter 6
```

ATE RMSE @0.02 s, mean ± sd over the 6 members:

| mode | room1 | room2 | room3 | room4 | room5 | room6 | 6-room mean |
|---|---|---|---|---|---|---|---|
| stereo | 0.0636 ±0.0102 | 0.0684 ±0.0095 | 0.0951 ±0.0048 | 0.0472 ±0.0100 | 0.0692 ±0.0100 | 0.0379 ±0.0052 | **0.0636 ±0.0045** |
| mono | 0.0762 ±0.0112 | 0.1001 ±0.0212 | 0.1343 ±0.0087 | 0.0805 ±0.0073 | 0.1065 ±0.0121 | 0.0594 ±0.0099 | **0.0928 ±0.0067** |

Two things to carry forward:

* **XIVO's spread is 5–8× OpenVINS'** on the 6-room mean (0.0045–0.0067 vs
  0.0006–0.0013) and 3× on single sequences (peak-to-peak up to 0.048 m on mono
  room2 vs 0.016 m for OpenVINS). Both filters are chaotic in their gating; the
  MSCKF's is evidently better damped here.
* **The jitter mean is not the seed mean.** XIVO mono goes 0.0888 (6 seeds, all
  identical) → 0.0928 ±0.0067 (6 jitter members); stereo 0.0599 → 0.0636 ±0.0045.
  The seed "ensemble" is a single sample dressed up as six, and it happens to sit
  ~0.004 low. Report the jitter ensemble.

## Rules this implies

* **Never compare single-sequence ATE across configurations.** Anything under
  ~0.015 m on one room is noise.
* **Compare 6-room means, and quote ±0.001.** mono 0.0611 vs stereo 0.0680 is a
  ~5 sd separation, so "mono beats stereo on TUM-VI rooms with this config" is a
  real effect (see report for why that is plausible).
* Cross-**system** comparisons carry both systems' spreads: OpenVINS ±0.001 and
  XIVO ±0.005–0.007 on the mean ([[xivo-single-run-ate-is-noise]]). A gap has to
  clear the larger one. That is why the report calls the mono comparison
  (0.061 vs 0.093) decisive and the stereo one (0.068 vs 0.064) a tie.
* Timing is *not* deterministic and must not be read off an accuracy run — see
  the throughput section of the report; use `--onecore`.
