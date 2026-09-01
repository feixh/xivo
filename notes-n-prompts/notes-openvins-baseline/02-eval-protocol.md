# Scoring OpenVINS, and making it comparable to XIVO (2026-08-27)

## Groundtruth: there is only one file, under four names

All of these are byte-identical for room1 (same values, different formatting), so
there is no "which groundtruth" question to worry about:

| path | format |
|---|---|
| `data/tumvi/dataset-room1_512_16/mav0/mocap0/data.csv` | ASL csv, ns, `p,qw,qx,qy,qz` |
| `data/tumvi/dataset-room1_512_16/dso/gt_imu.csv` | same, different header |
| `experiments/open_vins/ov_data/tum_vi/dataset-room1_512_16.{csv,txt}` | OpenVINS' shipped copy |
| `data/reference/tumvi_room1_gt` | what XIVO's eval uses, TUM txt, 6 dp |

It is the **IMU-frame** pose (TUM-VI publishes mocap already transformed into the
IMU frame), which is also what OpenVINS estimates and what `run_euroc_folder`
writes. So no body-frame offset has to be absorbed by the alignment — good,
because a right-multiplied `T_imu_marker` is *not* absorbable by a global
alignment and would show up as centimetres of fake error.

The harness regenerates its own TUM copy from `mocap0/data.csv`
(`asl_gt_to_tum.awk`) keeping full nanosecond timestamps, because the shipped
`.txt` is rounded to 5 decimals and the association windows below are 1–20 ms.

## Two scorers, and they agree

* `evaluate_ate.py` (TUM RGB-D tool, from `xivo/scripts/tum_rgbd_benchmark_tools/`)
  — nearest-neighbour association within a window, Horn SE(3) alignment, RMSE of
  translation. **This is the scorer XIVO is measured with in this workspace**, so
  it is the one that makes the cross-system comparison legitimate.
* `ov_eval error_singlerun posyaw` — OpenVINS' own: interpolates the estimate onto
  groundtruth times, aligns position+yaw only (correct for a gravity-aligned,
  metric-scale VIO), and additionally reports orientation ATE and RPE.

Agreement over the 12 runs is 0.000–0.008 m (mean |diff| 0.002 m), e.g. stereo
room1 0.0784 vs 0.0790, mono room5 0.0758 vs 0.0830. Two independent
implementations, two different alignment groups, same answer — the trajectories
are being read and aligned correctly.

## The 0.001 s association window is unusable on OpenVINS output

`xivo/scripts/run_and_eval_pyxivo.py` scores ATE with `--max_difference 0.001`,
and RESULTS.md quotes XIVO at that window (see [[xivo-ate-eval-protocol]]: it
covers only ~26% of frames, in contiguous blocks, skipping the whole init phase).
On XIVO that is a *biased but stable* subsample. On OpenVINS it is neither:

| | room1 | room2 | room3 | room4 | room5 | room6 |
|---|---|---|---|---|---|---|
| pairs @ 0.02 (of ~2700 poses) | 2641 | 2503 | 2501 | 2107 | 2750 | 2490 |
| pairs @ 0.001 | **3** | 90 | 1006 | 1138 | 260 | 60 |
| ATE @ 0.02 (mono) | 0.0531 | 0.0791 | 0.0834 | 0.0348 | 0.0758 | 0.0461 |
| ATE @ 0.001 (mono) | 0.0040 | 0.0128 | 0.0605 | 0.0335 | 0.0123 | 0.0800 |

Why the pair count collapses: OpenVINS stamps each pose at
`t_cam + dt_CAMtoIMU`, where `dt_CAMtoIMU` is *estimated online* (it converges to
about −0.12 ms here and keeps moving). XIVO stamps poses at raw image times. So
for OpenVINS the estimate-to-groundtruth phase offset is an estimator output, and
a 1 ms window keeps a near-arbitrary 0.1%–42% of the trajectory. room1's
"0.0040 m" is the RMSE of **three poses**.

Consequence for the report: headline at 0.02 for both systems (98% coverage,
whole run, including initialization where the largest errors live). The 0.001
column is kept in `summary.csv` and shown once, labelled as not usable, so nobody
later compares it against RESULTS.md's XIVO column by accident.

## Comparing against XIVO fairly

XIVO's stored best-config trajectories (`results/final/triangulation_configs/sweep_dlt_nodesc/tumvi_room*_cam0`,
mono+IMU) are re-scored with the *same* `evaluate_ate.py`, the same window, and
the same regenerated groundtruth — `experiments/openvins/score_xivo_reference.sh`.
That removes evaluator, window and groundtruth as explanations for any gap, and
leaves only the estimators. Caveats that remain and cannot be removed:

* XIVO's numbers come from an 8-member ensemble in RESULTS.md but the stored
  per-sequence trajectories are single runs, and XIVO's single-run ATE carries
  ±0.007 of chaotic noise ([[xivo-single-run-ate-is-noise]]). OpenVINS' own
  6-room mean carries ±0.001 (below). So the XIVO column is the noisier one.
* Both are "mono+IMU, authors' own config", but XIVO's config was tuned in this
  workspace over many sweeps while OpenVINS' TUM-VI config is upstream's as
  shipped. Neither was tuned *for this comparison*.
