# Comparing OpenVINS against XIVO fairly (2026-08-27)

The point of the whole harness is a cross-system number that survives scrutiny.
This is the protocol, and the two ways it can silently go wrong.

## The protocol

One runner per system, writing the **same directory layout**, so one scorer
covers both and there is no second code path to get wrong:

```
<out>/gt/<seq>.txt                 groundtruth, TUM format, ns precision
<out>/<mode>/<seq>_r<k>/traj.txt   estimate, TUM format
<out>/<mode>/<seq>_r<k>/stats.txt  frames_processed, wall_total_s, peak_rss_mb, ...
<out>/run_info.txt                 invocation + git describe of the system
<out>/summary.csv, summary.md      written by score_openvins.py
```

* `run_openvins.sh` → OpenVINS via `run_euroc_folder`.
* `run_xivo_reference.sh` → XIVO via `scripts/pyxivo.py -mode eval`, copying
  `dump/tumvi_<seq>_cam0` to `traj.txt` ([[xivo-trajectory-file-comparison]]).
* `score_openvins.py` → `evaluate_ate.py` @0.02 s and @0.001 s, plus
  `ov_eval error_singlerun posyaw` (position ATE, orientation ATE, RPE 8 m).

What that pins down: same groundtruth file, same evaluator, same association
window, same alignment, same host, same six sequences, both systems re-run at
HEAD. Accuracy runs go one cpu per run concurrently; throughput runs go
`--onecore` / `--timing`, which is one cpu, ASLR off, all thread pools at 1
(see [04-efficiency](04-efficiency.md)).

Both systems are scored as 6-member ensembles under a physically null
perturbation, never as single runs — see [03-determinism-and-noise](03-determinism-and-noise.md).

## Trap 1: the stored XIVO trajectories are stale

The obvious shortcut is to score the trajectories already on disk.
`score_xivo_reference.sh` does exactly that, over
`results/final/triangulation_configs/sweep_dlt_nodesc`, which is what `RESULTS.md`
publishes. It reproduces `RESULTS.md` exactly (@0.001 6-room mean **0.1209**),
which is how the evaluator wiring was validated — and it is **the wrong
comparator**:

| XIVO source | ATE@0.02, 6-room mean |
|---|---|
| stored `sweep_dlt_nodesc` (pre-M-series) | 0.1518 |
| re-run at HEAD `0476a98`, jitter ensemble | **0.0636** stereo / 0.0928 mono |

A factor of 2.4. Using the stored files would have made OpenVINS look like it
halves XIVO's error when in stereo it actually ties. Also note the shipped-config
schema drift that produced part of that old number
([[xivo-shipped-cfg-schema-drift]]).

**Rule: always re-run the comparator at HEAD.** `score_xivo_reference.sh` is kept
only for reproducing `RESULTS.md`.

## Trap 2: the 0.001 s association window

XIVO's own pipeline uses `--max_difference 0.001`, so it is the natural choice for
comparability — and it is unusable for OpenVINS. OpenVINS stamps poses at camera
time plus its *online-estimated* camera–IMU offset, which drifts a few ms, so a
1 ms window associates between **3 and 1138** of ~2700 poses. room1 mono scores
"0.0040 m" off three of 2689. Both windows are in `summary.csv`; only 0.02 s (~98%
coverage) is quotable. Details in [02-eval-protocol](02-eval-protocol.md).

This is a *different* failure from [[xivo-ate-eval-protocol]], which is about the
same window scoring XIVO's own frames in blocks and skipping its init phase.

## What the protocol still does not control

Worth restating whenever a number from here is quoted:

* **Tuning effort.** OpenVINS runs its authors' shipped `config/tum_vi` untouched;
  XIVO runs the shipped config on a codebase tuned in this workspace against
  these same six sequences. Favours XIVO.
* **State capacity.** OpenVINS 11 clones + 50 SLAM + 200 tracked vs XIVO 90
  features / 45 groups. Neither was re-tuned to the other's compute budget, so
  the FPS column is "what it costs each system to reach its own ATE", not a
  per-feature cost.
* **Availability.** OpenVINS emits nothing for the first 4–7 s (static-init
  detector); XIVO starts immediately. Invisible to ATE, visible to a robot.
* **One dataset, one motion profile.** Six handheld indoor mocap sequences.

## Adding a third system

Write a `run_<system>.sh` that produces the layout above and reuse
`score_openvins.py` unchanged — that is the only contract. Minimum it must emit
per run: `traj.txt` in TUM format (`t x y z qx qy qz qw`, seconds) and a
`stats.txt` with at least `frames_processed` and `wall_total_s` so `fps_wall` is
computable. `score_openvins.py` tolerates a run with `stats.txt` and no
`traj.txt`, which is what makes a `--timing` pass scorable.
