# XIVO vs OpenVINS on EuRoC MAV -- plan and milestones

Goal: run the same head-to-head we ran on TUM-VI, on the EuRoC MAV dataset, in
**stereo + IMU** mode, with **one XIVO configuration shared by all 11
sequences**, scoring both accuracy and runtime efficiency, and tune XIVO until it
matches or beats OpenVINS. Each milestone lands as a commit; feature work happens
on branches in separate `git worktree`s and is merged back to `auto` at the end.

Starting point: `auto` @ `cbd345f`. Reference: OpenVINS v2.7 (`v2.7-20-g6948812`),
already built ROS-free in `experiments/ov_build`, with `ov_eval` in
`experiments/ov_build_eval`.

## Why EuRoC is not just "TUM-VI with different files"

| | TUM-VI room1-6 | EuRoC MAV |
|---|---|---|
| camera model | equidistant fisheye, 512x512 | **pinhole + radtan**, 752x480 |
| sequences | 6, all in the mocap room | **11**, three environments (MH, V1, V2) |
| ground truth | `mav0/mocap0/data.csv`, room-only | `mav0/state_groundtruth_estimate0/data.csv`, whole sequence |
| motion | handheld, slow-ish | **MAV flight**, incl. fast/dark V1_03, V2_03 |
| stereo rate | 20 Hz | 20 Hz |
| IMU | 200 Hz | 200 Hz |

The two consequences that shape the plan: XIVO has no `euroc` dataset loader and
`savers.py` hardcodes the TUM-VI GT path, so M2 is real code work; and the
difficulty spread across 11 sequences is much wider than across room1-6, so a
single shared configuration is a genuine constraint rather than a formality.

## Milestones

### M0 -- data and harness plumbing
* Download EuRoC (ASL folder format) and extract to `data/euroc/<SEQ>/mav0/...`.
  The canonical host `robotics.ethz.ch` is unreachable from this box; using the
  HuggingFace mirror `GlowBond/EuRoC_MAV_Dataset` (24.7 GB, 3 zips).
* Verify the layout of all 11 sequences: `cam0`, `cam1`, `imu0`,
  `state_groundtruth_estimate0`, and image counts.
* Add `experiments/openvins/profiles/euroc_mav.sh` (the harness already has a
  profile mechanism and an EuRoC-aware ASL-GT-to-TUM converter).
* Smoke-run one sequence through OpenVINS.
* **Commit:** harness support for EuRoC.

### M1 -- OpenVINS EuRoC baseline (the target)
* Stereo+IMU, all 11 sequences, OpenVINS' own shipped `euroc_mav` config,
  unmodified. Scored with both `evaluate_ate.py` and `ov_eval error_singlerun
  posyaw` (ATE position/orientation, RPE at 8 m).
* Error bars from the `--gravity_mag` 9th-significant-digit perturbation
  ensemble, since OpenVINS repeats are byte-identical.
* One-core throughput and peak RSS under the established protocol: `taskset -c
  0`, ASLR off, all thread pools at 1, serial, idle box, n=3.
* **Commit:** OpenVINS EuRoC baseline results + notes.

### M2 -- XIVO EuRoC support (branch `auto-euroc`, worktree `xivo-euroc`)
* `scripts/pyxivo.py`: add a `euroc` dataset branch
  (`<root>/<seq>/mav0/cam{0,1}/data`, `mav0/imu0/data.csv`).
* `scripts/savers.py`: stop hardcoding `mav0/mocap0/data.csv`; resolve the GT
  file per dataset.
* A config generator that reads each sequence's `sensor.yaml` and emits **one**
  stereo config (pinhole-radtan intrinsics, `T_BS` extrinsics, IMU noise
  densities, gravity), with a check that the 11 sequences' calibrations agree
  closely enough for a single config to be honest.
* Smoke-run, keep `ctest` green.
* **Commit:** EuRoC dataset support in XIVO.

### M3 -- XIVO EuRoC baseline
* Run all 11 sequences stereo with a straight port of the tuned TUM-VI config,
  score accuracy and one-core efficiency. Expect this to be worse than OpenVINS
  in places: the TUM-VI tuning was done against fisheye handheld data.
* Catalogue failures (divergence, init failures, tracker starvation) per
  sequence. **Commit:** baseline results.

### M4 -- accuracy tuning (branch `auto-eurocacc`)
* Work the knobs that mattered on TUM-VI (feature budget, OOS window,
  triangulation, initialisation thresholds, CLAHE, measurement noise) plus
  anything EuRoC-specific the M3 failures point at.
* Hard constraint: **one config for all 11 sequences.** No per-sequence overrides.
* Score with jitter ensembles, not single runs -- single-run ATE on this filter
  is noise at the +-0.007 m level.
* Target: beat OpenVINS on ATE position, ATE orientation, RPE position, RPE
  orientation. **Commit** each accepted change.

### M5 -- efficiency tuning (branch `auto-eurocfps`)
* One-core FPS and peak RSS against OpenVINS on the same 11 sequences, at the M4
  accuracy config. Alternated A/B for any delta below ~0.05 ms/frame.
* Only exact or accuracy-neutral changes; anything that trades accuracy gets
  measured on both axes before it ships. **Commit** each accepted change.

### M6 -- final evaluation
* One config, 11 sequences, stereo+IMU, n>=3 replicates for both accuracy and
  efficiency, XIVO and OpenVINS measured in the same session.
  * **Revised during M6 to n=10.** n=3 is not enough on EuRoC: the shipped
    configuration re-run at n=6 and n=10, with members 0-2 bit-identical, moved
    from 0.098 to 0.102 to 0.103 m of `ate_002`. `V2_03_difficult` alone carries
    +-0.040 m of member spread. **Screen at n=3, ship at n=10** -- and M5's
    accuracy tables, all n=3, were corrected accordingly.
* Full metric table with error bars, plus a per-sequence breakdown.
* Done: `notes-euroc/m6-final-evaluation.md`.

### M7 -- documentation and merge
* `notes-n-prompts/report-xivo-vs-openvins-euroc.md`: what was done, every
  measurement, every negative result, the protocol, and the repro commands.
* Merge all EuRoC branches back into `auto`, re-verify on the merged tree
  (branch numbers do not compose -- that was the lesson from the TUM-VI round).
* Update `README.md` with what was done and the outcome.
* Done. The three branches are a linear chain, so they went in as three
  `--no-ff` merges (`faa494e`, `4c683bd`, `b562164`) and
  `git diff --stat auto-eurocfps HEAD` is empty. Re-verified on the merged tree
  rather than assumed: `make -j32` clean, `ctest` 23/23, and an n=3 stereo pass
  over all 11 sequences (`results/euroc_m7_merged`) matched members 0-2 of
  `euroc_m6_final/fast` in all 165 metric cells to the last printed digit
  (33-run `ate_002` mean 0.097856 on both sides). Report and README written.

## Protocol carried over from the TUM-VI round

These are settled and not up for re-litigation:

* Never compare single runs; use jitter ensembles and report the sd of the
  sequence-mean across members.
* Throughput: `fps_wall = frames/wall_total_s`, one core, ASLR off, pools at 1.
  Alternate arms `A B A B A B` within one session for small effects.
* Peak RSS from `/usr/bin/time -f %M`; stereo repeats to ~0.1 MB.
* `-mode runOnly` dumps no trajectory; use `-mode eval` when you need one.
* Score with both `evaluate_ate.py` (`--max_difference 0.02`) and `ov_eval
  ... posyaw`; the former is blind to a global rotation, the latter charges
  roll/pitch to orientation in full.
