# M0 -- EuRoC data and harness plumbing

## Getting the data

The canonical host is unreachable from this box: `robotics.ethz.ch` resolves
(129.132.38.186) but `curl` hangs at TCP connect on both 80 and 443.
`projects.asl.ethz.ch`, `huggingface.co`, `zenodo.org`, `github.com`, AWS S3 and
pypi all answer 200, so it is that one host, not the network.

Used the HuggingFace mirror `GlowBond/EuRoC_MAV_Dataset`:

```bash
base=https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset/resolve/main
for f in machine_hall.zip vicon_room1.zip vicon_room2.zip; do
  curl -sSL -C - -o "$f" "$base/$f" &
done; wait        # 24.7 GB, ~5 min on this box, resumable (server honours 206)
```

The three archives are **nested**: each holds, per sequence, both a ROS `.bag`
and the ASL-folder `.zip`. Only the inner zips are wanted -- extracting the bags
too would cost ~13 GB and nothing else:

```bash
for z in machine_hall vicon_room1 vicon_room2; do
  unzip -o -j -q "$z.zip" "$z/*/*.zip" -d inner
done
for z in inner/*.zip; do s=$(basename "$z" .zip)
  unzip -o -q "$z" 'mav0/*' -d "data/euroc/$s"; done
```

Result: `data/euroc/<SEQ>/mav0/{cam0,cam1,imu0,state_groundtruth_estimate0}/`,
20 GB, all 11 sequences.

## What the data looks like

| sequence | cam0 | cam1 | imu0 | gt poses |
|---|---|---|---|---|
| MH_01_easy | 3682 | 3682 | 36820 | 36382 |
| MH_02_easy | 3040 | 3040 | 30400 | 29993 |
| MH_03_medium | 2700 | 2700 | 27008 | 26302 |
| MH_04_difficult | 2033 | **2032** | 20320 | 19753 |
| MH_05_difficult | 2273 | 2273 | 22721 | 22212 |
| V1_01_easy | 2912 | 2912 | 29120 | 28712 |
| V1_02_medium | 1710 | **1711** | 17100 | 16702 |
| V1_03_difficult | 2149 | 2149 | 21500 | 20932 |
| V2_01_easy | 2280 | 2280 | 22800 | 22401 |
| V2_02_medium | 2348 | 2348 | 23490 | 23091 |
| V2_03_difficult | **1922** | 2336 | 23370 | 22970 |

Three sequences have unequal left/right counts. MH_04 and V1_02 are off by one;
**V2_03_difficult is missing 414 left images**, which is a known property of the
release, not a bad download (the inner zip's own listing has 1922 entries under
`cam0/data`). Both systems pair on timestamp and drop unmatched frames, so this
costs V2_03 about 18% of its stereo frames on both sides equally.

Groundtruth is `state_groundtruth_estimate0` -- the Leica/Vicon + IMU fusion,
covering the **whole** trajectory rather than a mocap volume, so unlike TUM-VI
outside room1-6 ([[tumvi-gt-is-mocap-room-only]]) every pose is scorable and ATE
is meaningful on all 11. Its body frame is imu0 (`T_BS` = identity in
`imu0/sensor.yaml`), which is also what XIVO's `gsb` and OpenVINS' state report,
so no frame conversion is needed on either side.

## One configuration is honest here

`mav0/{cam0,cam1,imu0}/sensor.yaml` is **byte-identical across all 11
sequences** (md5 `84411ac5` / `dec090ef` / `ec43620a`, 11 of 11 each), and those
values match OpenVINS' shipped `config/euroc_mav/kalibr_imucam_chain.yaml`
exactly. So a single shared configuration is not a compromise imposed for
fairness -- the dataset genuinely has one calibration.

Calibration used by both systems:

| | cam0 | cam1 |
|---|---|---|
| model | pinhole + radtan | pinhole + radtan |
| resolution | 752 x 480 | 752 x 480 |
| fx, fy | 458.654, 457.296 | 457.587, 456.134 |
| cx, cy | 367.215, 248.375 | 379.999, 255.238 |
| k1, k2, p1, p2 | -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05 | -0.28368365, 0.07451284, -0.00010473, -3.555907e-05 |

IMU: 200 Hz, gyro noise density 1.6968e-4 rad/s/sqrt(Hz), random walk 1.9393e-5;
accel 2.0e-3 m/s^2/sqrt(Hz), random walk 3.0e-3. Gravity 9.81.

## Harness

`experiments/openvins/` already had the profile mechanism, so EuRoC needed one
new file, `profiles/euroc_mav.sh` (copied here as `harness/euroc_mav.sh`):

* `PROFILE_SEQS` = the 11 full sequence names.
* `seq_folder` = `$PROFILE_ROOT/<SEQ>` (EuRoC has no `dataset-*_512_16` wrapper).
* `seq_gt_csv` = `.../state_groundtruth_estimate0/data.csv`. `asl_gt_to_tum.awk`
  handles it unchanged: EuRoC's leading columns are `ts, p_xyz, q_wxyz`, the same
  as TUM-VI's `mocap0`, and the extra velocity/bias columns are ignored.
* `seq_config` = the authors' shipped `config/euroc_mav/estimator_config.yaml`,
  unmodified, for stereo. **`seq_extra` is empty for every sequence** -- unlike
  TUM-VI, where room6 needs its own `init_imu_thresh`, OpenVINS' EuRoC config
  needs no per-sequence override, so OpenVINS is also running one config for all
  11.
* Mono is that config with `use_stereo: false` / `max_cameras: 1`, kept in
  `experiments/openvins/configs/euroc_mav_mono/` alongside copies of the two
  kalibr chains, because OpenVINS resolves `relative_config_*` against the
  config file's own directory. Mono is a diagnostic here; the headline
  comparison is stereo + IMU.

Smoke test, V1_01_easy stereo, 4 cores: ATE 0.0561 m (`evaluate_ate.py` 0.02 s
window), 0.0560 m (`ov_eval posyaw`), 95.2 FPS, peak RSS 100.2 MB, init delay
5.60 s. That ATE is in line with the published OpenVINS EuRoC numbers, so the
profile, the GT conversion and the scoring path are all wired up correctly.
