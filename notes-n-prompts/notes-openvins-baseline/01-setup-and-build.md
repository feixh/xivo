# Setting up OpenVINS in `experiments/open_vins` (2026-08-27)

Goal: get upstream OpenVINS running on TUM-VI room1–6 in both stereo+IMU and
mono+IMU, as an external baseline to compare XIVO against.

## Docker was requested first — it is not available

The prompt asked to prefer the docker images OpenVINS provides. There is no
`docker` binary on this box (`which docker` → not found, and no daemon socket),
and the six `Dockerfile_ros{1,2}_*` files in the repo are all `FROM osrf/ros:*`
development images intended for `catkin build` inside a container, not published
run-ready images. Installing docker would mean adding a system service to a host
we do not own, and would buy nothing: the containers only exist to supply ROS +
ceres, and ceres is one `apt-get install` away here.

So: **native ROS-free build**. This is a first-class upstream configuration
(`.github/workflows/build.yml` = "ROS Free Workflow"), not a hack.

## Toolchain

Ubuntu 26.04, gcc 15.2, cmake 4.2.3, OpenCV 4.10 (system `libopencv-dev`,
which does include `aruco`), Eigen 3.4.0, Boost 1.90, Ceres 2.2, no ROS.

```bash
sudo apt-get install -y libboost-all-dev libceres-dev libgoogle-glog-dev \
                        libgflags-dev libsuitesparse-dev
```

**OpenVINS v2.7 compiles clean under gcc-15 with zero source changes.** Worth
recording explicitly, because XIVO needed six separate porting fixes for the
same toolchain (see [[xivo-modern-toolchain-build-fixes]]): no missing
`<cstdint>`, no OpenCV-3-only enums, no `fmt`/Sophus breakage. Only the build
system needed touching:

1. **cmake 4 vs `cmake_minimum_required(VERSION 3.3)`** — every OpenVINS package
   declares 3.3, which cmake 4.x refuses outright. Fix is a flag, not a patch:
   `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.
2. **Boost 1.90 has no `system` component** — Boost.System became header-only in
   1.69 and Ubuntu 26.04 ships no `libboost_system.so` and no
   `boost_systemConfig.cmake`, so `find_package(Boost REQUIRED COMPONENTS system
   ...)` hard-fails. Dropped `system` from the component list in
   `ov_msckf/CMakeLists.txt` and `ov_eval/CMakeLists.txt`; nothing links against
   it any more.
3. **`ov_eval` ROS-free build never looks for OpenCV** — it compiles ov_core's
   sources into `ov_eval_lib`, and those include `opencv2/...`. Added
   `find_package(OpenCV 4 REQUIRED)` plus include/link in
   `ov_eval/cmake/ROS1.cmake`. (Upstream CI only builds `ov_msckf` ROS-free, so
   this path is untested upstream.)
4. `-DDISABLE_MATPLOTLIB=ON` for ov_eval — its matplotlib-cpp wrapper against
   Python 3.14 headers is not worth the fight; we only need the numbers.

## The missing piece: there is no ROS-free dataset player

ROS-free, OpenVINS builds `run_simulation`, `test_sim_meas`, `test_sim_repeat` —
all simulator-only. Both real-data entry points (`run_subscribe_msckf`,
`ros1_serial_msckf`) are inside `if (catkin_FOUND AND ENABLE_ROS)`, and
`ros1_serial_msckf` reads a **rosbag**, which we do not have (TUM-VI ships ASL
folders; converting 6×1.6 GB to bags would need ROS anyway).

So I added `ov_msckf/src/run_euroc_folder.cpp` (~380 lines, new file, no
estimator code touched). It is a line-by-line port of the replay logic in
`ros1_serial_msckf.cpp` + `ROS1Visualizer::callback_{inertial,monocular,stereo}`:

* reads `mav0/imu0/data.csv` and `mav0/cam{0,1}/data.csv` + image folders;
* groups cameras into synchronized frames with the same ±20 ms rule, skipping
  frames without a partner in every camera;
* applies the same `track_frequency` throttle (21 Hz vs TUM-VI's 20 Hz image
  rate, so nothing is dropped);
* **keeps the same ordering guarantee**: a camera frame is fed only once an IMU
  sample beyond `t_cam + dt_CAMtoIMU` has been consumed, i.e.
  `while (!queue.empty() && queue[0].t < t_imu - dt_CAMtoIMU) feed(queue[0])`.
  This is the one thing worth getting right — feed a frame too early and the
  propagation has no IMU to integrate over, which silently degrades accuracy;
* writes the IMU pose in the world frame in TUM format, stamped
  `state->_timestamp + dt_CAMtoIMU`, exactly as `publish_state()` does
  (OpenVINS' JPL `q_GtoI` components equal Hamilton `q_ItoG`, so the quaternion
  is written straight out, again as upstream does);
* times each `feed_measurement_camera` call and writes `timing.csv` +
  `stats.txt` (mean/median/p95/max ms, FPS, realtime factor, peak RSS, init
  delay), which is what the FPS numbers in the report come from;
* CLI overrides for the knobs an experiment sweeps: `--max_cameras`,
  `--use_stereo`, `--init_imu_thresh`, `--gravity_mag`,
  `--num_opencv_threads`, `--start`, `--duration`.

Deliberately *not* ported: the async subscriber/publisher threads
(`use_multi_threading_{subs,pubs} = false`), the visualizer, and image
publishing. Serial replay is what `ros1_serial_msckf` does too, and it makes the
run reproducible and the timing meaningful.

## Configs

Mono uses `config/tum_vi/estimator_config_mono.yaml`, a copy of the authors'
shipped TUM-VI config with `max_cameras: 1` and `use_stereo: false` — the two
knobs the upstream launch file exposes for this. Everything else (masks, online
camera intrinsic/extrinsic/timeoffset calibration, 11 clones, 50 SLAM features,
`num_pts: 200`, KLT, `init_imu_thresh: 0.45`) is left exactly as shipped, so the
numbers are "OpenVINS as its authors configured it for this dataset", not a
tuned variant.

Gotcha that cost 10 minutes: **`cv::FileStorage` cannot parse a colon inside an
end-of-line comment**. `use_stereo: false # PATCH: mono variant` fails with
`the node use_stereo has an invalid boolean type of []`, because the parser
treats `PATCH:` as a nested key. Drop the colon.

## Validation that the player is wired up correctly

* `init_imu_thresh` / static init fires at 4.1–6.6 s into each sequence, and
  identically for mono and stereo (init only uses cam0 disparity + IMU
  excitation) — consistent with the shipped config's expectations.
* Online calibration converges to sane values within a couple of seconds
  (cam0 intrinsics 191.9/191.8/255.0/256.9 vs the kalibr file's 190.98/190.97/254.93/256.90,
  camera-IMU timeoffset settling at −0.12 ms).
* Resulting ATE is in the same band as the OpenVINS paper/docs report for
  TUM-VI rooms (0.03–0.10 m), so nothing is structurally broken.
* Zero unsynchronized and zero rate-dropped frames on all six sequences, and
  `frames_processed` equals the image count in `cam0/data.csv` exactly
  (e.g. room1: 2821 = 2821), so nothing is being silently skipped. The
  trajectory files are shorter (room1: 2689 poses) purely because poses are only
  written once the filter has initialized, 4–7 s in.
