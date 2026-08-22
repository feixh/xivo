# XIVO: X Inertial-aided Visual Odometry and Sparse Mapping


![Demo](misc/demo_ucla_e6.gif)

XIVO is an open-source repository for visual-inertial odometry/mapping. It is a simplified version of Corvis \[[Jones *et al.*][jones_ijrr11],[Tsotsos *et al.*][tsotsos_icra15]\], designed for pedagogical purposes, and incorporates odometry (relative motion of the sensor platform), local mapping (pose relative to a reference frame of the oldest visible features), and global mapping (pose relative to a global frame, including loop-closure and global re-localization — this feature, present in Corvis, is not yet incorporated in XIVO).

XIVO runs at 140FPS on stored data (here from a RealSense D435i sensor) or on live streams with latency of around 1-7ms, depending on the hardware. It takes as input video frames from a calibrated camera and inertial measurements from an IMU, and outputs a sparse point cloud with attribute features and 6 DOF pose of the camera. It performs auto-calibration of the relative pose between the camera and the IMU as well as the time-stamp alignment. More demos are available [here](https://github.com/ucla-vision/xivo/wiki/Background-and-History#semantic-mapping-demo-corvis). The aproach is described in this [paper][tsotsos_icra15].

[jones_ijrr11]: http://vision.ucla.edu/papers/jonesS10IJRR.pdf
[tsotsos_icra15]: http://vision.ucla.edu/papers/tsotsosCS15.pdf
[dong_cvpr17]: http://openaccess.thecvf.com/content_cvpr_2017/papers/Dong_Visual-Inertial-Semantic_Scene_Representation_CVPR_2017_paper.pdf
[fei_eccv18]: http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaohan_Fei_Visual-Inertial_Object_Detection_ECCV_2018_paper.pdf
[fei_icra19]: https://arxiv.org/abs/1807.11130v3
[visma_repo]: https://github.com/feixh/VISMA-tracker


## Requirements

This software is primarily built and tested on Ubuntu 20.04 with compiler g++9. We may support other platforms as we feel like. A full list of supported platforms is listed in our [build page](https://github.com/ucla-vision/xivo/wiki/Build-Instructions).


## Dependencies

- [OpenCV][opencv]: Feature detection and tracking.
- [Eigen][eigen]: Linear algebra.
- [Pangolin][pangolin]: Lightweight visualization.
- [glog][glog]: Logging.
- [gflags][gflags]: Command-line options.
- [jsoncpp][jsoncpp]: Configuration.
- (optional) [googletest][gtest]: Unit tests.
- (optional) [g2o][g2o]: To use pose graph optimization.
- (optional) [ROS][ros]: To use in live mode with ROS.
- (optional) [pybind11][pybind11]: Python binding.
<!-- - [abseil-cpp][absl]: General utilities. -->

All dependencies, except for OpenCV, are included in the `thirdparty` directory.

[opencv]: https://opencv.org/
[eigen]: http://eigen.tuxfamily.org/index.php?title=Main_Page
[g2o]: https://github.com/RainerKuemmerle/g2o
[pangolin]: https://github.com/stevenlovegrove/Pangolin
[absl]: https://abseil.io/
[gtest]: https://github.com/google/googletest
[glog]: https://github.com/google/glog
[gflags]: https://github.com/gflags/gflags
[jsoncpp]: https://github.com/open-source-parsers/jsoncpp
[pybind11]: https://github.com/pybind/pybind11
[ros]: https://www.ros.org/


## Build and Usage

Please see our [wiki](https://github.com/ucla-vision/xivo/wiki) for usage instructions and more detailed information about the algorithm.



## New Features (in the era of coding agent, developed by coding agent)

### Stereo + IMU

This branch adds stereo-camera support: a second camera in the registry, a fixed
rig geometry, left→right feature matching, depth seeding by triangulation at first
observation, and right-image measurements in the EKF update. A config turns it on
by setting `"stereo": true` and supplying `camera1_cfg` and `stereo_cfg` blocks;
without those keys every stereo code path is skipped and behaviour is unchanged.
`cfg/tumvi_stereo.json` is a worked example for TUM-VI, generated from the
dataset's own `dso/camchain.yaml` by `scripts/make_stereo_cfg.py`.

Run it exactly like a monocular config — the config, not a flag, decides whether
the loader reads image pairs:

    python scripts/pyxivo.py -root /path/to/tumvi -dataset tumvi \
      -seq room1 -cfg cfg/tumvi_stereo.json -mode eval -dump /tmp/out

On TUM-VI room1–room6 this reduces mean ATE from 0.121 m (monocular, upstream
capacity) to 0.048 m. Roughly half of that gain is stereo itself and half is the
capacity increase described below; a monocular run at the *same* capacity scores
0.079 m, so stereo is worth 40% against a like-for-like control. Per-sequence
numbers, both ATE association protocols, and how to reproduce the controls are in
[`RESULTS_STEREO.md`](RESULTS_STEREO.md).

That accuracy costs speed, and the default trades in favour of accuracy:

| config | EKF / tracker | FPS, one core | mean ATE |
| --- | --- | --- | --- |
| monocular, upstream capacity | 30 / 60 | 89 | 0.140 |
| **stereo, shipped default** | 90 / 180 | **11.5** | **0.058** |
| stereo, real-time-friendly | 60 / 120 | 24 | 0.063 |

So the shipped config is ~7.5× slower than upstream and runs at about 0.6× real time
against TUM-VI's 20 Hz cameras; almost all of that is the EKF covariance update
growing with the feature count, not the stereo front end. Lower `EKF_MAX_FEATURES`
and `tracker_cfg.num_features_max` together (60 / 120) to get back above real time
for 0.005 m of accuracy. Full breakdown, including where each millisecond goes, in
[`RESULTS_STEREO.md`](RESULTS_STEREO.md#speed-and-memory).

#### EKF capacity is a build option

The number of features and groups the filter can hold is a compile-time constant.
It used to require editing `add_definitions` in `src/CMakeLists.txt`; it is now a
cache variable:

    cmake -S . -B build -DEKF_MAX_FEATURES=90 -DEKF_MAX_GROUPS=45

Those are the defaults, chosen because `cfg/tumvi_stereo.json` needs them —
`core.h` still falls back to 30/15 if the definitions are absent. Two things to
know:

- **Capacity has to be raised in two places.** `tracker_cfg.num_features_max`
  caps what reaches the filter, so raising `EKF_MAX_FEATURES` alone does nothing:
  builds at 60 and at 90 were bit-identical while the tracker stayed at 60.
- **`memory.max_features` is neither of those.** It sizes a fixed pre-allocated
  object pool, and exhausting it is fatal mid-run. Keep it at ≥2× the tracker cap;
  `src/factory.cpp` checks this at startup, along with `require_ekf_max_features`
  in the config, so a mismatched build fails immediately instead of silently
  scoring worse.

Build with `-DEKF_MAX_FEATURES=30 -DEKF_MAX_GROUPS=15` to reproduce numbers taken
at upstream capacity. `-DXIVO_OUTPUT_SUFFIX=_foo` puts a variant build in
`bin_foo/` and `lib_foo/` so it can sit beside the default one; `scripts/pyxivo.py`
reads `XIVO_LIB` to pick which to import.


---
## [LICENSE AND DISCLAIMER ARE COPIED FROM THE ORIGINAL REPO]

## License and Disclaimer 

This software is property of the UC Regents, and is provided free of charge for research purposes only. It comes with no warranties, expressed or implied, according to these [terms and conditions](LICENSE). For commercial use, please contact [UCLA TDG](https://tdg.ucla.edu).

## <a name="ack-anchor"></a> Acknowledgment



If you make use of any part of this code or the datasets provided, please acknowledge this repository by citing the following:
```
@misc{fei2019xivo,
title={XIVO: An Open-Source Software for Visual-Inertial Odometry},
author={Fei, Xiaohan and Soatto, Stefano},
year={2019},
howpublished = "\url{https://github.com/ucla-vision/xivo}"
}
```
or

```
@article{fei2019geo,
  title={Geo-supervised visual depth prediction},
  author={Fei, Xiaohan and Wong, Alex and Soatto, Stefano},
  journal={IEEE Robotics and Automation Letters},
  volume={4},
  number={2},
  pages={1661--1668},
  year={2019},
  publisher={IEEE}
}
```


