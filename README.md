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
numbers and how to reproduce the controls are in
[`RESULTS_STEREO.md`](RESULTS_STEREO.md).

Those three figures are `evaluate_ate.py` at `--max_difference 0.001`; at the
tool's own default of 0.02 the same three arms are 0.140 / 0.058 / 0.095 m. The
argument is the association radius in seconds, and it decides which poses are
scored at all: TUM-VI ground truth is 120 Hz against 20 Hz cameras, so 0.001 s
keeps only the ~26% of frames that happen to fall within 1 ms of a mocap sample,
while anything above half a ground-truth period (0.0042 s) keeps ~98% of them.
Which subset survives also decides the Horn alignment the error is measured
after, and at 0.001 s that subset shifts between runs. **Prefer 0.02**; the tight
window is reported only because the monocular baseline this work is compared
against was quoted at it.

That accuracy costs speed, and the default trades in favour of accuracy:

| config | EKF / tracker | FPS, one core | mean ATE@0.02 |
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

### Out-of-state (MSCKF) update

When the tracker drops a track that never made it into the state, its observations
are triangulated and the 3-D point is marginalized out through the left nullspace
of its Jacobian, so the track still constrains the poses it was seen from. A
sliding window of past poses is kept in the state to give those observations
something to constrain. In stereo the right observation of a dropped track is used
alongside the left where the matcher found one, which yields `4n - 3` rows from an
n-view track instead of `2n - 3`.

The update is off unless `use_OOS` is set. `cfg/tumvi_stereo_oos.json` is the
stereo config above plus `use_OOS: true` and an `OOS` block, and differs from
`cfg/tumvi_stereo.json` in nothing else:

    python scripts/pyxivo.py -root /path/to/tumvi -dataset tumvi \
      -seq room1 -cfg cfg/tumvi_stereo_oos.json -mode eval -dump /tmp/out

It improves both modes and costs nothing measurable — 12.5 against 12.4 FPS for
stereo with and without it, in the same batch. Six-room means, each a 6-member
ensemble (members perturb the initial velocity by ~1e-6 m/s, six orders of
magnitude inside the filter's own prior, so the spread is the scale below which a
difference is not attributable):

| arm | ATE@0.02 | ATE@0.001 | RPE_tra |
| --- | --- | --- | --- |
| monocular control, OOS off | 0.0945 ± 0.0083 | 0.0784 ± 0.0061 | 0.0228 |
| monocular control, OOS on | 0.0852 ± 0.0051 | 0.0686 ± 0.0034 | 0.0213 |
| stereo, OOS off | 0.0637 ± 0.0044 | 0.0556 ± 0.0032 | 0.0138 |
| **stereo, OOS on** | **0.0591 ± 0.0029** | **0.0453 ± 0.0024** | **0.0132** |

Stereo + OOS is the best arm on every metric and the tightest of the four. The
gain is the update itself rather than the right-camera rows: running the same
config with `OOS.use_stereo: false` moves nothing outside its own noise on these
sequences, because `too_short` accounts for 10538 of 14115 dropped-track
candidates and accepted candidates average only 2.0 in-state views. The rows are
the better-posed measurement for slow or forward motion, which room1–room6 do not
contain. Derivation of the deltas, their Welch statistics, and the byte-identity
checks that bound what the change touched are in
[`RESULTS_MERGE.md`](RESULTS_MERGE.md).

### Where this lands against other open-source VIO

Our [wiki](https://github.com/ucla-vision/xivo/wiki/Performance-Evaluation) carries
the TUM-VI room1–room6 comparison from the benchmark paper. Against it, stereo XIVO
with the OOS update is **at parity with OKVIS**, which is the only other stereo
entry in that table:

| method | mean ATE | RPE_tra, 1 s |
| --- | --- | --- |
| OKVIS (stereo, keyframe optimization) | 0.063 | 0.0127 |
| VINS-Mono (keyframe optimization) | 0.095 | 0.0183 |
| ROVIO (filter) | 0.150 | 0.0263 |
| XIVO, monocular (wiki) | 0.093 | 0.0343 |
| **XIVO, stereo + IMU + OOS** | **0.0591** | **0.0132** |

Read that as a tie, not a win. Three qualifications, all of which cut against the
last row:

- **The mean advantage is one sequence.** Head-to-head against OKVIS it is two
  wins, one tie and three losses; the whole −0.004 comes from room2, where OKVIS
  is anomalously bad (0.11 m, its worst room by 57%). Excluding room2, XIVO is
  0.006 m behind.
- **The external numbers are third-party and quoted to two decimals**, so each
  carries ±0.005 of rounding — more than the gap. They also come from a different
  implementation and evaluation protocol than ours.
- **Rotational RPE is not comparable across the table.** The wiki's XIVO column
  matches `evaluate_rpe.py` run on our own output to within 0.01 deg in all six
  rooms, and that evaluator matches timestamps by nearest neighbour, which scores
  a *perfect* trajectory at ~0.28 deg/s on these sequences. Ours is 0.62 deg/s on
  that evaluator and 0.51 with `evaluate_rpe_interp.py`, against 0.54 for OKVIS
  from the paper's own tooling — three numbers from two pipelines, so no ranking
  is supportable. Rotational claims on TUM-VI need the interpolated evaluator.

The cost claim elsewhere in the wiki — comparable accuracy at a fraction of the
compute, 140 FPS against OKVIS's ~20 Hz — is about the *upstream monocular*
configuration, and does not carry over to this one. Stereo at the raised capacity
runs at about 12 FPS on one core, i.e. below both real time and OKVIS's quoted
rate; the accuracy parity was bought with that compute. The 60 / 120 capacity
point is the configuration that reproduces the original claim: 0.063 m, which is
OKVIS's number, at 24 FPS.


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


