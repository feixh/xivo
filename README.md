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
- (optional) [libdeflate][libdeflate]: Faster grayscale-PNG decode (`src/pngfast.cpp`).
  Detected by CMake; without it that path falls back to zlib and keeps about half of
  its speedup.
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
[libdeflate]: https://github.com/ebiggers/libdeflate
[ros]: https://www.ros.org/


## Build and Usage

Please see our [wiki](https://github.com/ucla-vision/xivo/wiki) for usage instructions and more detailed information about the algorithm.



## New Features (in the era of coding agent, developed by coding agent)

### Intro (written by human)
The original repo does not implement the stereo + IMU combination, and only partially implements the out-of-state (also known as multi-state constraint kalman filter or MSCKF) update. 

Here, I used Claude Code to implement both as an experiment: Is coding agent as of today (August 2026) is capable of implementing such features that require domain-specific knowledge. It turns out that the coding agent is doing an Okay job -- it is able to implement the algorithm, implement some tests, and also tunes and tests the implementation against a simple benchmark (room1 - room6 in TUM-VI, 6 sequences in total) to ensure that 1/ there is no performance regression for the existing features, and 2/ the algorithms implemented outperform the original system which is the expected outcome.

**Note**: Even though the coding agent manages to implement the algorithm and advances the performance, there is no guarantee that the implemented algorithm is free of bugs and mathematicall correct. I didn't manually check the implementation. Again, this is an experiment aiming to test coding agent's performance in domains that require deep expertise.

Now, the rest of this section below are written by a coding agent, the algorithms described in these sub-sections, of course, are implemented also by a coding agent. This intro is written by me, a 35-year old flesh-and-blood human being.


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
| monocular, upstream capacity | 30 / 60 | 89 † | 0.140 |
| **stereo, shipped default** | 90 / 180 | **44** | **0.058** |
| stereo, real-time-friendly | 60 / 120 | 24 † | 0.063 |

† not re-measured since [the efficiency work](#efficiency) below, which sped the
shipped stereo config up 3.6× (11.5 → 44 FPS) at unchanged capacity and accuracy;
the two dagger rows would also be faster now.

So the shipped config runs at about 2.2× real time against TUM-VI's 20 Hz cameras,
and most of what it spends goes to the EKF covariance update growing with the
feature count, not to the stereo front end. Lower `EKF_MAX_FEATURES` and
`tracker_cfg.num_features_max` together (60 / 120) to buy more headroom for 0.005 m
of accuracy. Full breakdown, including where each millisecond goes, in
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

### Out-of-state update

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

### Efficiency

The filter above was then made **4.8× faster monocular and 3.6× faster stereo** at
unchanged EKF capacity, unchanged configs, and unchanged accuracy. Nothing was
traded away: no capacity reduction, no looser gates, no dropped features. All
figures are one core (`OMP_NUM_THREADS=1`, `setarch -R`), TUM-VI room1 and room6,
whole-process wall clock including PNG decode, measured in a single interleaved
batch:

| setting | FPS before | FPS after | speedup | peak RSS |
| --- | --- | --- | --- | --- |
| monocular + IMU | 21.1 | **100.4** | **4.8×** | 450 → 132 MB |
| stereo + IMU | 12.3 | **44.0** | **3.6×** | 459 → 138 MB |

Stereo therefore moves from 0.6× to 2.2× real time against TUM-VI's 20 Hz cameras.

Accuracy is held, verified by 8-member ensembles over room1–room6 (members perturb
the initial velocity by ~1e-6 m/s, so the spread is the scale below which a
difference is not attributable):

| setting | ATE@0.02 | ATE@0.001 | RPE_rot | RPE_tra |
| --- | --- | --- | --- | --- |
| monocular, before → after | 0.0958 → 0.0945 | 0.0796 → 0.0786 | 0.5126 → 0.5126 | 0.0222 → 0.0222 |
| stereo, before → after | 0.0632 → 0.0630 | 0.0551 → 0.0549 | 0.5128 → 0.5128 | 0.0132 → 0.0132 |

RPE is `evaluate_rpe_interp.py`, per the caveat in the section below. Both ATE
columns move well inside the ensemble spread (±0.005), in the improving direction,
and every RPE statistic is unchanged to four decimals — including the two stock ones
not shown, except mono stock RPE_tra at 0.0227 → 0.0226. The stronger check is that
most of this work is algebraically exact rather than approximate, so it admits a
test far tighter than ATE: **91 of the 96 ensemble runs are byte-for-byte identical
to the pre-optimization code**, and all five that differ are on room3, the one
sequence whose accept/reject gating is chaotic.

Where the speedup comes from — each row is a commit, validated before the next
began, with FPS as the two-sequence mean:

| | change | mono | stereo |
| --- | --- | --- | --- |
| baseline | | 21.1 | 12.3 |
| M1 | gate on the structurally nonzero columns of `J` — of 564, only ~33 can be nonzero, and `MHGating` was reading all 2.5 MB of `P` ~90× per frame | 26.0 | 15.7 |
| M2 | restructure the covariance update: compute `H P` once instead of twice, block-sparsely, and replace two N³ Joseph products with an O(mN²) form | 54.2 | 27.3 |
| M3 | apply the 24×540 motion-to-structure correlation once per frame instead of on all ~30 Prince-Dormand substeps | 77.0 | 32.1 |
| M4 | stop building the left image pyramid twice per stereo frame and stop cloning the input image | 77.2 | 36.9 |
| M5 | run the update over the occupied extent of `P` (339 of 564 dims) rather than the full state | 85.9 | 41.0 |
| M6 | propagation internals: dense fixed-size containers instead of `SparseMatrix`, `F`'s nine nonzero rows, `G Qimu Gᵀ` as four 3×3 blocks — 0.170 → 0.032 ms/call | 99.5 | 43.8 |
| M7 | drop `EIGEN_INITIALIZE_MATRICES_BY_ZERO`, which was zero-filling 310 MB of pooled Jacobians one double at a time at startup | 100.4 | 43.9 |

M7 is the memory result rather than a speed one, and it was not free: the define
turned out to be load-bearing in five places, masking five read-before-write bugs
that are fixed here. Valgrind memcheck reported 0 errors and `MALLOC_PERTURB_` gave
a bit-identical trajectory while all five were live; `-DXIVO_EIGEN_INIT=nan` plus an
`FE_INVALID` trap is what found them, and ships as a build knob for the next time a
trajectory goes strange.

Two ideas were measured and rejected: `-flto` and `-fvisibility=hidden` are both
noise here (+0.4% at best), because the hot code is templated Eigen in headers and
there is no cross-TU inlining left to win.

Those FPS figures are the `cfg/tumvi_*` configs at this point in the history. The
accuracy work that came next cost 41% of the throughput, and it was then won back and
more: the shipped `cfg/eff_*` configs run at **123.4 (mono) and 70.0 (stereo) FPS** over
room1–room6 today — see [the OpenVINS comparison](#head-to-head-against-openvins-orientation-position-efficiency).

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
configuration, and the parity above was originally bought back with compute: stereo
at the raised capacity ran at about 12 FPS on one core, below both real time and
OKVIS's quoted rate. After [the efficiency work](#efficiency) it runs at **44 FPS**,
so the parity now comes at 2.2× real time on a single core, and the qualitative
form of the original claim holds again at the shipped capacity.

Every number in this subsection is third-party and quoted. The next subsection redoes
the exercise properly against one system — OpenVINS — built and run on this machine,
scored by the same code, on the same core.


### Head-to-head against OpenVINS: orientation, position, efficiency

The brief for this round was concrete: **match or beat [OpenVINS](https://github.com/rpng/open_vins)
on orientation, on monocular position, and on runtime cost**, borrowing ideas or code
from it but taking no dependency on it. The reference is OpenVINS v2.7
(`v2.7-20-g6948812`), built ROS-free and run on this machine rather than quoted from a
paper, so both systems see the same sequences, the same ground truth, one evaluation
code path and one core. Six branches were developed on isolated `git worktree`s and
merged into `auto`; `ctest` is 22/22 at the endpoint.

Six-room means over TUM-VI room1–room6. Bold beats OpenVINS. `±` is the spread of
6-member ensembles whose members perturb the initial velocity by ~1e-6 m/s — six
orders of magnitude inside the filter's own prior, so it is the scale below which a
difference is not attributable:

| metric | XIVO before | **XIVO now** | OpenVINS | margin |
| --- | --- | --- | --- | --- |
| mono ATE@0.02 [m] | 0.0928 ± 0.0067 | **0.0555 ± 0.0026** | 0.0621 | 11% |
| mono ATE position, `posyaw` [m] | 0.0968 | **0.0575 ± 0.0028** | 0.0638 | 10% |
| mono ATE orientation [deg] | 1.8243 | **0.8788 ± 0.0303** | 1.5742 | 44% |
| mono RPE 8 m, position [m] | 0.0480 | **0.0265 ± 0.0009** | 0.0308 | 14% |
| mono RPE 8 m, orientation [deg] | 0.5153 | **0.5131 ± 0.0033** | 0.6445 | 20% |
| stereo ATE@0.02 [m] | 0.0636 ± 0.0045 | **0.0490 ± 0.0022** | 0.0677 | 28% |
| stereo ATE position, `posyaw` [m] | 0.0688 | **0.0507 ± 0.0022** | 0.0697 | 27% |
| stereo ATE orientation [deg] | 1.7982 | **0.8921 ± 0.0557** | 1.4440 | 38% |
| stereo RPE 8 m, position [m] | 0.0292 | **0.0215 ± 0.0008** | 0.0265 | 19% |
| stereo RPE 8 m, orientation [deg] | 0.5074 | **0.5161 ± 0.0080** | 0.5837 | 12% |

Efficiency, one core (`taskset -c 0`, `setarch -R`, every thread pool at 1, idle box),
whole-process wall clock including PNG decode, peak RSS from `/usr/bin/time`. Means of
three passes taken alternately in one session:

| | mono FPS | stereo FPS | mono peak RSS | stereo peak RSS |
| --- | --- | --- | --- | --- |
| XIVO before this round | 101.8 | 45.0 | 134.1 MB | 139.3 MB |
| **XIVO now** | **123.4** | 70.0 | **86.7 MB** | **94.6 MB** |
| OpenVINS | 114.4 | 71.1 | 88.2 MB | 95.4 MB |
| ratio | **1.08×** | 0.98× | **0.98×** | **0.99×** |

**Thirteen of the fourteen numbers above clear OpenVINS.** The one that does not is
stereo throughput, and it misses by 1.6% — 14.285 ms/frame against 14.057 — uniformly
across sequences rather than on one bad room. Mono wins all six sequences on speed.

The two shipped configs carry every key this round tuned; run them like any other:

    python scripts/pyxivo.py -root /path/to/tumvi -dataset tumvi \
      -seq room1 -cfg cfg/eff_mono.json   -mode eval -dump /tmp/out
    python scripts/pyxivo.py -root /path/to/tumvi -dataset tumvi \
      -seq room1 -cfg cfg/eff_stereo.json -mode eval -dump /tmp/out

(`-mode eval` writes the trajectory; `-mode runOnly` imports no savers and dumps
nothing, which is why timing and scoring are separate passes. `XIVO_DUMP_PRECISE=1`
switches the dump to 17 significant digits, which is what the bit-identity claims below
are checked with. The build gained one optional dependency, libdeflate; without it the
PNG path falls back to zlib and keeps about half of its speedup.)

What each branch did, in merge order. Each throughput ratio is paired against *that
branch's own* baseline, so they do not multiply out to the table above:

| | branch | what | effect |
| --- | --- | --- | --- |
| 1 | `auto-orient` | publish the pose in the gravity-aligned frame (`Estimator::gwb()`/`gwc()`, `gravity_align_output`, default `true`), and fix the 4-DoF gauge about **gravity** instead of the gauge group's body z-axis | orientation ATE 1.82 → **0.95** deg mono, 1.80 → **0.98** stereo; the estimate itself is bit-identical |
| 2 | `auto-speed` | single-channel decode at the entry points, batched block-sparse EKF products, three computed quantities nothing reads, and a pooled-Jacobian memory pass | 1.39× mono, 1.42× stereo; peak RSS −52 MB in both modes |
| 3 | `auto-position` | seven config keys, of which CLAHE equalization is the largest single effect in the whole project: `histogram_method: CLAHE`, `subpix_refine`, `use_OOS` + `OOS.pose_window: 20` + `min_observations: 2`, `consistent_init`, `oos_meas_std: 1.0`, `grayscale` | mono ATE 0.0928 → 0.0566 m; costs 41% of throughput |
| 4 | `auto-oosfast` | column-sparse out-of-state and feature-promotion products, a cheap consistent init, one shared OOS buffer | 1.23× mono, 1.13× stereo — 2.25 and 2.68 ms/frame recovered; −8 MB peak RSS; **bit-identical**, 72 paired runs |
| 5 | `auto-frontfast` | `src/pngfast.cpp` (libdeflate + fused unfilter + 16→8 strip): 2.81 → 1.42 ms per image with **zero changed output bytes**; a cheaper stereo match (`back_track: false`, `max_level: 2`); temporal KLT `max_level` 5 → 4 | 1.14× mono, 1.31× stereo; the decoder alone is +9.3% in both modes at zero accuracy cost |
| 6 | `auto-covrun` | apply the update's rows in `C` sequential chunks against the covariance the previous chunks left, re-predicting each chunk's innovation — exactly the batch update, since the information form is additive, and it makes the innovation covariance `(m/C)²` instead of `m²` | 1.03× mono, 1.10× stereo — −1.38 ms/frame and −10.5 MB stereo peak RSS, at zero accuracy cost |

Tests: `unitTests_ekf_update` (23 cases, including the chunked update checked against
both the batch downdate and an independent dense Joseph form at the real dimensions)
and `unitTests_pngfast` (the fast decoder pinned against `cv::imdecode`) are new.

Six qualifications, because several of these numbers are easy to over-read:

- **The orientation win is a frame-convention fix, not better attitude estimation.**
  XIVO's spatial frame `S` is the body frame of the first IMU sample, tilted by whatever
  the rig's attitude happened to be at startup, and nothing applied the filter's own
  `Rsg` to the published pose. The benchmark had been comparing a tilted frame against
  OpenVINS' level one. After the fix the residual tilt is 0.307 deg against OpenVINS'
  0.312 — **that is the benchmark's floor**, not XIVO's remaining error. What is a real
  win is yaw: 0.704 deg against 1.473, better on all six sequences.
- **ATE@0.02 is blind to a global rotation**, which is why the `posyaw` rows are quoted
  beside it: those align position and yaw only, so a roll/pitch offset lands in the
  reported orientation error undiminished.
- **The accuracy gains come from the config, not the code.** Every new key is a no-op
  when absent, so merging the code without the config is bit-identical. `use_OOS` and
  `consistent_init` must ship *together* — `consistent_init` is +0.0057 m without the
  OOS window and −0.0138 m with it.
- **One deliberate accuracy trade** exists in the whole round: stereo spends 0.0019 m
  of ATE on the cheaper stereo match for +30.7% throughput. Nothing else was traded, and
  0.0187 m of stereo margin is left unspent.
- **Two borrowed ideas ship off**: first-estimates Jacobians (implemented, correct,
  measured inside the noise floor) and two-view epipolar rejection (+0.0030 m once CLAHE
  is in — the two remove the same bad correspondences).
- **A seventh branch was measured and deliberately not merged.** `auto-covscratch`
  removes Eigen's copy of the innovation covariance; it is bit-identical and 0.022
  ms/frame faster, and it reproducibly costs 1.6 MB of stereo peak RSS, which turns 13
  of 14 metrics into 12. It is kept as a branch, not shipped.

Nothing from OpenVINS is linked or included: `grep -rni` over `src`, `pybind11`,
`CMakeLists.txt` and `cfg` finds eleven matches and all eleven are comments citing a
file as the origin of a recipe. What was borrowed — the gravity-aligned output frame,
the Joseph/invertible-init update recipe, CLAHE's two constants, `back_track: false` —
is reimplemented, and one borrowed idea (`measurement_compress_inplace`) was rejected
on the numbers, because it only pays when the stacked residual has more rows than the
state and XIVO's has fewer.

The full write-up, including the measurement protocol, the leads that turned out to be
wrong, and how to reproduce every table, is
[`notes-n-prompts/report-xivo-vs-openvins.md`](notes-n-prompts/report-xivo-vs-openvins.md);
per-branch notes are in `notes-n-prompts/notes-{orient,position,speed,oosfast,frontfast}/`
(`notes-oosfast/` also holds the chunked update, m6, and the unmerged m7), and the
OpenVINS baseline itself is in `notes-n-prompts/notes-openvins-baseline/` and
[`report-openvins-baseline.md`](notes-n-prompts/report-openvins-baseline.md).

Everything above is TUM-VI room1–room6, six sequences in one mocap room. The next
subsection repeats the exercise on a second, harder dataset.


### EuRoC MAV: eleven sequences, one shared configuration

The brief: run the same head-to-head on the [EuRoC MAV dataset](https://docs.openvins.com/gs-datasets.html#gs-data-euroc)
in **stereo + IMU**, with **one XIVO configuration shared by all eleven sequences**
rather than one per sequence, benchmark accuracy *and* runtime, and tune XIVO to
match or beat OpenVINS. Same reference (OpenVINS v2.7 built ROS-free, run on this
machine), same evaluation code, same core. Three branches on isolated `git
worktree`s, merged into `auto`; `ctest` is 23/23 on the merged tree.

XIVO is n=10, OpenVINS n=6, all eleven sequences, stereo. `±` is the standard error
of the eleven-sequence mean; bold is best of three:

| metric | XIVO `acc` | **XIVO shipped** | OpenVINS |
| --- | --- | --- | --- |
| ATE@0.02 [m] | 0.0950 ± 0.0009 | 0.1028 ± 0.0016 | **0.0941** ± 0.0006 |
| ATE position, `posyaw` [m] | 0.1035 ± 0.0009 | 0.1102 ± 0.0016 | **0.0972** ± 0.0006 |
| ATE orientation [deg] | 1.709 ± 0.009 | **1.706** ± 0.010 | 1.773 ± 0.010 |
| RPE 8 m, position [m] | **0.1093** ± 0.0003 | 0.1109 ± 0.0005 | 0.1168 ± 0.0007 |
| RPE 8 m, orientation [deg] | **0.852** ± 0.002 | 0.867 ± 0.003 | 0.902 ± 0.005 |

**XIVO wins three of the five metrics** — both orientation metrics and the position
drift rate — each by 4.6–9.9 combined standard errors, **ties ATE@0.02** at the
accurate operating point (0.8 σ) and **loses absolute ATE position**. Across the 55
per-sequence-per-metric cells XIVO takes 32, OpenVINS 23. Divergences, stereo: 0 of
110, 0 of 110, 0 of 66 — the XIVO baseline this round started from diverged on 15 of
66.

Efficiency, one core, same protocol as above, whole-process wall clock including PNG
decode, `-mode runOnly`:

| | ms/frame | FPS | peak RSS | vs OpenVINS |
| --- | --- | --- | --- | --- |
| XIVO after accuracy tuning (M4) | 14.756 | 67.8 | 95.3 MB | +37.4% |
| XIVO `acc` | 13.921 | 71.8 | 96.2 MB | +29.7% |
| **XIVO shipped** | **11.593** | **86.3** | **97.1 MB** | **+8.0%** |
| OpenVINS | 10.737 | 93.1 | 99.2 MB | — |

So 8.0% behind end-to-end at 4.3× real time on one core, using 3.0 MB less peak
RSS. XIVO's decode (2.972 vs 3.197 ms) and its front end are both *faster*; the
whole residual gap is the EKF path — 90 in-state features and a 20-pose OOS window
against OpenVINS' 50 SLAM features and 11 clones — which is state size, not
implementation slack. Estimator-only, the gap is +14.3%.

**The eleven-sequence near-tie is two large opposite effects cancelling**, and that
is the most substantive result of the round:

| ATE@0.02 | XIVO `acc` | XIVO shipped | OpenVINS |
| --- | --- | --- | --- |
| Machine Hall (5 sequences) | **0.0848** | 0.0883 | 0.1351 |
| Vicon Room (6 sequences) | 0.1036 | 0.1148 | **0.0599** |

XIVO is **37% better on Machine Hall** (large, dim, distant structure, slow flight;
OpenVINS' two worst sequences on the dataset are both here) and OpenVINS is **42%
better on Vicon Room** (small, bright, fast, high feature churn). On the two
sequences XIVO loses worst, RPE-8 disagrees with ATE — V2_03 is 1.8× worse on ATE
and *equal* on RPE-8 — so those losses are a few transient excursions that ATE
integrates and RPE-8 averages out, not a uniformly worse motion estimate.

Run it like any other dataset; `cfg/euroc_*.json` are generated from the dataset's
own per-sensor calibration, which is byte-identical across all eleven sequences, so
one shared config is what EuRoC itself says rather than a concession to the brief:

    python scripts/make_euroc_cfg.py --base cfg/eff_stereo.json \
      --seqdir /path/to/euroc/MH_01_easy --out cfg/euroc_stereo.json
    python scripts/pyxivo.py -root /path/to/euroc -dataset euroc \
      -seq MH_01_easy -cfg cfg/euroc_stereo.json -mode eval -dump /tmp/out

What each branch did:

| | branch | what | effect |
| --- | --- | --- | --- |
| 1 | `auto-euroc` | dataset support — a `euroc` loader branch, per-dataset ground-truth paths, and `scripts/make_euroc_cfg.py` generating the shared config from `sensor.yaml` — plus both baselines | XIVO runs all 11; 15 of 66 stereo runs diverge, and it already wins MH_02/03/05 |
| 2 | `auto-eurocacc` | `P.Wsg` 3.01 → 0.002 (a 1.73 rad prior on which way is down); a `gravity_init_max_accel_dev` stationarity gate; and `Estimator::AdaptVisualMeasNoise`, a χ²(2)-median consistency loop on the visual measurement noise | ATE 0.138 → **0.095**, and **0 of 66 divergences** in each mode |
| 3 | `auto-eurocfps` | per-stage front-end instrumentation, then substituting `FAST.threshold 7` on the raw image for CLAHE at 20 — matched candidate supply (6357 vs 6913 per detecting frame) for 0.36 ms instead of 2.06 | 14.756 → **11.593** ms/frame, +0.008 m; the final evaluation |

Four qualifications:

- **The one-configuration constraint is the whole gap, and it is structural.**
  `visual_meas_std` wants 0.75 px on all five Machine Hall sequences and 1.8–2.4 px
  on five of six Vicon Room ones — the scenes really do differ ~3× in tracking
  noise. A per-sequence oracle (which the brief forbids) scores 0.098, level with
  OpenVINS; one fixed value costs ~40%. That is what motivated measuring the noise
  online instead of choosing it, which is worth 40% against its own control.
- **OpenVINS was fixed, not handicapped.** With its shipped config it diverges on
  MH_04 in 6 of 6 members (~9349 m): its initializer uses feature disparity < 10 px
  as a proxy for stillness, and MH_04 takes off at 0.47 m/s in a scene tens of
  metres deep, so it asserts zero velocity on a moving platform. It was given its
  own dynamic initializer (`--init_dyn_use 1`) uniformly on all eleven — all five
  Machine Hall sequences improve and **all six Vicon Room sequences are
  bit-identical**. Every OpenVINS number here is the fixed baseline.
- **Two operating points, and the shipped one is not the more accurate one.**
  `acc` (CLAHE + `FAST.threshold 20`, two flags away) is ≥ shipped on all five
  metrics for 2.328 ms/frame more. Shipped wins on three grounds: it is the only
  one of the three configs that keeps *monocular* MH_01 alive (`acc` diverges there
  in 10 of 10 members), it closes the throughput gap to 8%, and it is what the
  generator emits.
- **Monocular is reported but not tuned, and OpenVINS wins it 4 metrics to 1**
  (0.185 vs 0.145 m ATE@0.02). Losing the stereo baseline's scale observability
  amplifies exactly the Vicon Room weakness above.

The full write-up — every measurement, every negative result, the protocol, the
methodological findings, and the commands to reproduce each table — is
[`notes-n-prompts/report-xivo-vs-openvins-euroc.md`](notes-n-prompts/report-xivo-vs-openvins-euroc.md);
per-milestone notes are in `notes-n-prompts/notes-euroc/`.

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


