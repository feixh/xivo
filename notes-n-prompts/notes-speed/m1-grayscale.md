# M1 -- decode one channel instead of three

## The bug

`pybind11/pyxivo.cpp` read every frame with

    cv::Mat image = cv::imread(image_path);

`cv::imread`'s default flag is `IMREAD_COLOR`, which forces an 8-bit **3-channel**
result whatever the file holds. TUM-VI's `cam0`/`cam1` PNGs are

    512 x 512, colour type 0 (grayscale), bit depth 16

so libpng was decoding one plane and OpenCV was then broadcasting it into three
identical planes, and every stage downstream -- `buildOpticalFlowPyramid`,
`calcOpticalFlowPyrLK`, `FastFeatureDetector`, `Canvas::Update` -- ran on three
copies of the same data. XIVO never wanted colour: `Tracker` uses intensity only
and `Canvas::Update` already has a `channels() == 1` branch that does the
`cvtColor(GRAY2RGB)` itself.

Cost of the redundancy, measured standalone on room3 cam0, 300 images, one core
(`harness/bench_io.cpp`, KLT settings taken from `cfg/eff_stereo.json`: win 15,
5 levels, 30 iters, eps 0.01, 180 points):

```
COLOR (8UC3)  decode  3.692  pyramid  3.610  klt  4.569  fast  0.307  ms/image
GRAY  (8UC1)  decode  2.775  pyramid  1.165  klt  2.805  fast  0.149  ms/image
```

Note that the *decode* saving is the smallest part of it: inflating the PNG is
the same work either way and only the final expansion is saved. The pyramid
(3.1x) and the KLT (1.6x) are where it hurts. Stereo runs two decodes, two
pyramids and three KLT passes per frame against mono's one, one and one, which
is exactly why stereo was 2.2x the cost of mono while OpenVINS' stereo is only
1.6x its mono.

## The change

`pybind11/pyxivo.cpp`: one helper in the anonymous namespace,

    cv::Mat ReadImage(const std::string &path) {
      return cv::imread(path, cv::IMREAD_GRAYSCALE);
    }

routed into the three file-path entry points (`VisualMeas`, `VisualMeasStereo`,
`VisualMeasTrackerOnly`). The two numpy-buffer overloads that go through
`CloneImageFromBuffer` are deliberately untouched, so a caller who hands XIVO an
HxWx3 array still gets the old behaviour. `src/app/vio.cpp` and
`src/app/feature_tracker_only.cpp` got the same flag so the C++ apps and the
Python driver do not diverge.

`scripts/pyxivo.py` is unchanged -- it passes paths, and the decode was always
inside C++.

## Result: FPS

Paired run, candidate on cpu 128 and `auto` baseline on cpu 129 in the same time
window, `--timing --no-score --seqs "room1 room3"`, 2821 frames each:

| | wall base | wall M1 | ratio | FPS base | FPS M1 |
|---|---|---|---|---|---|
| mono room1 | 29.81 s | 23.75 s | **1.255x** | 94.6 | 118.8 |
| mono room3 | 28.52 s | 22.75 s | **1.253x** | 98.9 | 124.0 |
| stereo room1 | 66.83 s | 52.80 s | **1.266x** | 42.2 | 53.4 |
| stereo room3 | 63.09 s | 50.97 s | **1.238x** | 44.7 | 55.3 |

Peak RSS falls too, because the frame, its pyramid and the KLT scratch are all a
third of the size:

| | RSS base | RSS M1 | ratio |
|---|---|---|---|
| mono room1/room3 | 132.1 / 133.6 MB | 124.0 / 122.5 MB | 0.93x |
| stereo room1/room3 | 137.1 / 137.8 MB | 127.7 / 126.7 MB | 0.93x |

Estimator timers, room3 (see `split-estimator-vs-io.md` for the full table):
mono `track` 3.79 -> 2.32 ms, stereo `track` 10.25 -> 6.79 ms. The EKF is
untouched, as expected: mono `actual-update` 1.75 -> 1.72, stereo 3.95 -> 4.04.

## Result: accuracy

**Not bit-identical, by construction.** OpenCV's `LKTrackerInvoker` accumulates
its window sums over channels, so with three identical channels `A11`, `A12`,
`A22`, `b1` and `b2` are all exactly 3x larger. The flow increment
`(A12*b2 - A22*b1) / D` is mathematically invariant, but the float rounding is
not, so tracks diverge in the last bits and the filter -- which is chaotic under
MH gating -- takes a different path. Two smaller effects go the same way:
`minEigThreshold` (default 1e-4) is an absolute threshold on a `minEig` that
OpenCV normalizes by `2 * win * win` but *not* by the channel count, so
grayscale is effectively a 3x stricter gate; and the `D < FLT_EPSILON`
degeneracy test is 9x smaller.

Both were checked for capacity loss and neither costs any: the per-run census is
unchanged to three digits.

| room3 census | base mono | M1 mono | base stereo | M1 stereo |
|---|---|---|---|---|
| feature-slots | 72.68/90 | 72.66/90 | 72.13/90 | 72.17/90 |
| group-slots | 7.10/45 | 7.28/45 | 7.18/45 | 7.31/45 |
| rows | 145.9 | 145.7 | 282.4 | 282.8 |

(room1: 76.27 -> 76.26 mono, 75.92 -> 75.84 stereo.) So the stricter `minEig`
gate does not actually reject anything the filter would have used; no
`minEigThreshold = 1e-4/3` compensation is needed. It stays out of the diff
rather than being added "to be safe", because adding it would itself be an
unmeasured change.

Six-member ensemble (`--jitter 6`, `X.Vsb = [k*1e-6, 0, 0]`), all six TUM-VI
rooms, both modes -- `experiments/results/speed_acc_m1` against
`experiments/results/xivo_ref_jitter`. sd is over the six members' 6-room means.

| metric | mode | base | M1 | delta | in sd_base |
|---|---|---|---|---|---|
| ATE RMSE, 0.02 s window [m] | mono | 0.0928 (sd 0.0067) | 0.0968 | +0.0039 | +0.59 |
| | stereo | 0.0636 (sd 0.0045) | 0.0651 | +0.0015 | +0.33 |
| ATE RMSE, 0.001 s window [m] | mono | 0.0770 (sd 0.0033) | 0.0797 | +0.0027 | +0.81 |
| | stereo | 0.0556 (sd 0.0032) | 0.0560 | +0.0004 | +0.11 |
| ov_eval ATE position [m] | mono | 0.0968 (sd 0.0065) | 0.1006 | +0.0039 | +0.60 |
| | stereo | 0.0688 (sd 0.0042) | 0.0704 | +0.0016 | +0.39 |
| ov_eval ATE orientation [deg] | mono | 1.824 (sd 0.040) | 1.807 | **-0.018** | -0.44 |
| | stereo | 1.798 (sd 0.040) | 1.759 | **-0.040** | -0.98 |
| RPE 8 m, translation [m] | mono | 0.0480 (sd 0.0017) | 0.0473 | **-0.0007** | -0.39 |
| | stereo | 0.0292 (sd 0.0012) | 0.0288 | **-0.0004** | -0.36 |
| RPE 8 m, rotation [deg] | mono | 0.5153 (sd 0.0066) | 0.5061 | **-0.0092** | -1.40 |
| | stereo | 0.5074 (sd 0.0063) | 0.5069 | **-0.0005** | -0.08 |

Every ATE mean moves by less than one ensemble sd, in the worse direction;
every orientation and RPE mean moves in the better direction, one of them by
1.4 sd. Sign-mixed and sub-sd on both sides is what "the same estimator, re-rolled"
looks like. Per-sequence it is a reshuffle rather than a trend: mono room1,
room2 and room4 improve, room3, room5 and room6 degrade.

## What was considered and rejected

- **`IMREAD_ANYDEPTH | IMREAD_GRAYSCALE`** (keep 16 bits). Would change pixel
  values, hence the algorithm, and would make the pyramid and KLT wider again.
- **`IMREAD_REDUCED_GRAYSCALE_2`.** Halves the resolution; that is a capacity
  reduction.
- **Faster PNG.** 2.78 ms per image is libpng inflate plus unfiltering inside
  the shared `dependencies/opencv_install`. OpenVINS pays the same 5.6 ms for two
  images, so this is a floor for both systems, and touching a shared dependency
  would neither be a XIVO change nor survive the paired-ratio test.
