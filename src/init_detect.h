// Is the sensor platform static or moving when the filter is asked to start?
//
// XIVO's initializer averages accelerometer samples and calls the average
// gravity, which leaves the initial velocity and both bias states at zero. That
// is right at rest and wrong in motion, so something has to decide which case
// this is. The decision is harder than it looks, and three of the four obvious
// statistics do not work:
//
//   * |a| - |g|, which is what `gravity_init_max_accel_dev` uses, is blind to
//     constant-velocity motion in principle. Specific force is R'(a - g); at
//     constant velocity a = 0 and the magnitude reads exactly |g|.
//   * accelerometer sample variance at a fixed instant does not separate the
//     classes on real data. On EuRoC, V1_01 (0.831) and V2_03 (1.093) are
//     stationary and noisier than MH_02 (0.499) is while moving at 0.48 m/s --
//     someone is picking the rig up. Only the *minimum over candidate windows*
//     separates, and only by 1.93x.
//   * raw pixel disparity, which is what OpenVINS thresholds at 10 px, cannot
//     tell translation from rotation: disparity from rotation happens at any
//     depth. TUM-VI room2 shows 0.765 px of it with the rig essentially at rest.
//   * gyro-de-rotated flow removes the rotation but assumes an unbiased gyro,
//     and at turn-on EuRoC's gyro bias (0.079-0.085 rad/s) is essentially all of
//     what it reads while still. That is 1.8 px of predicted motion per frame
//     gap that never happened -- larger than the signal. And the bias is one of
//     the quantities we are trying to estimate, so using it here is circular.
//
// What does work is to fit the rotation *from the images* and measure how much
// flow it fails to explain. Some rotation explains a rotating camera's flow
// exactly at any depth and any rate; no rotation explains parallax, because
// parallax depends on depth and rotation does not. The residual is therefore a
// translation signal that owes nothing to the gyro and nothing to scene scale.
// Measured over the 17 sequences of the two datasets XIVO is evaluated on, its
// margin is 6.8x (0.610 px moving against 0.090 px static) where accelerometer
// variance gets 1.93x and gyro de-rotation gets 1.37x.
//
// Its sensitivity scales as |t|/depth, so a far-field scene raises the floor.
// That is the same limitation OpenVINS' disparity threshold has, and it is why
// the accelerometer cue is kept as a second, independent trigger rather than
// discarded: the two are blind to different things.
#pragma once

#include <deque>
#include <vector>

#include <opencv2/core.hpp>

#include "alias.h"

namespace xivo {

/** What the detector concluded, and the two statistics behind it, so a caller
 *  can log why rather than only what. */
struct MotionVerdict {
  enum Kind {
    kUndecided, ///< not enough data yet; neither path may be chosen
    kStatic,    ///< average accelerometer samples for gravity, v = 0, b = 0
    kDynamic    ///< solve the window bundle adjustment
  };

  Kind kind{kUndecided};
  /** Smallest accelerometer sample sd over any candidate window, m/s^2. */
  number_t accel_sd{-1};
  /** Smallest best-fit-rotation flow residual over any candidate window, px.
   *  Negative when too few frames tracked to have an opinion. */
  number_t flow_px{-1};
  int frame_pairs{0};
  int imu_samples{0};
  /** Norm of the rotation rate the images disagree with the gyro about, rad/s.
   *  While the rig is still this is the gyro bias, and it is a free byproduct:
   *  a starting point for the bias the bundle adjustment then refines. */
  number_t gyro_bias_hint{0};

  const char *KindName() const {
    switch (kind) {
    case kStatic:
      return "static";
    case kDynamic:
      return "dynamic";
    default:
      return "undecided";
    }
  }
};

/** Accumulates IMU samples and images and decides static vs dynamic.
 *
 *  Deliberately self-contained: it runs its own corner detection and optical
 *  flow rather than borrowing XIVO's `Tracker`. The tracker is entangled with
 *  the feature memory pool and the group/feature lifecycle, and running it
 *  before the filter exists -- then discarding its output -- would put pool and
 *  id state at risk. Keeping the pre-init window out of that machinery is what
 *  makes "the static path is untouched" a structural property rather than a
 *  hope. The cost is one duplicated KLT.
 */
class MotionDetector {
public:
  struct Options {
    /** Candidate window length, seconds. Both statistics are minimised over
     *  windows of this length rather than evaluated once. */
    number_t window_sec{0.5};
    /** Give up waiting and decide with what is available after this long. */
    number_t horizon_sec{2.0};
    /** Accelerometer sample sd above which there is unambiguous acceleration,
     *  m/s^2. Measured: 15 static sequences peak at 0.258, the two moving ones
     *  floor at 0.497. Only consulted when the visual cue has no opinion --
     *  gravity sweeping through the body frame of a stationary but rotating rig
     *  produces about 0.42 m/s^2 at 0.3 rad/s, so this statistic cannot be
     *  trusted to mean translation whenever vision is available. */
    number_t imu_thresh{0.35};
    /** Best-fit-rotation flow residual above which the rig is translating, px.
     *  Measured: 15 static sequences peak at 0.090, the two moving ones floor
     *  at 0.610. */
    number_t flow_thresh{0.25};
    /** Below this many surviving tracks the visual cue has no opinion and the
     *  decision falls back to the accelerometer alone. */
    int min_tracks{15};
    int max_tracks{200};
    /** Forward-backward reprojection tolerance for a track to count, px. The
     *  only outlier rejection here, and it has to be strict: the residual of a
     *  bad track is indistinguishable from parallax. */
    number_t fb_thresh{1.0};
    /** Huber knee for the rotation fit, radians. */
    number_t huber_rad{0.002};
    /** Minimum samples in a window for its sd to count. */
    int min_window_samples{10};
    int cam_id{0};
    /** Camera-to-body rotation, `Xb = Rbc Xc + Tbc`. Used *only* to express the
     *  gyro's rotation in the camera frame for `gyro_bias_hint`; the verdict
     *  itself needs no extrinsics. Leaving this at identity does not bias the
     *  decision, but it does make the hint meaningless whenever the real
     *  extrinsic rotation is large -- conjugation preserves a rotation's angle
     *  but not the angle of a *product* of two rotations, so TUM-VI's ~180 degree
     *  camera flip turns a 0.02 rad/s discrepancy into 0.3. */
    Mat3 Rbc{Mat3::Identity()};
  };

  // Two constructors rather than one with `= Options{}`: a nested class's
  // default member initializers are not parsed until the *enclosing* class is
  // complete, so `Options{}` cannot appear as a default argument here.
  MotionDetector();
  explicit MotionDetector(const Options &opt);

  /** Feed one IMU sample. `t` is seconds; only monotonically increasing
   *  timestamps are meaningful. Cheap: O(1). */
  void AddImu(number_t t, const Vec3 &gyro, const Vec3 &accel);

  /** Feed one grayscale image. Runs a KLT step against the previous one, so
   *  this is the expensive call -- roughly one extra pyramidal flow per frame
   *  for as long as the detector is undecided. */
  void AddImage(number_t t, const cv::Mat &gray);

  /** Decide. Returns `kUndecided` until either cue has enough data. */
  MotionVerdict Classify() const;

  /** True once `Classify()` will not return `kUndecided`, so a caller can stop
   *  feeding without evaluating the (cheap but not free) statistics. */
  bool Ready() const;

  void Reset();

  /** Seconds spanned by the buffered IMU samples. */
  number_t ImuSpan() const;

  const Options &opt() const { return opt_; }

private:
  /** Pooled accelerometer sample sd, minimised over windows of
   *  `window_sec`. Pooled over the three axes, which is the same statistic
   *  OpenVINS' StaticInitializer compares against `init_imu_thresh`. */
  number_t MinWindowAccelSd() const;
  /** Per-frame-pair flow residuals, averaged within a window and minimised over
   *  windows. Mirrors the IMU cue's structure so one threshold argument covers
   *  both: a single noisy instant must not decide either. */
  number_t MinWindowFlow() const;

  Options opt_;

  struct ImuSample {
    number_t t;
    Vec3 gyro;
    Vec3 accel;
  };
  std::deque<ImuSample> imu_;

  /** (time, median flow residual in px, implied gyro-bias magnitude). */
  struct FlowSample {
    number_t t;
    number_t px;
    number_t bias;
  };
  std::vector<FlowSample> flow_;

  cv::Mat prev_img_;
  std::vector<cv::Point2f> prev_pts_;
  number_t prev_t_{0};
  bool have_prev_{false};
};

/** The rotation that best explains unit bearings `u1` landing at `u2`.
 *
 *  Wahba's problem: minimise sum_i w_i |u2_i - R u1_i|^2. With
 *  `M = sum_i w_i u2_i u1_i'` and `M = U S V'`, the minimiser is
 *  `R = U diag(1, 1, det(U V')) V'`. Closed form -- no seed, no line search, no
 *  local minimum to fall into.
 *
 *  Posed on the unit sphere rather than in normalized image coordinates on
 *  purpose. Normalized coordinates are the tangent of the field angle, so on a
 *  wide-angle camera they diverge toward the image edge; a least-squares fit in
 *  them weights edge points by orders of magnitude more than centre points and
 *  its residual is not comparable across the image. Unit bearings are uniformly
 *  conditioned for any camera model, which is what lets one threshold serve
 *  both TUM-VI's ~180 degree fisheye and EuRoC's 80 degree radtan.
 *
 *  `iters` passes of Huber IRLS; `iters == 1` is the plain unweighted fit.
 *  Exposed (rather than kept private) because it is independently testable and
 *  the bundle adjustment reuses it to seed relative attitude.
 */
Mat3 FitRotationWahba(const std::vector<Vec3> &u1, const std::vector<Vec3> &u2,
                      int iters = 4, number_t huber_rad = 0.002);

} // namespace xivo
