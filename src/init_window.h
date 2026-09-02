// Feature tracking over the initialization window.
//
// Separate from `MotionDetector`'s tracking on purpose, even though both run KLT
// over the same frames, because the two need opposite policies. The detector
// wants *frame-to-frame* flow and so re-detects corners every frame, which keeps
// its sample count up over a window only ten frames long. Stage A wants *long
// chains*: a track observed in one frame pair contributes a rank-deficient block
// and gets dropped, so re-detecting every frame would leave the solver with
// nothing. Merging them would mean one of the two doing something it does not
// want.
//
// Also separate from XIVO's `Tracker`, for the reason in plan-dyninit.md section
// 4: `Tracker` is entangled with the feature memory pool and the group/feature
// lifecycle, and running it before the filter exists -- then discarding its
// output -- puts pool and id state at risk. Keeping the pre-init window outside
// that machinery is what makes "the static path is bit-identical" a structural
// property rather than a hope.
#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "alias.h"
#include "init_problem.h"

namespace xivo {

class InitWindowTracker {
public:
  struct Options {
    /** Frames retained. Once full, further images are ignored: the window is a
     *  fixed slice starting where the caller reset it, not a sliding buffer.
     *
     *  31 is 1.5 s at EuRoC's 20 Hz, chosen by measurement rather than
     *  convenience: pixel noise is what a longer window averages down, and at
     *  0.3 px the velocity error falls from 1.14 m/s at 0.5 s to 0.060 at 1.5 s.
     *  Longer is not monotonically better, because the window also integrates
     *  the bias Stage A holds fixed -- on real data with a zero bias prior the
     *  curve is U-shaped with a 1.0-1.5 s plateau. See
     *  notes-n-prompts/notes-dyninit/m2-linear.md. */
    int max_frames{31};
    /** Minimum seconds between retained frames. Zero takes every image; raising
     *  it buys window span at the cost of KLT baseline per step. */
    number_t frame_gap{0.0};
    int max_tracks{160};
    number_t quality_level{0.01};
    number_t min_feature_dist{12.0};
    /** Forward-backward reprojection tolerance, px. Tighter than the detector's
     *  1.0 because a chained track accumulates its drift. */
    number_t fb_thresh{0.5};
    /** Detect replacements when the live track count falls below this. New
     *  tracks are masked away from surviving ones so a corner is never entered
     *  twice under two ids. */
    int topup_below{100};
    int cam_id{0};
    Mat3 Rbc{Mat3::Identity()};
    Vec3 Tbc{Vec3::Zero()};
    number_t gravity{9.81};
  };

  // Two constructors rather than one with `= Options{}`: a nested class's
  // default member initializers are not parsed until the enclosing class is
  // complete, so `Options{}` cannot appear as a default argument here.
  InitWindowTracker();
  explicit InitWindowTracker(const Options &opt);

  void Reset();
  /** IMU sample with `Cg` / `Ca` already applied and the biases still in. */
  void AddImu(number_t t, const Vec3 &gyro, const Vec3 &accel);
  void AddImage(number_t t, const cv::Mat &gray);

  int num_frames() const { return static_cast<int>(frame_t_.size()); }
  bool Full() const { return num_frames() >= opt_.max_frames; }
  number_t Span() const {
    return frame_t_.size() < 2 ? 0 : frame_t_.back() - frame_t_.front();
  }
  /** Live tracks in the most recent frame. */
  int num_live() const { return static_cast<int>(prev_ids_.size()); }
  const Options &opt() const { return opt_; }

  /** Assemble the solver problem, preintegrating each frame from frame 0 at the
   *  given bias prior. Tracks seen in fewer than `min_track_frames` frames are
   *  dropped. Returns false if there is not enough to solve. */
  bool Build(const Vec3 &bg, const Vec3 &ba, int min_track_frames,
             InitProblem *prob) const;

  /** Gravity acceleration in `I0` from the windowed accelerometer mean, with
   *  magnitude `opt.gravity`.
   *
   *  Rotating each reading into `I0` first makes the mean
   *  `mean(a_world)_{I0} - g_{I0}`, so the error is exactly the window's mean
   *  specific force -- a few degrees for hand-carried motion, and zero if the
   *  rig happens to be static. Feeds `LinearInitOptions::gravity_prior`, whose
   *  job is to catch the linear cost's bimodality; see `init_linear.h`. */
  Vec3 GravityFromAccelMean(const Vec3 &bg, const Vec3 &ba) const;

private:
  Options opt_;

  struct Ob {
    int id{-1};
    Vec2 xn{Vec2::Zero()};
    Vec2 px{Vec2::Zero()};
  };
  std::vector<number_t> frame_t_;
  std::vector<std::vector<Ob>> frame_obs_;
  std::vector<InitImu> imu_;

  cv::Mat prev_img_;
  std::vector<cv::Point2f> prev_pts_;
  std::vector<int> prev_ids_;
  int next_id_{0};
  bool have_prev_{false};
};

} // namespace xivo
