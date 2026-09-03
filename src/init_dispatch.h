// M4: choosing an initializer, and holding the messages while the choice is made.
//
// The decision needs about a second of images (see init_detect.h for why nothing
// cheaper separates the classes), but XIVO's static initializer is ready after
// 20 accelerometer samples -- 0.1 s. So something has to give, and what gives is
// *latency*, not the static path:
//
//   1. While the dispatcher is buffering, every message is diverted into it and
//      the estimator is not called at all. No filter state is touched -- not a
//      counter, not a clock, not the accelerometer buffer.
//   2. On a **static** verdict the buffered messages are replayed in order
//      through the ordinary entry points. Because the estimator never saw them
//      the first time, the state it reaches is the state it would have reached
//      with this whole file deleted. `MaintainBuffer` still pops exactly one
//      message per message pushed, so at the instant of the decision the replay
//      has executed exactly as many messages as the unbuffered filter would
//      have, in the same order: the two are in step from the handoff onward, not
//      merely close.
//   3. On a **dynamic** verdict Stage A and Stage B solve the window, the
//      estimator is seeded with the velocity, gravity and gyro bias they
//      recovered, and only the messages *after* the window's last frame are
//      replayed. The earlier ones are not lost: their information is what the
//      bundle adjustment just consumed.
//
// So the cost of enabling this is a one-off startup latency and one extra KLT
// per frame until the verdict lands -- never a per-frame cost afterwards.
//
// Two things a static sequence *does* lose, and neither is a defect in the
// replay:
//
//   - The poses inside the init window are never reported. The filter has not
//     started, so there is nothing to report; this is the latency of step 1 made
//     visible, and it is the same property OpenVINS' initializer has.
//   - With `USE_ONLINE_TEMPORAL_CALIB` the *enqueue* timestamp of an image is
//     stamped with the current `X_.td` (estimator.cpp, `VisualMeas`), because the
//     message heap has to sort images against IMU samples on the corrected clock.
//     Inside the init window no EKF update has run, so every buffered frame
//     carries `td_0`, whereas the unbuffered filter had already nudged `td` by a
//     few hundred nanoseconds. That sub-microsecond difference is enough to flip
//     an image/IMU tie in the heap, and from there the trajectories diverge
//     chaotically. Freezing the calibration (`P.td = 0`) removes the last
//     difference and the shared poses match to the last digit -- which is how
//     `notes-n-prompts/notes-dyninit/harness/m4_bitident.sh` tests the replay,
//     and how the divert was shown to be exact rather than argued to be. Note
//     that `td_0` is the *defensible* value here: the filter genuinely has no
//     temporal-calibration estimate yet when those frames are enqueued.
//
// Deliberately holds no reference to `Estimator`: it accumulates, decides, and
// reports. What to do with the answer, and when to replay, is the estimator's
// business. That keeps this file testable without a filter.
#pragma once

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "alias.h"
#include "init_ba.h"
#include "init_detect.h"
#include "init_linear.h"
#include "init_window.h"

namespace xivo {

/** What the dispatcher concluded, and everything the estimator needs to act on
 *  it. Distances are metres, angles radians, and every vector is expressed in
 *  the body frame of the **handoff frame** -- the last frame of the window --
 *  because that is the instant the filter is about to call `Rsb = I`. */
struct InitDecision {
  enum class Path {
    kWaiting,  ///< still buffering; the estimator must keep diverting
    kStatic,   ///< replay everything, change nothing
    kDynamic   ///< seed from the fields below, replay after `t_handoff`
  };

  Path path{Path::kWaiting};
  MotionVerdict verdict;

  /** Velocity at the handoff instant, in that instant's body frame -- which is
   *  also the spatial frame, since the filter starts with `Rsb = I`. Goes
   *  straight into `X_.Vsb`. */
  Vec3 Vsb{Vec3::Zero()};
  /** Gravity *acceleration* at the handoff instant in the same frame, i.e. the
   *  vector the filter's `Rsg * g_` has to reproduce. */
  Vec3 gravity_body{Vec3::Zero()};
  Vec3 bg{Vec3::Zero()};
  Vec3 ba{Vec3::Zero()};

  /** Window time of the handoff frame, on the caller's own window clock (see
   *  `AddImu`). Messages at or before this are dropped on the dynamic path;
   *  everything after is replayed. Negative if there is no handoff. */
  number_t t_handoff{-1};

  /** Why this path, in words, for the log. Never null. */
  const char *why{"waiting"};

  /** Populated on the dynamic path whether or not it succeeded, so a caller can
   *  log the diagnosis of a *failed* dynamic solve rather than only its verdict.
   *  `ba_result.ok` false with `path == kStatic` is the fallback case. */
  LinearInitResult stage_a;
  BAResult stage_b;
};

class InitDispatcher {
public:
  struct Options {
    /** Off by default. With this false the dispatcher is never constructed and
     *  the estimator's message path is byte-for-byte the pre-M4 one. */
    bool enabled{false};

    /** Fewest frames the bundle adjustment will accept. Under this the dynamic
     *  branch is abandoned for the static one rather than solved badly.
     *  `window.max_frames` times the frame period is the startup latency, so the
     *  two together are what M5 tunes. */
    int min_frames{12};
    /** Tracks seen in fewer frames than this are dropped from the problem. Two
     *  is the minimum that can be triangulated at all, and is what `linear_probe`
     *  validated the pipeline at; exposed so a probe configuration maps onto the
     *  dispatcher one-for-one. */
    int min_track_frames{2};

    /** Hard cap on buffering, seconds. If the detector has not decided by then
     *  the dispatcher gives up and takes the static path -- which is the safe
     *  direction, since it is what the filter did before this existed. Also the
     *  bound on how much memory the diverted images can occupy. */
    number_t max_wait_sec{3.0};

    MotionDetector::Options detect;
    InitWindowTracker::Options window;
    BAOptions ba;

    /** Reject the dynamic solve and fall back to static if the bundle
     *  adjustment's median reprojection error exceeds this, px. A window whose
     *  bulk does not fit has not measured anything, and seeding the filter from
     *  it is worse than seeding it from rest. Measured on EuRoC's two dynamic
     *  windows: 0.335 and 0.253 px, against a static-branch worst case of 0.76,
     *  so 1.5 rejects a genuinely broken solve without touching a working one. */
    number_t max_pixel_median{1.5};
    /** Reject if the recovered speed exceeds this, m/s. A hand-carried rig does
     *  not start at 10 m/s, and a Stage A branch flip looks exactly like that. */
    number_t max_speed{5.0};
  };

  InitDispatcher();
  explicit InitDispatcher(const Options &opt);

  /** Feed one IMU sample, already `Cg`/`Ca`-calibrated with the biases left in --
   *  the same convention `InitWindowTracker::AddImu` takes.
   *
   *  `t` is seconds on the *caller's* window clock, i.e. relative to whatever
   *  origin it chose. Deliberately not converted here: `t_handoff` is reported
   *  back on the same clock, and the caller decides which buffered messages to
   *  replay by recomputing each one's `t` and comparing. Both sides therefore
   *  compare doubles produced by the identical expression, and the split is exact
   *  rather than within-an-epsilon. */
  void AddImu(number_t t, const Vec3 &gyro, const Vec3 &accel);
  /** Feed one grayscale image. Runs up to two KLT steps -- the detector's and the
   *  window's -- so this is the expensive call, and it stops being called the
   *  moment `Decide()` returns anything but `kWaiting`. */
  void AddImage(number_t t, const cv::Mat &gray);

  /** Decide, if there is enough to decide on. Cheap to call per message. Once it
   *  returns a path other than `kWaiting` the answer is cached: the solve runs
   *  once, and asking again does not re-run it. */
  const InitDecision &Decide();

  /** The decision so far, without attempting to make one. */
  const InitDecision &decision() const { return decision_; }

  bool waiting() const { return decision_.path == InitDecision::Path::kWaiting; }

  /** Seconds from the first message to the most recent one. */
  number_t Elapsed() const;

  int num_frames() const { return win_.num_frames(); }
  const Options &opt() const { return opt_; }

private:
  /** Run Stage A then Stage B on the buffered window and fill `decision_`.
   *  Returns false (and leaves a reason in `decision_.why`) if any stage failed
   *  or the result did not survive the sanity gates, in which case the caller
   *  falls back to the static path. */
  bool SolveDynamic();

  Options opt_;
  MotionDetector detect_;
  InitWindowTracker win_;
  InitDecision decision_;
  number_t t0_{-1};
  number_t t_last_{-1};
  /** Set once `Decide()` has committed, so the solve cannot run twice. */
  bool decided_{false};
};

} // namespace xivo
