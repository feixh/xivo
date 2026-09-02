// The initialization window, as both solver stages see it.
//
// Deliberately a plain data structure with no OpenCV and no dataset in it:
// `init_window.{h,cpp}` fills it from images and IMU, the unit tests fill it
// from closed-form synthetic motion, and neither stage of the solver can tell
// the difference. That is the property that makes "the Jacobians are right"
// checkable at all -- a synthetic problem here has an exact answer, so a
// disagreement is a bug rather than a data question.
//
// Frame 0 is the reference. All geometry is expressed in `I0`, the body frame at
// `frames[0].t`, which is also the frame XIVO's `S` will be if this window's
// *last* frame is taken as the init instant (see the handoff in
// notes-n-prompts/plan-dyninit.md section 3).
#pragma once

#include <vector>

#include "alias.h"
#include "init_preint.h"

namespace xivo {

/** Camera-to-body extrinsics, in XIVO's convention: `Xb = Rbc * Xc + Tbc`
 *  (`feature.cpp:729`). */
struct InitCamera {
  Mat3 Rbc{Mat3::Identity()};
  Vec3 Tbc{Vec3::Zero()};
};

struct InitFrame {
  number_t t{0};
  /** Preintegral from `frames[0].t` to `t`. Identity for frame 0. */
  Preintegral pre;
};

/** One image measurement: feature `track` seen in frame `frame` by camera `cam`
 *  at normalized (undistorted) coordinates `xn`, so its bearing is
 *  `[xn(0), xn(1), 1]`. Undistortion happens once, at insertion, because both
 *  stages want the same numbers and the camera model is the expensive part. */
struct InitObservation {
  int frame{0};
  int track{0};
  int cam{0};
  Vec2 xn{Vec2::Zero()};
};

struct InitProblem {
  std::vector<InitCamera> cams;
  std::vector<InitFrame> frames;
  std::vector<InitObservation> obs;
  int num_tracks{0};
  /** `|g|`, m/s^2. Enforced exactly by Stage A and held fixed by Stage B. */
  number_t gravity{9.81};

  /** Number of *distinct frames* each track is seen in. A track confined to one
   *  frame contributes a rank-2 block and cannot be triangulated, however many
   *  cameras saw it there; a stereo pair in one frame is two rows of the same
   *  rank-2 defect only if the baseline is zero, so this counts frames and the
   *  solver additionally checks the block's conditioning. */
  std::vector<int> TrackFrameCounts() const {
    const int nf = static_cast<int>(frames.size());
    std::vector<int> n(num_tracks, 0);
    if (nf == 0 || num_tracks == 0)
      return n;
    // Dense (track, frame) occupancy rather than "did the frame index change
    // since the last observation of this track", which would silently over-count
    // if a caller ever handed the observations over in a different order.
    std::vector<char> seen(static_cast<size_t>(num_tracks) * nf, 0);
    for (const auto &o : obs) {
      if (o.track < 0 || o.track >= num_tracks || o.frame < 0 || o.frame >= nf)
        continue;
      char &cell = seen[static_cast<size_t>(o.track) * nf + o.frame];
      if (!cell) {
        cell = 1;
        ++n[o.track];
      }
    }
    return n;
  }

  number_t Span() const {
    return frames.size() < 2 ? 0 : frames.back().t - frames.front().t;
  }
};

} // namespace xivo
