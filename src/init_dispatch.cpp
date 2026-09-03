#include "init_dispatch.h"

#include <cmath>

#include "glog/logging.h"

namespace xivo {

InitDispatcher::InitDispatcher() : InitDispatcher(Options{}) {}

InitDispatcher::InitDispatcher(const Options &opt)
    : opt_{opt}, detect_{opt.detect}, win_{opt.window} {}

number_t InitDispatcher::Elapsed() const {
  return t0_ < 0 ? 0 : t_last_ - t0_;
}

void InitDispatcher::AddImu(number_t t, const Vec3 &gyro, const Vec3 &accel) {
  if (decided_)
    return;
  if (t0_ < 0)
    t0_ = t;
  t_last_ = t;
  // Every sample goes to the window, including those before the first frame and
  // after the last: preintegration interpolates at the window edges and needs a
  // sample on each side.
  win_.AddImu(t, gyro, accel);
  if (decision_.verdict.kind == MotionVerdict::kUndecided)
    detect_.AddImu(t, gyro, accel);
}

void InitDispatcher::AddImage(number_t t, const cv::Mat &gray) {
  if (decided_)
    return;
  if (t0_ < 0)
    t0_ = t;
  t_last_ = t;
  // The detector is asked once and its answer is cached, so it stops consuming
  // images as soon as it has one. Two reasons, and the second is the important
  // one: it saves a KLT per frame over the rest of the window, and it makes the
  // verdict a function of a fixed prefix of the data rather than something that
  // can flip half way through filling the window -- which would leave the
  // filter's start depending on exactly when `Decide()` happened to be called.
  if (decision_.verdict.kind == MotionVerdict::kUndecided)
    detect_.AddImage(t, gray);
  win_.AddImage(t, gray);
}

const InitDecision &InitDispatcher::Decide() {
  if (decided_)
    return decision_;

  if (decision_.verdict.kind == MotionVerdict::kUndecided && detect_.Ready())
    decision_.verdict = detect_.Classify();

  const bool timed_out = Elapsed() >= opt_.max_wait_sec;
  const auto TakeStatic = [this](const char *why) {
    decision_.path = InitDecision::Path::kStatic;
    decision_.why = why;
    decided_ = true;
  };

  switch (decision_.verdict.kind) {
  case MotionVerdict::kStatic:
    // Commit immediately, without waiting for the window to fill. The static
    // case is both the common one and the one that has to stay bit-identical, so
    // spending startup latency on a window that will be thrown away would be
    // paying for nothing.
    TakeStatic("detector: static");
    break;

  case MotionVerdict::kDynamic:
    if (!win_.Full() && !timed_out)
      break; // keep filling; the verdict is already cached
    if (win_.num_frames() < opt_.min_frames) {
      // A dynamic verdict the window cannot act on. Falling back to static is
      // wrong -- the platform *is* moving -- but it is wrong in the direction
      // the filter already handled before this code existed, whereas a bundle
      // adjustment over four frames is wrong in a new and worse way.
      TakeStatic("dynamic verdict, too few frames tracked");
      break;
    }
    if (SolveDynamic()) {
      decision_.path = InitDecision::Path::kDynamic;
      decided_ = true;
    } else {
      // `why` was set by SolveDynamic.
      decision_.path = InitDecision::Path::kStatic;
      decided_ = true;
    }
    break;

  default:
    if (timed_out)
      TakeStatic("no verdict within max_wait_sec");
    break;
  }

  if (decided_) {
    LOG(INFO) << "===== dynamic init dispatch =====";
    LOG(INFO) << "path=" << (decision_.path == InitDecision::Path::kDynamic
                                 ? "dynamic"
                                 : "static")
              << " (" << decision_.why << ")";
    LOG(INFO) << "verdict=" << decision_.verdict.KindName()
              << " accel_sd=" << decision_.verdict.accel_sd
              << " flow_px=" << decision_.verdict.flow_px
              << " pairs=" << decision_.verdict.frame_pairs
              << " imu=" << decision_.verdict.imu_samples;
    LOG(INFO) << "window frames=" << win_.num_frames()
              << " span=" << win_.Span() << " waited=" << Elapsed() << " s";
    if (decision_.path == InitDecision::Path::kDynamic) {
      LOG(INFO) << "v_b=" << decision_.Vsb.transpose() << " |v|="
                << decision_.Vsb.norm();
      LOG(INFO) << "g_b=" << decision_.gravity_body.transpose();
      LOG(INFO) << "bg=" << decision_.bg.transpose()
                << " ba=" << decision_.ba.transpose();
      LOG(INFO) << "pix med=" << decision_.stage_b.pixel_median
                << " rms=" << decision_.stage_b.pixel_rms
                << " imu_rms=" << decision_.stage_b.imu_rms
                << " it=" << decision_.stage_b.iterations
                << " handoff_t=" << decision_.t_handoff;
    }
  }
  return decision_;
}

bool InitDispatcher::SolveDynamic() {
  // Bias prior: zero, at both stages. The detector's `gyro_bias_hint` is a
  // magnitude and not a vector, so there is nothing here to seed a direction
  // with, and Stage B recovers `bg` from a zero start to within 3% of EuRoC's own
  // solved value (notes-n-prompts/notes-dyninit/m3-ba.md).
  const Vec3 bg0 = Vec3::Zero(), ba0 = Vec3::Zero();

  InitProblem prob;
  if (!win_.Build(bg0, ba0, opt_.min_track_frames, &prob)) {
    decision_.why = "window build failed";
    return false;
  }

  LinearInitOptions lopt;
  // The accelerometer mean is what breaks the depth-scaled cost's bimodality;
  // `Check` takes the (far more accurate) constrained solve unless it disagrees
  // with that prior, which is the production setting `init_linear.h` argues for.
  lopt.gravity_prior = win_.GravityFromAccelMean(bg0, ba0);
  lopt.prior_mode = LinearInitOptions::PriorMode::Check;
  decision_.stage_a = SolveLinearInit(prob, lopt);
  if (!decision_.stage_a.ok) {
    decision_.why = decision_.stage_a.why;
    return false;
  }

  BAState seed;
  if (!SeedBAState(prob, decision_.stage_a, &seed)) {
    decision_.why = "BA seed failed";
    return false;
  }
  decision_.stage_b = SolveInitBA(prob, seed, opt_.ba);
  const BAResult &b = decision_.stage_b;
  if (!b.ok) {
    decision_.why = b.why;
    return false;
  }

  // Hand off at the window's *last* frame. That is the instant the filter is
  // about to declare `Rsb = I`, so it is the only frame whose body coordinates
  // the filter's state means anything in.
  const int k = b.state.num_frames() - 1;
  if (k < 0 || static_cast<size_t>(k) >= prob.frames.size()) {
    decision_.why = "BA returned no frames";
    return false;
  }
  const Vec3 v = b.state.VelocityInBody(k);
  const Vec3 g = b.state.GravityInBody(k);

  // Sanity gates. Every one of these is a *rejection* rather than a repair: an
  // initializer that has failed should hand the problem back to the static path,
  // which is merely wrong about the velocity, rather than seed the filter with a
  // state that is wrong about the geometry.
  if (!v.allFinite() || !g.allFinite() || !b.state.bg.allFinite() ||
      !b.state.ba.allFinite()) {
    decision_.why = "non-finite BA state";
    return false;
  }
  if (b.pixel_median > opt_.max_pixel_median) {
    decision_.why = "BA reprojection median above max_pixel_median";
    return false;
  }
  if (v.norm() > opt_.max_speed) {
    decision_.why = "recovered speed above max_speed";
    return false;
  }

  decision_.Vsb = v;
  decision_.gravity_body = g;
  decision_.bg = b.state.bg;
  decision_.ba = b.state.ba;
  decision_.t_handoff = prob.frames[k].t;
  decision_.why = "detector: dynamic, BA converged";
  return true;
}

} // namespace xivo
