#include "init_dispatch.h"

#include <cmath>

#include "glog/logging.h"

namespace xivo {

namespace {
using Clock = std::chrono::steady_clock;
/// Milliseconds since `t0`, on a clock that cannot be stepped by NTP.
number_t MsSince(const Clock::time_point &t0) {
  return std::chrono::duration<number_t, std::milli>(Clock::now() - t0).count();
}
} // namespace

InitDispatcher::Options InitDispatcher::OptionsFromJson(const Json::Value &dyn,
                                                       const Mat3 &Rbc,
                                                       const Vec3 &Tbc,
                                                       number_t gravity) {
  Options opt;
  opt.max_wait_sec = dyn.get("max_wait_sec", opt.max_wait_sec).asDouble();
  opt.min_frames = dyn.get("min_frames", opt.min_frames).asInt();
  opt.min_track_frames =
      dyn.get("min_track_frames", opt.min_track_frames).asInt();
  opt.max_pixel_median =
      dyn.get("max_pixel_median", opt.max_pixel_median).asDouble();
  opt.max_speed = dyn.get("max_speed", opt.max_speed).asDouble();

  const int cam_id = 0; // the window is monocular; see `VisualStereo::image`.
  opt.detect.cam_id = cam_id;
  opt.detect.Rbc = Rbc;
  opt.detect.window_sec =
      dyn.get("detect_window_sec", opt.detect.window_sec).asDouble();
  opt.detect.horizon_sec =
      dyn.get("detect_horizon_sec", opt.detect.horizon_sec).asDouble();
  opt.detect.imu_thresh = dyn.get("imu_thresh", opt.detect.imu_thresh).asDouble();
  opt.detect.flow_thresh =
      dyn.get("flow_thresh", opt.detect.flow_thresh).asDouble();
  opt.detect.min_tracks =
      dyn.get("detect_min_tracks", opt.detect.min_tracks).asInt();

  opt.window.cam_id = cam_id;
  opt.window.Rbc = Rbc;
  opt.window.Tbc = Tbc;
  opt.window.gravity = gravity;
  opt.window.max_frames =
      dyn.get("window_frames", opt.window.max_frames).asInt();
  opt.window.frame_gap =
      dyn.get("window_frame_gap", opt.window.frame_gap).asDouble();
  opt.window.max_tracks =
      dyn.get("window_max_tracks", opt.window.max_tracks).asInt();

  opt.ba.sigma_pix = dyn.get("sigma_pix", opt.ba.sigma_pix).asDouble();
  opt.ba.max_iterations = dyn.get("ba_iters", opt.ba.max_iterations).asInt();
  // `sigma_ba_prior` is deliberately not configurable: it is not a tuning knob
  // but the statement that `ba` is not estimated over a window this short. See
  // the measurements at its declaration in init_ba.h.
  return opt;
}

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
  const auto tic = Clock::now();
  if (t0_ < 0)
    t0_ = t;
  t_last_ = t;
  ++num_images_;
  // The detector is asked once and its answer is cached, so it stops consuming
  // images as soon as it has one. Two reasons, and the second is the important
  // one: it saves a KLT per frame over the rest of the window, and it makes the
  // verdict a function of a fixed prefix of the data rather than something that
  // can flip half way through filling the window -- which would leave the
  // filter's start depending on exactly when `Decide()` happened to be called.
  if (decision_.verdict.kind == MotionVerdict::kUndecided)
    detect_.AddImage(t, gray);
  win_.AddImage(t, gray);
  buffer_ms_ += MsSince(tic);
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
    if (!win_.ImuCoversFrames() && !timed_out)
      // The window filled on an image whose timestamp coincides with an IMU
      // sample that has not been handed over yet -- the common case, not a rare
      // one, since on EuRoC every image coincides with one and the tie order is
      // unspecified at both callers. Wait for it: one more message, <=5 ms at
      // 200 Hz, and the decision stops depending on that order. Without this the
      // solve is attempted against a window `Build` cannot preintegrate to its
      // own last frame, and a moving platform is demoted to the static path.
      break;
    if (win_.num_frames() < opt_.min_frames) {
      // A dynamic verdict the window cannot act on. Falling back to static is
      // wrong -- the platform *is* moving -- but it is wrong in the direction
      // the filter already handled before this code existed, whereas a bundle
      // adjustment over four frames is wrong in a new and worse way.
      TakeStatic("dynamic verdict, too few frames tracked");
      break;
    }
    {
      // Timed here rather than inside `SolveDynamic`, which returns from eight
      // places: a rejected solve costs its compute too, and the cost of the
      // rejection is exactly what MH_04 and MH_05 pay.
      const auto tic = Clock::now();
      const bool solved = SolveDynamic();
      solve_ms_ = MsSince(tic);
      // `why` was set by SolveDynamic if it failed.
      decision_.path = solved ? InitDecision::Path::kDynamic
                              : InitDecision::Path::kStatic;
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
    LOG(INFO) << "cost: buffer=" << buffer_ms_ << " ms over " << num_images_
              << " images, solve=" << solve_ms_ << " ms, total="
              << buffer_ms_ + solve_ms_ << " ms (one-off)";
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
