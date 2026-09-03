// Run Stage A (the closed-form linear initializer) on a real dataset window and
// print what it recovered, in `I0`.
//
// This is the tool the M2 correctness claim is made with. The unit tests pin the
// algebra against synthetic data with exact ground truth; this binary answers the
// only question they cannot, which is whether real KLT tracks over a real EuRoC
// window carry enough signal to recover the initial velocity. Compare its output
// against `state_groundtruth_estimate0` with
// notes-n-prompts/notes-dyninit/harness/linear_check.py.
//
// Constructs no Estimator and touches no filter state, so it is safe to point at
// anything.
//
//   bin/linear_probe -cfg cfg/euroc_stereo.json -dataset euroc \
//       -root ../data/euroc -seq MH_01_easy -start 1.1 -frames 31
//
// `-start` is seconds after the first IMU sample, not after the first image, so
// that it lines up with `init_probe`'s clock.
#include <cstdio>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/imgcodecs.hpp"

#include "camera_manager.h"
#include "init_ba.h"
#include "init_linear.h"
#include "init_window.h"
#include "loader.h"
#include "message_types.h"

DEFINE_string(cfg, "cfg/euroc_stereo.json",
              "Estimator configuration file; supplies camera_cfg and X.Wbc.");
DEFINE_string(root, "../data/euroc", "Dataset root directory.");
DEFINE_string(dataset, "euroc", "euroc | tumvi | xivo");
DEFINE_string(seq, "MH_01_easy", "Sequence name.");
DEFINE_int32(cam_id, 0, "Camera id.");
DEFINE_double(start, 1.1, "Window start, seconds after the first IMU sample.");
DEFINE_int32(frames, 31, "Frames in the window.");
DEFINE_double(frame_gap, 0.0, "Minimum seconds between retained frames.");
DEFINE_int32(max_tracks, 160, "Corner budget.");
DEFINE_int32(min_track_frames, 2, "Drop tracks seen in fewer frames than this.");
DEFINE_double(bgx, 0.0, "Gyro bias prior x.");
DEFINE_double(bgy, 0.0, "Gyro bias prior y.");
DEFINE_double(bgz, 0.0, "Gyro bias prior z.");
DEFINE_double(bax, 0.0, "Accel bias prior x.");
DEFINE_double(bay, 0.0, "Accel bias prior y.");
DEFINE_double(baz, 0.0, "Accel bias prior z.");
DEFINE_double(prior_thresh, 0.30,
              "Gravity/prior disagreement (rad) above which the constrained "
              "solve is taken as branch-flipped; <=0 disables the check.");
DEFINE_bool(ba, false,
            "Also run Stage B (the bundle adjustment) and print its columns "
            "after Stage A's, so a caller parsing only Stage A keeps working.");
// Stage B knobs. Every one of these is applied *only if it was passed on the
// command line*, so that an unflagged run measures the shipped `BAOptions`
// defaults rather than whatever value happens to be written below. Repeating the
// defaults here would be a trap, and was one: with `sigma_ba_prior` declared 0 to
// match an older library default, an unflagged run silently measured the
// unpriored solve -- 4.5 degrees of gravity tilt -- and reported it as the
// shipping configuration. The values below are therefore only what `--help`
// shows; `Set` decides what takes effect.
DEFINE_double(sigma_pix, 1.0, "Stage B image noise, pixels.");
DEFINE_double(cauchy, 3.0, "Stage B Cauchy scale, in units of sigma_pix; 0 off.");
DEFINE_int32(ba_iters, 30, "Stage B iteration budget.");
DEFINE_double(sigma_yaw, 1e-3, "Stage B world-yaw gauge sigma, radians.");
DEFINE_double(sigma_bg_prior, 0.0,
              "Stage B gyro-bias prior sigma, rad/s; 0 disables.");
DEFINE_double(sigma_ba_prior, 0.01,
              "Stage B accel-bias prior sigma, m/s^2; 0 disables.");
DEFINE_int32(at_frame, 0,
             "Which window frame Stage B's state is reported at; -1 is the last. "
             "This is not a display detail. Stage A only ever speaks about frame "
             "0, but `InitDispatcher::SolveDynamic` seeds the filter from the "
             "**last** frame -- that is the instant the filter is about to call "
             "`Rsb = I` -- so 0 scores a quantity nothing consumes. The default "
             "stays 0 so that the M2/M3 numbers remain reproducible.");
DEFINE_bool(cov, false,
            "Also report Stage B's marginal covariance over the three seeded "
            "quantities, as extra `#cov` lines: nine sigmas, then the full 9x9. "
            "Rotated into the **last** frame's body coordinates -- the frame the "
            "handoff speaks in, and the one whose velocity block is "
            "gauge-invariant -- irrespective of `-at_frame`.");
DEFINE_bool(header, false, "Print a column header before the row.");

using namespace xivo;

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto cfg = LoadJson(FLAGS_cfg);
  Camera::Create(cfg["camera_cfg"], FLAGS_cam_id);

  std::string image_dir, imu_dir, mocap_dir;
  std::tie(image_dir, imu_dir, mocap_dir) =
      GetDirs(FLAGS_dataset, FLAGS_root, FLAGS_seq, FLAGS_cam_id);
  DataLoader loader{image_dir, imu_dir};

  InitWindowTracker::Options opt;
  opt.max_frames = FLAGS_frames;
  opt.frame_gap = FLAGS_frame_gap;
  opt.max_tracks = FLAGS_max_tracks;
  opt.cam_id = FLAGS_cam_id;
  // Rbc maps camera into body: Xb = Rbc Xc + Tbc.
  opt.Rbc = GetMatrixFromJson<number_t, 3, 3>(cfg["X"], "Wbc",
                                              JsonMatLayout::RowMajor);
  opt.Tbc = GetVectorFromJson<number_t, 3>(cfg["X"], "Tbc");
  InitWindowTracker win(opt);

  const Vec3 bg{FLAGS_bgx, FLAGS_bgy, FLAGS_bgz};
  const Vec3 ba{FLAGS_bax, FLAGS_bay, FLAGS_baz};

  number_t t0 = -1, t_first_frame = -1;
  for (int i = 0; i < loader.size(); ++i) {
    auto *msg = loader.Get(i);
    const number_t t = std::chrono::duration<number_t>(msg->ts_).count();
    if (auto *imu = dynamic_cast<msg::IMU *>(msg)) {
      if (t0 < 0)
        t0 = t;
      // Every sample is offered, including those before `-start`: preintegration
      // interpolates at the window edges and needs a sample on each side.
      win.AddImu(t - t0, imu->gyro_, imu->accel_);
    } else if (auto *im = dynamic_cast<msg::Image *>(msg)) {
      if (t0 < 0 || t - t0 < FLAGS_start)
        continue;
      if (win.Full())
        continue;
      cv::Mat gray = cv::imread(im->image_path_, cv::IMREAD_GRAYSCALE);
      if (gray.empty()) {
        LOG(WARNING) << "failed to read " << im->image_path_;
        continue;
      }
      win.AddImage(t - t0, gray);
      if (t_first_frame < 0 && win.num_frames() > 0)
        t_first_frame = t - t0;
    }
    // Stop one IMU sample *after* the window closes so the last frame has an
    // interval to be preintegrated over.
    if (win.Full() && dynamic_cast<msg::IMU *>(msg) != nullptr)
      break;
  }

  InitProblem prob;
  if (!win.Build(bg, ba, FLAGS_min_track_frames, &prob)) {
    printf("%-26s BUILD_FAILED frames=%d live=%d span=%.3f\n", FLAGS_seq.c_str(),
           win.num_frames(), win.num_live(), win.Span());
    return 1;
  }

  const Vec3 g_prior = win.GravityFromAccelMean(bg, ba);

  LinearInitOptions lopt;
  lopt.gravity_prior = g_prior;
  lopt.max_prior_disagreement = FLAGS_prior_thresh;
  lopt.prior_mode = FLAGS_prior_thresh > 0
                        ? LinearInitOptions::PriorMode::Check
                        : LinearInitOptions::PriorMode::Ignore;
  const auto res = SolveLinearInit(prob, lopt);
  if (!res.ok) {
    printf("%-26s SOLVE_FAILED %s\n", FLAGS_seq.c_str(), res.why);
    return 1;
  }

  if (FLAGS_header)
    printf("%-26s %8s %8s %6s %5s %6s %6s %9s %9s %9s %9s %9s %9s %9s %9s "
           "%9s %9s %7s %5s %9s\n",
           "sequence", "t0", "span", "frames", "trks", "rows", "obs", "vx", "vy",
           "vz", "gx", "gy", "gz", "gpx", "gpy", "gpz", "resid", "pr_ang",
           "flip", "gcond");
  printf("%-26s %8.4f %8.4f %6d %5d %6d %6d %9.5f %9.5f %9.5f %9.5f %9.5f "
         "%9.5f %9.5f %9.5f %9.5f %9.3e %7.4f %5d %9.2e\n",
         FLAGS_seq.c_str(), t_first_frame, win.Span(), win.num_frames(),
         res.tracks_used, res.rows, static_cast<int>(prob.obs.size()), res.v(0),
         res.v(1), res.v(2), res.g(0), res.g(1), res.g(2), g_prior(0),
         g_prior(1), g_prior(2), res.residual, res.prior_disagreement,
         res.gravity_flipped ? 1 : 0, res.g_cond);

  if (FLAGS_ba) {
    BAState seed;
    if (!SeedBAState(prob, res, &seed)) {
      printf("  BA_SEED_FAILED\n");
      return 1;
    }
    // Override only what the caller actually asked for; see the note at the flag
    // declarations. `is_default` is false exactly when the flag appeared on the
    // command line, which is the distinction the value alone cannot make.
    BAOptions bopt;
    const auto Set = [](const char *name, auto *dst, auto value) {
      if (!gflags::GetCommandLineFlagInfoOrDie(name).is_default)
        *dst = value;
    };
    Set("sigma_pix", &bopt.sigma_pix, FLAGS_sigma_pix);
    Set("cauchy", &bopt.cauchy_c, FLAGS_cauchy);
    Set("ba_iters", &bopt.max_iterations, FLAGS_ba_iters);
    Set("sigma_yaw", &bopt.sigma_yaw, FLAGS_sigma_yaw);
    Set("sigma_bg_prior", &bopt.sigma_bg_prior, FLAGS_sigma_bg_prior);
    Set("sigma_ba_prior", &bopt.sigma_ba_prior, FLAGS_sigma_ba_prior);
    bopt.want_covariance = FLAGS_cov;
    const BAResult b = SolveInitBA(prob, seed, bopt);
    const int at = FLAGS_at_frame < 0 ? b.state.num_frames() + FLAGS_at_frame
                                      : FLAGS_at_frame;
    if (at < 0 || at >= b.state.num_frames()) {
      printf("  BA_FRAME_OUT_OF_RANGE %d of %d\n", at, b.state.num_frames());
      return 1;
    }
    const Vec3 bv = b.state.VelocityInBody(at), bg_b = b.state.GravityInBody(at);
    if (FLAGS_header)
      printf("%-26s %9s %9s %9s %9s %9s %9s %10s %10s %10s %9s %9s %9s "
             "%8s %8s %8s %4s %4s %4s\n",
             "#ba", "b_vx", "b_vy", "b_vz", "b_gx", "b_gy", "b_gz", "b_bgx",
             "b_bgy", "b_bgz", "b_bax", "b_bay", "b_baz", "b_pix", "b_pmed",
             "b_imu", "b_it", "b_rj", "b_ok");
    printf("%-26s %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f %10.6f %10.6f %10.6f "
           "%9.5f %9.5f %9.5f %8.3f %8.3f %8.3f %4d %4d %4d\n",
           FLAGS_seq.c_str(), bv(0), bv(1), bv(2), bg_b(0), bg_b(1), bg_b(2),
           b.state.bg(0), b.state.bg(1), b.state.bg(2), b.state.ba(0),
           b.state.ba(1), b.state.ba(2), b.pixel_rms, b.pixel_median,
           b.imu_rms, b.iterations,
           b.rejections, b.ok ? 1 : 0);
    if (FLAGS_at_frame != 0)
      printf("#at_frame %d of %d, t=%.4f s\n", at, b.state.num_frames(),
             prob.frames[at].t);
    if (FLAGS_cov) {
      if (!b.cov_ok) {
        printf("#cov ok=0\n");
      } else {
        // Body frame of the last frame, for velocity; `bg` and `ba` are body
        // quantities already. See BAResult::cov for why the rotation matters.
        Mat9 T = Mat9::Identity();
        T.block<3, 3>(0, 0) = b.state.R.back().transpose();
        const Mat9 C = T * b.cov * T.transpose();
        const Vec9 s = C.diagonal().cwiseSqrt();
        printf("#cov ok=1 s_v=%.6e %.6e %.6e s_bg=%.6e %.6e %.6e "
               "s_ba=%.6e %.6e %.6e\n",
               s(0), s(1), s(2), s(3), s(4), s(5), s(6), s(7), s(8));
        printf("#cov9");
        for (int i = 0; i < 9; ++i)
          for (int j = 0; j < 9; ++j)
            printf(" %.6e", C(i, j));
        printf("\n");
      }
    }
  }
  return 0;
}
