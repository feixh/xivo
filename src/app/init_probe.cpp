// Run `MotionDetector` over a dataset sequence and print its verdict.
//
// This is the C++ counterpart of notes-n-prompts/notes-dyninit/harness/flow_diag.py:
// that script chose the cue by measuring four candidate statistics offline in
// Python; this binary confirms the shipped C++ implementation reaches the same
// conclusion through the real camera model, the real KLT, and the real dataset
// loader. It is the tool the M1 classification claim is made with, and it stays
// in the tree because "what did the detector think of this sequence, and why" is
// the first question to ask whenever initialization misbehaves.
//
// It does not construct an Estimator and touches no filter state, so it is safe
// to point at anything.
//
//   bin/init_probe -cfg cfg/euroc_stereo.json -dataset euroc \
//       -root ../data/euroc -seq MH_01_easy
//
// Feeds messages in the loader's order -- which is timestamp order, starting at
// the first IMU sample -- and reports the verdict at the moment `Ready()` first
// returns true, because that is the moment the estimator will ask.
//
// With `-dispatch` it instead drives the whole `InitDispatcher` and reports the
// path, the seed and the millisecond cost -- the measurement M5's "the cost is a
// startup latency, not a throughput regression" claim rests on. It is the only
// way to see that cost: it is ~0.9 s on a dynamic start, which is a couple of
// percent of a whole run's compute and so is not separable from a wall clock:
//
//   bin/init_probe -dispatch -header -repeat 5 -cfg cfg/euroc_stereo.json \
//       -dataset euroc -root ../data/euroc -seq MH_01_easy
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/imgcodecs.hpp"

#include "camera_manager.h"
#include "init_detect.h"
#include "init_dispatch.h"
#include "loader.h"
#include "message_types.h"

DEFINE_string(cfg, "cfg/euroc_stereo.json",
              "Estimator configuration file; supplies camera_cfg.");
DEFINE_string(root, "../data/euroc", "Dataset root directory.");
DEFINE_string(dataset, "euroc", "euroc | tumvi | xivo");
DEFINE_string(seq, "MH_01_easy", "Sequence name.");
DEFINE_int32(cam_id, 0, "Camera id.");
DEFINE_double(window, 0.5, "MotionDetector window_sec.");
DEFINE_double(horizon, 2.0, "MotionDetector horizon_sec.");
DEFINE_double(flow_thresh, 0.25, "MotionDetector flow_thresh, px.");
DEFINE_double(imu_thresh, 0.35, "MotionDetector imu_thresh, m/s^2.");
DEFINE_double(start, 0.0,
              "Ask the detector at `start` seconds into the sequence instead of "
              "at its beginning: every message before that instant is dropped, "
              "so the probe sees exactly what an estimator turned on mid-flight "
              "would see. This is the counterpart of `pyxivo.py -start_sec`, and "
              "the two use the same epoch (the first IMU sample), so the same "
              "number means the same instant in both. It exists because the "
              "detector's hard case is not a rig on a table -- it is a rig "
              "already moving, and no public dataset starts that way.");
DEFINE_bool(header, false, "Print a column header before the row.");
DEFINE_bool(dispatch, false,
            "Drive the whole `InitDispatcher` -- detector, window, Stage A, "
            "Stage B and the sanity gates -- configured from the `dynamic_init` "
            "block of `-cfg`, and report which path it took and what it cost in "
            "milliseconds. Without this the probe asks only the detector, which "
            "is what the M1 verdict claims are made with. Two uses: reading a "
            "sequence's `why` when initialization misbehaves, and measuring the "
            "one-off compute the feature adds, which cannot be resolved from the "
            "wall clock of a whole run. Still constructs no Estimator, so the "
            "cost measured here is the feature's alone.");
DEFINE_int32(repeat, 1,
             "In `-dispatch` mode, run the whole initialization this many times "
             "and report the mean cost. A single solve is a few tens of "
             "milliseconds and shares a machine with whatever else is running.");

using namespace xivo;

namespace {
/// True if the flag was given on the command line, as opposed to left at its
/// default. In `-dispatch` mode the config is the source of truth, so a flag may
/// only override it when the caller actually set it.
bool WasGiven(const char *name) {
  gflags::CommandLineFlagInfo info;
  return gflags::GetCommandLineFlagInfo(name, &info) && !info.is_default;
}

/// `X.Wbc` as the estimator reads it: a rotation vector if it has three
/// elements, a row-major matrix otherwise (estimator.cpp, `Estimator::Estimator`).
Mat3 RbcFromJson(const Json::Value &X) {
  try {
    return SO3::exp(GetVectorFromJson<number_t, 3>(X, "Wbc")).matrix();
  } catch (const Json::LogicError &) {
    Mat3 R = GetMatrixFromJson<number_t, 3, 3>(X, "Wbc", JsonMatLayout::RowMajor);
    return (Sophus::isOrthogonal(R) && R.determinant() > 0.0)
               ? R
               : SO3::fitToSO3(R).matrix();
  }
}

/** `-dispatch`: run the real initializer and report the path it chose, the seed
 *  it produced and what the whole thing cost.
 *
 *  IMU samples are fed raw, as `linear_probe` and the detector branch also do.
 *  The estimator feeds `Cg`/`Ca`-calibrated samples; on EuRoC `imu_calib` is the
 *  identity, so the two are the same numbers, and a config where they are not
 *  would need this to apply the calibration.
 */
int RunDispatch(const Json::Value &cfg, DataLoader &loader, number_t t_cut) {
  const Json::Value dyn = cfg.get("dynamic_init", Json::Value());
  auto opt = InitDispatcher::OptionsFromJson(
      dyn, RbcFromJson(cfg["X"]),
      GetVectorFromJson<number_t, 3>(cfg["X"], "Tbc"),
      GetMatrixFromJson<number_t, 3, 1>(cfg, "gravity").norm());
  // The config wins unless the caller explicitly overrode a threshold, so
  // `-dispatch` reports what the shipped configuration does rather than what this
  // binary's flag defaults happen to be.
  if (WasGiven("window"))
    opt.detect.window_sec = FLAGS_window;
  if (WasGiven("horizon"))
    opt.detect.horizon_sec = FLAGS_horizon;
  if (WasGiven("flow_thresh"))
    opt.detect.flow_thresh = FLAGS_flow_thresh;
  if (WasGiven("imu_thresh"))
    opt.detect.imu_thresh = FLAGS_imu_thresh;

  number_t buf = 0, slv = 0;
  int images = 0, frames = 0;
  InitDecision last;
  const int reps = std::max(1, FLAGS_repeat);
  for (int r = 0; r < reps; ++r) {
    InitDispatcher disp{opt};
    number_t t0 = -1;
    for (int i = 0; i < loader.size() && disp.waiting(); ++i) {
      auto *msg = loader.Get(i);
      const number_t t = std::chrono::duration<number_t>(msg->ts_).count();
      if (t < t_cut)
        continue;
      if (auto *imu = dynamic_cast<msg::IMU *>(msg)) {
        if (t0 < 0)
          t0 = t;
        disp.AddImu(t - t0, imu->gyro_, imu->accel_);
      } else if (auto *im = dynamic_cast<msg::Image *>(msg)) {
        if (t0 < 0)
          continue;
        // Read outside the timed section on purpose: image decode is a cost the
        // estimator pays with or without this feature.
        cv::Mat gray = cv::imread(im->image_path_, cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
          LOG(WARNING) << "failed to read " << im->image_path_;
          continue;
        }
        disp.AddImage(t - t0, gray);
      } else {
        continue;
      }
      disp.Decide();
    }
    buf += disp.buffer_ms();
    slv += disp.solve_ms();
    images += disp.num_images();
    frames += disp.num_frames();
    last = disp.decision();
  }
  buf /= reps;
  slv /= reps;

  const char *path = last.path == InitDecision::Path::kDynamic  ? "dynamic"
                     : last.path == InitDecision::Path::kStatic ? "static"
                                                                : "waiting";
  if (FLAGS_header)
    printf("%-26s %8s %7s %7s %8s %8s %8s %8s %7s %8s %5s  %s\n", "sequence",
           "path", "frames", "images", "buf_ms", "slv_ms", "tot_ms", "ms/img",
           "pmed", "|v|", "it", "why");
  printf("%-26s %8s %7d %7d %8.1f %8.1f %8.1f %8.3f %7.3f %8.4f %5d  %s\n",
         FLAGS_seq.c_str(), path, frames / reps, images / reps, buf, slv,
         buf + slv, images ? buf / (images / reps) : 0.0,
         last.stage_b.pixel_median, last.Vsb.norm(), last.stage_b.iterations,
         last.why);
  // buf is the whole cost of buffering (both KLTs), which is new work: while the
  // dispatcher holds the messages the estimator runs nothing at all. slv runs
  // once. Neither recurs -- see `num_images()` in init_dispatch.h.
  return last.path == InitDecision::Path::kWaiting ? 1 : 0;
}
} // namespace

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto cfg = LoadJson(FLAGS_cfg);
  Camera::Create(cfg["camera_cfg"], FLAGS_cam_id);

  std::string image_dir, imu_dir, mocap_dir;
  std::tie(image_dir, imu_dir, mocap_dir) =
      GetDirs(FLAGS_dataset, FLAGS_root, FLAGS_seq, FLAGS_cam_id);
  // Monocular loader on purpose: the detector reads one camera, and pairing the
  // stereo streams would only drop frames whose partner is missing.
  DataLoader loader{image_dir, imu_dir};

  MotionDetector::Options opt;
  opt.window_sec = FLAGS_window;
  opt.horizon_sec = FLAGS_horizon;
  opt.flow_thresh = FLAGS_flow_thresh;
  opt.imu_thresh = FLAGS_imu_thresh;
  opt.cam_id = FLAGS_cam_id;
  // Only the gyro-bias hint needs this; the verdict does not.
  opt.Rbc = RbcFromJson(cfg["X"]);
  MotionDetector det(opt);

  number_t t0 = -1, t_decided = -1;
  MotionVerdict v;
  // `-start` is measured from the first IMU sample of the whole sequence, which
  // has to be found before anything can be dropped relative to it.
  number_t t_first = -1;
  for (int i = 0; i < loader.size() && t_first < 0; ++i)
    if (dynamic_cast<msg::IMU *>(loader.Get(i)) != nullptr)
      t_first = std::chrono::duration<number_t>(loader.Get(i)->ts_).count();
  const number_t t_cut = t_first + FLAGS_start;

  if (FLAGS_dispatch)
    return RunDispatch(cfg, loader, t_cut);
  for (int i = 0; i < loader.size(); ++i) {
    auto *msg = loader.Get(i);
    // timestamp_t is a duration (nanoseconds), not a time_point.
    const number_t t = std::chrono::duration<number_t>(msg->ts_).count();
    if (t < t_cut)
      continue;
    if (auto *imu = dynamic_cast<msg::IMU *>(msg)) {
      if (t0 < 0)
        t0 = t;
      det.AddImu(t - t0, imu->gyro_, imu->accel_);
    } else if (auto *im = dynamic_cast<msg::Image *>(msg)) {
      // Images that arrive before the first IMU sample cannot be part of the
      // initialization window: there is no IMU interval to compare them against.
      if (t0 < 0)
        continue;
      cv::Mat gray = cv::imread(im->image_path_, cv::IMREAD_GRAYSCALE);
      if (gray.empty()) {
        LOG(WARNING) << "failed to read " << im->image_path_;
        continue;
      }
      det.AddImage(t - t0, gray);
    } else {
      continue;
    }
    if (det.Ready()) {
      v = det.Classify();
      t_decided = t - t0;
      break;
    }
  }

  if (FLAGS_header)
    printf("%-26s %9s %8s %9s %8s %6s %10s\n", "sequence", "verdict", "t_dec",
           "flow_px", "accel_sd", "pairs", "bias_hint");
  printf("%-26s %9s %8.3f %9.4f %8.4f %6d %10.4f\n", FLAGS_seq.c_str(),
         v.KindName(), t_decided, v.flow_px, v.accel_sd, v.frame_pairs,
         v.gyro_bias_hint);
  // `t_dec` above is relative to `-start`, so it is the latency the estimator
  // pays, not a position in the file.
  return v.kind == MotionVerdict::kUndecided ? 1 : 0;
}
