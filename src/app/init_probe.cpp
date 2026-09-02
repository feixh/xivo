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
#include <cstdio>
#include <memory>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/imgcodecs.hpp"

#include "camera_manager.h"
#include "init_detect.h"
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
  opt.Rbc = GetMatrixFromJson<number_t, 3, 3>(cfg["X"], "Wbc",
                                              JsonMatLayout::RowMajor);
  MotionDetector det(opt);

  number_t t0 = -1, t_decided = -1;
  MotionVerdict v;
  for (int i = 0; i < loader.size(); ++i) {
    auto *msg = loader.Get(i);
    // timestamp_t is a duration (nanoseconds), not a time_point.
    const number_t t = std::chrono::duration<number_t>(msg->ts_).count();
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
  return v.kind == MotionVerdict::kUndecided ? 1 : 0;
}
