// Author: Xiaohan Fei
#include "unistd.h"
#include <algorithm>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/highgui/highgui.hpp"

#include "estimator.h"
#include "estimator_process.h"
#include "metrics.h"
#include "tracker.h"
#include "loader.h"
#include "viewer.h"
#include "visualize.h"
#include "graphwriter.h"

// flags
DEFINE_string(cfg, "cfg/vio.json",
              "Configuration file for the VIO application.");
DEFINE_string(root, "/home/feixh/Data/tumvi/exported/euroc/512_16/",
              "Root directory containing tumvi dataset folder.");
DEFINE_string(dataset, "tumvi", "xivo | euroc | tumvi");
DEFINE_string(seq, "room1", "Sequence of TUM VI benchmark to play with.");
DEFINE_int32(cam_id, 0, "Camera id.");
DEFINE_string(out, "out_state", "Output file path.");
DEFINE_string(graphout, "", ".dot file to save output graph to");
DEFINE_int32(max_entries, 0,
             "Stop after this many dataset entries; 0 (default) plays the "
             "whole sequence. Useful for short runs under a sanitizer.");

using namespace xivo;


int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto cfg = LoadJson(FLAGS_cfg);
  bool verbose = cfg.get("verbose", false).asBool();

  std::string image_dir, imu_dir, mocap_dir;
  std::tie(image_dir, imu_dir, mocap_dir) =
      GetDirs(FLAGS_dataset, FLAGS_root, FLAGS_seq, FLAGS_cam_id);

  // The estimator config decides whether this is a stereo run, so read it before
  // choosing a loader.
  auto est_cfg = LoadJson(cfg["estimator_cfg"].asString());
  const bool stereo = est_cfg.get("stereo", false).asBool();

  std::unique_ptr<DataLoader> loader;
  if (stereo) {
    const int cam_id_r = est_cfg.get("cam_id_right", 1).asInt();
    std::string image_dir_r =
        StereoPairDir(image_dir, FLAGS_cam_id, cam_id_r);
    LOG(INFO) << "stereo run: left=" << image_dir << " right=" << image_dir_r;
    loader.reset(new DataLoader{image_dir, image_dir_r, imu_dir});
  } else {
    loader.reset(new DataLoader{image_dir, imu_dir});
  }

  // create estimator
  // auto est = std::make_unique<Estimator>(
  //     LoadJson(cfg["estimator_cfg"].asString()));
  auto est = CreateSystem(est_cfg);

  // create viewer
  std::unique_ptr<Viewer> viewer;
  if (cfg.get("visualize", false).asBool()) {
    viewer = std::make_unique<Viewer>(
        LoadJson(cfg["viewer_cfg"].asString()), FLAGS_seq);
  }

  // setup I/O for saving results
  if (std::ofstream ostream{FLAGS_out, std::ios::out}) {

    int num_entries = loader->size();
    if (FLAGS_max_entries > 0 && FLAGS_max_entries < num_entries) {
      num_entries = FLAGS_max_entries;
    }

    for (int i = 0; i < num_entries; ++i) {
      auto raw_msg = loader->Get(i);

      if (verbose && i % 1000 == 0) {
        std::cout << i << "/" << num_entries << std::endl;
      }

      // `StereoImage` does not derive from `Image`, so the two branches are
      // mutually exclusive and a stereo pair cannot be mistaken for a left-only
      // frame.
      bool did_visual = false;
      // IMREAD_GRAYSCALE, not the default IMREAD_COLOR: the estimator only ever
      // uses one channel, and decoding a grayscale PNG into 8UC3 makes the
      // decode, the pyramid and the KLT solve carry three identical planes.
      // Kept in step with pybind11/pyxivo.cpp's ReadImage(); see
      // notes-speed/m1-grayscale.md.
      if (auto msg = dynamic_cast<msg::StereoImage *>(raw_msg)) {
        auto image = cv::imread(msg->image_path_, cv::IMREAD_GRAYSCALE);
        auto image_r = cv::imread(msg->image_path_r_, cv::IMREAD_GRAYSCALE);
        est->VisualMeasStereo(msg->ts_, image, image_r);
        did_visual = true;
      } else if (auto msg = dynamic_cast<msg::Image *>(raw_msg)) {
        auto image = cv::imread(msg->image_path_, cv::IMREAD_GRAYSCALE);
        est->VisualMeas(msg->ts_, image);
        did_visual = true;
      }

      if (did_visual) {
        if (est->UsingLoopClosure()) {
          est->CloseLoop();
        }

        if (viewer) {
          viewer->Update_gsb(est->gsb());
          viewer->Update_gsc(est->gsc());

          cv::Mat disp = Canvas::instance()->display();

          if (!disp.empty()) {
            LOG(INFO) << "Display image is ready";
            viewer->Update(disp);
            viewer->Refresh();
          }
        }
      } else if (auto msg = dynamic_cast<msg::IMU *>(raw_msg)) {
        est->InertialMeas(msg->ts_, msg->gyro_, msg->accel_);
        // if (viewer) {
        //   viewer->Update_gsb(est->gsb());
        //   viewer->Update_gsc(est->gsc());
        // }
      } else {
        LOG(FATAL) << "Invalid entry type.";
      }

      // The pose is streamed straight to `ostream`; it used to also be
      // accumulated into a std::vector<msg::Pose> that nothing ever read, which
      // grew by 96 bytes for every dataset entry -- image *and* IMU -- and was
      // the largest single accumulating block in the process (~3.1 MB by the end
      // of room1, ~4.7 MB across the last reallocation).
      Vec3 Tsb = (Vec3)est->gsb().translation();
      Vec3 Wsb = (Vec3)est->gsb().so3().log();
      ostream << StrFormat("%ld", est->ts().count()) << " "
        << Tsb.transpose() << " "
        << Wsb.transpose() << std::endl;

      // std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    // Dump output graph
    if (!FLAGS_graphout.empty()) {
      GraphWriter GW;
      GW.CollectGraph(Graph::instance());
#ifdef USE_MAPPER
      GW.CollectGraph(Mapper::instance());
#endif
      GW.WriteDot(FLAGS_graphout);
    }

  } else {
    LOG(FATAL) << "failed to open output file @ " << FLAGS_out;
  }
  // while (viewer) {
  //   viewer->Refresh();
  //   usleep(30);
  // }
}
