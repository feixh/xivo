// Author: Parth Agrawal
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
#include "pngfast.h"
#include "viewer.h"
#include "visualize.h"
#include "graphwriter.h"

// flags
DEFINE_string(cfg, "cfg/tracker_only.json",
              "Configuration file for the VIO application.");
DEFINE_string(root, "xivo_dataset/tumvi",
              "Root directory containing tumvi dataset folder.");
DEFINE_string(dataset, "tumvi", "xivo | euroc | tumvi");
DEFINE_string(seq, "room1", "Sequence of TUM VI benchmark to play with.");
DEFINE_int32(cam_id, 0, "Camera id.");
DEFINE_string(out, "out_state", "Output file path.");
DEFINE_string(graphout, "", ".dot file to save output graph to");

using namespace xivo;


int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto cfg = LoadJson(FLAGS_cfg);
  bool verbose = cfg.get("verbose", false).asBool();

  std::string image_dir, imu_dir, mocap_dir;
  std::tie(image_dir, imu_dir, mocap_dir) =
      GetDirs(FLAGS_dataset, FLAGS_root, FLAGS_seq, FLAGS_cam_id);

  bool tracker_only = true;

  // read the data
  std::unique_ptr<DataLoader> loader(new DataLoader{image_dir});

  // create estimator
  auto est = CreateSystemTrackerOnly(
      LoadJson(cfg["estimator_cfg"].asString()));

  // create viewer
  std::unique_ptr<Viewer> viewer;
  if (cfg.get("visualize", false).asBool()) {
    viewer = std::make_unique<Viewer>(
        LoadJson(cfg["viewer_cfg"].asString()),
        FLAGS_seq,
        tracker_only);
  }

  // setup I/O for saving results
  if (std::ofstream ostream{FLAGS_out, std::ios::out}) {

    for (int i = 0; i < loader->size(); ++i) {
      auto raw_msg = loader->Get(i);

      if (verbose && i % 1000 == 0) {
        std::cout << i << "/" << loader->size() << std::endl;
      }

      if (auto msg = dynamic_cast<msg::Image *>(raw_msg)) {
        // One channel is all the tracker uses; see pybind11/pyxivo.cpp's
        // ReadImage() and notes-speed/m1-grayscale.md. Same decoder as the
        // other two drivers, fast PNG path included (src/pngfast.h).
        auto image = ReadGrayImage(msg->image_path_);
        est->VisualMeasTrackerOnly(msg->ts_, image);

        if (viewer) {

          cv::Mat disp = Canvas::instance()->display();

          if (!disp.empty()) {
            LOG(INFO) << "Display image is ready";
            viewer->Update(disp);
            viewer->Refresh();
          }
        }
      }
      else {
        LOG(INFO) << "Invalid entry type.";
      }
    }

  } else {
    LOG(FATAL) << "failed to open output file @ " << FLAGS_out;
  }
}
