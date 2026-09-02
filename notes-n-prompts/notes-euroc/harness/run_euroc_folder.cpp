/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// ROS-free serial runner over an ASL/EuRoC-format dataset folder (mav0/...).
//
// This is the non-ROS equivalent of ros1_serial_msckf.cpp: instead of a rosbag it
// reads mav0/imu0/data.csv and mav0/cam{0,1}/data.csv + image folders, and replays
// them into VioManager in timestamp order using exactly the same buffering rule as
// ROS1Visualizer (a camera frame is only fed once an IMU sample past it has been
// seen). It writes the IMU-frame trajectory in TUM format plus per-frame timing.
//
// Added for the auto-slam-engineer benchmark harness; not part of upstream OpenVINS.

#include <algorithm>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

using namespace ov_msckf;

struct ImageEntry {
  double timestamp;
  std::vector<std::string> paths; // one per camera
};

// Split a csv line on commas, trimming whitespace
static std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t b = item.find_first_not_of(" \t\r\n");
    size_t e = item.find_last_not_of(" \t\r\n");
    out.push_back(b == std::string::npos ? "" : item.substr(b, e - b + 1));
  }
  return out;
}

// Load mav0/imu0/data.csv -> [ts(ns), wx,wy,wz, ax,ay,az]
static std::vector<ov_core::ImuData> load_imu(const std::string &path) {
  std::vector<ov_core::ImuData> data;
  std::ifstream file(path);
  if (!file.is_open()) {
    PRINT_ERROR(RED "unable to open imu file: %s\n" RESET, path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line.at(0) == '#')
      continue;
    auto v = split_csv(line);
    if (v.size() < 7)
      continue;
    ov_core::ImuData meas;
    meas.timestamp = 1e-9 * std::stod(v.at(0));
    meas.wm << std::stod(v.at(1)), std::stod(v.at(2)), std::stod(v.at(3));
    meas.am << std::stod(v.at(4)), std::stod(v.at(5)), std::stod(v.at(6));
    data.push_back(meas);
  }
  std::sort(data.begin(), data.end());
  return data;
}

// Load mav0/camX/data.csv -> [ts(ns), filename]
static std::vector<std::pair<double, std::string>> load_camera(const std::string &folder) {
  std::vector<std::pair<double, std::string>> data;
  std::string path = folder + "/data.csv";
  std::ifstream file(path);
  if (!file.is_open()) {
    PRINT_ERROR(RED "unable to open camera file: %s\n" RESET, path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line.at(0) == '#')
      continue;
    auto v = split_csv(line);
    if (v.size() < 2)
      continue;
    data.emplace_back(1e-9 * std::stod(v.at(0)), folder + "/data/" + v.at(1));
  }
  std::sort(data.begin(), data.end());
  return data;
}

int main(int argc, char **argv) {

  // ---------------------------------------------------------------- arguments
  if (argc < 2) {
    printf("usage: run_euroc_folder <config.yaml> --dataset <dir containing mav0> [options]\n");
    printf("options:\n");
    printf("  --traj <file>            output trajectory, TUM format (default: none)\n");
    printf("  --timing <file>          output per-frame timing csv (default: none)\n");
    printf("  --stats <file>           output run summary as key=value lines (default: none)\n");
    printf("  --max_cameras <n>        override max_cameras (1 = mono, 2 = stereo)\n");
    printf("  --use_stereo <0|1>       override use_stereo\n");
    printf("  --init_imu_thresh <x>    override init_imu_thresh (room6 wants 0.25)\n");
    printf("  --init_max_disparity <x> override init_max_disparity, the pixel\n");
    printf("                           disparity below which the platform is called\n");
    printf("                           stationary. Depth-dependent: EuRoC MH_04\n");
    printf("                           hovers 0.4 m/s in a far scene and still\n");
    printf("                           reads as still at the shipped 10.0.\n");
    printf("  --init_dyn_use <0|1>     override init_dyn_use, the moving-start\n");
    printf("                           (dynamic MLE) initializer\n");
    printf("  --gravity_mag <x>        override gravity_mag (useful as a neutral knob\n");
    printf("                           to probe run-to-run sensitivity)\n");
    printf("  --num_opencv_threads <n> override num_opencv_threads\n");
    printf("  --start <sec>            skip this many seconds from the start\n");
    printf("  --duration <sec>         only play this many seconds (default: all)\n");
    printf("  --verbosity <level>      ALL, DEBUG, INFO, WARNING, ERROR, SILENT\n");
    return EXIT_FAILURE;
  }
  std::string config_path = argv[1];
  std::string path_dataset, path_traj, path_timing, path_stats, verbosity_override;
  int opt_max_cameras = -1, opt_use_stereo = -1, opt_cv_threads = -99;
  double opt_init_imu_thresh = -1, opt_gravity_mag = -1, play_start = 0.0, play_duration = -1.0;
  double opt_init_max_disparity = -1;
  int opt_init_dyn_use = -1;
  for (int i = 2; i < argc; i++) {
    std::string key = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        PRINT_ERROR(RED "missing value for %s\n" RESET, key.c_str());
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (key == "--dataset")
      path_dataset = next();
    else if (key == "--traj")
      path_traj = next();
    else if (key == "--timing")
      path_timing = next();
    else if (key == "--stats")
      path_stats = next();
    else if (key == "--max_cameras")
      opt_max_cameras = std::stoi(next());
    else if (key == "--use_stereo")
      opt_use_stereo = std::stoi(next());
    else if (key == "--init_imu_thresh")
      opt_init_imu_thresh = std::stod(next());
    else if (key == "--init_max_disparity")
      opt_init_max_disparity = std::stod(next());
    else if (key == "--init_dyn_use")
      opt_init_dyn_use = std::stoi(next());
    else if (key == "--gravity_mag")
      opt_gravity_mag = std::stod(next());
    else if (key == "--num_opencv_threads")
      opt_cv_threads = std::stoi(next());
    else if (key == "--start")
      play_start = std::stod(next());
    else if (key == "--duration")
      play_duration = std::stod(next());
    else if (key == "--verbosity")
      verbosity_override = next();
    else {
      PRINT_ERROR(RED "unknown argument: %s\n" RESET, key.c_str());
      return EXIT_FAILURE;
    }
  }
  if (path_dataset.empty()) {
    PRINT_ERROR(RED "--dataset is required\n" RESET);
    return EXIT_FAILURE;
  }
  while (path_dataset.size() > 1 && path_dataset.back() == '/')
    path_dataset.pop_back();

  // ------------------------------------------------------------------- config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  if (!verbosity_override.empty())
    verbosity = verbosity_override;
  ov_core::Printer::setPrintLevel(verbosity);

  VioManagerOptions params;
  params.print_and_load(parser);
  if (opt_max_cameras > 0) {
    params.state_options.num_cameras = opt_max_cameras;
    params.init_options.num_cameras = opt_max_cameras;
  }
  if (opt_use_stereo >= 0) {
    params.use_stereo = (opt_use_stereo != 0);
  }
  if (opt_init_imu_thresh > 0) {
    params.init_options.init_imu_thresh = opt_init_imu_thresh;
  }
  if (opt_init_max_disparity > 0) {
    params.init_options.init_max_disparity = opt_init_max_disparity;
  }
  if (opt_init_dyn_use >= 0) {
    params.init_options.init_dyn_use = (opt_init_dyn_use != 0);
  }
  if (opt_gravity_mag > 0) {
    params.gravity_mag = opt_gravity_mag;
    params.init_options.gravity_mag = opt_gravity_mag;
  }
  if (opt_cv_threads != -99) {
    params.num_opencv_threads = opt_cv_threads;
  }
  // We replay serially, so no async subscriber / publisher threads.
  params.use_multi_threading_subs = false;
  params.use_multi_threading_pubs = false;
  if (!parser->successful()) {
    PRINT_ERROR(RED "[EUROC]: unable to parse all parameters, please fix\n" RESET);
    return EXIT_FAILURE;
  }
  int num_cameras = params.state_options.num_cameras;
  if (num_cameras != 1 && num_cameras != 2) {
    PRINT_ERROR(RED "[EUROC]: only 1 or 2 cameras supported (got %d)\n" RESET, num_cameras);
    return EXIT_FAILURE;
  }

  auto sys = std::make_shared<VioManager>(params);

  // --------------------------------------------------------------- load index
  std::string folder_mav0 = path_dataset + "/mav0";
  if (!boost::filesystem::exists(folder_mav0)) {
    // allow being pointed directly at a mav0 folder
    folder_mav0 = path_dataset;
  }
  std::vector<ov_core::ImuData> imu_data = load_imu(folder_mav0 + "/imu0/data.csv");
  std::vector<std::vector<std::pair<double, std::string>>> cam_data;
  for (int i = 0; i < num_cameras; i++) {
    cam_data.push_back(load_camera(folder_mav0 + "/cam" + std::to_string(i)));
  }
  if (imu_data.empty() || cam_data.at(0).empty()) {
    PRINT_ERROR(RED "[EUROC]: no imu or camera data found under %s\n" RESET, folder_mav0.c_str());
    return EXIT_FAILURE;
  }

  // Group the cameras into synchronized frames (same rule as ros1_serial_msckf:
  // a frame needs a match in every camera within 20ms, else it is skipped).
  std::vector<ImageEntry> frames;
  size_t num_unsynced = 0;
  for (const auto &ref : cam_data.at(0)) {
    ImageEntry entry;
    entry.timestamp = ref.first;
    entry.paths.push_back(ref.second);
    for (int i = 1; i < num_cameras; i++) {
      const auto &other = cam_data.at(i);
      auto it = std::lower_bound(other.begin(), other.end(), std::make_pair(ref.first - 0.02, std::string()));
      if (it != other.end() && std::abs(it->first - ref.first) < 0.02) {
        entry.paths.push_back(it->second);
      }
    }
    if ((int)entry.paths.size() != num_cameras) {
      num_unsynced++;
      continue;
    }
    frames.push_back(entry);
  }
  if (num_unsynced > 0) {
    PRINT_WARNING(YELLOW "[EUROC]: skipped %zu unsynchronized camera frames\n" RESET, num_unsynced);
  }

  // Apply the requested play window, relative to the first camera frame
  double time_first = frames.at(0).timestamp;
  double time_start = time_first + play_start;
  double time_end = (play_duration < 0) ? std::numeric_limits<double>::max() : time_start + play_duration;
  PRINT_INFO("[EUROC]: %zu imu, %zu frames (%d cam), %.1f sec of data\n", imu_data.size(), frames.size(), num_cameras,
             frames.back().timestamp - time_first);

  // --------------------------------------------------------------- output files
  std::ofstream of_traj, of_timing;
  if (!path_traj.empty()) {
    boost::filesystem::path p(path_traj);
    if (!p.parent_path().empty())
      boost::filesystem::create_directories(p.parent_path());
    of_traj.open(path_traj);
    of_traj << "# timestamp(s) tx ty tz qx qy qz qw" << std::endl;
    of_traj << std::fixed;
  }
  if (!path_timing.empty()) {
    boost::filesystem::path p(path_timing);
    if (!p.parent_path().empty())
      boost::filesystem::create_directories(p.parent_path());
    of_timing.open(path_timing);
    of_timing << "# frame timestamp(s) time_track_and_update(s) time_imread(s) initialized" << std::endl;
    of_timing << std::fixed;
  }

  // --------------------------------------------------------------- replay loop
  std::deque<ov_core::CameraData> camera_queue;
  std::map<int, double> camera_last_timestamp;
  std::vector<double> times_update;
  double time_imread_total = 0.0;
  size_t frame_index = 0, num_processed = 0, num_dropped_rate = 0;
  double first_init_time = -1, last_frame_time = -1;

  auto wall_start = boost::posix_time::microsec_clock::local_time();

  // Feed a camera frame that is now safe to process (an IMU past it has arrived)
  auto process_camera = [&](const ov_core::CameraData &message) {
    auto rT1 = boost::posix_time::microsec_clock::local_time();
    sys->feed_measurement_camera(message);
    auto rT2 = boost::posix_time::microsec_clock::local_time();
    double dt = (rT2 - rT1).total_microseconds() * 1e-6;
    times_update.push_back(dt);
    num_processed++;
    last_frame_time = message.timestamp;

    // Record the state (IMU pose in global), in the IMU clock frame
    std::shared_ptr<State> state = sys->get_state();
    bool inited = sys->initialized();
    if (inited && first_init_time < 0)
      first_init_time = state->_timestamp;
    if (inited && of_traj.is_open()) {
      double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
      double timestamp_inI = state->_timestamp + t_ItoC;
      Eigen::Vector4d q = state->_imu->quat(); // JPL q_GtoI == Hamilton q_ItoG components
      Eigen::Vector3d p = state->_imu->pos();
      of_traj << std::setprecision(9) << timestamp_inI << " " << std::setprecision(9) << p(0) << " " << p(1) << " " << p(2) << " " << q(0)
              << " " << q(1) << " " << q(2) << " " << q(3) << std::endl;
    }
    if (of_timing.is_open()) {
      of_timing << (num_processed - 1) << " " << std::setprecision(9) << message.timestamp << " " << std::setprecision(6) << dt << " " << 0.0
                << " " << (inited ? 1 : 0) << std::endl;
    }
  };

  for (size_t i = 0; i < imu_data.size(); i++) {

    const ov_core::ImuData &meas = imu_data.at(i);
    if (meas.timestamp < time_start || meas.timestamp > time_end)
      continue;

    // Queue up any camera frames whose "message" has now arrived (bag order)
    while (frame_index < frames.size() && frames.at(frame_index).timestamp <= meas.timestamp) {
      const ImageEntry &entry = frames.at(frame_index);
      frame_index++;
      if (entry.timestamp < time_start || entry.timestamp > time_end)
        continue;

      // Same frame-rate throttle as the ROS callbacks
      double time_delta = 1.0 / params.track_frequency;
      if (camera_last_timestamp.find(0) != camera_last_timestamp.end() && entry.timestamp < camera_last_timestamp.at(0) + time_delta) {
        num_dropped_rate++;
        continue;
      }
      camera_last_timestamp[0] = entry.timestamp;

      ov_core::CameraData message;
      message.timestamp = entry.timestamp;
      auto rT1 = boost::posix_time::microsec_clock::local_time();
      for (int cam_id = 0; cam_id < num_cameras; cam_id++) {
        cv::Mat img = cv::imread(entry.paths.at(cam_id), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
          PRINT_ERROR(RED "[EUROC]: unable to read image %s\n" RESET, entry.paths.at(cam_id).c_str());
          std::exit(EXIT_FAILURE);
        }
        message.sensor_ids.push_back(cam_id);
        message.images.push_back(img);
        if (params.use_mask) {
          message.masks.push_back(params.masks.at(cam_id));
        } else {
          message.masks.push_back(cv::Mat::zeros(img.rows, img.cols, CV_8UC1));
        }
      }
      auto rT2 = boost::posix_time::microsec_clock::local_time();
      time_imread_total += (rT2 - rT1).total_microseconds() * 1e-6;
      camera_queue.push_back(message);
      std::sort(camera_queue.begin(), camera_queue.end());
    }

    // Feed the inertial reading
    sys->feed_measurement_imu(meas);

    // Then drain any camera frames that this IMU reading has made processable.
    // (identical condition to ROS1Visualizer::callback_inertial)
    double timestamp_imu_inC = meas.timestamp - sys->get_state()->_calib_dt_CAMtoIMU->value()(0);
    while (!camera_queue.empty() && camera_queue.at(0).timestamp < timestamp_imu_inC) {
      process_camera(camera_queue.at(0));
      camera_queue.pop_front();
    }

    // Stop once we have consumed every frame we were going to
    if (frame_index >= frames.size() && camera_queue.empty())
      break;
  }

  auto wall_end = boost::posix_time::microsec_clock::local_time();
  double wall_total = (wall_end - wall_start).total_microseconds() * 1e-6;

  // ------------------------------------------------------------------ summary
  double sum = 0, max_dt = 0;
  for (double dt : times_update) {
    sum += dt;
    max_dt = std::max(max_dt, dt);
  }
  double mean_dt = times_update.empty() ? 0 : sum / (double)times_update.size();
  std::vector<double> sorted = times_update;
  std::sort(sorted.begin(), sorted.end());
  auto pct = [&](double q) { return sorted.empty() ? 0.0 : sorted.at(std::min(sorted.size() - 1, (size_t)(q * sorted.size()))); };
  // Duration of the data we actually played (not of the whole dataset)
  double data_duration = (last_frame_time > 0 ? last_frame_time : time_first) - std::max(time_start, time_first);
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  double peak_rss_mb = usage.ru_maxrss / 1024.0;

  PRINT_INFO(REDPURPLE "\n[EUROC]: processed %zu frames (%zu rate-dropped, %zu unsynced)\n" RESET, num_processed, num_dropped_rate,
             num_unsynced);
  PRINT_INFO(REDPURPLE "[EUROC]: track+update  mean %.2f ms (%.2f hz) | median %.2f | p95 %.2f | max %.2f ms\n" RESET, 1e3 * mean_dt,
             (mean_dt > 0 ? 1.0 / mean_dt : 0.0), 1e3 * pct(0.5), 1e3 * pct(0.95), 1e3 * max_dt);
  PRINT_INFO(REDPURPLE "[EUROC]: wall %.2f sec for %.2f sec of data (%.2fx realtime), imread %.2f sec, peak rss %.1f MB\n" RESET, wall_total,
             data_duration, (wall_total > 0 ? data_duration / wall_total : 0.0), time_imread_total, peak_rss_mb);

  if (!path_stats.empty()) {
    std::ofstream of(path_stats);
    of << std::fixed << std::setprecision(6);
    of << "frames_processed=" << num_processed << "\n";
    of << "frames_rate_dropped=" << num_dropped_rate << "\n";
    of << "frames_unsynced=" << num_unsynced << "\n";
    of << "data_duration_s=" << data_duration << "\n";
    of << "wall_total_s=" << wall_total << "\n";
    of << "wall_imread_s=" << time_imread_total << "\n";
    of << "update_mean_ms=" << 1e3 * mean_dt << "\n";
    of << "update_median_ms=" << 1e3 * pct(0.5) << "\n";
    of << "update_p95_ms=" << 1e3 * pct(0.95) << "\n";
    of << "update_max_ms=" << 1e3 * max_dt << "\n";
    of << "fps_mean=" << (mean_dt > 0 ? 1.0 / mean_dt : 0.0) << "\n";
    of << "fps_median=" << (pct(0.5) > 0 ? 1.0 / pct(0.5) : 0.0) << "\n";
    of << "realtime_factor=" << (wall_total > 0 ? data_duration / wall_total : 0.0) << "\n";
    of << "peak_rss_mb=" << peak_rss_mb << "\n";
    of << "init_time_s=" << (first_init_time < 0 ? -1.0 : first_init_time - time_first) << "\n";
    of << "num_cameras=" << num_cameras << "\n";
    of << "use_stereo=" << (params.use_stereo ? 1 : 0) << "\n";
    of << "num_opencv_threads=" << params.num_opencv_threads << "\n";
    of.close();
  }

  if (num_processed == 0 || first_init_time < 0) {
    PRINT_ERROR(RED "[EUROC]: system never initialized!\n" RESET);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
