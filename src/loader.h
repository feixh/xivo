// Dataloader for ASL-compatible dataset.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "core.h"
#include "message_types.h"

namespace xivo {

class DataLoader {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DataLoader(const std::string &image_dir, const std::string &imu_dir);
  DataLoader(const std::string &image_dir);
  /** Stereo: pair the two cameras' frames by timestamp and interleave with IMU.
   *
   * Emits `msg::StereoImage` rather than `msg::Image`, so a consumer written for
   * the monocular loader will not silently treat a stereo pair as a left-only
   * frame -- its `dynamic_cast<msg::Image *>` simply will not match.
   *
   * A frame present in one camera but not the other is dropped, with a count
   * logged. Two frames pair up only when their timestamps are *exactly* equal:
   * TUM-VI triggers both cameras together, so any difference means the two
   * directories do not belong to the same recording and a tolerance would paper
   * over that.
   */
  DataLoader(const std::string &image_dir, const std::string &image_dir_r,
             const std::string &imu_dir);
  std::vector<msg::Pose> LoadGroundTruthState(const std::string &state_dir);

  msg::Message *Get(int i) const { return entries_[i].get(); };
  int size() const { return entries_.size(); }

private:
  std::vector<std::unique_ptr<msg::Message>> entries_;
  std::vector<msg::Pose> poses_;
};

using TUMVILoader = DataLoader;
using EuRoCLoader = DataLoader;

// Get image, imu and groundtruth directories for TUMVI and EuRoC dataset
std::tuple<std::string, std::string, std::string>
GetDirs(const std::string dataset, const std::string root,
        const std::string seq, int cam_id);

/** Image directory of the *other* camera of a stereo pair.
 *
 * Derived from the primary camera's directory by swapping the `camN` component,
 * so it cannot drift from `GetDirs`' own path construction.
 */
std::string StereoPairDir(const std::string &image_dir, int cam_id,
                          int cam_id_r);

} // namespace xivo
