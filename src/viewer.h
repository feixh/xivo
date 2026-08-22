// Viewer for VIO.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

#include <array>
#include <memory>
#include <string>

#include "opencv2/core/core.hpp"
#include "pangolin/pangolin.h"

#include "core.h"

namespace xivo {

using XYZRGB = std::array<float, 6>;

class Viewer {
public:
  Viewer(const Json::Value &cfg, const std::string &name = "", bool tracker_only=false);
  ~Viewer();

  void Update_gsb(const SE3 &gsb);
  void Update_gbc(const SE3 &gbc);
  void Update_gsc(const SE3 &gsc);
  void Update(const cv::Mat &img);
  void Refresh();

private:
  std::string window_name_;
  std::unique_ptr<pangolin::OpenGlRenderState> camera_state_;
  std::unique_ptr<pangolin::OpenGlRenderState> image_state_;
  std::unique_ptr<pangolin::GlTexture> texture_;
  /** pangolin::View stores a *non-owning* Handler*, so whatever is passed to
   *  View::SetHandler has to be owned here. See ~Viewer, which unhooks these
   *  from the views before dropping them. */
  std::unique_ptr<pangolin::Handler3D> image_handler_;
  std::unique_ptr<pangolin::Handler3D> camera_handler_;
  Json::Value cfg_;
  bool tracker_only_;

  // viewport attributes
  int height_, width_;
  Mat3 K_, Kinv_;
  number_t fx_, fy_, cx_, cy_;
  number_t znear_, zfar_;

  cv::Mat image_;

  SE3 Rg_, gsb_, gbc_, gsc_;
  std::vector<Vec3> trace_; // body frame trajectory

  float bg_color_[4]; // background color (rgba)
  static int counter_;
};

} // xivo
