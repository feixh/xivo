// Pangolin backed 2D and 3D viewer.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "glog/logging.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "json/json.h"

#include "core.h"
#include "estimator.h"
#include "viewer.h"

namespace xivo {

const static Vec3f kYellow{1., 1., 0};
const static Vec3f kGreen{0, 1., 0};
const static Vec3f kCyan{0, 1., 1.};

Viewer::~Viewer() {
  if (camera_state_) {
    delete camera_state_;
  }
  if (image_state_) {
    delete image_state_;
  }
  if (texture_) {
    delete texture_;
  }
}

Viewer::Viewer(const Json::Value &cfg, const std::string &name, bool tracker_only)
    : window_name_{name.empty() ? "XIVO Display" : name},
      tracker_only_{tracker_only},
      camera_state_{nullptr}, image_state_{nullptr}, texture_{nullptr},
      cfg_{cfg} {

  pangolin::CreateWindowAndBind(window_name_, cfg_["window"]["width"].asInt(),
                                cfg_["window"]["height"].asInt());

  // aspect ratio setup
  auto viewport = cfg_["viewport"];
  height_ = viewport["height"].asInt();
  width_ = viewport["width"].asInt();
  fx_ = viewport["fx"].asDouble();
  fy_ = viewport["fy"].asDouble();
  cx_ = viewport["cx"].asDouble();
  cy_ = viewport["cy"].asDouble();
  znear_ = viewport["znear"].asDouble();
  zfar_ = viewport["zfar"].asDouble();

  K_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
  Kinv_ = K_.inverse();

  float aspect = width_ / (float)height_;

  // Display Video Sequence
  image_state_ = new pangolin::OpenGlRenderState(
      pangolin::ProjectionMatrix(width_, height_, fx_, fy_, cx_, cy_, znear_,
                                 zfar_),
      pangolin::ModelViewLookAt(-1, 1, -1, 0, 0, 0, pangolin::AxisY));

  pangolin::View &image_view = pangolin::Display("image").SetAspect(aspect);

  image_view.SetHandler(new pangolin::Handler3D(*image_state_));

  // background setup
  bg_color_[0] = cfg_["bg_color"].get("r", 0).asFloat();
  bg_color_[1] = cfg_["bg_color"].get("g", 0).asFloat();
  bg_color_[2] = cfg_["bg_color"].get("b", 0).asFloat();
  bg_color_[3] = cfg_["bg_color"].get("a", 0).asFloat();

  // Display Map + Video Sequence
  if(!tracker_only_) {
    camera_state_ = new pangolin::OpenGlRenderState();

    camera_state_->SetProjectionMatrix(pangolin::ProjectionMatrixRDF_TopLeft(
        width_, height_, fx_, fy_, cx_, cy_, znear_, zfar_));

    auto modelView = cfg_["ModelViewMatrix"];
    Vec3 upVec = GetVectorFromJson<double, 3>(modelView, "upVector");
    camera_state_->SetModelViewMatrix(
      pangolin::ModelViewLookAtRDF(modelView["x"].asDouble(),
                                  modelView["y"].asDouble(),
                                  modelView["z"].asDouble(),
                                  0.0f, 0.0f, 0.0f,
                                  upVec(0), upVec(1), upVec(2)));

    pangolin::View &camera_view = pangolin::Display("cam").SetAspect(aspect);

    camera_view.SetHandler(new pangolin::Handler3D(*camera_state_));

    pangolin::DisplayBase()
        .SetBounds(0, 1, 0, 1)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(camera_view)
        .AddDisplay(image_view);
  }
  else {
    pangolin::DisplayBase()
        .SetBounds(0, 1, 0, 1)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(image_view);
  }

  // NOTE: have to unset the current context from the main thread
  // otherwise segfault
  // Reference:
  // https://github.com/stevenlovegrove/Pangolin/blob/master/examples/HelloPangolinThreads/main.cpp
  pangolin::GetBoundWindow()->RemoveCurrent();
}

void Viewer::Update(const cv::Mat &image) {
  if (image.empty())
    return;

  image_ = image.clone();
  cv::cvtColor(image_, image_, cv::COLOR_RGB2BGR);
  cv::flip(image_, image_, 0);

  int rows(image_.rows);
  int cols(image_.cols);
  // bind to context
  pangolin::BindToContext(window_name_);
  if (!texture_) {
    texture_ = new pangolin::GlTexture(cols, rows, GL_RGB, false, 0, GL_RGB,
                                       GL_UNSIGNED_BYTE);
  }
  texture_->Upload((uint8_t *)image_.data, GL_RGB, GL_UNSIGNED_BYTE);
  // unset context
  pangolin::GetBoundWindow()->RemoveCurrent();
}

void Viewer::Update_gsb(const SE3 &gsb) {
  gsb_ = gsb;
  trace_.push_back(gsb_.translation());
}

void Viewer::Update_gbc(const SE3 &gbc) { gbc_ = gbc; }

void Viewer::Update_gsc(const SE3 &gsc) { gsc_ = gsc; }

void Viewer::Refresh() {

  // bind to context
  pangolin::BindToContext(window_name_);

  if(!tracker_only_) {

    pangolin::View &camera_view = pangolin::Display("cam");
    camera_view.Activate(*camera_state_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(bg_color_[0], bg_color_[1], bg_color_[2], bg_color_[3]);

    // DrawGrid(half_grid_size_);
    glColor3f(0.25f, 0.25f, 0.25f);
    pangolin::glDraw_z0(0.2, cfg_.get("grid_size", 20).asInt());

    // draw axis for body frame
    pangolin::glDrawAxis(gsb_.matrix(), 0.2);

    // draw trace
    glColor3f(kYellow(0), kYellow(1), kYellow(2));

    if (cfg_.get("draw_trace_as_dots", false).asBool()) {
      pangolin::glDrawVertices(trace_, GL_POINTS); // as points
    } else {
      pangolin::glDrawVertices(trace_, GL_LINE_STRIP); // as line strips
    }

    // draw frustrum for camera frame
    glColor3f(kGreen(0), kGreen(1), kGreen(2));
    pangolin::glDrawFrustum(Kinv_, width_, height_, gsc_.matrix(), 0.2);

     // tracker view
    if (texture_) {
      pangolin::View &image_view = pangolin::Display("image");

      camera_view.Activate(*image_state_);

      image_view.Activate();
      glColor3f(1.0, 1.0, 1.0);
      texture_->RenderToViewport();
    }
  } else {

    // tracker view
    if (texture_) {
      pangolin::View &image_view = pangolin::Display("image");

      image_view.Activate();
      glColor3f(1.0, 1.0, 1.0);
      texture_->RenderToViewport();
    }
  }

  pangolin::FinishFrame();
  // unbind context
  pangolin::GetBoundWindow()->RemoveCurrent();
}

} // namespace xivo
