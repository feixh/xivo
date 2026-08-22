#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "estimator.h"
#include "camera_manager.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "utils.h"

// for visualization
#include "viewer.h"
#include "visualize.h"

namespace py = pybind11;
using namespace xivo;

namespace {

/** Copies a contiguous HxW or HxWxC uint8 numpy image into a `cv::Mat`.
 *
 *  Both properties matter.
 *
 *  *Shape from the shape*: the previous code derived the geometry from
 *  `info.strides` and hard-coded `CV_8UC3`, so a 2-D grayscale array became a
 *  `num_col = strides[0]/strides[1]`-wide 3-channel image and was read about
 *  twice past the end of the buffer.
 *
 *  *Copy, not wrap*: the `cv::Mat` handed to `VisualMeas` does not own the numpy
 *  buffer, and `VisualMeas` does not process the frame it is given — it buffers
 *  the message and processes the *oldest* one it holds
 *  (`Estimator::MaintainBuffer`), so the pixels are read long after this call
 *  returns, by which time python may have freed the array. */
cv::Mat CloneImageFromBuffer(const py::buffer_info &info) {
  if (info.ndim != 2 && info.ndim != 3) {
    throw std::runtime_error(
        "expecting a HxW or HxWxC uint8 image array, got ndim=" +
        std::to_string(info.ndim));
  }
  const int channels =
      info.ndim == 3 ? static_cast<int>(info.shape[2]) : 1;
  if (channels < 1 || channels > 4) {
    throw std::runtime_error("expecting 1-4 image channels, got " +
                             std::to_string(channels));
  }
  cv::Mat borrowed(static_cast<int>(info.shape[0]),
                   static_cast<int>(info.shape[1]), CV_8UC(channels),
                   info.ptr);
  return borrowed.clone();
}

} // namespace

class EstimatorWrapper {
public:
  EstimatorWrapper(const std::string &cfg_path,
                   const std::string &viewer_cfg_path,
                   const std::string &name,
                   bool tracker_only)
    : name_{name}, imu_calls_{0}, visual_calls_{0}, tracker_only_{tracker_only} {

    if (!glog_init_) {
      google::InitGoogleLogging("pyxivo");
      glog_init_ = true;
    }

    auto cfg = LoadJson(cfg_path);
    estimator_ = CreateSystem(cfg);
    camera_ = CameraManager::instance();

    if (!viewer_cfg_path.empty()) {
      auto viewer_cfg = LoadJson(viewer_cfg_path);
      viewer_ = std::unique_ptr<Viewer>(new Viewer{viewer_cfg, name, tracker_only_});
    }
  }

  void InertialMeas(uint64_t ts, double wx, double wy, double wz, double ax,
                    double ay, double az) {

    estimator_->InertialMeas(timestamp_t{ts}, {wx, wy, wz}, {ax, ay, az});

    if (viewer_) {
      viewer_->Update_gsb(estimator_->gsb());
      viewer_->Update_gsc(estimator_->gsc());
    }
  }


  void VisualMeasPointCloud(uint64_t ts,
                            const Eigen::Ref<const VecXi> &feature_ids,
                            const Eigen::Ref<const MatX3> &xp_with_depths) {
    estimator_->VisualMeasPointCloud(timestamp_t{ts}, feature_ids, xp_with_depths);
    if (viewer_) {
      auto disp = Canvas::instance()->display();
      if (!disp.empty()) {
        LOG(INFO) << "Display image is ready";
        viewer_->Update(disp);
      }
    }
  }

  void VisualMeasPointCloudTrackerOnly(uint64_t ts,
                            const Eigen::Ref<const VecXi> &feature_ids,
                            const Eigen::Ref<const MatX3> &xp_with_depths) {
    estimator_->VisualMeasPointCloudTrackerOnly(timestamp_t{ts}, feature_ids, xp_with_depths);
    if (viewer_) {
      auto disp = Canvas::instance()->display();
      if (!disp.empty()) {
        LOG(INFO) << "Display image is ready";
        viewer_->Update(disp);
      }
    }
  }


  void VisualMeas(uint64_t ts, std::string &image_path) {

    auto image = cv::imread(image_path);

    estimator_->VisualMeas(timestamp_t{ts}, image);

    if (viewer_) {
      auto disp = Canvas::instance()->display();

      if (!disp.empty()) {
        LOG(INFO) << "Display image is ready";
        viewer_->Update(disp);
      }
    }
  }

  void VisualMeas(uint64_t ts,
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> b)
  {
    py::buffer_info info = b.request();

    cv::Mat image = CloneImageFromBuffer(info);

    estimator_->VisualMeas(timestamp_t{ts}, image);

    if (viewer_) {
      auto disp = Canvas::instance()->display();
      if (!disp.empty()) {
        LOG(INFO) << "Display image is ready";
        viewer_->Update(disp);
      }
    }
  }

  void VisualMeasTrackerOnly(uint64_t ts, std::string &image_path) {

    auto image = cv::imread(image_path);

    estimator_->VisualMeasTrackerOnly(timestamp_t{ts}, image);

    if (viewer_) {
      auto disp = Canvas::instance()->display();

      if (!disp.empty()) {
        LOG(INFO) << "Display image is ready";
        viewer_->Update(disp);
      }
    }
  }

  void VisualMeasTrackerOnly(uint64_t ts,
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> b)
  {
    py::buffer_info info = b.request();

    cv::Mat image = CloneImageFromBuffer(info);

    estimator_->VisualMeasTrackerOnly(timestamp_t{ts}, image);

    if (viewer_) {
      auto disp = Canvas::instance()->display();
      if (!disp.empty()) {
        LOG(INFO) << "Display image is ready";
        viewer_->Update(disp);
      }
    }
  }

  void CloseLoop() {
    estimator_->CloseLoop();
  }

  std::vector<std::tuple<int, Vec2, MatXf>> tracked_features() {
    return estimator_->tracked_features();
  }
  std::vector<std::tuple<int, Vec2>> tracked_features_no_descriptor() {
    return estimator_->tracked_features_no_descriptor();
  }

  VecXi JustDroppedFeatureIDs() {
    return estimator_->JustDroppedFeatureIDs();
  }

  void InitWithSimDepths() { estimator_->InitWithSimDepths(); }

  void ScaleInitVelocity(double scale) { estimator_->ScaleInitVelocity(scale); }

  Eigen::Matrix<double, 3, 4> gsb() { return estimator_->gsb().matrix3x4(); }
  Eigen::Matrix<double, 3, 4> gsc() { return estimator_->gsc().matrix3x4(); }
  Eigen::Matrix<double, 3, 4> gbc() { return estimator_->gbc().matrix3x4(); }
  Eigen::Matrix<double, -1, -1> Pstate() { return estimator_->Pstate(); }
  Eigen::Matrix<double, -1, -1> P() { return estimator_-> P(); }
  Vec3 Vsb() { return estimator_->Vsb(); }
  Vec3 bg() { return estimator_->bg(); }
  Vec3 ba() { return estimator_->ba(); }
  Mat3 Rsg() { return estimator_->Rsg().matrix(); }
  number_t td() { return estimator_->td(); }
  Mat3 Ca() { return estimator_->Ca(); }
  Mat3 Cg() { return estimator_->Cg(); }

  bool MeasurementUpdateInitialized() {
    return estimator_->MeasurementUpdateInitialized();
  }

  uint64_t now() const { return estimator_->ts().count(); }

  int gauge_group() { return estimator_->gauge_group(); }

  MatX3 InstateFeaturePositions(int n_output) {
    return estimator_->InstateFeaturePositions(n_output);
  }

  MatX3 InstateFeaturePositions() {
    return estimator_->InstateFeaturePositions();
  }

  MatX6 InstateFeatureCovs(int n_output) {
    return estimator_->InstateFeatureCovs(n_output);
  }

  MatX6 InstateFeatureCovs() {
    return estimator_->InstateFeatureCovs();
  }

  VecXi InstateFeatureIDs(int n_output) {
    return estimator_->InstateFeatureIDs(n_output);
  }

  VecXi InstateFeatureIDs() {
    return estimator_->InstateFeatureIDs();
  }

  VecXi InstateFeatureSinds(int n_output) {
    return estimator_->InstateFeatureSinds(n_output);
  }

  VecXi InstateFeatureRefGroups(int n_output) {
    return estimator_->InstateFeatureRefGroups(n_output);
  }

  MatX3 InstateFeatureXc(int n_output) {
    return estimator_->InstateFeatureXc(n_output);
  }

  MatX3 InstateFeatureXc() {
    return estimator_->InstateFeatureXc();
  }

  MatX3 InstateFeaturexc(int n_output) {
    return estimator_->InstateFeaturexc(n_output);
  }

  MatX3 InstateFeaturexc() {
    return estimator_->InstateFeaturexc();
  }

  MatX2 InstateFeaturePreds(int n_output) {
    return estimator_->InstateFeaturePreds(n_output);
  }

  MatX2 InstateFeaturePreds() {
    return estimator_->InstateFeaturePreds();
  }

  MatX2 InstateFeatureMeas() {
    return estimator_->InstateFeatureMeas();  
  }

  MatX2 InstateFeatureMeas(int n_output) {
    return estimator_->InstateFeatureMeas(n_output);
  }

  VecXi InstateFeatureSinds() {
    return estimator_->InstateFeatureSinds();
  }

  VecXi InstateFeatureRefGroups() {
    return estimator_->InstateFeatureRefGroups();
  }

  VecXi InstateGroupIDs() {
    return estimator_->InstateGroupIDs();
  }

  MatX7 InstateGroupPoses() {
    return estimator_->InstateGroupPoses();
  }

  MatX InstateGroupCovs() {
    return estimator_->InstateGroupCovs();
  }

  VecXi InstateGroupSinds() {
    return estimator_->InstateGroupSinds();
  }

  Vec9 CameraIntrinsics() {
    return camera_->GetIntrinsics();
  }

  int CameraDistortionType() {
    return int(camera_->GetDistortionType());
  }

  int num_instate_features() { return estimator_->num_instate_features(); }

  int num_instate_groups() { return estimator_->num_instate_groups(); }

  int num_mh_rejected() { return estimator_->num_mh_rejected(); }

  int num_oneptransac_rejected() { return estimator_->num_oneptransac_rejected(); }

  int num_tracker_outlier_rejected() { return estimator_->num_tracker_outlier_rejected(); }

  int num_tracker_failed_to_track() { return estimator_->num_tracker_failed_to_track(); }

  int num_tracker_new_detections() { return estimator_->num_tracker_new_detections(); }

  bool UsingLoopClosure() {
    return estimator_->UsingLoopClosure();
  }

  bool VisionInitialized() {
    return estimator_->VisionInitialized();
  }

  void Visualize() {
    if (viewer_)
      viewer_->Refresh();
  }

private:
  EstimatorPtr estimator_;
  CameraPtr camera_;
  std::unique_ptr<Viewer> viewer_;
  static bool glog_init_;
  std::string name_;
  int imu_calls_, visual_calls_;
  bool tracker_only_;
};

bool EstimatorWrapper::glog_init_{false};

PYBIND11_MODULE(pyxivo, m) {
  m.doc() = "python binding of XIVO (Xiaohan's Inertial-aided Visual Odometry)";
  py::class_<EstimatorWrapper>(m, "Estimator")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, bool>())
      .def("InertialMeas", &EstimatorWrapper::InertialMeas)
      .def("VisualMeas", py::overload_cast<uint64_t, std::string &>(&EstimatorWrapper::VisualMeas))
      .def("VisualMeas", py::overload_cast<uint64_t, py::array_t<unsigned char, py::array::c_style | py::array::forcecast>>(&EstimatorWrapper::VisualMeas))
      .def("VisualMeasTrackerOnly", py::overload_cast<uint64_t, std::string &>(&EstimatorWrapper::VisualMeasTrackerOnly))
      .def("VisualMeasTrackerOnly", py::overload_cast<uint64_t, py::array_t<unsigned char, py::array::c_style | py::array::forcecast>>(&EstimatorWrapper::VisualMeasTrackerOnly))
      .def("VisualMeasPointCloud", &EstimatorWrapper::VisualMeasPointCloud)
      .def("VisualMeasPointCloudTrackerOnly", &EstimatorWrapper::VisualMeasPointCloudTrackerOnly)
      .def("CloseLoop", &EstimatorWrapper::CloseLoop)
      .def("InitWithSimDepths", &EstimatorWrapper::InitWithSimDepths)
      .def("ScaleInitVelocity", &EstimatorWrapper::ScaleInitVelocity)
      .def("gbc", &EstimatorWrapper::gbc)
      .def("gsb", &EstimatorWrapper::gsb)
      .def("gsc", &EstimatorWrapper::gsc)
      .def("Vsb", &EstimatorWrapper::Vsb)
      .def("Pstate", &EstimatorWrapper::Pstate)
      .def("P", &EstimatorWrapper::P)
      .def("bg", &EstimatorWrapper::bg)
      .def("ba", &EstimatorWrapper::ba)
      .def("Rg", &EstimatorWrapper::Rsg)
      .def("td", &EstimatorWrapper::td)
      .def("Ca", &EstimatorWrapper::Ca)
      .def("Cg", &EstimatorWrapper::Cg)
      .def("InstateFeaturePositions", py::overload_cast<int>(&EstimatorWrapper::InstateFeaturePositions))
      .def("InstateFeaturePositions", py::overload_cast<>(&EstimatorWrapper::InstateFeaturePositions))
      .def("InstateFeatureCovs", py::overload_cast<int>(&EstimatorWrapper::InstateFeatureCovs))
      .def("InstateFeatureCovs", py::overload_cast<>(&EstimatorWrapper::InstateFeatureCovs))
      .def("InstateFeatureIDs", py::overload_cast<int>(&EstimatorWrapper::InstateFeatureIDs))
      .def("InstateFeatureIDs", py::overload_cast<>(&EstimatorWrapper::InstateFeatureIDs))
      .def("InstateFeatureSinds", py::overload_cast<>(&EstimatorWrapper::InstateFeatureSinds))
      .def("InstateFeatureSinds", py::overload_cast<int>(&EstimatorWrapper::InstateFeatureSinds))
      .def("InstateFeatureRefGroups", py::overload_cast<>(&EstimatorWrapper::InstateFeatureRefGroups))
      .def("InstateFeatureRefGroups", py::overload_cast<int>(&EstimatorWrapper::InstateFeatureRefGroups))
      .def("InstateFeatureXc", py::overload_cast<int>(&EstimatorWrapper::InstateFeatureXc))
      .def("InstateFeatureXc", py::overload_cast<>(&EstimatorWrapper::InstateFeatureXc))
      .def("InstateFeaturexc", py::overload_cast<int>(&EstimatorWrapper::InstateFeaturexc))
      .def("InstateFeaturexc", py::overload_cast<>(&EstimatorWrapper::InstateFeaturexc))
      .def("InstateFeaturePreds", py::overload_cast<int>(&EstimatorWrapper::InstateFeaturePreds))
      .def("InstateFeaturePreds", py::overload_cast<>(&EstimatorWrapper::InstateFeaturePreds))
      .def("InstateFeatureMeas", py::overload_cast<int>(&EstimatorWrapper::InstateFeatureMeas))
      .def("InstateFeatureMeas", py::overload_cast<>(&EstimatorWrapper::InstateFeatureMeas))
      .def("InstateGroupIDs", &EstimatorWrapper::InstateGroupIDs)
      .def("InstateGroupSinds", &EstimatorWrapper::InstateGroupSinds)
      .def("InstateGroupPoses", &EstimatorWrapper::InstateGroupPoses)
      .def("InstateGroupCovs", &EstimatorWrapper::InstateGroupCovs)
      .def("num_instate_features", &EstimatorWrapper::num_instate_features)
      .def("num_instate_groups", &EstimatorWrapper::num_instate_groups)
      .def("num_mh_rejected", &EstimatorWrapper::num_mh_rejected)
      .def("num_oneptransac_rejected", &EstimatorWrapper::num_oneptransac_rejected)
      .def("num_tracker_outlier_rejected", &EstimatorWrapper::num_tracker_outlier_rejected)
      .def("num_tracker_failed_to_track", &EstimatorWrapper::num_tracker_failed_to_track)
      .def("num_tracker_new_detections", &EstimatorWrapper::num_tracker_new_detections)
      .def("UsingLoopClosure", &EstimatorWrapper::UsingLoopClosure)
      .def("VisionInitialized", &EstimatorWrapper::VisionInitialized)
      .def("now", &EstimatorWrapper::now)
      .def("Visualize", &EstimatorWrapper::Visualize)
      .def("gauge_group", &EstimatorWrapper::gauge_group)
      .def("CameraIntrinsics", &EstimatorWrapper::CameraIntrinsics)
      .def("CameraDistortionType", &EstimatorWrapper::CameraDistortionType)
      .def("MeasurementUpdateInitialized", &EstimatorWrapper::MeasurementUpdateInitialized)
      .def("JustDroppedFeatureIDs", &EstimatorWrapper::JustDroppedFeatureIDs)
      .def("tracked_features", &EstimatorWrapper::tracked_features)
      .def("tracked_features_no_descriptor", &EstimatorWrapper::tracked_features_no_descriptor);
}
