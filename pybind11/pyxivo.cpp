#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "estimator.h"
#include "camera_manager.h"
#include "stereo.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "utils.h"

// for visualization
#include "viewer.h"
#include "visualize.h"

namespace py = pybind11;
using namespace xivo;

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

    int size_row = info.strides[0];
    int num_col = size_row / info.strides[1] / info.itemsize;
    int num_row = info.size / size_row;

    cv::Mat image(num_row, num_col, CV_8UC3, info.ptr);

    estimator_->VisualMeas(timestamp_t{ts}, image);

    if (viewer_) {
      auto disp = Canvas::instance()->display();
      if (!disp.empty()) {
        LOG(INFO) << "Display image is ready";
        viewer_->Update(disp);
      }
    }
  }

  void VisualMeasStereo(uint64_t ts, std::string &image_path,
                        std::string &image_path_r) {

    auto image = cv::imread(image_path);
    auto image_r = cv::imread(image_path_r);
    if (image.empty()) {
      LOG(FATAL) << "failed to read left image " << image_path;
    }
    if (image_r.empty()) {
      LOG(FATAL) << "failed to read right image " << image_path_r;
    }

    estimator_->VisualMeasStereo(timestamp_t{ts}, image, image_r);

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

    int size_row = info.strides[0];
    int num_col = size_row / info.strides[1] / info.itemsize;
    int num_row = info.size / size_row;

    cv::Mat image(num_row, num_col, CV_8UC3, info.ptr);

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

  int num_stereo_frames() { return estimator_->num_stereo_frames(); }

  int num_stereo_matched() { return estimator_->num_stereo_matched(); }

  int num_stereo_attempted() { return estimator_->num_stereo_attempted(); }

  int num_stereo_rejected_klt() {
    return estimator_->num_stereo_rejected_klt();
  }

  int num_stereo_rejected_epipolar() {
    return estimator_->num_stereo_rejected_epipolar();
  }

  int num_stereo_rejected_circular() {
    return estimator_->num_stereo_rejected_circular();
  }

  int num_stereo_rejected_disparity() {
    return estimator_->num_stereo_rejected_disparity();
  }

  int num_stereo_init_ok() { return estimator_->num_stereo_init_ok(); }

  int num_stereo_init_no_match() {
    return estimator_->num_stereo_init_no_match();
  }

  int num_stereo_init_rejected() {
    return estimator_->num_stereo_init_rejected();
  }
  int num_stereo_init_rej_degenerate() {
    return estimator_->num_stereo_init_rej_degenerate();
  }
  int num_stereo_init_rej_gap() { return estimator_->num_stereo_init_rej_gap(); }
  int num_stereo_init_rej_range() {
    return estimator_->num_stereo_init_rej_range();
  }
  int num_stereo_init_rej_std() { return estimator_->num_stereo_init_rej_std(); }
  int num_stereo_upd_used() { return estimator_->num_stereo_upd_used(); }
  int num_stereo_upd_rej_geom() {
    return estimator_->num_stereo_upd_rej_geom();
  }
  int num_stereo_upd_rej_mh() { return estimator_->num_stereo_upd_rej_mh(); }

  bool StereoEnabled() { return StereoRig::enabled(); }

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
      .def("VisualMeasStereo", &EstimatorWrapper::VisualMeasStereo)
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
      .def("num_stereo_frames", &EstimatorWrapper::num_stereo_frames)
      .def("num_stereo_matched", &EstimatorWrapper::num_stereo_matched)
      .def("num_stereo_attempted", &EstimatorWrapper::num_stereo_attempted)
      .def("num_stereo_rejected_klt", &EstimatorWrapper::num_stereo_rejected_klt)
      .def("num_stereo_rejected_epipolar", &EstimatorWrapper::num_stereo_rejected_epipolar)
      .def("num_stereo_rejected_circular", &EstimatorWrapper::num_stereo_rejected_circular)
      .def("num_stereo_rejected_disparity", &EstimatorWrapper::num_stereo_rejected_disparity)
      .def("num_stereo_init_ok", &EstimatorWrapper::num_stereo_init_ok)
      .def("num_stereo_init_no_match", &EstimatorWrapper::num_stereo_init_no_match)
      .def("num_stereo_init_rejected", &EstimatorWrapper::num_stereo_init_rejected)
      .def("num_stereo_init_rej_degenerate", &EstimatorWrapper::num_stereo_init_rej_degenerate)
      .def("num_stereo_init_rej_gap", &EstimatorWrapper::num_stereo_init_rej_gap)
      .def("num_stereo_init_rej_range", &EstimatorWrapper::num_stereo_init_rej_range)
      .def("num_stereo_init_rej_std", &EstimatorWrapper::num_stereo_init_rej_std)
      .def("num_stereo_upd_used", &EstimatorWrapper::num_stereo_upd_used)
      .def("num_stereo_upd_rej_geom", &EstimatorWrapper::num_stereo_upd_rej_geom)
      .def("num_stereo_upd_rej_mh", &EstimatorWrapper::num_stereo_upd_rej_mh)
      .def("StereoEnabled", &EstimatorWrapper::StereoEnabled)
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
