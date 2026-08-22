// Singleton camera manager to create and manage different camera models.  
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <memory>
#include <ostream>
#include <variant>
#include <vector>

#include "camera_autocalib.h"
#include "alias.h"
#include "glog/logging.h"
#include "utils.h"
#include "json/json.h"

namespace xivo {

template <typename T> using UnknownCamera = T;  // dummy ...

class CameraManager {
public:
  using Unknown = UnknownCamera<number_t>;
  using ATAN = A_ATANCamera;
  using EquiDist = A_EquidistantCamera;
  // using EquiDist = EquidistantCamera<number_t>;
  using RadTan = A_RadialTangentialCamera;
  using Pinhole = A_PinholeCamera;

  /** Create the camera for slot `cam_id` (0 = left/primary). Calling this twice
   * for the same slot leaves the existing camera in place, matching the
   * original single-camera behaviour. */
  static CameraManager *Create(const Json::Value &cfg, int cam_id = 0);
  /** Camera for slot `cam_id`. `instance()` is the primary camera, so every
   * existing monocular call site keeps working unchanged. */
  static CameraManager *instance(int cam_id = 0) {
    return cam_id >= 0 && cam_id < static_cast<int>(instances_.size())
               ? instances_[cam_id].get()
               : nullptr;
  }
  /** Number of cameras configured (1 for monocular, 2 for stereo). */
  static int num_cameras() { return static_cast<int>(instances_.size()); }

  // project a point from camera coordinatex xc to pixel coordinates xp.
  // xc: a point in camera coordinates.
  // jac: jacobian matrix dxp/dxc
  // jacc: jacobian matrix of xp w.r.t. camera intrinsics
  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> Project(
      const Eigen::MatrixBase<Derived> &xc,
      Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac = nullptr,
      Eigen::Matrix<typename Derived::Scalar, 2, -1> *jacc = nullptr) const {
    if (std::holds_alternative<ATAN>(model_)) {
      return std::get<ATAN>(model_).Project(xc, jac, jacc);
    } else if (std::holds_alternative<EquiDist>(model_)) {
      return std::get<EquiDist>(model_).Project(xc, jac, jacc);
    } else if (std::holds_alternative<RadTan>(model_)) {
      return std::get<RadTan>(model_).Project(xc, jac, jacc);
    } else if (std::holds_alternative<Pinhole>(model_)) {
      return std::get<Pinhole>(model_).Project(xc, jac, jacc);
    } else {
      LOG(FATAL) << "unknown camera model";
    }
  }

  // unproject a point from pixel coordinatex xp to camera coordinates xc.
  // xp: a point in pixel coordinates.
  // jac: jacobian matrix dxc/dxp
  // jacc: jacobian matrix of xc w.r.t. camera intrinsics
  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> UnProject(
      const Eigen::MatrixBase<Derived> &xp,
      Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac = nullptr,
      Eigen::Matrix<typename Derived::Scalar, 2, -1> *jacc = nullptr) const {
    if (jacc != nullptr) {
      LOG(FATAL) << "jacobian w.r.t. camera intrinsics (jacc) NOT implemented";
    }

    if (std::holds_alternative<ATAN>(model_)) {
      return std::get<ATAN>(model_).UnProject(xp, jac, jacc);
    } else if (std::holds_alternative<EquiDist>(model_)) {
      return std::get<EquiDist>(model_).UnProject(xp, jac, jacc);
    } else if (std::holds_alternative<RadTan>(model_)) {
      return std::get<RadTan>(model_).UnProject(xp, jac, jacc);
    } else if (std::holds_alternative<Pinhole>(model_)) {
      return std::get<Pinhole>(model_).UnProject(xp, jac, jacc);
    } else {
      LOG(FATAL) << "unknown camera model";
    }
  }

  void Print(std::ostream &out) const {
    if (std::holds_alternative<ATAN>(model_)) {
      std::get<ATAN>(model_).Print(out);
    } else if (std::holds_alternative<EquiDist>(model_)) {
      std::get<EquiDist>(model_).Print(out);
    } else if (std::holds_alternative<RadTan>(model_)) {
      std::get<RadTan>(model_).Print(out);
    } else if (std::holds_alternative<Pinhole>(model_)) {
      std::get<Pinhole>(model_).Print(out);
    } else {
      LOG(FATAL) << "unknown camera model";
    }
  }

  void UpdateState(const VecX &dX) {
    if (std::holds_alternative<ATAN>(model_)) {
      std::get<ATAN>(model_).UpdateState(dX.head<ATAN::DIM>());
    } else if (std::holds_alternative<EquiDist>(model_)) {
      std::get<EquiDist>(model_).UpdateState(dX.head<EquiDist::DIM>());
    } else if (std::holds_alternative<RadTan>(model_)) {
      std::get<RadTan>(model_).UpdateState(dX.head<RadTan::DIM>());
    } else if (std::holds_alternative<Pinhole>(model_)) {
      std::get<Pinhole>(model_).UpdateState(dX.head<Pinhole::DIM>());
    } else {
      LOG(FATAL) << "unknown camera model";
    }
    // also update intrinsics for the camera manager ...
    fx_ += dX(0);
    fy_ += dX(1);
    cx_ += dX(2);
    cy_ += dX(3);
    fl_ = std::sqrt(0.5 * (fx_ * fx_ + fy_ * fy_));
  }

  number_t GetFocalLength() const { return fl_; }
  int rows() const { return rows_; }
  int cols() const { return cols_; }
  number_t fx() const { return fx_; }
  number_t fy() const { return fy_; }
  number_t cx() const { return cx_; }
  number_t cy() const { return cy_; }
  int dim() const { return dim_; }

  Vec9 GetIntrinsics() { 
    if (std::holds_alternative<ATAN>(model_)) {
      return std::get<ATAN>(model_).GetIntrinsics();
    } else if (std::holds_alternative<EquiDist>(model_)) {
      return std::get<EquiDist>(model_).GetIntrinsics();
    } else if (std::holds_alternative<RadTan>(model_)) {
      return std::get<RadTan>(model_).GetIntrinsics();
    } else if (std::holds_alternative<Pinhole>(model_)) {
      return std::get<Pinhole>(model_).GetIntrinsics();
    } else {
      LOG(FATAL) << "unknown camera model";
    }
  }

  DistortionType GetDistortionType() {
    if (std::holds_alternative<ATAN>(model_)) {
      return std::get<ATAN>(model_).GetDistortionType();
    } else if (std::holds_alternative<EquiDist>(model_)) {
      return std::get<EquiDist>(model_).GetDistortionType();
    } else if (std::holds_alternative<RadTan>(model_)) {
      return std::get<RadTan>(model_).GetDistortionType();
    } else if (std::holds_alternative<Pinhole>(model_)) {
      return std::get<Pinhole>(model_).GetDistortionType();
    } else {
      LOG(FATAL) << "unknown camera model";
    }
  }

  void BackupState() {
    if (std::holds_alternative<ATAN>(model_)) {
      std::get<ATAN>(model_).BackupState();
    } else if (std::holds_alternative<EquiDist>(model_)) {
      std::get<EquiDist>(model_).BackupState();
    } else if (std::holds_alternative<RadTan>(model_)) {
      std::get<RadTan>(model_).BackupState();
    } else if (std::holds_alternative<Pinhole>(model_)) {
      std::get<Pinhole>(model_).BackupState();
    } else {
      LOG(FATAL) << "unknown camera model";
    }
  }

  void RestoreState() {
    if (std::holds_alternative<ATAN>(model_)) {
      std::get<ATAN>(model_).RestoreState();
    } else if (std::holds_alternative<EquiDist>(model_)) {
      std::get<EquiDist>(model_).RestoreState();
    } else if (std::holds_alternative<RadTan>(model_)) {
      std::get<RadTan>(model_).RestoreState();
    } else if (std::holds_alternative<Pinhole>(model_)) {
      std::get<Pinhole>(model_).RestoreState();
    } else {
      LOG(FATAL) << "unknown camera model";
    }
  }

private:
  CameraManager &operator=(const CameraManager &) = delete;
  CameraManager(const CameraManager &) = delete;

  CameraManager(const Json::Value &cfg);
  static std::vector<std::unique_ptr<CameraManager>> instances_;

  int rows_, cols_;
  number_t fx_, fy_, cx_, cy_;
  number_t fl_; // focal length
  std::variant<Unknown, ATAN, EquiDist, RadTan, Pinhole> model_;
  int dim_; // number of intrinsic parameters
};

} // namespace xivo
