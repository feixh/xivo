// ATAN camera model.
// Reference:
//  Straight lines have to be straight ...
//  https://hal.inria.fr/inria-00267247/file/distcalib.pdf
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "camera_base.h"

namespace xivo {

template <typename T> class ATANCamera : public BaseCamera<T, ATANCamera<T>> {
public:
  using MyBase = BaseCamera<T, ATANCamera<T>>;
  static constexpr int DIM = 5; // size of intrinsic parameters

  ATANCamera(int rows, int cols, T fx, T fy, T cx, T cy, T w)
      : BaseCamera<T, ATANCamera<T>>{rows, cols, fx, fy, cx, cy}, w_(w),
        invw_(1.0 / w), w2_(2.0 * std::tan(w * 0.5)),
        w0_{0.0}, invw0_{0.0}, w20_{0.0} {}

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> Project(
      const Eigen::MatrixBase<Derived> &xc,
      Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac = nullptr,
      Eigen::Matrix<typename Derived::Scalar, 2, -1> *jacc = nullptr) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    using f_t = typename Derived::Scalar;

    Eigen::Matrix<f_t, 2, 1> xp;

    f_t R = xc.norm();
    bool singular = (R < 0.0001 || w_ == 0);

    // The R -> 0 limit of f = atan(w2 R) / (w R) is w2 / w, not 1, because
    // atan(w2 R) -> w2 R. Substituting 1 made the projection discontinuous
    // across the R = 1e-4 threshold by a factor 2 tan(w/2) / w -- roughly 9 %
    // for a typical w near 1 -- and put the same error in the Jacobian below.
    // (For w -> 0 the model degenerates to a pinhole and the limit really is 1,
    // but w2 / w is 0/0 there, so that case stays explicit.)
    const f_t f_limit = (w_ == 0) ? f_t(1) : f_t(invw_ * w2_);
    f_t f{singular ? f_limit : f_t(invw_ * std::atan(w2_ * R) / R)};

    // Project through distortion model
    xp(0) = fx_ * f * xc(0) + cx_;
    xp(1) = fy_ * f * xc(1) + cy_;

    if (jac != nullptr) {
      auto &J{*jac};
      // compute jacobians
      if (singular) {
        // The off-diagonals were left untouched -- `jac` points at whatever the
        // caller last had there (`JacobianCache::dxp_dxcn` is a reused member),
        // so this branch returned a Jacobian with two stale entries.
        J << fx_ * f, 0, 0, fy_ * f;
      } else {
        // FIXME: optimize computation
        f_t df_dx, df_dy, df_dR;
        f_t a = w2_ * R;
        df_dR = invw_ * (1. / (1 + a * a) * a - std::atan(a)) / R / R;
        df_dx = df_dR * xc(0) / R;
        df_dy = df_dR * xc(1) / R;

        J << fx_ * f + fx_ * xc(0) * df_dx, fx_ * xc(0) * df_dy,
            fy_ * xc(1) * df_dx, fy_ * f + fy_ * xc(1) * df_dy;
      }
    }

    if (jacc != nullptr) {
      auto &J{*jacc};
      J.setZero(2, DIM); // d[x, y]_d[fx, fy, cx, cy, w]
      if (singular) {
        // Same wrong limit as above: dxp/dfx is f * xc(0), and f is w2/w here,
        // not 1. The dxp/dw column was left at zero, but d/dw [2 tan(w/2) / w]
        // is only zero at w = 0.
        const f_t df_dw_limit =
            (w_ == 0) ? f_t(0)
                      : f_t((w_ / (std::cos(w_ * 0.5) * std::cos(w_ * 0.5)) -
                             w2_) *
                            invw_ * invw_);
        J(0, 0) = f * xc(0);
        J(0, 2) = 1;
        J(1, 1) = f * xc(1);
        J(1, 3) = 1;
        J(0, 4) = fx_ * xc(0) * df_dw_limit;
        J(1, 4) = fy_ * xc(1) * df_dw_limit;
      } else {
        J(0, 0) = f * xc(0);
        J(0, 2) = 1;
        J(1, 1) = f * xc(1);
        J(1, 3) = 1;
        // f = inv(w) * atan(w2 * R) / R
        // R is constant w.r.t. w
        f_t df_dinvw = std::atan(w2_ * R) / R;
        f_t dinvw_dw = -invw_ * invw_;

        f_t df_datanw2R = invw_ / R;
        // datan(x)_dx = 1 / (1 + x * x)
        f_t datanw2R_dw2R = 1 / (1 + (w2_ * R) * (w2_ * R));
        f_t dw2R_dw2 = R; // w2R = w2 * R
        // Recall: w2 = 2 * tan(w * 0.5)
        // dw2_dw = 2.0 / cos^2(w*0.5) * 0.5 = 1.0 / cos^2(2 * 0.5);
        f_t dw2_dw = 1 / std::cos(w_ * 0.5);
        dw2_dw *= dw2_dw;
        f_t df_dw = df_dinvw * dinvw_dw +
                    df_datanw2R * datanw2R_dw2R * dw2R_dw2 * dw2_dw;
        J(0, 4) = fx_ * xc(0) * df_dw;
        J(1, 4) = fy_ * xc(1) * df_dw;
      }
    }
    return xp;
  }

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> UnProject(
      const Eigen::MatrixBase<Derived> &xp,
      Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac = nullptr,
      Eigen::Matrix<typename Derived::Scalar, 2, -1> *jacc = nullptr) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);

    using f_t = typename Derived::Scalar;
    Eigen::Matrix<f_t, 2, 1> xc;

    Eigen::Matrix<f_t, 2, 1> tmp((xp(0) - cx_) / fx_, (xp(1) - cy_) / fy_);
    f_t R = tmp.norm();

    const bool singular = !(R > 0.01) || w_ == 0;
    // Mirror of the Project bug: the R -> 0 limit of tan(R w) / (w2 R) is
    // w / w2, not 1.
    const f_t f_limit = (w_ == 0) ? f_t(1) : f_t(w_ / w2_);

    f_t f;
    if (singular) {
      f = f_limit;
    } else {
      // R * w >= pi/2 is outside the model's image circle. Past it tan flips
      // sign and this returned a ray pointing backwards through the principal
      // point, reported as a valid unprojection.
      f_t Rw = R * w_;
      constexpr f_t kMaxRw = f_t(1.5706); // just under pi/2
      if (!std::isfinite(Rw)) {
        Rw = f_t(0);
      } else if (Rw > kMaxRw) {
        Rw = kMaxRw;
      } else if (Rw < -kMaxRw) {
        Rw = -kMaxRw;
      }
      f = std::tan(Rw) / (w2_ * R);
    }
    xc = f * tmp;

    if (jac != nullptr) {
      // The test used to be `f == 1`: an exact float comparison against the
      // value the singular branch happened to assign, which both misses the
      // singular case whenever the limit is not 1 and fires spuriously if the
      // regular branch lands on 1 exactly. And, as in Project, the branch only
      // wrote the diagonal, leaving the caller's stale off-diagonals in place.
      if (singular) {
        (*jac) << f_limit / fx_, 0, 0, f_limit / fy_;
      } else {
        f_t df_dR;
        f_t a = std::tan(R * w_);
        df_dR = 1.0 / w2_ * (((1 + a * a) * w_ * R - a) / R / R);

        f_t df_dx, df_dy;
        df_dx = df_dR * tmp(0) / R / fx_;
        df_dy = df_dR * tmp(1) / R / fy_;

        (*jac) << tmp(0) * df_dx + f / fx_, tmp(0) * df_dy, tmp(1) * df_dx,
            tmp(1) * df_dy + f / fy_;
      }
    }
    return xc;
  }

  void Print(std::ostream &out) const {
    out << "ATAN Camera" << std::endl
        << "[rows, cols]=" << rows_ << "," << cols_ << "]" << std::endl
        << "[fx, fy, cx, cy, w]=[" << fx_ << "," << fy_ << "," << cx_ << ","
        << cy_ << "," << w_ << "]" << std::endl;
  }

  Eigen::Matrix<T, 9, 1> GetIntrinsics() {
    Eigen::Matrix<T, 9, 1> output;
    output << fx_, fy_, cx_, cy_, w_, 0, 0, 0, 0;
    return output;
  }

  DistortionType GetDistortionType() { return DistortionType::ATAN; }

  void BackupState() {
    MyBase::BackupState();
    w0_ = w_;
    invw0_ = invw_;
    w20_ = w2_;
  }

  void RestoreState() {
    MyBase::RestoreState();
    w_ = w0_;
    invw_ = invw0_;
    w2_ = w20_;
  }


protected:
  using MyBase::rows_;
  using MyBase::cols_;
  using MyBase::fx_;
  using MyBase::fy_;
  using MyBase::cx_;
  using MyBase::cy_;

  T w_, invw_, w2_;

  // backup states
  T w0_, invw0_, w20_;
};

} // namespace xivo
