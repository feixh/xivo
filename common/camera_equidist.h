// Equidistant camera model. 
// Reference:
//  A Generic Camera Model and Calibration Method ...
//  http://www.ee.oulu.fi/research/mvmp/mvg/files/pdf/pdf_697.pdf
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "camera_base.h"

namespace xivo {

template <typename T>
class EquidistantCamera : public BaseCamera<T, EquidistantCamera<T>> {
public:
  using MyBase = BaseCamera<T, EquidistantCamera<T>>;
  static constexpr int DIM = 8; // size of intrinsic parameters

  EquidistantCamera(int rows, int cols, T fx, T fy, T cx, T cy, T k0, T k1,
                    T k2, T k3, int max_iter = 15)
      : BaseCamera<T, EquidistantCamera<T>>{rows, cols, fx, fy, cx, cy},
        k0_{k0}, k1_{k1}, k2_{k2}, k3_{k3}, max_iter_{max_iter},
        k00_{0.0}, k10_{0.0}, k20_{0.0}, k30_{0.0} {}

  template <typename Derived>
  Eigen::Matrix<typename Derived::Scalar, 2, 1> Project(
      const Eigen::MatrixBase<Derived> &xc,
      Eigen::Matrix<typename Derived::Scalar, 2, 2> *jac = nullptr,
      Eigen::Matrix<typename Derived::Scalar, 2, -1> *jacc = nullptr) const {
    EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 2, 1);
    using f_t = typename Derived::Scalar;
    Eigen::Matrix<f_t, 2, 1> xp;

    f_t xy_norm2 = xc.squaredNorm();
    f_t xy_norm = sqrt(xy_norm2);
    f_t xyz_norm2 = xy_norm2 + 1;

    f_t th = std::atan2(xy_norm, 1.0);

    f_t phi = std::atan2(xc[1], xc[0]);

    f_t th2 = th * th;
    f_t th3 = th2 * th;
    f_t th4 = th3 * th;
    f_t th5 = th3 * th2;
    f_t th6 = th5 * th;
    f_t th7 = th5 * th2;
    f_t th8 = th7 * th;
    f_t th9 = th7 * th2;
    f_t r = th + k0_ * th3 + k1_ * th5 + k2_ * th7 + k3_ * th9;

    f_t cos_phi = std::cos(phi);
    f_t sin_phi = std::sin(phi);

    f_t u = fx_ * r * cos_phi + cx_;
    f_t v = fy_ * r * sin_phi + cy_;

    // fill in xp
    xp[0] = u;
    xp[1] = v;

    if (jac != nullptr) {
      f_t dphi_dx = -xc[1] / xy_norm2;
      f_t dphi_dy = xc[0] / xy_norm2;

      f_t dth_dx = xc[0] / xyz_norm2 / xy_norm;
      f_t dth_dy = xc[1] / xyz_norm2 / xy_norm;

      f_t dr_dth =
          1 + k0_ * 3 * th2 + k1_ * 5 * th4 + k2_ * 7 * th6 + k3_ * 9 * th8;

      f_t du_dx = fx_ * cos_phi * dr_dth * dth_dx - fx_ * r * sin_phi * dphi_dx;
      f_t du_dy = fx_ * cos_phi * dr_dth * dth_dy - fx_ * r * sin_phi * dphi_dy;

      f_t dv_dx = fy_ * sin_phi * dr_dth * dth_dx + fy_ * r * cos_phi * dphi_dx;
      f_t dv_dy = fy_ * sin_phi * dr_dth * dth_dy + fy_ * r * cos_phi * dphi_dy;

      // fill in jacobians
      (*jac) << du_dx, du_dy, dv_dx, dv_dy;
    }

    if (jacc != nullptr) {
      auto &J{*jacc};
      J.setZero(2, 8); // d[x,y]_[fx. fy, cx, cy, k0, k1, k2, k3]

      J(0, 0) = r * cos_phi; // dx_dfx
      J(0, 2) = 1;           // dx_dcx
      J(1, 1) = r * sin_phi; // dy_dfy
      J(1, 3) = 1;           // dy_dcy

      Eigen::Matrix<f_t, 1, 4> dr_dk{th3, th5, th7,
                                     th9}; // dr_d[k0, k1, k2, k3]
      J.template block<1, 4>(0, 4) = fx_ * cos_phi * dr_dk;
      J.template block<1, 4>(1, 4) = fy_ * sin_phi * dr_dk;
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
    using Vec2 = Eigen::Matrix<f_t, 2, 1>;
    Vec2 xc;

    f_t xn = xp[0] - cx_;
    f_t yn = xp[1] - cy_;

    f_t b(fx_ * yn), a(fy_ * xn);
    f_t phi = std::atan2(b, a);
    f_t cos_phi = std::cos(phi);
    f_t sin_phi = std::sin(phi);

    // `Project` gives xn = fx * rth * cos_phi and yn = fy * rth * sin_phi, so
    // rth is recoverable either as xn / (fx cos_phi) or as the radius
    // sqrt((xn/fx)^2 + (yn/fy)^2). The two agree analytically -- the xn cancels
    // against cos_phi = fy xn / sqrt(a^2 + b^2) -- but the division form is 0/0
    // on the whole line xp[0] == cx_, where cos_phi is exactly zero, and loses
    // all significance next to it: cos_phi carries an absolute error of about
    // eps, so once |cos_phi| ~ 1e-8 the quotient is noise. The live TUM-VI
    // config uses this model, and the vertical centre line of the image is not a
    // corner case. Use the radius.
    f_t rth = std::sqrt((xn / fx_) * (xn / fx_) + (yn / fy_) * (yn / fy_));

    f_t th = rth;
    // solve th:
    // th + k0*th**3 + k1*th**5 + k2*th**7 + k3*th**9 = rth

    f_t th2, th3, th4, th6, x0, x1;
    for (int i = 0; i < max_iter_; i++) {
      // f = (th + k0*th**3 + k1*th**5 + k2*th**7 + k3*th**9 - rth)^2
      th2 = th * th;
      th3 = th2 * th;
      th4 = th2 * th2;
      th6 = th4 * th2;
      x0 = k0_ * th3 + k1_ * th4 * th + k2_ * th6 * th + k3_ * th6 * th3 - rth +
           th;
      x1 = 3 * k0_ * th2 + 5 * k1_ * th4 + 7 * k2_ * th6 + 9 * k3_ * th6 * th2 +
           1;
      f_t d = 2 * x0 * x1;
      f_t d2 = 4 * th * x0 * (3 * k0_ + 10 * k1_ * th2 + 21 * k2_ * th4 +
                              36 * k3_ * th6) +
               2 * x1 * x1;
      f_t delta = d / d2;
      th -= delta;
    }
    // `Project` builds th as atan2(|xy|, 1), so it is confined to [0, pi/2):
    // theta = pi/2 is the ray parallel to the image plane and there is no
    // (X/Z, Y/Z) that represents it. The Newton iteration above has no such
    // constraint, and for a pixel outside the model's valid radius it happily
    // overshoots past pi/2 -- where tan(th) turns negative and this function
    // returns a ray pointing *backwards* through the principal point, i.e. a
    // mirrored measurement reported as a valid one. Clamp to the model's domain.
    constexpr f_t kMaxTh = f_t(1.5706); // just under pi/2
    if (!std::isfinite(th)) {
      th = f_t(0);
    } else if (th > kMaxTh) {
      th = kMaxTh;
    } else if (th < -kMaxTh) {
      th = -kMaxTh;
    }

    f_t tan_th = std::tan(th);
    xc[0] = tan_th * cos_phi;
    xc[1] = tan_th * sin_phi;

    if (jac != nullptr) {
      f_t a2b2 = a * a + b * b;
      if (!(a2b2 > f_t(0)) || !(rth > f_t(0))) {
        // The principal point. phi is undefined there and the expressions below
        // are all 0/0; the limit of the model as xp -> [cx, cy] is
        // xp ~ [fx * xc0 + cx, fy * xc1 + cy], so d xc / d xp is this diagonal.
        (*jac) << 1 / fx_, 0, 0, 1 / fy_;
        return xc;
      }
      Vec2 dphi_dxy(-b / a2b2 * fy_, a / a2b2 * fx_);
      Vec2 dcosphi_dxy(-sin_phi * dphi_dxy);
      Vec2 dsinphi_dxy(cos_phi * dphi_dxy);
      // d rth / d[xn, yn] for rth = sqrt((xn/fx)^2 + (yn/fy)^2).
      Vec2 drth_dxy(xn / (fx_ * fx_ * rth), yn / (fy_ * fy_ * rth));

      // d rth / d th of the distortion polynomial, at the *final* th. The loop
      // variable `x1` held this at the second-to-last iterate, and was read
      // uninitialised altogether when max_iter_ was 0.
      const f_t thj2 = th * th;
      const f_t thj4 = thj2 * thj2;
      const f_t thj6 = thj4 * thj2;
      const f_t drth_dth = 1 + 3 * k0_ * thj2 + 5 * k1_ * thj4 +
                           7 * k2_ * thj6 + 9 * k3_ * thj6 * thj2;

      f_t cos_th(cos(th));
      Vec2 dtanth_dxy(drth_dxy / drth_dth / (cos_th * cos_th));
      Vec2 doutx_dxy(cos_phi * dtanth_dxy + tan_th * dcosphi_dxy);
      Vec2 douty_dxy(sin_phi * dtanth_dxy + tan_th * dsinphi_dxy);
      (*jac) << doutx_dxy[0], doutx_dxy[1], douty_dxy[0], douty_dxy[1];
    }
    return xc;
  }

  void Print(std::ostream &out) const {
    out << "Equidistant Camera" << std::endl
        << "[rows, cols]=" << rows_ << "," << cols_ << "]" << std::endl
        << "[fx, fy, cx, cy]=[" << fx_ << "," << fy_ << "," << cx_ << "," << cy_
        << "]" << std::endl
        << "[k0, k1, k2, k3]=[" << k0_ << "," << k1_ << "," << k2_ << "," << k3_
        << "]" << std::endl;
  }

  Eigen::Matrix<T, 9, 1> GetIntrinsics() {
    Eigen::Matrix<T, 9, 1> output;
    output << fx_, fy_, cx_, cy_, k0_, k1_, k2_, k3_, 0;
    return output;
  }

  DistortionType GetDistortionType() { return DistortionType::EQUI; }

  void BackupState() {
    MyBase::BackupState();
    k00_ = k0_;
    k10_ = k1_;
    k20_ = k2_;
    k30_ = k3_;
  }

  void RestoreState() {
    MyBase::RestoreState();
    k0_ = k00_;
    k1_ = k10_;
    k2_ = k20_;
    k3_ = k30_;
  }

protected:
  using MyBase::rows_;
  using MyBase::cols_;
  using MyBase::fx_;
  using MyBase::fy_;
  using MyBase::cx_;
  using MyBase::cy_;

  T k0_, k1_, k2_, k3_;
  int max_iter_;

  // backup states
  T k00_, k10_, k20_, k30_;
};

} // namespace xivo
