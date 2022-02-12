#include <algorithm>

#include "estimator.h"
#include "feature.h"
#include "group.h"
#include "helpers.h"
#include "mm.h"
#include "param.h"
#include "alias.h"
#include "rodrigues.h"

#include "glog/logging.h"

namespace xivo {

// Feature
int Feature::counter_ = Feature::counter0;
JacobianCache Feature::cache_ = {};

////////////////////////////////////////
// FACTORY METHODS
////////////////////////////////////////
FeaturePtr Feature::Create(number_t x, number_t y) {
  auto f = MemoryManager::instance()->GetFeature();
#ifndef NDEBUG
  CHECK(f);
#endif
  f->Reset(x, y);
  return f;
}

void Feature::Delete(FeaturePtr f) {
  MemoryManager::instance()->ReturnFeature(f);
}

void Feature::Reset(number_t x, number_t y) {
  id_ = counter_++;
  sind_ = -1;
  init_counter_ = 0;
  lifetime_ = 0;
  status_ = FeatureStatus::CREATED;
  ref_ = nullptr;
  Track::Reset(x, y);
  x_ << x, y, 2.0;
  pred_ << -1, -1;
  J_.setZero();
  inn_ << 0, 0;
  outlier_counter_ = 0;

  sim_.Xs << -1, -1, -1;
  sim_.xp << -1, -1;
  sim_.xc << -1, -1;
  sim_.z = -1;
  sim_.lifetime = -1;

#ifdef APPROXIMATE_INIT_COVARIANCE
  cov_.clear();
  cov_xc_.setZero();
  cov_xr_.setZero();
#endif
}

////////////////////////////////////////
// SOME ACCESSORS
////////////////////////////////////////
Vec3 Feature::Xc(Mat3 *J) {
#ifdef USE_INVDEPTH
  Xc_ = unproject_invz(x_, J);
#else
  Xc_ = unproject_logz(x_, J);
#endif
  return Xc_;
}

Vec3 Feature::Xs(const SE3 &gbc, Mat3 *J) {
  // Rsb * (Rbc*Xc + Tbc) + Tsb
#ifndef NDEBUG
  CHECK(ref_) << "feature #" << id_ << " null ref";
#endif
  SE3 gsc = ref_->gsb() * gbc;
  Xs_ = gsc * Xc(J); // J = dXc_dx, where x is the local parametrization
  if (J) {
    *J = gsc.R().matrix() * (*J);
  }
  return Xs_;
}

number_t Feature::z() const {
#ifdef USE_INVDEPTH
  return 1.0 / x_(2);
#else
  return exp(x_(2));
#endif
}

bool Feature::instate() const { return status_ == FeatureStatus::INSTATE; }

number_t Feature::score() const {
#ifndef NDEBUG
  CHECK(!instate())
      << "score function should only be called for feature not-instate yet";
#endif
  // TODO: come up with better scoring
  // confidence (negative uncertainty) in depth as score
  // return -P_(0, 0) * P_(1, 1) * P_(2, 2);
  return -P_(2, 2);
}

void Feature::Initialize(number_t z0, const Vec3 &std_xyz) {
  x_.head<2>() = Camera::instance()->UnProject(back());
#ifdef USE_INVDEPTH
  x_(2) = 1.0 / z0;
#else
  x_(2) = log(z0);
#endif

  // number_t rho = 1.0 / z0;
  // number_t rho_max = std::max(1.0 / (z0 - std_xyz(2)), 0.10);  // 0.10 is
  // inverse of max possible depth
  // number_t rho_min = 1.0 / (z0 + std_xyz(2));
  // number_t std_rho = std::max(fabs(rho - rho_min), fabs(rho - rho_max));

  P_ << std_xyz(0), 0, 0, 0, std_xyz(1), 0, 0, 0, std_xyz(2);
  P_ *= P_;
  status_ = FeatureStatus::INITIALIZING;
}

void Feature::SetRef(GroupPtr ref) {
#ifndef NDEBUG
  CHECK(ref_ == nullptr) << "reference already set!";
#endif
  // be very careful when reset references
  VLOG(0) << "ref group# " << ref->id() << " -> feature #" << id_;
  ref_ = ref;
}

void Feature::ResetRef(GroupPtr nref) {

  std::string str;
  if (nref == nullptr) {
    str = "nullptr";
  } else {
    str = "#" + std::to_string(nref->id());
  }

  VLOG(0) << "feature #" << id_ << " reset ref from #" << ref_->id() << " to "
          << str;

  ref_ = nref;
}

void Feature::SubfilterUpdate(const SE3 &gsb, const SE3 &gbc,
                              const SubfilterOptions &options) {

#ifndef NDEBUG
  CHECK(track_status() == TrackStatus::TRACKED);
  CHECK(status_ == FeatureStatus::INITIALIZING ||
        status_ == FeatureStatus::READY);
#endif

  init_counter_++;
  // depth sub-filter update
  Mat3 dXc_dx;
  Vec3 Xc = this->Xc(&dXc_dx);
  SE3 gtot = (gsb * gbc).inv() * ref()->gsb() * gbc; // g(curr cam <- ref cam)
  Vec3 Xcn = gtot * Xc;                              // predicted Xc
  Mat3 dXcn_dXc = gtot.rotation();
  Mat23 dxcn_dXcn;
  Vec2 xcn = project(Xcn, &dxcn_dXcn);

  Mat2 dxp_dxcn;
  Vec2 xp = Camera::instance()->Project(xcn, &dxp_dxcn);

  Mat23 H = dxp_dxcn * dxcn_dXcn * dXcn_dXc * dXc_dx;
  Vec2 inn = this->xp() - xp;

  Mat2 S = H * P_ * H.transpose();
  number_t Rtri = options.Rtri;
  S(0, 0) += Rtri;
  S(1, 1) += Rtri;

  number_t ratio{inn.dot(S.ldlt().solve(inn)) / options.MH_thresh};

  if (ratio > 1) {
    S(0, 0) += Rtri * (ratio - 1);
    S(1, 1) += Rtri * (ratio - 1);
    outlier_counter_ += sqrt(ratio);
  } else {
    outlier_counter_ = 0;
  }

  Mat32 K = P_ * H.transpose() * S.inverse(); // kalman gain

  x_ += K * inn;
  Mat3 I_KH = Mat3::Identity() - K * H;
  P_ = I_KH * P_ * I_KH.transpose() + K * Rtri * K.transpose();

  if (init_counter_ > options.ready_steps) {
    SetStatus(FeatureStatus::READY);
  } else {
    SetStatus(FeatureStatus::INITIALIZING);
  }
}

bool Feature::RefineDepth(const SE3 &gbc,
                          const std::vector<Observation> &observations,
                          const RefinementOptions &options) {

  std::vector<Observation> views;
  if (options.two_view) {
    auto[first, last] =
        std::minmax_element(std::begin(observations), std::end(observations),
                            [](const Observation &o1, const Observation &o2) {
                              return o1.g->id() < o1.g->id();
                            });
    views = {*first, *last};
  } else {
    views = observations;
  }

  Mat3 H, H0; // F'* invC *F, where F is measurement Jacobian, invC is inverse of measurement covariance
  Vec3 b;           // F' * invC * residual

  number_t res_norm0{0}; // norm of residual corresponding to optimal state
  // information matrix
  Mat2 invC;
  invC(0, 0) = 1. / options.Rtri;
  invC(1, 1) = 1. / options.Rtri;

  // using JacResTuple = std::tuple<Eigen::Matrix<number_t, 2, 3>, Eigen::Matrix<number_t, 2, 1>>;
  // std::vector<JacResTuple> jac_res;

  for (int iter = 0; iter < options.max_iters; ++iter) {
    Mat3 dXs_dx;
    Vec3 Xs = this->Xs(gbc, &dXs_dx); // ref_->gsb() * gbc * this->Xc();

    H.setZero();
    b.setZero();
    number_t res_norm{0};

    for (const auto &obs : views) {
      // skip reference group
      if (obs.g->id() == ref_->id())
        continue;

      SE3 g_cn_s = (obs.g->gsb() * gbc).inv(); // spatial -> camera new
      Vec3 Xcn = g_cn_s * Xs;
      // Mat3 dXc_dXs = gcs.rotation();
      Mat3 dXcn_dx = g_cn_s.R().matrix() * dXs_dx;

      Mat23 dxcn_dXcn;
      Vec2 xcn = project(Xcn, &dxcn_dXcn);

      Mat2 dxp_dxcn;
      Vec2 xp = Camera::instance()->Project(xcn, &dxp_dxcn);

      Mat23 dxp_dx = dxp_dxcn * dxcn_dXcn * dXcn_dx;

      H += (dxp_dx.transpose() * invC * dxp_dx); //  / (views.size() - 1);
      Vec2 res = xp - obs.xp;
      b += dxp_dx.transpose() * invC * res; //  / (views.size() - 1);

      // jac_res.push_back(std::make_tuple(dxp_dx, res));

      res_norm += res.norm(); //  / (views.size() - 1);
    }

    if (iter > 0 && res_norm > res_norm0) {
      // current state not good, revert
      RestoreState();
      break;
    }

    VLOG_IF(0, iter > 0) << StrFormat("iter=%d; |res|:%0.4f->%0.4f", 
        iter, res_norm0 / (views.size() - 1), res_norm / (views.size() - 1) );

    // auto ldlt = H.ldlt();
    // std::cout << "D=" << ldlt.vectorD().transpose() << std::endl;
    // Vec3 delta = H.ldlt().solve(b);
    Vec3 delta = H.completeOrthogonalDecomposition().solve(b);

    /*
    MatX J;
    J.setZero(2 * jac_res.size(), 3);
    VecX r;
    r.setZero(2 * jac_res.size());
    for (int i = 0; i < jac_res.size(); ++i) {
      const auto& tup{jac_res[i]};
      J.block<2, 3>(i * 2, 0) = std::get<0>(tup);
      r.segment<2>(i * 2) = std::get<1>(tup);
    }
    auto H = J.transpose() * J;
    auto ldlt = H.ldlt();
    std::cout << "D=" << ldlt.vectorD().transpose() << std::endl;
    auto b = J.transpose() * r;
    auto delta = ldlt.solve(b);
    */

    BackupState();
    x_ -= delta;
    res_norm0 = res_norm;

    // not much to progress
    if (delta.lpNorm<Eigen::Infinity>() < options.eps) {
      break;
    }
  }

  if (res_norm0 > options.max_res_norm) {
    VLOG(0) << StrFormat("feature #%d; status=%d; |res|=%f\n", id_,
                               as_integer(status_), res_norm0);
    return false;
  } 
    // std::cout << "H=\n" << H << std::endl;
    // std::cout << "H.inv=\n" << H.inverse() << std::endl;
    // std::cout << "P=\n" << P_ << std::endl;

  if (options.use_hessian) {
    // auto Hinv = H.inverse();
    // Pseudo-Inverse, since H is rank 2 (3x2 matrix times 2x2 matrix times 2x3 matrix)
    Mat3 H_pinv{H.completeOrthogonalDecomposition().pseudoInverse()};
    if (anynan(H_pinv)) {
      std::cout << "hessian as information matrix: nan in H.inv!!!" << std::endl;
      return false;
    }
    P_ = H_pinv;

#ifdef APPROXIMATE_INIT_COVARIANCE
    // std::cout << "approximating covariance using inverse of Hessian" << std::endl;
    // compute correlation blocks
    Mat3 dXc_dx;
    Vec3 Xc = this->Xc(&dXc_dx);

    SO3 Rr{ref_->gsb().R()};
    Vec3 Tr{ref_->gsb().T()};

    SO3 Rbc{gbc.R()};
    Vec3 Tbc{gbc.T()};

    // total rotation & translation w.r.t. body pose and alignment
    Mat3 dWtot_dWr, dWtot_dWbc;
    Mat3 dTtot_dWr, dTtot_dTr, dTtot_dTbc;
    // compose and compute the Jacobians
    auto [Rtot, Ttot] = Compose(Rr, Tr, Rbc, Tbc,
        &dWtot_dWr, &dWtot_dWbc,
        &dTtot_dWr, &dTtot_dTr, &dTtot_dTbc);
    // 3D point in spatial frame (Xs) w.r.t. total rotation & translation, and 3D point in camera frame (Xc)
    Mat3 dXs_dWtot, dXs_dTtot, dXs_dXc;
    auto Xs = Transform(Rtot, Ttot, Xc, &dXs_dWtot, &dXs_dTtot, &dXs_dXc);

    Mat3 dXs_dWr{dXs_dWtot * dWtot_dWr + dXs_dTtot * dTtot_dWr};
    Mat3 dXs_dTr{dXs_dTtot * dTtot_dTr};
    Mat3 dXs_dWbc{dXs_dWtot * dWtot_dWbc}; //  + dXs_dTtot * dTtot_dWbc};
    Mat3 dXs_dTbc{dXs_dTtot * dTtot_dTbc};
    Mat3 dXs_dx{dXs_dXc * dXc_dx};

    // allocate Jacobian matrices
    Eigen::Matrix<number_t, 2, kFeatureSize> Hx;  // dxp_dx
    Eigen::Matrix<number_t, 2, kGroupSize> Hc;  // dxp_d[Wbc, Tbc]
    Eigen::Matrix<number_t, 2, kGroupSize> Hr;  // dxp_d[Wr, Tr]
    Eigen::Matrix<number_t, 2, kGroupSize> Hg;  // dxp_d[Wbs, Tbs]
    Eigen::Matrix<number_t, 2, -1> Hstack;  // Jacobian stack: [Hx, Hc, Hr, Hg]
    Hstack.setZero(2, kFeatureSize + kGroupSize * 3);
    // allocate information matrices
    // Note: Ix, Ic and Ir should be accumulated
    MatX Ixcr;  // information matrix of [x, camera-body-alignment, reference group pose]
    Ixcr.setZero(kFeatureSize + kGroupSize * 2, kFeatureSize + kGroupSize * 2);

    for (const auto &obs : views) {
      Group* g{obs.g};
      if (g->id() == ref_->id() || !g->instate())
        continue;
      // Feeling too lasy to derive the Jacobians on paper,
      // so I'm gonna use chain rule to compute them.
      SO3 Rsb{g->gsb().R()};
      Vec3 Tsb{g->gsb().T()};
      // compute the total transformation from spatial frame to new camera frame
      Mat3 dWi_dWsb, dWi_dWbc; 
      Mat3 dTi_dWsb, dTi_dTsb, dTi_dTbc;
      // [Ri, Ti] = spatial to camera transformation
      auto [Ri, Ti] = InverseOfCompose(Rsb, Tsb,
          Rbc, Tbc,
          &dWi_dWsb, &dWi_dWbc, 
          &dTi_dWsb, &dTi_dTsb, &dTi_dTbc);

      // transfrom from spatial frame to new camera frame
      Mat3 dXcn_dWi, dXcn_dTi, dXcn_dXs;
      Vec3 Xcn = Transform(Ri, Ti, Xs,
          &dXcn_dWi, &dXcn_dTi, &dXcn_dXs);

      // intermediate Jacobians
      Mat3 dXcn_dx = dXcn_dXs * dXs_dx;
      Mat3 dXcn_dWsb = dXcn_dWi * dWi_dWsb + dXcn_dTi * dTi_dWsb;
      Mat3 dXcn_dTsb = dXcn_dTi * dTi_dTsb;
      Mat3 dXcn_dWbc = dXcn_dWi * dWi_dWbc + dXcn_dXs * dXs_dWbc; //  + dXcn_dTi * dTi_dWbc;
      Mat3 dXcn_dTbc = dXcn_dTi * dTi_dTbc + dXcn_dXs * dXs_dTbc;
      Mat3 dXcn_dWr = dXcn_dXs * dXs_dWr;
      Mat3 dXcn_dTr = dXcn_dXs * dXs_dTr;

      // perspective projection
      Mat23 dxcn_dXcn;
      Vec2 xcn = project(Xcn, &dxcn_dXcn);

      // apply distortion model
      Mat2 dxp_dxcn;
      Vec2 xp = Camera::instance()->Project(xcn, &dxp_dxcn);

      // fill-in Jacobian w.r.t. the group
      Mat23 dxp_dXcn{dxp_dxcn * dxcn_dXcn};
      Hg.block<2, 3>(0, 0) = dxp_dXcn * dXcn_dWsb; 
      Hg.block<2, 3>(0, 3) = dxp_dXcn * dXcn_dTsb;

      // update Jacobian w.r.t. the pose of the reference group
      Hr.block<2, 3>(0, 0) = dxp_dXcn * dXcn_dWr;
      Hr.block<2, 3>(0, 3) = dxp_dXcn * dXcn_dTr;

      // update Jacobian w.r.t. the pose of the camera-body alignment
      Hc.block<2, 3>(0, 0) = dxp_dXcn * dXcn_dWbc;
      Hc.block<2, 3>(0, 3) = dxp_dXcn * dXcn_dTbc;

      // update Jacobian w.r.t. local parametrization of the feature
      Hx = dxp_dXcn * dXcn_dx;
      Hstack << Hx, Hc, Hr, Hg;
      // std::cout << "Hstack=\n" << Hstack << std::endl;
      auto I = Hstack.transpose() * invC * Hstack;  // hessian as information matrix
      auto P = I.completeOrthogonalDecomposition().pseudoInverse(); // info mat -> cov mat
      // accumulate the top left corner of the info mat
      Ixcr += I.topLeftCorner(kFeatureSize + kGroupSize * 2, kFeatureSize + kGroupSize * 2);
      // keep the covariance block between feature state and group state
      int g_offset = kGroupBegin + kGroupSize * g->sind();
      cov_[g_offset] = P.block<kFeatureSize, kGroupSize>(0, kFeatureSize + kGroupSize * 2);
    }
    // convert info mat to cov mat
    auto P = Ixcr.completeOrthogonalDecomposition().pseudoInverse();
    cov_xc_ = P.block<kFeatureSize, kGroupSize>(0, kFeatureSize);
    cov_xr_ = P.block<kFeatureSize, kGroupSize>(0, kFeatureSize + kGroupSize);

    if (!P.isZero(0)) {
      std::cout << "P=\n" << P << std::endl;
      std::cout << "cov_xc=\n" << cov_xc_ << std::endl;
      std::cout << "cov_xr=\n" << cov_xr_ << std::endl;
    }
#endif
  }
  return true;
}

void Feature::ComputeJacobian(const Mat3 &Rsb, const Vec3 &Tsb, const Mat3 &Rbc,
                              const Vec3 &Tbc, const Vec3 &gyro, const Mat3 &Cg,
                              const Vec3 &bg, const Vec3 &Vsb, number_t td,
                              const VecX &error_state) {

  Mat3 Rsb_t = Rsb.transpose();
  Mat3 Rbc_t = Rbc.transpose();

  Mat3 Rr = ref_->Rsb();
  Vec3 Tr = ref_->Tsb();

  cache_.Xc = Xc(&cache_.dXc_dx);

  // Get components of the error state
  int offset = kGroupBegin + kGroupSize*(ref_->sind());
  Vec3 Wsb_err = error_state.segment<3>(Index::Wsb);
  Vec3 Tsb_err = error_state.segment<3>(Index::Tsb);
  Vec3 Wbc_err = error_state.segment<3>(Index::Wbc);
  Vec3 Tbc_err = error_state.segment<3>(Index::Tbc);
  Vec3 Wr_err  = error_state.segment<3>(offset);
  Vec3 Tr_err  = error_state.segment<3>(offset+3);

  // Get derivatives of error state matrix exponentials w.r.t vector.
  Mat93 dR_dWsb_err, dR_dWbc_err, dR_dWr_err;
  Mat3 R_Wsb_err = rodrigues(Wsb_err, &dR_dWsb_err);
  Mat3 R_Wbc_err = rodrigues(Wbc_err, &dR_dWbc_err);
  Mat3 R_Wr_err  = rodrigues(Wr_err, &dR_dWr_err);

  // Xc(ref) to Xs
  cache_.Xs = Rr * Rbc * cache_.Xc + Rr * Tbc + Tr;
  cache_.dXs_dx = Rr * Rbc * cache_.dXc_dx;
  cache_.dXs_dTbc = Rr;
  cache_.dXs_dTr = Mat3::Identity();
  for (int i=0; i<3; i++) {
    Mat3 dR_dWr_err_i  = unstack(dR_dWr_err.block<9,1>(0,i));
    Mat3 dR_dWbc_err_i = unstack(dR_dWbc_err.block<9,1>(0,i));
    // Compute derivatives
    Vec3 dXs_dWri = Rr * dR_dWr_err_i * (Rbc * cache_.Xc + Tbc);
    Vec3 dXs_dWbci = Rr * Rbc * dR_dWbc_err_i * cache_.Xc;
    // Fill in columns in cache_
    cache_.dXs_dWr.block<3,1>(0,i) = dXs_dWri;
    cache_.dXs_dWbc.block<3,1>(0,i) = dXs_dWbci;
  }

  // Xs back to Xc(new)
  cache_.Xcn = Rbc_t * Rsb_t * (cache_.Xs - Tsb) - Rbc_t * Tbc;
  cache_.dXcn_dXs = Rbc_t * Rsb_t;
  //cache_.dXcn_dTsb = -Rbc_t * Rsb_t;
  cache_.dXcn_dTsb = -cache_.dXcn_dXs;
  cache_.dXcn_dTbc = -Rbc_t + (cache_.dXcn_dXs * cache_.dXs_dTbc);
  for (int i=0; i<3; i++) {
    // Reshape columns of output of rodrigues()
    Mat3 dR_dWbc_err_i = unstack(dR_dWbc_err.block<9,1>(0,i));
    Mat3 dR_dWsb_err_i = unstack(dR_dWsb_err.block<9,1>(0,i));
    Mat3 dR_dWbc_err_i_t = dR_dWbc_err_i.transpose();
    Mat3 dR_dWsb_err_i_t = dR_dWsb_err_i.transpose();
    // Compute derivatives
    Vec3 dXcn_dWsb_err_i = Rbc_t * dR_dWsb_err_i_t * Rsb_t * (cache_.Xs - Tsb);
    Vec3 dXcn_dWbc_err_i = (dR_dWbc_err_i_t * Rbc_t * Rsb_t * cache_.Xs)
      + (R_Wbc_err.transpose() * Rbc_t * Rsb_t * cache_.dXs_dWbc.block<3,1>(0,i))
      - (dR_dWbc_err_i_t * Rbc_t * (Rsb_t*Tsb + Tbc));
    // Fill in columns in cache_
    cache_.dXcn_dWsb.block<3,1>(0,i) = dXcn_dWsb_err_i;
    cache_.dXcn_dWbc.block<3,1>(0,i) = dXcn_dWbc_err_i;
  }

  cache_.dXcn_dx = cache_.dXcn_dXs * cache_.dXs_dx;
  cache_.dXcn_dWr = cache_.dXcn_dXs * cache_.dXs_dWr;
  cache_.dXcn_dTr = cache_.dXcn_dXs * cache_.dXs_dTr;


#ifdef USE_ONLINE_TEMPORAL_CALIB
  Vec3 gyro_calib = Cg * gyro - bg;
  cache_.dXcn_dtd =
      -Rbc_t * (hat(gyro_calib) * Rsb_t * (cache_.Xs - Tsb) + Rsb_t * Vsb);

  // since imu.Cg is used here, also need to compute jacobian block w.r.t. Cg
  auto dXcn_dW =
      dAB_dB<3, 1>(Rbc_t * hat(Rsb_t * (cache_.Xs - Tsb)) * td); // W=Cg * Wm
#ifdef USE_ONLINE_IMU_CALIB
  Eigen::Matrix<number_t, 3, 9> dW_dCg;
  for (int i = 0; i < 3; ++i) {
    dW_dCg.block<1, 3>(i, 3 * i) = gyro;
  }
  cache_.dXcn_dCg = dXcn_dW * dW_dCg;
#endif
  cache_.dXcn_dbg = -dXcn_dW;
#endif

  // xc(new)
  cache_.xcn = project(cache_.Xcn, &cache_.dxcn_dXcn);

#ifdef USE_ONLINE_CAMERA_CALIB
  Eigen::Matrix<number_t, 2, -1> jacc;
  cache_.xp = Camera::instance()->Project(cache_.xcn, &cache_.dxp_dxcn, &jacc);
#else
  cache_.xp = Camera::instance()->Project(cache_.xcn, &cache_.dxp_dxcn);
#endif

  cache_.dxp_dXcn = cache_.dxp_dxcn * cache_.dxcn_dXcn;

  // set jacobians
  J_.setZero();
  J_.block<2, 3>(0, Index::Wsb) = cache_.dxp_dXcn * cache_.dXcn_dWsb;
  J_.block<2, 3>(0, Index::Tsb) = cache_.dxp_dXcn * cache_.dXcn_dTsb;
  J_.block<2, 3>(0, Index::Wbc) = cache_.dxp_dXcn * cache_.dXcn_dWbc;
  J_.block<2, 3>(0, Index::Tbc) = cache_.dxp_dXcn * cache_.dXcn_dTbc;
#ifdef USE_ONLINE_TEMPORAL_CALIB
  J_.block<2, 1>(0, Index::td) = cache_.dxp_dXcn * cache_.dXcn_dtd;
#ifdef USE_ONLINE_IMU_CALIB
  J_.block<2, 9>(0, Index::Cg) = cache_.dxp_dXcn * cache_.dXcn_dCg;
#endif
  J_.block<2, 3>(0, Index::bg) = cache_.dxp_dXcn * cache_.dXcn_dbg;
#endif

#ifndef NDEBUG
  CHECK(ref_->sind() != -1);
  CHECK(sind() != -1);
#endif
  int goff = kGroupBegin + 6 * ref_->sind();
  int foff = kFeatureBegin + 3 * sind();

  J_.block<2, 3>(0, goff) = cache_.dxp_dXcn * cache_.dXcn_dWr;
  J_.block<2, 3>(0, goff + 3) = cache_.dxp_dXcn * cache_.dXcn_dTr;
  J_.block<2, 3>(0, foff) = cache_.dxp_dXcn * cache_.dXcn_dx;

#ifdef USE_ONLINE_CAMERA_CALIB
  // fill-in jacobian w.r.t. camera intrinsics
  int dim{Camera::instance()->dim()};
  J_.block(0, kCameraBegin, 2, dim) = jacc.block(0, 0, 2, dim);
#endif

  // innovation
  cache_.inn = back() - cache_.xp;
  inn_ = cache_.inn;
}

void Feature::FillJacobianBlock(MatX &H, int offset) {
  H.block<2, 3>(offset, Index::Wsb) = J_.block<2, 3>(0, Index::Wsb);
  H.block<2, 3>(offset, Index::Tsb) = J_.block<2, 3>(0, Index::Tsb);
  H.block<2, 3>(offset, Index::Wbc) = J_.block<2, 3>(0, Index::Wbc);
  H.block<2, 3>(offset, Index::Tbc) = J_.block<2, 3>(0, Index::Tbc);

#ifdef USE_ONLINE_TEMPORAL_CALIB
  H.block<2, 1>(offset, Index::td) = J_.block<2, 1>(0, Index::td);
#ifdef USE_ONLINE_IMU_CALIB
  H.block<2, 9>(offset, Index::Cg) = J_.block<2, 9>(0, Index::Cg);
#endif
  H.block<2, 3>(offset, Index::bg) = J_.block<2, 3>(0, Index::bg);
#endif

  int goff = kGroupBegin + 6 * ref_->sind();
  int foff = kFeatureBegin + 3 * sind();

  H.block<2, 3>(offset, goff) = J_.block<2, 3>(0, goff);
  H.block<2, 3>(offset, goff) = J_.block<2, 3>(0, goff + 3);
  H.block<2, 3>(offset, foff) = J_.block<2, 3>(0, foff);

#ifdef USE_ONLINE_CAMERA_CALIB
  // fill-in jacobian w.r.t. camera intrinsics
  int dim{Camera::instance()->dim()};
  H.block(offset, kCameraBegin, 2, dim) = J_.block(0, kCameraBegin, 2, dim);
#endif
}

void Feature::Triangulate(const SE3 &gsb, const SE3 &gbc,
                          const TriangulateOptions &options) {
#ifndef NDEBUG
  CHECK(size() == 2);
#endif
  Vec2 xc1 = CameraManager::instance()->UnProject(front());
  Vec2 xc2 = CameraManager::instance()->UnProject(back());
  SE3 g12 = (ref_->gsb() * gbc).inv() * (gsb * gbc);

  // Vec3 Xc1 = options.method == 3 ? Triangulate1(g12, xc1, xc2)
  //                                : Triangulate2(g12, xc1, xc2);

  Vec3 Xc1;

  if(options.method == 1)
  {
    Xc1 = Triangulate1(g12, xc1, xc2);
  }

  else if(options.method == 2)
  {
    Xc1 = Triangulate2(g12, xc1, xc2);
  }

  else if(options.method == 3)
  {
    Xc1 = Triangulate1(g12, xc1, xc2);
  }

  else
  {
    std::cout << "[ERROR] Incorrect Method for Triangulation:" << options.method << std::endl;
    exit(1);
  }

  if (auto z = Xc1(2); z < options.zmin || z > options.zmax) {
    // triangulated depth is not great
    // stick to the constant depth
  } else {
    x_.head<2>() = Xc1.head<2>() / z;
#ifdef USE_INVDEPTH
    x_(2) = 1.0 / z;
#else
    x_(2) = log(z);
#endif
  }
}

void Feature::FillCovarianceBlock(MatX &P) {
  int size = P.rows();
  int offset = kFeatureBegin + kFeatureSize * sind_;
  // zero-out
  P.block(offset, 0, kFeatureSize, size).setZero();
  P.block(0, offset, size, kFeatureSize).setZero();
  // copy local covariance obtained during initialization to state covariance
  P.block<kFeatureSize, kFeatureSize>(offset, offset) = P_;

#ifdef APPROXIMATE_INIT_COVARIANCE
  // cross correlation of featur state (x) and spatial alignment (c)
  P.block<kFeatureSize, kGroupSize>(offset, Index::Wbc) = cov_xc_;
  P.block<kGroupSize, kFeatureSize>(Index::Wbc, offset) = cov_xc_.transpose();
  // cross correlation of x and reference group
  int ref_offset = kGroupBegin + kGroupSize * ref_->sind();
  P.block<kFeatureSize, kGroupSize>(offset, ref_offset) = cov_xr_;
  P.block<kGroupSize, kFeatureSize>(ref_offset, offset) = cov_xr_.transpose();

  for (auto [g_offset, cov] : cov_) {
    P.block<kFeatureSize, kGroupSize>(offset, g_offset) = cov;
    P.block<kGroupSize, kFeatureSize>(g_offset, offset) = cov.transpose();
  }
#endif
}

} // xivo
