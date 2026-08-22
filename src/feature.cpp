#include <algorithm>
#include <cstring>

#include "estimator.h"
#include "feature.h"
#include "group.h"
#include "helpers.h"
#include "mm.h"
#include "param.h"
#include "alias.h"
#include "rodrigues.h"
#include "stereo.h"

#include "glog/logging.h"

namespace xivo {

// Feature
int Feature::counter_ = Feature::counter0;
int Feature::num_good_triangulations_ = 0;
int Feature::num_bad_triangulations_ = 0;
JacobianCache Feature::cache_ = {};

// Operations for FeatureAdj
void FeatureAdj::Add(const Observation &obs) { insert({obs.g->id(), obs.xp}); }
void FeatureAdj::Remove(int id) { erase(id); }


namespace {

/** Copy a BRIEF descriptor out of the cv::Mat row OpenCV wrote it into.
 *
 *  A copy, not a cast of `mat.data`: FastBrief::TDescriptor is a value type
 *  (see fastbrief.h), the row is 32 bytes of CV_8U with no alignment guarantee,
 *  and DBoW2 keeps whatever it is given for as long as the vocabulary lives,
 *  which is longer than any pooled Feature's descriptor vector.
 */
FastBrief::TDescriptor ToDBoWDesc(const cv::Mat &mat) {
  FastBrief::TDescriptor desc;
  CHECK_EQ(mat.total() * mat.elemSize(), sizeof(desc))
      << "descriptor is not " << sizeof(desc) << " bytes";
  std::memcpy(desc.data(), mat.data, sizeof(desc));
  return desc;
}

} // namespace

FastBrief::TDescriptor Track::GetDBoWDesc() {
  return ToDBoWDesc(descriptors_.back());
}

std::vector<FastBrief::TDescriptor> Track::GetAllDBoWDesc() {
  std::vector<FastBrief::TDescriptor> ret;
  ret.reserve(descriptors_.size());
  for (const auto &d: descriptors_) {
    ret.push_back(ToDBoWDesc(d));
  }
  return ret;
}


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

FeaturePtr Feature::PointCloudWorldCreate(int fid, number_t x, number_t y) {
  FeaturePtr f = Feature::Create(x, y);
  f->id_ = fid;
  return f;
}

void Feature::Deactivate(FeaturePtr f) {
  MemoryManager::instance()->DeactivateFeature(f);
}

void Feature::Destroy(FeaturePtr f) {
  MemoryManager::instance()->DestroyFeature(f);
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
  // Features are recycled from the memory pool, so this must be cleared here
  // too: otherwise a fresh feature could inherit the right observation of
  // whichever feature previously occupied this slot.
  xp_r_ << -1, -1;
  has_right_ = false;
  stereo_seeded_ = false;
  J_.setZero();
  inn_ << 0, 0;
  J_r_.setZero();
  inn_r_ << 0, 0;
  right_jac_valid_ = false;
  outlier_counter_ = 0;
  lc_match_ = -1;
  triangulation_successful_ = false;

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
    *J = gsc.so3().matrix() * (*J);
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

bool Feature::instate() const {
  return (status_ == FeatureStatus::INSTATE) ||
         (status_ == FeatureStatus::GAUGE);
}

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

bool Feature::Merge(FeaturePtr f, const SE3& gbc) {

  // Change coordinates of new feature's estimates
  bool success = f->ChangeOwner(ref_, gbc);
  if (success) {
    // Covariance-weighted fusion of two independent Gaussian estimates:
    //
    //   P = (P1^-1 + P2^-1)^-1        = P1 (P1 + P2)^-1 P2
    //   x = P (P1^-1 x1 + P2^-1 x2)   = P2 (P1 + P2)^-1 x1 + P1 (P1 + P2)^-1 x2
    //
    // Each mean is weighted by the *other* estimate's covariance. The previous
    // code had both parts wrong: it weighted each mean by its own covariance, so
    // the more uncertain estimate dominated, and its fused covariance was
    // `(P1 + P2)^-1 (P1 + P2)`, i.e. the identity -- a merged feature came out
    // with a variance of 1 in normalised-pixel and log-depth units regardless of
    // what either input said.
    const Mat3 P1 = P_;
    const Mat3 P2 = f->P();
    const Mat3 S = P1 + P2;
    if (!S.allFinite() || std::abs(S.determinant()) < 1e-12) {
      return false;
    }
    const Mat3 S_inv = S.inverse();
    x_ = S_inv * (P2 * x_ + P1 * f->x());
    const Mat3 P_fused = P1 * S_inv * P2;
    // Symmetric in exact arithmetic; enforce it against round-off.
    P_ = 0.5 * (P_fused + P_fused.transpose());
    Xc();
    Xs(gbc);

    // Merge observations
    for (auto px: *f) {
      UpdateTrack(px);
    }
    // Merge descriptors
    for (auto desc: f->GetAllDescriptors()) {
      descriptors_.push_back(desc);
    }
  }
  return success;
}


bool Feature::ChangeOwner(GroupPtr nref, const SE3 &gbc,
                          ReanchorJacobians *jac_out) {
  // now transfer
  SE3 g_cn_s =
      (nref->gsb() * gbc)
          .inverse(); // spatial (s) to camera of the new reference (cn)
  Mat3 dXs_dx;
  Vec3 Xcn = g_cn_s * Xs(gbc, &dXs_dx);
  // Mat3 dXcn_dXs = gcb.R() * gbs.R();
  Mat3 dXcn_dx = g_cn_s.so3().matrix() * dXs_dx;
  Mat3 dxn_dXcn;

  if (Xcn(2) < 0) {
    return false;
  }

#ifdef USE_INVDEPTH
  Vec3 xn = project_invz(Xcn, &dxn_dXcn);
#else
  Vec3 xn = project_logz(Xcn, &dxn_dXcn);
#endif

  x_ = xn;
  Mat3 J = dxn_dXcn * dXcn_dx;

  // `P_` is the covariance only for an out-of-state (depth sub-filter) feature,
  // whose reference pose is treated as exact -- there `J` alone is the whole
  // story. Once the feature is in the EKF state its covariance lives in the 3x3
  // block of `Estimator::P_` at `kFeatureBegin + 3 * sind()` plus the
  // cross-covariance rows/columns against the rest of the state, none of which
  // is reachable from here.
  P_ = J * P_ * J.transpose();

  if (jac_out != nullptr) {
    // xn also depends on the *poses* of both the outgoing and the incoming
    // reference group, and both of those are error-state blocks:
    //
    //   xn = pi( (gsb_n gbc)^-1 gsb_o gbc pi^-1(x) )
    //
    // For an in-state feature the covariance transform is therefore a row
    // operation over three blocks, not a similarity transform on one. Dropping
    // the two pose terms understates the re-anchored uncertainty, and the
    // outgoing group is about to be deleted from the state without being
    // marginalized -- so if its contribution is not folded in here it is simply
    // lost. Derivatives use the same right (local) perturbation convention as
    // `ComputeJacobian`: Rsb <- Rsb exp(dW), Tsb <- Tsb + dT.
    const Mat3 Rbc_t = gbc.so3().matrix().transpose();
    const Mat3 R_cn_s = g_cn_s.so3().matrix();

    const Mat3 Rsb_o = ref_->Rsb().matrix();
    const Vec3 Xb_o = Rsb_o.transpose() * (Xs_ - ref_->Tsb());
    const Mat3 Rsb_n = nref->Rsb().matrix();
    const Vec3 Xb_n = Rsb_n.transpose() * (Xs_ - nref->Tsb());

    jac_out->dxn_dx = J;
    // d Xs / d[dW_o, dT_o] = [-Rsb_o hat(Xb_o), I], then through R_cn_s.
    jac_out->dxn_dref_old.leftCols<3>() =
        dxn_dXcn * R_cn_s * (-Rsb_o * SO3::hat(Xb_o));
    jac_out->dxn_dref_old.rightCols<3>() = dxn_dXcn * R_cn_s;
    // d Xcn / d[dW_n, dT_n] = [Rbc^T hat(Xb_n), -R_cn_s].
    jac_out->dxn_dref_new.leftCols<3>() =
        dxn_dXcn * (Rbc_t * SO3::hat(Xb_n));
    jac_out->dxn_dref_new.rightCols<3>() = dxn_dXcn * (-R_cn_s);
  }

  // Update other parameters
  ResetRef(nref);
  Xc();
  Xs(gbc);

  return true;
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
  SE3 gtot = (gsb * gbc).inverse() * ref()->gsb() * gbc; // g(curr cam <- ref cam)
  Vec3 Xcn = gtot * Xc;                              // predicted Xc
  Mat3 dXcn_dXc = gtot.so3().matrix();
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
  ClampLogDepth();
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
    // `o1.g->id() < o1.g->id()` compares o1 with itself and is therefore always
    // false, so `minmax_element` returned two positions fixed by the algorithm's
    // tie-breaking rather than the oldest and newest observation. When both
    // landed on the same element the two "views" were identical, `H` stayed zero,
    // and the refinement silently did nothing.
    auto[first, last] =
        std::minmax_element(std::begin(observations), std::end(observations),
                            [](const Observation &o1, const Observation &o2) {
                              return o1.g->id() < o2.g->id();
                            });
    views = {*first, *last};
  } else {
    views = observations;
  }

  Mat3 H, H0; // F'* invC *F, where F is measurement Jacobian, invC is inverse of measurement covariance
  Vec3 b;           // F' * invC * residual

  number_t res_norm0{0}; // norm of residual corresponding to optimal state
  // How many views actually contributed -- the reference group is skipped, and
  // `views` may hold duplicates. Needed to turn the summed residual into a
  // per-observation one for the acceptance test at the end.
  int num_res0{0};
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
    int num_res{0};

    for (const auto &obs : views) {
      // skip reference group
      if (obs.g->id() == ref_->id())
        continue;

      SE3 g_cn_s = (obs.g->gsb() * gbc).inverse(); // spatial -> camera new
      Vec3 Xcn = g_cn_s * Xs;
      // Mat3 dXc_dXs = gcs.rotation();
      Mat3 dXcn_dx = g_cn_s.so3().matrix() * dXs_dx;

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
      ++num_res;
    }

    if (num_res == 0) {
      // Every view was the reference group: nothing to refine against.
      return false;
    }

    if (iter > 0 && res_norm > res_norm0) {
      // current state not good, revert
      RestoreState();
      break;
    }

    VLOG_IF(0, iter > 0) << StrFormat("iter=%d; |res|:%0.4f->%0.4f",
        iter, res_norm0 / num_res0, res_norm / num_res );

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

    // H is rank-deficient when the views in `views` carry no parallax (e.g. a
    // near-stationary camera), and the least-squares solve can then return
    // non-finite values. Applying such a delta poisons x_ with NaN, which
    // propagates into the filter state and eventually aborts in
    // SO3_from_rotvec(). Treat it the same way as the NaN Hessian below:
    // abandon the refinement for this feature.
    if (anynan(delta)) {
      VLOG(0) << StrFormat("feature #%d: nan in depth-refinement delta", id_);
      return false;
    }

    BackupState();
    x_ -= delta;
    ClampLogDepth();
    res_norm0 = res_norm;
    num_res0 = num_res;

    // not much to progress
    if (delta.lpNorm<Eigen::Infinity>() < options.eps) {
      break;
    }
  }

  if (num_res0 == 0) {
    // The iteration never completed a usable pass.
    return false;
  }

  // `res_norm0` is a *sum* of per-observation reprojection-error norms, while
  // `max_res_norm` is a per-observation threshold -- the commented-out
  // `/ (views.size() - 1)` on each accumulation, and the log line above that
  // divides before printing, both say so. Comparing the sum made the test
  // strictly harder the more observations a feature had, so well-fitted
  // long-lived features were rejected while poorly-fitted two-view ones passed.
  const number_t mean_res = res_norm0 / num_res0;
  if (mean_res > options.max_res_norm) {
    VLOG(0) << StrFormat("feature #%d; status=%d; |res|=%f\n", id_,
                               as_integer(status_), mean_res);
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
                              const Vec3 &bg, const Vec3 &Vsb, number_t td) {

  // `J_` is a full-width row block of which only a handful of column blocks are
  // ever written, so stale columns survive from one call to the next. That
  // matters after `ChangeOwner`: the old reference group's six columns keep the
  // Jacobian w.r.t. a group this feature is no longer anchored to, and
  // `MHGating` forms `J_ * P_ * J_^T` over the *whole* row. Clear it.
  J_.setZero();

  Mat3 Rsb_t = Rsb.transpose();
  Mat3 Rbc_t = Rbc.transpose();

  Mat3 Rsbr = ref_->Rsb().matrix();
  Vec3 Tsbr = ref_->Tsb();

  // Index of feature in the whole group
  int offset = kGroupBegin + kGroupSize*(ref_->sind());

  // Compute position of feature in multiple coordinate frames
  cache_.Xc = Xc(&cache_.dXc_dx);
  cache_.Xbr = Rbc * cache_.Xc + Tbc;
  cache_.Xs = Rsbr * cache_.Xbr + Tsbr;
  cache_.Xb = Rsb_t * (cache_.Xs - Tsb);
  cache_.Xcn = Rbc_t * (cache_.Xb - Tbc);

  // Xbr to Xc
  cache_.dXbr_dXc = Rbc;
  cache_.dXbr_dTbc = Mat3::Identity();
  cache_.dXbr_dWbc = -Rbc * SO3::hat(cache_.Xc);

  // Xs to Xbr
  cache_.dXs_dXbr = Rsbr;
  cache_.dXs_dTsbr = Mat3::Identity();
  cache_.dXs_dWsbr = -Rsbr * SO3::hat(cache_.Xbr);

  // Xb to Xs
  cache_.dXb_dXs = Rsb_t;
  cache_.dXb_dTsb = -Rsb_t;
  cache_.dXb_dWsb = SO3::hat(cache_.Xb);

  // Xcn to Xb
  cache_.dXcn_dXb = Rbc_t;
  cache_.dXcn_dTbc = -Rbc_t +
    cache_.dXcn_dXb * cache_.dXb_dXs * cache_.dXs_dXbr * cache_.dXbr_dTbc;
  cache_.dXcn_dWbc = SO3::hat(cache_.Xcn) +
    cache_.dXcn_dXb * cache_.dXb_dXs * cache_.dXs_dXbr * cache_.dXbr_dWbc;

  // Chain rule values
  cache_.dXcn_dTsb = cache_.dXcn_dXb * cache_.dXb_dTsb;
  cache_.dXcn_dWsb = cache_.dXcn_dXb * cache_.dXb_dWsb;
  cache_.dXcn_dTsbr = cache_.dXcn_dXb * cache_.dXb_dXs * cache_.dXs_dTsbr;
  cache_.dXcn_dWsbr = cache_.dXcn_dXb * cache_.dXb_dXs * cache_.dXs_dWsbr;
  cache_.dXcn_dXs = cache_.dXcn_dXb * cache_.dXb_dXs;
  cache_.dXcn_dx = cache_.dXcn_dXs * cache_.dXs_dXbr * cache_.dXbr_dXc * cache_.dXc_dx;

#ifdef USE_ONLINE_TEMPORAL_CALIB
  Vec3 gyro_calib = Cg * gyro - bg;
  cache_.dXcn_dtd =
      -Rbc_t * (SO3::hat(gyro_calib) * Rsb_t * (cache_.Xs - Tsb) + Rsb_t * Vsb);

  // since imu.Cg is used here, also need to compute jacobian block w.r.t. Cg
  auto dXcn_dW =
      dAB_dB<3, 1>(Rbc_t * SO3::hat(Rsb_t * (cache_.Xs - Tsb)) * td); // W=Cg * Wm
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

  J_.block<2, 3>(0, goff) = cache_.dxp_dXcn * cache_.dXcn_dWsbr;
  J_.block<2, 3>(0, goff + 3) = cache_.dxp_dXcn * cache_.dXcn_dTsbr;
  J_.block<2, 3>(0, foff) = cache_.dxp_dXcn * cache_.dXcn_dx;

#ifdef USE_ONLINE_CAMERA_CALIB
  // fill-in jacobian w.r.t. camera intrinsics
  int dim{Camera::instance()->dim()};
  J_.block(0, kCameraBegin, 2, dim) = jacc.block(0, 0, 2, dim);
#endif

  // innovation
  cache_.inn = back() - cache_.xp;
  inn_ = cache_.inn;

  // The right camera's rows, while `cache_` still describes this feature.
  right_jac_valid_ = false;
  if (has_right_ && StereoRig::enabled()) {
    ComputeRightJacobian();
  }
}

void Feature::ComputeRightJacobian() {
  auto cam1 = Camera::instance(1);
  if (cam1 == nullptr) {
    return;
  }
  const StereoRig &rig = *StereoRig::instance();

  // The rig extrinsics are fixed and live outside the error state (see
  // stereo.h), so the *entire* dependence of the right observation on the state
  // flows through `cache_.Xcn` -- the same 3D point, in the left camera frame of
  // the current group, that the left rows were linearized about. That is what
  // lets these rows reuse the left camera's whole `dXcn_d*` chain verbatim: the
  // two cameras differ only in the last two links of the chain, the rigid hop
  // into camera 1 and camera 1's own projection.
  const Vec3 Xc1 = rig.Rc1c0() * cache_.Xcn + rig.Tc1c0();
  if (!(Xc1(2) > 0.0)) {
    // The current state predicts the point *behind* the right camera while the
    // tracker nonetheless matched something there. There is no meaningful
    // linearization of the projection here (and `project` would divide by a
    // negative depth), so contribute no rows rather than a garbage one.
    return;
  }

  Mat23 dxc1_dXc1;
  const Vec2 xc1 = project(Xc1, &dxc1_dXc1);
  Mat2 dxp1_dxc1;
  // NOTE: no `jacc` output, i.e. no USE_ONLINE_CAMERA_CALIB block below. Only
  // camera 0's intrinsics are in the error state; camera 1's stay fixed at their
  // calibrated values, which is consistent with holding the rig fixed.
  const Vec2 xp1 = cam1->Project(xc1, &dxp1_dxc1);
  if (!xp1.allFinite() || !dxp1_dxc1.allFinite()) {
    return;
  }
  const Mat23 dxp1_dXcn = dxp1_dxc1 * dxc1_dXc1 * rig.Rc1c0();

  J_r_.setZero();
  J_r_.block<2, 3>(0, Index::Wsb) = dxp1_dXcn * cache_.dXcn_dWsb;
  J_r_.block<2, 3>(0, Index::Tsb) = dxp1_dXcn * cache_.dXcn_dTsb;
  J_r_.block<2, 3>(0, Index::Wbc) = dxp1_dXcn * cache_.dXcn_dWbc;
  J_r_.block<2, 3>(0, Index::Tbc) = dxp1_dXcn * cache_.dXcn_dTbc;
#ifdef USE_ONLINE_TEMPORAL_CALIB
  J_r_.block<2, 1>(0, Index::td) = dxp1_dXcn * cache_.dXcn_dtd;
#ifdef USE_ONLINE_IMU_CALIB
  J_r_.block<2, 9>(0, Index::Cg) = dxp1_dXcn * cache_.dXcn_dCg;
#endif
  J_r_.block<2, 3>(0, Index::bg) = dxp1_dXcn * cache_.dXcn_dbg;
#endif

  int goff = kGroupBegin + 6 * ref_->sind();
  int foff = kFeatureBegin + 3 * sind();
  J_r_.block<2, 3>(0, goff) = dxp1_dXcn * cache_.dXcn_dWsbr;
  J_r_.block<2, 3>(0, goff + 3) = dxp1_dXcn * cache_.dXcn_dTsbr;
  J_r_.block<2, 3>(0, foff) = dxp1_dXcn * cache_.dXcn_dx;

  inn_r_ = xp_r_ - xp1;
  right_jac_valid_ = true;
}

void Feature::FillJacobianBlockFrom(
    MatX &H, int offset, const Eigen::Matrix<number_t, 2, kFullSize> &J) {
  H.block<2, 3>(offset, Index::Wsb) = J.block<2, 3>(0, Index::Wsb);
  H.block<2, 3>(offset, Index::Tsb) = J.block<2, 3>(0, Index::Tsb);
  H.block<2, 3>(offset, Index::Wbc) = J.block<2, 3>(0, Index::Wbc);
  H.block<2, 3>(offset, Index::Tbc) = J.block<2, 3>(0, Index::Tbc);

#ifdef USE_ONLINE_TEMPORAL_CALIB
  H.block<2, 1>(offset, Index::td) = J.block<2, 1>(0, Index::td);
#ifdef USE_ONLINE_IMU_CALIB
  H.block<2, 9>(offset, Index::Cg) = J.block<2, 9>(0, Index::Cg);
#endif
  H.block<2, 3>(offset, Index::bg) = J.block<2, 3>(0, Index::bg);
#endif

  int goff = kGroupBegin + 6 * ref_->sind();
  int foff = kFeatureBegin + 3 * sind();

  // Both of these used to be written to `goff`, so the rotation block was
  // overwritten by the translation Jacobian and the translation block at
  // `goff + 3` was left at zero (`H` is zeroed once per update in
  // `FilterUpdate`). Every stacked measurement therefore saw a wrong reference-
  // group rotation Jacobian and no translation Jacobian at all. `J_` itself was
  // always correct -- only this copy was broken, which is why
  // `unitTests_Jacobians` never caught it, and why OnePointRANSAC (which reads
  // `J_` directly) was unaffected.
  //
  // Read from the `J` parameter, not from `J_`: this body is shared with
  // `FillRightJacobianBlock`, which passes `J_r_`.
  H.block<2, 3>(offset, goff) = J.block<2, 3>(0, goff);
  H.block<2, 3>(offset, goff + 3) = J.block<2, 3>(0, goff + 3);
  H.block<2, 3>(offset, foff) = J.block<2, 3>(0, foff);

#ifdef USE_ONLINE_CAMERA_CALIB
  // fill-in jacobian w.r.t. camera intrinsics
  int dim{Camera::instance()->dim()};
  H.block(offset, kCameraBegin, 2, dim) = J.block(0, kCameraBegin, 2, dim);
#endif
}

void Feature::FillJacobianBlock(MatX &H, int offset) {
  FillJacobianBlockFrom(H, offset, J_);
}

void Feature::FillRightJacobianBlock(MatX &H, int offset) {
#ifndef NDEBUG
  CHECK(right_jac_valid_);
#endif
  FillJacobianBlockFrom(H, offset, J_r_);
}

void Feature::Triangulate(const SE3 &gsb, const SE3 &gbc,
                          const TriangulateOptions &options) {
#ifndef NDEBUG
  CHECK(size() == 2);
#endif
  Vec2 xc1 = CameraManager::instance()->UnProject(front());
  Vec2 xc2 = CameraManager::instance()->UnProject(back());
  SE3 g12 = (ref_->gsb() * gbc).inverse() * (gsb * gbc);

  Vec3 Xc1;
  bool return_output;

  if(options.method == "direct_linear_transform_svd")
  {
    return_output = DirectLinearTransformSVD(g12, xc1, xc2, Xc1);
  }

  else if(options.method == "direct_linear_transform_avg")
  {
    return_output = DirectLinearTransformAvg(g12, xc1, xc2, Xc1);
  }

  else if(options.method == "l1_angular")
  {
    return_output = L1Angular(g12, xc1, xc2, Xc1, options.max_theta_thresh, options.beta_thesh);
  }

  else if(options.method == "l2_angular")
  {
    return_output = L2Angular(g12, xc1, xc2, Xc1, options.max_theta_thresh, options.beta_thesh);
  }

  else if(options.method == "linf_angular")
  {
    return_output = LinfAngular(g12, xc1, xc2, Xc1, options.max_theta_thresh, options.beta_thesh);
  }

  else
  {
    LOG(ERROR) << "[ERROR] Incorrect Method for Triangulation: " << options.method;
    exit(1);
  }

  if(!return_output)
  {
    num_bad_triangulations_++;
    return;
  }

  // Written as a negated "is good" test, not "is bad": with `z < zmin ||
  // z > zmax` a NaN depth passed (every comparison against NaN is false) and
  // fell into the success branch, which set x_ to NaN, x_(2) to log(NaN), and
  // triangulation_successful_ to true. ClampLogDepth cannot rescue that either.
  // The NaN then propagated into J_, inn_ and the covariance update, and MH
  // gating could not reject it.
  if (auto z = Xc1(2); !(std::isfinite(z) && z >= options.zmin &&
                         z <= options.zmax && Xc1.allFinite())) {
    // triangulated depth is not great
    // stick to the constant depth
    num_bad_triangulations_++;
  } else {
    x_.head<2>() = Xc1.head<2>() / z;
    #ifdef USE_INVDEPTH
      x_(2) = 1.0 / z;
    #else
      x_(2) = log(z);
    #endif
    num_good_triangulations_++;
    triangulation_successful_ = true;
  }

  return;
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
