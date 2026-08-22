#include <algorithm>
#include <cmath>

#include "feature.h"
#include "group.h"
#include "helpers.h"

namespace xivo {

namespace {

/** Pick at most `max_obs` observations out of `obs` (assumed sorted by group
 *  id, i.e. oldest first), evenly spaced, always keeping the first and the last
 *  one. The depth information of an out-of-state constraint lives in the
 *  parallax between the two ends of the track, while the cost of the update
 *  grows with the number of rows -- so when a track is very long we thin it out
 *  instead of dropping it. */
std::vector<Observation> ThinObservations(const std::vector<Observation> &obs,
                                          int max_obs) {
  const int n = obs.size();
  if (n <= max_obs) {
    return obs;
  }
  std::vector<Observation> out;
  out.reserve(max_obs);
  for (int i = 0; i < max_obs; ++i) {
    // i = 0 -> 0 and i = max_obs - 1 -> n - 1; strictly increasing since
    // (n - 1) / (max_obs - 1) > 1.
    int j = static_cast<int>(std::lround(i * static_cast<double>(n - 1) /
                                         (max_obs - 1)));
    out.push_back(obs[j]);
  }
  return out;
}

} // namespace

std::vector<Observation>
Feature::SelectOOSObservations(const std::vector<Observation> &vobs,
                              const OOSOptions &options) const {
  // Only observations made from a group that is part of the filter state are
  // usable: after the 3D point is marginalized out, what is left of the
  // measurement Jacobian multiplies group-pose error states, so a group that is
  // not in the state cannot be corrected (and its pose is treated as known).
  std::vector<Observation> out;
  out.reserve(vobs.size());
  for (const auto &obs : vobs) {
    if (obs.g->instate()) {
      out.push_back(obs);
    }
  }

  // Deterministic order (oldest group first): `GraphBase::GetObservationsOf`
  // walks a hash map, and both the thinning below and the row order of the
  // stacked Jacobian should not depend on that.
  std::sort(out.begin(), out.end(),
            [](const Observation &a, const Observation &b) {
              return a.g->id() < b.g->id();
            });

  const int max_obs = options.max_observations > 1
                          ? std::min(options.max_observations, kMaxGroup)
                          : kMaxGroup;
  return ThinObservations(out, max_obs);
}

bool Feature::RefineOOSDepth(const SE3 &gbc,
                             const std::vector<Observation> &views,
                             const OOSOptions &options) {
  // A single view carries no depth information at all.
  if (views.size() < 2) {
    return false;
  }

  const number_t invC = 1.0 / options.Rtri;

  Mat3 H;
  Vec3 b;
  Vec3 x_best{x_};
  number_t res_best{std::numeric_limits<number_t>::infinity()};
  bool converged{false};

  // One extra pass so that the state produced by the last Gauss-Newton step is
  // evaluated (and kept, if it is the best one) instead of being discarded.
  for (int iter = 0; iter <= options.max_iters; ++iter) {
    Mat3 dXs_dx;
    Vec3 Xs = this->Xs(gbc, &dXs_dx); // ref_->gsb() * gbc * this->Xc()

    H.setZero();
    b.setZero();
    number_t res_norm{0};

    for (const auto &obs : views) {
      SE3 g_cn_s = (obs.g->gsb() * gbc).inverse(); // spatial -> this camera
      Vec3 Xcn = g_cn_s * Xs;
      Mat3 dXcn_dx = g_cn_s.so3().matrix() * dXs_dx;

      Mat23 dxcn_dXcn;
      Vec2 xcn = project(Xcn, &dxcn_dXcn);

      Mat2 dxp_dxcn;
      Vec2 xp = Camera::instance()->Project(xcn, &dxp_dxcn);

      Mat23 dxp_dx = dxp_dxcn * dxcn_dXcn * dXcn_dx;
      Vec2 res = xp - obs.xp;

      H.noalias() += dxp_dx.transpose() * dxp_dx * invC;
      b.noalias() += dxp_dx.transpose() * res * invC;
      res_norm += res.norm();
    }
    // Mean -- not sum -- of the per-view residual norms, so that the acceptance
    // threshold below has the same meaning (pixels of reprojection error)
    // whatever the length of the track. `Feature::RefineDepth` gates on the sum,
    // which silently rejects every well-tracked feature.
    res_norm /= views.size();

    if (!std::isfinite(res_norm)) {
      // A view behind the camera (or a saturated log-depth) makes the
      // projection blow up; nothing to salvage.
      x_ = x_best;
      return false;
    }

    if (res_norm < res_best) {
      res_best = res_norm;
      x_best = x_;
    } else if (iter > 0) {
      // Gauss-Newton stepped uphill: keep the best state seen so far.
      break;
    }

    if (converged || iter == options.max_iters) {
      break;
    }

    Vec3 delta = H.completeOrthogonalDecomposition().solve(b);
    if (anynan(delta)) {
      // H is rank deficient when the views carry no parallax; the solve can
      // then return non-finite values which would poison x_ with NaN.
      x_ = x_best;
      return false;
    }

    x_ -= delta;
    ClampLogDepth();
    converged = delta.lpNorm<Eigen::Infinity>() < options.eps;
  }

  x_ = x_best;
  oos_mean_reproj_err_ = res_best;

  if (res_best > options.max_mean_reproj_err) {
    return false;
  }
  // Depth in the reference camera. An out-of-state constraint ties together a
  // whole window of poses, so a badly triangulated point does much more damage
  // than a single bad in-state measurement -- be strict here.
  number_t depth = this->z();
  if (!(depth > options.zmin) || !(depth < options.zmax)) {
    return false;
  }
  return true;
}

int Feature::ComputeOOSJacobian(const std::vector<Observation> &vobs,
                                const Mat3 &Rbc, const Vec3 &Tbc,
                                const OOSOptions &options) {
  oos_jac_counter_ = 0;
  oos_num_obs_ = 0;

  std::vector<Observation> views{SelectOOSObservations(vobs, options)};

  // The 3D point has 3 degrees of freedom and each view contributes 2
  // equations, so at least 2 views are needed for `2n - 3` to be positive.
  if (static_cast<int>(views.size()) < std::max(2, options.min_observations)) {
    return 0;
  }

  cache_.Xs = this->Xs(SE3{SO3{Rbc}, Tbc});
  for (const auto &obs : views) {
    ComputeOOSJacobianInternal(obs, Rbc, Tbc);
  }
  // `ComputeOOSJacobianInternal` uses oos_jac_counter_ as the observation
  // counter; from here on it holds the number of rows of the marginalized
  // measurement (see `ro()` / `Ho()`).
  oos_num_obs_ = oos_jac_counter_;
  oos_jac_counter_ = MarginalizeOOSPoint(2 * oos_num_obs_);

  return oos_jac_counter_;
}

int Feature::MarginalizeOOSPoint(int rows) {
  if (rows < 4) {
    return 0;
  }
  CHECK_LE(rows, oos_.Hf.rows());

  // Householder QR of Hf (rows x 3): Hf = Q * [R; 0] with R upper triangular,
  // hence every column of Hf lies in the span of the first three columns of Q
  // and A := Q(:, 3:) satisfies A' * Hf = 0 exactly -- also when Hf is rank
  // deficient (no parallax), in which case one row of information is simply
  // thrown away.
  //
  // Q is orthonormal, which matters: the projected measurement noise covariance
  // is A' * (sigma^2 I) * A = sigma^2 I only for an orthonormal A, and the
  // update assumes exactly that (it feeds `Roos_` as a diagonal). The previous
  // implementation used the (non-orthonormal) kernel of a FullPivLU instead,
  // and it also operated on the whole over-sized buffer -- including rows left
  // over from the previous feature -- rather than on the `rows` just filled.
  Eigen::HouseholderQR<MatX> qr(oos_.Hf.topRows(rows));
  MatX Q = qr.householderQ();
  const int out_rows = rows - 3;
  auto A = Q.rightCols(out_rows);

  MatX Hx = A.transpose() * oos_.Hx.topRows(rows);
  VecX inn = A.transpose() * oos_.inn.head(rows);
  oos_.Hx.topRows(out_rows) = Hx;
  oos_.inn.head(out_rows) = inn;

  return out_rows;
}

void Feature::ComputeOOSJacobianInternal(const Observation &obs,
                                         const Mat3 &Rbc, const Vec3 &Tbc) {

  auto g = obs.g;
  CHECK(g->sind() != -1);

  int goff = kGroupBegin + 6 * obs.g->sind();
  Mat3 Rsb = g->Rsb().matrix();
  Mat3 Rsb_t = Rsb.transpose();
  Vec3 Tsb = g->Tsb();
  Mat3 Rbc_t = Rbc.transpose();

  // Xb to Xs
  cache_.Xb = Rsb_t * (cache_.Xs - Tsb);
  cache_.dXb_dXs = Rsb_t;
  cache_.dXb_dTsb = -Rsb_t;
  cache_.dXb_dWsb = SO3::hat(cache_.Xb);

  // Xcn to Xb
  cache_.Xcn = Rbc_t * (cache_.Xb - Tbc);
  cache_.dXcn_dXb = Rbc_t;
  cache_.dXcn_dWbc = SO3::hat(cache_.Xcn);
  cache_.dXcn_dTbc = -Rbc_t;

  // Other values
  cache_.dXcn_dXs = cache_.dXcn_dXb * cache_.dXb_dXs;
  cache_.dXcn_dWsb = cache_.dXcn_dXb * cache_.dXb_dWsb;
  cache_.dXcn_dTsb = cache_.dXcn_dXb * cache_.dXb_dTsb;

  cache_.xcn = project(cache_.Xcn, &cache_.dxcn_dXcn);

  cache_.xp = Camera::instance()->Project(cache_.xcn, &cache_.dxp_dxcn);

  cache_.dxp_dXcn = cache_.dxp_dxcn * cache_.dxcn_dXcn;

  oos_.inn.segment<2>(2 * oos_jac_counter_) = obs.xp - cache_.xp;

  oos_.Hf.block<2, 3>(2 * oos_jac_counter_, 0) =
      cache_.dxp_dXcn * cache_.dXcn_dXb * cache_.dXb_dXs;

  oos_.Hx.block<2, kFullSize>(2 * oos_jac_counter_, 0).setZero();
  oos_.Hx.block<2, 3>(2 * oos_jac_counter_, goff) =
      cache_.dxp_dXcn * cache_.dXcn_dXb * cache_.dXb_dWsb;
  oos_.Hx.block<2, 3>(2 * oos_jac_counter_, goff + 3) =
      cache_.dxp_dXcn * cache_.dXcn_dXb * cache_.dXb_dTsb;
  oos_.Hx.block<2, 3>(2 * oos_jac_counter_, Index::Wbc) =
      cache_.dxp_dXcn * cache_.dXcn_dWbc;
  oos_.Hx.block<2, 3>(2 * oos_jac_counter_, Index::Tbc) =
      cache_.dxp_dXcn * cache_.dXcn_dTbc;
  ++oos_jac_counter_;
}


void Feature::ComputeLCJacobian(const Obs &obs, const Mat3 &Rbc,
                                const Vec3 &Tbc,
                                int match_counter, MatX &H, VecX &inn)
{
  auto g = obs.g;

  int goff = kGroupBegin + 6 * obs.g->sind();
  Mat3 Rsb = g->Rsb().matrix();
  Mat3 Rsb_t = Rsb.transpose();
  Vec3 Tsb = g->Tsb();
  Mat3 Rbc_t = Rbc.transpose();
  SE3 gbc(SO3(Rbc), Tbc);

  // Xb to Xs
  cache_.Xb = Rsb_t * (Xs(gbc) - Tsb);
  cache_.dXb_dXs = Rsb_t;
  cache_.dXb_dTsb = -Rsb_t;
  cache_.dXb_dWsb = SO3::hat(cache_.Xb);

  // Xcn to Xb
  cache_.Xcn = Rbc_t * (cache_.Xb - Tbc);
  cache_.dXcn_dXb = Rbc_t;
  cache_.dXcn_dTbc = -Rbc_t;
  cache_.dXcn_dWbc = SO3::hat(cache_.Xcn);
  cache_.dXcn_dXs = cache_.dXcn_dXb * cache_.dXb_dXs;

  // Chain rule values
  cache_.dXcn_dTsb = cache_.dXcn_dXb * cache_.dXb_dTsb;
  cache_.dXcn_dWsb = cache_.dXcn_dXb * cache_.dXb_dWsb;

  cache_.xcn = project(cache_.Xcn, &cache_.dxcn_dXcn);

#ifdef USE_ONLINE_CAMERA_CALIB
  Eigen::Matrix<number_t, 2, -1> jacc;
  cache_.xp = Camera::instance()->Project(cache_.xcn, &cache_.dxp_dxcn, &jacc);
#else
  cache_.xp = Camera::instance()->Project(cache_.xcn, &cache_.dxp_dxcn);
#endif

  cache_.dxp_dXcn = cache_.dxp_dxcn * cache_.dxcn_dXcn;

  int st = 2*match_counter;
  H.block<2, 3>(st, goff) = cache_.dxp_dXcn * cache_.dXcn_dWsb;
  H.block<2, 3>(st, goff + 3) = cache_.dxp_dXcn * cache_.dXcn_dTsb;
  H.block<2, 3>(st, Index::Wbc) = cache_.dxp_dXcn * cache_.dXcn_dWbc;
  H.block<2, 3>(st, Index::Tbc) = cache_.dxp_dXcn * cache_.dXcn_dTbc;

#ifdef USE_ONLINE_CAMERA_CALIB
  int dim{Camera::instance()->dim()};
  H.block(st, kCameraBegin, 2, dim) = jacc;
#endif

  inn.segment<2>(st) = obs.xp - cache_.xp;
}

} // namespace xivo
