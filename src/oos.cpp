#include "feature.h"
#include "helpers.h"
#include "group.h"
#include "estimator.h"

namespace xivo {

int Feature::ComputeOOSJacobian(const std::vector<Observation> &vobs,
                                const Mat3 &Rbc, const Vec3 &Tbc) {

  int num_constraints =
      std::count_if(vobs.begin(), vobs.end(),
                    [](const Observation &obs) { return obs.g->instate(); });

// A constraint should involve at least 2 poses
  if (num_constraints >= Estimator::instance()->OOS_update_min_observations()) {
    cache_.Xs = this->Xs(SE3{SO3{Rbc}, Tbc});
    oos_jac_counter_ = 0;
    for (auto obs : vobs) {
      if (obs.g->instate()) {
        ComputeOOSJacobianInternal(obs, Rbc, Tbc);
      }
    }

    // perform givens elimination
//    oos_jac_counter_ = Givens(oos_.inn, oos_.Hx, oos_.Hf, 2 * oos_jac_counter_);
    // Two rows per observation, and only those rows are filled -- the rest of
    // `oos_` still holds whatever the previous user of this pooled Feature left
    // there. `SlowGivens` used to read the whole buffer.
    const int filled_rows = 2 * oos_jac_counter_;
    MatX A;
    oos_jac_counter_ = SlowGivens(oos_.Hf, oos_.Hx, oos_.inn, A, filled_rows);
    // std::cout << "feature #" << id_ << " got " << oos_jac_counter_ << " oos
    // jac blocks\n";
  } else {
    oos_jac_counter_ = 0;
  }

  return oos_jac_counter_;
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
