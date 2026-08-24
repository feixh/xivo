// The EKF measurement update. See ekf_update.h for the algebra.
// Author: efficiency work (branch auto-efficiency)
#include <cmath>

#include "glog/logging.h"

#include "ekf_update.h"

namespace xivo {

void MeasurementTimesCov(const MatX &H, const MatX &P,
                         const std::vector<MeasBlock> &blocks, MatX &M) {
  M.resize(H.rows(), P.cols());
  for (const auto &b : blocks) {
    auto dst = M.middleRows(b.row, b.rows);
    if (!b.sparse()) {
      dst.noalias() = H.middleRows(b.row, b.rows) * P;
      continue;
    }
    // Only the rows of `P` this block's nonzero columns select contribute, so
    // the product is a sum of a handful of short-and-wide gemms instead of one
    // rows x N x N gemm.
    ColRun runs[kJacRuns];
    MeasurementRuns(b.gsind, b.fsind, runs);
    dst.setZero();
    for (const auto &r : runs) {
      dst.noalias() += H.block(b.row, r.start, b.rows, r.len) *
                       P.middleRows(r.start, r.len);
    }
  }
}

namespace {

/** `S = M H^T`, using the same block sparsity on the *columns* of the result. */
void CovTimesMeasurementT(const MatX &M, const MatX &H,
                          const std::vector<MeasBlock> &blocks, MatX &S) {
  S.resize(M.rows(), H.rows());
  for (const auto &b : blocks) {
    auto dst = S.middleCols(b.row, b.rows);
    if (!b.sparse()) {
      dst.noalias() = M * H.middleRows(b.row, b.rows).transpose();
      continue;
    }
    ColRun runs[kJacRuns];
    MeasurementRuns(b.gsind, b.fsind, runs);
    dst.setZero();
    for (const auto &r : runs) {
      dst.noalias() += M.middleCols(r.start, r.len) *
                       H.block(b.row, r.start, b.rows, r.len).transpose();
    }
  }
}

} // namespace

void EkfUpdateJoseph(MatX &P, const MatX &H, const VecX &inn, const VecX &diagR,
                     VecX &err) {
  // `H * P` once, not twice as the original did -- otherwise this is the same
  // sequence of operations, including the `ldlt` solve and the separate
  // `K R K^T` term with the square root folded into `K`.
  MatX HP = H * P;
  MatX S = HP * H.transpose();
  for (int i = 0; i < diagR.size(); ++i) {
    S(i, i) += diagR(i);
  }

  MatX K(P.rows(), H.rows());
  K.transpose() = S.ldlt().solve(HP);
  err = K * inn;

  // As in the original: this is actually `K H - I`, but the covariance update is
  // quadratic in it, so the sign does not matter.
  MatX I_KH = K * H;
  for (int i = 0; i < I_KH.rows(); ++i) {
    I_KH(i, i) -= 1;
  }
  P = I_KH * P * I_KH.transpose();

  for (int i = 0; i < K.cols(); ++i) {
    K.col(i) *= std::sqrt(diagR(i));
  }
  P.noalias() += K * K.transpose();
}

bool EkfUpdateDowndate(MatX &P, const MatX &H, const VecX &inn,
                       const VecX &diagR, const std::vector<MeasBlock> &blocks,
                       VecX &err) {
  MatX M, S;
  MeasurementTimesCov(H, P, blocks, M);
  CovTimesMeasurementT(M, H, blocks, S);
  for (int i = 0; i < diagR.size(); ++i) {
    S(i, i) += diagR(i);
  }

  // `S` is symmetric positive definite whenever `P` is positive semidefinite and
  // the measurement noise is positive, which is the whole point of the Joseph
  // form's guarantee upstream. If it is not -- a covariance that has already
  // gone indefinite, or an `S` so ill-conditioned that its factor does not
  // exist in double -- there is no factor to downdate through, and the caller
  // falls back.
  Eigen::LLT<MatX> llt(S);
  if (llt.info() != Eigen::Success) {
    return false;
  }

  // In place: `M` becomes `W = L^-1 (H P)`.
  llt.matrixL().solveInPlace(M);
  VecX u = inn;
  llt.matrixL().solveInPlace(u);
  err.noalias() = M.transpose() * u;

  // `P -= W^T W`, on the lower triangle only, then mirrored. Going through the
  // factor is what keeps the subtracted term symmetric and positive
  // semidefinite by construction; forming `K M` instead would leave a
  // difference of two rounded products whose asymmetry grows without bound.
  P.selfadjointView<Eigen::Lower>().rankUpdate(M.transpose(), -1.0);
  for (int j = 1; j < P.cols(); ++j) {
    P.block(0, j, j, 1) = P.block(j, 0, 1, j).transpose();
  }
  return true;
}

} // namespace xivo
