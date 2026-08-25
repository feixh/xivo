// The EKF measurement update. See ekf_update.h for the algebra.
// Author: efficiency work (branch auto-efficiency)
#include <cmath>

#include "glog/logging.h"

#include "ekf_update.h"

namespace xivo {

namespace {

#if !defined(NDEBUG) || defined(XIVO_CHECK_OCCUPIED_STATE)
/** The indices `live` does *not* cover. */
std::vector<int> IndicesOutside(const StateRuns &live, int n) {
  std::vector<bool> in(n, false);
  for (int i = 0; i < live.nruns; ++i) {
    for (int k = 0; k < live.runs[i].len; ++k) {
      in[live.runs[i].start + k] = true;
    }
  }
  std::vector<int> out;
  for (int i = 0; i < n; ++i) {
    if (!in[i]) {
      out.push_back(i);
    }
  }
  return out;
}

/** Verifies the two premises that make restricting the update to `live` exact
 *  (see `StateRuns` in core.h):
 *
 *  1. `H` has no nonzero column outside `live` -- a measurement cannot reference
 *     an unoccupied slot;
 *  2. every row and column of `P` outside `live` is zero *apart from its own
 *     diagonal entry* -- a slot nothing has observed is uncorrelated, though it
 *     does keep the prior variance `P_.setIdentity` gave it.
 *
 *  Note which claim this is *not* making. "Outside `live`, `P` is zero" is the
 *  intuitive statement and it is false: an untouched slot sits at variance 1 for
 *  the whole run. What matters is that it has no cross terms, because then its
 *  column of `H P` is `P(i,i) * H(:,i) = 0` and the update cannot move it or be
 *  moved by it.
 *
 *  O(N^2) per update, so it is compiled out of release builds; reachable from an
 *  optimized build with `-DXIVO_CHECK_OCCUPIED_STATE`, which is how it was run
 *  over the TUM-VI sequences. */
void CheckLiveExtent(const MatX &P, const MatX &H, const StateRuns &live) {
  for (int i : IndicesOutside(live, P.rows())) {
    CHECK_EQ(H.col(i).cwiseAbs().maxCoeff(), 0)
        << "H references state " << i << ", which is outside the live extent";
    for (int k = 0; k < P.rows(); ++k) {
      if (k == i) {
        continue;
      }
      CHECK_EQ(P(i, k), 0) << "P(" << i << "," << k << ") couples a state "
                              "outside the live extent";
      CHECK_EQ(P(k, i), 0) << "P(" << k << "," << i << ") couples a state "
                              "outside the live extent";
    }
  }
}
#endif

/** Zeros the columns of `M` that fall in the gaps between the runs of `live`. */
void ZeroOutsideRuns(MatX &M, const StateRuns &live) {
  int c = 0;
  for (int i = 0; i < live.nruns; ++i) {
    if (live.runs[i].start > c) {
      M.middleCols(c, live.runs[i].start - c).setZero();
    }
    c = live.runs[i].start + live.runs[i].len;
  }
  if (c < M.cols()) {
    M.middleCols(c, M.cols() - c).setZero();
  }
}

} // namespace

void MeasurementTimesCov(const MatX &H, const MatX &P,
                         const std::vector<MeasBlock> &blocks,
                         const StateRuns &live, MatX &M) {
  M.resize(H.rows(), P.cols());
  // The columns outside the occupied extent are columns of zeros of `P`, so they
  // are columns of zeros of the product; write them once here rather than let
  // three later loops step over them.
  ZeroOutsideRuns(M, live);
  for (const auto &b : blocks) {
    ColRun runs[kJacRuns];
    const bool sparse = b.sparse();
    if (sparse) {
      MeasurementRuns(b.gsind, b.fsind, runs);
    }
    for (int j = 0; j < live.nruns; ++j) {
      const ColRun &cj = live.runs[j];
      auto dst = M.block(b.row, cj.start, b.rows, cj.len);
      if (!sparse) {
        // A dense block still only meets the occupied rows of `P`, so the sum is
        // over `live` on both sides: `nruns^2` gemms rather than one rows x N x N.
        dst.setZero();
        for (int i = 0; i < live.nruns; ++i) {
          const ColRun &ci = live.runs[i];
          dst.noalias() += H.block(b.row, ci.start, b.rows, ci.len) *
                           P.block(ci.start, cj.start, ci.len, cj.len);
        }
        continue;
      }
      // Only the rows of `P` this block's nonzero columns select contribute, so
      // the product is a sum of a handful of short-and-wide gemms.
      dst.setZero();
      for (const auto &r : runs) {
        dst.noalias() += H.block(b.row, r.start, b.rows, r.len) *
                         P.block(r.start, cj.start, r.len, cj.len);
      }
    }
  }
}

namespace {

/** `S = M H^T`, using the block sparsity on the *columns* of the result and the
 *  occupied extent on the summation index. Restricting the latter is exact
 *  whatever `H` looks like there, because `M` is zero outside `live`. */
void CovTimesMeasurementT(const MatX &M, const MatX &H,
                          const std::vector<MeasBlock> &blocks,
                          const StateRuns &live, MatX &S) {
  S.resize(M.rows(), H.rows());
  for (const auto &b : blocks) {
    auto dst = S.middleCols(b.row, b.rows);
    if (!b.sparse()) {
      dst.setZero();
      for (int i = 0; i < live.nruns; ++i) {
        const ColRun &ci = live.runs[i];
        dst.noalias() += M.middleCols(ci.start, ci.len) *
                         H.block(b.row, ci.start, b.rows, ci.len).transpose();
      }
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

/** Mirrors the lower triangle of the `live` x `live` part of `P` into the upper.
 *  The diagonal blocks go column by column, as the whole matrix used to; each
 *  off-diagonal block pair is one blocked transpose. */
void MirrorLowerTriangle(MatX &P, const StateRuns &live) {
  for (int i = 0; i < live.nruns; ++i) {
    const ColRun &ri = live.runs[i];
    auto D = P.block(ri.start, ri.start, ri.len, ri.len);
    for (int j = 1; j < ri.len; ++j) {
      D.block(0, j, j, 1) = D.block(j, 0, 1, j).transpose();
    }
    for (int j = 0; j < i; ++j) {
      const ColRun &rj = live.runs[j]; // rj.start < ri.start
      P.block(rj.start, ri.start, rj.len, ri.len) =
          P.block(ri.start, rj.start, ri.len, rj.len).transpose();
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
                       const StateRuns &live, VecX &err) {
#if !defined(NDEBUG) || defined(XIVO_CHECK_OCCUPIED_STATE)
  CheckLiveExtent(P, H, live);
#endif
  MatX M, S;
  MeasurementTimesCov(H, P, blocks, live, M);
  CovTimesMeasurementT(M, H, blocks, live, S);
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

  // In place: `M` becomes `W = L^-1 (H P)`. Only the occupied columns; the rest
  // are zero, and `L^-1 0 = 0`.
  for (int i = 0; i < live.nruns; ++i) {
    llt.matrixL().solveInPlace(M.middleCols(live.runs[i].start, live.runs[i].len));
  }
  VecX u = inn;
  llt.matrixL().solveInPlace(u);
  err.setZero(P.rows());
  for (int i = 0; i < live.nruns; ++i) {
    const ColRun &r = live.runs[i];
    err.segment(r.start, r.len).noalias() =
        M.middleCols(r.start, r.len).transpose() * u;
  }

  // `P -= W^T W`, on the lower triangle only, then mirrored. Going through the
  // factor is what keeps the subtracted term symmetric and positive
  // semidefinite by construction; forming `K M` instead would leave a
  // difference of two rounded products whose asymmetry grows without bound.
  //
  // Restricted to the occupied extent: one `rankUpdate` per run on the diagonal
  // and one gemm per run pair below it, in place of a single `kFullSize`-square
  // `rankUpdate`. That step reads and writes the whole of `P` -- 2.5 MB -- and is
  // bandwidth-bound, so dropping the ~47% of it that is provably zero is close to
  // a proportional saving rather than a flop-count one.
  for (int i = 0; i < live.nruns; ++i) {
    const ColRun &ri = live.runs[i];
    auto Wi = M.middleCols(ri.start, ri.len);
    P.block(ri.start, ri.start, ri.len, ri.len)
        .selfadjointView<Eigen::Lower>()
        .rankUpdate(Wi.transpose(), -1.0);
    for (int j = 0; j < i; ++j) {
      const ColRun &rj = live.runs[j];
      P.block(ri.start, rj.start, ri.len, rj.len).noalias() -=
          Wi.transpose() * M.middleCols(rj.start, rj.len);
    }
  }
  MirrorLowerTriangle(P, live);
  return true;
}

} // namespace xivo
