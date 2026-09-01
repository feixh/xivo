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

/** One past the last *fixed* shared run: `kJacSharedRuns` ends with the group
 *  sentinel, whose start depends on the block, and the runs before it are the
 *  same columns for every visual measurement in the update. */
constexpr int kJacFixedRuns = static_cast<int>(kJacSharedRuns.size()) - 1;

/** The reference group's column run of a sparse block. */
inline ColRun GroupRun(const MeasBlock &b) {
  return {kGroupBegin + kGroupSize * b.gsind, kGroupSize};
}

/** The feature's own column run of a sparse block. */
inline ColRun FeatureRun(const MeasBlock &b) {
  return {kFeatureBegin + kFeatureSize * b.fsind, kFeatureSize};
}

/** One past the last block of the maximal run of consecutive sparse blocks
 *  starting at `b0`, and the number of rows of `H` it spans. Blocks cover the
 *  rows of `H` in order and without gaps, so a run of blocks is a run of rows. */
inline int SparseSpanEnd(const std::vector<MeasBlock> &blocks, int b0,
                         int &rows) {
  int b1 = b0;
  while (b1 < static_cast<int>(blocks.size()) && blocks[b1].sparse()) {
    ++b1;
  }
  rows = blocks[b1 - 1].row + blocks[b1 - 1].rows - blocks[b0].row;
  return b1;
}

/** One past the last block in `[b0, b1)` that names the same reference group as
 *  `b0` (`Group`) or the same group *and* feature (`Feature`), and the rows it
 *  spans. A feature's left- and right-camera rows are two adjacent blocks with
 *  identical slots, so the `Feature` grouping halves the block count in stereo,
 *  and a group is usually shared by several consecutive features. */
enum class MergeBy { Group, Feature };
inline int MergeEnd(const std::vector<MeasBlock> &blocks, int b0, int b1,
                    MergeBy by, int &rows) {
  int k = b0 + 1;
  while (k < b1 && blocks[k].gsind == blocks[b0].gsind &&
         (by == MergeBy::Group || blocks[k].fsind == blocks[b0].fsind)) {
    ++k;
  }
  rows = blocks[k - 1].row + blocks[k - 1].rows - blocks[b0].row;
  return k;
}

/** The column runs a dense block's summation index has to cover: its own recorded
 *  runs when it has them, otherwise the whole live extent. Returned by value into
 *  a caller-owned `RunSet` so both loops below can iterate one thing.
 *
 *  A block's runs are a subset of `live` in practice (an out-of-state track only
 *  references in-state groups), but nothing here depends on that: a run outside
 *  `live` contributes `H(:, i) * P(i, i)` with `H(:, i)` zero. */
inline void DenseSumRuns(const MeasBlock &b, const StateRuns &live, RunSet &out) {
  out.Clear();
  if (b.runs != nullptr) {
    for (int i = 0; i < b.runs->nruns; ++i) {
      out.Add(b.runs->runs[i].start, b.runs->runs[i].len);
    }
    return;
  }
  for (int i = 0; i < live.nruns; ++i) {
    out.Add(live.runs[i].start, live.runs[i].len);
  }
}

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

  // The flop count of this product is tiny -- 283 rows x 25 nonzero columns x 331
  // live columns is 2.3 MFLOP -- but issuing it as one gemm per (row block,
  // column run, live run) meant ~1700 calls per stereo update with M as small as
  // 2 and K as small as 1, and Eigen's per-call cost then dominated. It is
  // batched three ways below, all of them exact rather than approximate:
  //
  //  - the fixed shared runs (`Wsb Tsb`, `bg`, `Wbc Tbc`, `td`) are the *same*
  //    columns for every visual measurement, so they are driven once over the
  //    whole span of consecutive sparse blocks;
  //  - a feature's left- and right-camera rows are two adjacent blocks with the
  //    same group and feature slot, and consecutive features usually share a
  //    reference group, so the group run is driven over merged spans;
  //  - only the feature run is left per feature.
  //
  // Each output element still accumulates its runs in the same order -- the
  // fixed shared runs ascending, then the group, then the feature -- and no
  // merge changes K (a merged group gemm still has K = 6, a merged feature gemm
  // K = 3). That makes this *mathematically* the same sum, but it is **not
  // bit-identical**: merging changes M, and Eigen's gemm is not shape-invariant
  // in the last bit -- a different M means a different LHS packing and a
  // different row-peeling path through `gebp_kernel`, hence different rounding.
  //
  // Measured, by computing both forms and comparing element by element over a
  // whole room3 run: they differ on every update, in ~1-50% of the elements of
  // `M` and `S`, by at most 2e-13 of the matrix's own max magnitude (typically
  // 2e-16, i.e. one ulp; the large *relative* differences are all on elements
  // near zero, where the sum cancels). This is a reassociation, not an error --
  // `unitTests_ekf_update` checks the result against the dense Joseph form.
  // The branch carries a full 6-member ensemble as its accuracy proof; see
  // notes-speed/m2-batched-sparse-products.md.
  const int nb = static_cast<int>(blocks.size());
  for (int b0 = 0; b0 < nb;) {
    if (!blocks[b0].sparse()) {
      // A dense block still only meets the occupied rows of `P`, so the sum is
      // over `live` on both sides: `nruns^2` gemms rather than one rows x N x N.
      // When the block knows its own columns (an out-of-state one does) the
      // summation index shrinks from the whole live extent to those, which is
      // where the `live x live` read of `P` -- 1.9 MB per block -- goes away.
      const MeasBlock &b = blocks[b0];
      RunSet sum;
      DenseSumRuns(b, live, sum);
      for (int j = 0; j < live.nruns; ++j) {
        const ColRun &cj = live.runs[j];
        auto dst = M.block(b.row, cj.start, b.rows, cj.len);
        dst.setZero();
        for (int i = 0; i < sum.nruns; ++i) {
          const ColRun &ci = sum.runs[i];
          dst.noalias() += H.block(b.row, ci.start, b.rows, ci.len) *
                           P.block(ci.start, cj.start, ci.len, cj.len);
        }
      }
      ++b0;
      continue;
    }

    int span_rows = 0;
    const int b1 = SparseSpanEnd(blocks, b0, span_rows);
    const int r0 = blocks[b0].row;

    for (int j = 0; j < live.nruns; ++j) {
      const ColRun &cj = live.runs[j];
      M.block(r0, cj.start, span_rows, cj.len).setZero();
    }
    for (int s = 0; s < kJacFixedRuns; ++s) {
      const ColRun &rs = kJacSharedRuns[s];
      for (int j = 0; j < live.nruns; ++j) {
        const ColRun &cj = live.runs[j];
        M.block(r0, cj.start, span_rows, cj.len).noalias() +=
            H.block(r0, rs.start, span_rows, rs.len) *
            P.block(rs.start, cj.start, rs.len, cj.len);
      }
    }
    for (int i = b0; i < b1;) {
      int rows = 0;
      const int k = MergeEnd(blocks, i, b1, MergeBy::Group, rows);
      const ColRun rg = GroupRun(blocks[i]);
      for (int j = 0; j < live.nruns; ++j) {
        const ColRun &cj = live.runs[j];
        M.block(blocks[i].row, cj.start, rows, cj.len).noalias() +=
            H.block(blocks[i].row, rg.start, rows, rg.len) *
            P.block(rg.start, cj.start, rg.len, cj.len);
      }
      i = k;
    }
    for (int i = b0; i < b1;) {
      int rows = 0;
      const int k = MergeEnd(blocks, i, b1, MergeBy::Feature, rows);
      const ColRun rf = FeatureRun(blocks[i]);
      for (int j = 0; j < live.nruns; ++j) {
        const ColRun &cj = live.runs[j];
        M.block(blocks[i].row, cj.start, rows, cj.len).noalias() +=
            H.block(blocks[i].row, rf.start, rows, rf.len) *
            P.block(rf.start, cj.start, rf.len, cj.len);
      }
      i = k;
    }
    b0 = b1;
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
  // Batched exactly as `MeasurementTimesCov` above, and for the same reason: the
  // columns of `S` a block owns are its rows of `H`, so a span of consecutive
  // sparse blocks owns a contiguous span of columns and the fixed shared runs
  // collapse into one gemm each over the whole span. Same caveat as above: the
  // sum is the same one, reassociated at the last bit, not bit-identical.
  const int nb = static_cast<int>(blocks.size());
  for (int b0 = 0; b0 < nb;) {
    if (!blocks[b0].sparse()) {
      const MeasBlock &b = blocks[b0];
      RunSet sum;
      DenseSumRuns(b, live, sum);
      auto dst = S.middleCols(b.row, b.rows);
      dst.setZero();
      for (int i = 0; i < sum.nruns; ++i) {
        const ColRun &ci = sum.runs[i];
        dst.noalias() += M.middleCols(ci.start, ci.len) *
                         H.block(b.row, ci.start, b.rows, ci.len).transpose();
      }
      ++b0;
      continue;
    }

    int span_rows = 0;
    const int b1 = SparseSpanEnd(blocks, b0, span_rows);
    const int r0 = blocks[b0].row;

    S.middleCols(r0, span_rows).setZero();
    for (int s = 0; s < kJacFixedRuns; ++s) {
      const ColRun &rs = kJacSharedRuns[s];
      S.middleCols(r0, span_rows).noalias() +=
          M.middleCols(rs.start, rs.len) *
          H.block(r0, rs.start, span_rows, rs.len).transpose();
    }
    for (int i = b0; i < b1;) {
      int rows = 0;
      const int k = MergeEnd(blocks, i, b1, MergeBy::Group, rows);
      const ColRun rg = GroupRun(blocks[i]);
      S.middleCols(blocks[i].row, rows).noalias() +=
          M.middleCols(rg.start, rg.len) *
          H.block(blocks[i].row, rg.start, rows, rg.len).transpose();
      i = k;
    }
    for (int i = b0; i < b1;) {
      int rows = 0;
      const int k = MergeEnd(blocks, i, b1, MergeBy::Feature, rows);
      const ColRun rf = FeatureRun(blocks[i]);
      S.middleCols(blocks[i].row, rows).noalias() +=
          M.middleCols(rf.start, rf.len) *
          H.block(blocks[i].row, rf.start, rows, rf.len).transpose();
      i = k;
    }
    b0 = b1;
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
