// The EKF measurement update, as two free functions over (P, H, inn, R) rather
// than over Estimator's members, so that the fast one can be tested against the
// slow one on arbitrary inputs.
//
// Author: efficiency work (branch auto-efficiency)
#pragma once
#include <vector>

#include "core.h"

namespace xivo {

/** One row block of the stacked measurement Jacobian `H`, and what is known
 *  about its sparsity.
 *
 *  A visual measurement's two (or four, with the right camera) rows are nonzero
 *  in only `kJacCols` of the `kFullSize` columns -- the motion states, its
 *  reference group's slot, its own slot (see `kJacSharedRuns` in core.h). An
 *  out-of-state block, by contrast, spans every group its track was observed
 *  from, so it is treated as dense. Recording this per block as `H` is filled
 *  costs nothing and is what lets `H P` be formed in O(rows x kJacCols x N)
 *  instead of O(rows x N^2). */
struct MeasBlock {
  int row;   ///< first row of the block in `H`
  int rows;  ///< number of rows
  int gsind; ///< reference group slot, or -1 if the block is dense
  int fsind; ///< feature slot, unused when `gsind < 0`

  bool sparse() const { return gsind >= 0; }
};

/** The measurement update in the form the filter used before this work: form
 *  `S`, solve for the gain, and apply the Joseph (symmetrized) covariance
 *  update. Dense throughout, O(N^3).
 *
 *  It survives as the reference `EkfUpdateDowndate` is tested against, and as
 *  the fallback when the innovation covariance is not numerically positive
 *  definite -- the case where the cheap form has no Cholesky factor to work
 *  with. `unitTests_ekf_update` is what pins the two together.
 *
 *  `P` is updated in place, `err` is set to `K * inn` (not accumulated). */
void EkfUpdateJoseph(MatX &P, const MatX &H, const VecX &inn, const VecX &diagR,
                     VecX &err);

/** The same update, in the form that exploits (a) the block sparsity of `H` and
 *  (b) the identity that makes Joseph's extra terms redundant at the optimal
 *  gain.
 *
 *  With `M = H P` and `S = M H^T + R`, the optimal gain is `K = P H^T S^-1 =
 *  M^T S^-1` (using the symmetry of `P`), so
 *
 *      (I-KH) P (I-KH)^T + K R K^T = P - K M - M^T K^T + K S K^T
 *                                  = P - M^T S^-1 M
 *                                  = P - W^T W,      W = L^-1 M,  S = L L^T
 *
 *  because `K S K^T = M^T S^-1 M = K M`. The two N^3 products of the Joseph form
 *  collapse into one rank-`m` symmetric downdate, and applying it through the
 *  Cholesky factor keeps the subtracted term symmetric positive semidefinite by
 *  construction. This is the update OpenVINS uses (`StateHelper::EKFUpdate`).
 *
 *  Costs, at the shipped capacity with 76 in-state features (N = 564, m = 153):
 *  `H P` block-sparsely 2 MFLOP instead of 49, `S` 0.6, the triangular solve 7,
 *  the downdate 24 -- against ~600 MFLOP for `EkfUpdateJoseph`, which also
 *  computed `H P` twice.
 *
 *  `blocks` must cover every row of `H` exactly once, in order. Returns false
 *  and leaves `P` and `err` untouched if `S` is not numerically positive
 *  definite; the caller is expected to fall back to `EkfUpdateJoseph`. */
bool EkfUpdateDowndate(MatX &P, const MatX &H, const VecX &inn,
                       const VecX &diagR, const std::vector<MeasBlock> &blocks,
                       VecX &err);

/** `M = H P`, using the block sparsity. Exposed for the test, which checks it
 *  against the dense product. */
void MeasurementTimesCov(const MatX &H, const MatX &P,
                         const std::vector<MeasBlock> &blocks, MatX &M);

} // namespace xivo
