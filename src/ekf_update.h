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

/** The same update, in the form that exploits (a) the block sparsity of `H`,
 *  (b) the identity that makes Joseph's extra terms redundant at the optimal
 *  gain, and (c) the fact that most of the state is unoccupied.
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
 *  `live` names the rows and columns of `P` the update has to touch (see
 *  `StateRuns` in core.h). Everything -- the columns of `M = H P`, the triangular
 *  solve, the downdate, the mirror and `err` -- is restricted to them. That is
 *  exact rather than approximate because a state outside `live` is *uncorrelated*
 *  with every other (it may still carry a variance), and `H` is zero there, so its
 *  column of `H P` is zero and the update neither moves it nor is moved by it.
 *  Debug builds, and `-DXIVO_CHECK_OCCUPIED_STATE`, verify both premises against
 *  the actual `P` and `H`.
 *
 *  Pass `WholeState()` to restrict nothing. The runs are permitted to be larger
 *  than the true occupied set; a slot that is inside a run but vacant costs
 *  arithmetic and changes no result.
 *
 *  Costs, at the shipped capacity with 76 in-state features (N = 564, m = 153,
 *  live dim ~350): `H P` block-sparsely 1.3 MFLOP instead of 49, `S` 0.6, the
 *  triangular solve 4, the downdate 9 -- against ~600 MFLOP for
 *  `EkfUpdateJoseph`, which also computed `H P` twice.
 *
 *  `blocks` must cover every row of `H` exactly once, in order, and every slot
 *  they name must lie inside `live`. Returns false and leaves `P` and `err`
 *  untouched if `S` is not numerically positive definite; the caller is expected
 *  to fall back to `EkfUpdateJoseph`. */
bool EkfUpdateDowndate(MatX &P, const MatX &H, const VecX &inn,
                       const VecX &diagR, const std::vector<MeasBlock> &blocks,
                       const StateRuns &live, VecX &err);

/** `M = H P`, using the block sparsity of `H` on its rows and the occupied
 *  extent `live` on its columns. Equal to the dense `H * P` in full: the columns
 *  outside `live` are set to zero rather than left alone, which is what they are
 *  in the product anyway. Exposed for the test. */
void MeasurementTimesCov(const MatX &H, const MatX &P,
                         const std::vector<MeasBlock> &blocks,
                         const StateRuns &live, MatX &M);

} // namespace xivo
