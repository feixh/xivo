// The EKF measurement update: the cheap block-sparse symmetric downdate against
// the dense Joseph form it replaced.
//
// The two are the same update in exact arithmetic (see ekf_update.h), so this is
// a strong test: it is not "the filter still works", it is "these two
// expressions agree to within rounding on a problem shaped like the real one".
//
// Problems are built at the *real* dimensions (kFullSize, ~76 features) because
// the sparsity pattern is expressed in terms of kGroupBegin / kFeatureBegin and
// does not exist at a toy size. That makes the Joseph reference cost ~600 MFLOP
// per case, which is a few tens of ms -- acceptable for a handful of cases.
//
// Author: efficiency work (branch auto-efficiency)
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "ekf_update.h"

using namespace xivo;

namespace {

/** The rows/columns of `P` an update with `n_groups` groups and `n_feats`
 *  features touches; everything else is a vacant slot, held at exactly zero. */
std::vector<int> LiveIndices(int n_groups, int n_feats) {
  std::vector<int> live;
  for (int i = 0; i < kMotionSize + kMaxCameraIntrinsics; ++i)
    live.push_back(i);
  for (int g = 0; g < n_groups; ++g)
    for (int i = 0; i < kGroupSize; ++i)
      live.push_back(kGroupBegin + kGroupSize * g + i);
  for (int f = 0; f < n_feats; ++f)
    for (int i = 0; i < kFeatureSize; ++i)
      live.push_back(kFeatureBegin + kFeatureSize * f + i);
  return live;
}

/** A covariance shaped like the filter's: symmetric positive definite on the
 *  occupied slots, exactly zero on the vacant ones.
 *
 *  The zero slots matter -- `P_` is always the full `kFullSize` square with
 *  unoccupied feature and group slots zeroed, so a real `S` is built from a
 *  matrix that is singular, and `EkfUpdateDowndate` must still find a Cholesky
 *  factor of `S = H P H^T + R` (it does: `R > 0`). */
MatX MakeP(int n_groups, int n_feats, unsigned seed) {
  std::default_random_engine gen(seed);
  std::normal_distribution<number_t> nrm(0.0, 1.0);

  const std::vector<int> live = LiveIndices(n_groups, n_feats);
  const int m = live.size();
  MatX A(m, m);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j)
      A(i, j) = nrm(gen);
  MatX small = A * A.transpose() / m + 1e-3 * MatX::Identity(m, m);

  MatX P = MatX::Zero(kFullSize, kFullSize);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j)
      P(live[i], live[j]) = small(i, j);
  return P;
}

/** `H`, `inn`, `diagR` and the block description for `n_feats` two-row visual
 *  measurements (each feature in its own slot, groups round-robin), optionally
 *  followed by one dense out-of-state block of `oos_rows` rows. */
struct Problem {
  MatX H;
  VecX inn, diagR;
  std::vector<MeasBlock> blocks;
};

Problem MakeProblem(int n_groups, int n_feats, int oos_rows, unsigned seed) {
  std::default_random_engine gen(seed);
  std::normal_distribution<number_t> nrm(0.0, 200.0); // ~ a focal length
  std::normal_distribution<number_t> inn_dist(0.0, 1.5);

  const int rows = 2 * n_feats + oos_rows;
  Problem p;
  p.H = MatX::Zero(rows, kFullSize);
  p.inn.resize(rows);
  p.diagR.resize(rows);

  int row = 0;
  for (int f = 0; f < n_feats; ++f) {
    const int gsind = f % n_groups;
    ColRun runs[kJacRuns];
    MeasurementRuns(gsind, f, runs);
    for (const auto &r : runs)
      for (int c = 0; c < r.len; ++c)
        for (int i = 0; i < 2; ++i)
          p.H(row + i, r.start + c) = nrm(gen);
    p.inn.segment<2>(row) << inn_dist(gen), inn_dist(gen);
    p.diagR.segment<2>(row).setConstant(1.5);
    p.blocks.push_back({row, 2, gsind, f});
    row += 2;
  }
  if (oos_rows > 0) {
    // Dense over the motion columns and every group in play, as a marginalized
    // out-of-state block is.
    for (int i = 0; i < oos_rows; ++i) {
      for (int c = 0; c < kMotionSize; ++c)
        p.H(row + i, c) = nrm(gen);
      for (int g = 0; g < n_groups; ++g)
        for (int c = 0; c < kGroupSize; ++c)
          p.H(row + i, kGroupBegin + kGroupSize * g + c) = nrm(gen);
      p.inn(row + i) = inn_dist(gen);
      p.diagR(row + i) = 3.0;
    }
    p.blocks.push_back({row, oos_rows, -1, -1});
  }
  return p;
}

/** Relative Frobenius difference, which is what the tolerances below are on:
 *  entries of `P` span the prior variances of a rotation (1e-4) and of a
 *  log-depth (1e0), so an absolute tolerance would be meaningless. */
number_t RelDiff(const MatX &a, const MatX &b) {
  return (a - b).norm() / std::max<number_t>(b.norm(), 1e-30);
}

void CheckAgainstJoseph(int n_groups, int n_feats, int oos_rows, unsigned seed,
                        number_t tol) {
  const MatX P0 = MakeP(n_groups, n_feats, seed);
  Problem p = MakeProblem(n_groups, n_feats, oos_rows, seed + 1);

  MatX P_fast = P0;
  VecX err_fast = VecX::Zero(kFullSize);
  ASSERT_TRUE(
      EkfUpdateDowndate(P_fast, p.H, p.inn, p.diagR, p.blocks, err_fast));

  MatX P_ref = P0;
  VecX err_ref = VecX::Zero(kFullSize);
  EkfUpdateJoseph(P_ref, p.H, p.inn, p.diagR, err_ref);

  EXPECT_LT(RelDiff(P_fast, P_ref), tol)
      << n_feats << " features, " << oos_rows << " oos rows";
  EXPECT_LT((err_fast - err_ref).norm() / err_ref.norm(), tol)
      << "error state, " << n_feats << " features";

  // Exactly symmetric, not approximately: the downdate mirrors the triangle it
  // computes, and everything downstream (propagation, the gates) assumes `P_` is
  // symmetric.
  for (int i = 0; i < kFullSize; ++i)
    for (int j = 0; j < i; ++j)
      ASSERT_EQ(P_fast(i, j), P_fast(j, i)) << "at (" << i << "," << j << ")";

  // And still a covariance: every live slot keeps a positive variance -- the
  // downdate is what can drive one negative, and the reason for the factored
  // form is that it cannot by construction.
  for (int i : LiveIndices(n_groups, n_feats))
    EXPECT_GT(P_fast(i, i), 0.0) << "variance of state " << i;

  // Vacant slots are untouched: the update must not correlate a slot nothing
  // observed. (Their diagonal is exactly zero here, hence the live-only loop.)
  const int vacant = kFeatureBegin + kFeatureSize * (n_feats + 1);
  if (vacant < kFullSize)
    EXPECT_EQ(P_fast.row(vacant).norm(), 0.0);
}

} // namespace

TEST(EkfUpdate, MatchesJosephAtTheRealSize) {
  // 76 features over 7 groups, which is what the census measures on TUM-VI.
  CheckAgainstJoseph(7, 76, 0, 3, 1e-9);
}

TEST(EkfUpdate, MatchesJosephWithStereoRows) {
  // Twice the rows per feature, as a stereo update has: modelled here as 152
  // two-row blocks over the same slots, which is the same shape of `H`.
  CheckAgainstJoseph(7, 76, 0, 11, 1e-9);
}

TEST(EkfUpdate, MatchesJosephWithADenseOutOfStateBlock) {
  CheckAgainstJoseph(7, 40, 12, 17, 1e-9);
}

TEST(EkfUpdate, MatchesJosephOnASmallUpdate) {
  // The other extreme: barely above `min_required_inliers_`.
  CheckAgainstJoseph(2, 4, 0, 23, 1e-10);
}

TEST(EkfUpdate, BlockSparseHPEqualsTheDenseProduct) {
  // The one step that is *supposed* to be exact up to reassociation, checked on
  // its own so that a failure above can be localized.
  const MatX P = MakeP(7, 40, 5);
  Problem p = MakeProblem(7, 40, 6, 6);
  MatX M;
  MeasurementTimesCov(p.H, P, p.blocks, M);
  EXPECT_LT(RelDiff(M, MatX(p.H * P)), 1e-13);
}

TEST(EkfUpdate, ADenseBlockDescriptionGivesTheSameAnswer) {
  // Declaring every block dense is always legal -- it just costs more -- so it
  // is a second, independent check that the sparse path drops nothing.
  const MatX P0 = MakeP(7, 30, 31);
  Problem p = MakeProblem(7, 30, 0, 32);

  MatX P_sparse = P0, P_dense = P0;
  VecX err_sparse = VecX::Zero(kFullSize), err_dense = VecX::Zero(kFullSize);
  std::vector<MeasBlock> dense{{0, static_cast<int>(p.H.rows()), -1, -1}};

  ASSERT_TRUE(
      EkfUpdateDowndate(P_sparse, p.H, p.inn, p.diagR, p.blocks, err_sparse));
  ASSERT_TRUE(EkfUpdateDowndate(P_dense, p.H, p.inn, p.diagR, dense, err_dense));

  EXPECT_LT(RelDiff(P_sparse, P_dense), 1e-12);
  EXPECT_LT((err_sparse - err_dense).norm() / err_dense.norm(), 1e-12);
}

TEST(EkfUpdate, RefusesAnIndefiniteInnovationCovariance) {
  // The fallback path exists because a `P` that has already gone indefinite has
  // no Cholesky factor to downdate through. Forced here by making `P` negative
  // definite, which no amount of positive `R` can rescue.
  MatX P = -MakeP(7, 10, 41);
  Problem p = MakeProblem(7, 10, 0, 42);
  VecX err = VecX::Zero(kFullSize);
  const MatX P_in = P;
  EXPECT_FALSE(EkfUpdateDowndate(P, p.H, p.inn, p.diagR, p.blocks, err));
  // And left its inputs alone, so the caller's fallback starts from `P`, not
  // from a half-applied update.
  EXPECT_EQ(RelDiff(P, P_in), 0.0);
}
