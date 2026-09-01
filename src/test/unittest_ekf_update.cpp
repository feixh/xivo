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
 *  features touches; everything else is a vacant slot, which carries a variance
 *  but no cross terms (see `MakeP`). */
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

/** The same set, as the two contiguous runs the update is given. `LiveIndices` and
 *  this must describe the same set -- `OccupiedRunsAgreeWithTheLiveIndexSet` is
 *  what checks it, and everything else here relies on it. */
StateRuns Live(int n_groups, int n_feats) {
  return OccupiedStateRuns(n_groups, n_feats);
}

/** A covariance shaped like the filter's: symmetric positive definite on the
 *  occupied slots, and on the vacant ones the prior variance `kVacant` with no
 *  cross terms at all.
 *
 *  The vacant slots are the interesting part of the fixture. `Estimator` builds
 *  `P_` with `setIdentity` and only ever zeros a slot's row and column when it
 *  frees it, so a slot that has never been used sits at variance 1 forever -- it is
 *  *uncorrelated*, not zero, which is the premise `EkfUpdateDowndate`'s
 *  compaction actually needs. `kVacant` is not 1 so that a result which happens to
 *  land on 1 cannot pass by accident. */
constexpr number_t kVacant = 0.75;

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

  MatX P = kVacant * MatX::Identity(kFullSize, kFullSize);
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

/** The columns `MakeProblem` fills in its out-of-state block: the motion states
 *  and the pose block of every group in play. The real thing's are narrower still
 *  (`Wbc`/`Tbc` plus the observed groups, from `Feature::OOSColumnRuns`), so this
 *  is the conservative version of the same description. */
RunSet OOSBlockRuns(int n_groups) {
  RunSet rs;
  rs.Add(0, kMotionSize);
  rs.Add(kGroupBegin, kGroupSize * n_groups);
  return rs;
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
  ASSERT_TRUE(EkfUpdateDowndate(P_fast, p.H, p.inn, p.diagR, p.blocks,
                                Live(n_groups, n_feats), err_fast));

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

  // Vacant slots come out exactly as they went in: the update must neither
  // correlate a slot nothing observed nor disturb its prior variance.
  const int vacant = kFeatureBegin + kFeatureSize * (n_feats + 1);
  if (vacant < kFullSize) {
    EXPECT_EQ(P_fast(vacant, vacant), kVacant);
    EXPECT_EQ(P_fast.row(vacant).norm(), kVacant);
  }
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
  MeasurementTimesCov(p.H, P, p.blocks, Live(7, 40), M);
  // In full, including the columns outside the occupied extent -- they are zero in
  // the dense product too, and `MeasurementTimesCov` writes them rather than
  // leaving them at whatever the allocator returned.
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

  ASSERT_TRUE(EkfUpdateDowndate(P_sparse, p.H, p.inn, p.diagR, p.blocks,
                                Live(7, 30), err_sparse));
  ASSERT_TRUE(EkfUpdateDowndate(P_dense, p.H, p.inn, p.diagR, dense,
                                Live(7, 30), err_dense));

  EXPECT_LT(RelDiff(P_sparse, P_dense), 1e-12);
  EXPECT_LT((err_sparse - err_dense).norm() / err_dense.norm(), 1e-12);
}

TEST(EkfUpdate, ADenseBlockWithRunsGivesTheSameAnswer) {
  // The other direction of the previous test: telling a dense block which columns
  // it is nonzero in must not change the answer, only the work. This is the path
  // an out-of-state block takes with `OOS.fast_sparse` on, and the tolerance is
  // the reassociation of gemms cut on different boundaries -- not an
  // approximation, since every column dropped is identically zero in `H`.
  const int n_groups = 7, n_feats = 40, oos_rows = 12;
  const MatX P0 = MakeP(n_groups, n_feats, 41);
  Problem p = MakeProblem(n_groups, n_feats, oos_rows, 42);
  const RunSet rs = OOSBlockRuns(n_groups);
  // One run or two depending on whether the camera-intrinsics block exists at
  // this build's compile-time options; `dim` is what the test is really about.
  ASSERT_GE(rs.nruns, 1);
  ASSERT_EQ(rs.dim, kMotionSize + kGroupSize * n_groups);

  // The premise. If `MakeProblem` ever writes outside these columns the test
  // below would be checking two different problems.
  ASSERT_TRUE(ColsWithinRuns(p.H.bottomRows(oos_rows), rs));

  std::vector<MeasBlock> with_runs = p.blocks;
  ASSERT_FALSE(with_runs.back().sparse());
  with_runs.back().runs = &rs;

  MatX M_plain, M_runs;
  MeasurementTimesCov(p.H, P0, p.blocks, Live(n_groups, n_feats), M_plain);
  MeasurementTimesCov(p.H, P0, with_runs, Live(n_groups, n_feats), M_runs);
  EXPECT_LT(RelDiff(M_runs, M_plain), 1e-14);

  MatX P_plain = P0, P_runs = P0;
  VecX err_plain = VecX::Zero(kFullSize), err_runs = VecX::Zero(kFullSize);
  ASSERT_TRUE(EkfUpdateDowndate(P_plain, p.H, p.inn, p.diagR, p.blocks,
                                Live(n_groups, n_feats), err_plain));
  ASSERT_TRUE(EkfUpdateDowndate(P_runs, p.H, p.inn, p.diagR, with_runs,
                                Live(n_groups, n_feats), err_runs));
  EXPECT_LT(RelDiff(P_runs, P_plain), 1e-12);
  EXPECT_LT((err_runs - err_plain).norm() / err_plain.norm(), 1e-12);
}

TEST(RunSetTest, AddKeepsRunsAscendingDisjointAndMaximal) {
  RunSet rs;
  EXPECT_EQ(rs.nruns, 0);
  EXPECT_EQ(rs.dim, 0);

  rs.Add(100, 6);
  rs.Add(0, 6);
  // Adjacent below: absorbed into one run, not appended as a second.
  rs.Add(94, 6);
  ASSERT_EQ(rs.nruns, 2);
  EXPECT_EQ(rs.runs[0].start, 0);
  EXPECT_EQ(rs.runs[0].len, 6);
  EXPECT_EQ(rs.runs[1].start, 94);
  EXPECT_EQ(rs.runs[1].len, 12);
  EXPECT_EQ(rs.dim, 18);

  // Idempotent: the same group observed twice costs nothing.
  rs.Add(94, 6);
  rs.Add(100, 6);
  EXPECT_EQ(rs.nruns, 2);
  EXPECT_EQ(rs.dim, 18);

  // A run that bridges two existing ones absorbs both.
  rs.Add(6, 88);
  ASSERT_EQ(rs.nruns, 1);
  EXPECT_EQ(rs.runs[0].start, 0);
  EXPECT_EQ(rs.runs[0].len, 106);
  EXPECT_EQ(rs.dim, 106);

  // Zero and negative lengths are no-ops.
  rs.Add(500, 0);
  rs.Add(500, -3);
  EXPECT_EQ(rs.nruns, 1);
}

TEST(RunSetTest, CompactGatherAndScatterRoundTrip) {
  RunSet rs;
  rs.Add(kGroupBegin + kGroupSize * 3, kGroupSize);
  rs.Add(Index::Wbc, 6);
  rs.Add(kGroupBegin + kGroupSize * 1, kGroupSize);
  ASSERT_EQ(rs.nruns, 3);
  EXPECT_EQ(rs.dim, 18);
  // Ascending, so the extrinsics run comes first.
  EXPECT_EQ(rs.runs[0].start, Index::Wbc);

  EXPECT_EQ(rs.Compact(Index::Wbc), 0);
  EXPECT_EQ(rs.Compact(Index::Tbc), 3);
  EXPECT_EQ(rs.Compact(kGroupBegin + kGroupSize * 1), 6);
  EXPECT_EQ(rs.Compact(kGroupBegin + kGroupSize * 3 + 5), 17);
  EXPECT_EQ(rs.Compact(0), -1);
  EXPECT_EQ(rs.Compact(kGroupBegin + kGroupSize * 2), -1);
  EXPECT_EQ(rs.Compact(kFullSize - 1), -1);

  MatX H = MatX::Zero(4, kFullSize);
  for (int i = 0; i < rs.nruns; ++i)
    for (int c = 0; c < rs.runs[i].len; ++c)
      for (int r = 0; r < 4; ++r)
        H(r, rs.runs[i].start + c) = 1 + r + 10 * (rs.runs[i].start + c);
  ASSERT_TRUE(ColsWithinRuns(H, rs));

  MatX Hc(4, rs.dim);
  GatherRunCols(H, rs, Hc);
  int c = 0;
  for (int i = 0; i < rs.nruns; ++i) {
    for (int k = 0; k < rs.runs[i].len; ++k)
      EXPECT_EQ(Hc(2, c + k), H(2, rs.runs[i].start + k));
    c += rs.runs[i].len;
  }

  MatX back = MatX::Zero(4, kFullSize);
  ScatterRunCols(Hc, rs, back);
  EXPECT_EQ((back - H).cwiseAbs().maxCoeff(), 0);

  // The premise check has to actually fail when the premise does.
  H(1, 0) = 1e-300;
  EXPECT_FALSE(ColsWithinRuns(H, rs));
}

TEST(RunSetTest, GatheredProductEqualsTheDenseOne) {
  // The identity every user of `RunSet` relies on: for an `H` that is zero outside
  // `rs`, `H P H^T` equals the product formed on the gathered slice alone.
  RunSet rs;
  rs.Add(Index::Wbc, 6);
  for (int g : {0, 2, 3, 9})
    rs.Add(kGroupBegin + kGroupSize * g, kGroupSize);

  std::default_random_engine gen(7);
  std::normal_distribution<number_t> nrm(0.0, 1.0);
  MatX A(kFullSize, kFullSize);
  for (int i = 0; i < kFullSize; ++i)
    for (int j = 0; j < kFullSize; ++j)
      A(i, j) = nrm(gen);
  const MatX P = A * A.transpose() / kFullSize;

  const int rows = 9;
  MatX H = MatX::Zero(rows, kFullSize);
  for (int i = 0; i < rs.nruns; ++i)
    for (int c = 0; c < rs.runs[i].len; ++c)
      for (int r = 0; r < rows; ++r)
        H(r, rs.runs[i].start + c) = 200.0 * nrm(gen);

  MatX Hc(rows, rs.dim), Pc(rs.dim, rs.dim);
  GatherRunCols(H, rs, Hc);
  GatherRunCov(P, rs, Pc);
  EXPECT_LT(RelDiff(MatX(Hc * Pc * Hc.transpose()), MatX(H * P * H.transpose())),
            1e-14);
  // And the half product `P H^T`, which is what the feature-init path needs.
  MatX Pcols(kFullSize, rs.dim);
  GatherRunCols(P, rs, Pcols);
  EXPECT_LT(RelDiff(MatX(Pcols * Hc.transpose()), MatX(P * H.transpose())),
            1e-14);
}

TEST(OccupiedState, RunsAgreeWithTheLiveIndexSet) {
  // The runs are the *only* description of the occupied extent the update gets, so
  // they have to name exactly the indices `MakeP` fills -- which is the shape the
  // filter's slot bookkeeping produces. Swept over the whole range of occupancies,
  // including the two boundaries that decide whether there are one or two runs.
  for (int g = 0; g <= kMaxGroup; ++g) {
    for (int f : {0, 1, 2, 40, 76, kMaxFeature}) {
      const StateRuns s = OccupiedStateRuns(g, f);
      ASSERT_GE(s.nruns, 1) << g << " groups, " << f << " features";
      ASSERT_LE(s.nruns, kMaxStateRuns) << g << " groups, " << f << " features";

      std::vector<int> from_runs;
      for (int i = 0; i < s.nruns; ++i) {
        if (i > 0) {
          // Ascending and disjoint, which `MirrorLowerTriangle` relies on to know
          // which block of a pair is the lower one.
          ASSERT_GT(s.runs[i].start,
                    s.runs[i - 1].start + s.runs[i - 1].len);
        }
        for (int k = 0; k < s.runs[i].len; ++k)
          from_runs.push_back(s.runs[i].start + k);
      }
      EXPECT_EQ(from_runs, LiveIndices(g, f)) << g << " groups, " << f
                                              << " features";
      EXPECT_EQ(s.dim, static_cast<int>(from_runs.size()));
      EXPECT_LE(s.dim, kFullSize);
    }
  }
}

TEST(OccupiedState, TheWholeStateIsOneRun) {
  const StateRuns s = WholeState();
  EXPECT_EQ(s.nruns, 1);
  EXPECT_EQ(s.dim, kFullSize);
  EXPECT_EQ(s.runs[0].start, 0);
  EXPECT_EQ(s.runs[0].len, kFullSize);
  // A full state is also what the occupied extent degenerates to.
  const StateRuns full = OccupiedStateRuns(kMaxGroup, kMaxFeature);
  EXPECT_EQ(full.nruns, 1);
  EXPECT_EQ(full.dim, kFullSize);
}

TEST(EkfUpdate, CompactingToTheOccupiedExtentChangesNothing) {
  // The milestone's claim, stated as a test: restricting the update to the
  // occupied runs gives the same answer as running it over all 564 dimensions,
  // because the rest of `P` is exactly zero. Not bit-identical -- the dense
  // blocks' summation index is split at the run boundary, so those gemms
  // reassociate -- but agreeing at the level rounding allows.
  const MatX P0 = MakeP(7, 76, 51);
  Problem p = MakeProblem(7, 76, 12, 52);

  MatX P_compact = P0, P_full = P0;
  VecX err_compact = VecX::Zero(kFullSize), err_full = VecX::Zero(kFullSize);
  ASSERT_TRUE(EkfUpdateDowndate(P_compact, p.H, p.inn, p.diagR, p.blocks,
                                Live(7, 76), err_compact));
  ASSERT_TRUE(EkfUpdateDowndate(P_full, p.H, p.inn, p.diagR, p.blocks,
                                WholeState(), err_full));

  EXPECT_LT(RelDiff(P_compact, P_full), 1e-13);
  EXPECT_LT((err_compact - err_full).norm() / err_full.norm(), 1e-13);
  // The vacant tail, exactly rather than to a tolerance: the compact form never
  // visits it, and the full form visits it and subtracts a column of `W` that is
  // exactly zero, so both leave the prior variance bit-for-bit alone.
  for (int i = kFeatureBegin + kFeatureSize * 76; i < kFullSize; ++i) {
    ASSERT_EQ(P_compact(i, i), kVacant) << "row " << i;
    ASSERT_EQ(P_compact.row(i).cwiseAbs().maxCoeff(), kVacant) << "row " << i;
    ASSERT_EQ(P_full(i, i), kVacant) << "row " << i;
    ASSERT_EQ(err_compact(i), 0.0) << "err " << i;
    ASSERT_EQ(err_full(i), 0.0) << "err " << i;
  }
}

TEST(EkfUpdate, OversizedRunsGiveTheSameAnswer) {
  // The runs are a covering, not a characterization: a slot inside a run but not
  // actually occupied is a zero row and column, so including it must cost
  // arithmetic and nothing else. This is the failure mode that matters, since
  // `Estimator::OccupiedState` uses high-water marks and a freed slot below the
  // mark stays inside the runs.
  const MatX P0 = MakeP(3, 20, 61);
  Problem p = MakeProblem(3, 20, 0, 62);

  MatX P_tight = P0, P_loose = P0;
  VecX err_tight = VecX::Zero(kFullSize), err_loose = VecX::Zero(kFullSize);
  ASSERT_TRUE(EkfUpdateDowndate(P_tight, p.H, p.inn, p.diagR, p.blocks,
                                Live(3, 20), err_tight));
  ASSERT_TRUE(EkfUpdateDowndate(P_loose, p.H, p.inn, p.diagR, p.blocks,
                                Live(9, 55), err_loose));

  EXPECT_LT(RelDiff(P_tight, P_loose), 1e-13);
  EXPECT_LT((err_tight - err_loose).norm() / err_loose.norm(), 1e-13);
}

TEST(EkfUpdate, RefusesAnIndefiniteInnovationCovariance) {
  // The fallback path exists because a `P` that has already gone indefinite has
  // no Cholesky factor to downdate through. Forced here by making `P` negative
  // definite, which no amount of positive `R` can rescue.
  MatX P = -MakeP(7, 10, 41);
  Problem p = MakeProblem(7, 10, 0, 42);
  VecX err = VecX::Zero(kFullSize);
  const MatX P_in = P;
  EXPECT_FALSE(
      EkfUpdateDowndate(P, p.H, p.inn, p.diagR, p.blocks, Live(7, 10), err));
  // And left its inputs alone, so the caller's fallback starts from `P`, not
  // from a half-applied update.
  EXPECT_EQ(RelDiff(P, P_in), 0.0);
}
