// Unit tests for the out-of-state (MSCKF) measurement model:
//   * the marginalization of the 3D point (left-nullspace projection),
//   * the multi-view depth refinement used to triangulate an OOS feature,
//   * observation selection / thinning,
//   * and the copy of the in-state Jacobian into the stacked H (which had no
//     test, and was wrong).
// Author: generated for the OOS update work (branch auto-oos)
#include <gtest/gtest.h>

#define private public

#include "alias.h"
#include "camera_manager.h"
#include "feature.h"
#include "graph.h"
#include "group.h"
#include "helpers.h"
#include "mm.h"

using namespace Eigen;
using namespace xivo;

namespace {

// A short, gently curving trajectory looking at a point in front of it. Enough
// parallax to triangulate, small enough rotation that the linearization is
// accurate.
SE3 PoseAt(int i) {
  Vec3 W(0.004 * i, -0.006 * i, 0.002 * i);
  Vec3 T(0.09 * i, 0.02 * i, -0.01 * i);
  return SE3{SO3::exp(W), T};
}

const SE3 &Gbc() {
  static SE3 gbc{SO3::exp(Vec3{0.02, -0.03, 0.01}), Vec3{0.05, -0.02, 0.01}};
  return gbc;
}

const Vec3 &TruePoint() {
  static Vec3 Xs{0.35, -0.15, 3.4};
  return Xs;
}

Vec2 ProjectFrom(const SE3 &gsb, const Vec3 &Xs) {
  Vec3 Xc = (gsb * Gbc()).inverse() * Xs;
  return Camera::instance()->Project(project(Xc));
}

} // namespace

class OOSUpdateTest : public ::testing::Test {
protected:
  void SetUp() override {
    MemoryManager::Create(256, 128);
    auto cfg = LoadJson("src/test/camera_configs.json");
    Camera::Create(cfg["perfect_pinhole"]);

    options_.min_observations = 2;
    options_.max_observations = kMaxGroup;
    options_.max_iters = 10;
    options_.Rtri = 1.0;
    options_.max_mean_reproj_err = 1.5;
    options_.zmin = 0.05;
    options_.zmax = 50.0;

    BuildTrack(5);
  }

  void TearDown() override {
    for (auto g : groups_) {
      Group::Destroy(g);
    }
    groups_.clear();
    obs_.clear();
    if (f_) {
      Feature::Destroy(f_);
      f_ = nullptr;
    }
  }

  /** `n` groups, all in the state, all observing `TruePoint()` exactly. The
   *  feature is anchored at the first group and its state is set so that its 3D
   *  point is exactly `TruePoint()`. */
  void BuildTrack(int n) {
    for (int i = 0; i < n; ++i) {
      SE3 gsb = PoseAt(i);
      GroupPtr g = Group::Create(gsb.so3(), gsb.translation());
      g->SetSind(i);
      g->SetStatus(GroupStatus::INSTATE);
      groups_.push_back(g);
      obs_.push_back(Observation{g, ProjectFrom(gsb, TruePoint())});
    }

    f_ = Feature::Create(obs_[0].xp(0), obs_[0].xp(1));
    f_->ref_ = groups_[0];
    f_->SetSind(0);
    f_->SetStatus(FeatureStatus::READY);
    SetExactPoint();
  }

  void SetExactPoint() {
    Vec3 Xc = (groups_[0]->gsb() * Gbc()).inverse() * TruePoint();
    f_->x_ << Xc(0) / Xc(2), Xc(1) / Xc(2), std::log(Xc(2));
  }

  /** Fills the shared scratch `{Hf,Hx,inn}` without marginalizing, and stashes copies of the
   *  filled rows. A copy of `Hf` is parked in the (otherwise unused) feature
   *  columns of `Hx`, so that after marginalization those columns hold
   *  `A' * Hf`, which must vanish. */
  int FillJacobians() {
    f_->cache_.Xs = f_->Xs(Gbc());
    int rows = 0;
    for (const auto &obs : obs_) {
      rows += f_->ComputeOOSJacobianInternal(obs, Gbc().so3().matrix(),
                                             Gbc().translation(), rows,
                                             options_);
    }
    // The un-marginalized stack lives in the shared scratch now; `f_->oos_`
    // holds only what `MarginalizeOOSPoint` writes back.
    OOSJacobian &s = Feature::oos_scratch();
    Hf0_ = s.Hf.topRows(rows);
    s.Hx.block(0, kFeatureBegin, rows, 3) = Hf0_;
    Hx0_ = s.Hx.topRows(rows);
    inn0_ = s.inn.head(rows);
    return rows;
  }

  /** Orthogonal projector onto the left nullspace of `Hf0_`. */
  MatX NullspaceProjector() const {
    int rows = Hf0_.rows();
    return MatX::Identity(rows, rows) -
           Hf0_ * (Hf0_.transpose() * Hf0_).inverse() * Hf0_.transpose();
  }

  OOSOptions options_;
  std::vector<GroupPtr> groups_;
  std::vector<Observation> obs_;
  FeaturePtr f_{nullptr};

  MatX Hf0_, Hx0_;
  VecX inn0_;
};

TEST_F(OOSUpdateTest, RowCountIsTwoNMinusThree) {
  int rows = f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                    Gbc().translation(), options_);
  EXPECT_EQ(rows, 2 * static_cast<int>(obs_.size()) - 3);
  EXPECT_EQ(rows, f_->oos_inn_size());
  EXPECT_EQ(f_->oos_num_obs(), static_cast<int>(obs_.size()));
  EXPECT_EQ(f_->Ho().rows(), rows);
  EXPECT_EQ(f_->ro().size(), rows);
}

TEST_F(OOSUpdateTest, TooFewObservationsIsRejected) {
  options_.min_observations = 6;
  EXPECT_EQ(f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                   Gbc().translation(), options_),
            0);

  // Observations from groups that are not in the state do not count.
  options_.min_observations = 2;
  for (size_t i = 1; i < groups_.size(); ++i) {
    groups_[i]->SetStatus(GroupStatus::FLOATING);
  }
  EXPECT_EQ(f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                   Gbc().translation(), options_),
            0);
}

TEST_F(OOSUpdateTest, MarginalizationAnnihilatesPointJacobian) {
  int rows = FillJacobians();
  int out = f_->MarginalizeOOSPoint(rows);
  ASSERT_EQ(out, rows - 3);

  // These columns held a copy of Hf, so they now hold A' * Hf.
  MatX AtHf = Feature::oos_result_Hx(out).block(0, kFeatureBegin, out, 3);
  EXPECT_LT(AtHf.norm(), 1e-9 * Hf0_.norm());
}

TEST_F(OOSUpdateTest, MarginalizationBasisIsOrthonormal) {
  int rows = FillJacobians();
  int out = f_->MarginalizeOOSPoint(rows);
  ASSERT_EQ(out, rows - 3);

  MatX Hx1 = MatX(Feature::oos_result_Hx(out));
  VecX inn1 = Feature::oos_result().inn.head(out);
  MatX Pi = NullspaceProjector();

  // For an orthonormal basis A of the left nullspace, A * A' is exactly the
  // projector Pi, so the sufficient statistics of the projected measurement are
  // invariant to the choice of basis and must match those of the unprojected
  // one squeezed through Pi. This is what breaks if A's columns are merely
  // independent (e.g. a LU kernel): the update assumes A' R A = sigma^2 I.
  MatX lhs = Hx1.transpose() * Hx1;
  MatX rhs = Hx0_.transpose() * Pi * Hx0_;
  EXPECT_LT((lhs - rhs).norm(), 1e-8 * rhs.norm());

  VecX lhs_b = Hx1.transpose() * inn1;
  VecX rhs_b = Hx0_.transpose() * Pi * inn0_;
  EXPECT_LT((lhs_b - rhs_b).norm(), 1e-8 * std::max<number_t>(rhs_b.norm(), 1));

  EXPECT_NEAR(inn1.squaredNorm(), inn0_.dot(Pi * inn0_),
              1e-8 * std::max<number_t>(inn0_.squaredNorm(), 1));
}

TEST_F(OOSUpdateTest, NoiseFreeResidualIsZero) {
  int rows = f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                    Gbc().translation(), options_);
  ASSERT_GT(rows, 0);
  EXPECT_LT(f_->ro().lpNorm<Infinity>(), 1e-8);
}

TEST_F(OOSUpdateTest, PointErrorIsMarginalizedOut) {
  // A wrong 3D point (here: 10% off in depth) produces a large raw innovation,
  // but the projected one must stay ~0: that is the whole point of the
  // nullspace projection.
  f_->x_(2) += 0.1;
  int rows = FillJacobians();
  number_t raw_norm = inn0_.lpNorm<Infinity>();
  EXPECT_GT(raw_norm, 1.0); // pixels: a clearly visible reprojection error

  int out = f_->MarginalizeOOSPoint(rows);
  ASSERT_EQ(out, rows - 3);
  // Only second-order terms survive.
  EXPECT_LT(Feature::oos_result().inn.head(out).lpNorm<Infinity>(), 1e-2 * raw_norm);
}

TEST_F(OOSUpdateTest, RefineRecoversPoint) {
  // Start 35% off in depth and slightly off in the bearing.
  f_->x_(0) += 0.02;
  f_->x_(1) -= 0.02;
  f_->x_(2) += 0.3;

  ASSERT_TRUE(f_->RefineOOSDepth(Gbc(), obs_, options_));
  Vec3 Xs = f_->Xs(Gbc());
  EXPECT_NEAR(Xs(0), TruePoint()(0), 1e-4);
  EXPECT_NEAR(Xs(1), TruePoint()(1), 1e-4);
  EXPECT_NEAR(Xs(2), TruePoint()(2), 1e-4);
  EXPECT_LT(f_->oos_mean_reproj_err(), 1e-4);

  // ... and the residual of the marginalized measurement is then ~0.
  ASSERT_GT(f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                   Gbc().translation(), options_),
            0);
  EXPECT_LT(f_->ro().lpNorm<Infinity>(), 1e-4);
}

TEST_F(OOSUpdateTest, RefineRejectsInconsistentTrack) {
  // One badly mismatched observation: no 3D point explains the track, and the
  // feature must not be handed to the filter.
  obs_[3].xp += Vec2{25.0, -18.0};
  EXPECT_FALSE(f_->RefineOOSDepth(Gbc(), obs_, options_));
  EXPECT_GT(f_->oos_mean_reproj_err(), options_.max_mean_reproj_err);
}

TEST_F(OOSUpdateTest, RefineRejectsOutOfRangeDepth) {
  options_.zmax = 2.0; // the true depth is ~3.4 m
  EXPECT_FALSE(f_->RefineOOSDepth(Gbc(), obs_, options_));
}

TEST_F(OOSUpdateTest, RefineNeedsTwoViews) {
  std::vector<Observation> one{obs_[0]};
  EXPECT_FALSE(f_->RefineOOSDepth(Gbc(), one, options_));
}

TEST_F(OOSUpdateTest, SelectionSkipsFloatingGroupsAndSortsByAge) {
  groups_[1]->SetStatus(GroupStatus::FLOATING);
  std::vector<Observation> shuffled{obs_[3], obs_[0], obs_[4], obs_[1],
                                    obs_[2]};
  auto sel = f_->SelectOOSObservations(shuffled, options_);
  ASSERT_EQ(sel.size(), 4u);
  for (size_t i = 1; i < sel.size(); ++i) {
    EXPECT_LT(sel[i - 1].g->id(), sel[i].g->id());
  }
  for (const auto &obs : sel) {
    EXPECT_TRUE(obs.g->instate());
  }
}

TEST_F(OOSUpdateTest, SelectionThinsLongTracks) {
  TearDown();
  BuildTrack(12);
  options_.max_observations = 5;

  auto sel = f_->SelectOOSObservations(obs_, options_);
  ASSERT_EQ(sel.size(), 5u);
  // First and last observation kept: that is where the parallax is.
  EXPECT_EQ(sel.front().g->id(), obs_.front().g->id());
  EXPECT_EQ(sel.back().g->id(), obs_.back().g->id());
  for (size_t i = 1; i < sel.size(); ++i) {
    EXPECT_LT(sel[i - 1].g->id(), sel[i].g->id());
  }

  // Selection is idempotent, so the refinement and the Jacobian see the same
  // rows even though each of them selects on its own.
  auto sel2 = f_->SelectOOSObservations(sel, options_);
  ASSERT_EQ(sel2.size(), sel.size());
  for (size_t i = 0; i < sel.size(); ++i) {
    EXPECT_EQ(sel2[i].g->id(), sel[i].g->id());
  }

  int rows = f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                    Gbc().translation(), options_);
  EXPECT_EQ(f_->oos_num_obs(), 5);
  EXPECT_EQ(rows, 2 * 5 - 3);
}

// `OOSOptions::fast_sparse` records which error-state columns the stacked
// Jacobian can be nonzero in and then forms the products over those alone. The
// two forms are the same matrices up to gemm reassociation, so these tests are
// what license the config key.
TEST_F(OOSUpdateTest, ColumnRunsCoverExtrinsicsAndEveryObservedGroup) {
  groups_[0]->SetSind(4);
  groups_[1]->SetSind(5); // adjacent to 4: the two slots must merge into one run
  groups_[2]->SetSind(9);
  groups_[3]->SetSind(9); // a repeat costs nothing
  groups_[4]->SetSind(-1); // not in the state; contributes no columns

  const RunSet rs = Feature::OOSColumnRuns(obs_);
  EXPECT_EQ(rs.nruns, 3);
  EXPECT_EQ(rs.dim, 6 + 12 + 6);
  EXPECT_EQ(rs.runs[0].start, xivo::Index::Wbc);
  EXPECT_EQ(rs.runs[0].len, 6);
  EXPECT_EQ(rs.runs[1].start, kGroupBegin + kGroupSize * 4);
  EXPECT_EQ(rs.runs[1].len, 12);
  EXPECT_EQ(rs.runs[2].start, kGroupBegin + kGroupSize * 9);

  // The anchor group, which `ComputeInitJacobian` also writes, on request.
  const RunSet with_anchor = Feature::OOSColumnRuns(obs_, 7);
  EXPECT_EQ(with_anchor.nruns, 4);
  EXPECT_EQ(with_anchor.dim, rs.dim + 6);
}

TEST_F(OOSUpdateTest, FastSparseOOSJacobianMatchesTheDenseForm) {
  options_.fast_sparse = false;
  ASSERT_GT(f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                   Gbc().translation(), options_), 0);
  const MatX Ho_dense = f_->Ho();
  const VecX ro_dense = f_->ro();
  EXPECT_EQ(f_->oos_runs().nruns, 0) << "no runs are recorded when the key is off";

  options_.fast_sparse = true;
  ASSERT_EQ(f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                   Gbc().translation(), options_),
            Ho_dense.rows());
  const MatX Ho_fast = f_->Ho();
  const VecX ro_fast = f_->ro();

  // Same shape, and nonzero only where the run set says.
  ASSERT_EQ(Ho_fast.rows(), Ho_dense.rows());
  ASSERT_EQ(Ho_fast.cols(), kFullSize);
  EXPECT_GT(f_->oos_runs().nruns, 0);
  EXPECT_TRUE(ColsWithinRuns(Ho_fast, f_->oos_runs()));
  // And the dense form is too -- which is the premise, not a consequence.
  EXPECT_TRUE(ColsWithinRuns(Ho_dense, f_->oos_runs()));

  EXPECT_LT((Ho_fast - Ho_dense).norm() / Ho_dense.norm(), 1e-14);
  EXPECT_LT((ro_fast - ro_dense).norm() / std::max(ro_dense.norm(), 1e-30), 1e-14);
}

TEST_F(OOSUpdateTest, FastSparseLeavesNoStaleScratchBehind) {
  // The trap the restricted clear opens: the scratch is only cleared inside the
  // *current* feature's runs, so a feature whose runs differ from the previous
  // one's would read its leftovers if anything ever looked outside them. Run two
  // features with disjoint group slots back to back and check the second.
  options_.fast_sparse = true;
  ASSERT_GT(f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                   Gbc().translation(), options_), 0);

  for (size_t i = 0; i < groups_.size(); ++i) {
    groups_[i]->SetSind(static_cast<int>(i) + 20);
  }
  ASSERT_GT(f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                   Gbc().translation(), options_), 0);
  EXPECT_TRUE(ColsWithinRuns(f_->Ho(), f_->oos_runs()))
      << "the second feature's Jacobian kept the first one's columns";
}

TEST_F(OOSUpdateTest, FastSparseInitJacobianMatchesTheDenseForm) {
  // `ComputeInitJacobian` is the `consistent_init` path: same stack, but the
  // anchor's and the extrinsics' contribution *through* the point is added back
  // and only the three invertible rows are kept.
  Mat3 Hl_dense, Hl_fast;
  Eigen::Matrix<number_t, 3, kFullSize> Hx_dense, Hx_fast;
  Vec3 res_dense, res_fast;

  const RunSet rs = Feature::OOSColumnRuns(obs_, f_->ref()->sind());
  ASSERT_TRUE(f_->ComputeInitJacobian(obs_, Gbc().so3().matrix(),
                                      Gbc().translation(), options_, &Hl_dense,
                                      &Hx_dense, &res_dense, nullptr));
  ASSERT_TRUE(f_->ComputeInitJacobian(obs_, Gbc().so3().matrix(),
                                      Gbc().translation(), options_, &Hl_fast,
                                      &Hx_fast, &res_fast, &rs));

  EXPECT_TRUE(ColsWithinRuns(Hx_dense, rs));
  EXPECT_TRUE(ColsWithinRuns(Hx_fast, rs));
  EXPECT_LT((Hl_fast - Hl_dense).norm() / Hl_dense.norm(), 1e-14);
  EXPECT_LT((Hx_fast - Hx_dense).norm() / Hx_dense.norm(), 1e-14);
  EXPECT_LT((res_fast - res_dense).norm() /
                std::max(res_dense.norm(), number_t(1e-30)),
            1e-12);
}

// Regression test for the in-state Jacobian copy: every block of J_ must reach
// H. Both group blocks used to be written to `goff`, so the reference group's
// rotation Jacobian was overwritten by its translation Jacobian and the
// translation columns stayed zero.
TEST_F(OOSUpdateTest, FillJacobianBlockCopiesEveryBlock) {
  groups_[0]->SetSind(2);
  f_->SetSind(3);

  // A pattern that makes every column distinguishable.
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < kFullSize; ++j) {
      f_->J_(i, j) = 1 + i * kFullSize + j;
    }
  }

  MatX H(4, kFullSize);
  H.setZero();
  const int offset = 2;
  f_->FillJacobianBlock(H, offset);

  const int goff = kGroupBegin + 6 * groups_[0]->sind();
  const int foff = kFeatureBegin + 3 * f_->sind();
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 3; ++k) {
      EXPECT_EQ(H(offset + i, xivo::Index::Wsb + k), f_->J_(i, xivo::Index::Wsb + k));
      EXPECT_EQ(H(offset + i, xivo::Index::Tsb + k), f_->J_(i, xivo::Index::Tsb + k));
      EXPECT_EQ(H(offset + i, xivo::Index::Wbc + k), f_->J_(i, xivo::Index::Wbc + k));
      EXPECT_EQ(H(offset + i, xivo::Index::Tbc + k), f_->J_(i, xivo::Index::Tbc + k));
      EXPECT_EQ(H(offset + i, goff + k), f_->J_(i, goff + k));
      EXPECT_EQ(H(offset + i, goff + 3 + k), f_->J_(i, goff + 3 + k));
      EXPECT_EQ(H(offset + i, foff + k), f_->J_(i, foff + k));
    }
  }
  // Rows outside the block are untouched.
  EXPECT_EQ(H.row(0).norm(), 0);
}
