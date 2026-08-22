// Unit tests for the *stereo* out-of-state (MSCKF) measurement model: the two
// extra rows a right-camera observation contributes to a dropped track, and
// their effect on the marginalization, the whitening and the depth refinement.
//
// Run from the repository root: the config path below is relative to it.
//
// A separate binary from unitTests_OOSUpdate on purpose: `CameraManager`'s
// registry and `StereoRig` are process-wide singletons that keep whatever was
// installed first, and this test needs the real TUM-VI fisheye pair in slots
// 0/1 while that one installs a perfect pinhole in slot 0.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#define private public

#include "alias.h"
#include "feature.h"
#include "group.h"
#include "helpers.h"
#include "mm.h"
#include "stereo.h"

using namespace xivo;

namespace {

const char *kStereoCfg = "cfg/tumvi_stereo.json";

/** Sideways-drifting, gently rotating trajectory: enough parallax for a
 *  monocular track to triangulate, small enough that the point stays in both
 *  images. `scale` shrinks the translation to make the track (nearly)
 *  degenerate for a monocular reconstruction. */
SE3 PoseAt(int i, number_t scale = 1.0) {
  Vec3 W(0.003 * i, -0.004 * i, 0.002 * i);
  Vec3 T = scale * Vec3(0.07 * i, 0.015 * i, -0.01 * i);
  return SE3{SO3::exp(W), T};
}

const SE3 &Gbc() {
  static SE3 gbc{SO3::exp(Vec3{0.02, -0.03, 0.01}), Vec3{0.05, -0.02, 0.01}};
  return gbc;
}

/** A point a few metres ahead of the first camera, comfortably inside both
 *  fields of view. */
const Vec3 &TruePoint() {
  static Vec3 Xs = PoseAt(0) * Gbc() * Vec3{0.25, -0.1, 3.2};
  return Xs;
}

Vec2 ProjectLeft(const SE3 &gsb, const Vec3 &Xs) {
  Vec3 Xc0 = (gsb * Gbc()).inverse() * Xs;
  return Camera::instance(0)->Project(project(Xc0));
}

Vec2 ProjectRight(const SE3 &gsb, const Vec3 &Xs) {
  Vec3 Xc0 = (gsb * Gbc()).inverse() * Xs;
  Vec3 Xc1 = StereoRig::instance()->ToCam1(Xc0);
  return Camera::instance(1)->Project(project(Xc1));
}

} // namespace

class OOSStereoTest : public ::testing::Test {
protected:
  void SetUp() override {
    MemoryManager::Create(256, 128);
    auto cfg = LoadJson(kStereoCfg);
    ASSERT_FALSE(cfg.isNull())
        << "could not load " << kStereoCfg << "; run tests from the repo root";
    Camera::Create(cfg["camera_cfg"], 0);
    ASSERT_NE(Camera::Create(cfg["camera1_cfg"], 1), nullptr);
    ASSERT_NE(StereoRig::Create(cfg["stereo_cfg"]), nullptr);

    options_.min_observations = 2;
    options_.max_observations = kMaxGroup;
    options_.max_iters = 10;
    options_.Rtri = 1.0;
    options_.max_mean_reproj_err = 1.5;
    options_.zmin = 0.05;
    options_.zmax = 50.0;
    options_.use_stereo = true;
    options_.stereo_R_scale = 1.0;

    BuildTrack(4);
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

  /** `n` in-state groups, all observing `TruePoint()` exactly, in both cameras.
   *  The feature is anchored at the first group and its state is set so that its
   *  3D point is exactly `TruePoint()`. */
  void BuildTrack(int n, number_t scale = 1.0) {
    for (auto g : groups_) {
      Group::Destroy(g);
    }
    groups_.clear();
    obs_.clear();
    if (f_) {
      Feature::Destroy(f_);
      f_ = nullptr;
    }

    for (int i = 0; i < n; ++i) {
      SE3 gsb = PoseAt(i, scale);
      GroupPtr g = Group::Create(gsb.so3(), gsb.translation());
      g->SetSind(i);
      g->SetStatus(GroupStatus::INSTATE);
      groups_.push_back(g);

      Observation obs{g, ProjectLeft(gsb, TruePoint())};
      obs.has_right = true;
      obs.xp_r = ProjectRight(gsb, TruePoint());
      obs_.push_back(obs);
    }

    f_ = Feature::Create(obs_[0].xp(0), obs_[0].xp(1));
    f_->ref_ = groups_[0];
    f_->SetSind(0);
    f_->SetStatus(FeatureStatus::READY);
    SetPoint(1.0);
  }

  /** Puts the feature's state on the true point, with its depth multiplied by
   *  `depth_factor`. */
  void SetPoint(number_t depth_factor) {
    Vec3 Xc = (groups_[0]->gsb() * Gbc()).inverse() * TruePoint();
    f_->x_ << Xc(0) / Xc(2), Xc(1) / Xc(2), std::log(depth_factor * Xc(2));
  }

  void DropRightObservations() {
    for (auto &obs : obs_) {
      obs.has_right = false;
    }
  }

  /** Fills `oos_.{Hf,Hx,inn}` without marginalizing, and stashes copies of the
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
    Hf0_ = f_->oos_.Hf.topRows(rows);
    f_->oos_.Hx.block(0, kFeatureBegin, rows, 3) = Hf0_;
    Hx0_ = f_->oos_.Hx.topRows(rows);
    inn0_ = f_->oos_.inn.head(rows);
    return rows;
  }

  OOSOptions options_;
  std::vector<GroupPtr> groups_;
  std::vector<Observation> obs_;
  FeaturePtr f_{nullptr};

  MatX Hf0_, Hx0_;
  VecX inn0_;
};

// A stereo view contributes 4 rows instead of 2, so an n-view track yields
// `4n - 3` rows -- and a track whose right observations are missing, or whose
// right rows are switched off, is bit-for-bit the monocular measurement.
TEST_F(OOSStereoTest, RowCounts) {
  const int n = obs_.size();
  EXPECT_EQ(FillJacobians(), 4 * n);
  EXPECT_EQ(f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                   Gbc().translation(), options_),
            4 * n - 3);
  EXPECT_EQ(f_->oos_num_obs(), n);
  EXPECT_EQ(f_->oos_num_right_obs(), n);

  MatX Ho_stereo = f_->Ho();

  options_.use_stereo = false;
  EXPECT_EQ(FillJacobians(), 2 * n);
  EXPECT_EQ(f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                   Gbc().translation(), options_),
            2 * n - 3);
  EXPECT_EQ(f_->oos_num_right_obs(), 0);
  MatX Ho_mono_off = f_->Ho();

  options_.use_stereo = true;
  DropRightObservations();
  EXPECT_EQ(f_->ComputeOOSJacobian(obs_, Gbc().so3().matrix(),
                                   Gbc().translation(), options_),
            2 * n - 3);
  EXPECT_EQ(f_->oos_num_right_obs(), 0);
  EXPECT_TRUE(f_->Ho().isApprox(Ho_mono_off))
      << "no right observations must reproduce the monocular measurement";
  EXPECT_GT(Ho_stereo.rows(), Ho_mono_off.rows());
}

// The whole point of the extra rows: they enter the *same* nullspace projection
// as the left ones, because they constrain the same 3D point.
TEST_F(OOSStereoTest, MarginalizationKillsStereoRows) {
  int rows = FillJacobians();
  ASSERT_EQ(rows, 4 * static_cast<int>(obs_.size()));

  int out = f_->MarginalizeOOSPoint(rows);
  EXPECT_EQ(out, rows - 3);

  // Those parked columns now hold A' * Hf, which must be exactly zero (up to
  // the rounding of the products) for the marginalization to have eliminated
  // the point.
  MatX AtHf = f_->oos_.Hx.block(0, kFeatureBegin, out, 3);
  EXPECT_LT(AtHf.cwiseAbs().maxCoeff(), 1e-9 * Hf0_.cwiseAbs().maxCoeff())
      << "A' * Hf =\n" << AtHf;
}

// The observations are exact projections of the point the feature's state
// encodes, so every innovation -- left and right -- must vanish.
TEST_F(OOSStereoTest, ExactObservationsGiveZeroInnovation) {
  int rows = FillJacobians();
  EXPECT_LT(inn0_.cwiseAbs().maxCoeff(), 1e-8) << "inn =\n" << inn0_.transpose();
  EXPECT_EQ(rows, 4 * static_cast<int>(obs_.size()));
}

// `stereo_R_scale` is applied as a whitening factor on the rows as they are
// written, so that the filter can keep feeding a scalar `Roos_`.
TEST_F(OOSStereoTest, RightRowsAreWhitened) {
  // A nonzero innovation to check, and a Jacobian that is not at the optimum.
  SetPoint(1.05);

  FillJacobians();
  MatX Hf1 = Hf0_, Hx1 = Hx0_;
  VecX inn1 = inn0_;

  options_.stereo_R_scale = 4.0; // -> weight 1/2
  FillJacobians();

  const int n = obs_.size();
  for (int i = 0; i < n; ++i) {
    const int lrow = 4 * i, rrow = 4 * i + 2;
    // Left rows untouched ...
    EXPECT_TRUE(Hf0_.middleRows<2>(lrow).isApprox(Hf1.middleRows<2>(lrow)));
    EXPECT_TRUE(Hx0_.middleRows<2>(lrow).isApprox(Hx1.middleRows<2>(lrow)));
    EXPECT_TRUE(inn0_.segment<2>(lrow).isApprox(inn1.segment<2>(lrow)));
    // ... right rows halved, all three of them consistently.
    EXPECT_TRUE(Hf0_.middleRows<2>(rrow).isApprox(0.5 * Hf1.middleRows<2>(rrow)));
    EXPECT_TRUE(Hx0_.middleRows<2>(rrow).isApprox(0.5 * Hx1.middleRows<2>(rrow)));
    EXPECT_TRUE(inn0_.segment<2>(rrow).isApprox(0.5 * inn1.segment<2>(rrow)));
  }
  EXPECT_GT(inn1.cwiseAbs().maxCoeff(), 1e-3) << "test would be vacuous";
}

// Finite differences on the right camera's two rows, against the same
// perturbation conventions the filter uses (`State::operator+=`: a *right*
// perturbation on each rotation).
TEST_F(OOSStereoTest, RightRowsMatchNumericDifferentiation) {
  BuildTrack(2);
  // Off the optimum, so that the innovation check below is not vacuous.
  SetPoint(1.05);
  ASSERT_EQ(FillJacobians(), 8);

  // Second view, so that its group block is not the anchor's and the pose
  // perturbations are not degenerate.
  const int view = 1;
  const int rrow = 4 * view + 2;
  const SE3 gsb_nom = PoseAt(view);
  const int goff = kGroupBegin + 6 * groups_[view]->sind();
  const Vec3 Xs_nom = f_->Xs(Gbc());

  auto predict = [&](const Vec3 &dWsb, const Vec3 &dTsb, const Vec3 &dWbc,
                     const Vec3 &dTbc, const Vec3 &dXs) {
    SE3 gsb{gsb_nom.so3() * SO3::exp(dWsb), gsb_nom.translation() + dTsb};
    SE3 gbc{Gbc().so3() * SO3::exp(dWbc), Gbc().translation() + dTbc};
    Vec3 Xc0 = (gsb * gbc).inverse() * (Xs_nom + dXs);
    Vec3 Xc1 = StereoRig::instance()->ToCam1(Xc0);
    return Camera::instance(1)->Project(project(Xc1));
  };

  const number_t delta = 1e-6;
  // One-sided differences on entries that are O(100) pixels per unit, so the
  // tolerance has to scale with the entry rather than be absolute.
  auto tol_for = [](number_t v) { return 1e-3 * std::max<number_t>(1.0, std::abs(v)); };
  const Vec3 zero = Vec3::Zero();
  const Vec2 xp0 = predict(zero, zero, zero, zero, zero);

  // Each column of each block, one perturbed variable at a time.
  for (int k = 0; k < 3; ++k) {
    Vec3 d = Vec3::Zero();
    d(k) = delta;
    const Vec2 num_Wsb = (predict(d, zero, zero, zero, zero) - xp0) / delta;
    const Vec2 num_Tsb = (predict(zero, d, zero, zero, zero) - xp0) / delta;
    const Vec2 num_Wbc = (predict(zero, zero, d, zero, zero) - xp0) / delta;
    const Vec2 num_Tbc = (predict(zero, zero, zero, d, zero) - xp0) / delta;
    const Vec2 num_Xs = (predict(zero, zero, zero, zero, d) - xp0) / delta;

    for (int r = 0; r < 2; ++r) {
      EXPECT_NEAR(Hx0_(rrow + r, goff + k), num_Wsb(r), tol_for(num_Wsb(r)))
          << "Wsb col " << k;
      EXPECT_NEAR(Hx0_(rrow + r, goff + 3 + k), num_Tsb(r), tol_for(num_Tsb(r)))
          << "Tsb col " << k;
      EXPECT_NEAR(Hx0_(rrow + r, Index::Wbc + k), num_Wbc(r), tol_for(num_Wbc(r)))
          << "Wbc col " << k;
      EXPECT_NEAR(Hx0_(rrow + r, Index::Tbc + k), num_Tbc(r), tol_for(num_Tbc(r)))
          << "Tbc col " << k;
      EXPECT_NEAR(Hf0_(rrow + r, k), num_Xs(r), tol_for(num_Xs(r)))
          << "Hf col " << k;
    }
  }

  // And the innovation is `observed - predicted`, like every other row.
  EXPECT_LT((inn0_.segment<2>(rrow) - (obs_[view].xp_r - xp0)).norm(), 1e-9);
}

// The rows of a view the right camera cannot see are simply not there: the
// left ones still are, and everything below shifts up.
TEST_F(OOSStereoTest, ViewWithoutRightObservationContributesTwoRows) {
  obs_[1].has_right = false;
  const int rows = FillJacobians();
  EXPECT_EQ(rows, 4 * 3 + 2);
  // Rows 4..5 are view 1's left rows; view 2's left rows follow immediately.
  const int goff2 = kGroupBegin + 6 * groups_[2]->sind();
  const number_t view2_block_max =
      Hx0_.block<2, 6>(6, goff2).cwiseAbs().maxCoeff();
  EXPECT_GT(view2_block_max, 0.0);
}

// The right observations pin the depth of a track with too little parallax for
// the left camera alone -- which is the reason to have them at all. The
// comparison needs measurement noise to be meaningful: with exact observations
// even a millimetre of parallax determines the depth exactly, and it is the
// *amplification* of a matching error by a short baseline that ruins the
// monocular estimate.
TEST_F(OOSStereoTest, RightObservationsPinDepthOfALowParallaxTrack) {
  // 21 mm of motion over the whole track, against a ~100 mm stereo baseline.
  BuildTrack(4, 0.1);
  const number_t z_true = f_->z();

  // A third of a pixel of matching error on the left observations, alternating
  // so that it is not a pure image shift (which the point's x/y absorb).
  for (size_t i = 0; i < obs_.size(); ++i) {
    const number_t s = (i % 2) ? 1.0 : -1.0;
    obs_[i].xp += 0.33 * Vec2{s, -s};
  }
  // The threshold exists to catch a *bad* triangulation; here we are measuring
  // how far a legitimate one is off, so let both arms through it.
  options_.max_mean_reproj_err = 100.0;

  SetPoint(1.4);
  options_.use_stereo = false;
  ASSERT_TRUE(f_->RefineOOSDepth(Gbc(), obs_, options_));
  const number_t err_mono = std::abs(f_->z() - z_true);

  SetPoint(1.4);
  options_.use_stereo = true;
  ASSERT_TRUE(f_->RefineOOSDepth(Gbc(), obs_, options_));
  const number_t err_stereo = std::abs(f_->z() - z_true);

  EXPECT_LT(err_stereo, 0.1 * z_true)
      << "z_true=" << z_true << " err_stereo=" << err_stereo;
  EXPECT_LT(err_stereo, err_mono / 3)
      << "err_mono=" << err_mono << " err_stereo=" << err_stereo;
}

// ... and they are part of the triangulation gate, so a bad right match is
// caught rather than marginalized into the pose window.
TEST_F(OOSStereoTest, BadRightMatchFailsTheTriangulationGate) {
  for (auto &obs : obs_) {
    obs.xp_r += Vec2{12.0, -9.0};
  }

  options_.use_stereo = false;
  SetPoint(1.0);
  EXPECT_TRUE(f_->RefineOOSDepth(Gbc(), obs_, options_))
      << "the left observations are still exact";

  options_.use_stereo = true;
  SetPoint(1.0);
  EXPECT_FALSE(f_->RefineOOSDepth(Gbc(), obs_, options_));
  EXPECT_GT(f_->oos_mean_reproj_err(), options_.max_mean_reproj_err);
}

// The Jacobian buffer is sized `2 * kMaxGroup` rows and shared by every pooled
// feature, so the view budget is halved when the right rows are in play instead.
TEST_F(OOSStereoTest, ViewBudgetIsHalvedForStereo) {
  std::vector<Observation> many = obs_;
  // `SelectOOSObservations` only ever thins, so a synthetic list of the right
  // length is enough here; the groups are reused.
  while (static_cast<int>(many.size()) < 2 * kMaxGroup) {
    many.push_back(obs_[many.size() % obs_.size()]);
  }

  options_.use_stereo = true;
  auto sel_stereo = f_->SelectOOSObservations(many, options_);
  EXPECT_EQ(static_cast<int>(sel_stereo.size()), kMaxGroup / 2);
  EXPECT_LE(4 * static_cast<int>(sel_stereo.size()), f_->oos_.Hf.rows());

  options_.use_stereo = false;
  auto sel_mono = f_->SelectOOSObservations(many, options_);
  EXPECT_EQ(static_cast<int>(sel_mono.size()), kMaxGroup);
  EXPECT_LE(2 * static_cast<int>(sel_mono.size()), f_->oos_.Hf.rows());
}
