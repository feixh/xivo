// Finite-difference check on the right camera's measurement Jacobian (M5), plus
// coverage of the copy of both cameras' rows into the big H matrix.
//
// Run from the repository root: the config path below is relative to it.
//
// Why a separate binary from unitTests_Jacobians: `CameraManager`'s registry is
// process-wide and keeps whatever camera was installed first in each slot, so a
// test needing the real fisheye pair in slots 0/1 cannot share a process with
// one that installs a perfect pinhole in slot 0.
#include <gtest/gtest.h>

#include <random>
#include <utility>
#include <vector>

#define private public

#include "alias.h"
#include "feature.h"
#include "graph.h"
#include "group.h"
#include "mm.h"
#include "stereo.h"

using namespace xivo;

namespace {

const char *kStereoCfg = "cfg/tumvi_stereo.json";

// The error-state variables the right rows must depend on *and* that a finite
// difference through `ComputeRightPixel` can reach. The online-calibration
// columns are live too when their flags are on, but they do not move the
// predicted pixel -- see `LiveBlocks` below, which is the full list.
enum class Var { Wsb, Tsb, Wbc, Tbc, Wsbr, Tsbr, x };

} // namespace

class StereoJacobiansTest : public ::testing::Test {
protected:
  void SetUp() override {
    MemoryManager::Create(256, 128);
    auto cfg = LoadJson(kStereoCfg);
    ASSERT_FALSE(cfg.isNull())
        << "could not load " << kStereoCfg << "; run tests from the repo root";
    cam0 = Camera::Create(cfg["camera_cfg"], 0);
    cam1 = Camera::Create(cfg["camera1_cfg"], 1);
    rig = StereoRig::Create(cfg["stereo_cfg"]);
    ASSERT_NE(cam1, nullptr);
    ASSERT_NE(rig, nullptr);

    delta = 1e-6;

    // Nominal states. The current body pose is a *small* motion away from the
    // reference group's, as it is in practice at 20 Hz, which keeps the point in
    // front of both cameras -- with two independently random poses the predicted
    // point lands behind a camera most of the time and there is no Jacobian to
    // compare against.
    std::default_random_engine generator(7);
    gsbr_nom = SE3::sampleUniform(generator);
    gsb_nom = gsbr_nom * SE3(SO3::exp(Vec3{0.01, -0.02, 0.015}),
                             Vec3{0.03, -0.01, 0.02});
    gbc_nom = SE3::sampleUniform(generator);

    // These reach only the online-calibration columns; none of them moves the
    // predicted pixel (the pose `ComputeJacobian` is handed is already the
    // td-corrected one), so they are free to be nonzero without disturbing any
    // of the finite differences below.
    gyro = Vec3::Random();
    Cg_nom = Mat3::Identity();
    bg_nom = Vec3::Zero();
    Vsb_nom = Vec3::Random();
    // Nonzero on purpose, and about the magnitude the filter actually estimates
    // on TUM-VI: `dXcn_dbg` carries a factor of td, so at td = 0 the whole bg
    // block would be zero and `NoBlockIsAccidentallyZero` could not tell a
    // filled block from a forgotten one.
    td_nom = 5e-3;

    ClearErrors();

    // A left observation well inside the fisheye's field, at ~7.4 m (log-depth
    // 2.0, which is what Feature::Reset seeds).
    Vec2 xp{200.0, 250.0};
    f = Feature::Create(xp(0), xp(1));
    Vec2 xc = cam0->UnProject(xp);
    f->x_(0) = xc(0);
    f->x_(1) = xc(1);
    group = Group::Create(gsbr_nom.so3(), gsbr_nom.translation());
    group->SetSind(0);
    f->ref_ = group;
    f->SetSind(0);

    // A right observation. Its *value* only affects the innovation, not the
    // Jacobian, so put it a couple of pixels off the true projection: that way
    // the innovation is nonzero and a sign error in it would show.
    Vec2 xp1_true = ComputeRightPixel();
    ASSERT_TRUE(xp1_true.allFinite());
    f->SetRightObs(xp1_true + Vec2{2.0, -1.5});

    f->ComputeJacobian(gsb_nom.so3().matrix(), gsb_nom.translation(),
                       gbc_nom.so3().matrix(), gbc_nom.translation(), gyro,
                       Cg_nom, bg_nom, Vsb_nom, td_nom);
    ASSERT_TRUE(f->right_jac_valid())
        << "the right rows were not computed; the rest of this fixture is moot";
  }

  void ClearErrors() {
    Wsb_err.setZero();
    Tsb_err.setZero();
    Wbc_err.setZero();
    Tbc_err.setZero();
    Wsbr_err.setZero();
    Tsbr_err.setZero();
    x_err.setZero();
  }

  /** The predicted *right* pixel as a function of the current error variables.
   * Deliberately written out from scratch rather than by calling into the
   * cache, so the numbers it produces are independent of the code under test. */
  Vec2 ComputeRightPixel() {
    SO3 Rsbr = gsbr_nom.so3() * SO3::exp(Wsbr_err);
    Vec3 Tsbr = gsbr_nom.translation() + Tsbr_err;
    SO3 Rsb = gsb_nom.so3() * SO3::exp(Wsb_err);
    Vec3 Tsb = gsb_nom.translation() + Tsb_err;
    SO3 Rbc = gbc_nom.so3() * SO3::exp(Wbc_err);
    Vec3 Tbc = gbc_nom.translation() + Tbc_err;

    // The feature's own error state is additive on x_ = (X/Z, Y/Z, log Z), so
    // perturb there and unproject, exactly as `Feature::UpdateState` does.
    Vec3 x_pert = f->x_ + x_err;
    Vec3 Xc = unproject_logz(x_pert);

    Vec3 Xs = Rsbr * (Rbc * Xc + Tbc) + Tsbr;
    Vec3 Xcn = SE3(Rbc, Tbc).inverse() * (SE3(Rsb, Tsb).inverse() * Xs);
    Vec3 Xc1 = rig->ToCam1(Xcn);
    return cam1->Project(project(Xc1));
  }

  /** Central-difference column i of d(right pixel)/d(var). */
  Vec2 NumericColumn(Var v, int i) {
    number_t *p = nullptr;
    switch (v) {
    case Var::Wsb:  p = &Wsb_err(i);  break;
    case Var::Tsb:  p = &Tsb_err(i);  break;
    case Var::Wbc:  p = &Wbc_err(i);  break;
    case Var::Tbc:  p = &Tbc_err(i);  break;
    case Var::Wsbr: p = &Wsbr_err(i); break;
    case Var::Tsbr: p = &Tsbr_err(i); break;
    case Var::x:    p = &x_err(i);    break;
    }
    *p = delta;
    Vec2 plus = ComputeRightPixel();
    *p = -delta;
    Vec2 minus = ComputeRightPixel();
    *p = 0.0;
    return (plus - minus) / (2 * delta);
  }

  /** Column offset of `v` in the 2 x kFullSize row pair. */
  int ColumnOf(Var v) {
    switch (v) {
    case Var::Wsb:  return Index::Wsb;
    case Var::Tsb:  return Index::Tsb;
    case Var::Wbc:  return Index::Wbc;
    case Var::Tbc:  return Index::Tbc;
    case Var::Wsbr: return kGroupBegin + 6 * group->sind();
    case Var::Tsbr: return kGroupBegin + 6 * group->sind() + 3;
    case Var::x:    return kFeatureBegin + 3 * f->sind();
    }
    return -1;
  }

  /** Every (column, width) block the right rows are allowed to touch.
   *
   * The `Var` list is the part a finite difference can reach. The
   * online-calibration blocks are live as well whenever their flag is set --
   * `USE_ONLINE_TEMPORAL_CALIB` was off when this file was written and is on
   * since the auto-bugfix merge -- but `td`, `bg` and `Cg` do not appear in the
   * predicted pixel at all, only in its derivative, so there is no finite
   * difference to compare them against here. They still have to be listed, or
   * the structural tests below would read a correctly filled column as a stray
   * write. `TemporalCalibColumnsShareTheLeftChain` is what checks their values.
   */
  std::vector<std::pair<int, int>> LiveBlocks() {
    std::vector<std::pair<int, int>> blocks;
    for (Var v : {Var::Wsb, Var::Tsb, Var::Wbc, Var::Tbc, Var::Wsbr, Var::Tsbr,
                  Var::x}) {
      blocks.emplace_back(ColumnOf(v), 3);
    }
#ifdef USE_ONLINE_TEMPORAL_CALIB
    blocks.emplace_back(Index::td, 1);
#ifdef USE_ONLINE_IMU_CALIB
    blocks.emplace_back(Index::Cg, 9);
#endif
    blocks.emplace_back(Index::bg, 3);
#endif
    return blocks;
  }

  void CheckBlock(Var v, const char *name) {
    const int col = ColumnOf(v);
    for (int i = 0; i < 3; ++i) {
      Vec2 num = NumericColumn(v, i);
      Vec2 ana = f->J_r_.block<2, 1>(0, col + i);
      // The entries span several orders of magnitude (a 190 px focal length
      // times a chain of O(1) terms, against near-zero columns), so the
      // tolerance is relative with an absolute floor.
      const number_t tol = 1e-5 * (1.0 + ana.norm());
      EXPECT_NEAR(num(0), ana(0), tol) << name << " column " << i;
      EXPECT_NEAR(num(1), ana(1), tol) << name << " column " << i;
    }
  }

  CameraManager *cam0;
  CameraManager *cam1;
  StereoRig *rig;
  GroupPtr group;
  FeaturePtr f;

  SE3 gsbr_nom, gsb_nom, gbc_nom;
  Mat3 Cg_nom;
  Vec3 bg_nom, Vsb_nom, gyro;
  number_t td_nom;

  Vec3 Wsb_err, Tsb_err, Wbc_err, Tbc_err, Wsbr_err, Tsbr_err, x_err;

  number_t delta;
};

TEST_F(StereoJacobiansTest, RightInnovationIsObservedMinusPredicted) {
  // The prediction the fixture built the observation from must come back out of
  // the implementation: `inn_r` is `xp_r - predicted`, and the fixture offset
  // the observation by exactly (+2, -1.5).
  ClearErrors();
  Vec2 xp1 = ComputeRightPixel();
  EXPECT_NEAR((f->xp_r() - xp1)(0), f->inn_r()(0), 1e-9);
  EXPECT_NEAR((f->xp_r() - xp1)(1), f->inn_r()(1), 1e-9);
  EXPECT_NEAR(f->inn_r()(0), 2.0, 1e-9);
  EXPECT_NEAR(f->inn_r()(1), -1.5, 1e-9);
}

TEST_F(StereoJacobiansTest, RightPredictionUsesTheRightCamera) {
  // A guard against the easy mistake of projecting the right-camera point with
  // camera 0's intrinsics, or of skipping the rigid hop into camera 1: either
  // would leave the two cameras' predictions much closer together than the
  // ~19 px disparity a 101 mm baseline gives at 7.4 m.
  ClearErrors();
  Vec2 xp1 = ComputeRightPixel();
  Vec2 xp0 = f->back();
  EXPECT_GT((xp1 - xp0).norm(), 5.0)
      << "left " << xp0.transpose() << " right " << xp1.transpose();
}

TEST_F(StereoJacobiansTest, Wsb) { CheckBlock(Var::Wsb, "Wsb"); }
TEST_F(StereoJacobiansTest, Tsb) { CheckBlock(Var::Tsb, "Tsb"); }
TEST_F(StereoJacobiansTest, Wbc) { CheckBlock(Var::Wbc, "Wbc"); }
TEST_F(StereoJacobiansTest, Tbc) { CheckBlock(Var::Tbc, "Tbc"); }
TEST_F(StereoJacobiansTest, Wsbr) { CheckBlock(Var::Wsbr, "Wsbr"); }
TEST_F(StereoJacobiansTest, Tsbr) { CheckBlock(Var::Tsbr, "Tsbr"); }
TEST_F(StereoJacobiansTest, FeatureState) { CheckBlock(Var::x, "x"); }

TEST_F(StereoJacobiansTest, NoBlockIsAccidentallyZero) {
  // Every block checked above is compared against a finite difference, but a
  // block that is zero in *both* would pass silently -- which is exactly the
  // failure mode of forgetting to fill one. None of these are zero for a
  // generic pose.
  for (auto [col, width] : LiveBlocks()) {
    const number_t n = f->J_r_.block(0, col, 2, width).norm();
    EXPECT_GT(n, 1e-6) << "block at column " << col << " is zero";
  }
}

TEST_F(StereoJacobiansTest, AllOtherColumnsAreZero) {
  // The rig extrinsics are fixed and camera 1's intrinsics are not in the state,
  // so the right rows must touch no columns beyond the ones above. In particular
  // no *other* group's or feature's slot: a stray write there would silently
  // corrupt an unrelated part of the state.
  Eigen::Matrix<number_t, 2, kFullSize> J = f->J_r_;
  for (auto [col, width] : LiveBlocks()) {
    J.block(0, col, 2, width).setZero();
  }
  EXPECT_NEAR(J.norm(), 0.0, 1e-12);
}

TEST_F(StereoJacobiansTest, RightRowsAreNotTheLeftRows) {
  // Both cameras see the same 3D point through the same state, so their
  // Jacobians are similar in structure -- but a copy-paste that projected with
  // camera 0 would make them *identical*, which no real stereo pair is.
  EXPECT_GT((f->J_r_ - f->J_).norm() / f->J_.norm(), 1e-3);
}

#ifdef USE_ONLINE_TEMPORAL_CALIB
TEST_F(StereoJacobiansTest, TemporalCalibColumnsShareTheLeftChain) {
  // The td and bg columns are the one live part of the right rows a finite
  // difference cannot reach (they do not move the predicted pixel), and turning
  // USE_ONLINE_TEMPORAL_CALIB on made them live for the first time. They are
  // still checkable: each camera's column is *its own* projection Jacobian times
  // the *same* chain vector (`cache_.dXcn_dtd`, `cache_.dXcn_dbg`), so
  //
  //     [ dxp0_dXcn ]        [ J_  (:, c) ]
  //     [ dxp1_dXcn ] * v =  [ J_r_(:, c) ]
  //
  // must be consistent for some 3-vector v. Each dxp_dXcn is recovered from the
  // Tsb block -- which the finite differences above already pin down -- divided
  // by the analytically known dXcn_dTsb = -Rbc^T Rsb^T. The obvious copy-paste,
  // building the right column with the *left* camera's dxp_dXcn, fails this: the
  // two projection Jacobians have different null directions, so no single v
  // explains both halves.
  ClearErrors();
  const Mat3 Rsb = gsb_nom.so3().matrix();
  const Mat3 Rbc = gbc_nom.so3().matrix();
  const Mat3 dTsb_inv = (-Rbc.transpose() * Rsb.transpose()).inverse();

  Eigen::Matrix<number_t, 4, 3> M;
  M.topRows<2>() = f->J_.block<2, 3>(0, Index::Tsb) * dTsb_inv;
  M.bottomRows<2>() = f->J_r_.block<2, 3>(0, Index::Tsb) * dTsb_inv;

  for (auto [begin, width] :
       std::vector<std::pair<int, int>>{{Index::td, 1}, {Index::bg, 3}}) {
    for (int i = 0; i < width; ++i) {
      const int c = begin + i;
      Eigen::Matrix<number_t, 4, 1> rhs;
      rhs.head<2>() = f->J_.block<2, 1>(0, c);
      rhs.tail<2>() = f->J_r_.block<2, 1>(0, c);
      EXPECT_GT(rhs.tail<2>().norm(), 1e-6) << "right column " << c << " is zero";
      const Vec3 v = M.colPivHouseholderQr().solve(rhs);
      EXPECT_LT((M * v - rhs).norm(), 1e-8 * (1.0 + rhs.norm()))
          << "column " << c << " is not the left chain seen through camera 1";
    }
  }
}
#endif

TEST_F(StereoJacobiansTest, FillJacobianBlockCopiesEveryLiveBlock) {
  // The deferred M0 coverage test: `FillJacobianBlock` used to write the
  // reference group's translation block over its rotation block, a bug invisible
  // to any test that only looked at `J_`. Check both cameras' copies.
  MatX H;
  H.setZero(4, kFullSize);
  f->FillJacobianBlock(H, 0);
  f->FillRightJacobianBlock(H, 2);

  for (auto [col, width] : LiveBlocks()) {
    EXPECT_NEAR(
        (H.block(0, col, 2, width) - f->J_.block(0, col, 2, width)).norm(), 0.0,
        1e-12)
        << "left rows, column " << col;
    EXPECT_NEAR(
        (H.block(2, col, 2, width) - f->J_r_.block(0, col, 2, width)).norm(),
        0.0, 1e-12)
        << "right rows, column " << col;
  }

  // And nothing outside those blocks was written, in either row pair.
  MatX H2 = H;
  for (auto [col, width] : LiveBlocks()) {
    H2.block(0, col, 4, width).setZero();
  }
  EXPECT_NEAR(H2.norm(), 0.0, 1e-12);
}

TEST_F(StereoJacobiansTest, NoRightMatchMeansNoRightRows) {
  // `has_right()` is cleared at the start of every stereo frame, so a feature
  // that loses its match must stop contributing rows -- otherwise a stale
  // observation would be paired with a fresh prediction.
  f->ClearRightObs();
  f->ComputeJacobian(gsb_nom.so3().matrix(), gsb_nom.translation(),
                     gbc_nom.so3().matrix(), gbc_nom.translation(), gyro,
                     Cg_nom, bg_nom, Vsb_nom, td_nom);
  EXPECT_FALSE(f->right_jac_valid());
}

TEST_F(StereoJacobiansTest, PointBehindTheRightCameraContributesNoRows) {
  // A match can survive the tracker's gates while the *current state* predicts
  // the point behind the cameras -- a badly diverged pose or depth. There is no
  // valid linearization there, so the right rows must be dropped rather than
  // computed from a negative depth.
  //
  // The construction moves the current body pose 10 m along the *reference*
  // camera's optical axis, i.e. straight past a point 7.4 m down that axis.
  // Note the baseline is lateral, so no depth in front of camera 0 alone puts a
  // point behind camera 1; the current pose has to overshoot the point.
  Vec3 axis_s = gsbr_nom.so3() * (gbc_nom.so3() * Vec3{0.0, 0.0, 1.0});
  Vec3 Tsb_far = gsb_nom.translation() + 10.0 * axis_s;
  f->SetRightObs(Vec2{100.0, 100.0});
  f->ComputeJacobian(gsb_nom.so3().matrix(), Tsb_far,
                     gbc_nom.so3().matrix(), gbc_nom.translation(), gyro,
                     Cg_nom, bg_nom, Vsb_nom, td_nom);
  Vec3 Xc1 = rig->ToCam1(f->cache_.Xcn);
  ASSERT_LT(Xc1(2), 0.0) << "the fixture failed to put the point behind cam1";
  EXPECT_FALSE(f->right_jac_valid());
}

////////////////////////////////////////////////////////////////////////////////
// The compacted innovation covariance (M1).
//
// `InnovationCov` (core.h) evaluates `J P J^T + R I` from the `kJacCols` columns
// of `P` that `J` can reach instead of all `kFullSize` of them. That is only
// equal to the dense product if `J` really is zero everywhere else, so the two
// claims are tested together and against *both* cameras' rows.
////////////////////////////////////////////////////////////////////////////////
namespace {

/** A symmetric positive-definite matrix the size of `P_`, deterministic and
 *  built once: 564x564 is large enough that rebuilding it per test is the
 *  slowest thing in this binary. Off-diagonal correlations are strong (a random
 *  Gram matrix), which is what makes the skipped-column claim load-bearing --
 *  with a diagonal `P` the compaction would be trivially correct. */
const MatX &RandomP() {
  static const MatX P = [] {
    std::default_random_engine gen(11);
    std::normal_distribution<number_t> n(0.0, 1.0);
    MatX A(kFullSize, kFullSize);
    for (int i = 0; i < kFullSize; ++i)
      for (int j = 0; j < kFullSize; ++j)
        A(i, j) = n(gen);
    return (A * A.transpose() / kFullSize +
            0.1 * MatX::Identity(kFullSize, kFullSize))
        .eval();
  }();
  return P;
}

/** `J` with every column `MeasurementRuns` claims for this measurement blanked,
 *  i.e. the part the compacted product silently drops. */
Eigen::Matrix<number_t, 2, kFullSize>
OutsideTheRuns(const Eigen::Matrix<number_t, 2, kFullSize> &J, int gsind,
               int fsind) {
  ColRun runs[kJacRuns];
  MeasurementRuns(gsind, fsind, runs);
  Eigen::Matrix<number_t, 2, kFullSize> rest = J;
  for (const auto &r : runs) {
    rest.middleCols(r.start, r.len).setZero();
  }
  return rest;
}

} // namespace

TEST_F(StereoJacobiansTest, MeasurementRunsCoverEveryLiveBlock) {
  // The complement of this is `NothingOutsideTheMeasurementRunsIsNonzero` below,
  // which would also pass if the runs covered the whole state. Together they pin
  // `kJacSharedRuns` to exactly what `ComputeJacobian` writes, so the two cannot
  // drift apart as columns are added.
  ColRun runs[kJacRuns];
  MeasurementRuns(group->sind(), f->sind(), runs);

  int total = 0;
  for (const auto &r : runs) {
    total += r.len;
  }
  EXPECT_EQ(total, kJacCols) << "MeasurementRuns and kJacCols disagree";

  for (auto [col, width] : LiveBlocks()) {
    bool covered = false;
    for (const auto &r : runs) {
      if (col >= r.start && col + width <= r.start + r.len) {
        covered = true;
        break;
      }
    }
    EXPECT_TRUE(covered) << "live block at column " << col << " (width " << width
                         << ") is not in any column run; the compacted update "
                            "would drop it";
  }
}

TEST_F(StereoJacobiansTest, NothingOutsideTheMeasurementRunsIsNonzero) {
  // At slots other than 0/0, so that a run computed with the wrong stride would
  // leave the real group/feature columns outside and fail here. The left rows are
  // covered as well as the right: `MHGating` compacts both.
  group->SetSind(7);
  f->SetSind(33);
  f->ComputeJacobian(gsb_nom.so3().matrix(), gsb_nom.translation(),
                     gbc_nom.so3().matrix(), gbc_nom.translation(), gyro,
                     Cg_nom, bg_nom, Vsb_nom, td_nom);
  ASSERT_TRUE(f->right_jac_valid());

  EXPECT_EQ(OutsideTheRuns(f->J_, 7, 33).norm(), 0.0);
  EXPECT_EQ(OutsideTheRuns(f->J_r_, 7, 33).norm(), 0.0);
}

TEST_F(StereoJacobiansTest, CompactInnovationCovMatchesTheDenseProduct) {
  const number_t R = 1.5;

  for (auto [gsind, fsind] : {std::pair<int, int>{0, 0},
                              std::pair<int, int>{7, 33},
                              std::pair<int, int>{kMaxGroup - 1,
                                                  kMaxFeature - 1}}) {
    group->SetSind(gsind);
    f->SetSind(fsind);
    f->ComputeJacobian(gsb_nom.so3().matrix(), gsb_nom.translation(),
                       gbc_nom.so3().matrix(), gbc_nom.translation(), gyro,
                       Cg_nom, bg_nom, Vsb_nom, td_nom);
    ASSERT_TRUE(f->right_jac_valid());

    const MatX &P = RandomP();
    for (const auto *J : {&f->J_, &f->J_r_}) {
      Mat2 dense = (*J) * P * J->transpose();
      dense(0, 0) += R;
      dense(1, 1) += R;
      Mat2 compact = InnovationCov(*J, P, gsind, fsind, R);
      // Relative, not absolute: `S` here is O(1e5) (a 190 px focal length
      // squared times an O(1) covariance). Reassociating the sums is the only
      // difference allowed, so the tolerance is a few ulps of the magnitude.
      EXPECT_LT((compact - dense).norm() / dense.norm(), 1e-12)
          << "slots " << gsind << "/" << fsind << "\ndense =\n"
          << dense << "\ncompact =\n"
          << compact;
    }
  }
}

TEST_F(StereoJacobiansTest, CompactInnovationCovDependsOnTheRightSlots) {
  // Guards the test above from passing vacuously: if the group and feature
  // columns of `J` were zero (or the runs pointed at the wrong slots and got
  // zeros), reading the *wrong* slots would give the same answer.
  const MatX &P = RandomP();
  Mat2 right = InnovationCov(f->J_, P, group->sind(), f->sind(), 1.5);
  Mat2 wrong_group = InnovationCov(f->J_, P, group->sind() + 1, f->sind(), 1.5);
  Mat2 wrong_feat = InnovationCov(f->J_, P, group->sind(), f->sind() + 1, 1.5);
  EXPECT_GT((wrong_group - right).norm() / right.norm(), 1e-6);
  EXPECT_GT((wrong_feat - right).norm() / right.norm(), 1e-6);
}
