// Numerical validation of the Jacobians `Feature::ChangeOwner` reports when a
// feature is re-anchored to a new reference group.
//
// Re-anchoring is a change of state coordinates:
//
//     xn = pi( (gsb_n gbc)^-1 gsb_o gbc pi^-1(x) )
//
// so the filter covariance has to be pushed through a *row* of Jacobians -- one
// w.r.t. the feature's own three parameters and one w.r.t. each of the two
// groups' 6-DOF poses. Only the first of the three existed (and it was applied
// to the feature's dead local copy of P, never to the filter block).
#include <gtest/gtest.h>

#define private public

#include "alias.h"
#include "camera_manager.h"
#include "feature.h"
#include "group.h"
#include "mm.h"
#include "project.h"

#include "unittest_helpers.h"

using namespace xivo;

namespace {

// Same convention as the rest of the filter (core.h): the rotation error is a
// RIGHT perturbation, the translation error is additive.
SE3 Perturb(const SE3 &g, const Vec3 &dW, const Vec3 &dT) {
  return SE3(g.so3() * SO3::exp(dW), g.translation() + dT);
}

} // namespace

class ReanchorJacobianTest : public ::testing::Test {
protected:
  void SetUp() override {
    MemoryManager::Create(256, 128);
    auto cfg = LoadJson("src/test/camera_configs.json");
    Camera::Create(cfg["perfect_pinhole"]);

    // Deliberately not random poses: the two cameras have to look at the same
    // point from a sane baseline, or `ChangeOwner` bails on negative depth and
    // there is nothing to differentiate.
    gbc = SE3(SO3::exp(Vec3{0.02, -0.03, 0.01}), Vec3{0.05, -0.02, 0.01});
    gsb_o = SE3(SO3::exp(Vec3{0.05, 0.10, -0.07}), Vec3{0.30, -0.10, 0.20});
    gsb_n = SE3(SO3::exp(Vec3{-0.08, 0.04, 0.11}), Vec3{0.55, 0.15, 0.10});

    go = Group::Create(gsb_o.so3(), gsb_o.translation());
    gn = Group::Create(gsb_n.so3(), gsb_n.translation());
    go->SetSind(0);
    gn->SetSind(1);

    f = Feature::Create(30.0, -12.0);
    // A point about 4 m out, off-axis in both directions.
    x0 << 0.11, -0.07, std::log(4.0);
    f->x_ = x0;
    f->ref_ = go;
    f->SetSind(0);
    f->status_ = FeatureStatus::INSTATE;
  }

  /** The quantity whose derivatives we are after: the feature's local
   *  parameterization in the *new* reference camera, as a function of the
   *  feature's own error and both groups' pose errors. Written out from the
   *  geometry rather than reusing any of the code under test. */
  Vec3 Reparameterized(const Vec3 &dx, const Vec3 &dW_o, const Vec3 &dT_o,
                       const Vec3 &dW_n, const Vec3 &dT_n) const {
    const Vec3 Xc_o = unproject_logz(Vec3(x0 + dx));
    const SE3 gsc_o = Perturb(gsb_o, dW_o, dT_o) * gbc;
    const SE3 gsc_n = Perturb(gsb_n, dW_n, dT_n) * gbc;
    const Vec3 Xcn = (gsc_n.inverse() * gsc_o) * Xc_o;
    return project_logz(Xcn);
  }

  /** Central-difference one 3-vector argument of `Reparameterized`. */
  Mat3 NumericalBlock(int which) const {
    constexpr number_t eps = 1e-6;
    Mat3 J;
    for (int i = 0; i < 3; ++i) {
      Vec3 d[5] = {Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), Vec3::Zero(),
                   Vec3::Zero()};
      d[which](i) = eps;
      const Vec3 plus = Reparameterized(d[0], d[1], d[2], d[3], d[4]);
      d[which](i) = -eps;
      const Vec3 minus = Reparameterized(d[0], d[1], d[2], d[3], d[4]);
      J.col(i) = (plus - minus) / (2 * eps);
    }
    return J;
  }

  SE3 gbc, gsb_o, gsb_n;
  GroupPtr go{nullptr}, gn{nullptr};
  FeaturePtr f{nullptr};
  Vec3 x0;
  static constexpr number_t tol = 1e-6;
};

TEST_F(ReanchorJacobianTest, ChangeOwnerSucceedsOnThisGeometry) {
  Feature::ReanchorJacobians jac;
  ASSERT_TRUE(f->ChangeOwner(gn, gbc, &jac));
  // Sanity: the re-parameterized mean must match the geometry, otherwise the
  // Jacobian comparisons below are being made at the wrong point.
  const Vec3 expected =
      Reparameterized(Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), Vec3::Zero(),
                      Vec3::Zero());
  CheckVectorEquality(f->x_, expected, 1e-9);
  EXPECT_EQ(f->ref(), gn);
}

TEST_F(ReanchorJacobianTest, JacobianWrtOwnParameters) {
  Feature::ReanchorJacobians jac;
  ASSERT_TRUE(f->ChangeOwner(gn, gbc, &jac));
  CheckMatrixEquality(jac.dxn_dx, NumericalBlock(0), tol);
  // Not vacuous: this block is nowhere near zero or identity.
  EXPECT_GT(jac.dxn_dx.norm(), 1e-2);
}

TEST_F(ReanchorJacobianTest, JacobianWrtOutgoingReferencePose) {
  Feature::ReanchorJacobians jac;
  ASSERT_TRUE(f->ChangeOwner(gn, gbc, &jac));
  CheckMatrixEquality(jac.dxn_dref_old.leftCols<3>(), NumericalBlock(1), tol);
  CheckMatrixEquality(jac.dxn_dref_old.rightCols<3>(), NumericalBlock(2), tol);
  EXPECT_GT(jac.dxn_dref_old.norm(), 1e-2);
}

TEST_F(ReanchorJacobianTest, JacobianWrtIncomingReferencePose) {
  Feature::ReanchorJacobians jac;
  ASSERT_TRUE(f->ChangeOwner(gn, gbc, &jac));
  CheckMatrixEquality(jac.dxn_dref_new.leftCols<3>(), NumericalBlock(3), tol);
  CheckMatrixEquality(jac.dxn_dref_new.rightCols<3>(), NumericalBlock(4), tol);
  EXPECT_GT(jac.dxn_dref_new.norm(), 1e-2);
}

TEST_F(ReanchorJacobianTest, PoseJacobiansAreNotInterchangeable) {
  // The two 3x6 blocks are genuinely different maps -- a regression that swapped
  // them, or reported the same one twice, has to fail.
  Feature::ReanchorJacobians jac;
  ASSERT_TRUE(f->ChangeOwner(gn, gbc, &jac));
  EXPECT_FALSE(jac.dxn_dref_old.isApprox(jac.dxn_dref_new, 1e-3));
  EXPECT_FALSE(jac.dxn_dref_old.isApprox(-jac.dxn_dref_new, 1e-3));
}

TEST_F(ReanchorJacobianTest, RejectsNegativeDepthWithoutMutating) {
  // The documented contract: "If change in ownership results in negative depth,
  // no changes in any members of this feature are made."
  const SE3 behind(SO3(), Vec3{0.0, 0.0, 20.0}); // new camera past the point
  GroupPtr gbad = Group::Create(behind.so3(), behind.translation());
  gbad->SetSind(2);
  Feature::ReanchorJacobians jac;
  EXPECT_FALSE(f->ChangeOwner(gbad, gbc, &jac));
  CheckVectorEquality(f->x_, x0, 1e-12);
  EXPECT_EQ(f->ref(), go);
}
