#include <gtest/gtest.h>

#include "core.h"

#include <random>

using namespace Eigen;
using namespace xivo;



TEST(CamerasAtan, atanProjectUnproject) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["atan_cam"]);

  std::default_random_engine generator;
  std::uniform_real_distribution<number_t> distribution(0.0, 5.0);

  Vec2 px;
  px(0) = distribution(generator);
  px(1) = distribution(generator);

  Vec2 px_projected = cam->Project(px);
  Vec2 px2 = cam->UnProject(px_projected);

  EXPECT_FLOAT_EQ(px(0), px2(0));
  EXPECT_FLOAT_EQ(px(1), px2(1));
}


// Both Project and UnProject switch to a "singular" branch near R = 0 and used
// f = 1 there. The true limits are 2 tan(w/2) / w and its reciprocal, so the
// model was discontinuous across the threshold.
TEST(CamerasAtan, atanIsContinuousAcrossTheSingularThreshold) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["atan_cam"]);

  // Project's threshold is R < 1e-4; straddle it.
  Vec2 inside(0.9e-4 / std::sqrt(2.0), 0.9e-4 / std::sqrt(2.0));
  Vec2 outside(1.1e-4 / std::sqrt(2.0), 1.1e-4 / std::sqrt(2.0));
  Vec2 pin = cam->Project(inside);
  Vec2 pout = cam->Project(outside);
  // Over a 2e-5 change in xc the projected pixel must move by well under a pixel.
  EXPECT_LT((pout - pin).norm(), 1e-2);

  // UnProject's threshold is R > 0.01, i.e. a pixel radius of ~0.01 * f.
  Mat2 J_in, J_out;
  Vec2 c = cam->UnProject(Vec2(0.0, 0.0));
  Vec2 uin = cam->UnProject(cam->Project(Vec2(0.9e-2 / std::sqrt(2.0),
                                              0.9e-2 / std::sqrt(2.0))),
                            &J_in);
  Vec2 uout = cam->UnProject(cam->Project(Vec2(1.1e-2 / std::sqrt(2.0),
                                               1.1e-2 / std::sqrt(2.0))),
                             &J_out);
  // Round trip must hold on both sides of the threshold.
  EXPECT_NEAR(uin(0), 0.9e-2 / std::sqrt(2.0), 1e-6);
  EXPECT_NEAR(uout(0), 1.1e-2 / std::sqrt(2.0), 1e-6);
  // ...and the Jacobian must not jump across it.
  EXPECT_LT((J_out - J_in).norm(), 1e-2 * J_in.norm());
  ASSERT_TRUE(c.allFinite());
}


// Both singular branches wrote only the diagonal, so the off-diagonals were
// whatever the caller's matrix already held.
TEST(CamerasAtan, atanSingularJacobiansAreFullyWritten) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["atan_cam"]);

  const number_t poison = 12345.0;

  Mat2 J = Mat2::Constant(poison);
  cam->Project(Vec2(0.0, 0.0), &J);
  EXPECT_NE(J(0, 1), poison);
  EXPECT_NE(J(1, 0), poison);
  EXPECT_EQ(J(0, 1), 0.0);
  EXPECT_EQ(J(1, 0), 0.0);

  Mat2 J2 = Mat2::Constant(poison);
  cam->UnProject(cam->Project(Vec2(0.0, 0.0)), &J2);
  EXPECT_NE(J2(0, 1), poison);
  EXPECT_NE(J2(1, 0), poison);
  EXPECT_EQ(J2(0, 1), 0.0);
  EXPECT_EQ(J2(1, 0), 0.0);
}


// R * w >= pi/2 is outside the model's image circle; tan flips sign there and the
// unprojected ray came back mirrored through the principal point.
TEST(CamerasAtan, atanUnprojectDoesNotMirrorFarPixels) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["atan_cam"]);

  Vec2 dir(0.6, 0.8);
  Vec2 principal = cam->Project(Vec2(0.0, 0.0));
  for (number_t r = 50.0; r < 5000.0; r += 173.0) {
    Vec2 xc = cam->UnProject(Vec2(principal + r * dir));
    ASSERT_TRUE(std::isfinite(xc(0))) << "r=" << r;
    ASSERT_TRUE(std::isfinite(xc(1))) << "r=" << r;
    EXPECT_GE(xc.dot(dir), 0.0) << "mirrored ray at r=" << r;
  }
}


TEST(CamerasAtan, atanProjectionJac) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["atan_cam"]);

  std::default_random_engine generator;
  std::uniform_real_distribution<number_t> distribution(0.0, 5.0);

  number_t delta = 1e-7;

  Vec2 px;
  Vec2 px_projected;
  Mat2 px_jac;
  px(0) = distribution(generator);
  px(1) = distribution(generator);

  px_projected = cam->Project(px, &px_jac);

  Vec2 px1 = px;
  px1(0) += delta;
  Vec2 px1_projected = cam->Project(px1);

  Vec2 px2 = px;
  px2(1) += delta;
  Vec2 px2_projected = cam->Project(px2);

  Vec2 approx_dx1 = (px1_projected - px_projected) / delta;
  Vec2 approx_dx2 = (px2_projected - px_projected) / delta;

  EXPECT_FLOAT_EQ(approx_dx1(0), px_jac(0,0));
  EXPECT_FLOAT_EQ(approx_dx1(1), px_jac(1,0));
  EXPECT_FLOAT_EQ(approx_dx2(0), px_jac(0,1));
  EXPECT_FLOAT_EQ(approx_dx2(1), px_jac(1,1));
}


TEST(CamerasAtan, atanProjectionJacc) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["atan_cam"]);

  std::default_random_engine generator;
  std::uniform_real_distribution<number_t> distribution(0.0, 5.0);

  number_t delta = 1e-7;

  Vec2 px;
  Vec2 px_proj;
  Eigen::Matrix<number_t, 2, Eigen::Dynamic> px_jacc;
  px(0) = distribution(generator);
  px(1) = distribution(generator);

  px_proj = cam->Project(px, nullptr, &px_jacc);

  Vec9 Intrinsics = cam->GetIntrinsics();

  Vec8 dX_fx;
  dX_fx << delta, 0, 0, 0, 0, 0, 0, 0;
  cam->UpdateState(dX_fx);
  Vec2 px_proj_fx = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_fx(0) - px_proj(0)) / delta, px_jacc(0,0));
  EXPECT_FLOAT_EQ((px_proj_fx(1) - px_proj(1)) / delta, px_jacc(1,0));
  cam->UpdateState(-dX_fx);

  Vec8 dX_fy;
  dX_fy << 0, delta, 0, 0, 0, 0, 0, 0;
  cam->UpdateState(dX_fy);
  Vec2 px_proj_fy = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_fy(0) - px_proj(0)) / delta, px_jacc(0,1));
  EXPECT_FLOAT_EQ((px_proj_fy(1) - px_proj(1)) / delta, px_jacc(1,1));
  cam->UpdateState(-dX_fy);

  Vec8 dX_cx;
  dX_cx << 0, 0, delta, 0, 0, 0, 0, 0;
  cam->UpdateState(dX_cx);
  Vec2 px_proj_cx = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_cx(0) - px_proj(0)) / delta, px_jacc(0,2));
  EXPECT_FLOAT_EQ((px_proj_cx(1) - px_proj(1)) / delta, px_jacc(1,2));
  cam->UpdateState(-dX_cx);

  Vec8 dX_cy;
  dX_cy << 0, 0, 0, delta, 0, 0, 0, 0;
  cam->UpdateState(dX_cy);
  Vec2 px_proj_cy = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_cy(0) - px_proj(0)) / delta, px_jacc(0,3));
  EXPECT_FLOAT_EQ((px_proj_cy(1) - px_proj(1)) / delta, px_jacc(1,3));
  cam->UpdateState(-dX_cy);

  Vec8 dX_w;
  dX_w << 0, 0, 0, 0, delta, 0, 0, 0;
  cam->UpdateState(dX_w);
  Vec2 px_proj_w = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_w(0) - px_proj(0)) / delta, px_jacc(0,4));
  EXPECT_FLOAT_EQ((px_proj_w(1) - px_proj(1)) / delta, px_jacc(1,4));
  cam->UpdateState(-dX_w);
}


TEST(CamerasAtan, atanSingularProjectUnproject) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["atan_singular"]);

  std::default_random_engine generator;
  std::uniform_real_distribution<number_t> distribution(0.0, 5.0);

  Vec2 px;
  px(0) = distribution(generator);
  px(1) = distribution(generator);

  Vec2 px_projected = cam->Project(px);
  Vec2 px2 = cam->UnProject(px_projected);

  EXPECT_FLOAT_EQ(px(0), px2(0));
  EXPECT_FLOAT_EQ(px(1), px2(1));
}


TEST(CamerasAtan, atanSingularProjectionJac) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["atan_singular"]);

  std::default_random_engine generator;
  std::uniform_real_distribution<number_t> distribution(0.0, 5.0);

  number_t delta = 1e-7;

  Vec2 px;
  Vec2 px_projected;
  Mat2 px_jac;
  px(0) = distribution(generator);
  px(1) = distribution(generator);

  px_projected = cam->Project(px, &px_jac);

  Vec2 px1 = px;
  px1(0) += delta;
  Vec2 px1_projected = cam->Project(px1);

  Vec2 px2 = px;
  px2(1) += delta;
  Vec2 px2_projected = cam->Project(px2);

  Vec2 approx_dx1 = (px1_projected - px_projected) / delta;
  Vec2 approx_dx2 = (px2_projected - px_projected) / delta;

  EXPECT_FLOAT_EQ(approx_dx1(0), px_jac(0,0));
  EXPECT_FLOAT_EQ(approx_dx1(1), px_jac(1,0));
  EXPECT_FLOAT_EQ(approx_dx2(0), px_jac(0,1));
  EXPECT_FLOAT_EQ(approx_dx2(1), px_jac(1,1));
}


TEST(CamerasAtan, atanSingularProjectionJacc) {
  auto cfg_ = LoadJson("src/test/camera_configs.json");
  CameraManager *cam = Camera::Create(cfg_["atan_singular"]);

  std::default_random_engine generator;
  std::uniform_real_distribution<number_t> distribution(0.0, 5.0);

  number_t delta = 1e-7;

  Vec2 px;
  Vec2 px_proj;
  Eigen::Matrix<number_t, 2, Eigen::Dynamic> px_jacc;
  px(0) = distribution(generator);
  px(1) = distribution(generator);

  px_proj = cam->Project(px, nullptr, &px_jacc);

  Vec9 Intrinsics = cam->GetIntrinsics();

  Vec8 dX_fx;
  dX_fx << delta, 0, 0, 0, 0, 0, 0, 0;
  cam->UpdateState(dX_fx);
  Vec2 px_proj_fx = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_fx(0) - px_proj(0)) / delta, px_jacc(0,0));
  EXPECT_FLOAT_EQ((px_proj_fx(1) - px_proj(1)) / delta, px_jacc(1,0));
  cam->UpdateState(-dX_fx);

  Vec8 dX_fy;
  dX_fy << 0, delta, 0, 0, 0, 0, 0, 0;
  cam->UpdateState(dX_fy);
  Vec2 px_proj_fy = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_fy(0) - px_proj(0)) / delta, px_jacc(0,1));
  EXPECT_FLOAT_EQ((px_proj_fy(1) - px_proj(1)) / delta, px_jacc(1,1));
  cam->UpdateState(-dX_fy);

  Vec8 dX_cx;
  dX_cx << 0, 0, delta, 0, 0, 0, 0, 0;
  cam->UpdateState(dX_cx);
  Vec2 px_proj_cx = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_cx(0) - px_proj(0)) / delta, px_jacc(0,2));
  EXPECT_FLOAT_EQ((px_proj_cx(1) - px_proj(1)) / delta, px_jacc(1,2));
  cam->UpdateState(-dX_cx);

  Vec8 dX_cy;
  dX_cy << 0, 0, 0, delta, 0, 0, 0, 0;
  cam->UpdateState(dX_cy);
  Vec2 px_proj_cy = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_cy(0) - px_proj(0)) / delta, px_jacc(0,3));
  EXPECT_FLOAT_EQ((px_proj_cy(1) - px_proj(1)) / delta, px_jacc(1,3));
  cam->UpdateState(-dX_cy);

  Vec8 dX_w;
  dX_w << 0, 0, 0, 0, delta, 0, 0, 0;
  cam->UpdateState(dX_w);
  Vec2 px_proj_w = cam->Project(px);
  EXPECT_FLOAT_EQ((px_proj_w(0) - px_proj(0)) / delta, px_jacc(0,4));
  EXPECT_FLOAT_EQ((px_proj_w(1) - px_proj(1)) / delta, px_jacc(1,4));
  cam->UpdateState(-dX_w);
}
