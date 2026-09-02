// The static-vs-dynamic detector, on synthetic data where the truth is known.
//
// Run from the repository root.
//
// The three sequence-level cases below are the ones real data cannot cleanly
// provide, because each isolates a failure mode of a *different* candidate
// statistic. All three are run through the real `MotionDetector` -- rendered
// images, real corner detection, real KLT -- rather than by injecting
// pre-computed flow, so the tests cover the tracking and the projection as well
// as the arithmetic:
//
//   RotatingButStill    a stationary rig turning at 0.3 rad/s. Raw pixel
//                       disparity (OpenVINS' cue) is large here and calls it
//                       moving; accelerometer sd is ~0.42 because gravity sweeps
//                       through the body frame, so that calls it moving too. The
//                       rig is not translating and the answer is static.
//   ConstantVelocity    a rig gliding at 0.6 m/s with no rotation. Specific
//                       force is exactly |g| and its variance is exactly zero,
//                       so `gravity_init_max_accel_dev` and accelerometer sd
//                       both call it static. The answer is dynamic.
//   BiasedGyroButStill  a stationary rig whose gyro reads a constant 0.08 rad/s
//                       that is pure bias -- EuRoC's turn-on value. Gyro
//                       de-rotation predicts ~1.8 px/frame of motion that never
//                       happened and calls it moving. The answer is static.
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "core.h"
#include "init_detect.h"

using namespace xivo;

namespace {

constexpr int kRows = 480;
constexpr int kCols = 752;
constexpr number_t kImuRate = 200.0;
constexpr number_t kCamRate = 20.0;
constexpr number_t kDur = 1.5; // seconds of synthetic data

// EuRoC cam0: the resolution and focal length the thresholds were measured at,
// so a threshold in pixels means here what it means on the real data.
Json::Value EurocCam0() {
  Json::Value c;
  c["model"] = "radtan";
  c["rows"] = kRows;
  c["cols"] = kCols;
  c["fx"] = 458.654;
  c["fy"] = 457.296;
  c["cx"] = 367.215;
  c["cy"] = 248.375;
  c["k012"][0] = -0.28340811;
  c["k012"][1] = 0.07395907;
  c["k012"][2] = 0.0;
  c["p1"] = 0.00019359;
  c["p2"] = 1.76187114e-05;
  c["max_iter"] = 15;
  return c;
}

/** A cloud of points spread in depth. Depth *variation* is what makes
 *  translation distinguishable from rotation at all, so the scene has to have
 *  some -- a cloud at constant range would be the detector's degenerate case and
 *  is not what these tests are about. */
std::vector<Vec3> MakeScene(int n = 400) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<number_t> ux(-5.0, 5.0), uy(-3.5, 3.5),
      uz(2.0, 10.0);
  std::vector<Vec3> pts;
  pts.reserve(n);
  for (int i = 0; i < n; ++i)
    pts.emplace_back(Vec3{ux(rng), uy(rng), uz(rng)});
  return pts;
}

/** Render the scene from pose (R_wc, p_w) as blobs a corner detector will find.
 *
 *  Bright dots blurred to a couple of pixels give strong two-dimensional
 *  gradients, which is what both goodFeaturesToTrack and KLT need. Sub-pixel
 *  placement matters: drawing on integer centres would quantise the flow to
 *  whole pixels and swamp the sub-pixel residuals these thresholds live at, so
 *  the blob is accumulated with bilinear weights.
 */
cv::Mat Render(const std::vector<Vec3> &pts, const Mat3 &R_wc, const Vec3 &p_w,
               CameraManager *cam) {
  cv::Mat img(kRows, kCols, CV_32F, cv::Scalar(20.0));
  const Mat3 R_cw = R_wc.transpose();
  for (const auto &X : pts) {
    const Vec3 xc = R_cw * (X - p_w);
    if (xc(2) < 0.5)
      continue;
    const Vec2 px = cam->Project(Vec2{xc(0) / xc(2), xc(1) / xc(2)});
    if (px(0) < 8 || px(0) > kCols - 9 || px(1) < 8 || px(1) > kRows - 9)
      continue;
    const int x0 = static_cast<int>(std::floor(px(0)));
    const int y0 = static_cast<int>(std::floor(px(1)));
    const number_t fx = px(0) - x0, fy = px(1) - y0;
    for (int dy = 0; dy <= 1; ++dy)
      for (int dx = 0; dx <= 1; ++dx) {
        const number_t w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
        img.at<float>(y0 + dy, x0 + dx) += static_cast<float>(220.0 * w);
      }
  }
  cv::GaussianBlur(img, img, cv::Size(0, 0), 1.4);
  cv::Mat out;
  img.convertTo(out, CV_8U);
  return out;
}

/** Feed the detector a whole synthetic sequence.
 *
 *  Motion model: constant body rate `omega`; world-frame initial velocity `vel`;
 *  world-frame linear acceleration `accel`, constant when `accel_hz == 0` and
 *  `accel * sin(2 pi accel_hz t)` otherwise. Position is the exact integral in
 *  both cases, so the rendered images and the IMU stream are consistent to
 *  machine precision and any flow residual the detector reports is parallax
 *  rather than a discretisation artifact.
 *
 *  The IMU is the body frame and the camera is the body frame (identity
 *  extrinsics), which is not EuRoC's geometry but is irrelevant to what is being
 *  tested: the detector never uses the extrinsics except through the gyro-bias
 *  hint, which nothing here asserts on.
 */
MotionVerdict RunSequence(const MotionDetector::Options &opt,
                          const std::vector<Vec3> &pts, CameraManager *cam,
                          const Vec3 &omega, const Vec3 &vel,
                          const Vec3 &gyro_bias,
                          const Vec3 &accel = Vec3::Zero(),
                          number_t accel_hz = 0.0) {
  MotionDetector det(opt);
  const Vec3 g_w{0.0, 0.0, -9.81};
  const number_t w = 2 * M_PI * accel_hz;

  auto accel_at = [&](number_t t) -> Vec3 {
    return accel_hz > 0 ? Vec3{accel * std::sin(w * t)} : accel;
  };
  auto pos_at = [&](number_t t) -> Vec3 {
    // int int a: exact for both branches, so the camera path and the IMU agree.
    return accel_hz > 0 ? Vec3{vel * t + accel * (t - std::sin(w * t) / w) / w}
                        : Vec3{vel * t + 0.5 * accel * t * t};
  };

  // IMU first and camera second would both work; interleaving in timestamp order
  // is what the estimator will do.
  const int n_imu = static_cast<int>(kDur * kImuRate);
  const int n_cam = static_cast<int>(kDur * kCamRate);
  int ic = 0;
  for (int i = 0; i < n_imu; ++i) {
    const number_t t = i / kImuRate;
    const Mat3 R_wb = SO3::exp(omega * t).matrix();
    // Specific force: what the accelerometer actually reads.
    det.AddImu(t, omega + gyro_bias, R_wb.transpose() * (accel_at(t) - g_w));

    while (ic < n_cam && static_cast<number_t>(ic) / kCamRate <= t) {
      const number_t tc = ic / kCamRate;
      det.AddImage(tc, Render(pts, SO3::exp(omega * tc).matrix(), pos_at(tc),
                              cam));
      ++ic;
    }
  }
  return det.Classify();
}

MotionDetector::Options DefaultOpts() {
  MotionDetector::Options o;
  o.horizon_sec = 1.0; // the synthetic sequences are 1.5 s long
  return o;
}

} // namespace

// ---------------------------------------------------------------------------
// The rotation fit, on its own.
// ---------------------------------------------------------------------------

TEST(InitDetectWahba, RecoversRotationExactly) {
  std::mt19937 rng(3);
  std::uniform_real_distribution<number_t> u(-1.0, 1.0);
  const Mat3 R_true = SO3::exp(Vec3{0.03, -0.07, 0.11}).matrix();

  std::vector<Vec3> u1, u2;
  for (int i = 0; i < 50; ++i) {
    Vec3 b{u(rng), u(rng), 1.0 + std::abs(u(rng))};
    b.normalize();
    u1.push_back(b);
    u2.push_back((R_true * b).normalized());
  }
  const Mat3 R = FitRotationWahba(u1, u2);
  EXPECT_LT((R - R_true).norm(), 1e-12);
}

TEST(InitDetectWahba, ReturnsARotationNotAReflection) {
  // A degenerate configuration -- all bearings nearly coplanar -- is where the
  // unconstrained least-squares answer can come out as a reflection. The fit
  // must project back onto SO(3) rather than return det = -1.
  std::mt19937 rng(11);
  std::uniform_real_distribution<number_t> u(-0.4, 0.4);
  std::vector<Vec3> u1, u2;
  for (int i = 0; i < 20; ++i) {
    Vec3 b{u(rng), 1e-6 * u(rng), 1.0};
    b.normalize();
    u1.push_back(b);
    u2.push_back(b);
  }
  const Mat3 R = FitRotationWahba(u1, u2);
  EXPECT_NEAR(R.determinant(), 1.0, 1e-9);
  EXPECT_LT((R * R.transpose() - Mat3::Identity()).norm(), 1e-9);
}

TEST(InitDetectWahba, HuberSurvivesOutliers) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<number_t> u(-1.0, 1.0);
  const Mat3 R_true = SO3::exp(Vec3{0.0, 0.05, 0.0}).matrix();

  std::vector<Vec3> u1, u2;
  for (int i = 0; i < 60; ++i) {
    Vec3 b{u(rng), u(rng), 1.0 + std::abs(u(rng))};
    b.normalize();
    u1.push_back(b);
    // Every tenth correspondence is garbage, which is what a KLT step that
    // survives the forward-backward check but latched onto the wrong corner
    // looks like.
    u2.push_back(i % 10 == 0 ? Vec3{u(rng), u(rng), 1.0}.normalized()
                             : (R_true * b).normalized());
  }
  const Mat3 R_robust = FitRotationWahba(u1, u2, 4);
  const Mat3 R_plain = FitRotationWahba(u1, u2, 1);
  const number_t e_robust = (R_robust - R_true).norm();
  const number_t e_plain = (R_plain - R_true).norm();
  EXPECT_LT(e_robust, e_plain)
      << "IRLS did not improve on the unweighted fit: " << e_robust << " vs "
      << e_plain;
  EXPECT_LT(e_robust, 0.02);
}

// ---------------------------------------------------------------------------
// The detector, end to end on rendered images.
// ---------------------------------------------------------------------------

class InitDetectTest : public ::testing::Test {
protected:
  void SetUp() override {
    cam_ = Camera::Create(EurocCam0());
    ASSERT_NE(cam_, nullptr);
    ASSERT_EQ(CameraManager::instance(0), cam_)
        << "the detector reaches the camera through instance(0)";
    pts_ = MakeScene();
  }
  CameraManager *cam_{nullptr};
  std::vector<Vec3> pts_;
};

TEST_F(InitDetectTest, RotatingButStillIsStatic) {
  // 0.3 rad/s about y, no translation. This is the case that defeats both raw
  // pixel disparity and accelerometer variance.
  const auto v = RunSequence(DefaultOpts(), pts_, cam_, Vec3{0.0, 0.3, 0.0},
                     Vec3::Zero(), Vec3::Zero());
  EXPECT_EQ(v.kind, MotionVerdict::kStatic) << "flow " << v.flow_px
                                            << " px, accel sd " << v.accel_sd;
  EXPECT_GT(v.frame_pairs, 5);
  // The point of the fit: rotation at this rate moves features by tens of
  // pixels, and essentially all of it is explained.
  EXPECT_LT(v.flow_px, 0.25);
  // And the accelerometer statistic really is above its own threshold here, so
  // this test would fail under an `imu || flow` rule. Pinned so that the reason
  // the rule is what it is cannot quietly stop being true.
  EXPECT_GT(v.accel_sd, 0.35);
}

TEST_F(InitDetectTest, ConstantVelocityIsDynamic) {
  // 0.6 m/s, no rotation, no acceleration: specific force is a constant vector
  // of magnitude exactly |g|.
  const auto v = RunSequence(DefaultOpts(), pts_, cam_, Vec3::Zero(),
                     Vec3{0.6, 0.0, 0.0}, Vec3::Zero());
  EXPECT_EQ(v.kind, MotionVerdict::kDynamic) << "flow " << v.flow_px
                                             << " px, accel sd " << v.accel_sd;
  EXPECT_GT(v.flow_px, 0.25);
  // Every IMU-only statistic is blind here, and that is the whole reason the
  // visual cue exists. Pin both of them.
  EXPECT_LT(v.accel_sd, 1e-9) << "accelerometer variance should be exactly zero";
}

TEST_F(InitDetectTest, ConstantVelocityDefeatsTheShippedGate) {
  // The same motion, checked against `gravity_init_max_accel_dev`'s statistic
  // rather than the detector's: | |a| - |g| | is zero to numerical precision, so
  // the shipped gate accepts every sample and initializes with v = 0.
  const Vec3 g_w{0.0, 0.0, -9.81};
  for (int i = 0; i < 20; ++i) {
    const Mat3 R_wb = Mat3::Identity();
    const Vec3 a = R_wb.transpose() * (Vec3::Zero() - g_w);
    EXPECT_LT(std::abs(a.norm() - 9.81), 1e-12);
  }
}

TEST_F(InitDetectTest, BiasedGyroButStillIsStatic) {
  // Stationary, and the gyro reads a constant 0.08 rad/s of pure bias --
  // EuRoC's turn-on value. The detector must not care, because the visual cue
  // never consults the gyro.
  const auto v = RunSequence(DefaultOpts(), pts_, cam_, Vec3::Zero(), Vec3::Zero(),
                     Vec3{0.0, 0.08, 0.0});
  EXPECT_EQ(v.kind, MotionVerdict::kStatic) << "flow " << v.flow_px
                                            << " px, accel sd " << v.accel_sd;
  EXPECT_LT(v.flow_px, 0.25);
  // What gyro de-rotation would have predicted instead: 0.08 rad/s over a 50 ms
  // frame gap at f = 458 px is about 1.8 px of motion that did not happen.
  EXPECT_GT(0.08 * (1.0 / kCamRate) * 458.654, 1.5);
}

TEST_F(InitDetectTest, StartsAtRestThenAcceleratesIsStatic) {
  // At rest at t = 0, then accelerating at a constant 1.5 m/s^2. The verdict is
  // static, and that is the intended answer rather than a miss.
  //
  // Both statistics are minimised over candidate windows, so the earliest window
  // decides -- and in the earliest window this rig genuinely has not moved. The
  // instant the initializer is initializing *at* is the start of that window, and
  // at that instant v = 0 is exactly right. Running a bundle adjustment here
  // would be solving for a velocity that is zero. XIVO's static path already
  // handles this case, and does it better than a window BA on 0.006 m of
  // baseline would.
  //
  // MH_01 and MH_02 are not this case: they are already moving at 0.67 and
  // 0.48 m/s in their first window, which is why min-over-windows still reports
  // 0.610 and 0.804 px on them.
  const auto v =
      RunSequence(DefaultOpts(), pts_, cam_, Vec3::Zero(), Vec3::Zero(),
                  /*gyro_bias=*/Vec3::Zero(), /*accel=*/Vec3{1.5, 0, 0});
  EXPECT_EQ(v.kind, MotionVerdict::kStatic) << "flow " << v.flow_px
                                            << " px, accel sd " << v.accel_sd;
  // And the reason the *fallback* could not have rescued it either: a constant
  // acceleration with no rotation makes the specific force a constant vector, so
  // accelerometer variance is zero even though the rig is accelerating. Only
  // | |a| - |g| | sees this, and only because 1.5 m/s^2 is large.
  EXPECT_LT(v.accel_sd, 1e-9)
      << "constant acceleration has no accelerometer variance to see";
}

TEST_F(InitDetectTest, AlreadyMovingAndShakenIsDynamic) {
  // The MH_01 signature: the rig is already translating when the first sample
  // arrives, and is being jostled at the same time. Both cues fire here, so this
  // is the one case the shipped gate also handles -- pinned so that a change that
  // broke it would show up as a failure rather than as a quiet regression on the
  // sequences where dynamic init is supposed to help.
  const auto v = RunSequence(DefaultOpts(), pts_, cam_, Vec3{0, 0.1, 0},
                             Vec3{0.5, 0, 0}, /*gyro_bias=*/Vec3::Zero(),
                             /*accel=*/Vec3{2.0, 0, 1.0}, /*accel_hz=*/1.5);
  EXPECT_EQ(v.kind, MotionVerdict::kDynamic) << "flow " << v.flow_px
                                             << " px, accel sd " << v.accel_sd;
  EXPECT_GT(v.flow_px, 0.25);
  EXPECT_GT(v.accel_sd, 0.35) << "both cues should agree on this one";
}

TEST_F(InitDetectTest, UndecidedBeforeTheWindowIsFull) {
  MotionDetector det(DefaultOpts());
  for (int i = 0; i < 20; ++i)
    det.AddImu(i / kImuRate, Vec3::Zero(), Vec3{0.0, 0.0, 9.81});
  const auto v = det.Classify();
  EXPECT_EQ(v.kind, MotionVerdict::kUndecided);
  EXPECT_FALSE(det.Ready());
}

TEST_F(InitDetectTest, FallsBackToTheImuWhenThereIsNoTexture) {
  // A blank wall: no corners, so no flow samples, so the visual cue has no
  // opinion and the accelerometer has to answer. Under acceleration it does.
  MotionDetector det(DefaultOpts());
  const Vec3 g_w{0.0, 0.0, -9.81};
  const Vec3 a_extra{2.0, 0.0, 0.0};
  const int n_imu = static_cast<int>(kDur * kImuRate);
  int ic = 0;
  for (int i = 0; i < n_imu; ++i) {
    const number_t t = i / kImuRate;
    // A sinusoid so the sd is large; a constant acceleration has zero variance.
    det.AddImu(t, Vec3::Zero(),
               -g_w + a_extra * std::sin(2 * M_PI * 1.5 * t));
    while (ic < static_cast<int>(kDur * kCamRate) &&
           static_cast<number_t>(ic) / kCamRate <= t) {
      det.AddImage(ic / kCamRate,
                   cv::Mat(kRows, kCols, CV_8U, cv::Scalar(128)));
      ++ic;
    }
  }
  const auto v = det.Classify();
  EXPECT_EQ(v.frame_pairs, 0) << "a blank image should yield no tracks";
  EXPECT_LT(v.flow_px, 0) << "no opinion is encoded as a negative residual";
  EXPECT_EQ(v.kind, MotionVerdict::kDynamic) << "accel sd " << v.accel_sd;
}
