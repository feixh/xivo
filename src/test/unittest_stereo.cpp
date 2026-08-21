// Tests for the multi-camera registry and the fixed stereo rig geometry (M1).
//
// Run from the repository root, as the other camera tests do: the config paths
// below are relative to it.
#include <gtest/gtest.h>

#include <random>

#include "core.h"
#include "stereo.h"

using namespace Eigen;
using namespace xivo;

namespace {

// The generated TUM-VI stereo config is the one the system actually runs with,
// so testing against it also guards the converter's output.
const char *kStereoCfg = "cfg/tumvi_stereo.json";

Json::Value StereoCfg() {
  auto cfg = LoadJson(kStereoCfg);
  CHECK(!cfg.isNull()) << "could not load " << kStereoCfg
                       << "; run tests from the repository root";
  return cfg;
}

// Both cameras, created once for the whole binary. CameraManager's registry
// keeps the first camera installed in a slot, so creating them per-test would
// silently reuse whatever the first test happened to install.
struct Rig {
  Rig() {
    auto cfg = StereoCfg();
    cam0 = Camera::Create(cfg["camera_cfg"], 0);
    cam1 = Camera::Create(cfg["camera1_cfg"], 1);
    rig = StereoRig::Create(cfg["stereo_cfg"]);
  }
  CameraManager *cam0;
  CameraManager *cam1;
  StereoRig *rig;
};

const Rig &TheRig() {
  static Rig r;
  return r;
}

} // namespace

TEST(StereoRegistry, BothCamerasLoadAndAreDistinct) {
  const auto &r = TheRig();

  ASSERT_NE(r.cam0, nullptr);
  ASSERT_NE(r.cam1, nullptr);
  EXPECT_EQ(Camera::num_cameras(), 2);

  // instance() with no argument must remain the primary camera, since every
  // pre-existing monocular call site relies on that.
  EXPECT_EQ(Camera::instance(), r.cam0);
  EXPECT_EQ(Camera::instance(0), r.cam0);
  EXPECT_EQ(Camera::instance(1), r.cam1);
  EXPECT_NE(r.cam0, r.cam1);

  // The two TUM-VI cameras have genuinely different intrinsics; if they came
  // out equal the second slot would be a copy of the first.
  EXPECT_GT((r.cam0->GetIntrinsics() - r.cam1->GetIntrinsics()).norm(), 1e-3);

  // Out-of-range slots return null rather than reading past the registry.
  EXPECT_EQ(Camera::instance(2), nullptr);
  EXPECT_EQ(Camera::instance(-1), nullptr);
}

TEST(StereoRegistry, CreateIsIdempotentPerSlot) {
  const auto &r = TheRig();
  auto cfg = StereoCfg();
  // Re-creating a populated slot must not swap the camera out from under code
  // holding a pointer to it.
  EXPECT_EQ(Camera::Create(cfg["camera1_cfg"], 1), r.cam1);
  EXPECT_EQ(Camera::num_cameras(), 2);
}

TEST(StereoRegistry, ProjectUnprojectRoundTripBothCameras) {
  const auto &r = TheRig();

  std::mt19937 gen(42);
  // Normalized coordinates spanning roughly the fisheye's usable field.
  std::uniform_real_distribution<number_t> dist(-1.2, 1.2);

  for (CameraManager *cam : {r.cam0, r.cam1}) {
    for (int i = 0; i < 200; ++i) {
      Vec2 xc{dist(gen), dist(gen)};
      Vec2 xp = cam->Project(xc);
      Vec2 xc2 = cam->UnProject(xp);
      EXPECT_NEAR(xc(0), xc2(0), 1e-9) << "cam " << cam << " sample " << i;
      EXPECT_NEAR(xc(1), xc2(1), 1e-9) << "cam " << cam << " sample " << i;
    }
  }
}

TEST(StereoRig, BaselineAndExtrinsicsMatchCalibration) {
  const auto &r = TheRig();
  ASSERT_NE(r.rig, nullptr);
  EXPECT_TRUE(StereoRig::enabled());

  // TUM-VI's camchain gives a 101.09 mm baseline; a wrong unit or a transform
  // composed in the wrong direction would show up here first.
  EXPECT_NEAR(r.rig->baseline(), 0.10109, 1e-4);

  // gc0c1 and gc1c0 must be genuine inverses.
  SE3 id = r.rig->gc0c1() * r.rig->gc1c0();
  EXPECT_NEAR((id.rotationMatrix() - Mat3::Identity()).norm(), 0.0, 1e-12);
  EXPECT_NEAR(id.translation().norm(), 0.0, 1e-12);

  // Rc1c0 / Tc1c0 are the cached pieces of gc1c0; keep them in sync.
  EXPECT_NEAR((r.rig->Rc1c0() - r.rig->gc1c0().rotationMatrix()).norm(), 0.0,
              1e-12);
  EXPECT_NEAR((r.rig->Tc1c0() - r.rig->gc1c0().translation()).norm(), 0.0,
              1e-12);
  // Rotation is a valid rotation.
  EXPECT_NEAR((r.rig->Rc1c0().transpose() * r.rig->Rc1c0() - Mat3::Identity())
                  .norm(),
              0.0, 1e-12);
  EXPECT_NEAR(r.rig->Rc1c0().determinant(), 1.0, 1e-12);

  // ToCam1 agrees with gc1c0 applied directly.
  Vec3 Xc0{0.3, -0.2, 4.0};
  EXPECT_NEAR((r.rig->ToCam1(Xc0) - r.rig->gc1c0() * Xc0).norm(), 0.0, 1e-12);

  // The cameras are side-by-side, so nearly all of the baseline is along x.
  EXPECT_GT(std::abs(r.rig->Tc1c0()(0)) / r.rig->baseline(), 0.99);
}

TEST(StereoRig, TriangulateRecoversKnownDepth) {
  const auto &r = TheRig();

  std::mt19937 gen(7);
  std::uniform_real_distribution<number_t> lat(-1.0, 1.0);
  // Depths from just past the baseline out to where a 10 cm baseline stops
  // resolving well.
  std::uniform_real_distribution<number_t> depth(0.5, 20.0);

  for (int i = 0; i < 500; ++i) {
    number_t z = depth(gen);
    Vec3 Xc0{lat(gen) * z * 0.5, lat(gen) * z * 0.5, z};
    Vec3 Xc1 = r.rig->ToCam1(Xc0);
    ASSERT_GT(Xc1(2), 0.0);

    // Noise-free normalized observations.
    Vec2 xc0{Xc0(0) / Xc0(2), Xc0(1) / Xc0(2)};
    Vec2 xc1{Xc1(0) / Xc1(2), Xc1(1) / Xc1(2)};

    Vec3 Xhat;
    number_t gap = -1.0;
    ASSERT_TRUE(r.rig->Triangulate(xc0, xc1, &Xhat, &gap)) << "sample " << i;
    // Exact observations: the rays intersect and the point is recovered to
    // numerical precision, in metres -- this is the metric-scale property that
    // stereo buys over the monocular depth prior.
    EXPECT_NEAR(gap, 0.0, 1e-9) << "sample " << i;
    EXPECT_NEAR((Xhat - Xc0).norm(), 0.0, 1e-9)
        << "sample " << i << " true z=" << z << " got z=" << Xhat(2);
  }
}

TEST(StereoRig, TriangulateRejectsDegenerateAndBehindCamera) {
  const auto &r = TheRig();
  Vec3 Xhat;

  // Identical bearings: rays are parallel, so there is no parallax to use.
  Vec2 x{0.1, 0.05};
  EXPECT_FALSE(r.rig->Triangulate(x, x, &Xhat));

  // A pair whose rays meet behind the cameras. The baseline points along -x
  // from cam0 to cam1, so reversing the disparity sign puts the intersection
  // behind both.
  Vec3 Xbehind{0.2, 0.1, -3.0};
  Vec3 Xbehind1 = r.rig->ToCam1(Xbehind);
  Vec2 b0{Xbehind(0) / Xbehind(2), Xbehind(1) / Xbehind(2)};
  Vec2 b1{Xbehind1(0) / Xbehind1(2), Xbehind1(1) / Xbehind1(2)};
  EXPECT_FALSE(r.rig->Triangulate(b0, b1, &Xhat));
}

TEST(StereoRig, EpipolarResidualZeroForTrueCorrespondence) {
  const auto &r = TheRig();

  std::mt19937 gen(11);
  std::uniform_real_distribution<number_t> lat(-1.0, 1.0);
  std::uniform_real_distribution<number_t> depth(0.5, 20.0);

  for (int i = 0; i < 200; ++i) {
    number_t z = depth(gen);
    Vec3 Xc0{lat(gen) * z * 0.5, lat(gen) * z * 0.5, z};
    Vec3 Xc1 = r.rig->ToCam1(Xc0);
    Vec2 xc0{Xc0(0) / Xc0(2), Xc0(1) / Xc0(2)};
    Vec2 xc1{Xc1(0) / Xc1(2), Xc1(1) / Xc1(2)};

    EXPECT_NEAR(r.rig->EpipolarResidual(xc0, xc1), 0.0, 1e-12)
        << "sample " << i;
  }
}

TEST(StereoRig, EpipolarResidualGrowsWithOffEpipolarError) {
  const auto &r = TheRig();

  Vec3 Xc0{0.4, -0.3, 5.0};
  Vec3 Xc1 = r.rig->ToCam1(Xc0);
  Vec2 xc0{Xc0(0) / Xc0(2), Xc0(1) / Xc0(2)};
  Vec2 xc1{Xc1(0) / Xc1(2), Xc1(1) / Xc1(2)};

  // Displacing the right observation perpendicular to the epipolar direction
  // must raise the residual monotonically, and the residual is the sine of the
  // angular miss -- so a 1e-3 normalized offset gives ~1e-3 rad, which is what
  // makes the gating threshold interpretable.
  number_t prev = r.rig->EpipolarResidual(xc0, xc1);
  for (number_t d : {1e-4, 1e-3, 1e-2, 1e-1}) {
    Vec2 off = xc1;
    off(1) += d; // the baseline is nearly along x, so y is across the epiline
    number_t res = r.rig->EpipolarResidual(xc0, off);
    EXPECT_GT(res, prev) << "offset " << d;
    prev = res;
  }
  Vec2 off = xc1;
  off(1) += 1e-3;
  EXPECT_NEAR(r.rig->EpipolarResidual(xc0, off), 1e-3, 2e-4);
}
