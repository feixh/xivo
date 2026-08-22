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

// ---------------------------------------------------------------------------
// TriangulateFromPixels: the pixel-space entry point used for depth seeding
// (M4). Unlike Triangulate it also reports a log-depth standard deviation,
// which is what the EKF actually consumes as P_(2,2).
// ---------------------------------------------------------------------------

TEST(StereoInit, RecoversKnownDepthFromPixelObservations) {
  const auto &r = TheRig();

  std::mt19937 gen(23);
  std::uniform_real_distribution<number_t> lat(-0.8, 0.8);
  std::uniform_real_distribution<number_t> depth(0.5, 8.0);

  int tested = 0;
  for (int i = 0; i < 500; ++i) {
    number_t z = depth(gen);
    Vec3 Xc0{lat(gen) * z * 0.5, lat(gen) * z * 0.5, z};
    Vec3 Xc1 = r.rig->ToCam1(Xc0);
    ASSERT_GT(Xc1(2), 0.0);

    // Round-trip through the actual fisheye projections, so this exercises the
    // same UnProject path the estimator uses.
    Vec2 xp0 = r.cam0->Project(Vec2{Xc0(0) / Xc0(2), Xc0(1) / Xc0(2)});
    Vec2 xp1 = r.cam1->Project(Vec2{Xc1(0) / Xc1(2), Xc1(1) / Xc1(2)});
    if (xp0(0) < 0 || xp0(0) >= 512 || xp0(1) < 0 || xp0(1) >= 512) {
      continue; // outside the image; the tracker would never produce this
    }

    Vec3 Xhat;
    number_t std_z = -1.0, gap = -1.0;
    ASSERT_TRUE(r.rig->TriangulateFromPixels(xp0, xp1, 0.5, &Xhat, &std_z,
                                             &gap))
        << "sample " << i;
    EXPECT_NEAR(gap, 0.0, 1e-8) << "sample " << i;
    // Metric depth, recovered to numerical precision from noise-free pixels.
    EXPECT_NEAR(Xhat(2), z, 1e-6 * z) << "sample " << i;
    EXPECT_GT(std_z, 0.0) << "sample " << i;
    ++tested;
  }
  EXPECT_GT(tested, 300) << "too many samples skipped; the sampler is off";
}

TEST(StereoInit, ReturnedPointLiesOnTheLeftRay) {
  const auto &r = TheRig();

  // `Feature::Initialize` pairs the returned depth with the *left* bearing
  // UnProject(back()), so a point off the left ray would name a different 3D
  // point than the one that was triangulated. With imperfect (rounded) pixels
  // the two rays no longer intersect, which is exactly when this matters.
  std::mt19937 gen(29);
  std::uniform_real_distribution<number_t> lat(-0.6, 0.6);
  std::uniform_real_distribution<number_t> depth(0.8, 6.0);

  for (int i = 0; i < 200; ++i) {
    number_t z = depth(gen);
    Vec3 Xc0{lat(gen) * z * 0.5, lat(gen) * z * 0.5, z};
    Vec3 Xc1 = r.rig->ToCam1(Xc0);
    Vec2 xp0 = r.cam0->Project(Vec2{Xc0(0) / Xc0(2), Xc0(1) / Xc0(2)});
    Vec2 xp1 = r.cam1->Project(Vec2{Xc1(0) / Xc1(2), Xc1(1) / Xc1(2)});
    if (xp0(0) < 0 || xp0(0) >= 512 || xp0(1) < 0 || xp0(1) >= 512) {
      continue;
    }
    // Round the right pixel to integers: a realistic, non-intersecting pair.
    xp1 = Vec2{std::round(xp1(0)), std::round(xp1(1))};

    Vec3 Xhat;
    number_t std_z, gap;
    if (!r.rig->TriangulateFromPixels(xp0, xp1, 0.5, &Xhat, &std_z, &gap)) {
      continue;
    }
    Vec2 xc0 = r.cam0->UnProject(xp0);
    // Xhat must be exactly Xhat(2) * (xc0.x, xc0.y, 1).
    EXPECT_NEAR(Xhat(0), Xhat(2) * xc0(0), 1e-9 * z) << "sample " << i;
    EXPECT_NEAR(Xhat(1), Xhat(2) * xc0(1), 1e-9 * z) << "sample " << i;
  }
}

TEST(StereoInit, LogDepthStdGrowsWithDepthAndMatchesClosedForm) {
  const auto &r = TheRig();

  // In the image centre the equidistant fisheye is close to a pinhole, so the
  // textbook stereo formula applies there and gives an independent check on the
  // numerically propagated uncertainty:
  //     sigma_z = z^2 sigma_d / (f b)   =>   sigma_logz = z sigma_d / (f b)
  // Away from the centre the effective focal length changes, which is precisely
  // why the implementation propagates numerically instead of using the formula.
  const number_t sigma_px = 0.5;
  const number_t b = r.rig->baseline();
  const number_t f = r.cam0->GetIntrinsics()(0); // fx
  ASSERT_GT(f, 100.0);

  number_t prev = -1.0;
  for (number_t z : {1.0, 2.0, 3.0, 4.0, 6.0}) {
    Vec3 Xc0{0.0, 0.0, z};
    Vec3 Xc1 = r.rig->ToCam1(Xc0);
    Vec2 xp0 = r.cam0->Project(Vec2{0.0, 0.0});
    Vec2 xp1 = r.cam1->Project(Vec2{Xc1(0) / Xc1(2), Xc1(1) / Xc1(2)});

    Vec3 Xhat;
    number_t std_z, gap;
    ASSERT_TRUE(
        r.rig->TriangulateFromPixels(xp0, xp1, sigma_px, &Xhat, &std_z, &gap))
        << "z=" << z;

    // Monotone in depth: a 10 cm baseline resolves near features far better
    // than far ones, and the EKF must be told so.
    EXPECT_GT(std_z, prev) << "z=" << z;
    prev = std_z;

    const number_t expected = z * sigma_px / (f * b);
    // 15% agreement: the closed form assumes a pinhole and a purely horizontal
    // baseline, neither of which is exactly true here.
    EXPECT_NEAR(std_z, expected, 0.15 * expected) << "z=" << z;
  }
}

TEST(StereoInit, RejectsZeroDisparityAndBehindCamera) {
  const auto &r = TheRig();
  Vec3 Xhat;
  number_t std_z, gap;

  // Identical pixels in both images: zero disparity, i.e. a point at infinity.
  // There is no finite depth to seed, so this must fail rather than return a
  // huge one.
  Vec2 xp{256.0, 256.0};
  EXPECT_FALSE(r.rig->TriangulateFromPixels(xp, xp, 0.5, &Xhat, &std_z, &gap));

  // Disparity of the wrong sign puts the intersection behind the cameras.
  Vec3 Xbehind{0.2, 0.1, -3.0};
  Vec3 Xbehind1 = r.rig->ToCam1(Xbehind);
  Vec2 p0 = r.cam0->Project(Vec2{Xbehind(0) / Xbehind(2),
                                 Xbehind(1) / Xbehind(2)});
  Vec2 p1 = r.cam1->Project(Vec2{Xbehind1(0) / Xbehind1(2),
                                 Xbehind1(1) / Xbehind1(2)});
  EXPECT_FALSE(r.rig->TriangulateFromPixels(p0, p1, 0.5, &Xhat, &std_z, &gap));
}

TEST(StereoInit, GapIsReportedForAnInconsistentMatch) {
  const auto &r = TheRig();

  // A right observation displaced *across* the epipolar line produces rays that
  // cannot meet. `gap` is what lets the estimator reject such a match even
  // though the tracker's epipolar gate has a finite threshold.
  Vec3 Xc0{0.3, -0.2, 3.0};
  Vec3 Xc1 = r.rig->ToCam1(Xc0);
  Vec2 xp0 = r.cam0->Project(Vec2{Xc0(0) / Xc0(2), Xc0(1) / Xc0(2)});
  Vec2 xp1 = r.cam1->Project(Vec2{Xc1(0) / Xc1(2), Xc1(1) / Xc1(2)});

  Vec3 Xhat;
  number_t std_z, gap_true = -1.0, gap_off = -1.0;
  ASSERT_TRUE(
      r.rig->TriangulateFromPixels(xp0, xp1, 0.5, &Xhat, &std_z, &gap_true));
  EXPECT_NEAR(gap_true, 0.0, 1e-8);

  Vec2 xp1_off{xp1(0), xp1(1) + 5.0}; // 5 px across the epipolar line
  ASSERT_TRUE(r.rig->TriangulateFromPixels(xp0, xp1_off, 0.5, &Xhat, &std_z,
                                           &gap_off));
  EXPECT_GT(gap_off, 1e-3) << "an off-epipolar match must show a nonzero gap";
}
