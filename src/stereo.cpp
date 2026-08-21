#include "stereo.h"

#include "glog/logging.h"

#include "core.h"  // the `Camera` alias for CameraManager

#include "utils.h"

namespace xivo {

std::unique_ptr<StereoRig> StereoRig::instance_ = nullptr;

StereoRig *StereoRig::Create(const Json::Value &cfg) {
  if (!instance_) {
    instance_ = std::unique_ptr<StereoRig>(new StereoRig(cfg));
  }
  return instance_.get();
}

StereoRig::StereoRig(const Json::Value &cfg) {
  // Two accepted spellings of the left->right leg:
  //
  //   "T_c1c0": [[...4x4 row-major...]]  -- kalibr's `T_cn_cnm1`, i.e. the
  //             transform taking a point from cam0's frame to cam1's frame.
  //             This is what TUM-VI's dso/camchain.yaml provides verbatim.
  //   "Wc0c1" / "Tc0c1"  -- rotation vector + translation for the pose of cam1
  //             in cam0's frame, matching the "Wbc"/"Tbc" style used elsewhere
  //             in these configs.
  if (cfg.isMember("T_c1c0")) {
    Mat3 R = GetMatrixFromJson<number_t, 3, 3>(cfg, "T_c1c0",
                                               JsonMatLayout::RowMajor);
    Vec3 T;
    for (int i = 0; i < 3; ++i) {
      T(i) = cfg["T_c1c0"][i][3].asDouble();
    }
    if (!Sophus::isOrthogonal(R) || R.determinant() <= 0.0) {
      LOG(WARNING) << "stereo: T_c1c0 rotation is not in SO(3); projecting";
      gc1c0_ = SE3(SO3::fitToSO3(R), T);
    } else {
      gc1c0_ = SE3(SO3(R), T);
    }
    gc0c1_ = gc1c0_.inverse();
  } else if (cfg.isMember("Wc0c1") && cfg.isMember("Tc0c1")) {
    gc0c1_ = SE3(SO3::exp(GetVectorFromJson<number_t, 3>(cfg, "Wc0c1")),
                 GetVectorFromJson<number_t, 3>(cfg, "Tc0c1"));
    gc1c0_ = gc0c1_.inverse();
  } else {
    LOG(FATAL) << "stereo config needs either \"T_c1c0\" (kalibr T_cn_cnm1, "
                  "4x4 row-major) or both \"Wc0c1\" and \"Tc0c1\"";
  }

  Rc1c0_ = gc1c0_.rotationMatrix();
  Tc1c0_ = gc1c0_.translation();
  E_ = SO3::hat(Tc1c0_) * Rc1c0_;

  LOG(INFO) << "stereo rig: baseline=" << baseline() << " m";

  // A rig whose cameras sit on top of each other, or metres apart, means the
  // config is wrong (units, or a transform composed in the wrong direction).
  // Better to fail loudly here than to silently produce garbage depths.
  if (baseline() < 1e-3 || baseline() > 1.0) {
    LOG(FATAL) << "stereo baseline of " << baseline()
               << " m is implausible; check the sign/direction convention of "
                  "T_c1c0 (or Wc0c1/Tc0c1) in the config";
  }
}

bool StereoRig::Triangulate(const Vec2 &xc0, const Vec2 &xc1, Vec3 *Xc0,
                            number_t *gap) const {
  // Bearing vectors (not normalized to unit length -- the (x, y, 1) form is
  // what the rest of the codebase uses and the algebra below does not care).
  Vec3 b0{xc0(0), xc0(1), 1.0};
  // Camera 1's ray, rotated into camera 0's frame.
  Vec3 b1 = gc0c1_.rotationMatrix() * Vec3{xc1(0), xc1(1), 1.0};
  // Camera 1's optical centre in camera 0's frame.
  const Vec3 &c1 = gc0c1_.translation();

  // Closest approach of the two rays  X = s * b0  and  X = c1 + t * b1.
  // Solve the 2x2 normal equations for (s, t).
  number_t b00 = b0.dot(b0);
  number_t b01 = b0.dot(b1);
  number_t b11 = b1.dot(b1);
  number_t det = b00 * b11 - b01 * b01;

  // det -> 0 means the rays are parallel: no usable parallax. Scale the
  // threshold by the ray magnitudes so it is a genuine angular test rather
  // than an absolute one.
  if (!(det > 1e-12 * b00 * b11)) {
    return false;
  }

  number_t r0 = b0.dot(c1);
  number_t r1 = b1.dot(c1);
  number_t s = (b11 * r0 - b01 * r1) / det;
  number_t t = (b01 * r0 - b00 * r1) / det;

  Vec3 P0 = s * b0;         // closest point on camera 0's ray
  Vec3 P1 = c1 + t * b1;    // closest point on camera 1's ray

  if (gap) {
    *gap = (P0 - P1).norm();
  }

  Vec3 X = 0.5 * (P0 + P1);

  // Reject points behind either camera.
  if (!(X(2) > 0.0)) {
    return false;
  }
  if (!(ToCam1(X)(2) > 0.0)) {
    return false;
  }
  if (!X.allFinite()) {
    return false;
  }

  *Xc0 = X;
  return true;
}

bool StereoRig::TriangulateFromPixels(const Vec2 &xp0, const Vec2 &xp1,
                                      number_t sigma_px, Vec3 *Xc0,
                                      number_t *log_depth_std,
                                      number_t *gap) const {
  auto cam0 = Camera::instance(0);
  auto cam1 = Camera::instance(1);
  if (cam0 == nullptr || cam1 == nullptr) {
    return false;
  }

  const Vec2 xc0 = cam0->UnProject(xp0);

  Vec3 X;
  if (!Triangulate(xc0, cam1->UnProject(xp1), &X, gap)) {
    return false;
  }

  // `Triangulate` returns the midpoint of the two rays' closest approach, which
  // in general lies *off* the left ray by half the gap. The consumer
  // (`Feature::Initialize`) pairs the depth we return with the left bearing
  // `UnProject(back())`, so the point has to lie on that same ray -- otherwise
  // the seeded (bearing, depth) pair names a different 3D point than the one we
  // triangulated. Project X onto ray 0, keeping all three components coherent.
  const Vec3 b0{xc0(0), xc0(1), 1.0};
  auto onto_left_ray = [&b0](const Vec3 &P) {
    return Vec3{b0 * (b0.dot(P) / b0.squaredNorm())};
  };
  X = onto_left_ray(X);

  if (log_depth_std) {
    // Displace the right observation along x. For a rig whose baseline is
    // essentially horizontal -- TUM-VI's is 99.9% along x, checked by the M1
    // rig test -- x is the disparity direction, so this perturbs the quantity
    // that actually carries the depth information. Perturbing along y instead
    // would mostly slide the point along the epipolar line and understate the
    // depth uncertainty.
    Vec2 xp1_pert{xp1(0) + sigma_px, xp1(1)};
    Vec3 X_pert;
    if (!Triangulate(xc0, cam1->UnProject(xp1_pert), &X_pert, nullptr)) {
      // The perturbed ray pair is degenerate even though the nominal one was
      // not: the feature sits at the edge of usable parallax, so its depth is
      // not trustworthy. Report failure rather than a fabricated uncertainty.
      return false;
    }
    // Same ray convention for both, so the difference is a pure depth change.
    number_t z = X(2), z_pert = onto_left_ray(X_pert)(2);
    if (!(z > 0.0) || !(z_pert > 0.0)) {
      return false;
    }
    *log_depth_std = std::abs(std::log(z_pert) - std::log(z));
  }

  *Xc0 = X;
  return true;
}

number_t StereoRig::EpipolarResidual(const Vec2 &xc0, const Vec2 &xc1) const {
  Vec3 b0 = Vec3{xc0(0), xc0(1), 1.0}.normalized();
  Vec3 b1 = Vec3{xc1(0), xc1(1), 1.0}.normalized();
  // Normalizing by |E * b0| turns the algebraic coplanarity residual into the
  // sine of the angular miss, so the threshold is interpretable in radians and
  // does not drift with position in the image.
  Vec3 Eb0 = E_ * b0;
  number_t n = Eb0.norm();
  if (!(n > 0.0)) {
    return std::numeric_limits<number_t>::infinity();
  }
  return std::abs(b1.dot(Eb0) / n);
}

} // namespace xivo
