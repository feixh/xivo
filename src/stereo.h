// Fixed stereo rig geometry: the rigid transform between the two cameras, plus
// the primitives built on top of it (epipolar residual, triangulation).
//
// The stereo extrinsics are deliberately NOT part of the EKF state. TUM-VI (and
// most rigs) ship factory calibration good to ~1e-3, and keeping them fixed
// avoids disturbing the error-state layout in core.h (Index, kMotionSize,
// kFullSize) and the covariance blocks keyed off it. The camera-to-body
// alignment `gbc` remains in the state as before; the rig only adds the
// left->right leg on top of it.
#pragma once
#include <memory>

#include "json/json.h"

#include "alias.h"
#include "camera_manager.h"

namespace xivo {

/** Geometry of a two-camera rig, held fixed.
 *
 * Frame conventions, matching the rest of the codebase (`gbc` maps a point in
 * the camera frame to the body frame):
 *   - camera 0 ("left") is the primary camera; `gbc` in `State` refers to it,
 *     and all existing monocular code paths mean camera 0 when they say
 *     "the camera".
 *   - `gc0c1` maps a point in the camera-1 frame into the camera-0 frame.
 *     Equivalently it is the pose of camera 1 expressed in camera 0's frame.
 *
 * So a point known in camera 0 lands in camera 1 via `gc1c0 = gc0c1.inverse()`:
 *     Xc1 = gc1c0 * Xc0
 */
class StereoRig {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** Build from a config block; see `FromJson` for the accepted schema. */
  static StereoRig *Create(const Json::Value &cfg);
  static StereoRig *instance() { return instance_.get(); }
  /** True when a stereo rig has been configured. Monocular runs leave this
   * false and every stereo code path is then skipped. */
  static bool enabled() { return instance_ != nullptr; }

  /** Pose of camera 1 in camera 0's frame. */
  const SE3 &gc0c1() const { return gc0c1_; }
  /** Maps a point from camera 0's frame to camera 1's frame. */
  const SE3 &gc1c0() const { return gc1c0_; }
  const Mat3 &Rc1c0() const { return Rc1c0_; }
  const Vec3 &Tc1c0() const { return Tc1c0_; }

  /** Distance between the two optical centres, in metres. Sanity/diagnostics. */
  number_t baseline() const { return gc0c1_.translation().norm(); }

  /** Transform a point from camera 0's frame into camera 1's frame. */
  Vec3 ToCam1(const Vec3 &Xc0) const { return Rc1c0_ * Xc0 + Tc1c0_; }

  /** Triangulate from a pair of *normalized* (already unprojected) coordinates.
   *
   * `xc0` and `xc1` are (X/Z, Y/Z) in their respective camera frames. Returns
   * the 3D point in camera 0's frame and writes the residual of the two rays'
   * closest approach to `gap` when non-null -- that residual is the natural
   * quality gate, since well-matched points give near-intersecting rays.
   *
   * Uses the midpoint of the closest approach of the two rays, which is the
   * least-squares optimum for isotropic ray noise and is numerically better
   * behaved at low parallax than the DLT.
   *
   * Returns false when the rays are too close to parallel to intersect
   * meaningfully, or when the result lands behind either camera.
   */
  bool Triangulate(const Vec2 &xc0, const Vec2 &xc1, Vec3 *Xc0,
                   number_t *gap = nullptr) const;

  /** Triangulate from *pixel* observations, and report how uncertain the
   * resulting log-depth is.
   *
   * `xp0`/`xp1` are pixel coordinates in cameras 0 and 1; both are unprojected
   * internally, so callers do not need to know the distortion models.
   *
   * `log_depth_std` receives the standard deviation of log(z) implied by a
   * `sigma_px` matching error on the right observation. It is obtained by
   * re-triangulating with the right observation displaced by `sigma_px` pixels
   * and taking the change in log-depth, rather than from the rectified-stereo
   * formula sigma_z = z^2 sigma_d / (f b). The closed form needs a single focal
   * length, and on a 190-degree fisheye the effective focal length varies
   * substantially across the field, so a constant f would understate the
   * uncertainty at the periphery -- exactly where matches are worst. The
   * numerical version is exact for whatever camera model is configured and
   * costs one extra triangulation.
   *
   * log-depth is the right space for this because that is how features are
   * parameterized (`x_(2) = log z`), so the value drops straight into the
   * feature's covariance without a further Jacobian.
   *
   * Returns false if either triangulation fails (degenerate parallax or a point
   * behind a camera), in which case no output is written.
   */
  bool TriangulateFromPixels(const Vec2 &xp0, const Vec2 &xp1,
                             number_t sigma_px, Vec3 *Xc0,
                             number_t *log_depth_std,
                             number_t *gap = nullptr) const;

  /** Angular epipolar residual, in radians.
   *
   * For a fisheye rig the epipolar constraint is not a straight line in pixel
   * space, so this works on the unprojected bearing vectors instead: it returns
   * the absolute value of the (normalized) coplanarity residual
   *     b0' * E * b1,   E = [Tc1c0]_x * Rc1c0
   * which is the sine of the angle by which the two bearings miss being
   * coplanar with the baseline. Zero for a perfect match.
   */
  number_t EpipolarResidual(const Vec2 &xc0, const Vec2 &xc1) const;

private:
  StereoRig(const Json::Value &cfg);
  StereoRig(const StereoRig &) = delete;
  StereoRig &operator=(const StereoRig &) = delete;

  static std::unique_ptr<StereoRig> instance_;

  SE3 gc0c1_, gc1c0_;
  Mat3 Rc1c0_;
  Vec3 Tc1c0_;
  Mat3 E_; // essential matrix, cam0 -> cam1
};

} // namespace xivo
