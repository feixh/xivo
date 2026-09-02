// IMU preintegration for the initialization window.
//
// XIVO has none. The filter propagates the *full* state with an RK4 or
// Prince-Dormand step (`src/rk4.cpp`, `src/princedormand.cpp`) driven by the
// current bias estimate, which is exactly the wrong tool here: the initializer
// does not yet know the bias, and it needs to re-evaluate the same IMU interval
// many times as the optimizer moves the bias around. Preintegration is the
// standard answer -- integrate once at a fixed linearization bias, carry the
// derivative with respect to that bias, and correct to first order afterwards.
//
// The convention is pinned to `Estimator::ComposeMotion` (`estimator.cpp:1005`),
// because the whole point is that the result can be handed to that filter:
//
//     Tsb += Vsb * dt
//     Vsb += (Rsb * (Ca*a - ba) + Rsg * g_) * dt        g_ = [0, 0, -9.81]
//     Rsb *= SO3::exp((Cg*w - bg) * dt)
//
// so `Ca`/`Cg` are applied by the caller (they are calibration, not unknowns)
// and `g` below is the gravity *acceleration* vector -- it points down and has
// magnitude 9.81, matching `Rsg * g_`, not its negation. Getting that sign wrong
// produces a solve that converges beautifully to an upside-down world, so the
// synthetic tests check `g` against an analytically known vector rather than
// only checking that the residual went to zero.
//
// With `R_k = R_{I0<-Ik}`, `alpha_k`, `beta_k` the preintegrals from frame 0 to
// frame k and `dt_k = t_k - t_0`:
//
//     v_{Ik}^{I0} = v_{I0}^{I0} + g^{I0} * dt_k         + beta_k
//     p_{Ik}^{I0} = v_{I0}^{I0} * dt_k + 0.5 * g^{I0} * dt_k^2 + alpha_k
//
// Integration is midpoint in *both* the IMU reading and the rotation: the
// increment uses `R_k * exp(w*dt/2)` rather than `R_k`. That is not polish. With
// a constant world-frame acceleration and a constant body rate the exact answer
// is `beta = (a_w - g_w) T` and `alpha = 0.5 (a_w - g_w) T^2` regardless of how
// the rig is spinning, and the midpoint rotation reproduces it to machine
// precision where the left-endpoint rotation leaves an O(dt^2)-per-step error
// that grows with rate. That exactness is what lets the unit tests compare
// against a closed form instead of against a finer-grained version of
// themselves.
#pragma once

#include <vector>

#include "alias.h"

namespace xivo {

/** One IMU sample as the initializer consumes it: already multiplied by the
 *  calibration matrices `imu_.Cg()` / `imu_.Ca()`, with the biases still in. */
struct InitImu {
  number_t t{0};
  Vec3 gyro{Vec3::Zero()};
  Vec3 accel{Vec3::Zero()};
};

/** The right Jacobian of SO(3): `exp(w + dw) ~= exp(w) * exp(Jr(w) * dw)`. */
Mat3 SO3RightJacobian(const Vec3 &w);

/** IMU preintegral over one interval, at a fixed linearization bias.
 *
 *  `R` is `R_{i<-j}` (the later frame expressed in the earlier one), `beta` the
 *  velocity increment and `alpha` the position increment, both in frame `i`.
 *  The `d*_db*` blocks are derivatives with respect to a correction added to
 *  `bg` / `ba`; for the rotation the correction is applied on the right, i.e.
 *  `R(bg + db) ~= R * exp(dR_dbg * db)`. */
struct Preintegral {
  number_t dt{0};
  Mat3 R{Mat3::Identity()};
  Vec3 beta{Vec3::Zero()};
  Vec3 alpha{Vec3::Zero()};

  Mat3 dR_dbg{Mat3::Zero()};
  Mat3 dbeta_dbg{Mat3::Zero()};
  Mat3 dbeta_dba{Mat3::Zero()};
  Mat3 dalpha_dbg{Mat3::Zero()};
  Mat3 dalpha_dba{Mat3::Zero()};

  Vec3 bg{Vec3::Zero()}; ///< the linearization point the above was integrated at
  Vec3 ba{Vec3::Zero()};
  int samples{0};

  /** First-order correction of the preintegral to a different bias. Cheap; the
   *  alternative is re-running `Preintegrate`, which `RepropagateThreshold` in a
   *  full VIO would do but an initializer over half a second never needs. */
  Mat3 RAt(const Vec3 &bg_new) const;
  Vec3 BetaAt(const Vec3 &bg_new, const Vec3 &ba_new) const;
  Vec3 AlphaAt(const Vec3 &bg_new, const Vec3 &ba_new) const;
};

/** Linearly interpolate the IMU stream at `t`, clamping outside its span.
 *  `imu` must be sorted by time and non-empty. */
void InterpolateImu(const std::vector<InitImu> &imu, number_t t, Vec3 *gyro,
                    Vec3 *accel);

/** Preintegrate over `[t0, t1]` at the given linearization bias.
 *
 *  Sub-intervals are cut at every IMU sample time inside the interval, with the
 *  two endpoints interpolated, so a window frame boundary does not have to land
 *  on a sample. Returns an identity preintegral if `t1 <= t0`. */
Preintegral Preintegrate(const std::vector<InitImu> &imu, number_t t0,
                         number_t t1, const Vec3 &bg, const Vec3 &ba);

} // namespace xivo
