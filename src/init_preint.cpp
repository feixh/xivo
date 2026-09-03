#include "init_preint.h"

#include <algorithm>
#include <cmath>

#include "rodrigues.h"

namespace xivo {

Mat3 SO3RightJacobian(const Vec3 &w) {
  const number_t th2 = w.squaredNorm();
  const Mat3 W = hat(w);
  // Below ~1e-4 rad the closed form loses more to cancellation in
  // (theta - sin theta) than the truncated series loses to truncation: the
  // series error is O(theta^4) ~ 1e-16 while the subtraction of two nearly
  // equal quantities of size theta throws away ~theta^-2 * eps.
  if (th2 < 1e-8)
    return Mat3::Identity() - 0.5 * W + W * W / 6.0;
  const number_t th = std::sqrt(th2);
  return Mat3::Identity() - (1.0 - std::cos(th)) / th2 * W +
         (th - std::sin(th)) / (th2 * th) * W * W;
}

Mat3 SO3RightJacobianInverse(const Vec3 &w) {
  const number_t th2 = w.squaredNorm();
  const Mat3 W = hat(w);
  if (th2 < 1e-8)
    return Mat3::Identity() + 0.5 * W + W * W / 12.0;
  const number_t th = std::sqrt(th2);
  const number_t s = std::sin(th), c = std::cos(th);
  // The quadratic coefficient is 1/th^2 - (1 + cos th) / (2 th sin th). Near
  // th = pi both parts of that fraction vanish -- (1 + cos th) as (pi - th)^2/2
  // and sin th as (pi - th) -- so the quotient tends to zero and the whole
  // coefficient to 1/pi^2. That limit is reached numerically rather than
  // symbolically here: cos(pi) is exactly -1 in double, so the numerator is
  // exactly zero and no cancellation is left to go wrong.
  const number_t k = 1.0 / th2 - (1.0 + c) / (2.0 * th * s);
  return Mat3::Identity() + 0.5 * W + k * W * W;
}

Mat3 Preintegral::RAt(const Vec3 &bg_new) const {
  return R * SO3::exp(dR_dbg * (bg_new - bg)).matrix();
}

Vec3 Preintegral::BetaAt(const Vec3 &bg_new, const Vec3 &ba_new) const {
  return beta + dbeta_dbg * (bg_new - bg) + dbeta_dba * (ba_new - ba);
}

Vec3 Preintegral::AlphaAt(const Vec3 &bg_new, const Vec3 &ba_new) const {
  return alpha + dalpha_dbg * (bg_new - bg) + dalpha_dba * (ba_new - ba);
}

void InterpolateImu(const std::vector<InitImu> &imu, number_t t, Vec3 *gyro,
                    Vec3 *accel) {
  if (imu.empty()) {
    *gyro = Vec3::Zero();
    *accel = Vec3::Zero();
    return;
  }
  if (t <= imu.front().t) {
    *gyro = imu.front().gyro;
    *accel = imu.front().accel;
    return;
  }
  if (t >= imu.back().t) {
    *gyro = imu.back().gyro;
    *accel = imu.back().accel;
    return;
  }
  const auto it = std::lower_bound(
      imu.begin(), imu.end(), t,
      [](const InitImu &s, number_t v) { return s.t < v; });
  const InitImu &hi = *it;
  const InitImu &lo = *(it - 1);
  const number_t span = hi.t - lo.t;
  const number_t w = span > 0 ? (t - lo.t) / span : 0.0;
  *gyro = lo.gyro + w * (hi.gyro - lo.gyro);
  *accel = lo.accel + w * (hi.accel - lo.accel);
}

Preintegral Preintegrate(const std::vector<InitImu> &imu, number_t t0,
                         number_t t1, const Vec3 &bg, const Vec3 &ba) {
  Preintegral p;
  p.bg = bg;
  p.ba = ba;
  if (imu.empty() || t1 <= t0)
    return p;

  // Knot times: the interval endpoints plus every sample strictly inside. The
  // endpoints are interpolated, which is what lets a camera frame sit between
  // two IMU samples without biasing the interval length.
  std::vector<number_t> knots;
  knots.reserve(imu.size() + 2);
  knots.push_back(t0);
  for (const auto &s : imu)
    if (s.t > t0 && s.t < t1)
      knots.push_back(s.t);
  knots.push_back(t1);

  Vec3 g_prev, a_prev;
  InterpolateImu(imu, knots.front(), &g_prev, &a_prev);
  for (size_t k = 1; k < knots.size(); ++k) {
    const number_t dt = knots[k] - knots[k - 1];
    if (dt <= 0)
      continue;
    Vec3 g_next, a_next;
    InterpolateImu(imu, knots[k], &g_next, &a_next);

    // Midpoint of a linearly interpolated stream is the mean of the endpoints.
    const Vec3 w = 0.5 * (g_prev + g_next) - bg;
    const Vec3 f = 0.5 * (a_prev + a_next) - ba;

    const Vec3 u = w * (0.5 * dt);
    const Mat3 H = SO3::exp(u).matrix();
    const Mat3 Jr_half = SO3RightJacobian(u);
    const Mat3 Rm = p.R * H; // rotation at the midpoint of the sub-interval
    // d(Rm)/dbg on the right: R_k contributes H' * dR_dbg (moved through H),
    // the half step contributes -Jr(u) * dt/2.
    const Mat3 Jm = H.transpose() * p.dR_dbg - Jr_half * (0.5 * dt);

    const Vec3 Rf = Rm * f;
    // d(Rm * f)/dbg = -Rm * hat(f) * Jm, since Rm*exp(Jm db)*f ~= Rm*f - Rm*hat(f)*Jm*db.
    const Mat3 dRf_dbg = -Rm * hat(f) * Jm;
    const Mat3 dRf_dba = -Rm; // f = a - ba

    // alpha uses beta *before* it advances; order matters.
    p.alpha += p.beta * dt + 0.5 * Rf * dt * dt;
    p.dalpha_dbg += p.dbeta_dbg * dt + 0.5 * dRf_dbg * dt * dt;
    p.dalpha_dba += p.dbeta_dba * dt + 0.5 * dRf_dba * dt * dt;

    p.beta += Rf * dt;
    p.dbeta_dbg += dRf_dbg * dt;
    p.dbeta_dba += dRf_dba * dt;

    const Vec3 uf = w * dt;
    const Mat3 dR = SO3::exp(uf).matrix();
    p.dR_dbg = dR.transpose() * p.dR_dbg - SO3RightJacobian(uf) * dt;
    p.R = p.R * dR;

    g_prev = g_next;
    a_prev = a_next;
    ++p.samples;
  }

  // Repeated multiplication drifts off SO(3) slowly; the filter normalizes every
  // step (`Rsb.normalize()`), so do the same here.
  p.R = SO3(Eigen::Quaternion<number_t>(p.R).normalized()).matrix();
  p.dt = t1 - t0;
  return p;
}

} // namespace xivo
