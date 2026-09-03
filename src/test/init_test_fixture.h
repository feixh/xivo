// Synthetic-window fixture shared by the M2 and M3 initializer tests.
//
// Extracted rather than duplicated: both stages are checked against the *same*
// closed-form motion, so if the two files carried their own copies a fix to one
// (the camera-behind-the-scene bug recorded in `EurocishCam`, say) would silently
// not reach the other, and the two suites would stop testing the same thing while
// both still passing.
//
// The whole point of the fixture is that the motion has a closed form. Constant
// body rate plus constant world-frame acceleration gives
//
//     R(t)     = exp(omega t)
//     p(t)     = v0 t + 0.5 a_w t^2
//     alpha(T) = 0.5 (a_w - g_w) T^2        beta(T) = (a_w - g_w) T
//
// with alpha and beta independent of how the rig is spinning -- so a solver that
// agrees with those to 1e-12 while the rig turns at 1 rad/s is right for a
// reason, not right by construction.
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "alias.h"
#include "init_preint.h"
#include "init_problem.h"
#include "rodrigues.h"

namespace xivo {
namespace inittest {

constexpr number_t kG = 9.81;
const Vec3 kGw{0, 0, -kG}; // gravity acceleration, world; matches Estimator::g_

// ---------------------------------------------------------------------------
// synthetic IMU with an exact position/orientation history
// ---------------------------------------------------------------------------
struct Truth {
  Vec3 omega{Vec3::Zero()}; // constant body rate
  Vec3 a_w{Vec3::Zero()};   // constant world-frame acceleration
  Vec3 v0{Vec3::Zero()};    // world velocity at t = 0
  Vec3 bg{Vec3::Zero()};
  Vec3 ba{Vec3::Zero()};

  Mat3 R(number_t t) const { return SO3::exp(omega * t).matrix(); }
  Vec3 p(number_t t) const { return v0 * t + 0.5 * a_w * t * t; }
  Vec3 v(number_t t) const { return v0 + a_w * t; }

  // The preintegrals over [t0, t1], in the body frame at t0. Note the
  // R(t0)' factor: `beta` is expressed in I(t0), not in the world, so it is
  // *not* (a_w - g_w) * dt unless t0 = 0. Getting that wrong reads as a
  // preintegration bug -- it did, on the first run of
  // `EndpointsNeedNotLandOnSamples` -- when it is only a frame confusion in the
  // expected value.
  Vec3 BetaExact(number_t t0, number_t t1) const {
    return R(t0).transpose() * (a_w - kGw) * (t1 - t0);
  }
  Vec3 AlphaExact(number_t t0, number_t t1) const {
    const number_t dt = t1 - t0;
    return 0.5 * R(t0).transpose() * (a_w - kGw) * dt * dt;
  }
  Mat3 RExact(number_t t0, number_t t1) const {
    return R(t0).transpose() * R(t1);
  }
  /** What the IMU reports: specific force in the body frame, plus the bias. */
  InitImu Sample(number_t t) const {
    InitImu s;
    s.t = t;
    s.gyro = omega + bg;
    s.accel = R(t).transpose() * (a_w - kGw) + ba;
    return s;
  }
};

std::vector<InitImu> MakeImu(const Truth &tr, number_t t0, number_t t1,
                             number_t rate) {
  std::vector<InitImu> out;
  const number_t dt = 1.0 / rate;
  for (number_t t = t0; t <= t1 + 0.5 * dt; t += dt)
    out.push_back(tr.Sample(t));
  return out;
}

std::vector<Vec3> MakeScene(int n, unsigned seed = 11) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<number_t> ux(-4, 4), uy(-3, 3), uz(2, 9);
  std::vector<Vec3> pts;
  pts.reserve(n);
  for (int i = 0; i < n; ++i)
    pts.emplace_back(ux(rng), uy(rng), uz(rng));
  return pts;
}

/** Build a Stage A problem from exact geometry: no pixels, no camera model, no
 *  KLT. `SolveLinearInit` cannot tell this from `InitWindowTracker`'s output,
 *  and that is the point -- a failure here is arithmetic, never data. */
InitProblem MakeProblem(const Truth &tr, const std::vector<Vec3> &pts,
                        const InitCamera &cam, number_t t0, number_t span,
                        int nframes, const Vec3 &bg_prior, const Vec3 &ba_prior,
                        number_t noise_sd = 0, unsigned seed = 3,
                        bool exact_pre = false) {
  const std::vector<InitImu> imu =
      MakeImu(tr, t0 - 0.05, t0 + span + 0.05, 200.0);
  InitProblem prob;
  prob.cams.push_back(cam);
  prob.gravity = kG;
  prob.num_tracks = static_cast<int>(pts.size());

  std::mt19937 rng(seed);
  std::normal_distribution<number_t> gauss(0, noise_sd > 0 ? noise_sd : 1);

  const Mat3 R0 = tr.R(t0);
  const Vec3 p0 = tr.p(t0);
  for (int k = 0; k < nframes; ++k) {
    const number_t t = t0 + span * k / std::max(1, nframes - 1);
    const number_t t_prev = t0 + span * (k - 1) / std::max(1, nframes - 1);
    InitFrame fr;
    fr.t = t;
    if (exact_pre) {
      // The preintegrals a perfect integrator at a perfect bias prior would
      // produce. Lets a test isolate the solver from the O(dt^2) discretization
      // error of the preintegrator, which has its own tests. `bg_prior` and
      // `ba_prior` are ignored in this mode -- the linearization point *is* the
      // truth.
      //
      // Integrate first and overwrite only the integrals: the bias Jacobians and
      // the recorded linearization point have to survive, or Stage B would see a
      // residual that does not depend on the biases at all and would report
      // whatever it was seeded with while converging beautifully.
      fr.pre = Preintegrate(imu, t0, t, tr.bg, tr.ba);
      fr.pre.R = tr.RExact(t0, t);
      fr.pre.alpha = tr.AlphaExact(t0, t);
      fr.pre.beta = tr.BetaExact(t0, t);
      fr.pre.dt = t - t0;
      if (k > 0) {
        fr.pre_prev = Preintegrate(imu, t_prev, t, tr.bg, tr.ba);
        fr.pre_prev.R = tr.RExact(t_prev, t);
        fr.pre_prev.alpha = tr.AlphaExact(t_prev, t);
        fr.pre_prev.beta = tr.BetaExact(t_prev, t);
        fr.pre_prev.dt = t - t_prev;
      }
    } else {
      fr.pre = Preintegrate(imu, t0, t, bg_prior, ba_prior);
      if (k > 0)
        fr.pre_prev = Preintegrate(imu, t_prev, t, bg_prior, ba_prior);
    }
    prob.frames.push_back(fr);

    // Exact geometry, in I0: the body pose at t relative to the pose at t0.
    const Mat3 R_I0_Ik = R0.transpose() * tr.R(t);
    const Vec3 p_Ik_I0 = R0.transpose() * (tr.p(t) - p0);
    for (size_t i = 0; i < pts.size(); ++i) {
      const Vec3 pf_I0 = R0.transpose() * (pts[i] - p0);
      const Vec3 pf_Ik = R_I0_Ik.transpose() * (pf_I0 - p_Ik_I0);
      const Vec3 pf_c = cam.Rbc.transpose() * (pf_Ik - cam.Tbc);
      if (pf_c(2) < 0.3)
        continue;
      InitObservation o;
      o.frame = k;
      o.track = static_cast<int>(i);
      o.cam = 0;
      o.xn = Vec2{pf_c(0) / pf_c(2), pf_c(1) / pf_c(2)};
      if (noise_sd > 0)
        o.xn += Vec2{gauss(rng), gauss(rng)};
      prob.obs.push_back(o);
    }
  }
  return prob;
}

/** What `InitWindowTracker::GravityFromAccelMean` computes, from the truth
 *  instead of from a tracked window: gravity in `I0` from the mean of the
 *  window's accelerometer readings, each rotated into `I0` first. Its error is
 *  the window's mean specific force, which for this fixture's *constant* `a_w`
 *  is the full `atan(|a_w| / 9.81)` -- deliberately adversarial, since real
 *  motion averages much of that away (measured 1.1 deg on MH_01). */
Vec3 AccelMeanGravity(const Truth &tr, number_t t0, number_t span,
                      const Vec3 &bg, const Vec3 &ba) {
  const std::vector<InitImu> imu =
      MakeImu(tr, t0 - 0.05, t0 + span + 0.05, 200.0);
  Vec3 acc = Vec3::Zero();
  int n = 0;
  for (const auto &s : imu) {
    if (s.t < t0 || s.t > t0 + span)
      continue;
    acc += Preintegrate(imu, t0, s.t, bg, ba).R * (s.accel - ba);
    ++n;
  }
  const Vec3 g = -acc / std::max(1, n);
  return g * (kG / g.norm());
}

/** Angle between two vectors, in radians. */
number_t Angle(const Vec3 &a, const Vec3 &b) {
  const number_t c = a.dot(b) / (a.norm() * b.norm());
  return std::acos(std::max<number_t>(-1, std::min<number_t>(1, c)));
}

InitCamera EurocishCam() {
  InitCamera c;
  // A real 90-degree roll about the optical axis plus small tilts, and EuRoC
  // cam0's actual 6.9 cm offset. Non-trivial on purpose: with Rbc = I and
  // Tbc = 0 the extrinsics drop out of every equation and a sign error in them
  // is invisible.
  //
  // The optical axis has to stay pointed at the scene, which the first version
  // of this fixture got wrong -- a 1.55 rad rotation about *y* swings the axis
  // onto body -x while `MakeScene` puts everything at body +z, so 97 of 120
  // features were behind the camera and every downstream tolerance failed for a
  // reason that had nothing to do with the solver.
  c.Rbc = SO3::exp(Vec3{0.03, -0.02, 1.5708}).matrix();
  c.Tbc = Vec3{-0.0216, -0.0647, 0.0098};
  c.focal = 458.654; // EuRoC cam0's fx, so a "0.3 px" sigma means 0.3 px
  return c;
}
} // namespace inittest
} // namespace xivo
