// Prince-Dormand numerical integration
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "estimator.h"

#include <cmath>
#include <limits>

namespace xivo {

void Estimator::PrinceDormand(const Vec3 &gyro0, const Vec3 &accel0, number_t dt) {
  // reference 1:
  // http://www.mymathlib.com/c_source/diffeq/embedded_runge_kutta/embedded_prince_dormand_v3_4_5.c
  // reference 2:
  // http://depa.fquim.unam.mx/amyd/archivero/DormandPrince_19856.pdf
  static bool pd_initialized{false}, control_stepsize{false};
  static number_t tolerance, min_scale_factor, max_scale_factor, h, h0;
  static int attempts;

  if (!pd_initialized) {
    auto pd_cfg = cfg_["PrinceDormand"];
    control_stepsize = pd_cfg.get("control_stepsize", false).asBool();
    tolerance = pd_cfg.get("tolerance", 1e-3).asDouble();
    attempts = pd_cfg.get("attempts", 12).asInt();
    min_scale_factor = pd_cfg.get("min_scale_factor", 0.125).asDouble();
    max_scale_factor = pd_cfg.get("max_scale_factor", 4.0).asDouble();
    h = pd_cfg.get("stepsize", 0.002).asDouble();
    h0 = h;
    pd_initialized = true;
  }

  if (control_stepsize) {
    // NOTE: this branch only ever *grows or shrinks the next* step; it never
    // rejects and retries a step whose error exceeded `tolerance`, because
    // `PrinceDormandStep` commits directly to X_/P_ and there is no rollback. The
    // `attempts` config key exists for exactly that retry count and is read into
    // a local that nothing reads back. Leaving the retry unimplemented rather
    // than faking it: it needs a state/covariance snapshot per trial step. No
    // shipped config enables control_stepsize.
    static bool warned{false};
    if (!warned) {
      LOG(WARNING) << "PrinceDormand.control_stepsize is on, but step rejection "
                      "is not implemented (`attempts` is ignored)";
      warned = true;
    }

    number_t total_step = 0.0, scale = 1.0;

    if (h < 1e-6) {
      h = h0;
    }

    h = std::min(h, dt);

    while (total_step < dt) {
      number_t err = PrinceDormandStep(gyro0 + slope_gyro_ * total_step,
                                    accel0 + slope_accel_ * total_step, h);
      total_step += h;
      if (err == 0.0) {
        scale = max_scale_factor;
      } else {
        scale = 0.8 * sqrt(sqrt(tolerance * h / err));
        scale = std::min(std::max(scale, min_scale_factor),
                         max_scale_factor); // clipping
      }
      VLOG(1) << "err=" << err << " ;h=" << h << " ;s=" << scale;

      h *= scale;
      if (total_step < dt) {
        if (total_step + h > dt) {
          h = dt - total_step;
        } else if (total_step + h + 0.5 * h > dt) {
          h = 0.5 * h;
        }
      }
    }
  } else {
    // constant h0
    if (h0 < 0) {
      PrinceDormandStep(gyro0, accel0, dt);
    } else {
      number_t total_step = 0;

      Vec3 gyro{gyro0}, accel{accel0};
      while (total_step < dt) {
        number_t h = h0; // this shadows the static variable h
        if (total_step + h > dt) {
          h = dt - total_step;
        } else if (total_step + h + 0.5 * h > dt) {
          // half step trick
          h = 0.5 * h;
        }
        PrinceDormandStep(gyro, accel, h);
        gyro += slope_gyro_ * h;
        accel += slope_accel_ * h;
        total_step += h;
      }
    }
  }
}

number_t Estimator::PrinceDormandStep(const Vec3 &gyro0, const Vec3 &accel0,
                                   number_t dt) {
  static const number_t r_9 = 1.0 / 9.0;
  static const number_t r_2_9 = 2.0 / 9.0;
  static const number_t r_12 = 1.0 / 12.0;
  static const number_t r_324 = 1.0 / 324.0;
  static const number_t r_330 = 1.0 / 330.0;
  static const number_t r_28 = 1.0 / 28.0;
  static const number_t r_400 = 1.0 / 400.0;

  static State X0;
  static Vec3 K1, K2, K3, K4, K5, K6, K7;
  // Fixed-size, and 9x24 rather than 24x24: a stage's transition slope is
  // `F + F (...) dt`, and `F` is zero below row `kMotionDynSize`, so every `FK`
  // is too. The covariance slopes stay 24x24 -- `A + A'` fills nine rows *and*
  // nine columns, and the noise term adds two more diagonal blocks.
  static MatMotionDyn FK1, FK2, FK3, FK4, FK5, FK6, FK7;
  static MatMotion PK1, PK2, PK3, PK4, PK5, PK6, PK7;
  static MatMotion P0;

  number_t step;
  Eigen::Matrix<number_t, 6, 1> slope;
  slope << slope_gyro_, slope_accel_;
  Eigen::Matrix<number_t, 6, 1> gyro_accel0, gyro_accel;
  gyro_accel0 << gyro0, accel0;

  // The stage transition slopes are `F + F (combination) dt`. `F` is zero below
  // row `kMotionDynSize` and so is every `FK`, so the product only ever needs
  // `F`'s leading 9 columns against the 9 rows the combination actually has:
  // 9x9x24 instead of 24x24x24.
  //
  // `Fleft` is a *view* of `Fdyn_`, not a snapshot of it: each stage below
  // re-fills `Fdyn_` via `ComputeMotionJacobianAt` and then reads it through this
  // name, which is the intent -- a stage's slope uses the Jacobian at that
  // stage's state.
  const auto Fleft = Fdyn_.leftCols<kMotionDynSize>();

  X0 = X_;
  K1 = X0.Vsb;
  ComputeMotionJacobianAt(X0, gyro_accel0);
  FK1 = Fdyn_;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0);
  MotionCovSlope(Fdyn_, P0, X0.Rsb.matrix(), Qimu_, PK1);

  X0 = X_;
  step = r_2_9 * dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(X0, r_2_9 * (K1), gyro_accel, step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K2 = X0.Vsb;
  FK2 = Fdyn_ + (Fleft * FK1) * (r_2_9 * dt);
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) + (r_2_9 * dt) * PK1;
  MotionCovSlope(Fdyn_, P0, X0.Rsb.matrix(), Qimu_, PK2);

  X0 = X_;
  step = 3.0 * r_9 * dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(X0, r_12 * (K1 + 3.0 * K2), gyro_accel, step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K3 = X0.Vsb;
  FK3 = Fdyn_ + (Fleft * (FK1 + 3.0 * FK2)) * (r_12 * dt);
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) +
       (r_12 * dt) * (PK1 + 3.0 * PK2);
  MotionCovSlope(Fdyn_, P0, X0.Rsb.matrix(), Qimu_, PK3);

  X0 = X_;
  step = 5.0 * r_9 * dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(X0, r_324 * (55.0 * K1 - 75.0 * K2 + 200.0 * K3), gyro_accel,
                step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K4 = X0.Vsb;
  FK4 = Fdyn_ +
        (Fleft * (55.0 * FK1 - 75.0 * FK2 + 200.0 * FK3)) * (r_324 * dt);
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) +
       (r_324 * dt) * (55.0 * PK1 - 75.0 * PK2 + 200.0 * PK3);
  MotionCovSlope(Fdyn_, P0, X0.Rsb.matrix(), Qimu_, PK4);

  X0 = X_;
  step = 6.0 * r_9 * dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(X0, r_330 * (83.0 * K1 - 195.0 * K2 + 305.0 * K3 + 27.0 * K4),
                gyro_accel, step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K5 = X0.Vsb;
  FK5 = Fdyn_ + (Fleft * (83.0 * FK1 - 195.0 * FK2 + 305.0 * FK3 +
                          27.0 * FK4)) *
                    (r_330 * dt);
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) +
       (r_330 * dt) * (83.0 * PK1 - 195.0 * PK2 + 305.0 * PK3 + 27.0 * PK4);
  MotionCovSlope(Fdyn_, P0, X0.Rsb.matrix(), Qimu_, PK5);

  X0 = X_;
  step = dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(
      X0, r_28 * (-19.0 * K1 + 63.0 * K2 + 4.0 * K3 - 108.0 * K4 + 88.0 * K5),
      gyro_accel, step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K6 = X0.Vsb;
  FK6 = Fdyn_ + (Fleft * (-19.0 * FK1 + 63.0 * FK2 + 4.0 * FK3 - 108.0 * FK4 +
                          88.0 * FK5)) *
                    (r_28 * dt);
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) +
       (r_28 * dt) *
           (-19.0 * PK1 + 63.0 * PK2 + 4.0 * PK3 - 108.0 * PK4 + 88.0 * PK5);
  MotionCovSlope(Fdyn_, P0, X0.Rsb.matrix(), Qimu_, PK6);

  X0 = X_;
  step = dt;
  gyro_accel = gyro_accel0 + slope * step;
  ComposeMotion(X0, r_400 * (38.0 * K1 + 240.0 * K3 - 243.0 * K4 + 330.0 * K5 +
                             35.0 * K6),
                gyro_accel, step);
  ComputeMotionJacobianAt(X0, gyro_accel);
  K7 = X0.Vsb;
  FK7 = Fdyn_ + (Fleft * (38.0 * FK1 + 240.0 * FK3 - 243.0 * FK4 +
                          330.0 * FK5 + 35.0 * FK6)) *
                    (r_400 * dt);
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) +
       (r_400 * dt) *
           (38.0 * PK1 + 240.0 * PK3 - 243.0 * PK4 + 330.0 * PK5 + 35.0 * PK6);
  MotionCovSlope(Fdyn_, P0, X0.Rsb.matrix(), Qimu_, PK7);

  static Vec3 K;
  static MatMotionDyn FK, Fdt;
  static MatMotion PK;
  K = 0.0862 * K1 + 0.6660 * K3 - 0.7857 * K4 + 0.9570 * K5 + 0.0965 * K6 -
      0.0200 * K7;
  FK = 0.0862 * FK1 + 0.6660 * FK3 - 0.7857 * FK4 + 0.9570 * FK5 +
       0.0965 * FK6 - 0.0200 * FK7;
  PK = 0.0862 * PK1 + 0.6660 * PK3 - 0.7857 * PK4 + 0.9570 * PK5 +
       0.0965 * PK6 - 0.0200 * PK7;

  // apply the aggregated difference to state
  gyro_accel = gyro_accel0 + slope * dt;
  ComposeMotion(X_, K, gyro_accel, dt);

  P_.block<kMotionSize, kMotionSize>(0, 0).noalias() += PK * dt;
  // Record this step's contribution to the motion-to-structure correlation
  // instead of rewriting the two 24x540 blocks, which nothing reads until the
  // next image. The step transition is `I + [Fdt; 0]`; the identity is implicit.
  Fdt = FK * dt;
  AccumulateMotionStructureCorrelation(Fdt);
  // The embedded 4th-order solution differs from the 5th-order one by this
  // combination of the stage slopes (Prince-Dormand v3(4,5); see reference 1),
  // which is the local truncation error estimate the step controller in
  // `PrinceDormand` above needs. Returning a hardcoded 0 made that controller
  // take its `err == 0.0` branch on every single step, so it multiplied the step
  // size by `max_scale_factor` unconditionally: "adaptive" stepping degenerated
  // to "always take the largest step allowed". The slopes are increments per unit
  // time and `ComposeMotion` applies them over `dt`, so the error in the state
  // increment carries a factor of dt.
  const Vec3 diffK = 0.0002 * (44.0 * K1 - 330.0 * K3 + 891.0 * K4 -
                               660.0 * K5 - 45.0 * K6 + 100.0 * K7);
  const number_t err = diffK.cwiseAbs().maxCoeff() * std::fabs(dt);
  return std::isfinite(err) ? err : std::numeric_limits<number_t>::max();
}

}
