#include "estimator.h"

namespace xivo {

void Estimator::RK4(const Vec3 &gyro0, const Vec3 &accel0, number_t dt) {
  static bool rk4_initialized{false};
  static number_t stepsize{-1};
  if (!rk4_initialized) {
    stepsize = cfg_["RK4"].get("stepsize", 0.002).asDouble();
    rk4_initialized = true;
  }

  if (stepsize < 0) {
    RK4Step(gyro0, accel0, dt);
  } else {
    number_t total_step = 0;

    Vec3 gyro{gyro0}, accel{accel0};
    while (total_step < dt) {
      number_t h = stepsize;
      if (total_step + h > dt) {
        h = dt - total_step;
      } else if (total_step + h + 0.5 * h > dt) {
        // half step trick
        h = 0.5 * h;
      }
      RK4Step(gyro, accel, h);
      gyro += slope_gyro_ * h;
      accel += slope_accel_ * h;
      total_step += h;
    }
  }
}

void Estimator::RK4Step(const Vec3 &gyro0, const Vec3 &accel0, number_t dt) {
  number_t halfstep = 0.5 * dt;

  static State X0;
  static Vec3 K1, K2, K3, K4;
  // 9x24 and fixed-size; see `kMotionDynSize` and the comment in
  // `PrinceDormandStep`, which this mirrors stage for stage.
  static MatMotionDyn FK1, FK2, FK3, FK4;
  static MatMotion PK1, PK2, PK3, PK4;
  static MatMotion P0;

  Eigen::Matrix<number_t, 6, 1> slope;
  slope << slope_gyro_, slope_accel_;
  Eigen::Matrix<number_t, 6, 1> gyro_accel, gyro_accel0;
  gyro_accel0 << gyro0, accel0;

  // A view of `Fdyn_`, re-read after each `ComputeMotionJacobianAt`.
  const auto Fleft = Fdyn_.leftCols<kMotionDynSize>();

  X0 = X_;
  // uncomment the following to use non-standard RK4?
  // ComposeMotion(X0, X0.Vsb, gyro_accel0, dt);
  K1 = X0.Vsb;
  ComputeMotionJacobianAt(X0, gyro_accel0);
  FK1 = Fdyn_;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0);
  MotionCovSlope(Fdyn_, P0, X0.Rsb.matrix(), Qimu_, PK1);

  X0 = X_;
  gyro_accel = gyro_accel0 + halfstep * slope;
  ComposeMotion(X0, 0.5 * K1, gyro_accel, halfstep);
  K2 = X0.Vsb;
  ComputeMotionJacobianAt(X0, gyro_accel);
  FK2 = Fdyn_ + (Fleft * FK1) * halfstep;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) + halfstep * PK1;
  MotionCovSlope(Fdyn_, P0, X0.Rsb.matrix(), Qimu_, PK2);

  X0 = X_;
  gyro_accel = gyro_accel0 + halfstep * slope;
  ComposeMotion(X0, 0.5 * K2, gyro_accel, halfstep);
  K3 = X0.Vsb;
  ComputeMotionJacobianAt(X0, gyro_accel);
  FK3 = Fdyn_ + (Fleft * FK2) * halfstep;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) + halfstep * PK2;
  MotionCovSlope(Fdyn_, P0, X0.Rsb.matrix(), Qimu_, PK3);

  X0 = X_;
  gyro_accel = gyro_accel0 + halfstep * slope;
  ComposeMotion(X0, K3, gyro_accel, dt);
  K4 = X0.Vsb;
  ComputeMotionJacobianAt(X0, gyro_accel);
  FK4 = Fdyn_ + (Fleft * FK3) * dt;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) + dt * PK3;
  MotionCovSlope(Fdyn_, P0, X0.Rsb.matrix(), Qimu_, PK4);

  static Vec3 Ktot;
  static MatMotionDyn FK, Fdt;
  static MatMotion PK;
  Ktot = (K1 + 2.0 * (K2 + K3) + K4) / 6.0;
  FK = (FK1 + 2.0 * (FK2 + FK3) + FK4) / 6.0;
  PK = (PK1 + 2.0 * (PK2 + PK3) + PK4) / 6.0;

  // apply the aggregated difference to state
  gyro_accel = gyro_accel0 + dt * slope;
  ComposeMotion(X_, Ktot, gyro_accel, dt);

  P_.block<kMotionSize, kMotionSize>(0, 0).noalias() += PK * dt;
  // Deferred to one application per image; see
  // `AccumulateMotionStructureCorrelation`. The step transition is
  // `I + [Fdt; 0]`; the identity is implicit.
  Fdt = FK * dt;
  AccumulateMotionStructureCorrelation(Fdt);
}

} // namespace xivo
