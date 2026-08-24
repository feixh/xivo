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
  static MatX FK1, FK2, FK3, FK4;
  static MatX PK1, PK2, PK3, PK4;

  Eigen::Matrix<number_t, 6, 1> slope;
  slope << slope_gyro_, slope_accel_;
  Eigen::Matrix<number_t, 6, 1> gyro_accel, gyro_accel0;
  gyro_accel0 << gyro0, accel0;

  X0 = X_;
  // uncomment the following to use non-standard RK4?
  // ComposeMotion(X0, X0.Vsb, gyro_accel0, dt);
  K1 = X0.Vsb;
  ComputeMotionJacobianAt(X0, gyro_accel0);
  FK1 = F_;
  MatX P0 = P_.block<kMotionSize, kMotionSize>(0, 0);
  PK1 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();

  X0 = X_;
  gyro_accel = gyro_accel0 + halfstep * slope;
  ComposeMotion(X0, 0.5 * K1, gyro_accel, halfstep);
  K2 = X0.Vsb;
  ComputeMotionJacobianAt(X0, gyro_accel);
  FK2 = F_ + F_ * FK1 * halfstep;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) + halfstep * PK1;
  PK2 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();

  X0 = X_;
  gyro_accel = gyro_accel0 + halfstep * slope;
  ComposeMotion(X0, 0.5 * K2, gyro_accel, halfstep);
  K3 = X0.Vsb;
  ComputeMotionJacobianAt(X0, gyro_accel);
  FK3 = F_ + F_ * FK2 * halfstep;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) + halfstep * PK2;
  PK3 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();

  X0 = X_;
  gyro_accel = gyro_accel0 + halfstep * slope;
  ComposeMotion(X0, K3, gyro_accel, dt);
  K4 = X0.Vsb;
  ComputeMotionJacobianAt(X0, gyro_accel);
  FK4 = F_ + F_ * FK3 * dt;
  P0 = P_.block<kMotionSize, kMotionSize>(0, 0) + dt * PK3;
  PK4 = F_ * P0 + P0 * F_.transpose() + G_ * Qimu_ * G_.transpose();

  auto Ktot{(K1 + 2.0 * (K2 + K3) + K4) / 6.0};
  auto FK{(FK1 + 2.0 * (FK2 + FK3) + FK4) / 6.0};
  auto PK{(PK1 + 2.0 * (PK2 + PK3) + PK4) / 6.0};

  // apply the aggregated difference to state
  gyro_accel = gyro_accel0 + dt * slope;
  ComposeMotion(X_, Ktot, gyro_accel, dt);

  F_.setIdentity();
  F_ = F_ + FK * dt;

  P_.block<kMotionSize, kMotionSize>(0, 0) =
      P_.block<kMotionSize, kMotionSize>(0, 0) + PK * dt;
  // Deferred to one application per image; see
  // `AccumulateMotionStructureCorrelation`.
  AccumulateMotionStructureCorrelation();
}

} // namespace xivo
