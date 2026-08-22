// Gravity (initial attitude) initialization, M6.
//
// Run from the repository root: the config path below is relative to it.
//
// `Estimator::InitializeGravity` averages the first `gravity_init_counter`
// accelerometer samples and calls the mean gravity. Its log line calls them
// "stationary accel samples", but on TUM-VI's room sequences the rig is already
// turning at 0.11-0.32 rad/s when the first sample arrives, and that breaks the
// average two ways:
//
//   * a *short* window cannot average away the carrier's own linear
//     acceleration, and
//   * a *long* window averages body-frame vectors across the turn, which smears
//     the direction by roughly |w| * window -- so lengthening the window to fix
//     the first problem creates a bigger second one.
//
// De-rotating each sample into the body frame of the last sample removes the
// smearing, which is what makes a long window usable. These tests pin both
// halves of that claim: the de-rotated average is exact under pure rotation
// (where the plain average is off by degrees), and over a long window it
// averages away a linear acceleration that a short window does not.
//
// The fixture drives the real `InertialMeasInternal` entry point rather than the
// averaging code alone, so it also covers the parallel gyro/timestamp buffers
// and the `gravity_init_counter` trigger.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#define private public

#include "estimator.h"
#include "utils.h"

using namespace xivo;

namespace {

const char *kCfg = "cfg/tumvi_stereo.json";
constexpr number_t kRate = 199.0; // TUM-VI's IMU rate

number_t AngleDeg(const Vec3 &a, const Vec3 &b) {
  const number_t c = a.normalized().dot(b.normalized());
  return 180.0 / M_PI * std::acos(std::max<number_t>(-1.0, std::min<number_t>(1.0, c)));
}

} // namespace

class GravityInitTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto cfg = LoadJson(kCfg);
    ASSERT_FALSE(cfg.isNull())
        << "could not load " << kCfg << "; run tests from the repo root";
    est = CreateSystem(cfg);
    ASSERT_NE(est, nullptr);

    // The recovered direction is only equal to the *measured* one when the
    // accelerometer calibration is the identity and the bias is zero, which is
    // what every shipped config seeds. Say so out loud rather than having the
    // tolerances below quietly absorb a nonzero bias.
    ASSERT_LT((est->imu_.Ca() - Mat3::Identity()).norm(), 1e-12);
    ASSERT_LT(est->X_.ba.norm(), 1e-12);
    ASSERT_FALSE(est->clamp_signals_)
        << "signal clamping would rewrite the synthetic accelerations";

    // The true specific force in the body frame of the *last* buffered sample.
    // That is the frame the state starts propagating in (X.Wsb = 0 => Rsb = I),
    // so it is the frame gravity has to come out in.
    a_final = Vec3{0.6, -0.4, 9.79};
  }

  /** Replay `n` synthetic samples through `InertialMeasInternal` and return the
   * specific-force direction the filter ended up believing in.
   *
   * The rig turns at constant `omega` and feels `a_final` plus, optionally, a
   * zero-mean sinusoidal linear acceleration `lin_amp` of period `lin_period`
   * seconds -- both given in the final body frame. Sample k therefore measures
   * R_kN * (a_final + lin(t_k)).
   */
  Vec3 RunInit(int n, bool derotate, const Vec3 &omega,
               const Vec3 &lin_amp = Vec3::Zero(), number_t lin_period = 1.0) {
    est->gravity_initialized_ = false;
    est->gravity_init_buf_.clear();
    est->gravity_init_gyro_buf_.clear();
    est->gravity_init_time_buf_.clear();
    est->gravity_init_counter_ = n;
    est->gravity_init_derotate_ = derotate;
    est->X_.Rsg = SO3();

    const number_t dt = 1.0 / kRate;
    // R_0N, needed to express the final-frame quantities in frame k.
    const SO3 R_0N = SO3::exp(omega * (n - 1) * dt);
    for (int k = 0; k < n; ++k) {
      const number_t t = k * dt;
      const SO3 R_0k = SO3::exp(omega * t);
      const Vec3 lin = lin_amp * std::sin(2 * M_PI * t / lin_period);
      const Vec3 a_k = R_0k.inverse() * (R_0N * (a_final + lin));
      // Timestamps only ever move forward across the whole fixture, so
      // `GoodTimestamp` never rejects a sample.
      est->InertialMeasInternal(
          timestamp_t(static_cast<int64_t>((ts_base + t) * 1e9)), omega, a_k);
    }
    ts_base += n * dt + 1.0;

    EXPECT_TRUE(est->gravity_initialized_) << "gravity never initialized";
    // InitializeGravity enforces Rsb * accel + Rsg * g == 0 with Rsb = I, so the
    // specific force it settled on is -Rsg * g.
    return -(est->X_.Rsg * est->g_);
  }

  EstimatorPtr est;
  Vec3 a_final;
  number_t ts_base{1.0};
};

TEST_F(GravityInitTest, DerotatedAverageIsExactUnderPureRotation) {
  // With no linear acceleration the de-rotated samples are all equal to the
  // truth, so the mean is the truth -- to integration error only, and with a
  // constant rate the midpoint rule is exact.
  const Vec3 omega{0.1, -0.2, 0.15}; // 0.27 rad/s, the scale seen on room1-6
  EXPECT_LT(AngleDeg(RunInit(200, true, omega), a_final), 1e-6);
}

TEST_F(GravityInitTest, PlainAverageIsSmearedByTheTurn) {
  // The same 1 s window without de-rotation. This is the reason the shipped
  // window is only 20 samples long: lengthening it makes the smearing worse,
  // not better, so the linear acceleration can never be averaged out.
  const Vec3 omega{0.1, -0.2, 0.15};
  const number_t err = AngleDeg(RunInit(200, false, omega), a_final);
  EXPECT_GT(err, 3.0) << "expected degrees of smearing over a 15 deg turn";
  // And the error grows with the window, i.e. it really is the turn and not a
  // fixed offset.
  EXPECT_GT(err, AngleDeg(RunInit(50, false, omega), a_final));
}

TEST_F(GravityInitTest, DerotationChangesNothingWhenActuallyStationary) {
  // The claim is about moving starts. On a genuinely static start the two paths
  // must agree exactly, so turning the option on cannot regress a dataset that
  // does hold still.
  const Vec3 rest = Vec3::Zero();
  const Vec3 with = RunInit(200, true, rest);
  const Vec3 without = RunInit(200, false, rest);
  EXPECT_LT((with - without).norm(), 1e-12);
  EXPECT_LT(AngleDeg(with, a_final), 1e-9);
}

TEST_F(GravityInitTest, LongWindowAveragesAwayLinearAcceleration) {
  // The payoff. A 0.6 m/s^2 sway at 2 cycles per window: over the whole window
  // it integrates to zero, over the first tenth of it it does not. Both arms are
  // de-rotated, so the only difference is the window length -- which is exactly
  // the trade the shipped 20-sample window was stuck on.
  const Vec3 omega{0.1, -0.2, 0.15};
  const Vec3 sway{0.5, 0.3, 0.0};
  const number_t period = 0.5;

  const number_t err_long = AngleDeg(RunInit(200, true, omega, sway, period), a_final);
  const number_t err_short = AngleDeg(RunInit(20, true, omega, sway, period), a_final);
  EXPECT_LT(err_long, 0.05);
  EXPECT_GT(err_short, 1.0);
  EXPECT_LT(err_long, err_short / 10);
}

TEST_F(GravityInitTest, GravityStaysUninitializedBelowTheSampleCount) {
  // The counter is what gates the whole thing; a fencepost error here would
  // silently shorten every window by one sample.
  est->gravity_initialized_ = false;
  est->gravity_init_buf_.clear();
  est->gravity_init_gyro_buf_.clear();
  est->gravity_init_time_buf_.clear();
  est->gravity_init_counter_ = 100;
  est->gravity_init_derotate_ = true;

  const number_t dt = 1.0 / kRate;
  for (int k = 0; k < 99; ++k) {
    est->InertialMeasInternal(
        timestamp_t(static_cast<int64_t>((ts_base + k * dt) * 1e9)),
        Vec3::Zero(), a_final);
    ASSERT_FALSE(est->gravity_initialized_) << "initialized at sample " << k;
  }
  est->InertialMeasInternal(
      timestamp_t(static_cast<int64_t>((ts_base + 99 * dt) * 1e9)), Vec3::Zero(),
      a_final);
  EXPECT_TRUE(est->gravity_initialized_);
  // And the buffers are released, not left holding a window's worth of samples.
  EXPECT_TRUE(est->gravity_init_buf_.empty());
  EXPECT_TRUE(est->gravity_init_gyro_buf_.empty());
  EXPECT_TRUE(est->gravity_init_time_buf_.empty());
}
