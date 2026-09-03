// M2: preintegration and the linear initializer.
//
// The synthetic motion these tests are built on has a *closed form*, so they
// compare against an exact answer rather than against a finer-grained run of the
// same code. It lives in `init_test_fixture.h`, shared with the M3 suite.
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "init_linear.h"
#include "init_preint.h"
#include "init_problem.h"
#include "rodrigues.h"
#include "test/init_test_fixture.h"

using namespace xivo;
using namespace xivo::inittest;

// ---------------------------------------------------------------------------
// preintegration
// ---------------------------------------------------------------------------

TEST(InitPreint, RightJacobianMatchesCentralDifferences) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<number_t> u(-1.5, 1.5);
  for (int trial = 0; trial < 20; ++trial) {
    const Vec3 w{u(rng), u(rng), u(rng)};
    const Mat3 Jr = SO3RightJacobian(w);
    // exp(w + dw) ~= exp(w) exp(Jr dw): recover column j numerically.
    const number_t h = 1e-6;
    for (int j = 0; j < 3; ++j) {
      Vec3 dw = Vec3::Zero();
      dw(j) = h;
      const Mat3 Rp = SO3::exp(w + dw).matrix();
      const Mat3 Rm = SO3::exp(w - dw).matrix();
      const Vec3 up = SO3(Eigen::Quaternion<number_t>(
                              SO3::exp(w).matrix().transpose() * Rp))
                          .log();
      const Vec3 um = SO3(Eigen::Quaternion<number_t>(
                              SO3::exp(w).matrix().transpose() * Rm))
                          .log();
      const Vec3 num = (up - um) / (2 * h);
      EXPECT_LT((num - Jr.col(j)).norm(), 1e-7)
          << "column " << j << " of Jr at w = " << w.transpose();
    }
  }
  // The small-angle branch has to agree with the closed form across the switch.
  const Vec3 tiny{7e-5, -3e-5, 1e-5};
  const number_t th = tiny.norm(), th2 = th * th;
  const Mat3 W = hat(tiny);
  const Mat3 closed = Mat3::Identity() - (1 - std::cos(th)) / th2 * W +
                      (th - std::sin(th)) / (th2 * th) * W * W;
  EXPECT_LT((SO3RightJacobian(tiny) - closed).norm(), 1e-12);
}

TEST(InitPreint, ExactWhenTheSpecificForceIsConstant) {
  // No rotation: the specific force the IMU reports is then a constant vector,
  // linear interpolation between samples is exact, and so the preintegral must
  // be exact to machine precision. This is the test that pins the *algebra* --
  // the sign of gravity, the order of the alpha/beta update, the endpoint
  // handling -- with no discretisation error in the way to hide behind.
  Truth tr;
  tr.a_w = Vec3{0.9, -1.3, 0.5};
  const auto imu = MakeImu(tr, 0, 1.2, 200.0);
  for (number_t T : {0.1, 0.35, 0.5, 1.0}) {
    const Preintegral p = Preintegrate(imu, 0, T, Vec3::Zero(), Vec3::Zero());
    EXPECT_LT((p.beta - tr.BetaExact(0, T)).norm(), 1e-13) << "T = " << T;
    EXPECT_LT((p.alpha - tr.AlphaExact(0, T)).norm(), 1e-13) << "T = " << T;
    EXPECT_LT((p.R - Mat3::Identity()).norm(), 1e-14) << "T = " << T;
    EXPECT_NEAR(p.dt, T, 1e-12);
  }
}

TEST(InitPreint, IsSecondOrderAccurateWhileSpinning) {
  // With the rig spinning, the specific force rotates *within* each sample
  // interval, so averaging the two endpoint readings is only second-order
  // accurate no matter what the rotation scheme does. The claim worth checking
  // is therefore not a tolerance but an order: halving the sample period must
  // quarter the error. A first-order scheme -- which is what using the
  // left-endpoint rotation instead of the midpoint one would give -- would only
  // halve it, and the test would catch that.
  Truth tr;
  tr.omega = Vec3{0.7, -0.4, 1.1}; // ~1.4 rad/s, well outside small angles
  tr.a_w = Vec3{0.9, -1.3, 0.5};
  const number_t T = 1.0;

  number_t prev = -1;
  for (number_t rate : {200.0, 400.0, 800.0, 1600.0}) {
    const auto imu = MakeImu(tr, 0, 1.2, rate);
    const Preintegral p = Preintegrate(imu, 0, T, Vec3::Zero(), Vec3::Zero());
    const number_t err = (p.beta - tr.BetaExact(0, T)).norm();
    EXPECT_LT((p.R - tr.RExact(0, T)).norm(), 1e-13) << "rate = " << rate;
    if (prev > 0)
      EXPECT_NEAR(prev / err, 4.0, 0.35)
          << "convergence order at rate " << rate << ": " << prev << " -> "
          << err;
    prev = err;
  }

  // And the absolute size at the rate that actually ships. EuRoC's
  // accelerometer noise density is ~2e-3 m/s^2/sqrt(Hz), which over a 0.5 s
  // window is ~1.4e-3 m/s of velocity uncertainty, so discretisation has to be
  // far below that to be irrelevant -- which is the real claim.
  const auto imu = MakeImu(tr, 0, 1.2, 200.0);
  const Preintegral p = Preintegrate(imu, 0, 0.5, Vec3::Zero(), Vec3::Zero());
  EXPECT_LT((p.beta - tr.BetaExact(0, 0.5)).norm(), 1e-4);
  EXPECT_LT((p.alpha - tr.AlphaExact(0, 0.5)).norm(), 1e-5);
}

TEST(InitPreint, ReproducesTheFilterPropagation) {
  // The reason the convention is pinned: the same IMU stream, integrated by the
  // preintegrator and by Estimator::ComposeMotion's Euler step, must describe
  // the same trajectory. Written out here rather than trusted, because the sign
  // of `Rsg * g_` is the single easiest thing to get backwards, and getting it
  // backwards produces a solve that converges neatly to an upside-down world.
  Truth tr;
  tr.omega = Vec3{0.3, 0.2, -0.5};
  tr.a_w = Vec3{0.4, 0.7, -0.2};
  tr.v0 = Vec3{0.6, -0.2, 0.1};
  const auto imu = MakeImu(tr, 0, 0.6, 2000.0); // fine, so Euler is accurate

  // Euler-forward exactly as ComposeMotion does it, in the world frame, with
  // Rsb = R(t), Vsb the world velocity and Rsg * g_ = kGw.
  Vec3 T = Vec3::Zero(), V = tr.v0;
  Mat3 R = Mat3::Identity();
  for (size_t k = 1; k < imu.size(); ++k) {
    const number_t dt = imu[k].t - imu[k - 1].t;
    const Vec3 gyro = 0.5 * (imu[k].gyro + imu[k - 1].gyro);
    const Vec3 accel = 0.5 * (imu[k].accel + imu[k - 1].accel);
    const Mat3 Rm = R * SO3::exp(gyro * (0.5 * dt)).matrix();
    T += V * dt + 0.5 * (Rm * accel + kGw) * dt * dt;
    V += (Rm * accel + kGw) * dt;
    R = R * SO3::exp(gyro * dt).matrix();
    R = SO3(Eigen::Quaternion<number_t>(R).normalized()).matrix();
  }

  const number_t Tend = imu.back().t;
  const Preintegral p = Preintegrate(imu, 0, Tend, Vec3::Zero(), Vec3::Zero());
  // v(T) = v0 + g T + beta;  p(T) = v0 T + 0.5 g T^2 + alpha
  const Vec3 v_pre = tr.v0 + kGw * Tend + p.beta;
  const Vec3 p_pre = tr.v0 * Tend + 0.5 * kGw * Tend * Tend + p.alpha;
  // Agreement with the filter's own recursion is the claim; both share the same
  // second-order discretisation, so this is tight.
  EXPECT_LT((v_pre - V).norm(), 1e-9);
  EXPECT_LT((p_pre - T).norm(), 1e-9);
  // Agreement with the exact trajectory is looser by exactly the discretisation
  // error the two have in common -- at 2 kHz, ~4e-8 m/s.
  EXPECT_LT((v_pre - tr.v(Tend)).norm(), 1e-6);
  EXPECT_LT((p_pre - tr.p(Tend)).norm(), 1e-6);
}

TEST(InitPreint, BiasJacobiansMatchCentralDifferences) {
  Truth tr;
  tr.omega = Vec3{0.5, -0.9, 0.3};
  tr.a_w = Vec3{1.1, 0.4, -0.8};
  const auto imu = MakeImu(tr, 0, 0.8, 200.0);
  const Vec3 bg0{0.02, -0.03, 0.08}, ba0{0.05, -0.11, 0.07};
  const number_t T = 0.55;
  const Preintegral p = Preintegrate(imu, 0, T, bg0, ba0);

  const number_t h = 1e-6;
  for (int j = 0; j < 3; ++j) {
    Vec3 d = Vec3::Zero();
    d(j) = h;

    const Preintegral pgp = Preintegrate(imu, 0, T, bg0 + d, ba0);
    const Preintegral pgm = Preintegrate(imu, 0, T, bg0 - d, ba0);
    EXPECT_LT(((pgp.alpha - pgm.alpha) / (2 * h) - p.dalpha_dbg.col(j)).norm(),
              1e-7)
        << "dalpha/dbg column " << j;
    EXPECT_LT(((pgp.beta - pgm.beta) / (2 * h) - p.dbeta_dbg.col(j)).norm(),
              1e-7)
        << "dbeta/dbg column " << j;
    // R(bg + db) ~= R exp(dR_dbg db), so the numerical derivative is taken in
    // the right-tangent space of R, not entrywise.
    const Vec3 lp =
        SO3(Eigen::Quaternion<number_t>(p.R.transpose() * pgp.R)).log();
    const Vec3 lm =
        SO3(Eigen::Quaternion<number_t>(p.R.transpose() * pgm.R)).log();
    EXPECT_LT(((lp - lm) / (2 * h) - p.dR_dbg.col(j)).norm(), 1e-7)
        << "dR/dbg column " << j;

    const Preintegral pap = Preintegrate(imu, 0, T, bg0, ba0 + d);
    const Preintegral pam = Preintegrate(imu, 0, T, bg0, ba0 - d);
    EXPECT_LT(((pap.alpha - pam.alpha) / (2 * h) - p.dalpha_dba.col(j)).norm(),
              1e-7)
        << "dalpha/dba column " << j;
    EXPECT_LT(((pap.beta - pam.beta) / (2 * h) - p.dbeta_dba.col(j)).norm(),
              1e-7)
        << "dbeta/dba column " << j;
  }
}

TEST(InitPreint, FirstOrderCorrectionBeatsIgnoringTheBias) {
  Truth tr;
  tr.omega = Vec3{0.4, 0.25, -0.6};
  tr.a_w = Vec3{0.7, -0.5, 0.3};
  const auto imu = MakeImu(tr, 0, 0.8, 200.0);
  const Vec3 bg0 = Vec3::Zero(), ba0 = Vec3::Zero();
  // EuRoC-sized: 0.08 rad/s of gyro bias, 0.1 m/s^2 of accel bias.
  const Vec3 bg1{-0.003, 0.021, 0.0785}, ba1{-0.025, 0.137, 0.076};
  const number_t T = 0.6;

  const Preintegral at0 = Preintegrate(imu, 0, T, bg0, ba0);
  const Preintegral at1 = Preintegrate(imu, 0, T, bg1, ba1);

  const number_t raw = (at0.alpha - at1.alpha).norm();
  const number_t corrected = (at0.AlphaAt(bg1, ba1) - at1.alpha).norm();
  EXPECT_LT(corrected, 0.02 * raw) << "raw " << raw << " corrected " << corrected;

  const number_t raw_b = (at0.beta - at1.beta).norm();
  const number_t cor_b = (at0.BetaAt(bg1, ba1) - at1.beta).norm();
  EXPECT_LT(cor_b, 0.02 * raw_b);

  const Mat3 dR = at0.RAt(bg1).transpose() * at1.R;
  const number_t ang =
      SO3(Eigen::Quaternion<number_t>(dR)).log().norm();
  const number_t ang_raw =
      SO3(Eigen::Quaternion<number_t>(at0.R.transpose() * at1.R)).log().norm();
  EXPECT_LT(ang, 0.02 * ang_raw);
}

TEST(InitPreint, EndpointsNeedNotLandOnSamples) {
  // A camera frame lands between IMU samples; the interval has to be honoured
  // rather than snapped to the nearest sample, or every frame time picks up up
  // to half a sample period of error -- 2.5 ms on EuRoC. That is not subtle:
  // 2.5 ms of the gravity term alone is 0.025 m/s of beta.
  //
  // Checked with no rotation so the expected value is exact, which isolates the
  // interval arithmetic from the discretisation covered above; the spinning case
  // then repeats it at the looser tolerance discretisation forces.
  const number_t t0 = 0.00137;
  {
    Truth tr;
    tr.a_w = Vec3{0.5, 0.9, -0.4};
    const auto imu = MakeImu(tr, 0, 1.0, 200.0);
    for (number_t T : {0.3021, 0.4437, 0.61234}) {
      const Preintegral p = Preintegrate(imu, t0, T, Vec3::Zero(), Vec3::Zero());
      EXPECT_LT((p.beta - tr.BetaExact(t0, T)).norm(), 1e-13) << "T = " << T;
      EXPECT_LT((p.alpha - tr.AlphaExact(t0, T)).norm(), 1e-13) << "T = " << T;
      EXPECT_NEAR(p.dt, T - t0, 1e-14) << "T = " << T;
    }
  }
  {
    Truth tr;
    tr.omega = Vec3{0.2, -0.1, 0.35};
    tr.a_w = Vec3{0.5, 0.9, -0.4};
    const auto imu = MakeImu(tr, 0, 1.0, 200.0);
    for (number_t T : {0.3021, 0.4437, 0.61234}) {
      const Preintegral p = Preintegrate(imu, t0, T, Vec3::Zero(), Vec3::Zero());
      EXPECT_LT((p.beta - tr.BetaExact(t0, T)).norm(), 1e-5) << "T = " << T;
      EXPECT_LT((p.alpha - tr.AlphaExact(t0, T)).norm(), 1e-6) << "T = " << T;
      EXPECT_LT((p.R - tr.RExact(t0, T)).norm(), 1e-13) << "T = " << T;
    }
  }
}

// ---------------------------------------------------------------------------
// the sphere-constrained quadratic
// ---------------------------------------------------------------------------

namespace {

// Brute force over a near-uniform sphere grid, refined locally. Slow and dumb on
// purpose: it is the only thing here that does not share a line of code with the
// routine under test.
Vec3 BruteForceSphere(const Mat3 &D, const Vec3 &d, number_t r, int n = 900) {
  Vec3 best = Vec3::Zero();
  number_t best_q = std::numeric_limits<number_t>::infinity();
  auto q = [&](const Vec3 &g) { return g.dot(D * g) + 2 * d.dot(g); };
  // Fibonacci sphere.
  const number_t ga = M_PI * (3.0 - std::sqrt(5.0));
  for (int i = 0; i < n; ++i) {
    const number_t z = 1.0 - 2.0 * (i + 0.5) / n;
    const number_t rr = std::sqrt(std::max<number_t>(0, 1 - z * z));
    const number_t th = ga * i;
    const Vec3 u{rr * std::cos(th), rr * std::sin(th), z};
    const number_t v = q(r * u);
    if (v < best_q) {
      best_q = v;
      best = r * u;
    }
  }
  // Local refinement: random walk on the sphere with a shrinking step.
  std::mt19937 rng(17);
  std::normal_distribution<number_t> gs(0, 1);
  for (number_t step = 0.2; step > 1e-9; step *= 0.7) {
    for (int it = 0; it < 400; ++it) {
      Vec3 cand = best + step * r * Vec3{gs(rng), gs(rng), gs(rng)};
      cand *= r / cand.norm();
      const number_t v = q(cand);
      if (v < best_q) {
        best_q = v;
        best = cand;
      }
    }
  }
  return best;
}

} // namespace

TEST(InitSphere, MatchesBruteForceIncludingIndefiniteD) {
  std::mt19937 rng(23);
  std::uniform_real_distribution<number_t> u(-3, 3);
  auto q = [](const Mat3 &D, const Vec3 &d, const Vec3 &g) {
    return g.dot(D * g) + 2 * d.dot(g);
  };
  int indefinite = 0;
  for (int trial = 0; trial < 40; ++trial) {
    Mat3 A;
    for (int i = 0; i < 9; ++i)
      A(i / 3, i % 3) = u(rng);
    // Half positive definite, half indefinite: an indefinite D is where a
    // Newton iteration seeded at the unconstrained solution has no solution to
    // be seeded from at all, and it is exactly what a low-parallax window gives.
    Mat3 D = trial % 2 == 0 ? Mat3(A.transpose() * A + Mat3::Identity())
                            : Mat3(0.5 * (A + A.transpose()));
    Eigen::SelfAdjointEigenSolver<Mat3> es(D);
    if (es.eigenvalues()(0) < 0)
      ++indefinite;
    const Vec3 d{u(rng), u(rng), u(rng)};
    const number_t r = 9.81;

    Vec3 g;
    ASSERT_TRUE(SolveSphereConstrainedQuadratic(D, d, r, &g)) << "trial " << trial;
    EXPECT_NEAR(g.norm(), r, 1e-9) << "trial " << trial;
    const Vec3 gb = BruteForceSphere(D, d, r);
    // The routine must be at least as good as brute force, up to brute force's
    // own resolution.
    EXPECT_LE(q(D, d, g), q(D, d, gb) + 1e-6 * std::abs(q(D, d, gb)) + 1e-6)
        << "trial " << trial << " ours " << q(D, d, g) << " brute "
        << q(D, d, gb);
    EXPECT_LT((g - gb).norm(), 1e-3 * r) << "trial " << trial;
  }
  EXPECT_GT(indefinite, 5) << "the indefinite branch was never exercised";
}

TEST(InitSphere, HandlesTheDegenerateCase) {
  // d orthogonal to the smallest eigendirection: |g(lambda)| stays bounded as
  // lambda approaches lambda_min, so there is no root to find and the minimiser
  // sits on the boundary with a free null component. A root finder that assumes
  // a root exists returns whatever its bracket collapsed to.
  Mat3 D = Vec3{1.0, 4.0, 7.0}.asDiagonal();
  const Vec3 d{0.0, 0.5, -0.25}; // no component on eigenvector 0
  const number_t r = 9.81;
  Vec3 g;
  ASSERT_TRUE(SolveSphereConstrainedQuadratic(D, d, r, &g));
  EXPECT_NEAR(g.norm(), r, 1e-9);
  const Vec3 gb = BruteForceSphere(D, d, r);
  auto q = [&](const Vec3 &x) { return x.dot(D * x) + 2 * d.dot(x); };
  EXPECT_LE(q(g), q(gb) + 1e-6);
  // The free component is along eigenvector 0, and it is nearly all of g.
  EXPECT_GT(std::abs(g(0)), 0.95 * r);

  // d = 0 exactly: every point of the sphere in the lambda_min eigenspace is a
  // global minimiser; the routine must still return one of them, on the sphere.
  Vec3 g2;
  ASSERT_TRUE(SolveSphereConstrainedQuadratic(D, Vec3::Zero(), r, &g2));
  EXPECT_NEAR(g2.norm(), r, 1e-9);
  EXPECT_NEAR(std::abs(g2(0)), r, 1e-6);
}

TEST(InitSphere, RejectsNonsense) {
  Vec3 g;
  EXPECT_FALSE(SolveSphereConstrainedQuadratic(Mat3::Identity(), Vec3::Zero(),
                                               0.0, &g));
  EXPECT_FALSE(SolveSphereConstrainedQuadratic(Mat3::Identity(), Vec3::Zero(),
                                               -1.0, &g));
  EXPECT_FALSE(SolveSphereConstrainedQuadratic(Mat3::Identity(), Vec3::Zero(),
                                               1.0, nullptr));
}

// ---------------------------------------------------------------------------
// Stage A
// ---------------------------------------------------------------------------

TEST(InitLinear, RecoversVelocityAndGravityExactly) {
  // Exact preintegrals, no noise, no bias: the linear system is consistent, so
  // anything but an exact answer is an arithmetic error. Run at the 1.5 s window
  // that the real-data sweep settled on (notes-dyninit/m2-linear.md).
  Truth tr;
  tr.omega = Vec3{0.25, -0.4, 0.6};
  tr.a_w = Vec3{0.8, -0.6, 0.35};
  tr.v0 = Vec3{0.9, -0.35, 0.15}; // MH_01-sized
  const auto pts = MakeScene(120);
  const InitCamera cam = EurocishCam();
  const number_t t0 = 1.1, span = 1.5;

  const InitProblem prob =
      MakeProblem(tr, pts, cam, t0, span, 31, Vec3::Zero(), Vec3::Zero(), 0, 3,
                  /*exact_pre=*/true);
  const LinearInitResult res = SolveLinearInit(prob);
  ASSERT_TRUE(res.ok) << res.why;

  const Mat3 R0 = tr.R(t0);
  const Vec3 v_true = R0.transpose() * tr.v(t0);
  const Vec3 g_true = R0.transpose() * kGw;
  EXPECT_LT((res.v - v_true).norm(), 1e-9)
      << "got " << res.v.transpose() << " want " << v_true.transpose();
  EXPECT_LT((res.g - g_true).norm(), 1e-9)
      << "got " << res.g.transpose() << " want " << g_true.transpose();
  EXPECT_NEAR(res.g.norm(), kG, 1e-12); // exact by construction
  EXPECT_LT(res.residual, 1e-10);

  // Every track with two or more frames must be used, and no more. Asserting
  // against `pts.size()` instead would be wrong: `MakeProblem` drops
  // observations behind the camera, so a track can legitimately fall below two
  // frames and be dropped.
  const std::vector<int> nfr = prob.TrackFrameCounts();
  int expect_used = 0;
  for (int c : nfr)
    if (c >= 2)
      ++expect_used;
  EXPECT_EQ(res.tracks_used, expect_used);

  // The features, which are what the depth-weighted rows are really about.
  // Only the used ones: a dropped track is deliberately left at zero, so
  // including it would measure the true position's magnitude, not an error.
  const Vec3 p0 = tr.p(t0);
  number_t worst = 0;
  int checked = 0;
  for (size_t i = 0; i < pts.size(); ++i) {
    if (!res.used[i])
      continue;
    worst = std::max(worst,
                     (res.features[i] - R0.transpose() * (pts[i] - p0)).norm());
    ++checked;
  }
  EXPECT_GT(checked, 100);
  EXPECT_LT(worst, 1e-8) << "worst feature error " << worst;

  // With the real preintegrator in the loop instead of the closed form, the only
  // added error is its O(dt^2) discretization. Measured 4e-5 m/s at 200 Hz and
  // flat in span -- roughly 1500x below the pixel-noise term at this window
  // length, which is why the preintegrator is not the thing to improve.
  const LinearInitResult integrated = SolveLinearInit(
      MakeProblem(tr, pts, cam, t0, span, 31, Vec3::Zero(), Vec3::Zero()));
  ASSERT_TRUE(integrated.ok) << integrated.why;
  EXPECT_LT((integrated.v - v_true).norm(), 1e-4)
      << "end-to-end error with integrated preintegrals";
}

TEST(InitLinear, GravityMagnitudeIsEnforcedNotFitted) {
  // With a bias the solve cannot see, an unconstrained gravity absorbs the error
  // into its *magnitude* and reports a smaller residual than the truth has. The
  // constraint is what stops that, so check the constrained answer is on the
  // sphere and the unconstrained one is not -- the second half is why the
  // constraint has to exist at all.
  Truth tr;
  tr.omega = Vec3{0.3, 0.15, -0.2};
  tr.a_w = Vec3{0.5, 0.4, -0.3};
  tr.v0 = Vec3{0.7, 0.1, -0.2};
  tr.bg = Vec3{-0.003, 0.021, 0.0785};
  tr.ba = Vec3{-0.025, 0.137, 0.076};
  const auto pts = MakeScene(120);
  const InitProblem prob = MakeProblem(tr, pts, EurocishCam(), 1.1, 0.5, 11,
                                       Vec3::Zero(), Vec3::Zero());
  const LinearInitResult res = SolveLinearInit(prob);
  ASSERT_TRUE(res.ok) << res.why;
  EXPECT_NEAR(res.g.norm(), kG, 1e-9);

  // Unconstrained least squares on the same problem, by dense QR over the whole
  // 3N+6 system: independent of everything in init_linear.cpp.
  const int N = prob.num_tracks;
  Eigen::MatrixXd A(static_cast<int>(prob.obs.size()) * 2, 3 * N + 6);
  Eigen::VectorXd b(static_cast<int>(prob.obs.size()) * 2);
  A.setZero();
  int row = 0;
  for (const auto &o : prob.obs) {
    const InitFrame &fr = prob.frames[o.frame];
    const InitCamera &cm = prob.cams[o.cam];
    const number_t dt = fr.t - prob.frames.front().t;
    Eigen::Matrix<number_t, 2, 3> Hp;
    Hp << 1, 0, -o.xn(0), 0, 1, -o.xn(1);
    const Eigen::Matrix<number_t, 2, 3> HRc = Hp * cm.Rbc.transpose();
    const Eigen::Matrix<number_t, 2, 3> Y = HRc * fr.pre.R.transpose();
    A.block<2, 3>(row, 3 * o.track) = Y;
    A.block<2, 3>(row, 3 * N) = -dt * Y;
    A.block<2, 3>(row, 3 * N + 3) = -0.5 * dt * dt * Y;
    b.segment<2>(row) = Y * fr.pre.alpha + HRc * cm.Tbc;
    row += 2;
  }
  const Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
  const number_t g_free = x.tail<3>().norm();
  EXPECT_GT(std::abs(g_free - kG), 0.05)
      << "unconstrained |g| came out at " << g_free
      << ", so this problem does not actually exercise the constraint";
}

TEST(InitLinear, RefusesAWindowWithNoParallaxAndUsesTheLeverArmWhenThereIs) {
  Truth tr;
  tr.omega = Vec3{0.3, -0.5, 0.2};
  const auto pts = MakeScene(120);

  // A camera at the body origin under pure rotation genuinely has no parallax:
  // every bearing rotates, no depth is observable, and every per-track 3x3 block
  // is rank 2. The solve must decline rather than return a confident number.
  InitCamera centred;
  centred.Rbc = EurocishCam().Rbc;
  centred.Tbc = Vec3::Zero();
  const LinearInitResult none = SolveLinearInit(MakeProblem(
      tr, pts, centred, 1.1, 1.5, 31, Vec3::Zero(), Vec3::Zero()));
  EXPECT_FALSE(none.ok);
  EXPECT_LT(none.tracks_used, 4);

  // With the real 6.9 cm offset it is a different problem, and the first version
  // of this test had it wrong: rotating the *body* swings the camera through an
  // arc, so the camera does translate and there is real parallax to triangulate
  // from. The solve should succeed and recover v = 0.
  const LinearInitResult lever = SolveLinearInit(MakeProblem(
      tr, pts, EurocishCam(), 1.1, 1.5, 31, Vec3::Zero(), Vec3::Zero()));
  ASSERT_TRUE(lever.ok) << lever.why;
  EXPECT_GT(lever.tracks_used, 100);
  // 1e-4, not 0: these are integrated preintegrals, so the floor here is the
  // preintegrator's O(dt^2) discretization error, measured at ~2e-5.
  EXPECT_LT(lever.v.norm(), 1e-4)
      << "pure rotation, so the true velocity is zero; got "
      << lever.v.transpose();
  EXPECT_GT(lever.g_cond, 1e-6)
      << "the lever arm does constrain gravity, so this should not look "
         "degenerate";
}

TEST(InitLinear, RefusesSingleFrameTracks) {
  Truth tr;
  tr.omega = Vec3{0.1, 0.2, -0.1};
  tr.v0 = Vec3{0.5, 0, 0};
  const auto pts = MakeScene(60);
  InitProblem prob = MakeProblem(tr, pts, EurocishCam(), 1.1, 0.5, 11,
                                 Vec3::Zero(), Vec3::Zero());
  // Keep only frame 0's observations: every track is now seen once.
  std::vector<InitObservation> keep;
  for (const auto &o : prob.obs)
    if (o.frame == 0)
      keep.push_back(o);
  prob.obs = keep;
  const LinearInitResult res = SolveLinearInit(prob);
  EXPECT_FALSE(res.ok);
  EXPECT_EQ(res.tracks_used, 0);
}

TEST(InitLinear, LongerWindowsBeatPixelNoise) {
  // Pixel noise is the term a longer window actually fixes, and this is the
  // measurement that set the shipped window length. 0.3 px at EuRoC's ~458 px
  // focal length is what survives the tracker's forward-backward gate.
  //
  // Measured (6 seeds, 150 features, 20 frames/s):
  //     0.50 s  1.14 m/s      1.50 s  0.060 m/s
  //     0.75 s  0.44 m/s      2.00 s  0.036 m/s
  //     1.00 s  0.20 m/s
  // The 0.5 s window the plan originally specified costs more than half the true
  // speed; 1.5 s costs 7% of it. That is a 19x difference from one knob.
  Truth tr;
  tr.omega = Vec3{0.25, -0.4, 0.6};
  tr.a_w = Vec3{0.8, -0.6, 0.35};
  tr.v0 = Vec3{0.9, -0.35, 0.15};
  const auto pts = MakeScene(150);
  const Mat3 R0 = tr.R(1.1);
  const Vec3 v_true = R0.transpose() * tr.v(1.1);
  const number_t sigma = 0.3 / 458.0;

  auto worst_over_seeds = [&](number_t span, int nframes) {
    number_t worst = 0;
    for (unsigned seed = 1; seed <= 6; ++seed) {
      const InitProblem prob = MakeProblem(tr, pts, EurocishCam(), 1.1, span,
                                           nframes, Vec3::Zero(), Vec3::Zero(),
                                           sigma, seed);
      const LinearInitResult res = SolveLinearInit(prob);
      EXPECT_TRUE(res.ok) << res.why;
      if (res.ok)
        worst = std::max(worst, (res.v - v_true).norm());
    }
    return worst;
  };

  const number_t e_short = worst_over_seeds(0.5, 11);
  const number_t e_mid = worst_over_seeds(1.0, 21);
  const number_t e_long = worst_over_seeds(1.5, 31);

  EXPECT_LT(e_mid, e_short) << "span 1.0 s (" << e_mid << ") should beat 0.5 s ("
                            << e_short << ")";
  EXPECT_LT(e_long, e_mid) << "span 1.5 s (" << e_long << ") should beat 1.0 s ("
                           << e_mid << ")";
  EXPECT_LT(e_long, e_short / 5)
      << "lengthening the window from 0.5 s to 1.5 s should buy much more than "
         "a factor of 5; got " << e_short << " -> " << e_long;
  EXPECT_LT(e_long, 0.15) << "velocity error at the shipped window length "
                          << e_long;
}

TEST(InitLinear, BiasErrorIsWhatLimitsIt) {
  // Stage A's velocity error is set by the bias it holds at the prior, not by
  // the solver -- which is the whole reason Stage B exists. Unlike the noise
  // term above, this one is a systematic error in the estimator rather than a
  // variance, so a longer window barely helps: measured 2.09 m/s at 0.5 s and
  // still 1.79 m/s at 2.0 s on this fixture.
  //
  // The absolute size is fixture-specific and adversarial: `a_w` and `omega` are
  // held *constant* across the window here, whereas real hand-carried motion
  // averages both down. On real EuRoC windows the same bias costs 0.11-0.23 m/s
  // (notes-dyninit/m2-linear.md). So this test pins the three structural facts
  // -- large, span-insensitive, and removable by a correct prior -- and not a
  // number that would only mislead.
  Truth tr;
  tr.omega = Vec3{0.25, -0.4, 0.6};
  tr.a_w = Vec3{0.8, -0.6, 0.35};
  tr.v0 = Vec3{0.9, -0.35, 0.15};
  const auto pts = MakeScene(120);
  const Mat3 R0 = tr.R(1.1);
  const Vec3 v_true = R0.transpose() * tr.v(1.1);

  const Vec3 bg_euroc{-0.003, 0.021, 0.0785};
  const Vec3 ba_euroc{-0.025, 0.137, 0.076};

  const LinearInitResult clean = SolveLinearInit(MakeProblem(
      tr, pts, EurocishCam(), 1.1, 1.5, 31, Vec3::Zero(), Vec3::Zero()));
  ASSERT_TRUE(clean.ok) << clean.why;
  EXPECT_LT((clean.v - v_true).norm(), 1e-4);

  tr.bg = bg_euroc;
  tr.ba = ba_euroc;
  const LinearInitResult short_win = SolveLinearInit(MakeProblem(
      tr, pts, EurocishCam(), 1.1, 0.5, 11, Vec3::Zero(), Vec3::Zero()));
  const LinearInitResult long_win = SolveLinearInit(MakeProblem(
      tr, pts, EurocishCam(), 1.1, 2.0, 41, Vec3::Zero(), Vec3::Zero()));
  ASSERT_TRUE(short_win.ok) << short_win.why;
  ASSERT_TRUE(long_win.ok) << long_win.why;
  const number_t e_short = (short_win.v - v_true).norm();
  const number_t e_long = (long_win.v - v_true).norm();

  EXPECT_GT(e_short, 1.0) << "the planted bias barely registered: " << e_short;
  EXPECT_GT(e_long, 0.7 * e_short)
      << "a 4x longer window cut the bias error from " << e_short << " to "
      << e_long << "; if that is real, the bias is not the structural error "
                   "this test claims it is";

  // And a correct prior removes it entirely, which is the mechanism rather than
  // a coincidence: nothing else in the problem changed.
  const LinearInitResult known = SolveLinearInit(MakeProblem(
      tr, pts, EurocishCam(), 1.1, 1.5, 31, bg_euroc, ba_euroc));
  ASSERT_TRUE(known.ok) << known.why;
  EXPECT_LT((known.v - v_true).norm(), 1e-4)
      << "with the right prior the error should be back to the "
         "discretization floor";
}

TEST(InitLinear, TheLinearCostIsBimodalAndTheAccelPriorBreaksTheTie) {
  // The most consequential thing measured in M2, and it is a property of the
  // cost rather than of the solver: with an accelerometer bias held at zero, the
  // depth-scaled objective acquires a second minimum roughly 40 degrees away in
  // gravity direction whose cost is *lower* than the one near the truth by about
  // one part in 1e4. The sphere solve then correctly returns a physically wrong
  // answer, and the velocity is off by more than 10 m/s at every window length.
  //
  // No conditioning fix applies, because nothing in this cost distinguishes the
  // two branches. The accelerometer does, so `PriorMode::Check` uses its mean
  // direction purely as a discriminator. Never seen on real EuRoC data (0 flips
  // in 11 sequences), which is why the guard has to be tested here.
  Truth tr;
  tr.omega = Vec3{0.25, -0.4, 0.6};
  tr.a_w = Vec3{0.8, -0.6, 0.35};
  tr.v0 = Vec3{0.9, -0.35, 0.15};
  tr.ba = Vec3{-0.025, 0.137, 0.076};
  const auto pts = MakeScene(150);
  const Mat3 R0 = tr.R(1.1);
  const Vec3 v_true = R0.transpose() * tr.v(1.1);
  const Vec3 g_true = R0.transpose() * kGw;
  const InitProblem prob = MakeProblem(tr, pts, EurocishCam(), 1.1, 1.5, 31,
                                       Vec3::Zero(), Vec3::Zero());

  const LinearInitResult raw = SolveLinearInit(prob);
  ASSERT_TRUE(raw.ok) << raw.why;
  EXPECT_GT((raw.v - v_true).norm(), 5.0)
      << "this fixture is supposed to trigger the flip";
  EXPECT_GT(Angle(raw.g, g_true), 0.5);

  // The solver is not at fault: its answer beats the truth on the very objective
  // it is asked to minimise, and it also beats an independent scan of the sphere.
  auto cost = [&](const Vec3 &g) {
    return number_t(g.transpose() * raw.g_hess * g) - 2.0 * raw.g_rhs.dot(g);
  };
  EXPECT_LT(cost(raw.g), cost(g_true))
      << "the solve returned a *worse* point than the truth, so this is a solver "
         "bug and not the cost's bimodality";
  number_t best_scan = std::numeric_limits<number_t>::max();
  for (int i = 0; i < 20000; ++i) {
    const number_t z = 1.0 - 2.0 * (i + 0.5) / 20000.0;
    const number_t ph = i * 2.39996323; // golden-angle spiral
    const number_t rr = std::sqrt(std::max<number_t>(0, 1 - z * z));
    best_scan = std::min(
        best_scan,
        cost(Vec3{rr * std::cos(ph), rr * std::sin(ph), z} * kG));
  }
  EXPECT_LE(cost(raw.g), best_scan + 1e-6 * std::abs(best_scan));

  // With the accelerometer mean as a tie-breaker the flip is caught and the
  // answer falls back to something usable. The prior's own error here is the
  // full atan(|a_w|/9.81) = 6.2 deg, because this fixture holds `a_w` constant;
  // that is why the fallback is ~1.4 m/s rather than ~0.
  LinearInitOptions opt;
  opt.prior_mode = LinearInitOptions::PriorMode::Check;
  opt.gravity_prior = AccelMeanGravity(tr, 1.1, 1.5, Vec3::Zero(), Vec3::Zero());
  const LinearInitResult checked = SolveLinearInit(prob, opt);
  ASSERT_TRUE(checked.ok) << checked.why;
  EXPECT_TRUE(checked.gravity_flipped)
      << "prior disagreement was " << checked.prior_disagreement << " rad";
  EXPECT_LT((checked.v - v_true).norm(), 2.5)
      << "falling back to the prior should be far better than 10+ m/s";
  EXPECT_LT(Angle(checked.g, g_true), 0.2);

  // And the guard must not fire on a clean problem: same motion, no bias.
  Truth ok_tr = tr;
  ok_tr.ba = Vec3::Zero();
  const InitProblem ok_prob = MakeProblem(
      ok_tr, pts, EurocishCam(), 1.1, 1.5, 31, Vec3::Zero(), Vec3::Zero());
  LinearInitOptions ok_opt;
  ok_opt.prior_mode = LinearInitOptions::PriorMode::Check;
  ok_opt.gravity_prior =
      AccelMeanGravity(ok_tr, 1.1, 1.5, Vec3::Zero(), Vec3::Zero());
  const LinearInitResult ok_res = SolveLinearInit(ok_prob, ok_opt);
  ASSERT_TRUE(ok_res.ok) << ok_res.why;
  EXPECT_FALSE(ok_res.gravity_flipped)
      << "false positive: disagreement " << ok_res.prior_disagreement << " rad";
  EXPECT_LT((ok_res.v - v_true).norm(), 1e-4)
      << "a non-firing guard must leave the answer untouched";
}
