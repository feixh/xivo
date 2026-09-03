// M3: the initialization bundle adjustment.
//
// Stage B has two ways to be wrong that a converging solver hides completely. It
// can minimize the right cost with a wrong derivative, in which case it still
// descends -- more slowly, or to a nearby point -- and nothing in its output says
// so. Or it can have a residual that does not actually depend on the biases, in
// which case it reports the seed's biases with a beautifully small cost. So the
// tests here are built to catch exactly those:
//
//  * every column of the analytic Jacobian is checked against a central
//    difference of the same residual vector the solver whitens;
//  * the bias is planted at EuRoC's magnitude and has to be *recovered*, not
//    preserved;
//  * the gauge freedom is checked to be a freedom (the cost is invariant) rather
//    than assumed;
//  * and convergence is checked against a synthetic problem whose exact answer is
//    known in closed form, so "converged" and "correct" are separable claims.
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "init_ba.h"
#include "init_linear.h"
#include "init_preint.h"
#include "init_problem.h"
#include "rodrigues.h"
#include "test/init_test_fixture.h"

using namespace xivo;
using namespace xivo::inittest;

namespace {

/** The exact state, in a world frame that already satisfies Stage B's
 *  conventions: the fixture's world is gravity-aligned (`kGw = [0,0,-9.81]`), so
 *  only the origin has to move to frame 0's position. */
BAState TrueState(const Truth &tr, const std::vector<Vec3> &pts, number_t t0,
                  number_t span, int nframes) {
  BAState st;
  st.gravity = kG;
  const Vec3 p0 = tr.p(t0);
  for (int k = 0; k < nframes; ++k) {
    const number_t t = t0 + span * k / std::max(1, nframes - 1);
    st.R.push_back(tr.R(t));
    st.p.push_back(tr.p(t) - p0);
    st.v.push_back(tr.v(t));
  }
  st.bg = tr.bg;
  st.ba = tr.ba;
  for (const auto &q : pts)
    st.f.push_back(q - p0);
  st.used.assign(pts.size(), 1);
  return st;
}

/** A seed that is wrong the way a real Stage A seed is wrong: gravity tilted,
 *  velocity offset, biases unknown.
 *
 *  The tilt is applied as a *left* (world) rotation about world x and y, which
 *  leaves frame 0's world yaw untouched. That matters: yaw is the gauge, pinned
 *  to the seed, so a seed with a different yaw from the truth would make "the
 *  solver should reach the truth" false for a reason that has nothing to do with
 *  the solver. Tilt about x/y is not a gauge direction -- gravity is fixed in W --
 *  so the cost really does rise and really does have to come back down. */
BAState TiltedSeed(const BAState &truth, const Vec3 &tilt_xy, const Vec3 &dv,
                   const Vec3 &bg, const Vec3 &ba) {
  BAState st = truth;
  const Mat3 Q = SO3::exp(Vec3{tilt_xy(0), tilt_xy(1), 0}).matrix();
  for (int k = 0; k < truth.num_frames(); ++k) {
    st.R[k] = Q * truth.R[k];
    st.p[k] = Q * truth.p[k];
    st.v[k] = Q * truth.v[k] + dv;
  }
  for (int n = 0; n < truth.num_tracks(); ++n)
    st.f[n] = Q * truth.f[n];
  st.p[0].setZero();
  st.bg = bg;
  st.ba = ba;
  return st;
}

/** Perturb the parameter that dense Jacobian column `col` differentiates, using
 *  exactly the update rule `SolveStep` applies. */
BAState Bump(const BAState &st, const std::vector<int> &tcol, int M, int col,
             number_t h) {
  BAState o = st;
  const int K = st.num_frames();
  Vec3 e = Vec3::Zero();
  if (col < 9 * K) {
    const int k = col / 9, off = col % 9;
    e(off % 3) = h;
    if (off < 3)
      o.R[k] = st.R[k] * SO3::exp(e).matrix();
    else if (off < 6)
      o.p[k] += e;
    else
      o.v[k] += e;
  } else if (col < M) {
    const int off = col - 9 * K;
    e(off % 3) = h;
    if (off < 3)
      o.bg += e;
    else
      o.ba += e;
  } else {
    for (size_t n = 0; n < tcol.size(); ++n)
      if (tcol[n] >= 0 && col >= tcol[n] && col < tcol[n] + 3) {
        o.f[n](col - tcol[n]) += h;
        break;
      }
  }
  return o;
}

Truth EurocishTruth(const Vec3 &bg = Vec3::Zero(),
                    const Vec3 &ba = Vec3::Zero()) {
  Truth tr;
  tr.omega = Vec3{0.35, -0.22, 0.61};
  tr.a_w = Vec3{0.55, -0.72, 0.34};
  tr.v0 = Vec3{1.1, 0.45, -0.2};
  tr.bg = bg;
  tr.ba = ba;
  return tr;
}

// EuRoC's own solved biases at the start of MH_01, rounded: the magnitudes the
// planted-bias tests have to cope with.
const Vec3 kEurocBg{-0.0027, 0.0213, 0.0785};
const Vec3 kEurocBa{-0.0246, 0.1237, 0.0764};

number_t PixelSigma() { return 0.3 / 458.654; } // 0.3 px in normalized units

} // namespace

// ---------------------------------------------------------------------------
// the derivatives
// ---------------------------------------------------------------------------

TEST(InitBA, AnalyticJacobiansMatchCentralDifferences) {
  const Truth tr = EurocishTruth(kEurocBg, kEurocBa);
  const auto pts = MakeScene(14, 7);
  const InitCamera cam = EurocishCam();
  const number_t t0 = 0.4, span = 0.3;
  const int K = 4;
  // Integrated (not exact) preintegrals at a *wrong* bias prior, because that is
  // the case where the bias Jacobians are load-bearing: at the right prior the
  // correction terms are multiplied by zero and a sign error in them is invisible.
  const InitProblem prob =
      MakeProblem(tr, pts, cam, t0, span, K, Vec3::Zero(), Vec3::Zero());

  BAOptions opt;
  // No robust loss here. Cauchy reweighting makes the residual a different
  // function of the state than the one the Jacobian differentiates -- that is the
  // standard IRLS approximation, not a bug -- so a finite-difference check has to
  // be run on the underlying least-squares residual. The reweighting is exercised
  // instead by `CauchyContainsATrackThatJumped`.
  opt.cauchy_c = 0;
  opt.sigma_pix = 1.0;
  opt.sigma_bg_prior = 0.01; // exercise both prior families too
  opt.sigma_ba_prior = 0.05;
  opt.sigma_yaw = 1e-3;

  const BAState truth = TrueState(tr, pts, t0, span, K);
  // Evaluate away from the optimum: at a zero residual every column of J'r
  // vanishes and a wrong Jacobian would still look right.
  const BAState ref =
      TiltedSeed(truth, Vec3{0.04, -0.03, 0}, Vec3{0.12, -0.08, 0.05},
                 Vec3{0.01, -0.02, 0.03}, Vec3{0.05, -0.03, 0.02});
  BAState at = ref;
  at.p[2] += Vec3{0.01, -0.02, 0.015};
  at.R[1] = at.R[1] * SO3::exp(Vec3{0.01, 0.02, -0.015}).matrix();
  at.f[3] += Vec3{0.05, -0.03, 0.08};

  VecX r0;
  MatX J;
  std::vector<int> tcol;
  ASSERT_TRUE(InitBALinearize(prob, at, opt, ref, &r0, &J, &tcol));
  const int M = 9 * K + 6;
  ASSERT_EQ(J.rows(), r0.size());
  ASSERT_GT(J.cols(), M); // at least one track took part

  const number_t h = 1e-6;
  number_t worst = 0;
  int worst_col = -1;
  for (int c = 0; c < J.cols(); ++c) {
    VecX rp, rm;
    MatX dummy;
    ASSERT_TRUE(InitBALinearize(prob, Bump(at, tcol, M, c, h), opt, ref, &rp,
                                &dummy, nullptr));
    ASSERT_TRUE(InitBALinearize(prob, Bump(at, tcol, M, c, -h), opt, ref, &rm,
                                &dummy, nullptr));
    ASSERT_EQ(rp.size(), r0.size()) << "column " << c << " changed the row set";
    ASSERT_EQ(rm.size(), r0.size()) << "column " << c << " changed the row set";
    const VecX num = (rp - rm) / (2 * h);
    // Relative, because the whitened families differ in scale by four orders of
    // magnitude: an alpha row carries 1/sigma_alpha ~ 4e4, a pixel row ~ 1.
    const number_t scale = std::max<number_t>(1, J.col(c).cwiseAbs().maxCoeff());
    const number_t err = (num - J.col(c)).cwiseAbs().maxCoeff() / scale;
    if (err > worst) {
      worst = err;
      worst_col = c;
    }
  }
  EXPECT_LT(worst, 1e-6) << "worst relative Jacobian error is in column "
                         << worst_col << " of " << J.cols();
}

// ---------------------------------------------------------------------------
// the gauge
// ---------------------------------------------------------------------------

TEST(InitBA, WorldYawIsAGaugeFreedomAndTiltIsNot) {
  const Truth tr = EurocishTruth(kEurocBg, kEurocBa);
  const auto pts = MakeScene(20, 9);
  const InitCamera cam = EurocishCam();
  const number_t t0 = 0.4, span = 0.5;
  const int K = 6;
  const InitProblem prob = MakeProblem(tr, pts, cam, t0, span, K, tr.bg, tr.ba);
  const BAState truth = TrueState(tr, pts, t0, span, K);

  BAOptions opt;
  opt.cauchy_c = 0;
  opt.sigma_yaw = 0; // the prior is exactly what would hide the freedom
  const number_t c0 = InitBACost(prob, truth, opt, truth);

  // A rotation of the *whole* solution about world z leaves every residual
  // untouched, because Q g_W = g_W. That is why one prior row, of any weight, is
  // enough to pin it, and why the pin does not bias the answer.
  for (number_t yaw : {0.3, -1.1, 2.5}) {
    const Mat3 Q = SO3::exp(Vec3{0, 0, yaw}).matrix();
    BAState st = truth;
    for (int k = 0; k < K; ++k) {
      st.R[k] = Q * truth.R[k];
      st.p[k] = Q * truth.p[k];
      st.v[k] = Q * truth.v[k];
    }
    for (int n = 0; n < truth.num_tracks(); ++n)
      st.f[n] = Q * truth.f[n];
    EXPECT_NEAR(InitBACost(prob, st, opt, truth), c0, 1e-6)
        << "world yaw of " << yaw << " changed the cost";
  }

  // The same rotation about world x is *not* free: gravity is fixed in W, so
  // tilting everything makes the IMU residual pay for it. Without this half the
  // test above would pass on a cost that had simply stopped seeing gravity.
  {
    const Mat3 Q = SO3::exp(Vec3{0.05, 0, 0}).matrix();
    BAState st = truth;
    for (int k = 0; k < K; ++k) {
      st.R[k] = Q * truth.R[k];
      st.p[k] = Q * truth.p[k];
      st.v[k] = Q * truth.v[k];
    }
    for (int n = 0; n < truth.num_tracks(); ++n)
      st.f[n] = Q * truth.f[n];
    EXPECT_GT(InitBACost(prob, st, opt, truth), c0 + 1);
  }
}

// ---------------------------------------------------------------------------
// convergence on data with an exact answer
// ---------------------------------------------------------------------------

TEST(InitBA, RecoversTheExactStateFromATiltedSeed) {
  const Truth tr = EurocishTruth(kEurocBg, kEurocBa);
  const auto pts = MakeScene(70, 11);
  const InitCamera cam = EurocishCam();
  const number_t t0 = 0.4, span = 1.0;
  const int K = 11;
  // exact_pre: the preintegrals a perfect integrator at the true bias would
  // produce, so the cost at the true state is exactly zero and "did it converge"
  // and "is it correct" are the same question.
  const InitProblem prob = MakeProblem(tr, pts, cam, t0, span, K, tr.bg, tr.ba,
                                       0, 3, /*exact_pre=*/true);
  const BAState truth = TrueState(tr, pts, t0, span, K);

  BAOptions opt;
  opt.cauchy_c = 0;
  opt.max_iterations = 60;
  // Opt out of the shipped `sigma_ba_prior`. The default holds `ba` at the seed
  // because a real 1.5 s window cannot see it (see init_ba.h); this test asks the
  // narrower question of whether the *machinery* -- the bias Jacobians and the
  // first-order preintegral correction -- can recover it when the data does
  // determine it. Both questions matter and they need different settings.
  opt.sigma_ba_prior = 0;
  ASSERT_LT(InitBACost(prob, truth, opt, truth), 1e-12)
      << "the fixture's exact state is not a zero of the cost, so nothing below "
         "tests the solver";

  const BAState seed =
      TiltedSeed(truth, Vec3{0.06, -0.05, 0}, Vec3{0.3, -0.25, 0.15},
                 Vec3::Zero(), Vec3::Zero());
  const BAResult res = SolveInitBA(prob, seed, opt);
  ASSERT_TRUE(res.ok) << res.why;
  EXPECT_LT(res.cost_final, res.cost_init);

  // The plan's M3 gate asked for the velocity to 1e-6 here. That gate was wrong,
  // and this test records why rather than being loosened until it passes.
  //
  // Measured: the solver drives the cost to 1.41e-22, *below* the 1.76e-22 the
  // exact truth scores, with a reprojection RMS of 1.6e-13 px -- both are pure
  // double-precision roundoff on noiseless data, so to the last bit the state it
  // found fits the data at least as well as the state that generated it. Yet the
  // velocity is 0.0094 m/s off. The two facts are consistent because the window
  // has a nearly flat direction: a global scale change on (p, v, f) can be paid
  // for by an accelerometer bias along the trajectory's mean acceleration. Both
  // signatures are present -- the recovered scale is 1.00686, and dividing it out
  // leaves 5.2e-4 m/s -- and the flat direction is a property of a 1 s window,
  // not of the solver: 4x the angular rate and 8x the linear acceleration both
  // make it *worse* (0.034, 0.036), and only a longer span helps (0.0056 at 2 s).
  //
  // So this test asserts on the parts the data does determine, and on the scale-
  // corrected velocity. `sigma_ba_prior` exists precisely because this flat
  // direction is much worse on real windows; the shipped default closes it.
  EXPECT_LT(res.cost_final, InitBACost(prob, truth, opt, truth) * 2);
  EXPECT_LT(res.pixel_rms, 1e-9);
  // The gyro bias is fully determined -- it comes out at machine precision.
  EXPECT_LT((res.state.bg - tr.bg).norm(), 1e-9);  // measured 1.4e-16
  EXPECT_LT((res.state.ba - tr.ba).norm(), 1e-2);  // measured 2.8e-3, scale-bound
  // Gravity direction in frame 0 -- the quantity the filter's Rsg is built from.
  EXPECT_LT(Angle(res.state.GravityInBody(0), truth.GravityInBody(0)), 2e-3);

  // Scale, from the feature cloud: the least-squares s minimizing |s*f_true - f|.
  number_t num = 0, den = 0;
  for (int n = 0; n < truth.num_tracks(); ++n) {
    num += res.state.f[n].dot(truth.f[n]);
    den += truth.f[n].squaredNorm();
  }
  const number_t s = num / den;
  EXPECT_NEAR(s, 1.0, 0.02);                                   // measured 1.00686
  EXPECT_LT((res.state.v[0] / s - truth.v[0]).norm(), 2e-3);    // measured 5.2e-4
  // ... and the raw velocity error is small but *not* at the level the scale-
  // corrected one is, which is the asymmetry the paragraph above explains.
  EXPECT_LT((res.state.v[0] - truth.v[0]).norm(), 0.02);        // measured 9.4e-3

  // And the seed really was wrong, so none of the above is trivially satisfied.
  EXPECT_GT((seed.v[0] - truth.v[0]).norm(), 0.2);
  EXPECT_GT(Angle(seed.GravityInBody(0), truth.GravityInBody(0)), 0.05);
}

TEST(InitBA, CostNeverIncreasesWithAnExtraIteration) {
  const Truth tr = EurocishTruth(kEurocBg, kEurocBa);
  const auto pts = MakeScene(50, 13);
  const InitCamera cam = EurocishCam();
  const number_t t0 = 0.4, span = 1.0;
  const int K = 11;
  const InitProblem prob = MakeProblem(tr, pts, cam, t0, span, K, Vec3::Zero(),
                                       Vec3::Zero(), PixelSigma(), 17);
  const BAState truth = TrueState(tr, pts, t0, span, K);
  const BAState seed =
      TiltedSeed(truth, Vec3{0.06, -0.05, 0}, Vec3{0.3, -0.25, 0.15},
                 Vec3::Zero(), Vec3::Zero());

  // Monotonicity, tested from outside: LM only accepts a step that lowers the
  // cost, so the cost after n+1 iterations can never exceed the cost after n.
  // Running the whole solve at each budget also checks that the iteration count
  // is the only thing that changes -- an internal per-iteration log could not.
  number_t prev = std::numeric_limits<number_t>::infinity();
  for (int budget = 0; budget <= 12; ++budget) {
    BAOptions opt;
    opt.cauchy_c = 0;
    opt.max_iterations = budget;
    const BAResult res = SolveInitBA(prob, seed, opt);
    EXPECT_LE(res.state.num_frames(), K);
    EXPECT_LE(res.cost_final, prev * (1 + 1e-12))
        << "budget " << budget << " ended above budget " << budget - 1;
    EXPECT_LE(res.iterations, budget);
    prev = res.cost_final;
  }
  EXPECT_LT(prev, 1e300);
}

// ---------------------------------------------------------------------------
// what it is for: recovering a planted bias under noise
// ---------------------------------------------------------------------------

TEST(InitBA, RecoversAPlantedBiasUnderPixelNoise) {
  const InitCamera cam = EurocishCam();
  const number_t t0 = 0.4, span = 1.5;
  const int K = 31;

  // Six seeds: one draw of pixel noise says nothing about a solver whose whole
  // job is to average it down.
  number_t bg_err_sum = 0, ba_err_sum = 0, v_err_sum = 0, bg_err_max = 0;
  const int trials = 6;
  for (int s = 0; s < trials; ++s) {
    const Truth tr = EurocishTruth(kEurocBg, kEurocBa);
    const auto pts = MakeScene(150, 21 + s);
    // Zero bias prior, as at init: the preintegrals are integrated at zero and
    // the true bias has to come out of the optimization.
    const InitProblem prob = MakeProblem(tr, pts, cam, t0, span, K, Vec3::Zero(),
                                         Vec3::Zero(), PixelSigma(), 101 + s);
    const BAState truth = TrueState(tr, pts, t0, span, K);
    const BAState seed =
        TiltedSeed(truth, Vec3{0.05, -0.04, 0}, Vec3{0.25, -0.2, 0.1},
                   Vec3::Zero(), Vec3::Zero());

    BAOptions opt;
    opt.sigma_pix = 0.3;
    opt.max_iterations = 40;
    opt.sigma_ba_prior = 0; // see RecoversTheExactStateFromATiltedSeed
    const BAResult res = SolveInitBA(prob, seed, opt);
    ASSERT_TRUE(res.ok) << res.why;

    bg_err_sum += (res.state.bg - tr.bg).norm();
    bg_err_max = std::max(bg_err_max, (res.state.bg - tr.bg).norm());
    ba_err_sum += (res.state.ba - tr.ba).norm();
    v_err_sum += (res.state.v[0] - truth.v[0]).norm();
    // The reprojection error should land at the noise it was given, not below it
    // (over-fitted) or far above it (the model cannot express the data).
    EXPECT_GT(res.pixel_rms, 0.15);
    EXPECT_LT(res.pixel_rms, 1.0) << "seed " << s;
  }
  const number_t bg_err = bg_err_sum / trials, ba_err = ba_err_sum / trials,
                 v_err = v_err_sum / trials;

  // Measured, 6 seeds: mean |bg| error 8.4e-5 rad/s (max 1.3e-4) against a
  // planted 0.0814 -- a recovery to 0.1% -- mean |ba| error 1.1e-2 m/s^2 against
  // a planted 0.1475, and mean |v| error 3.0e-2 m/s. The velocity is the loosest
  // of the three because it inherits the scale/ba flat direction documented in
  // `RecoversTheExactStateFromATiltedSeed`; the biases are what this test is
  // about. Tolerances are ~2-3x the measurements: loose enough to survive a
  // different Eigen or a different -ffast-math, tight enough that losing a
  // Jacobian term breaks them -- dropping the rotation's bias Jacobian alone
  // takes the gyro error to 0.08, i.e. no recovery at all.
  EXPECT_LT(bg_err, 2.5e-4) << "mean gyro bias error over " << trials
                            << " seeds";
  EXPECT_LT(bg_err_max, 4.0e-4);
  EXPECT_LT(ba_err, 3.0e-2);
  EXPECT_LT(v_err, 8.0e-2);
  // And it is a recovery, not a preservation: the seed's bias was zero.
  EXPECT_GT(kEurocBg.norm() / bg_err, 20.0);
  EXPECT_GT(kEurocBa.norm() / ba_err, 5.0);
}

TEST(InitBA, ImprovesOnStageAEndToEnd) {
  const InitCamera cam = EurocishCam();
  const number_t t0 = 0.4, span = 1.5;
  const int K = 31;
  const Truth tr = EurocishTruth(kEurocBg, kEurocBa);
  const auto pts = MakeScene(150, 31);
  const InitProblem prob = MakeProblem(tr, pts, cam, t0, span, K, Vec3::Zero(),
                                       Vec3::Zero(), PixelSigma(), 5);
  const BAState truth = TrueState(tr, pts, t0, span, K);

  // The whole pipeline as M4 will run it, including Stage A's accelerometer-mean
  // guard against the bimodality of the linear cost.
  LinearInitOptions lopt;
  lopt.prior_mode = LinearInitOptions::PriorMode::Check;
  lopt.gravity_prior = AccelMeanGravity(tr, t0, span, Vec3::Zero(), Vec3::Zero());
  const LinearInitResult lin = SolveLinearInit(prob, lopt);
  ASSERT_TRUE(lin.ok) << lin.why;

  BAState seed;
  ASSERT_TRUE(SeedBAState(prob, lin, &seed));
  // The seed has to be Stage A's answer, in Stage B's frame -- not merely close
  // to it. A bug in the handoff would show up here and nowhere else.
  EXPECT_LT((seed.VelocityInBody(0) - lin.v).norm(), 1e-9);
  EXPECT_LT(Angle(seed.GravityInBody(0), lin.g), 1e-9);
  EXPECT_LT(seed.p[0].norm(), 1e-12);

  BAOptions opt;
  opt.sigma_pix = 0.3;
  opt.max_iterations = 40;
  const BAResult res = SolveInitBA(prob, seed, opt);
  ASSERT_TRUE(res.ok) << res.why;

  const number_t vA = (lin.v - truth.VelocityInBody(0)).norm();
  const number_t vB = (res.state.VelocityInBody(0) - truth.VelocityInBody(0)).norm();
  const number_t gA = Angle(lin.g, truth.GravityInBody(0));
  const number_t gB = Angle(res.state.GravityInBody(0), truth.GravityInBody(0));

  // Measured: velocity 1.199 -> 1.104 m/s, tilt 4.01 -> 3.30 deg. Both improve,
  // and both are terrible -- so read this test for what it is.
  //
  // The fixture is adversarial for *Stage A*: it holds |a_w| and |omega| constant
  // for the whole window, so the gyro bias integrates coherently instead of
  // averaging down, and Stage A comes out 1.2 m/s off where a real EuRoC window
  // costs it 0.11-0.23. From 1.2 m/s the reprojection residual at the seed is 62
  // px RMS and Stage B cannot walk back to the truth: it is in the wrong basin,
  // and it converges neatly to the wrong place. A graduated Cauchy schedule
  // (c = inf, 300, 100, 30, 10, 3) does not rescue it either, which is why no
  // annealing was added -- the evidence did not support it.
  //
  // So the arbiter for "Stage B improves on Stage A" is the real-data harness,
  // where the seed is 0.11-0.23 m/s and Stage B takes it to 0.017 on all 11 EuRoC
  // sequences (see notes-n-prompts/notes-dyninit/m3-ba.md). What this test can
  // honestly settle is the *plumbing*: that Stage A's answer arrives in Stage B's
  // frame unchanged, and that Stage B moves both metrics in the right direction
  // even from a hopeless seed. Asserting a factor here would be asserting a
  // property of the fixture.
  EXPECT_LT(vB, vA) << "Stage A " << vA << " -> Stage B " << vB;
  EXPECT_LT(gB, gA) << "Stage A " << gA << " -> Stage B " << gB;
  // The one quantity that *is* well determined from this seed, and the reason to
  // believe the machinery ran: the gyro bias, measured 4.4e-5 against 0.0814.
  EXPECT_LT((res.state.bg - tr.bg).norm(), 5e-4);
}

TEST(InitBA, CauchyBoundsTheInfluenceOfOutliers) {
  // What a robust loss buys is *bounded influence*: the answer must stop being a
  // function of how bad the worst tracks are. That, not "a better number on one
  // seed", is what this test measures -- it solves the same window at three
  // outlier amplitudes and asks which configuration's answer moves.
  //
  // Two outlier models, because the obvious one is nearly harmless. A constant
  // offset applied to *every* observation of a track (`mode 0`) is almost
  // consistent with that track's 3D point having moved, so the solve absorbs it
  // into the feature position and the loss has little to do; a real KLT failure
  // is a jump *partway* through a track (`mode 1`), which no single 3D point can
  // explain. Measured velocity error across amplitude, with the loss and without:
  //
  //   mode 0 (whole track):  cauchy 0.0812 0.0816 0.0822 | plain 0.0704 0.0718 0.0690
  //   mode 1 (mid-track):    cauchy 0.0829 0.0806 0.0806 | plain 0.0682 0.0456 0.1375
  //
  // The robust spread over all six cases is 0.0023 m/s; the non-robust spread is
  // 0.0919, forty times larger, and at the severe end the non-robust solve is the
  // worse of the two outright. Note that the non-robust answer is *better* in four
  // of the six -- which is exactly why a test asserting "Cauchy improves the
  // velocity" would be measuring the seed and not the mechanism. The claim is
  // stability, and it is checked as stability.
  //
  // (On real EuRoC windows, where the outliers are real, the loss also improves
  // the answer outright: velocity 0.0527 -> 0.0170 m/s and gyro bias
  // 0.0084 -> 0.0028 rad/s from `cauchy 0` to the default `cauchy 3`. See
  // notes-n-prompts/notes-dyninit/m3-ba.md.)
  const InitCamera cam = EurocishCam();
  const number_t t0 = 0.4, span = 1.5;
  const int K = 31;
  const Truth tr = EurocishTruth(kEurocBg, kEurocBa);
  const auto pts = MakeScene(150, 41);
  const BAState clean_truth = TrueState(tr, pts, t0, span, K);

  std::vector<number_t> v_robust, v_plain, med_robust, med_plain;
  for (int mode = 0; mode < 2; ++mode) {
    for (number_t amp : {10.0, 25.0, 60.0}) {
      InitProblem prob = MakeProblem(tr, pts, cam, t0, span, K, Vec3::Zero(),
                                     Vec3::Zero(), PixelSigma(), 7);
      std::mt19937 rng(3);
      std::uniform_real_distribution<number_t> off(-amp, amp);
      int spoiled = 0;
      for (int n = 0; n < prob.num_tracks; n += 17) {
        const Vec2 d{off(rng) / cam.focal, off(rng) / cam.focal};
        int tot = 0;
        for (const auto &o : prob.obs)
          if (o.track == n)
            ++tot;
        int seen = 0;
        for (auto &o : prob.obs)
          if (o.track == n) {
            if (mode == 0 || seen >= tot / 2)
              o.xn += d;
            ++seen;
          }
        ++spoiled;
      }
      ASSERT_GE(spoiled, 5);

      const BAState seed =
          TiltedSeed(clean_truth, Vec3{0.05, -0.04, 0}, Vec3{0.25, -0.2, 0.1},
                     Vec3::Zero(), Vec3::Zero());
      // Shipped defaults apart from the noise level: `cauchy_c` is the only thing
      // varied, so the comparison is about the loss. Note the default
      // `sigma_ba_prior` is left in place -- freeing `ba` here takes the robust
      // spread from 0.0023 to 0.0368, i.e. the stability below needs *both* the
      // bounded loss and the bias prior, and neither alone.
      BAOptions opt;
      opt.sigma_pix = 0.3;
      opt.max_iterations = 40;
      BAOptions plain = opt;
      plain.cauchy_c = 0;

      const BAResult w = SolveInitBA(prob, seed, opt);
      const BAResult p = SolveInitBA(prob, seed, plain);
      ASSERT_TRUE(w.ok) << w.why;
      ASSERT_TRUE(p.ok) << p.why;
      v_robust.push_back((w.state.v[0] - clean_truth.v[0]).norm());
      v_plain.push_back((p.state.v[0] - clean_truth.v[0]).norm());
      med_robust.push_back(w.pixel_median);
      med_plain.push_back(p.pixel_median);
    }
  }

  const auto spread = [](const std::vector<number_t> &x) {
    return *std::max_element(x.begin(), x.end()) -
           *std::min_element(x.begin(), x.end());
  };
  // Bounded influence, stated directly. Measured 0.0023 against 0.0919.
  EXPECT_LT(spread(v_robust), 0.02);
  EXPECT_GT(spread(v_plain), 4 * spread(v_robust))
      << "robust spread " << spread(v_robust) << ", plain " << spread(v_plain);

  // The same story in the residual, and here it is monotone: with the loss the
  // median reprojection error stays at the 0.3 px noise the data actually carries
  // (measured 0.349-0.355 across all six cases), while without it the median
  // degrades with the outlier amplitude -- 0.374, 0.443, 0.682 for mode 0. The
  // median is the right statistic for this because `pixel_rms` is unrobustified
  // and the outliers dominate it by construction.
  for (size_t i = 0; i < med_robust.size(); ++i) {
    EXPECT_GT(med_robust[i], 0.2) << "case " << i << ": suspiciously good, the "
                                     "loss should not fit below the noise";
    EXPECT_LT(med_robust[i], 0.45) << "case " << i;
  }
  EXPECT_GT(med_plain.back(), med_robust.back() * 1.3)
      << "at the worst outlier setting the non-robust median should be clearly "
         "inflated: plain " << med_plain.back() << " vs " << med_robust.back();
  EXPECT_LT(spread(med_robust), spread(med_plain) / 3);
}

TEST(InitBA, DefaultPriorHoldsTheAccelBiasNearZero) {
  // `BAOptions` ships `sigma_ba_prior = 0.01`, which is not a regularizer with a
  // convenient side effect -- it is a decision not to estimate `ba` at all. Over
  // a 1.5 s window `ba` is nearly indistinguishable from a gravity tilt, and on
  // real EuRoC windows freeing it costs 68% *more* gravity tilt than the Stage A
  // seed. This test pins the default so that a later "let's just relax that
  // prior" cannot pass silently; the two tests above deliberately opt out of it,
  // which is why the default needs its own coverage.
  const InitCamera cam = EurocishCam();
  const number_t t0 = 0.4, span = 1.5;
  const int K = 31;
  const Truth tr = EurocishTruth(kEurocBg, kEurocBa);
  const auto pts = MakeScene(150, 31);
  const InitProblem prob = MakeProblem(tr, pts, cam, t0, span, K, Vec3::Zero(),
                                       Vec3::Zero(), PixelSigma(), 5);
  const BAState truth = TrueState(tr, pts, t0, span, K);
  const BAState seed =
      TiltedSeed(truth, Vec3{0.05, -0.04, 0}, Vec3{0.25, -0.2, 0.1},
                 Vec3::Zero(), Vec3::Zero());

  BAOptions opt; // defaults, untouched -- that is the point
  opt.sigma_pix = 0.3;
  BAOptions freed = opt;
  freed.sigma_ba_prior = 0;
  const BAResult res = SolveInitBA(prob, seed, opt);
  const BAResult unpriored = SolveInitBA(prob, seed, freed);
  ASSERT_TRUE(res.ok) << res.why;
  ASSERT_TRUE(unpriored.ok) << unpriored.why;

  EXPECT_GT(opt.sigma_ba_prior, 0) << "the shipped default stopped holding ba";
  EXPECT_EQ(opt.sigma_bg_prior, 0) << "the gyro bias is observable and wants no "
                                      "prior; see init_ba.h";

  // Stated as a comparison, not as an absolute: how far the prior drags `ba` back
  // depends on how well the window determines it, and this synthetic window
  // determines it far better than a real one does. Measured here: |ba| = 0.091
  // with the default prior against 0.146 unpriored, on a planted 0.1475 -- so the
  // prior is doing real work even where the data is informative. On the 11 real
  // EuRoC windows, where it is not, unpriored |ba| runs to gravity-sized values
  // and the recovered gravity ends up 68% worse than the Stage A seed.
  EXPECT_LT(res.state.ba.norm(), 0.8 * unpriored.state.ba.norm())
      << "priored |ba| " << res.state.ba.norm() << " vs unpriored "
      << unpriored.state.ba.norm();
  EXPECT_LT(res.state.ba.norm(), kEurocBa.norm());
  // The gyro bias is untouched by any of this: it is observable from r_theta.
  // Measured 1.6e-4 priored, 7.3e-5 unpriored.
  EXPECT_LT((res.state.bg - tr.bg).norm(), 1.0e-3);
  EXPECT_LT((unpriored.state.bg - tr.bg).norm(), 1.0e-3);
}


// ---------------------------------------------------------------------------
// the marginal covariance
// ---------------------------------------------------------------------------

TEST(InitBA, MarginalCovarianceMatchesTheDenseHessian) {
  // `BAResult::cov` is computed by Schur-complementing the tracks out and solving
  // the reduced system against a nine-column selection matrix. That is a
  // different computation from what a marginal covariance *means*, so this test
  // asks for the meaning: invert the whole Hessian, tracks included, and read the
  // same nine rows and columns off it. The dense side comes from
  // `InitBALinearize`, i.e. from the accumulation the solver itself runs, so a
  // derivative error would cancel on both sides -- deliberately, since the
  // derivatives have their own test above and what is under test here is the
  // elimination and the gauge handling.
  const InitCamera cam = EurocishCam();
  const number_t t0 = 0.4, span = 0.8;
  const int K = 9;
  const Truth tr = EurocishTruth(kEurocBg, kEurocBa);
  const auto pts = MakeScene(60, 12);
  const InitProblem prob = MakeProblem(tr, pts, cam, t0, span, K, Vec3::Zero(),
                                       Vec3::Zero(), PixelSigma(), 5);
  const BAState truth = TrueState(tr, pts, t0, span, K);
  const BAState seed =
      TiltedSeed(truth, Vec3{0.05, -0.04, 0}, Vec3{0.25, -0.2, 0.1},
                 Vec3::Zero(), Vec3::Zero());

  BAOptions opt; // shipped defaults, including `sigma_ba_prior`
  opt.sigma_pix = 0.3;
  // No robust loss. Under Cauchy the reweighted normal equations are not the
  // Hessian of any one quadratic, so "the" covariance would not be well defined
  // and the two sides would be entitled to disagree.
  opt.cauchy_c = 0;
  opt.want_covariance = true;
  const BAResult res = SolveInitBA(prob, seed, opt);
  ASSERT_TRUE(res.ok) << res.why;
  ASSERT_TRUE(res.cov_ok) << "no covariance was produced";

  VecX r;
  MatX J;
  std::vector<int> tcol;
  // `seed` as the gauge reference, not `res.state`: the yaw prior's Jacobian is a
  // function of `R_0 R_ref_0'` (init_ba.cpp), so referencing it anywhere else is a
  // different Hessian -- by 1e-3 relative here, which is exactly the size of
  // disagreement that would otherwise be blamed on the elimination.
  ASSERT_TRUE(InitBALinearize(prob, res.state, opt, seed, &r, &J, &tcol));
  MatX H = J.transpose() * J;
  // The same translation gauge the solver pins: `p_0` held exactly, by zeroing
  // its rows and columns. Without this H is singular and the comparison would be
  // between two arbitrary pseudo-inverses.
  for (int i = 3; i < 6; ++i) {
    H.row(i).setZero();
    H.col(i).setZero();
    H(i, i) = 1;
  }
  const MatX dense_full = H.inverse();
  // Column layout, from init_ba.h: frame k at [9k, 9k+9) as (dtheta, dp, dv),
  // then bg, then ba. The reported order is (v at the last frame, bg, ba).
  int sel[9];
  for (int i = 0; i < 3; ++i) {
    sel[i] = 9 * (K - 1) + 6 + i;
    sel[3 + i] = 9 * K + i;
    sel[6 + i] = 9 * K + 3 + i;
  }
  Mat9 dense;
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j)
      dense(i, j) = dense_full(sel[i], sel[j]);

  ASSERT_GT(dense.diagonal().minCoeff(), 0);
  // Correlation-scaled, because the three blocks differ by four orders of
  // magnitude (m/s against rad/s) and an absolute tolerance would be a test of
  // the velocity block alone. The tolerance is 1e-4 rather than roundoff because
  // the two sides are not the same arithmetic: the dense side inverts a 267x267
  // Hessian that carries this window's near-flat scale/accel-bias direction (see
  // `RecoversTheExactStateFromATiltedSeed`), while the sparse side eliminates the
  // tracks first. Measured worst disagreement 6e-6, i.e. 16x inside the gate --
  // and a wrong elimination or a mismatched gauge misses by 1e-3, which this
  // catches: referencing the yaw prior to the wrong state does exactly that.
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j) {
      const number_t scale = std::sqrt(dense(i, i) * dense(j, j));
      EXPECT_NEAR(res.cov(i, j) / scale, dense(i, j) / scale, 1e-4)
          << "entry (" << i << ", " << j << ")";
    }

  // And the matrix is not trivially the prior handed back: on this window the
  // velocity is determined far better than the 0.5 m/s the filter's config
  // assumes, which is exactly the property M6 went on to measure against real
  // groundtruth (and found does not hold there -- see m6-covariance.md).
  EXPECT_LT(std::sqrt(res.cov.diagonal().head<3>().maxCoeff()), 0.1);
  // `ba` is priored, not estimated, so its sigma cannot exceed the prior by much.
  EXPECT_LT(std::sqrt(res.cov.diagonal().tail<3>().maxCoeff()),
            1.5 * opt.sigma_ba_prior);
}
