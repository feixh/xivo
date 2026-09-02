// Online estimation of the visual measurement noise, M4 of the EuRoC round.
//
// Run from the repository root: the config path below is relative to it.
//
// `Estimator::AdaptVisualMeasNoise` reads the Mahalanobis distances that
// `MHGating` has already computed and moves `R_` so that their median matches
// the median of a chi-square with two degrees of freedom. The claim is that this
// fixed point is the true measurement noise, and that the loop reaches it from
// either side without the operator having to pick a value per sequence.
//
// These tests drive the function directly with synthetic distance vectors rather
// than through a real update, so what they pin is the estimator itself: its fixed
// point, its convergence, its clamps, and the conditions under which it declines
// to move at all. They deliberately say nothing about the H P H' term -- the
// synthetic distances stand in for the whole innovation covariance, which is
// exactly the quantity the real gate forms. Whether the loop lands on a *useful*
// value on real data is an accuracy question, answered by the sequence results in
// notes-euroc/m4-xivo-accuracy-tuning.md, not by a unit test.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#define private public

#include "estimator.h"
#include "utils.h"

using namespace xivo;

namespace {

const char *kCfg = "cfg/tumvi_stereo.json";

// The median of a chi-square with 2 degrees of freedom, which is what the
// adaptation compares against. Spelled out here rather than imported so that a
// change to the constant in the estimator has to be made twice, on purpose.
constexpr number_t kChi2TwoDofMedian = 1.3862943611198906; // 2 ln 2

/** Quantiles of chi-square(2) at (k + 0.5) / n for k = 0..n-1, i.e. a
 *  deterministic stand-in for a sample of n distances from a *consistent*
 *  filter. chi-square(2) has CDF 1 - exp(-x/2), so its quantile function is
 *  -2 ln(1 - p). n is odd, so the middle element is the exact median and no
 *  interpolation enters the expectations below.
 *
 * Using quantiles rather than pseudo-random draws keeps every expectation an
 * equality instead of a tolerance, and keeps the test from depending on a
 * generator. */
std::vector<number_t> ConsistentDistances(int n = 61) {
  std::vector<number_t> d;
  d.reserve(n);
  for (int k = 0; k < n; ++k) {
    const number_t p = (k + 0.5) / n;
    d.push_back(-2.0 * std::log1p(-p));
  }
  return d;
}

/** The same sample as seen by a filter whose assumed variance is `assumed` when
 *  the truth is `truth`: every distance scales by truth/assumed, since the
 *  distance is a residual whitened by the assumed covariance. */
std::vector<number_t> ScaledDistances(number_t truth, number_t assumed,
                                      int n = 61) {
  auto d = ConsistentDistances(n);
  for (auto &x : d) {
    x *= truth / assumed;
  }
  return d;
}

} // namespace

class AdaptNoiseTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto cfg = LoadJson(kCfg);
    ASSERT_FALSE(cfg.isNull())
        << "could not load " << kCfg << "; run tests from the repo root";
    est = CreateSystem(cfg);
    ASSERT_NE(est, nullptr);
    Configure(1.0);
  }

  /** Put the adaptation in a known state: on, past its warmup, full step size,
   *  and starting from an assumed variance of `R0`. A full step (alpha = 1) makes
   *  the fixed-point and clamp expectations exact; convergence with alpha < 1 is
   *  tested separately. */
  void Configure(number_t R0, number_t alpha = 1.0) {
    est->adapt_R_ = true;
    est->adapt_R_alpha_ = alpha;
    est->adapt_R_min_ = 0.25;  // std 0.5 px
    est->adapt_R_max_ = 16.0;  // std 4.0 px
    est->adapt_R_warmup_ = 0;
    est->adapt_R_min_samples_ = 10;
    est->adapt_R_updates_ = 0;
    est->R_ = R0;
    est->R_pending_ = R0;
    est->adapt_R_std_min_ = est->adapt_R_std_max_ = std::sqrt(R0);
  }

  /** One adaptation step, returning the variance the next update would adopt. */
  number_t Step(const std::vector<number_t> &dist) {
    est->AdaptVisualMeasNoise(dist);
    return est->R_pending_;
  }

  /** `n` steps of the loop against a *fixed truth*, feeding back the estimate
   *  each time, the way a real run does. Returns the final assumed variance. */
  number_t Converge(number_t truth, number_t R0, number_t alpha, int n) {
    Configure(R0, alpha);
    for (int i = 0; i < n; ++i) {
      est->AdaptVisualMeasNoise(ScaledDistances(truth, est->R_));
      est->R_ = est->R_pending_; // MHGating does this at the top of the next update
    }
    return est->R_;
  }

  EstimatorPtr est;
};

TEST_F(AdaptNoiseTest, AConsistentFilterIsAFixedPoint) {
  // The whole design rests on this: when the assumed noise is already right, the
  // median of the distances is the chi-square(2) median and the estimate must not
  // drift. If it drifted, every sequence would end up wherever the loop's bias
  // pushed it rather than at its own noise level.
  Configure(2.25); // std 1.5 px
  const number_t after = Step(ConsistentDistances());
  EXPECT_NEAR(after, 2.25, 1e-12);
}

TEST_F(AdaptNoiseTest, TheEstimateMovesToTheTruthFromEitherSide) {
  // With a full step the estimate should land exactly on the truth, because the
  // ratio of medians *is* the ratio of variances. Both directions, since EuRoC
  // needs both: Machine Hall wants the estimate to come down from a loose start,
  // the Vicon room wants it to go up from a tight one.
  const number_t truth = 4.0; // std 2.0 px

  Configure(0.5625); // std 0.75 px, the shipped TUM-VI value: too tight
  EXPECT_NEAR(Step(ScaledDistances(truth, 0.5625)), truth, 1e-9);

  Configure(12.25); // std 3.5 px: too loose
  EXPECT_NEAR(Step(ScaledDistances(truth, 12.25)), truth, 1e-9);
}

TEST_F(AdaptNoiseTest, PartialStepsConvergeGeometrically) {
  // The shipped alpha is 0.05, so a step is deliberately a small fraction of the
  // way. Check that the loop still gets there, and that it is monotone rather
  // than oscillating -- an overshoot would widen the gate past the truth, which
  // is the one direction with a feedback path back into the residuals.
  const number_t truth = 4.0, R0 = 0.5625;
  number_t prev = R0;
  Configure(R0, 0.05);
  for (int i = 0; i < 200; ++i) {
    est->AdaptVisualMeasNoise(ScaledDistances(truth, est->R_));
    est->R_ = est->R_pending_;
    EXPECT_GT(est->R_, prev) << "step " << i << " did not advance";
    EXPECT_LE(est->R_, truth + 1e-9) << "step " << i << " overshot the truth";
    prev = est->R_;
  }
  // 200 steps of a 5% geometric approach closes log(4/0.5625) to within
  // 0.95^200 of itself, which is far below the tolerance here.
  EXPECT_NEAR(Converge(truth, R0, 0.05, 200), truth, 1e-3);
  // And from the other side, to the same place.
  EXPECT_NEAR(Converge(truth, 12.25, 0.05, 200), truth, 1e-3);
}

TEST_F(AdaptNoiseTest, ClampsBoundTheEstimateOnBothSides) {
  // The bounds are the safety property, not a nicety: `R_` sets the gate radius
  // as well as the weight, so an unbounded upward walk would admit ever worse
  // measurements, and an unbounded downward one would reject everything.
  Configure(4.0);
  EXPECT_NEAR(Step(ScaledDistances(1e6, 4.0)), est->adapt_R_max_, 1e-12);

  Configure(4.0);
  EXPECT_NEAR(Step(ScaledDistances(1e-6, 4.0)), est->adapt_R_min_, 1e-12);
}

TEST_F(AdaptNoiseTest, WarmupAndSampleCountHoldTheEstimateStill) {
  // Two guards against adapting on evidence that is not about the tracker: the
  // first updates after initialization, when the covariance is still the prior,
  // and any update with too few in-state features for a median to mean anything.
  Configure(2.25);
  est->adapt_R_warmup_ = 5;
  const auto loose = ScaledDistances(16.0, 2.25);
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(Step(loose), 2.25, 1e-12) << "moved during warmup step " << i;
  }
  EXPECT_GT(Step(loose), 2.25) << "never left warmup";

  Configure(2.25);
  std::vector<number_t> few(est->adapt_R_min_samples_ - 1, 10.0);
  EXPECT_NEAR(Step(few), 2.25, 1e-12);
  std::vector<number_t> enough(est->adapt_R_min_samples_, 10.0);
  EXPECT_GT(Step(enough), 2.25);
}

TEST_F(AdaptNoiseTest, NonFiniteDistancesAreDroppedNotRanked) {
  // A singular innovation covariance or a NaN in a Jacobian yields a non-finite
  // distance. Sorting those would put them wherever the comparator happened to
  // land them and could drag the median to an arbitrary place, so they have to be
  // removed rather than merely ignored by the gate.
  Configure(2.25);
  auto d = ConsistentDistances();
  const number_t clean = Step(d);

  Configure(2.25);
  d.push_back(std::numeric_limits<number_t>::quiet_NaN());
  d.push_back(std::numeric_limits<number_t>::infinity());
  d.push_back(-1.0); // negative: impossible for a quadratic form, so also junk
  // The three extra entries are dropped, which leaves the same odd-sized sample
  // and therefore exactly the same median.
  EXPECT_NEAR(Step(d), clean, 1e-12);
}

TEST_F(AdaptNoiseTest, EvenSampleSizesUseTheMeanOfTheTwoMiddleValues) {
  // `nth_element` only guarantees the prefix is no greater than the pivot, so the
  // even case has to hunt for the lower middle value. Getting that wrong would
  // bias every even-sized update, which is half of them.
  Configure(1.0);
  std::vector<number_t> d;
  const int n = 20;
  for (int k = 0; k < n; ++k) {
    d.push_back(1.0 + k); // 1..20, median 10.5
  }
  // Shuffled deterministically so the answer cannot come from the input order.
  std::rotate(d.begin(), d.begin() + 7, d.end());
  std::reverse(d.begin() + 3, d.begin() + 11);
  EXPECT_NEAR(Step(d), 10.5 / kChi2TwoDofMedian, 1e-9);
}

TEST_F(AdaptNoiseTest, TheReportedRangeTracksWhereTheEstimateWent) {
  // The census line reports the range walked, and the sequence-level argument for
  // this option is read off exactly that. A range that only ever grew upward, or
  // that missed the extremes, would make the run logs misleading.
  Configure(2.25, 0.5);
  for (int i = 0; i < 40; ++i) {
    est->AdaptVisualMeasNoise(ScaledDistances(9.0, est->R_)); // walk up to 3 px
    est->R_ = est->R_pending_;
  }
  for (int i = 0; i < 60; ++i) {
    est->AdaptVisualMeasNoise(ScaledDistances(1.0, est->R_)); // then down to 1 px
    est->R_ = est->R_pending_;
  }
  EXPECT_NEAR(est->adapt_R_std_max_, 3.0, 1e-3);
  EXPECT_NEAR(est->adapt_R_std_min_, 1.0, 1e-3);
  EXPECT_NEAR(std::sqrt(est->R_), 1.0, 1e-3);
}
