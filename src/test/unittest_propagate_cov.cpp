// Deferring the motion-to-structure correlation update of the covariance.
//
// The integrators used to propagate `P[0:24, 24:]` and its transpose on every
// Prince-Dormand/RK4 substep. They now accumulate the transition and apply it
// once per image. The two are the same thing because the update is linear:
//
//     F_n (... (F_1 P)) == (F_n ... F_1) P
//
// so these tests pin the equality, the accumulation *order* (the one thing that
// is easy to get backwards, and silent when wrong -- the matrices do not
// commute), and the two properties the rewrite has to preserve: that `P` comes
// out exactly symmetric, and that nothing outside the two correlation blocks
// moves.
//
// Author: efficiency work (branch auto-efficiency)
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core.h"

using namespace xivo;

namespace {

/** A motion transition shaped like the real one. `F_` is built as
 *  `I + FK*dt`, and `ComputeMotionJacobianAt` writes `FK` only in rows 0..8
 *  (Wsb, Tsb, Vsb -- the only states with dynamics), so `F_` is the identity
 *  outside those rows. `dt` is the integrator's stepsize, 2 ms. */
MatMotion MakeF(unsigned seed) {
  std::default_random_engine gen(seed);
  std::normal_distribution<number_t> nrm(0.0, 1.0);
  MatMotion F = MatMotion::Identity();
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < kMotionSize; ++j)
      F(i, j) += 0.002 * nrm(gen);
  return F;
}

/** A symmetric full-size covariance. Positive definiteness is irrelevant here --
 *  nothing factors it -- but symmetry is not: mirroring the upper correlation
 *  block into the lower one is only valid on a symmetric input. */
MatX MakeSymP(unsigned seed) {
  std::default_random_engine gen(seed);
  std::normal_distribution<number_t> nrm(0.0, 1.0);
  MatX A(kFullSize, kFullSize);
  for (int i = 0; i < kFullSize; ++i)
    for (int j = 0; j < kFullSize; ++j)
      A(i, j) = nrm(gen);
  MatX P = 0.5 * (A + A.transpose());
  P.diagonal().array() += kFullSize;
  return P;
}

/** The code this milestone replaced, verbatim: both blocks, once per step. */
void ApplyPerStep(MatX &P, const MatMotion &F) {
  P.block<kMotionSize, kStructureSize>(0, kMotionSize) =
      F * P.block<kMotionSize, kStructureSize>(0, kMotionSize);
  P.block<kStructureSize, kMotionSize>(kMotionSize, 0) =
      P.block<kStructureSize, kMotionSize>(kMotionSize, 0) * F.transpose();
}

number_t RelDiff(const MatX &a, const MatX &b) {
  return (a - b).norm() / std::max(number_t(1e-30), b.norm());
}

/** How far from symmetric `P` is, in the same relative units. */
number_t SymError(const MatX &P) {
  return (P - P.transpose()).norm() / std::max(number_t(1e-30), P.norm());
}

/** `n_steps` transitions, the reference applying each in turn and the fast path
 *  accumulating them the way `AccumulateMotionStructureCorrelation` does. */
struct Propagated {
  MatX ref, fast, in;
  MatMotion Fcross;
};

Propagated RunSteps(int n_steps, unsigned seed) {
  Propagated r;
  r.in = MakeSymP(seed);
  r.ref = r.in;
  r.fast = r.in;
  r.Fcross = MatMotion::Identity();
  for (int k = 0; k < n_steps; ++k) {
    const MatMotion F = MakeF(seed + 100 * k + 1);
    ApplyPerStep(r.ref, F);
    r.Fcross = F * r.Fcross;   // the order under test
  }
  ApplyMotionTransition(r.fast, r.Fcross);
  return r;
}

// ~3 substeps x ~10 IMU samples is what one image costs.
constexpr int kStepsPerImage = 30;

TEST(PropagateCov, DeferredTransitionMatchesPerStepApplication) {
  const Propagated r = RunSteps(kStepsPerImage, 7);
  EXPECT_LT(RelDiff(r.fast, r.ref), 1e-12);
}

TEST(PropagateCov, DeferredTransitionMatchesOverManyImages) {
  // 300 steps: ten images' worth without an intervening flush, which is more
  // than the filter ever accumulates. If the accumulated product were drifting,
  // this is where it would show.
  const Propagated r = RunSteps(10 * kStepsPerImage, 11);
  EXPECT_LT(RelDiff(r.fast, r.ref), 1e-10);
}

TEST(PropagateCov, OneStepIsBitIdenticalToTheOldUpperBlock) {
  // With a single step the accumulated transition *is* `F`, so the upper block
  // is the same expression the old code evaluated and must agree exactly. Any
  // difference would mean the accumulator is not starting from the identity.
  const Propagated r = RunSteps(1, 13);
  const auto fast = r.fast.block<kMotionSize, kStructureSize>(0, kMotionSize);
  const auto ref = r.ref.block<kMotionSize, kStructureSize>(0, kMotionSize);
  for (int i = 0; i < kMotionSize; ++i)
    for (int j = 0; j < kStructureSize; ++j)
      ASSERT_EQ(fast(i, j), ref(i, j)) << "at " << i << ", " << j;
}

TEST(PropagateCov, TheAccumulationOrderMatters) {
  // The same steps accumulated the other way round. The transitions do not
  // commute, so this must be *wrong* -- otherwise the test above would pass no
  // matter which order the implementation used.
  MatX P = MakeSymP(7);
  MatMotion Fcross = MatMotion::Identity();
  for (int k = 0; k < kStepsPerImage; ++k)
    Fcross = Fcross * MakeF(7 + 100 * k + 1);
  ApplyMotionTransition(P, Fcross);

  const Propagated r = RunSteps(kStepsPerImage, 7);
  EXPECT_GT(RelDiff(P, r.ref), 1e-6);
}

TEST(PropagateCov, TheResultIsExactlySymmetric) {
  const Propagated r = RunSteps(kStepsPerImage, 17);
  for (int i = 0; i < kFullSize; ++i)
    for (int j = 0; j < i; ++j)
      ASSERT_EQ(r.fast(i, j), r.fast(j, i)) << "at " << i << ", " << j;
  // The code it replaces came out symmetric too, and measurably *exactly* so
  // (this reads 0, not 1e-16): its two products, `F U` and `U^T F^T`, sum the
  // same terms in the same order inside Eigen's gemm, so they agree bit for bit.
  // Worth pinning, because it is what makes the mirror a pure halving of the
  // work rather than a change in the numbers -- but it is a property of the
  // kernel's blocking, not of the algebra, whereas the mirror is symmetric by
  // construction.
  EXPECT_LT(SymError(r.ref), 1e-15);
}

TEST(PropagateCov, NothingOutsideTheCorrelationBlocksMoves) {
  const Propagated r = RunSteps(kStepsPerImage, 23);
  const MatX motion_delta = r.fast.block<kMotionSize, kMotionSize>(0, 0) -
                            r.in.block<kMotionSize, kMotionSize>(0, 0);
  const MatX structure_delta =
      r.fast.block<kStructureSize, kStructureSize>(kMotionSize, kMotionSize) -
      r.in.block<kStructureSize, kStructureSize>(kMotionSize, kMotionSize);
  ASSERT_EQ(motion_delta.norm(), 0.0);
  ASSERT_EQ(structure_delta.norm(), 0.0);
}

} // namespace
