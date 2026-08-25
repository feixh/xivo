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
  for (int i = 0; i < kMotionDynSize; ++i)
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
  ApplyMotionTransition(r.fast, r.Fcross.topRows<kMotionDynSize>());
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

TEST(PropagateCov, OneStepMatchesTheOldUpperBlock) {
  // With a single step the accumulated transition *is* `F`, so the upper block
  // is the same quantity the old code computed -- but no longer via the same
  // expression, which is what this test now separates.
  //
  // Below row `kMotionDynSize` it *is* still bit-identical, and for a stronger
  // reason than before: the old code multiplied those rows of the upper block by
  // rows of the identity, so the only difference is that the new code does not
  // write them at all.
  const Propagated r = RunSteps(1, 13);
  const auto fast = r.fast.block<kMotionSize, kStructureSize>(0, kMotionSize);
  const auto ref = r.ref.block<kMotionSize, kStructureSize>(0, kMotionSize);
  for (int i = kMotionDynSize; i < kMotionSize; ++i)
    for (int j = 0; j < kStructureSize; ++j)
      ASSERT_EQ(fast(i, j), ref(i, j)) << "at " << i << ", " << j;

  // The nine dynamic rows are exactly the narrowed product and nothing else --
  // no extra scaling, no leftover accumulation, no rounding of its own.
  const MatX narrowed = r.Fcross.topRows<kMotionDynSize>() *
                        r.in.block<kMotionSize, kStructureSize>(0, kMotionSize);
  for (int i = 0; i < kMotionDynSize; ++i)
    for (int j = 0; j < kStructureSize; ++j)
      ASSERT_EQ(fast(i, j), narrowed(i, j)) << "at " << i << ", " << j;

  // Against the *old* 24-row product they agree only to the last bits. M6
  // narrowed the gemm from 24x24x540 to 9x24x540, and Eigen picks its panel
  // blocking from the shape, so the same dot products get summed in a different
  // order. This is the same class of reassociation M5 introduced in the update;
  // it is why this milestone can change a trajectory at all (see the note).
  const MatX dyn_ref = r.ref.block<kMotionDynSize, kStructureSize>(0, kMotionSize);
  EXPECT_LT(RelDiff(MatX(r.fast.block<kMotionDynSize, kStructureSize>(
                        0, kMotionSize)),
                    dyn_ref),
            1e-15);
}

TEST(PropagateCov, TheAccumulationOrderMatters) {
  // The same steps accumulated the other way round. The transitions do not
  // commute, so this must be *wrong* -- otherwise the test above would pass no
  // matter which order the implementation used.
  MatX P = MakeSymP(7);
  MatMotion Fcross = MatMotion::Identity();
  for (int k = 0; k < kStepsPerImage; ++k)
    Fcross = Fcross * MakeF(7 + 100 * k + 1);
  ApplyMotionTransition(P, Fcross.topRows<kMotionDynSize>());

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

TEST(PropagateCov, TheAccumulatedTransitionIsTheIdentityBelowTheDynamicRows) {
  // `ApplyMotionTransition` skips those rows on the strength of this, so it is
  // worth checking on the accumulated product rather than on one factor:
  // `(I + A)(I + B) = I + A + B + AB` keeps the property, but only if every
  // factor has it.
  const Propagated r = RunSteps(kStepsPerImage, 29);
  const Eigen::Matrix<number_t, kMotionSize - kMotionDynSize, kMotionSize>
      below = r.Fcross.bottomRows<kMotionSize - kMotionDynSize>();
  const MatMotion I = MatMotion::Identity();
  for (int i = 0; i < kMotionSize - kMotionDynSize; ++i)
    for (int j = 0; j < kMotionSize; ++j)
      ASSERT_EQ(below(i, j), I(i + kMotionDynSize, j))
          << "at " << i + kMotionDynSize << ", " << j;
}

TEST(PropagateCov, AccumulatingByTheDynamicRowsMatchesTheFullProduct) {
  // The accumulator applies `(I + [Fdt; 0]) Fcross` as
  // `Fcross + [Fdt Fcross; 0]`, which is 9x24x24 rather than 24x24x24. Same
  // matrix; the identity is added by hand instead of multiplied through.
  MatMotion full = MatMotion::Identity(), dyn = MatMotion::Identity();
  for (int k = 0; k < kStepsPerImage; ++k) {
    const MatMotionDyn Fdt =
        (MakeF(31 + 100 * k) - MatMotion::Identity()).topRows<kMotionDynSize>();
    MatMotion step = MatMotion::Identity();
    step.topRows<kMotionDynSize>() += Fdt;
    full = step * full;

    const MatMotionDyn scratch = Fdt * dyn;
    dyn.topRows<kMotionDynSize>() += scratch;
  }
  EXPECT_LT(RelDiff(full, dyn), 1e-12);
}

////////////////////////////////////////
// M6: the stage slope
////////////////////////////////////////

/** A rotation, since the accelerometer noise is rotated into the spatial
 *  frame. */
Mat3 MakeRsb(unsigned seed) {
  std::default_random_engine gen(seed);
  std::normal_distribution<number_t> nrm(0.0, 1.0);
  Vec3 w;
  w << nrm(gen), nrm(gen), nrm(gen);
  return SO3::exp(w).matrix();
}

/** `Qimu` as `Estimator` builds it: block diagonal, each 3x3 block diagonal and
 *  positive. */
MatX MakeQimu(unsigned seed) {
  std::default_random_engine gen(seed);
  std::uniform_real_distribution<number_t> pos(0.1, 4.0);
  MatX Q = MatX::Zero(12, 12);
  for (int i = 0; i < 12; ++i)
    Q(i, i) = pos(gen);
  return Q;
}

/** An exactly symmetric motion covariance -- `MotionCovSlope` requires it, and
 *  the reference below has to be driven with the same matrix for the comparison
 *  to mean anything. */
MatMotion MakeSymMotionP(unsigned seed) {
  std::default_random_engine gen(seed);
  std::normal_distribution<number_t> nrm(0.0, 1.0);
  MatMotion A;
  for (int i = 0; i < kMotionSize; ++i)
    for (int j = 0; j < kMotionSize; ++j)
      A(i, j) = nrm(gen);
  MatMotion P = A.transpose() * A;
  // `A'A` is symmetric in algebra but its gemm need not produce it bit for bit.
  P = 0.5 * (P + P.transpose()).eval();
  P.diagonal().array() += kMotionSize;
  return P;
}

/** The dynamic rows of a Jacobian shaped like `ComputeMotionJacobianAt`'s
 *  output: O(1) entries, not the near-identity that `MakeF` returns. */
MatMotionDyn MakeFdyn(unsigned seed) {
  std::default_random_engine gen(seed);
  std::normal_distribution<number_t> nrm(0.0, 1.0);
  MatMotionDyn F = MatMotionDyn::Zero();
  for (int i = 0; i < kMotionDynSize; ++i)
    for (int j = 0; j < kMotionSize; ++j)
      F(i, j) = nrm(gen);
  return F;
}

/** The expression the integrators used to evaluate, spelled out: the Jacobian
 *  padded back to 24x24, both products, and the noise term through the full
 *  24x12 noise Jacobian. */
MatMotion SlopeReference(const MatMotionDyn &Fdyn, const MatMotion &P,
                         const Mat3 &Rsb, const MatX &Qimu) {
  MatMotion F = MatMotion::Zero();
  F.topRows<kMotionDynSize>() = Fdyn;
  MatMotionNoise G;
  MotionNoiseJacobian(Rsb, G);
  return F * P + P * F.transpose() + G * Qimu * G.transpose();
}

TEST(MotionCovSlope, MatchesTheUnstructuredForm) {
  // The milestone's claim: `A + A'` with `A = F P` over nine rows, plus four
  // small noise blocks, is `F P + P F' + G Qimu G'`.
  for (unsigned seed : {3u, 5u, 41u}) {
    const MatMotionDyn Fdyn = MakeFdyn(seed);
    const MatMotion P = MakeSymMotionP(seed + 1);
    const Mat3 Rsb = MakeRsb(seed + 2);
    const MatX Qimu = MakeQimu(seed + 3);

    MatMotion out;
    MotionCovSlope(Fdyn, P, Rsb, Qimu, out);
    EXPECT_LT(RelDiff(out, SlopeReference(Fdyn, P, Rsb, Qimu)), 1e-13)
        << "seed " << seed;
  }
}

TEST(MotionCovSlope, TheSlopeIsExactlySymmetric) {
  // `P F' = (F P)'` is used *because* it makes this structural. The old form got
  // it only as far as its two gemms agreed.
  const MatMotionDyn Fdyn = MakeFdyn(7);
  const MatMotion P = MakeSymMotionP(8);
  MatMotion out;
  MotionCovSlope(Fdyn, P, MakeRsb(9), MakeQimu(10), out);
  for (int i = 0; i < kMotionSize; ++i)
    for (int j = 0; j < i; ++j)
      ASSERT_EQ(out(i, j), out(j, i)) << "at " << i << ", " << j;
}

TEST(MotionCovSlope, TheStructurallyZeroPartIsExactlyZero) {
  // Everything outside the nine dynamic rows, the nine dynamic columns and the
  // two bias blocks is untouched by the model, and `MotionCovSlope` writes into
  // a matrix the caller has been reusing since the previous step -- so "zero"
  // has to mean it was cleared, not that it happens to hold last step's value.
  MatMotion out = MatMotion::Constant(1234.5);
  MotionCovSlope(MakeFdyn(11), MakeSymMotionP(12), MakeRsb(13), MakeQimu(14),
                 out);
  for (int i = kMotionDynSize; i < kMotionSize; ++i) {
    for (int j = kMotionDynSize; j < kMotionSize; ++j) {
      const bool bias_block =
          (i >= Index::bg && i < Index::bg + 3 && j >= Index::bg &&
           j < Index::bg + 3) ||
          (i >= Index::ba && i < Index::ba + 3 && j >= Index::ba &&
           j < Index::ba + 3);
      if (!bias_block)
        ASSERT_EQ(out(i, j), 0.0) << "at " << i << ", " << j;
    }
  }
}

TEST(MotionNoise, TheFourBlockFormMatchesGQGt) {
  // `AddMotionNoiseCov` on its own, so a failure in the slope test can be
  // localized.
  const Mat3 Rsb = MakeRsb(17);
  const MatX Qimu = MakeQimu(18);
  MatMotion fast = MatMotion::Zero();
  AddMotionNoiseCov(Rsb, Qimu, fast);

  MatMotionNoise G;
  MotionNoiseJacobian(Rsb, G);
  const MatMotion ref = G * Qimu * G.transpose();
  EXPECT_LT(RelDiff(fast, ref), 1e-14);
  // And it really is only 18 entries out of 576.
  EXPECT_EQ((ref.array() != 0.0).count(), 3 + 9 + 3 + 3);
}

TEST(MotionNoise, TheBlockDiagonalPremiseIsLoadBearing) {
  // Correlate the gyro and accelerometer noises. The four-block form drops
  // exactly the cross terms that a `Qimu` like this would contribute, so it must
  // now be *wrong* -- otherwise the check `Estimator` runs when it builds
  // `Qimu_` would be guarding nothing.
  const Mat3 Rsb = MakeRsb(19);
  MatX Qimu = MakeQimu(20);
  Qimu.block<3, 3>(0, 3).setConstant(0.5);
  Qimu.block<3, 3>(3, 0).setConstant(0.5);

  MatMotion fast = MatMotion::Zero();
  AddMotionNoiseCov(Rsb, Qimu, fast);
  MatMotionNoise G;
  MotionNoiseJacobian(Rsb, G);
  EXPECT_GT(RelDiff(fast, MatMotion(G * Qimu * G.transpose())), 1e-3);
}

} // namespace
