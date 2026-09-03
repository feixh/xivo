// Stage A: the closed-form linear initializer.
//
// Unknowns are the window's feature positions, the initial velocity and gravity,
// all in `I0`: `x = [p_F1 ... p_FN, v, g]`, size `3N + 6`. Biases are *not*
// recovered here -- they are held at whatever prior the caller passes in, which
// is what makes the problem linear. Stage A only has to land inside the basin
// Stage B converges from.
//
// Each observation contributes two rows. With `H = [[1,0,-u],[0,1,-v]]`, which
// annihilates anything parallel to the bearing `[u,v,1]`, and
// `Y = H * Rbc' * R_k'`:
//
//     Y p_F  -  dt_k Y v  -  0.5 dt_k^2 Y g  =  Y alpha_k + H Rbc' Tbc
//
// This is not the reprojection error -- each row is scaled by the feature's
// depth, so far features are weighted up. That is the standard and accepted cost
// of linearity; Stage B minimises the actual pixel error.
//
// `|g|` must be *enforced*, not hoped for. Left free, the solve trades gravity's
// magnitude against the bias error it cannot see and against scene scale, and it
// will happily report 8.9 m/s^2 with a smaller residual than the truth has. So
// the features and the velocity are eliminated by Schur complement down to a
// 3-variable problem
//
//     min  g' D g + 2 d' g     s.t.  |g| = r
//
// which is the equality-constrained trust-region subproblem, and it has an exact
// characterisation: the global minimiser satisfies `(D - lambda I) g = -d` with
// `lambda <= lambda_min(D)`, and on that branch `|g(lambda)|` is monotone, so
// there is exactly one root to find and no local minima to fall into. See
// `SolveSphereConstrainedQuadratic`.
//
// OpenVINS solves the same subproblem by expanding the optimality conditions
// into a degree-6 polynomial, building its 6x6 companion matrix, running a
// general (non-symmetric) eigensolver, and rank-checking before it starts --
// a check that can and does bail out. The secular form below is ~40 lines, needs
// no rank precondition, and is provably the global minimiser; deriving it here
// also keeps the standing property that nothing from OpenVINS is linked or
// transcribed.
#pragma once

#include "alias.h"
#include "init_problem.h"

namespace xivo {

/** Global minimiser of `g' D g + 2 d' g` subject to `|g| = r`, for symmetric
 *  `D` (only its symmetric part is used) and `r > 0`.
 *
 *  Exposed separately from `SolveLinearInit` because it is the one piece here
 *  with a testable global-optimality claim: `unittest_init_linear` compares it
 *  against brute force over a fine sphere grid, including the indefinite and
 *  the degenerate ("hard") cases, which a Newton iteration seeded near the
 *  unconstrained solution would get wrong. */
bool SolveSphereConstrainedQuadratic(const Mat3 &D, const Vec3 &d, number_t r,
                                     Vec3 *g);

struct LinearInitResult {
  bool ok{false};
  /** Body velocity at frame 0, in `I0` coordinates. */
  Vec3 v{Vec3::Zero()};
  /** Gravity *acceleration* in `I0` coordinates: points down, `|g| = gravity`,
   *  matching the filter's `Rsg * g_` and not its negation. */
  Vec3 g{Vec3::Zero()};
  /** One entry per track, in `I0`. Tracks the solve had to drop are left zero
   *  and flagged in `used`. */
  std::vector<Vec3> features;
  std::vector<char> used;
  int tracks_used{0};
  int rows{0};
  /** RMS of the two-row residual, in the linear system's own (depth-scaled)
   *  units. Useful as a relative indicator across candidate windows, not as a
   *  pixel error. */
  number_t residual{0};
  /** Smallest / largest eigenvalue of the 3x3 gravity Hessian after both Schur
   *  eliminations. A tiny ratio means the window has no parallax to speak of.
   *
   *  Read it only as a parallax indicator, never as a health check on the
   *  answer: a planted gyro bias *raises* it by five orders of magnitude,
   *  because a wrong rotation chain injects signal outside the span of the
   *  {dt, dt^2/2} families whose near-collinearity is what makes a clean short
   *  window ill-conditioned. Gating on it would pass biased windows first. */
  number_t g_cond{0};
  /** The unconstrained (`|g|` free) minimiser, always computed. Kept because an
   *  accelerometer bias is very nearly a gauge direction of `g`: it perturbs
   *  `alpha_k` by `0.5 R ba dt^2`, which the row equation absorbs exactly with
   *  `dg = -R ba`, leaving `v` and the residual untouched. So under bias this
   *  solution is the accurate one and `g_free.norm() - gravity` estimates the
   *  bias component along gravity -- whereas under pixel noise it is the
   *  unstable one, since the weak eigendirection of the gravity Hessian is then
   *  no longer pinned by consistency. */
  Vec3 v_free{Vec3::Zero()};
  Vec3 g_free{Vec3::Zero()};
  /** The reduced gravity subproblem actually solved: minimise
   *  `g' g_hess g - 2 g_rhs' g` over `|g| = gravity`. Exposed so the test can
   *  check global optimality on *realistic* problems built by the accumulation
   *  above, not only on randomly generated `D` matrices, and so a caller can
   *  tell a genuine second minimum from a solver failure. */
  Mat3 g_hess{Mat3::Zero()};
  Vec3 g_rhs{Vec3::Zero()};
  /** Under `PriorMode::Check`: the constrained solve disagreed with the prior
   *  and was discarded. Worth surfacing -- it means this window's linear cost
   *  is bimodal, which is a property of the window, not of the solver. */
  bool gravity_flipped{false};
  /** Angle (rad) between the constrained solve's gravity and the prior, when a
   *  prior was supplied. */
  number_t prior_disagreement{0};
  const char *why{"not run"};
};

struct LinearInitOptions {
  /** Enforce `|g| = prob.gravity`. Measured to be worth ~35x under pixel noise
   *  over a 1.5 s window; leaving it off is only interesting as a diagnostic. */
  bool constrain_gravity{true};
  /** Hold gravity at `gravity_prior` and solve only for `v` and the features.
   *
   *  This exists because the depth-scaled cost is *bimodal*: its two minima can
   *  sit 40 degrees apart in gravity direction and differ in cost by one part in
   *  1e4, so past a small perturbation the global minimiser -- correctly found
   *  -- is the physically wrong one, and the resulting `v` is off by >10 m/s
   *  regardless of window span. No amount of conditioning fixes that, because
   *  nothing in this cost distinguishes the branches. The accelerometer does:
   *  averaged over the window it reads `R'(-g) + ba + mean(R'a)`, so as long as
   *  the mean specific force from real motion is small next to 9.81 its
   *  direction picks the right branch. Holding `g` there also removes the
   *  `v`-vs-`g` ambiguity outright, which is what makes `g_cond` ~1e-6. */
  enum class PriorMode {
    /** No prior: take the constrained solve, flip or not. */
    Ignore,
    /** Take the constrained solve, but if it disagrees with the prior by more
     *  than `max_prior_disagreement` treat it as branch-flipped and fall back
     *  to `Force`. This is the intended production setting: the constrained
     *  solve is far more accurate than the prior when it has not flipped
     *  (0.06 m/s vs 1.5 m/s under pixel noise over 1.5 s), and far worse when
     *  it has (13.7 m/s vs 1.4 m/s), so the prior earns its keep as a
     *  discriminator rather than as an estimate. */
    Check,
    /** Hold gravity at the prior and solve only `v` and the features. */
    Force
  };
  PriorMode prior_mode{PriorMode::Ignore};
  /** Gravity *acceleration* in `I0`; normalised to `prob.gravity` internally. */
  Vec3 gravity_prior{Vec3::Zero()};
  /** Radians. The gap to separate is wide -- a flip is 0.67-0.70 rad while an
   *  honest solve is within the prior's own error, which is
   *  `atan(|mean specific force| / 9.81)` and so a few degrees on real
   *  hand-carried data -- hence a threshold anywhere in between works. */
  number_t max_prior_disagreement{0.30};
};

/** Solve Stage A. `prob.frames[k].pre` must already be preintegrated from frame
 *  0 at the bias prior the caller intends. */
LinearInitResult SolveLinearInit(const InitProblem &prob,
                                 const LinearInitOptions &opt);
LinearInitResult SolveLinearInit(const InitProblem &prob);

} // namespace xivo
