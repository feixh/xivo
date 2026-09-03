// Stage B: the initialization bundle adjustment.
//
// Stage A recovers the velocity and gravity in closed form but pays for that
// linearity twice: it minimizes a depth-scaled surrogate rather than the pixel
// error, and it cannot see the IMU biases at all. On real EuRoC windows the
// second cost dominates -- holding the biases at zero leaves 0.11-0.23 m/s of
// velocity error and 4.1 degrees of gravity tilt, and handing the same code
// EuRoC's own solved biases collapses both by a factor of 5-10 (see
// notes-n-prompts/notes-dyninit/m2-linear.md). So the *gyro* bias is what Stage B
// is for; the velocity refinement follows from getting it right. Measured over
// the 11 EuRoC windows: velocity error 0.173 -> 0.017 m/s and gravity tilt
// 4.15 -> 0.90 degrees, improving on every one of the 11.
//
// The accelerometer bias is deliberately *not* estimated -- over 1.5 s it is
// nearly indistinguishable from a gravity tilt, and freeing it makes gravity
// worse than the seed. It is held near zero by a tight prior; see
// `sigma_ba_prior` for the measurements behind that decision, and note that the
// 0.90 degree tilt above is the floor that choice implies, not solver slack.
//
// Unknowns, in a **gravity-aligned world frame `W`** whose `z` axis is up:
//
//   per frame k:  R_{W<-Ik} (3, right increments)   p_{Ik}^W (3)   v_{Ik}^W (3)
//   global:       bg (3)   ba (3)
//   per track:    p_F^W (3)
//
// Gravity is *not* a variable: it is `[0, 0, -gravity]` by the definition of `W`.
// Its two free directions are the roll and pitch of frame 0, which are ordinary
// unknowns here, so the gravity direction is refined without a constraint to
// maintain and with no way for `|g|` to drift. What that leaves is a 4-dof gauge
// -- global translation and a rotation about world `z` -- and both are exact
// symmetries of every residual below (a yaw `Q` about `z` satisfies `Q g_W = g_W`,
// so `R_i'(p_j - p_i - v_i dt - 0.5 g dt^2)` is invariant under
// `p, v, f -> Qp, Qv, Qf` and `R -> QR`). Translation is removed by holding
// `p_0 = 0` exactly; yaw by one prior row. Because the yaw direction is an exact
// nullspace of the data terms, that prior changes the answer not at all and its
// weight can be modest -- unlike the usual 1e6-weight pin, which trades a
// nullspace for a badly conditioned Hessian.
//
// Residuals:
//
//   IMU, one 9-vector per consecutive frame pair (i, j = i+1), dt = t_j - t_i:
//     r_alpha = R_i'(p_j - p_i - v_i dt - 0.5 g_W dt^2) - alpha_ij(bg, ba)
//     r_beta  = R_i'(v_j - v_i - g_W dt)                - beta_ij(bg, ba)
//     r_theta = Log( R_ij(bg)' R_i' R_j )
//   Reprojection, one 2-vector per observation, Cauchy-robustified:
//     r_pix = pi( Rbc'(R_k'(f - p_k) - Tbc) ) - xn
//
// `alpha_ij`, `beta_ij` and `R_ij` are the preintegrals corrected to the current
// bias to first order through the Jacobians `init_preint` carries. That first-
// order correction is what makes the biases *observable* rather than merely
// present: without it the IMU residual would not depend on `bg` or `ba` at all
// and the solver would happily report whatever it was seeded with.
//
// Solver: Levenberg-Marquardt with the Schur complement on the track block, and
// Marquardt's `lambda * diag(H)` scaling rather than `lambda * I`, because these
// parameters differ in scale by six orders of magnitude (radians of gyro bias
// against metres of feature position) and `lambda * I` would damp them by wildly
// unequal amounts. Hand-rolled in Eigen; see plan-dyninit.md section 4 for why
// Ceres is deliberately not linked.
#pragma once

#include <vector>

#include "alias.h"
#include "init_linear.h"
#include "init_problem.h"

namespace xivo {

/** The full window state Stage B optimizes, in the gravity-aligned world frame
 *  `W`. `R[k]` is `R_{W<-Ik}`; `p[k]`, `v[k]` and `f[n]` are `W`-frame vectors.
 *
 *  `p[0]` is the origin of `W` and is held there. Frame 0's *attitude* is free
 *  except for its yaw, which is what carries the gravity-direction refinement. */
struct BAState {
  std::vector<Mat3> R;
  std::vector<Vec3> p;
  std::vector<Vec3> v;
  Vec3 bg{Vec3::Zero()};
  Vec3 ba{Vec3::Zero()};
  std::vector<Vec3> f;
  /** Per track: whether it takes part. Tracks Stage A dropped stay out, since a
   *  track without a triangulation is a free 3-vector with no constraint and
   *  would make the Schur block singular. */
  std::vector<char> used;
  number_t gravity{9.81};

  /** Gravity *acceleration* in `W`: down, magnitude `gravity`. */
  Vec3 GravityW() const { return Vec3{0, 0, -gravity}; }
  int num_frames() const { return static_cast<int>(R.size()); }
  int num_tracks() const { return static_cast<int>(f.size()); }
  /** Body velocity at frame `k`, in that frame's own coordinates -- the form the
   *  filter's `X_.Vsb` wants. */
  Vec3 VelocityInBody(int k) const { return R[k].transpose() * v[k]; }
  /** Gravity acceleration in frame `k`'s coordinates, i.e. what Stage A reports
   *  for `k = 0`. Feeds the `Rsg` handoff. */
  Vec3 GravityInBody(int k) const { return R[k].transpose() * GravityW(); }
};

struct BAOptions {
  /** Image noise, pixels. Divided by `InitCamera::focal` to reach the normalized
   *  coordinates the residual is written in. */
  number_t sigma_pix{1.0};
  /** Cauchy scale, in units of `sigma_pix`. Applies to the reprojection family
   *  only: an IMU edge over 50 ms has no outliers to speak of, while a KLT track
   *  that has drifted onto a different corner is the normal case. */
  number_t cauchy_c{3.0};
  /** IMU white-noise densities, in rad/s/sqrt(Hz) and m/s^2/sqrt(Hz). Defaults
   *  are EuRoC's ADIS16448 figures. Per edge these give
   *  `sigma_theta = sigma_g sqrt(dt)`, `sigma_beta = sigma_a sqrt(dt)` and
   *  `sigma_alpha = sigma_a sqrt(dt) dt / sqrt(3)`.
   *
   *  This is the diagonal approximation to the preintegration covariance: it
   *  drops the alpha/beta correlation and the gyro's contribution to alpha. Both
   *  matter for a filter that will carry the result forward as a covariance; for
   *  choosing *relative* weights between two residual families that differ by
   *  two orders of magnitude in accuracy it does not, and a full covariance
   *  recursion is 60 lines that M6 can add if it ever needs the covariance for
   *  its own sake. */
  number_t sigma_g{1.6968e-4};
  number_t sigma_a{2.0e-3};
  /** Prior sigmas pulling the biases toward the *seed's* values -- which at a
   *  cold start are zero, but are whatever a caller with a calibrated bias put
   *  there, which is the useful generalization.
   *
   *  These defaults are asymmetric because the two biases are not equally
   *  observable over 1.5 seconds, and measurement says so loudly. The gyro bias
   *  enters `r_theta` directly and is recovered to 0.0028 rad/s against EuRoC's
   *  own solved value -- a 3% error on a true 0.08 rad/s -- so it needs no prior
   *  at all, and sweeping `sigma_bg_prior` over two decades (0.02, 0.1, 1.0)
   *  moves the velocity error by 0.003 m/s and the bias error not at all. The
   *  accelerometer bias is a different matter: over a short window it is very
   *  nearly indistinguishable from a gravity tilt, and left free it *takes* that
   *  interpretation. Unpriored on the 11 EuRoC windows it runs to |ba| ~ 9 m/s^2
   *  -- gravity-sized -- and drags the recovered gravity direction with it, so
   *  that Stage B ends up 68% *worse* than the Stage A seed it started from
   *  (6.95 deg of tilt against 4.15). Priored at 0.01 it stays near zero and the
   *  tilt improves by 78% instead (0.90 deg).
   *
   *  So `ba` is, in effect, held at zero, and the honest way to read `0.01` is
   *  "not estimated". That is the same conclusion VINS-Mono reaches and states
   *  outright, and it costs exactly what theory says it should: with `ba` at
   *  zero, an accel bias of `b` perpendicular to gravity *must* appear as
   *  `|b|/9.81` radians of tilt, and across the 11 windows the measured tilt
   *  tracks that prediction at r = 0.93 with a slope of 0.82 (below 1 because
   *  `ba`'s component along gravity tilts nothing). The 0.90 deg mean is
   *  therefore a floor set by the window length, not slack in the solver; V1_01,
   *  whose true |ba| is 0.55 m/s^2 where every other sequence sits near 0.15, is
   *  duly the worst window at 2.6 deg. Beating it needs a longer window, not a
   *  better optimizer. Note also that this is a prior toward zero, not toward a
   *  guess: it does not inject a bias value, it declines to estimate one.
   *
   *  Zero disables either prior. See notes-n-prompts/notes-dyninit/m3-ba.md for
   *  the full sweep, including why 0.003 and 0.001 are worse than 0.01. */
  number_t sigma_bg_prior{0};
  number_t sigma_ba_prior{0.01};
  /** Sigma on frame 0's world yaw, radians. Pins the one remaining gauge
   *  direction; see the header comment for why a modest value is correct. */
  number_t sigma_yaw{1e-3};
  /** Predicted depths are clamped below at this, so an observation that lands
   *  behind the camera mid-solve produces a large but finite residual instead of
   *  a sign-flipped one. Note *clamped*, not dropped: dropping would make the row
   *  count depend on the state, and LM would then buy cost reductions by pushing
   *  features out of the problem. See the comment at the clamp in init_ba.cpp. */
  number_t min_depth{0.05};

  /** Also report the marginal covariance of the handoff state (`BAResult::cov`).
   *  Off by default because it costs one extra residual accumulation plus an
   *  LDLT of the reduced system -- 25-40 ms on a 41-frame window, next to a
   *  300-1300 ms solve -- and nothing on the shipped path reads it. `linear_probe
   *  -cov` is the only caller that sets it: M6 measured whether the filter should
   *  start from this matrix instead of its config priors, and the answer was no.
   *  See notes-n-prompts/notes-dyninit/m6-covariance.md. */
  bool want_covariance{false};

  int max_iterations{30};
  /** Stop when one accepted step improves the cost by less than this fraction. */
  number_t cost_tol{1e-10};
  /** Stop when the accepted step's infinity norm falls below this. */
  number_t step_tol{1e-12};
  number_t lambda_init{1e-4};
  number_t lambda_max{1e14};
  /** Consecutive rejected steps before giving up on an iteration. */
  int max_rejections{10};
};

struct BAResult {
  bool ok{false};
  BAState state;
  int iterations{0};
  /** Rejected LM steps, summed over all iterations. A large count next to
   *  `iterations` means the Jacobians and the cost disagree -- the signature of
   *  a derivative bug, which is why it is reported rather than hidden. */
  int rejections{0};
  number_t cost_init{0};
  number_t cost_final{0};
  /** RMS reprojection error in *pixels*, unrobustified, over the observations
   *  actually used. The number to quote: it is comparable across windows and
   *  across configurations, which the weighted cost is not. */
  number_t pixel_rms{0};
  /** Median reprojection error, px. Quote this next to `pixel_rms`, never
   *  instead of it: the median says whether the bulk of the window fits, the RMS
   *  says how much gross outlier energy the robust loss is carrying. Converged
   *  EuRoC windows read 0.03-0.68 px median (MH_01: 0.335) with an RMS up to 10
   *  on the same window -- which is the whole reason the median is here. */
  number_t pixel_median{0};
  /** RMS of the whitened IMU residual, dimensionless. ~1 means the IMU edges are
   *  fitted to their own noise level; >>1 means something the model cannot
   *  express (unmodelled bias drift, a bad extrinsic, a clock offset). */
  number_t imu_rms{0};
  int obs_used{0};
  int tracks_used{0};
  const char *why{"not run"};

  /** Marginal covariance of `(v_last, bg, ba)`, in that order, at the converged
   *  state: the corresponding 9x9 block of the inverse of the Schur-complemented
   *  reduced information matrix, undamped, with the translation gauge pinned the
   *  way `SolveStep` pins it. Velocity is in `W`; `BAState::VelocityInBody`'s
   *  rotation applies to it as `R_k' C R_k`, and because a change of gauge maps
   *  `R_k -> Q R_k` and `C -> Q C Q'`, the body-frame covariance is
   *  gauge-invariant while this one is not.
   *
   *  Only populated when `BAOptions::want_covariance` is set, and then only if
   *  the reduced system factorized -- read `cov_ok`, not the matrix. **It is the
   *  covariance of this cost function, not of the true error**: the IMU edges are
   *  whitened by the diagonal approximation documented at `BAOptions::sigma_g`,
   *  and `ba` is priored rather than estimated, so its block reports the prior
   *  back. M6 measured how far that is from the truth -- 10-500x too tight, and
   *  worst exactly where the window is hard -- which is why nothing consumes it;
   *  see notes-n-prompts/notes-dyninit/m6-covariance.md. */
  Mat9 cov{Mat9::Zero()};
  bool cov_ok{false};
};

/** Turn Stage A's answer into a Stage B seed.
 *
 *  Builds `W` by rotating Stage A's gravity onto `[0, 0, -gravity]`, which fixes
 *  frame 0's roll and pitch and leaves its yaw arbitrary -- the gauge freedom
 *  Stage B then pins. Frame poses come from the preintegral chain rather than
 *  from a fresh integration, so the seed is exactly the state Stage A's residual
 *  was evaluated at. Returns false if Stage A failed or left too few tracks. */
bool SeedBAState(const InitProblem &prob, const LinearInitResult &lin,
                 BAState *seed);

/** Refine `seed` in place of nothing -- returns a new state, leaving the seed
 *  alone so a caller can fall back to it. `prob.frames[k].pre_prev` must be
 *  populated. */
BAResult SolveInitBA(const InitProblem &prob, const BAState &seed,
                     const BAOptions &opt);
BAResult SolveInitBA(const InitProblem &prob, const BAState &seed);

/** The cost `SolveInitBA` minimizes, at an arbitrary state. Exposed because the
 *  monotone-descent and numerical-Jacobian tests need to evaluate it at states
 *  the solver never visits, and because a caller comparing two candidate windows
 *  wants the same number the solver used. */
number_t InitBACost(const InitProblem &prob, const BAState &state,
                    const BAOptions &opt, const BAState &gauge_ref);

/** The whitened residual and its dense Jacobian at `state`, for the numerical-
 *  Jacobian test.
 *
 *  The solver never forms this matrix -- it goes straight to the Schur-complemented
 *  normal equations. The dense rows are emitted by the *same* accumulation call as
 *  the sparse ones rather than by a parallel code path, which is the only version
 *  of this hook worth having: a dense assembler written separately would be a
 *  second implementation, and agreeing with finite differences would then say
 *  nothing about the one the solver runs.
 *
 *  Column layout: frame `k` occupies `[9k, 9k+9)` as `(dtheta, dp, dv)`, then
 *  `bg` and `ba`, then one 3-block per participating track -- `track_col[n]` gives
 *  each track's column, or -1 if it does not take part. Rotation columns are
 *  derivatives with respect to a *right* increment, `R <- R exp(dtheta)`. */
bool InitBALinearize(const InitProblem &prob, const BAState &state,
                     const BAOptions &opt, const BAState &gauge_ref, VecX *r,
                     MatX *J, std::vector<int> *track_col);

} // namespace xivo
