#include "init_linear.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>

namespace xivo {

namespace {

// A 3x3 block is accepted only if it is this well conditioned. Loose on purpose:
// the point is to reject the rank-deficient blocks (a track seen in one frame),
// not to second-guess marginal ones, and a dropped track costs nothing while a
// wrongly kept one poisons the Schur complement.
constexpr number_t kBlockCond = 1e-9;

} // namespace

bool SolveSphereConstrainedQuadratic(const Mat3 &D, const Vec3 &d, number_t r,
                                     Vec3 *g) {
  if (g == nullptr || !(r > 0))
    return false;

  const Mat3 Ds = 0.5 * (D + D.transpose());
  Eigen::SelfAdjointEigenSolver<Mat3> es(Ds);
  if (es.info() != Eigen::Success)
    return false;
  const Vec3 lam = es.eigenvalues(); // ascending
  const Mat3 V = es.eigenvectors();
  const Vec3 e = -(V.transpose() * d);
  const number_t lmin = lam(0);

  const number_t scale =
      std::max<number_t>({number_t(1), lam.cwiseAbs().maxCoeff(), e.norm() / r});
  const number_t tol = 1e-12 * scale;

  auto g_of = [&](number_t l) {
    Vec3 y;
    for (int i = 0; i < 3; ++i)
      y(i) = e(i) / (lam(i) - l);
    return Vec3(V * y);
  };

  // The degenerate ("hard") case: `e` has no component along the eigenspace of
  // lambda_min. Then |g(lambda)| stays bounded as lambda -> lambda_min and there
  // may be no root at all; the minimiser instead sits at lambda = lambda_min
  // with a free component along the null direction, chosen to meet |g| = r.
  // Ignoring this returns garbage exactly when D is near-degenerate, which is
  // what a window with weak parallax looks like.
  bool hard = true;
  for (int i = 0; i < 3; ++i)
    if (lam(i) - lmin <= tol && std::abs(e(i)) > tol)
      hard = false;

  if (hard) {
    Vec3 y = Vec3::Zero();
    for (int i = 0; i < 3; ++i)
      if (lam(i) - lmin > tol)
        y(i) = e(i) / (lam(i) - lmin);
    const Vec3 gp = V * y;
    const number_t n2 = gp.squaredNorm();
    if (n2 <= r * r) {
      // Pick the null direction with the smallest eigenvalue; any of them gives
      // the same objective value, so the choice is arbitrary but must be made.
      const number_t tau = std::sqrt(std::max<number_t>(0, r * r - n2));
      *g = gp + tau * V.col(0);
      return true;
    }
    // |gp| > r means a root does exist below lambda_min after all, so fall
    // through to the bracket below.
  }

  // Bracket. From |g(l)|^2 <= |e|^2 / (lmin - l)^2, |g| < r whenever
  // lmin - l > |e|/r, so `lo` is guaranteed feasible-side; the additive term
  // keeps the bracket non-degenerate when e is zero.
  number_t lo = lmin - e.norm() / r - 1e-3 * scale;
  number_t hi = lmin;
  // Bisect on 1/|g| - 1/r, which is monotone *decreasing* and far better
  // conditioned than |g|^2 - r^2, whose left side blows up like (lmin - l)^-2.
  auto h = [&](number_t l) {
    const number_t n = g_of(l).norm();
    return (n > 0 ? 1.0 / n : std::numeric_limits<number_t>::infinity()) -
           1.0 / r;
  };
  if (!(h(lo) > 0))
    return false; // should not happen given the bound above
  for (int it = 0; it < 200; ++it) {
    const number_t mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi)
      break;
    if (h(mid) > 0)
      lo = mid;
    else
      hi = mid;
  }
  *g = g_of(0.5 * (lo + hi));
  // Bisection converges on lambda, and |g| is stiff in lambda near the root, so
  // finish by projecting exactly onto the sphere rather than trusting the last
  // digit of the iteration.
  const number_t n = g->norm();
  if (!(n > 0))
    return false;
  *g *= r / n;
  return true;
}

LinearInitResult SolveLinearInit(const InitProblem &prob) {
  return SolveLinearInit(prob, LinearInitOptions{});
}

LinearInitResult SolveLinearInit(const InitProblem &prob,
                                 const LinearInitOptions &opt) {
  LinearInitResult out;
  out.features.assign(std::max(0, prob.num_tracks), Vec3::Zero());
  out.used.assign(std::max(0, prob.num_tracks), 0);

  if (prob.frames.size() < 2) {
    out.why = "fewer than two frames";
    return out;
  }
  if (prob.num_tracks < 1 || prob.obs.empty()) {
    out.why = "no observations";
    return out;
  }
  if (!(prob.gravity > 0)) {
    out.why = "non-positive gravity magnitude";
    return out;
  }

  const int N = prob.num_tracks;
  const int nf = static_cast<int>(prob.frames.size());
  const std::vector<int> nframes = prob.TrackFrameCounts();

  // Per-track normal blocks, plus the shared 6x6 motion block.
  std::vector<Mat3> H(N, Mat3::Zero());
  std::vector<Eigen::Matrix<number_t, 3, 6>> C(
      N, Eigen::Matrix<number_t, 3, 6>::Zero());
  std::vector<Vec3> bf(N, Vec3::Zero());
  Eigen::Matrix<number_t, 6, 6> M = Eigen::Matrix<number_t, 6, 6>::Zero();
  Eigen::Matrix<number_t, 6, 1> bm = Eigen::Matrix<number_t, 6, 1>::Zero();
  std::vector<int> nobs(N, 0);

  const number_t t0 = prob.frames.front().t;
  int rows = 0;
  for (const auto &o : prob.obs) {
    if (o.track < 0 || o.track >= N || o.frame < 0 || o.frame >= nf)
      continue;
    if (o.cam < 0 || o.cam >= static_cast<int>(prob.cams.size()))
      continue;
    if (nframes[o.track] < 2)
      continue; // rank-2 block; it would make H singular

    const InitFrame &fr = prob.frames[o.frame];
    const InitCamera &cm = prob.cams[o.cam];
    const number_t dt = fr.t - t0;

    Eigen::Matrix<number_t, 2, 3> Hp;
    Hp << 1, 0, -o.xn(0), 0, 1, -o.xn(1);
    const Eigen::Matrix<number_t, 2, 3> HRc = Hp * cm.Rbc.transpose();
    // fr.pre.R is R_{I0<-Ik}, so its transpose maps I0 into Ik.
    const Eigen::Matrix<number_t, 2, 3> Y = HRc * fr.pre.R.transpose();

    Eigen::Matrix<number_t, 2, 6> Z;
    Z.leftCols<3>() = -dt * Y;
    Z.rightCols<3>() = -0.5 * dt * dt * Y;
    const Vec2 r = Y * fr.pre.alpha + HRc * cm.Tbc;

    H[o.track] += Y.transpose() * Y;
    C[o.track] += Y.transpose() * Z;
    bf[o.track] += Y.transpose() * r;
    M += Z.transpose() * Z;
    bm += Z.transpose() * r;
    ++nobs[o.track];
    rows += 2;
  }
  out.rows = rows;
  if (rows < 12) {
    out.why = "too few usable rows";
    return out;
  }

  // Schur-eliminate the features. H is block diagonal by construction (a feature
  // position appears only in its own observations), which is the whole reason
  // this stays cheap as the window grows.
  Eigen::Matrix<number_t, 6, 6> S = M;
  Eigen::Matrix<number_t, 6, 1> s = bm;
  std::vector<Mat3> Hinv(N);
  for (int i = 0; i < N; ++i) {
    if (nobs[i] < 2)
      continue;
    Eigen::SelfAdjointEigenSolver<Mat3> es(H[i]);
    if (es.info() != Eigen::Success)
      continue;
    const Vec3 ev = es.eigenvalues();
    if (!(ev(0) > kBlockCond * std::max<number_t>(ev(2), 1e-30)))
      continue; // effectively rank deficient: no parallax on this track
    Hinv[i] = H[i].inverse();
    out.used[i] = 1;
    ++out.tracks_used;
    const Eigen::Matrix<number_t, 3, 6> HiC = Hinv[i] * C[i];
    S -= C[i].transpose() * HiC;
    s -= C[i].transpose() * (Hinv[i] * bf[i]);
  }
  if (out.tracks_used < 4) {
    out.why = "fewer than four triangulable tracks";
    return out;
  }

  // Eliminate the velocity too: it is unconstrained, so it can be solved for in
  // closed form given g, leaving the 3-variable sphere problem.
  const Mat3 Svv = S.topLeftCorner<3, 3>();
  const Mat3 Svg = S.topRightCorner<3, 3>();
  const Mat3 Sgg = S.bottomRightCorner<3, 3>();
  const Vec3 sv = s.head<3>();
  const Vec3 sg = s.tail<3>();

  Eigen::SelfAdjointEigenSolver<Mat3> esv(0.5 * (Svv + Svv.transpose()));
  if (esv.info() != Eigen::Success ||
      !(esv.eigenvalues()(0) >
        kBlockCond * std::max<number_t>(esv.eigenvalues()(2), 1e-30))) {
    out.why = "velocity block rank deficient";
    return out;
  }
  const Mat3 Svv_inv = Svv.inverse();
  const Mat3 D = Sgg - Svg.transpose() * Svv_inv * Svg;
  const Vec3 w = sg - Svg.transpose() * (Svv_inv * sv);

  out.g_hess = D;
  out.g_rhs = w;

  Eigen::SelfAdjointEigenSolver<Mat3> esd(0.5 * (D + D.transpose()));
  if (esd.info() == Eigen::Success) {
    const Vec3 ev = esd.eigenvalues();
    out.g_cond = std::abs(ev(2)) > 0 ? ev(0) / ev(2) : 0;
  }

  // The unconstrained minimiser of the full 6-variable system, by truncated
  // eigen-inverse rather than LDLT: `S` is deliberately allowed to be
  // near-singular here (that near-singularity *is* the v/g ambiguity of a short
  // window), and a factorisation would return a large spurious component along
  // the weak direction instead of dropping it.
  {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<number_t, 6, 6>> ess(
        0.5 * (S + S.transpose()));
    if (ess.info() == Eigen::Success) {
      const auto ev = ess.eigenvalues();
      const number_t floor = kBlockCond * std::max<number_t>(ev(5), 1e-30);
      Eigen::Matrix<number_t, 6, 1> y = ess.eigenvectors().transpose() * s;
      for (int i = 0; i < 6; ++i)
        y(i) = ev(i) > floor ? y(i) / ev(i) : number_t(0);
      const Eigen::Matrix<number_t, 6, 1> xf = ess.eigenvectors() * y;
      out.v_free = xf.head<3>();
      out.g_free = xf.tail<3>();
    }
  }

  const bool want_prior = opt.prior_mode != LinearInitOptions::PriorMode::Ignore;
  Vec3 g_prior = Vec3::Zero();
  if (want_prior) {
    const number_t n = opt.gravity_prior.norm();
    if (!(n > 0)) {
      out.why = "gravity prior requested but zero";
      return out;
    }
    g_prior = opt.gravity_prior * (prob.gravity / n);
  }

  Vec3 g;
  if (opt.prior_mode == LinearInitOptions::PriorMode::Force) {
    g = g_prior;
  } else if (opt.constrain_gravity) {
    if (!SolveSphereConstrainedQuadratic(D, -w, prob.gravity, &g)) {
      out.why = "sphere-constrained solve failed";
      return out;
    }
  } else {
    g = out.g_free;
  }

  if (opt.prior_mode == LinearInitOptions::PriorMode::Check) {
    const number_t c = std::max<number_t>(
        -1, std::min<number_t>(1, g.normalized().dot(g_prior.normalized())));
    out.prior_disagreement = std::acos(c);
    if (opt.max_prior_disagreement > 0 &&
        out.prior_disagreement > opt.max_prior_disagreement) {
      out.gravity_flipped = true;
      g = g_prior;
    }
  }
  const Vec3 v = Svv_inv * (sv - Svg * g);

  Eigen::Matrix<number_t, 6, 1> xm;
  xm.head<3>() = v;
  xm.tail<3>() = g;
  for (int i = 0; i < N; ++i)
    if (out.used[i])
      out.features[i] = Hinv[i] * (bf[i] - C[i] * xm);

  // Residual, recomputed from the observations rather than from the normal
  // equations, so a mistake in the accumulation above cannot hide inside it.
  number_t ss = 0;
  int cnt = 0;
  for (const auto &o : prob.obs) {
    if (o.track < 0 || o.track >= N || !out.used[o.track])
      continue;
    if (o.frame < 0 || o.frame >= nf || o.cam < 0 ||
        o.cam >= static_cast<int>(prob.cams.size()))
      continue;
    const InitFrame &fr = prob.frames[o.frame];
    const InitCamera &cm = prob.cams[o.cam];
    const number_t dt = fr.t - t0;
    Eigen::Matrix<number_t, 2, 3> Hp;
    Hp << 1, 0, -o.xn(0), 0, 1, -o.xn(1);
    const Eigen::Matrix<number_t, 2, 3> HRc = Hp * cm.Rbc.transpose();
    const Eigen::Matrix<number_t, 2, 3> Y = HRc * fr.pre.R.transpose();
    const Vec3 p = v * dt + 0.5 * g * dt * dt + fr.pre.alpha;
    const Vec2 res = Y * (out.features[o.track] - p) - HRc * cm.Tbc;
    ss += res.squaredNorm();
    cnt += 2;
  }
  out.residual = cnt > 0 ? std::sqrt(ss / cnt) : 0;
  out.v = v;
  out.g = g;
  out.ok = true;
  out.why = "ok";
  return out;
}

} // namespace xivo
