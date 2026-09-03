#include "init_ba.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "init_preint.h"
#include "rodrigues.h"

namespace xivo {
namespace {

using Mat3x9 = Eigen::Matrix<number_t, 3, 9>;
using Vec9 = Eigen::Matrix<number_t, 9, 1>;

Vec3 SO3Log(const Mat3 &R) {
  return SO3(Eigen::Quaternion<number_t>(R).normalized()).log();
}

Mat3 Orthonormalize(const Mat3 &R) {
  return SO3(Eigen::Quaternion<number_t>(R).normalized()).matrix();
}

// Column offsets in the reduced (non-track) parameter block. Every block is
// three wide, which is what lets one accumulation helper serve every residual
// family below.
inline int Ith(int k) { return 9 * k; }     // rotation increment, right/body
inline int Ip(int k) { return 9 * k + 3; }  // position
inline int Iv(int k) { return 9 * k + 6; }  // velocity

/** Normal equations, split the way the Schur complement wants them: the reduced
 *  block dense, the track blocks 3x3, and the coupling stored only for the
 *  (track, frame) pairs that exist. */
struct Normal {
  int M{0};
  MatX Hcc;
  VecX rhs; ///< -J'r on the reduced block
  std::vector<Mat3> Hff;
  std::vector<Vec3> rf; ///< -J'r on each track
  /** `Hfc[n][s]` couples track `n` to the frame in slot `s` of its own frame
   *  list. Sparse by slot rather than dense by frame, because the Schur update
   *  costs (frames this track was seen in)^2 and not (frames)^2. */
  std::vector<std::vector<Mat3x9>> Hfc;
  VecX diag_c;
  std::vector<Vec3> diag_f;
};

struct Stats {
  number_t pix_sq_px{0}; ///< sum of squared reprojection error, px^2
  int pix_obs{0};
  /** Every observation's reprojection error in px, so the *median* can be
   *  reported alongside the RMS. Necessary, not decorative: on MH_01 the two
   *  read 9.4 px and 0.35 px, because a handful of KLT tracks that walked onto a
   *  different corner dominate a sum of squares over ~4000 observations. Reading
   *  the RMS alone says "the solve is stuck" when the robust loss has in fact
   *  identified those tracks and the rest fit to a third of a pixel. */
  std::vector<number_t> pix_err_px;
  number_t imu_sq{0}; ///< sum of squared *whitened* IMU residual
  int imu_rows{0};
};

/** Optional dense output, filled by the same emit call that feeds the normal
 *  equations. Only the test asks for it. */
struct DenseLin {
  MatX J;
  VecX r;
  int row{0};
};

template <int NR> struct CamBlocks {
  int n{0};
  int col[8];
  Eigen::Matrix<number_t, NR, 3> J[8];
  void Add(int c, const Eigen::Matrix<number_t, NR, 3> &j) {
    col[n] = c;
    J[n] = j;
    ++n;
  }
};

/** Accumulate `J'J` and `-J'r` for one whitened residual over its parameter
 *  blocks. Both triangles are filled: the extra work is negligible next to the
 *  Schur complement and it removes a whole class of "which triangle did I mean"
 *  bug from the solve. */
template <int NR>
void AccumCam(const Eigen::Matrix<number_t, NR, 1> &r, const CamBlocks<NR> &b,
              MatX *H, VecX *rhs) {
  for (int i = 0; i < b.n; ++i) {
    rhs->segment<3>(b.col[i]).noalias() -= b.J[i].transpose() * r;
    for (int j = 0; j < b.n; ++j)
      H->block<3, 3>(b.col[i], b.col[j]).noalias() +=
          b.J[i].transpose() * b.J[j];
  }
}

/** Everything about the window that does not change as the state moves: the
 *  observation index, the per-edge IMU weights, and the gauge reference. Built
 *  once so that `Evaluate` is a pure function of the state -- which is what lets
 *  the cost used by the line search and the cost differentiated by the Jacobians
 *  be literally the same code. */
class Evaluator {
public:
  Evaluator(const InitProblem &prob, const BAOptions &opt, const BAState &ref)
      : prob_(prob), opt_(opt), ref_(ref) {
    K_ = static_cast<int>(prob.frames.size());
    N_ = prob.num_tracks;
    M_ = 9 * K_ + 6;
    ibg_ = 9 * K_;
    iba_ = 9 * K_ + 3;
    if (K_ < 2 || N_ <= 0 || prob.cams.empty())
      return;
    if (static_cast<int>(ref.R.size()) != K_ ||
        static_cast<int>(ref.p.size()) != K_ ||
        static_cast<int>(ref.v.size()) != K_ ||
        static_cast<int>(ref.f.size()) != N_ ||
        static_cast<int>(ref.used.size()) != N_)
      return;

    inv_sigma_xn_.resize(prob.cams.size());
    for (size_t c = 0; c < prob.cams.size(); ++c) {
      const number_t f = prob.cams[c].focal > 0 ? prob.cams[c].focal : 1;
      const number_t s = opt.sigma_pix > 0 ? opt.sigma_pix / f : 1;
      inv_sigma_xn_[c] = 1.0 / s;
    }
    // The Cauchy scale lives in whitened units, so `cauchy_c` is a multiple of
    // sigma_pix and needs no focal length.
    cauchy_c2_ = opt.cauchy_c > 0 ? opt.cauchy_c * opt.cauchy_c : 0;

    slot_.assign(static_cast<size_t>(N_) * K_, -1);
    frames_.resize(N_);
    obs_.resize(N_);
    for (size_t i = 0; i < prob.obs.size(); ++i) {
      const auto &o = prob.obs[i];
      if (o.track < 0 || o.track >= N_ || o.frame < 0 || o.frame >= K_)
        continue;
      if (o.cam < 0 || o.cam >= static_cast<int>(prob.cams.size()))
        continue;
      if (!ref.used[o.track])
        continue;
      obs_[o.track].push_back(static_cast<int>(i));
      int &s = slot_[static_cast<size_t>(o.track) * K_ + o.frame];
      if (s < 0) {
        s = static_cast<int>(frames_[o.track].size());
        frames_[o.track].push_back(o.frame);
      }
    }
    dense_col_.assign(N_, -1);
    for (int n = 0; n < N_; ++n)
      if (ref.used[n] && frames_[n].size() >= 2) {
        dense_col_[n] = M_ + 3 * static_cast<int>(tracks_.size());
        tracks_.push_back(n);
      }
    ok_ = !tracks_.empty();
  }

  bool ok() const { return ok_; }
  int M() const { return M_; }
  int K() const { return K_; }
  int N() const { return N_; }
  const std::vector<int> &tracks() const { return tracks_; }
  const std::vector<int> &frames(int n) const { return frames_[n]; }
  int slot(int n, int k) const { return slot_[static_cast<size_t>(n) * K_ + k]; }
  const std::vector<int> &dense_cols() const { return dense_col_; }
  int dense_width() const { return M_ + 3 * static_cast<int>(tracks_.size()); }
  int max_rows() const {
    return 9 * K_ + 2 * static_cast<int>(prob_.obs.size()) + 7;
  }

  /** The cost at `st`. If `nrm` is non-null the normal equations are accumulated
   *  at the same time; if `stats` is non-null the per-family error summaries are
   *  too; if `dn` is non-null the dense rows are written as well. */
  number_t Evaluate(const BAState &st, Normal *nrm, Stats *stats,
                    DenseLin *dn = nullptr) const;

private:
  /** Feed one whitened residual to whichever consumers are attached. Does not
   *  advance `dn->row`: the pixel family has a track column to write first. */
  template <int NR>
  void Emit(const Eigen::Matrix<number_t, NR, 1> &rw, const CamBlocks<NR> &b,
            Normal *nrm, DenseLin *dn) const {
    if (nrm != nullptr)
      AccumCam<NR>(rw, b, &nrm->Hcc, &nrm->rhs);
    if (dn != nullptr) {
      dn->r.template segment<NR>(dn->row) = rw;
      for (int i = 0; i < b.n; ++i)
        dn->J.template block<NR, 3>(dn->row, b.col[i]) += b.J[i];
    }
  }

  const InitProblem &prob_;
  const BAOptions &opt_;
  const BAState &ref_;
  bool ok_{false};
  int K_{0}, N_{0}, M_{0}, ibg_{0}, iba_{0};
  number_t cauchy_c2_{0};
  std::vector<number_t> inv_sigma_xn_;
  std::vector<int> slot_;
  std::vector<std::vector<int>> frames_;
  std::vector<std::vector<int>> obs_;
  std::vector<int> tracks_;
  std::vector<int> dense_col_;
};

number_t Evaluator::Evaluate(const BAState &st, Normal *nrm, Stats *stats,
                             DenseLin *dn) const {
  if (dn != nullptr) {
    dn->J.setZero(max_rows(), dense_width());
    dn->r.setZero(max_rows());
    dn->row = 0;
  }
  if (nrm != nullptr) {
    nrm->M = M_;
    nrm->Hcc.setZero(M_, M_);
    nrm->rhs.setZero(M_);
    nrm->Hff.assign(N_, Mat3::Zero());
    nrm->rf.assign(N_, Vec3::Zero());
    nrm->Hfc.assign(N_, {});
    for (int n : tracks_)
      nrm->Hfc[n].assign(frames_[n].size(), Mat3x9::Zero());
  }
  number_t cost = 0;

  // ---------------------------------------------------------------- IMU edges
  const Vec3 gW = st.GravityW();
  for (int j = 1; j < K_; ++j) {
    const int i = j - 1;
    const Preintegral &pre = prob_.frames[j].pre_prev;
    const number_t dt = pre.dt > 0 ? pre.dt : prob_.frames[j].t - prob_.frames[i].t;
    if (!(dt > 0))
      continue;
    const Vec3 dbg = st.bg - pre.bg;
    const Mat3 Rij = pre.RAt(st.bg);
    const Vec3 alpha = pre.AlphaAt(st.bg, st.ba);
    const Vec3 beta = pre.BetaAt(st.bg, st.ba);

    const Mat3 Ri = st.R[i], Rj = st.R[j];
    const Vec3 Av =
        st.p[j] - st.p[i] - st.v[i] * dt - 0.5 * gW * dt * dt;
    const Vec3 Bv = st.v[j] - st.v[i] - gW * dt;
    const Mat3 E = Rij.transpose() * Ri.transpose() * Rj;
    const Vec3 rt = SO3Log(E);

    Vec9 r;
    r.segment<3>(0) = Ri.transpose() * Av - alpha;
    r.segment<3>(3) = Ri.transpose() * Bv - beta;
    r.segment<3>(6) = rt;

    // Diagonal whitening; see BAOptions::sigma_g for what it drops.
    const number_t sq = std::sqrt(dt);
    const number_t s_a = std::max<number_t>(opt_.sigma_a * sq * dt / std::sqrt(3.0), 1e-12);
    const number_t s_b = std::max<number_t>(opt_.sigma_a * sq, 1e-12);
    const number_t s_t = std::max<number_t>(opt_.sigma_g * sq, 1e-12);
    Vec9 w;
    w.segment<3>(0).setConstant(1.0 / s_a);
    w.segment<3>(3).setConstant(1.0 / s_b);
    w.segment<3>(6).setConstant(1.0 / s_t);
    const Vec9 rw = r.cwiseProduct(w);
    cost += rw.squaredNorm();
    if (stats != nullptr) {
      stats->imu_sq += rw.squaredNorm();
      stats->imu_rows += 9;
    }
    if (nrm == nullptr && dn == nullptr)
      continue;

    const Mat3 Rit = Ri.transpose();
    const Mat3 Jr_inv = SO3RightJacobianInverse(rt);
    // `RAt` applies the whole bias offset inside one exponential, so the
    // derivative with respect to a further increment carries that offset's right
    // Jacobian. It is a 0.2% correction at EuRoC's bias magnitudes -- invisible
    // to a solver that converges anyway, and the difference between passing and
    // failing a 1e-6 central-difference check.
    const Mat3 Jbg = SO3RightJacobian(pre.dR_dbg * dbg) * pre.dR_dbg;

    CamBlocks<9> b;
    Eigen::Matrix<number_t, 9, 3> J;

    J.setZero(); // theta_i
    J.block<3, 3>(0, 0) = hat(Rit * Av);
    J.block<3, 3>(3, 0) = hat(Rit * Bv);
    J.block<3, 3>(6, 0) = -Jr_inv * Rj.transpose() * Ri;
    b.Add(Ith(i), w.asDiagonal() * J);

    J.setZero(); // p_i
    J.block<3, 3>(0, 0) = -Rit;
    b.Add(Ip(i), w.asDiagonal() * J);

    J.setZero(); // v_i
    J.block<3, 3>(0, 0) = -Rit * dt;
    J.block<3, 3>(3, 0) = -Rit;
    b.Add(Iv(i), w.asDiagonal() * J);

    J.setZero(); // theta_j
    J.block<3, 3>(6, 0) = Jr_inv;
    b.Add(Ith(j), w.asDiagonal() * J);

    J.setZero(); // p_j
    J.block<3, 3>(0, 0) = Rit;
    b.Add(Ip(j), w.asDiagonal() * J);

    J.setZero(); // v_j
    J.block<3, 3>(3, 0) = Rit;
    b.Add(Iv(j), w.asDiagonal() * J);

    J.setZero(); // bg
    J.block<3, 3>(0, 0) = -pre.dalpha_dbg;
    J.block<3, 3>(3, 0) = -pre.dbeta_dbg;
    J.block<3, 3>(6, 0) = -Jr_inv * E.transpose() * Jbg;
    b.Add(ibg_, w.asDiagonal() * J);

    J.setZero(); // ba
    J.block<3, 3>(0, 0) = -pre.dalpha_dba;
    J.block<3, 3>(3, 0) = -pre.dbeta_dba;
    b.Add(iba_, w.asDiagonal() * J);

    Emit<9>(rw, b, nrm, dn);
    if (dn != nullptr)
      dn->row += 9;
  }

  // ------------------------------------------------------------- reprojection
  for (int n : tracks_) {
    for (int oi : obs_[n]) {
      const auto &o = prob_.obs[oi];
      const InitCamera &cam = prob_.cams[o.cam];
      const int k = o.frame;
      const Mat3 Rkt = st.R[k].transpose();
      const Vec3 Xb = Rkt * (st.f[n] - st.p[k]);
      const Vec3 Xc = cam.Rbc.transpose() * (Xb - cam.Tbc);
      // Depth is clamped, not skipped. Skipping would make the *number of rows* a
      // function of the state, and LM -- which accepts any step that lowers "the
      // cost" -- then lowers it by pushing features behind the camera and deleting
      // their rows. It did: from a Stage A seed 1.2 m/s off, the cost fell to
      // 1e-10 with a 1.4 m/s velocity error and a pixel RMS of 2e-7 on data
      // carrying 0.3 px of noise, because most of the observations had quietly
      // left the problem. Clamping keeps the residual a continuous function of the
      // whole state, so descent means what it says.
      const bool clamped = Xc(2) < opt_.min_depth;
      const number_t inv_z = 1.0 / (clamped ? opt_.min_depth : Xc(2));
      const number_t u = Xc(0) * inv_z, v = Xc(1) * inv_z;
      const number_t iw = inv_sigma_xn_[o.cam];
      const Vec2 r{(u - o.xn(0)) * iw, (v - o.xn(1)) * iw};
      const number_t s = r.squaredNorm();

      // Cauchy: rho(s) = c^2 log(1 + s/c^2), rho'(s) = 1/(1 + s/c^2). Weighting
      // both residual and Jacobian by sqrt(rho') gives the Gauss-Newton step for
      // the robust cost, dropping the Triggs second-order term as usual.
      number_t rho = s, drho = 1;
      if (cauchy_c2_ > 0) {
        const number_t q = 1.0 + s / cauchy_c2_;
        rho = cauchy_c2_ * std::log(q);
        drho = 1.0 / q;
      }
      cost += rho;
      if (stats != nullptr) {
        const number_t f = cam.focal > 0 ? cam.focal : 1;
        const number_t e =
            (Vec2{u - o.xn(0), v - o.xn(1)}).norm() * f;
        stats->pix_sq_px += e * e;
        stats->pix_err_px.push_back(e);
        ++stats->pix_obs;
      }
      if (nrm == nullptr && dn == nullptr)
        continue;

      const number_t sw = std::sqrt(drho) * iw;
      // The exact derivative of the clamped projection: past the clamp `z` no
      // longer depends on the state, so the third column goes with it.
      Eigen::Matrix<number_t, 2, 3> Jpix;
      if (clamped)
        Jpix << inv_z, 0, 0, 0, inv_z, 0;
      else
        Jpix << inv_z, 0, -u * inv_z, 0, inv_z, -v * inv_z;
      const Eigen::Matrix<number_t, 2, 3> A =
          sw * Jpix * cam.Rbc.transpose();
      const Eigen::Matrix<number_t, 2, 3> Jf = A * Rkt;
      const Eigen::Matrix<number_t, 2, 3> Jth = A * hat(Xb);
      const Vec2 rw = std::sqrt(drho) * r;

      CamBlocks<2> b;
      b.Add(Ith(k), Jth);
      b.Add(Ip(k), -Jf); // exact: d Xb/dp = -d Xb/df
      Emit<2>(rw, b, nrm, dn);
      if (dn != nullptr) {
        dn->J.block<2, 3>(dn->row, dense_col_[n]) += Jf;
        dn->row += 2;
      }
      if (nrm == nullptr)
        continue;

      nrm->Hff[n].noalias() += Jf.transpose() * Jf;
      nrm->rf[n].noalias() -= Jf.transpose() * rw;
      Mat3x9 &C = nrm->Hfc[n][slot(n, k)];
      C.block<3, 3>(0, 0).noalias() += Jf.transpose() * Jth;
      C.block<3, 3>(0, 3).noalias() -= Jf.transpose() * Jf;
    }
  }

  // ------------------------------------------------------- priors and the gauge
  if (opt_.sigma_bg_prior > 0) {
    const number_t iw = 1.0 / opt_.sigma_bg_prior;
    const Vec3 r = (st.bg - ref_.bg) * iw;
    cost += r.squaredNorm();
    CamBlocks<3> b;
    b.Add(ibg_, Mat3(Mat3::Identity() * iw));
    Emit<3>(r, b, nrm, dn);
    if (dn != nullptr)
      dn->row += 3;
  }
  if (opt_.sigma_ba_prior > 0) {
    const number_t iw = 1.0 / opt_.sigma_ba_prior;
    const Vec3 r = (st.ba - ref_.ba) * iw;
    cost += r.squaredNorm();
    CamBlocks<3> b;
    b.Add(iba_, Mat3(Mat3::Identity() * iw));
    Emit<3>(r, b, nrm, dn);
    if (dn != nullptr)
      dn->row += 3;
  }
  if (opt_.sigma_yaw > 0) {
    const number_t iw = 1.0 / opt_.sigma_yaw;
    const Mat3 M0 = st.R[0] * ref_.R[0].transpose();
    const Vec3 y = SO3Log(M0);
    Eigen::Matrix<number_t, 1, 1> r;
    r(0) = y(2) * iw;
    cost += r.squaredNorm();
    const Eigen::Matrix<number_t, 1, 3> Jy =
        iw * (SO3RightJacobianInverse(y) * M0.transpose() * st.R[0]).row(2);
    CamBlocks<1> b;
    b.Add(Ith(0), Jy);
    Emit<1>(r, b, nrm, dn);
    if (dn != nullptr)
      dn->row += 1;
  }

  if (dn != nullptr) {
    dn->J.conservativeResize(dn->row, Eigen::NoChange);
    dn->r.conservativeResize(dn->row);
  }
  if (nrm != nullptr) {
    nrm->diag_c = nrm->Hcc.diagonal();
    nrm->diag_f.assign(N_, Vec3::Zero());
    for (int n : tracks_)
      nrm->diag_f[n] = nrm->Hff[n].diagonal();
  }
  return cost;
}

/** One damped Gauss-Newton step: Schur out the tracks, pin the gauge, solve the
 *  reduced system, back-substitute. Returns false only if the reduced system is
 *  not solvable, which LM answers by raising lambda. */
bool SolveStep(const Evaluator &ev, const Normal &nrm, number_t lambda,
               const BAState &st, BAState *out, number_t *step_inf) {
  const int M = nrm.M;
  MatX S = nrm.Hcc;
  VecX s = nrm.rhs;

  const number_t dmax = nrm.diag_c.size() > 0 ? nrm.diag_c.maxCoeff() : 1;
  const number_t floor_c = std::max<number_t>(dmax, 1) * 1e-12;
  for (int i = 0; i < M; ++i)
    S(i, i) += lambda * std::max(nrm.diag_c(i), floor_c);

  const auto &tracks = ev.tracks();
  std::vector<Mat3> Ainv(nrm.Hff.size(), Mat3::Zero());
  for (int n : tracks) {
    Mat3 A = nrm.Hff[n];
    const number_t fl = std::max<number_t>(nrm.diag_f[n].maxCoeff(), 1e-30) * 1e-12;
    for (int d = 0; d < 3; ++d)
      A(d, d) += lambda * std::max(nrm.diag_f[n](d), fl);
    Eigen::FullPivLU<Mat3> lu(A);
    if (!lu.isInvertible())
      return false;
    Ainv[n] = lu.inverse();

    const auto &fr = ev.frames(n);
    for (size_t a = 0; a < fr.size(); ++a) {
      const Mat3x9 &Ca = nrm.Hfc[n][a];
      const Eigen::Matrix<number_t, 9, 3> CaT_Ainv = Ca.transpose() * Ainv[n];
      for (size_t b = 0; b < fr.size(); ++b)
        S.block<9, 9>(9 * fr[a], 9 * fr[b]).noalias() -=
            CaT_Ainv * nrm.Hfc[n][b];
      s.segment<9>(9 * fr[a]).noalias() -= CaT_Ainv * nrm.rf[n];
    }
  }

  // Gauge: `p_0 = 0` exactly. Zeroing the row and column and putting 1 on the
  // diagonal is equivalent to deleting the three unknowns, without renumbering
  // everything after them -- and unlike a large-weight prior it costs the
  // conditioning of the remaining system nothing.
  for (int i = Ip(0); i < Ip(0) + 3; ++i) {
    S.row(i).setZero();
    S.col(i).setZero();
    S(i, i) = 1;
    s(i) = 0;
  }

  Eigen::LDLT<MatX> ldlt(0.5 * (S + S.transpose()));
  if (ldlt.info() != Eigen::Success)
    return false;
  const VecX dx = ldlt.solve(s);
  if (!dx.allFinite())
    return false;

  *out = st;
  const int K = ev.K();
  for (int k = 0; k < K; ++k) {
    out->R[k] = Orthonormalize(st.R[k] * SO3::exp(dx.segment<3>(Ith(k))).matrix());
    out->p[k] = st.p[k] + dx.segment<3>(Ip(k));
    out->v[k] = st.v[k] + dx.segment<3>(Iv(k));
  }
  out->bg = st.bg + dx.segment<3>(9 * K);
  out->ba = st.ba + dx.segment<3>(9 * K + 3);

  number_t inf = dx.cwiseAbs().maxCoeff();
  for (int n : tracks) {
    const auto &fr = ev.frames(n);
    Vec3 rhs = nrm.rf[n];
    for (size_t a = 0; a < fr.size(); ++a)
      rhs.noalias() -= nrm.Hfc[n][a] * dx.segment<9>(9 * fr[a]);
    const Vec3 df = Ainv[n] * rhs;
    if (!df.allFinite())
      return false;
    out->f[n] = st.f[n] + df;
    inf = std::max(inf, df.cwiseAbs().maxCoeff());
  }
  *step_inf = inf;
  return true;
}

} // namespace

bool SeedBAState(const InitProblem &prob, const LinearInitResult &lin,
                 BAState *seed) {
  if (seed == nullptr || !lin.ok)
    return false;
  const int K = static_cast<int>(prob.frames.size());
  const int N = prob.num_tracks;
  if (K < 2 || N <= 0 || static_cast<int>(lin.features.size()) != N ||
      static_cast<int>(lin.used.size()) != N)
    return false;
  if (lin.g.norm() <= 0)
    return false;

  *seed = BAState{};
  seed->gravity = prob.gravity;
  const Vec3 gW = seed->GravityW();
  // Rotate Stage A's gravity onto world down. This fixes frame 0's roll and
  // pitch -- the two directions that carry the gravity estimate -- and leaves its
  // yaw arbitrary, which is the gauge the solver pins.
  const Mat3 R0 = Eigen::Quaternion<number_t>::FromTwoVectors(
                      lin.g.normalized(), gW.normalized())
                      .toRotationMatrix();

  seed->R.resize(K);
  seed->p.resize(K);
  seed->v.resize(K);
  for (int k = 0; k < K; ++k) {
    const Preintegral &pre = prob.frames[k].pre;
    const number_t dt = pre.dt;
    // Exactly the geometry Stage A's rows were written against, so the seed sits
    // at Stage A's own optimum rather than near it.
    const Vec3 p_I0 = lin.v * dt + 0.5 * lin.g * dt * dt + pre.alpha;
    const Vec3 v_I0 = lin.v + lin.g * dt + pre.beta;
    seed->R[k] = Orthonormalize(R0 * pre.R);
    seed->p[k] = R0 * p_I0;
    seed->v[k] = R0 * v_I0;
  }
  seed->p[0].setZero();
  seed->bg = prob.frames.size() > 1 ? prob.frames[1].pre_prev.bg
                                    : prob.frames[0].pre.bg;
  seed->ba = prob.frames.size() > 1 ? prob.frames[1].pre_prev.ba
                                    : prob.frames[0].pre.ba;
  seed->f.resize(N);
  seed->used = lin.used;
  for (int n = 0; n < N; ++n)
    seed->f[n] = lin.used[n] ? Vec3(R0 * lin.features[n]) : Vec3::Zero();
  return true;
}

number_t InitBACost(const InitProblem &prob, const BAState &state,
                    const BAOptions &opt, const BAState &gauge_ref) {
  Evaluator ev(prob, opt, gauge_ref);
  if (!ev.ok())
    return std::numeric_limits<number_t>::quiet_NaN();
  return ev.Evaluate(state, nullptr, nullptr);
}

bool InitBALinearize(const InitProblem &prob, const BAState &state,
                     const BAOptions &opt, const BAState &gauge_ref, VecX *r,
                     MatX *J, std::vector<int> *track_col) {
  Evaluator ev(prob, opt, gauge_ref);
  if (!ev.ok() || r == nullptr || J == nullptr)
    return false;
  DenseLin dn;
  ev.Evaluate(state, nullptr, nullptr, &dn);
  *r = dn.r;
  *J = dn.J;
  if (track_col != nullptr)
    *track_col = ev.dense_cols();
  return true;
}

BAResult SolveInitBA(const InitProblem &prob, const BAState &seed,
                     const BAOptions &opt) {
  BAResult res;
  res.state = seed;
  Evaluator ev(prob, opt, seed);
  if (!ev.ok()) {
    res.why = "window too small, or seed does not match the problem";
    return res;
  }
  res.tracks_used = static_cast<int>(ev.tracks().size());

  BAState st = seed;
  Normal nrm;
  Stats s0;
  number_t cost = ev.Evaluate(st, &nrm, &s0);
  res.cost_init = cost;
  if (!std::isfinite(cost)) {
    res.why = "seed cost is not finite";
    return res;
  }

  number_t lambda = opt.lambda_init;
  bool converged = false;
  const char *why = "iteration budget";
  for (int iter = 0; iter < opt.max_iterations && !converged; ++iter) {
    bool accepted = false;
    for (int rej = 0; rej < opt.max_rejections; ++rej) {
      BAState cand;
      number_t step = 0;
      if (SolveStep(ev, nrm, lambda, st, &cand, &step)) {
        const number_t c = ev.Evaluate(cand, nullptr, nullptr);
        if (std::isfinite(c) && c < cost) {
          const number_t rel = (cost - c) / std::max<number_t>(cost, 1e-300);
          st = cand;
          cost = c;
          ++res.iterations;
          accepted = true;
          lambda = std::max<number_t>(lambda * 0.1, 1e-14);
          if (rel < opt.cost_tol) {
            converged = true;
            why = "cost converged";
          } else if (step < opt.step_tol) {
            converged = true;
            why = "step converged";
          }
          break;
        }
      }
      ++res.rejections;
      lambda *= 10;
      if (lambda > opt.lambda_max)
        break;
    }
    if (!accepted) {
      // Not a failure: LM stops raising lambda when no descent direction is left,
      // which at a minimum is the correct answer.
      why = lambda > opt.lambda_max ? "no further descent" : "step rejected";
      break;
    }
    if (!converged)
      ev.Evaluate(st, &nrm, &s0);
  }

  Stats fin;
  res.cost_final = ev.Evaluate(st, nullptr, &fin);
  res.state = st;
  res.obs_used = fin.pix_obs;
  res.pixel_rms = fin.pix_obs > 0 ? std::sqrt(fin.pix_sq_px / fin.pix_obs) : 0;
  if (!fin.pix_err_px.empty()) {
    auto &e = fin.pix_err_px;
    const size_t mid = e.size() / 2;
    std::nth_element(e.begin(), e.begin() + mid, e.end());
    res.pixel_median = e[mid];
  }
  res.imu_rms = fin.imu_rows > 0 ? std::sqrt(fin.imu_sq / fin.imu_rows) : 0;
  res.ok = std::isfinite(res.cost_final) && res.cost_final <= res.cost_init;
  res.why = res.ok ? why : "cost did not decrease";
  return res;
}

BAResult SolveInitBA(const InitProblem &prob, const BAState &seed) {
  return SolveInitBA(prob, seed, BAOptions{});
}

} // namespace xivo
