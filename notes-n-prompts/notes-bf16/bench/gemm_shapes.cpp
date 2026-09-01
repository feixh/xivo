// The three covariance kernels of XIVO's EKF, at the shapes the shipped config
// actually runs (kFullSize = 23 + 6*45 + 3*90 = 563), in fp64 / fp32 / bf16.
//
//   K1  Joseph-form covariance update   P <- A P A^T          (2 * n^3)
//   K2  innovation covariance           S = H P H^T           (m n^2 + m^2 n)
//   K3  the MH-gating sweep             S_i = J_i P J_i^T     (nfeat * 2 n^2)
//
// K1/K2 are compute bound; K3 streams the whole of P once per feature and is
// bandwidth bound, which is where halving the element width pays twice over.
//
// build: g++ -O3 -march=native -std=c++17 -I<eigen> gemm_shapes.cpp -o gemm_shapes
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "bf16_gemm_proto.h"

using Eigen::Dynamic;
using MatXd = Eigen::Matrix<double, Dynamic, Dynamic>;
using MatXf = Eigen::Matrix<float, Dynamic, Dynamic>;

static double secs(std::chrono::steady_clock::time_point a,
                   std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// Relative error of X against the fp64 reference, in the Frobenius norm.
static double relerr(const MatXd &ref, const MatXd &x) {
  return (x - ref).norm() / ref.norm();
}

int main(int argc, char **argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 563;  // state size
  const int m = argc > 2 ? atoi(argv[2]) : 96;   // measurement rows
  const int nfeat = argc > 3 ? atoi(argv[3]) : 90;
  const int reps = argc > 4 ? atoi(argv[4]) : 20;

  std::mt19937 rng(7);
  std::normal_distribution<double> g(0, 1);

  // A covariance-like P: symmetric positive definite with a wide dynamic range
  // across blocks, as the real one has (metres, radians, and inverse depths all
  // sit in the same matrix).
  MatXd L(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) L(i, j) = g(rng);
  Eigen::VectorXd scale(n);
  for (int i = 0; i < n; ++i) scale(i) = std::pow(10.0, -3.0 + 4.0 * (i % 7) / 6.0);
  L = scale.asDiagonal() * L;
  MatXd P = L * L.transpose() / n;
  MatXd A(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) A(i, j) = (i == j ? 1.0 : 0.0) + 0.01 * g(rng);
  MatXd H(m, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) H(i, j) = g(rng);

  MatXf Pf = P.cast<float>(), Af = A.cast<float>(), Hf = H.cast<float>();
  xivo::bf16::Workspace ws;

  printf("n=%d m=%d nfeat=%d reps=%d\n\n", n, m, nfeat, reps);
  printf("%-34s %10s %10s %10s\n", "kernel", "ms", "GFLOP/s", "rel.err");

  auto report = [&](const char *name, double ms, double flops, double err) {
    printf("%-34s %10.3f %10.1f %10.2e\n", name, ms, flops / (ms * 1e6), err);
  };

  // ---- K1: P <- A P A^T -------------------------------------------------
  const double f1 = 2.0 * 2.0 * n * n * (double)n;
  MatXd ref1;
  {
    auto t0 = std::chrono::steady_clock::now();
    MatXd tmp(n, n), out(n, n);
    for (int r = 0; r < reps; ++r) {
      tmp.noalias() = A * P;
      out.noalias() = tmp * A.transpose();
    }
    auto t1 = std::chrono::steady_clock::now();
    ref1 = out;
    report("K1 A*P*A^T   eigen fp64", secs(t0, t1) / reps * 1e3, f1, 0.0);
  }
  {
    auto t0 = std::chrono::steady_clock::now();
    MatXf tmp(n, n), out(n, n);
    for (int r = 0; r < reps; ++r) {
      tmp.noalias() = Af * Pf;
      out.noalias() = tmp * Af.transpose();
    }
    auto t1 = std::chrono::steady_clock::now();
    report("K1 A*P*A^T   eigen fp32", secs(t0, t1) / reps * 1e3, f1,
           relerr(ref1, out.cast<double>()));
  }
#if XIVO_HAS_BF16_GEMM
  {
    auto t0 = std::chrono::steady_clock::now();
    MatXf tmp(n, n), out(n, n);
    for (int r = 0; r < reps; ++r) {
      // tmp = A*P ; out = tmp * A^T = (A * tmp^T)^T -- but tmp is not symmetric,
      // so pass A^T as the 'B' operand by transposing through the packer:
      // out = tmp * A^T is a gemm with B = A^T, i.e. B(k,j) = A(j,k). Feed A as
      // a_trans on the left of the transposed product instead: out^T = A * tmp^T.
      xivo::bf16::gemm(n, n, n, Af.data(), n, false, Pf.data(), n, tmp.data(), n,
                       0.f, ws);
      // out^T = A * tmp^T : A(nxn) * tmp^T(nxn); tmp^T is tmp packed as a_trans
      // on the B side, which pack_b cannot do -- so form the product as
      // out = tmp * A^T by packing A^T explicitly (cheap: n^2).
      MatXf At = Af.transpose();
      xivo::bf16::gemm(n, n, n, tmp.data(), n, false, At.data(), n, out.data(), n,
                       0.f, ws);
    }
    auto t1 = std::chrono::steady_clock::now();
    report("K1 A*P*A^T   bf16 (fp32 acc)", secs(t0, t1) / reps * 1e3, f1,
           relerr(ref1, out.cast<double>()));
  }
#endif

  // ---- K2: S = H P H^T --------------------------------------------------
  const double f2 = 2.0 * m * n * (double)n + 2.0 * m * (double)m * n;
  MatXd ref2;
  {
    auto t0 = std::chrono::steady_clock::now();
    MatXd HP(m, n), S(m, m);
    for (int r = 0; r < reps; ++r) {
      HP.noalias() = H * P;
      S.noalias() = HP * H.transpose();
    }
    auto t1 = std::chrono::steady_clock::now();
    ref2 = S;
    report("K2 H*P*H^T   eigen fp64", secs(t0, t1) / reps * 1e3, f2, 0.0);
  }
  {
    auto t0 = std::chrono::steady_clock::now();
    MatXf HP(m, n), S(m, m);
    for (int r = 0; r < reps; ++r) {
      HP.noalias() = Hf * Pf;
      S.noalias() = HP * Hf.transpose();
    }
    auto t1 = std::chrono::steady_clock::now();
    report("K2 H*P*H^T   eigen fp32", secs(t0, t1) / reps * 1e3, f2,
           relerr(ref2, S.cast<double>()));
  }
#if XIVO_HAS_BF16_GEMM
  {
    auto t0 = std::chrono::steady_clock::now();
    MatXf HP(m, n), S(m, m), Ht = Hf.transpose();
    for (int r = 0; r < reps; ++r) {
      xivo::bf16::gemm(m, n, n, Hf.data(), m, false, Pf.data(), n, HP.data(), m,
                       0.f, ws);
      xivo::bf16::gemm(m, m, n, HP.data(), m, false, Ht.data(), n, S.data(), m,
                       0.f, ws);
    }
    auto t1 = std::chrono::steady_clock::now();
    report("K2 H*P*H^T   bf16 (fp32 acc)", secs(t0, t1) / reps * 1e3, f2,
           relerr(ref2, S.cast<double>()));
  }
#endif

  // ---- K3: nfeat x (2xn) P (nx2) ---------------------------------------
  // Every in-state feature is gated separately, so P is streamed nfeat times per
  // frame. The Jacobian rows are dense here, as they are in the code.
  const double f3 = nfeat * (2.0 * 2 * n * (double)n + 2.0 * 4 * (double)n);
  std::vector<MatXd> J(nfeat);
  std::vector<MatXf> Jf(nfeat);
  for (int i = 0; i < nfeat; ++i) {
    J[i].resize(2, n);
    for (int c = 0; c < n; ++c) { J[i](0, c) = g(rng); J[i](1, c) = g(rng); }
    Jf[i] = J[i].cast<float>();
  }
  double s64 = 0, s32 = 0, sbf = 0;
  {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
      for (int i = 0; i < nfeat; ++i) {
        Eigen::Matrix<double, 2, 2> S = J[i] * P * J[i].transpose();
        s64 += S(0, 0) + S(1, 1);
      }
    auto t1 = std::chrono::steady_clock::now();
    report("K3 nfeat x J*P*J^T  eigen fp64", secs(t0, t1) / reps * 1e3, f3, 0.0);
  }
  {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
      for (int i = 0; i < nfeat; ++i) {
        Eigen::Matrix<float, 2, 2> S = Jf[i] * Pf * Jf[i].transpose();
        s32 += S(0, 0) + S(1, 1);
      }
    auto t1 = std::chrono::steady_clock::now();
    report("K3 nfeat x J*P*J^T  eigen fp32", secs(t0, t1) / reps * 1e3, f3,
           std::abs(s32 - s64) / std::abs(s64));
  }
#if XIVO_HAS_BF16_GEMM
  {
    // P is packed once per frame and reused by every feature -- that is the
    // whole point: the bf16 copy is half the bytes to stream.
    const int nt = xivo::bf16::round_up(n, xivo::bf16::NR) / xivo::bf16::NR;
    const int kt = xivo::bf16::round_up(n, 2) / 2;
    std::vector<uint32_t> pb((size_t)nt * kt * xivo::bf16::NR);
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
      xivo::bf16::pack_b(n, n, Pf.data(), n, pb.data());
      for (int i = 0; i < nfeat; ++i) {
        Eigen::Matrix<float, 2, Dynamic> JP(2, n);
        std::vector<uint32_t> pa((size_t)kt * xivo::bf16::MR);
        xivo::bf16::pack_a(2, n, Jf[i].data(), 2, pa.data());
        xivo::bf16::gemm_packed(2, n, n, pa.data(), pb.data(), JP.data(), 2, 0.f);
        Eigen::Matrix<float, 2, 2> S = JP * Jf[i].transpose();
        sbf += S(0, 0) + S(1, 1);
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    report("K3 nfeat x J*P*J^T  bf16", secs(t0, t1) / reps * 1e3, f3,
           std::abs(sbf - s64) / std::abs(s64));
  }
#endif
  return 0;
}
