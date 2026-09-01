// Times the *shipped* kernel API -- xivo-bf16/common/bf16_gemm.h -- at the
// filter's real shapes, so the exit criterion for M3 ("per-kernel timings match
// the microbenchmark") is checked against the header the estimator actually
// calls rather than against the prototype in bf16_gemm.h next door.
//
// Three shapes, at n = 23 + 6*45 + 3*90 = 563:
//   Joseph      (I-KH) P (I-KH)^T : n x n x n, twice
//   Innovation  H P H^T           : m x n x n then m x m x n, m = 180
//   Gating      J_i P J_i^T       : 90 x (2 x n x n), the sweep MHGating runs
//
// and for the gating sweep, three ways of arranging it, because the arrangement
// turns out to matter as much as the precision:
//   per-feature Mul   -- one gemm per feature, P packed 90 times
//   packed rhs        -- P packed once, 90 small gemms against it
//   batched           -- all 180 Jacobian rows in one gemm
//
// build:
//   g++ -O3 -march=native -std=c++17 -funroll-loops \
//     -I../../../xivo-bf16/common -I../../../xivo-bf16/thirdparty/eigen \
//     kernel_api.cpp -o kernel_api
// run: taskset -c <idle core> ./kernel_api
//
// Author: Xiaohan Fei <hzhsfxh@gmail.com>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "bf16_gemm.h"

using namespace xivo;
using bfgemm::Prec;
using MatD = Eigen::Matrix<double, -1, -1>;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kN = 564;    // 24 + 6*45 + 3*90, the fixed capacity
constexpr int kMeas = 180;  // what the filter actually reaches on mono room1
constexpr int kFeat = 90;
constexpr int kReps = 5;

MatD Random(int r, int c, uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  MatD m(r, c);
  for (int j = 0; j < c; ++j)
    for (int i = 0; i < r; ++i) m(i, j) = nd(gen);
  return m;
}

MatD Covariance(int n, uint32_t seed) {
  MatD L = Random(n, n, seed);
  MatD P = (L * L.transpose()) / n;
  for (int i = 0; i < n; ++i) {
    const double s = std::pow(10.0, -3.0 + 6.0 * i / (n - 1));
    P.row(i) *= s;
    P.col(i) *= s;
  }
  return 0.5 * (P + P.transpose());
}

double RelErr(const MatD &a, const MatD &b) {
  return (a - b).norm() / b.norm();
}

/// Median of kReps timings, in ms. Median rather than min: the point is what a
/// frame costs, not what the best cache state allows.
template <typename F> double TimeMs(F &&f) {
  std::vector<double> ms;
  for (int r = 0; r < kReps; ++r) {
    const auto t0 = Clock::now();
    f();
    const auto t1 = Clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

struct Row {
  const char *arrangement;
  double ms[3];   // f64, f32, bf16
  double err[3];
};

void Print(const char *title, const std::vector<Row> &rows) {
  std::printf("\n%s\n", title);
  std::printf("  %-22s %10s %10s %10s   %9s %9s\n", "arrangement", "f64 ms",
              "f32 ms", "bf16 ms", "f32 err", "bf16 err");
  for (const Row &r : rows) {
    std::printf("  %-22s %10.2f %10.2f %10.2f   %9.1e %9.1e",
                r.arrangement, r.ms[0], r.ms[1], r.ms[2], r.err[1], r.err[2]);
    std::printf("   (f32 %.2fx, bf16 %.2fx)\n", r.ms[0] / r.ms[1],
                r.ms[0] / r.ms[2]);
  }
}

} // namespace

int main() {
  std::printf("n=%d  meas rows=%d  features=%d  reps=%d  bf16 hw=%s\n", kN,
              kMeas, kFeat, kReps, bfgemm::have_bf16() ? "yes" : "NO");

  const MatD P = Covariance(kN, 11);
  const Prec precs[3] = {Prec::kF64, Prec::kF32, Prec::kBF16};

  // ---- Joseph: (I-KH) P (I-KH)^T ----------------------------------------
  {
    const MatD A = Random(kN, kN, 12);
    MatD tmp(kN, kN), out(kN, kN), ref(kN, kN);
    bfgemm::Scratch sc;
    std::vector<Row> rows;
    Row plain{"two Mul", {}, {}}, packed{"packed rhs (P once)", {}, {}};
    for (int i = 0; i < 3; ++i) {
      plain.ms[i] = TimeMs([&] {
        bfgemm::Mul<double>(precs[i], A, false, P, false, tmp, false, sc);
        bfgemm::Mul<double>(precs[i], tmp, false, A, true, out, false, sc);
      });
      if (i == 0) ref = out;
      plain.err[i] = RelErr(out, ref);

      // Only the first product's rhs is P; the second's is A^T, so this
      // arrangement saves one of two packings. Included to show that it is not
      // where the gain is for this shape -- the shape is compute bound.
      packed.ms[i] = TimeMs([&] {
        if (bfgemm::effective(precs[i]) == Prec::kBF16) {
          bfgemm::PackRhs<double>(P, false, sc);
          bfgemm::MulRhs<double>(A, false, tmp, false, sc);
        } else {
          bfgemm::Mul<double>(precs[i], A, false, P, false, tmp, false, sc);
        }
        bfgemm::Mul<double>(precs[i], tmp, false, A, true, out, false, sc);
      });
      packed.err[i] = RelErr(out, ref);
    }
    rows.push_back(plain);
    rows.push_back(packed);
    Print("Joseph  (I-KH) P (I-KH)^T, n x n x n twice", rows);
  }

  // ---- Innovation: H P H^T ----------------------------------------------
  {
    const MatD H = Random(kMeas, kN, 13);
    MatD HP(kMeas, kN), S(kMeas, kMeas), ref(kMeas, kMeas);
    bfgemm::Scratch sc;
    Row row{"two Mul", {}, {}};
    for (int i = 0; i < 3; ++i) {
      row.ms[i] = TimeMs([&] {
        bfgemm::Mul<double>(precs[i], H, false, P, false, HP, false, sc);
        bfgemm::Mul<double>(precs[i], HP, false, H, true, S, false, sc);
      });
      if (i == 0) ref = S;
      row.err[i] = RelErr(S, ref);
    }
    Print("Innovation  H P H^T, m = 180", {row});
  }

  // ---- Gating sweep: 90 x J_i P J_i^T -----------------------------------
  {
    std::vector<MatD> J;
    for (int f = 0; f < kFeat; ++f) J.push_back(Random(2, kN, 300 + f));
    MatD Jall(2 * kFeat, kN);
    for (int f = 0; f < kFeat; ++f) Jall.middleRows(2 * f, 2) = J[f];

    MatD JP(2, kN), JallP(2 * kFeat, kN);
    std::vector<MatD> S(kFeat, MatD(2, 2)), ref(kFeat, MatD(2, 2));
    bfgemm::Scratch sc;

    Row per{"per-feature Mul", {}, {}};
    Row pk{"packed rhs (P once)", {}, {}};
    Row bt{"batched, one gemm", {}, {}};
    for (int i = 0; i < 3; ++i) {
      per.ms[i] = TimeMs([&] {
        for (int f = 0; f < kFeat; ++f) {
          bfgemm::Mul<double>(precs[i], J[f], false, P, false, JP, false, sc);
          bfgemm::Mul<double>(precs[i], JP, false, J[f], true, S[f], false, sc);
        }
      });
      if (i == 0) ref = S;
      double e = 0;
      for (int f = 0; f < kFeat; ++f) e += RelErr(S[f], ref[f]);
      per.err[i] = e / kFeat;

      pk.ms[i] = TimeMs([&] {
        const bool packed =
            bfgemm::effective(precs[i]) == Prec::kBF16 &&
            bfgemm::PackRhs<double>(P, false, sc);
        for (int f = 0; f < kFeat; ++f) {
          if (packed) bfgemm::MulRhs<double>(J[f], false, JP, false, sc);
          else bfgemm::Mul<double>(precs[i], J[f], false, P, false, JP, false, sc);
          bfgemm::Mul<double>(precs[i], JP, false, J[f], true, S[f], false, sc);
        }
      });
      double e2 = 0;
      for (int f = 0; f < kFeat; ++f) e2 += RelErr(S[f], ref[f]);
      pk.err[i] = e2 / kFeat;

      bt.ms[i] = TimeMs([&] {
        bfgemm::Mul<double>(precs[i], Jall, false, P, false, JallP, false, sc);
        for (int f = 0; f < kFeat; ++f) {
          bfgemm::Mul<double>(precs[i], MatD(JallP.middleRows(2 * f, 2)), false,
                              J[f], true, S[f], false, sc);
        }
      });
      double e3 = 0;
      for (int f = 0; f < kFeat; ++f) e3 += RelErr(S[f], ref[f]);
      bt.err[i] = e3 / kFeat;
    }
    Print("Gating  90 x J_i P J_i^T, 2 rows each", {per, pk, bt});
  }
  return 0;
}
