// Does Eigen 3.3 have a bf16 fast path on x86? Times the Joseph-form product
// A * P * A^T at the branch's fixed state size, with Eigen matrices of each
// candidate scalar type -- i.e. exactly what -DXIVO_NUMBER_T= gives the filter.
//
// Author: Xiaohan Fei <hzhsfxh@gmail.com>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "bf16.h"

using xivo::bf16;

template <class T> double time_one(int n, int reps) {
  using M = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  M A(n, n), P(n, n), C(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      A(i, j) = T(u(rng));
      P(i, j) = T(u(rng));
    }
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    C.noalias() = A * P;
    C = C * A.transpose();
    A(0, 0) = C(0, 0); // keep it live
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int main() {
  const int n = 563;
  const int reps = 5;
  const double d = time_one<double>(n, reps);
  const double f = time_one<float>(n, reps);
  const double b = time_one<bf16>(n, reps);
  std::printf("n=%d  A*P*A^T, Eigen matrices of each scalar type\n", n);
  std::printf("  double %8.2f ms   1.00x\n", d);
  std::printf("  float  %8.2f ms  %5.2fx\n", f, d / f);
  std::printf("  bf16   %8.2f ms  %5.2fx\n", b, d / b);
  return 0;
}
