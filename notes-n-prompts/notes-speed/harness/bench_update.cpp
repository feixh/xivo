// Times the pieces of xivo::EkfUpdateDowndate at the shipped capacity, and a few
// alternative formulations of the triangular solve, which the sampling profile
// says is 14% of the whole stereo run.
//
//   g++ -O3 -march=native -std=c++17 -I<eigen> bench_update.cpp -o bench_update
#include <chrono>
#include <cstdio>
#include <vector>

#include "Eigen/Dense"

using number_t = double;
using MatX = Eigen::Matrix<number_t, -1, -1>;
using VecX = Eigen::Matrix<number_t, -1, 1>;
using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char **argv) {
  const int N = 564;                             // kFullSize
  const int rows = argc > 1 ? atoi(argv[1]) : 282; // stereo measurement rows
  const int live0 = 90, live1 = 240;              // the two occupied runs
  const int reps = argc > 2 ? atoi(argv[2]) : 300;

  MatX P = MatX::Random(N, N);
  P = (P * P.transpose()).eval() + N * MatX::Identity(N, N);
  MatX Mfull = MatX::Random(rows, N);
  MatX S = MatX::Random(rows, rows);
  S = (S * S.transpose()).eval() + rows * MatX::Identity(rows, rows);

  struct Run { int start, len; };
  std::vector<Run> runs{{0, live0}, {324, live1}};

  double t_llt = 0, t_solve_run = 0, t_solve_all = 0, t_solve_right = 0,
         t_rank = 0, t_mirror = 0;
  double sink = 0;

  for (int r = 0; r < reps; ++r) {
    // --- LLT
    auto a = clk::now();
    Eigen::LLT<MatX> llt(S);
    auto b = clk::now();
    t_llt += ms(a, b);

    // --- (1) what the code does now: one solveInPlace per occupied run
    MatX M = Mfull;
    a = clk::now();
    for (auto &q : runs) llt.matrixL().solveInPlace(M.middleCols(q.start, q.len));
    b = clk::now();
    t_solve_run += ms(a, b);
    sink += M(0, 0);

    // --- (2) one solveInPlace over the whole width
    MatX M2 = Mfull;
    a = clk::now();
    llt.matrixL().solveInPlace(M2);
    b = clk::now();
    t_solve_all += ms(a, b);
    sink += M2(0, 0);

    // --- (3) transposed layout: W' = M' L^-T, a right-side solve on an
    //         (N x rows) matrix, which is the orientation the downdate wants
    //         anyway.
    MatX Mt = Mfull.transpose();
    a = clk::now();
    for (auto &q : runs)
      llt.matrixL().transpose().solveInPlace<Eigen::OnTheRight>(
          Mt.middleRows(q.start, q.len));
    b = clk::now();
    t_solve_right += ms(a, b);
    sink += Mt(0, 0);

    // --- rank update on the occupied runs (as the code does)
    MatX Pc = P;
    a = clk::now();
    for (size_t i = 0; i < runs.size(); ++i) {
      auto Wi = M.middleCols(runs[i].start, runs[i].len);
      Pc.block(runs[i].start, runs[i].start, runs[i].len, runs[i].len)
          .selfadjointView<Eigen::Lower>()
          .rankUpdate(Wi.transpose(), -1.0);
      for (size_t j = 0; j < i; ++j) {
        Pc.block(runs[i].start, runs[j].start, runs[i].len, runs[j].len)
            .noalias() -= Wi.transpose() * M.middleCols(runs[j].start, runs[j].len);
      }
    }
    b = clk::now();
    t_rank += ms(a, b);
    sink += Pc(0, 0);

    a = clk::now();
    for (size_t i = 0; i < runs.size(); ++i) {
      auto D = Pc.block(runs[i].start, runs[i].start, runs[i].len, runs[i].len);
      for (int j = 1; j < runs[i].len; ++j)
        D.block(0, j, j, 1) = D.block(j, 0, 1, j).transpose();
      for (size_t j = 0; j < i; ++j)
        Pc.block(runs[j].start, runs[i].start, runs[j].len, runs[i].len) =
            Pc.block(runs[i].start, runs[j].start, runs[i].len, runs[j].len)
                .transpose();
    }
    b = clk::now();
    t_mirror += ms(a, b);
    sink += Pc(1, 0);
  }
  printf("rows=%d reps=%d  (ms per update)\n", rows, reps);
  printf("  LLT(S)                       %8.4f\n", t_llt / reps);
  printf("  solve, per-run (current)     %8.4f\n", t_solve_run / reps);
  printf("  solve, whole width           %8.4f\n", t_solve_all / reps);
  printf("  solve, transposed/OnTheRight %8.4f\n", t_solve_right / reps);
  printf("  rank update + gemm           %8.4f\n", t_rank / reps);
  printf("  mirror                       %8.4f\n", t_mirror / reps);
  printf("  (sink %g)\n", sink);
  return 0;
}
