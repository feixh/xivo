// Tests for the matrix-product derivative helpers in common/rodrigues.h.
//
// Every derivative in that header flattens a matrix ROW-MAJOR: `dhat` documents
// its rows as (u00, u01, u02, u10, ...), `rodrigues`' dR_dw says "reshape into
// 9x1 column, row-major way", `dAt_dA` uses D(m*N+n, n*M+m), and the hand-built
// dWsb_dCg in Estimator::ComputeMotionJacobianAt puts row i of Cg at columns
// 3i..3i+2. `dAB_dA` and `dAB_dB` indexed their *output* rows as p*N+n, which is
// COLUMN-major, while indexing their input columns row-major. Each function was
// therefore self-inconsistent, and -- worse -- the two are multiplied together in
// ComputeMotionJacobianAt, so the mismatch silently transposed Rsb*Ca.
//
// The affected chain sits behind USE_ONLINE_IMU_CALIB, which the shipped build
// leaves undefined (src/CMakeLists.txt), so this is latent rather than live. The
// tests below pin the convention so it stays fixed.
#include <gtest/gtest.h>

#include "rodrigues.h"

#include "Eigen/Dense"

using namespace Eigen;
using namespace xivo;

namespace {

using num = double;

// Row-major flattening of an MxN matrix, matching the convention above.
template <int M, int N>
Matrix<num, M * N, 1> vec_rm(const Matrix<num, M, N> &A) {
  Matrix<num, M * N, 1> v;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      v(i * N + j) = A(i, j);
  return v;
}

Matrix<num, 3, 3> SomeA() {
  Matrix<num, 3, 3> A;
  A << 0.3, -1.7, 2.1, 0.9, 0.4, -0.6, -2.2, 1.1, 0.8;
  return A;
}

Matrix<num, 3, 3> SomeB() {
  Matrix<num, 3, 3> B;
  B << -0.5, 1.3, 0.2, 2.4, -0.9, 1.6, 0.7, 0.1, -1.4;
  return B;
}

} // namespace

// dAB_dA maps a row-major perturbation of A to a row-major perturbation of A*B.
TEST(Rodrigues, dABdAUsesRowMajorFlatteningOnBothSides) {
  const Matrix<num, 3, 3> A = SomeA();
  const Matrix<num, 3, 3> B = SomeB();
  const Matrix<num, 9, 9> D = dAB_dA<3, 3>(B);

  const num eps = 1e-6;
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      Matrix<num, 3, 3> dA = Matrix<num, 3, 3>::Zero();
      dA(n, m) = eps;
      const Matrix<num, 9, 1> num_col =
          (vec_rm<3, 3>((A + dA) * B) - vec_rm<3, 3>((A - dA) * B)) / (2 * eps);
      // Column n*M+m of D is the row-major slot of A(n, m).
      const Matrix<num, 9, 1> ana_col = D.col(n * 3 + m);
      for (int k = 0; k < 9; ++k) {
        EXPECT_NEAR(ana_col(k), num_col(k), 1e-8)
            << "dA(" << n << "," << m << ") row " << k;
      }
    }
  }
}

// Same for dAB_dB. This is the function whose row index was column-major.
TEST(Rodrigues, dABdBUsesRowMajorFlatteningOnBothSides) {
  const Matrix<num, 3, 3> A = SomeA();
  const Matrix<num, 3, 3> B = SomeB();
  const Matrix<num, 9, 9> D = dAB_dB<3, 3>(A);

  const num eps = 1e-6;
  for (int m = 0; m < 3; ++m) {
    for (int p = 0; p < 3; ++p) {
      Matrix<num, 3, 3> dB = Matrix<num, 3, 3>::Zero();
      dB(m, p) = eps;
      const Matrix<num, 9, 1> num_col =
          (vec_rm<3, 3>(A * (B + dB)) - vec_rm<3, 3>(A * (B - dB))) / (2 * eps);
      const Matrix<num, 9, 1> ana_col = D.col(m * 3 + p);
      for (int k = 0; k < 9; ++k) {
        EXPECT_NEAR(ana_col(k), num_col(k), 1e-8)
            << "dB(" << m << "," << p << ") row " << k;
      }
    }
  }
}

// The vector case: P == 1 makes the two row conventions coincide, so this one
// was always right. Keep it as a guard against the fix breaking it.
TEST(Rodrigues, dABdAAgainstAVectorOperand) {
  const Matrix<num, 3, 3> A = SomeA();
  const Matrix<num, 3, 1> b(0.7, -1.9, 2.3);
  const Matrix<num, 3, 9> D = dAB_dA<3, 3>(b);

  const num eps = 1e-6;
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      Matrix<num, 3, 3> dA = Matrix<num, 3, 3>::Zero();
      dA(n, m) = eps;
      const Matrix<num, 3, 1> num_col = ((A + dA) * b - (A - dA) * b) / (2 * eps);
      for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(D(k, n * 3 + m), num_col(k), 1e-8);
      }
    }
  }
}

// dAt_dA already used the row-major convention; assert it so the three agree.
TEST(Rodrigues, dAtdAUsesRowMajorFlattening) {
  const Matrix<num, 3, 3> A = SomeA();
  const Matrix<num, 9, 9> D = dAt_dA(A);

  const num eps = 1e-6;
  for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 3; ++m) {
      Matrix<num, 3, 3> dA = Matrix<num, 3, 3>::Zero();
      dA(n, m) = eps;
      const Matrix<num, 9, 1> num_col =
          (vec_rm<3, 3>((A + dA).transpose()) -
           vec_rm<3, 3>((A - dA).transpose())) /
          (2 * eps);
      for (int k = 0; k < 9; ++k) {
        EXPECT_NEAR(D(k, n * 3 + m), num_col(k), 1e-8);
      }
    }
  }
}

// dA_dAu wrote only its 6 upper-triangular rows and never called setZero(); the
// other 3 came out zero only because the build passes
// -DEIGEN_INITIALIZE_MATRICES_BY_ZERO. Poison-check by comparing the two
// overloads and asserting the exact expected pattern.
TEST(Rodrigues, dAdAuIsFullyInitialisedAndUpperTriangularOnly) {
  const Matrix<num, 9, 6> D = dA_dAu<num, 3>();
  const Matrix<num, 9, 6> D2 = dA_dAu<Matrix<num, 3, 3>, 3>(SomeA());
  EXPECT_TRUE(D.isApprox(D2));

  // Parameter order matches IMUState::operator+= : (0,0) (0,1) (0,2) (1,1)
  // (1,2) (2,2). Rows are the row-major slots of the 3x3 matrix.
  Matrix<num, 9, 6> expect = Matrix<num, 9, 6>::Zero();
  int idx = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j)
      expect(i * 3 + j, idx++) = 1;
  EXPECT_EQ(idx, 6);
  EXPECT_TRUE(D.isApprox(expect)) << D;

  // Ca is upper triangular (IMU::IMU CHECKs Ca(1,0)==Ca(2,0)==Ca(2,1)==0) and
  // operator+= only ever touches j >= i, so the strictly-lower rows are
  // structurally zero -- not a missing term.
  for (int i : {3, 6, 7}) { // row-major slots of (1,0), (2,0), (2,1)
    EXPECT_EQ(D.row(i).cwiseAbs().sum(), 0.0) << "row " << i;
  }
}

// The chain from Estimator::ComputeMotionJacobianAt: V += Rsb * Ca * accel * dt,
// differentiated w.r.t. the 6 free parameters of the upper-triangular Ca. This
// is the composition that the convention mismatch broke -- each factor was
// individually "plausible" but the product transposed Rsb*Ca.
TEST(Rodrigues, AccelCalibrationChainMatchesNumericalDerivative) {
  Matrix<num, 3, 3> Rsb;
  // A generic (non-symmetric) rotation; with a symmetric Rsb*Ca the bug hides.
  const num a = 0.4, b = -0.7, c = 1.1;
  Matrix<num, 3, 3> Rz, Ry, Rx;
  Rz << cos(a), -sin(a), 0, sin(a), cos(a), 0, 0, 0, 1;
  Ry << cos(b), 0, sin(b), 0, 1, 0, -sin(b), 0, cos(b);
  Rx << 1, 0, 0, 0, cos(c), -sin(c), 0, sin(c), cos(c);
  Rsb = Rz * Ry * Rx;

  const Matrix<num, 3, 1> accel(0.31, -9.6, 1.4);

  Matrix<num, 3, 3> Ca;
  Ca << 1.02, 0.013, -0.007, 0, 0.98, 0.021, 0, 0, 1.01;

  const Matrix<num, 3, 9> dV_dRCa = dAB_dA<3, 3>(accel);
  const Matrix<num, 9, 9> dRCa_dCafm = dAB_dB<3, 3>(Rsb);
  const Matrix<num, 9, 6> dCafm_dCa = dA_dAu<num, 3>();
  const Matrix<num, 3, 6> dV_dCa = dV_dRCa * dRCa_dCafm * dCafm_dCa;

  const num eps = 1e-7;
  for (int k = 0; k < 6; ++k) {
    // Perturb the k-th upper-triangular parameter, in operator+= order.
    Matrix<num, 3, 3> dCa = Matrix<num, 3, 3>::Zero();
    int idx = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j, ++idx)
        if (idx == k)
          dCa(i, j) = eps;

    const Matrix<num, 3, 1> num_col =
        (Rsb * (Ca + dCa) * accel - Rsb * (Ca - dCa) * accel) / (2 * eps);
    for (int r = 0; r < 3; ++r) {
      EXPECT_NEAR(dV_dCa(r, k), num_col(r), 1e-6) << "param " << k << " row " << r;
    }
  }
}
