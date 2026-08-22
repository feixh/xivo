#include "helpers.cpp"
#include "unittest_helpers.h"
#include "Eigen/SVD"
#include "gtest/gtest.h"

using namespace Eigen;
using namespace xivo;


// These tests come from here:
// https://www.math.usm.edu/lambers/mat610/sum10/lecture9.pdf 
// Can double-check using 'planerot' function in MATLAB.
// Note that we use the convention in Golub & Loan
// tests are accurate up to a sign.
TEST(NumericalLinearAlgebra, GivensSub) {
    double tol = 5e-4;

    Vec2 v1(0.9134, 0.6324);
    Mat2 G1 = givens(v1(0), v1(1));
    Vec2 y1 = G1.transpose() * v1;
    EXPECT_NEAR(abs(y1(0)), 1.1110, tol);
    EXPECT_NEAR(y1(1), 0.0, tol);
    EXPECT_NEAR(abs(G1(0,0)), 0.8222, tol);
    EXPECT_NEAR(abs(G1(0,1)), 0.5692, tol);
    EXPECT_NEAR(abs(G1(1,0)), 0.5692, tol);
    EXPECT_NEAR(abs(G1(1,1)), 0.8222, tol);

    Vec2 v2(0.1270, 1.1109);
    Mat2 G2 = givens(v2(0), v2(1));
    Vec2 y2 = G2.transpose() * v2;
    EXPECT_NEAR(abs(y2(0)), 1.1181, tol);
    EXPECT_NEAR(y2(1), 0.0, tol);
    EXPECT_NEAR(abs(G2(0,0)), 0.1136, tol);
    EXPECT_NEAR(abs(G2(0,1)), 0.9935, tol);
    EXPECT_NEAR(abs(G2(1,0)), 0.9935, tol);
    EXPECT_NEAR(abs(G2(1,1)), 0.1136, tol);
}


// `Givens` and `SlowGivens` both project onto the left nullspace of Hf, but they
// build *different orthonormal bases* of it -- one by a sequence of plane
// rotations, one from a Householder QR. The two results therefore differ by an
// arbitrary orthogonal transform of the (2M-3)-dimensional nullspace, and the
// original version of this test compared them element by element. That is not a
// property either function has or should have; it is why the test shipped
// failing.
//
// What the EKF actually consumes is basis-invariant: for an orthonormal A,
// H'H = Hx' Q Q' Hx and H'r = Hx' Q Q' r depend on Q only through the projector
// QQ', which is the same subspace for both. Assert that instead.
TEST(NumericalLinearAlgebra, SlowAndFastGivensMatch) {
    number_t tol = 1e-9;

    int M = 4;
    // Deliberately more state columns than Hf has: rotating only Hf.cols()
    // columns of Hx used to leave columns 3 and 4 unmixed.
    MatX Hf = MatX::Random(2 * M, 3);
    MatX Hx = MatX::Random(2 * M, 5);
    VecX r = VecX::Random(2 * M);

    MatX Hf2 = Hf, Hx2 = Hx;
    VecX r2 = r;

    const int rows1 = Givens(r, Hx, Hf);
    MatX A;
    const int rows2 = SlowGivens(Hf2, Hx2, r2, A);

    ASSERT_EQ(rows1, 2 * M - 3);
    ASSERT_EQ(rows2, 2 * M - 3);

    const MatX H1 = Hx.topRows(rows1);
    const MatX H2 = Hx2.topRows(rows2);
    const VecX r1 = r.head(rows1);
    const VecX r2h = r2.head(rows2);

    // The feature block is eliminated by both.
    CheckMatrixZero(Hf.topRows(rows1), 1e-9);
    CheckMatrixZero(MatX(A.transpose() * Hf2), 1e-9);

    // The information the update actually uses.
    CheckMatrixEquality(MatX(H1.transpose() * H1), MatX(H2.transpose() * H2), tol);
    CheckVectorEquality(VecX(H1.transpose() * r1), VecX(H2.transpose() * r2h), tol);
    EXPECT_NEAR(r1.squaredNorm(), r2h.squaredNorm(), tol);
}


// The nullspace basis has to be orthonormal, or the projected measurement noise
// is A'(sigma^2 I)A = sigma^2 A'A and no longer matches the isotropic R the
// filter assumes. `FullPivLU::kernel()`, which this used to return, is not.
TEST(NumericalLinearAlgebra, SlowGivensBasisIsOrthonormal) {
    int M = 5;
    MatX Hf = MatX::Random(2 * M, 3);
    MatX Hx = MatX::Random(2 * M, 5);
    VecX r = VecX::Random(2 * M);

    MatX A;
    const int rows = SlowGivens(Hf, Hx, r, A);
    ASSERT_EQ(rows, 2 * M - 3);
    ASSERT_EQ(A.cols(), rows);
    ASSERT_EQ(A.rows(), 2 * M);
    CheckMatrixEquality(MatX(A.transpose() * A), MatX(MatX::Identity(rows, rows)),
                        1e-9);
}


// `Feature::oos_` is a fixed 2 * kMaxGroup buffer of which only the rows for the
// observations of *this* feature are filled; the tail holds whatever the
// previous user of the pooled object left behind. Reading the whole buffer folded
// that garbage into both the nullspace and the projected measurement.
TEST(NumericalLinearAlgebra, SlowGivensIgnoresTheUnfilledTail) {
    const int kBuf = 20; // over-sized buffer, as in the real caller
    const int filled = 8;

    MatX Hf = MatX::Zero(kBuf, 3);
    MatX Hx = MatX::Zero(kBuf, 5);
    VecX r = VecX::Zero(kBuf);
    Hf.topRows(filled) = MatX::Random(filled, 3);
    Hx.topRows(filled) = MatX::Random(filled, 5);
    r.head(filled) = VecX::Random(filled);

    MatX Hf_dirty = Hf, Hx_dirty = Hx;
    VecX r_dirty = r;
    // Stale rows, as a recycled Feature would have.
    Hf_dirty.bottomRows(kBuf - filled) = MatX::Random(kBuf - filled, 3);
    Hx_dirty.bottomRows(kBuf - filled) = MatX::Random(kBuf - filled, 5);
    r_dirty.tail(kBuf - filled) = VecX::Random(kBuf - filled);

    MatX A, A_dirty;
    const int rows = SlowGivens(Hf, Hx, r, A, filled);
    const int rows_dirty =
        SlowGivens(Hf_dirty, Hx_dirty, r_dirty, A_dirty, filled);

    ASSERT_EQ(rows, filled - 3);
    ASSERT_EQ(rows_dirty, filled - 3);
    CheckMatrixEquality(Hx.topRows(rows), Hx_dirty.topRows(rows), 1e-9);
    CheckVectorEquality(VecX(r.head(rows)), VecX(r_dirty.head(rows)), 1e-9);
}


// The buffer must keep its size: the caller goes on writing into it with
// fixed-size blocks on the next frame, and shrinking it turned that into an
// out-of-bounds write that NDEBUG hides.
TEST(NumericalLinearAlgebra, SlowGivensPreservesBufferShape) {
    const int kBuf = 30;
    MatX Hf = MatX::Random(kBuf, 3);
    MatX Hx = MatX::Random(kBuf, 5);
    VecX r = VecX::Random(kBuf);

    MatX A;
    SlowGivens(Hf, Hx, r, A, 10);
    EXPECT_EQ(Hx.rows(), kBuf);
    EXPECT_EQ(Hx.cols(), 5);
    EXPECT_EQ(r.rows(), kBuf);
}


TEST(NumericalLinearAlgebra, QR) {
    int N = 4;  // state size
    int M = 8;  // measurement size
    VecX r;
    MatX Hf, Hx;
    r = MatX::Random(M, 1);
    Hx = MatX::Random(M, N);

    std::cout << "r=\n" << r.transpose() << std::endl;
    std::cout << "Hx=\n" << Hx << std::endl;
    int rows = QR(r, Hx);
    std::cout << "Effective rows: " << rows << std::endl;
    std::cout << "===== After givens =====\n";
    std::cout << "r=\n";
    std::cout << r.head(rows).transpose() << std::endl;
    std::cout << "TH=\n";
    std::cout << Hx.topRows(rows) << std::endl;

}