#include "helpers.h"

#include <algorithm>
#include <iostream>
#include <memory>

#include "glog/logging.h"

namespace xivo {

// Reference:
// http://www.math.usm.edu/lambers/mat610/sum10/lecture9.pdf
int SlowGivens(const MatX &Hf, MatX &Hx, VecX &inn, MatX &A,
               int effective_rows) {
  const int rows = (effective_rows < 0 ? static_cast<int>(Hf.rows())
                                       : effective_rows);
  const int cols = Hf.cols();

  CHECK(rows <= Hf.rows());
  CHECK(rows <= Hx.rows());
  CHECK(rows <= inn.rows());

  // The old implementation had three problems, all of which reached the filter:
  //
  //  1. `FullPivLU::kernel()` is *not* orthonormal. The MSCKF nullspace
  //     projection is only noise-preserving for an orthonormal basis; with a
  //     general basis the projected noise is sigma^2 * A'A while the update
  //     still assumed sigma^2 * I, so the OOS measurements were weighted by an
  //     essentially arbitrary matrix.
  //  2. It ignored how many rows were actually filled. `Feature::oos_` is a
  //     fixed 2 * kMaxGroup buffer written one observation at a time, so the
  //     unfilled tail -- stale rows from whichever feature last used this
  //     pooled object -- was folded into both the nullspace and the result.
  //  3. It *resized* `Hx` down to (rows - cols). Since `oos_.Hx` is the
  //     persistent buffer that `ComputeOOSJacobianInternal` writes into with
  //     fixed-size `block<2, kFullSize>(2 * counter, 0)`, a later feature with
  //     enough observations to need the rows that had been trimmed away wrote
  //     past the end of the matrix. Eigen's bounds assertions are disabled by
  //     NDEBUG in a release build, so this was a silent heap overwrite.
  //
  // Householder QR gives the orthonormal basis directly: Hf = Q R with R upper
  // triangular, so the trailing columns of Q span the left nullspace of Hf.
  const int rank = std::min(rows, cols);
  const int out_rows = rows - rank;
  if (out_rows <= 0) {
    A.resize(rows, 0);
    return 0;
  }

  Eigen::HouseholderQR<MatX> qr(Hf.topRows(rows));
  const MatX Q = qr.householderQ();
  A = Q.rightCols(out_rows);

  // Into temporaries: the destinations overlap the sources.
  const MatX Hx_proj = A.transpose() * Hx.topRows(rows);
  const VecX inn_proj = A.transpose() * inn.head(rows);
  Hx.topRows(out_rows) = Hx_proj;
  inn.head(out_rows) = inn_proj;

  return out_rows;
}

// Matrix computation. Golub & Loan.
// page 240, Algorithm 5.1.3
static Mat2 givens(number_t a, number_t b) {
  Mat2 G;
  number_t c, s, t;
  if (fabs(b) < eps) {
    c = 1;
    s = 0;
  } else {
    if (fabs(b) > fabs(a)) {
      t = -a / b;
      s = 1 / sqrt(1 + t * t);
      c = s * t;
    } else {
      t = -b / a;
      c = 1 / sqrt(1 + t * t);
      s = c * t;
    }
  }
  G << c, s, -s, c;
  return G;
}

int Givens(VecX &x, MatX &Hx, MatX &Hf, int effective_rows) {
  CHECK(x.rows() == Hx.rows());
  CHECK(x.rows() == Hf.rows());

  int rows = (effective_rows == -1 ? Hf.rows() : effective_rows);
  int cols = Hf.cols();

  // `n ^ 1` is bitwise XOR: it is non-zero -- i.e. the assertion passes -- for
  // every n except 1, so the intended "the row count is even, because
  // measurements come in (u, v) pairs" precondition was never actually checked.
  CHECK(rows % 2 == 0) << "expected an even row count, got " << rows;
  CHECK(effective_rows <= x.rows());

  // Hx has the full state width; Hf has three columns. Rotating only
  // `Hf.cols()` columns of Hx left every column from the fourth on
  // untouched, so the returned Hx was not A' * Hx for any A -- the rows had
  // been mixed by the rotations in their first three columns and not in the
  // rest. This is what the shipped `SlowAndFastGivensMatch` failure was
  // reporting: its Hx has five columns.
  const int hx_cols = Hx.cols();

  Mat2 Gt;
  for (int c = 0; c < cols; ++c) {
    for (int r = rows - 2; r >= c; --r) {
      Gt.transpose() = givens(Hf(r, c), Hf(r + 1, c));
      Hf.block(r, 0, 2, cols) = Gt * Hf.block(r, 0, 2, cols);
      Hx.block(r, 0, 2, hx_cols) = Gt * Hx.block(r, 0, 2, hx_cols);

      x.segment<2>(r) = Gt * x.segment<2>(r);
    }
  }
  if (rows <= cols) {
    // Nothing survives the elimination; the caller must not be handed a
    // negative row count.
    return 0;
  }
  // now strip the first #cols rows
  for (int r = 0; r < rows - cols; ++r) {
    x(r) = x(r + cols);
    Hx.row(r) = Hx.row(r + cols);
    Hf.row(r) = Hf.row(r + cols);
  }
  return rows - cols;
}

int QR(VecX &x, MatX &Hx, int effective_rows) {
  CHECK(x.rows() == Hx.rows());

  int rows = (effective_rows == -1 ? Hx.rows() : effective_rows);
  int cols = Hx.cols();

  CHECK(rows > cols);

  Mat2 Gt;
  for (int c = 0; c < cols; ++c) {
    for (int r = rows - 2; r >= c; --r) {
      Gt.transpose() = givens(Hx(r, c), Hx(r + 1, c));
      Hx.block(r, 0, 2, cols) = Gt * Hx.block(r, 0, 2, cols);
      x.segment<2>(r) = Gt * x.segment<2>(r);
    }
  }
  // in contrast to the Givens function used to eliminate measurement jacobian
  // Hf of feature,
  // here we want to keep the parts which are not eliminated -- which are the
  // first #rows
  // of both residual vector x and measurement jacobian Hx
  // return the number of rows, and let the caller to decide what to do with the
  // matrices.
  return rows;
}

bool DirectLinearTransformSVD(const SE3 &g12, const Vec2 &xc1, const Vec2 &xc2, Vec3 &X) {
  Vec3 t12{g12.translation()};
  Mat3 R12{g12.so3().matrix()};
  // [I | 0]: the fourth column is never assigned, so it has to be zeroed here
  // rather than left to -DEIGEN_INITIALIZE_MATRICES_BY_ZERO. `A.row(0)` and
  // `A.row(1)` read `P1.row(2)` in full, fourth column included.
  Mat34 P1{Mat34::Zero()};
  P1.block<3, 3>(0, 0).setIdentity();
  Mat34 P2;
  P2.block<3, 3>(0, 0) = R12.transpose();
  P2.block<3, 1>(0, 3) = -R12.transpose() * t12;
  Vec3 f1{xc1(0), xc1(1), 1.0};
  f1.normalize();
  Vec3 f2{xc2(0), xc2(1), 1.0};
  f2.normalize();

  Mat4 A;
  A.row(0) = f1(0) * P1.row(2) - f1(2) * P1.row(0);
  A.row(1) = f1(1) * P1.row(2) - f1(2) * P1.row(1);
  A.row(2) = f2(0) * P2.row(2) - f2(2) * P2.row(0);
  A.row(3) = f2(1) * P2.row(2) - f2(2) * P2.row(1);

  Eigen::JacobiSVD<Mat4> svd(A, Eigen::ComputeFullV);
  auto V = svd.matrixV();

  // V(3,3) -> 0 for a point at infinity, i.e. a zero-parallax pair -- routine
  // for a nearly static or purely rotating camera. Dividing anyway produced
  // inf/NaN and `return true` reported it as a good triangulation; the caller's
  // only sanity check is a zmin/zmax range test, which NaN passes (every
  // comparison against NaN is false).
  if (std::abs(V(3, 3)) < 1e-10) {
    return false;
  }
  X << V(0, 3), V(1, 3), V(2, 3);
  X /= V(3, 3);

  return X.allFinite();
}

bool DirectLinearTransformAvg(const SE3 &g12, const Vec2 &xc1, const Vec2 &xc2, Vec3 &X) {
  Vec3 t12{g12.translation()};
  Mat3 R12{g12.so3().matrix()};

  Vec3 f1{xc1(0), xc1(1), 1.0};
  f1.normalize();
  Vec3 f2{xc2(0), xc2(1), 1.0};
  f2.normalize();

  Vec3 f2_unrotated{R12 * f2};
  Vec2 b;
  b << t12.dot(f1), t12.dot(f2_unrotated);
  Mat2 A;
  A(0, 0) = f1.dot(f1);
  A(1, 0) = f1.dot(f2_unrotated);
  A(0, 1) = -A(1, 0);
  A(1, 1) = -f2_unrotated.dot(f2_unrotated);
  // A is singular when the two bearings are parallel (zero parallax);
  // `A.inverse()` returns inf/NaN rather than failing.
  if (std::abs(A.determinant()) < 1e-12) {
    return false;
  }
  Vec2 lambda = A.inverse() * b;
  Vec3 xm = lambda(0) * f1;
  Vec3 xn = t12 + lambda(1) * f2_unrotated;
  X = (xm + xn) / 2.0;

  return X.allFinite();
}


bool L1Angular(const SE3 &g01, const Vec2 &xc0, const Vec2 &xc1, Vec3 &X, float max_theta_thresh, float beta_thresh) {

  // Initalize the Rotation and Translation Matricies
  Vec3 t01{g01.translation()};
  Mat3 R01{g01.so3().matrix()};
  Mat3 R10{R01.transpose()};
  Vec3 t10{-1 * R01.transpose() * t01};

  // Create homogeneous coordinates
  Vec3 f0{xc0(0), xc0(1), 1.0};
  f0.normalize();
  Vec3 f1{xc1(0), xc1(1), 1.0};
  f1.normalize();

  Vec3 m0{R10 * f0};
  Vec3 m1{f1};

  float a0 = ((m0 / m0.norm()).cross(t10)).norm();
  float a1 = ((m1 / m1.norm()).cross(t10)).norm();

  Vec3 m0_prime;
  Vec3 m1_prime;

  if(a0 <= a1)
  {
    Vec3 n1 = m1.cross(t10);
    Vec3 n1_hat = n1 / n1.norm();
    m0_prime = m0 - (m0.dot(n1_hat)) * n1_hat;
    m1_prime = m1;
  }
  else
  {
    Vec3 n0 = m0.cross(t10);
    Vec3 n0_hat = n0 / n0.norm();
    m0_prime = m0;
    m1_prime = m1 - (m1.dot(n0_hat)) * n0_hat;
  }

  Vec3 Rf0_prime = m0_prime;
  Vec3 f1_prime = m1_prime;

  Vec3 z = f1_prime.cross(Rf0_prime);


  X = ((z.dot(t10.cross(Rf0_prime))) / pow(z.norm(),2)) * f1_prime;

  // Returns point from 1st frame of reference
  X = R01 * X + t01;

  // Check the conditions
  if(!check_cheirality(z, t10, f1_prime, Rf0_prime) ||
    !check_angular_reprojection(m0, Rf0_prime, m1, f1_prime, max_theta_thresh) ||
    !check_parallax(Rf0_prime, f1_prime, beta_thresh))
  {
    return false;
  }

  return true;
}


bool L2Angular(const SE3 &g01, const Vec2 &xc0, const Vec2 &xc1, Vec3 &X, float max_theta_thresh, float beta_thresh) {

  // Initalize the Rotation and Translation Matricies
  Vec3 t01{g01.translation()};
  Mat3 R01{g01.so3().matrix()};
  Mat3 R10{R01.transpose()};
  Vec3 t10{-1 * R01.transpose() * t01};

  // Create homogeneous coordinates
  Vec3 f0{xc0(0), xc0(1), 1.0};
  f0.normalize();
  Vec3 f1{xc1(0), xc1(1), 1.0};
  f1.normalize();

  Vec3 m0{R10 * f0};
  Vec3 m1{f1};

  Vec3 m0_hat = m0 / m0.norm();
  Vec3 m1_hat = m1 / m1.norm();

  Eigen::Matrix<double, 3, 2> A;
  A.row(0) << m0_hat(0), m1_hat(0);
  A.row(1) << m0_hat(1), m1_hat(1);
  A.row(2) << m0_hat(2), m1_hat(2);

  Vec3 t10_hat = t10 / t10.norm();
  Mat3 I = Eigen::Matrix3d::Identity();

  Eigen::Matrix<double, 2, 3> B;
  B = A.transpose() * (I - t10_hat * t10_hat.transpose());

  Eigen::JacobiSVD<Eigen::Matrix<double, 2, 3>> svd(B, Eigen::ComputeFullV);
  Mat3 V = svd.matrixV();
  Vec3 n_prime_hat = V.col(1);

  Vec3 m0_prime = m0 - m0.dot(n_prime_hat) * n_prime_hat;
  Vec3 m1_prime = m1 - m1.dot(n_prime_hat) * n_prime_hat;

  Vec3 Rf0_prime = m0_prime;
  Vec3 f1_prime = m1_prime;

  Vec3 z = f1_prime.cross(Rf0_prime);

  X = ((z.dot(t10.cross(Rf0_prime))) / pow(z.norm(),2)) * f1_prime;

  // Returns point from 1st frame of reference
  X = R01 * X + t01;

  // Check the conditions
  if(!check_cheirality(z, t10, f1_prime, Rf0_prime) ||
    !check_angular_reprojection(m0, Rf0_prime, m1, f1_prime, max_theta_thresh) ||
    !check_parallax(Rf0_prime, f1_prime, beta_thresh))
  {
    return false;
  }

  return true;
}

bool LinfAngular(const SE3 &g01, const Vec2 &xc0, const Vec2 &xc1, Vec3 &X, float max_theta_thresh, float beta_thresh) {

  // Initalize the Rotation and Translation Matricies
  Vec3 t01{g01.translation()};
  Mat3 R01{g01.so3().matrix()};
  Mat3 R10{R01.transpose()};
  Vec3 t10{-1 * R01.transpose() * t01};

  // Create homogeneous coordinates
  Vec3 f0{xc0(0), xc0(1), 1.0};
  f0.normalize();
  Vec3 f1{xc1(0), xc1(1), 1.0};
  f1.normalize();

  Vec3 m0{R10 * f0};
  Vec3 m1{f1};

  Vec3 m0_hat = m0 / m0.norm();
  Vec3 m1_hat = m1 / m1.norm();

  Vec3 n_a = (m0_hat + m1_hat).cross(t10);
  Vec3 n_b = (m0_hat - m1_hat).cross(t10);

  // `m - (m.n)n` is a projection onto the plane perpendicular to n only if n is
  // a *unit* vector. n_a/n_b are raw cross products with norms far from 1, so
  // without normalizing this over/under-corrected by ||n||^2. L1Angular and
  // L2Angular both use unit normals.
  Vec3 n_prime_hat = (n_a.norm() >= n_b.norm() ? n_a : n_b);
  if (n_prime_hat.norm() < 1e-10) {
    return false; // degenerate: both bearings parallel to the baseline
  }
  n_prime_hat.normalize();

  Vec3 m0_prime = m0 - m0.dot(n_prime_hat) * n_prime_hat;
  Vec3 m1_prime = m1 - m1.dot(n_prime_hat) * n_prime_hat;

  Vec3 Rf0_prime = m0_prime;
  Vec3 f1_prime = m1_prime;

  Vec3 z = f1_prime.cross(Rf0_prime);

  X = ((z.dot(t10.cross(Rf0_prime))) / pow(z.norm(),2)) * f1_prime;

  // Returns point from 1st frame of reference
  X = R01 * X + t01;

  // Check the conditions
  if(!check_cheirality(z, t10, f1_prime, Rf0_prime) ||
    !check_angular_reprojection(m0, Rf0_prime, m1, f1_prime, max_theta_thresh) ||
    !check_parallax(Rf0_prime, f1_prime, beta_thresh))
  {
    return false;
  }

  return true;
}


/** acos with its argument clamped to [-1, 1]. Ratios of floating-point dot
 * products and norms routinely land 1-2 ulp outside the domain, where acos
 * returns NaN. Returns NaN only if the input is already NaN. */
static float SafeAcos(float c) {
  if (std::isnan(c)) {
    return c;
  }
  return std::acos(std::clamp(c, -1.0f, 1.0f));
}

bool check_cheirality(const Vec3 &z, const Vec3 &t, const Vec3 &f1_prime, const Vec3 &Rf0_prime)
{

  float lambda0 = z.dot(t.cross(f1_prime)) / pow(z.norm(), 2);
  float lambda1 = z.dot(t.cross(Rf0_prime)) / pow(z.norm(), 2);

  if(lambda0 <= 0 || lambda1 <= 0)
  {
    LOG(WARNING) << "[WARNING] cheirality error in triangulation. lambda0=" << lambda0 << ", lamba1=" << lambda1;
    return false;
  }

  return true;
}


bool check_angular_reprojection(const Vec3 &Rf0, const Vec3 &Rf0_prime, const Vec3 &f1, const Vec3 &f1_prime, float max_theta_thresh)
{

  // The cosine has to be clamped: these methods leave one of the two bearings
  // *unchanged* by construction, so the ratio is 1 + O(ulp) and unclamped acos
  // returns NaN. std::max(NaN, x) returns NaN, and `NaN > thresh` is false --
  // so the gate silently accepted the outlier it was there to reject.
  float theta0 = SafeAcos(Rf0.dot(Rf0_prime) / (Rf0.norm() * Rf0_prime.norm()));
  float theta1 = SafeAcos(f1.dot(f1_prime) / (f1.norm() * f1_prime.norm()));

  float max_theta = std::max(theta0, theta1);

  if(!std::isfinite(max_theta) || max_theta > max_theta_thresh)
  {
    LOG(WARNING) << "[WARNING] angular reprojection error in triangulation";
    return false;
  }
  return true;
}

bool check_parallax(const Vec3 &Rf0_prime, const Vec3 &f1_prime, float beta_thresh)
{

  float beta = SafeAcos(f1_prime.dot(Rf0_prime) /
                        (f1_prime.norm() * Rf0_prime.norm()));

  if(!std::isfinite(beta) || beta < beta_thresh)
  {
    LOG(WARNING) << "[WARNING] parallax error in triangulation " << beta;
    return false;
  }

  return true;
}


SO3 SO3_from_rotvec(const Vec3 w) {
  AngleAxis aa(w.norm(), w.normalized());
  Quat q(aa);
  return SO3(q);
}

} // namespace xivo
