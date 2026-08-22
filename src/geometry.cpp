#include <iostream>

#include <algorithm>

#include "Eigen/Eigenvalues"
#include "Eigen/OrderingMethods"
#include "Eigen/SparseCore"
#include "Eigen/SparseQR"
#include "glog/logging.h"

#include "geometry.h"

namespace xivo {

// Reference:
// Frank C. Park, Bryan J. Martin
// Robot Sensor Calibration: Solving AX = XB on the Euclidan Group
SO3 HandEyeCalibration(const std::vector<SO3> &A, const std::vector<SO3> &B) {
  int n = A.size();

  Eigen::SparseMatrix<number_t, Eigen::RowMajor> M(3 * n, 9);
  using ordering = Eigen::COLAMDOrdering<int>;
  Eigen::SparseQR<Eigen::SparseMatrix<number_t>, ordering> solver;
  VecX y(3 * n);
  // solve Mx = y
  for (int i = 0; i < n; ++i) {
    auto a = A[i].log();
    a /= a.norm();
    auto b = B[i].log();
    b /= b.norm();
    // solve Rb = a
    // Let R = [r1.T\\r2.T\\r3.T]
    // x = stack(r1, r2, r3), 9x1 vector
    int offset = i * 3;
    y.segment<3>(offset) = a;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        M.coeffRef(offset + row, 3 * row + col) = b(col);
      }
    }
  }
  // solve Mx = y
  solver.compute(M);
  auto xstack = solver.solve(y);
  Mat3 X;
  for (int i = 0; i < 3; ++i) {
    X.row(i) = xstack.segment<3>(i * 3).transpose();
  }
  auto R = SO3::fitToSO3(X);

  // compute residuals
  number_t res(0);
  for (int i = 0; i < n; ++i) {
    // AR=RB
    SO3 tmp = A[i] * R * (R * B[i]).inverse();
    auto r = tmp.log();
    res += r.norm();
  }
  res = (res / n) / M_PI * 180;
  LOG(INFO) << "residual=" << res << " degrees" << std::endl;

  return R;
}

SE3 HandEyeCalibration(const std::vector<SE3> &A, const std::vector<SE3> &B) {
  throw std::runtime_error("Not implemented.");
}

SE3 TrajectoryAlignment(const std::vector<Vec3> &Y,
                        const std::vector<Vec3> &X) {
  if (Y.size() != X.size())
    LOG(FATAL) << "Input trajectories have different length.";
  LOG(INFO) << "effective data points on trajectory=" << Y.size();
  // Compute difference vector between two consecutive time steps dX and dY
  // and find the rotational alignment R first such that dY = R dX.
  // Then find translation T such that Y = RX + T
  // Finally, on-manifold optimization for refinement.
  std::vector<Vec3> dX, dY;
  for (int i = 0; i + 1 < X.size(); ++i) {
    auto dx = X[i + 1] - X[i];
    auto dy = Y[i + 1] - Y[i];
    if (dx.norm() > 0 && dy.norm() > 0) {
      dX.push_back(dx / dx.norm());
      dY.push_back(dy / dy.norm());
    }
  }

  Eigen::SparseMatrix<number_t, Eigen::RowMajor> M(3 * dX.size(), 9);
  using ordering = Eigen::COLAMDOrdering<int>;
  Eigen::SparseQR<Eigen::SparseMatrix<number_t>, ordering> solver;

  LOG(INFO) << "building coefficient matrix M for M x = y";
  VecX rhs(3 * dY.size());
  // solve Mx = y
  for (int i = 0; i < dX.size(); ++i) {
    // solve Rb = a
    // Let R = [r1.T\\r2.T\\r3.T]
    // x = stack(r1, r2, r3), 9x1 vector
    int offset = i * 3;
    rhs.segment<3>(offset) = dY[i];
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        M.coeffRef(offset + row, 3 * row + col) = dX[i][col];
      }
    }
  }

  LOG(INFO) << "solving M x = y";
  // solve Mx = rhs
  solver.compute(M);
  auto xstack = solver.solve(rhs);
  Mat3 Xmat;
  for (int i = 0; i < 3; ++i) {
    Xmat.row(i) = xstack.segment<3>(i * 3).transpose();
  }
  auto R = SO3::fitToSO3(Xmat);

  LOG(INFO) << "computing T";
  // compute translation
  Vec3 T{0, 0, 0};
  for (int i = 0; i < X.size(); ++i) {
    T += Y[i] - R * X[i];
  }
  T /= X.size();

  // on-manifold optimization
  // Y = R(I+hat(dW)) X + T+dT = (R X + T) + R hat(dW) X + dT
  // residual = Y - (R X + T) = f(dW, dT) = -R hat(X) dW + dT
  // [-R hat(X) | I3x3] [dW' dT']' = residual
  // Vec3 dW, dT;
  rhs.resize(X.size() * 3);
  M.resize(X.size() * 3, 6);
  M.setZero();
  for (int iter = 0; iter < 5; ++iter) {
    for (int i = 0; i < X.size(); ++i) {
      int offset = i * 3;
      rhs.segment<3>(offset) = Y[i] - (R * X[i] + T);
      auto block = -R.matrix() * SO3::hat(X[i]);
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          M.coeffRef(offset + row, col) = block(row, col);
          if (row == col)
            M.coeffRef(offset + row, 3 + col) = 1.0;
        }
      }
    }
    auto Mt = M.transpose();
    auto MtM = Mt * M;
    solver.compute(MtM);
    auto dWdT = solver.solve(Mt * rhs);
    R *= SO3::exp(dWdT.head<3>());
    T += dWdT.tail<3>();
    // std::cout << "iter=" << iter << " ;dWdT=" << dWdT.transpose() <<
    // std::endl;
  }

  // output pose
  SE3 g{R, T};
  LOG(INFO) << "trajectory alignment=\n" << g.matrix3x4();

  return g;
}


bool PointsAreCollinear(const std::vector<Vec3>& pts, number_t thresh) {
  // Three defects in the previous implementation, all of which mattered:
  //
  //  1. `pts[1] - pts[0]` was read unguarded, so a vector of 0 or 1 points read
  //     out of bounds.
  //  2. It compared |(p1-p0) x (pi-p0)|, an *area*, against a fixed threshold.
  //     That grows with the square of the distance between the points, so the
  //     same geometry passes or fails depending only on how far away it is; and
  //     it degenerates to "always collinear" whenever p0 and p1 happen to be
  //     close together, independently of where the other points lie.
  //  3. The verdict depended on which point landed at index 0. The only caller
  //     (`Graph::FindNewGaugeFeatures`) builds the vector by iterating an
  //     `unordered_set<FeaturePtr>`, whose order is the features' heap
  //     addresses -- which is why runs of two different binaries on identical
  //     input diverged (see notes-bugfix/m0-baseline.md).
  //
  // Use the second moment of the centred point set instead. Its eigenvalues are
  // invariant to the ordering, and the ratio of the largest spread orthogonal to
  // the best-fit line to the spread along it is dimensionless: `thresh` is now
  // sin(angle) of the residual off-line spread rather than an area in m^2.
  if (pts.size() < 3) {
    // Fewer than three points are always collinear.
    return true;
  }

  Vec3 centroid = Vec3::Zero();
  for (const auto &p : pts) {
    centroid += p;
  }
  centroid /= static_cast<number_t>(pts.size());

  Mat3 M = Mat3::Zero();
  for (const auto &p : pts) {
    Vec3 d = p - centroid;
    M += d * d.transpose();
  }

  Eigen::SelfAdjointEigenSolver<Mat3> eig(M);
  if (eig.info() != Eigen::Success) {
    return true; // cannot tell; treat as degenerate
  }
  // Ascending order: s(2) is the variance along the best-fit line, s(1) the
  // largest variance orthogonal to it.
  const Vec3 s = eig.eigenvalues();
  if (!(s(2) > 0)) {
    return true; // every point coincides
  }
  return std::sqrt(std::max<number_t>(s(1), 0) / s(2)) < thresh;
}


} // namespace xivo
