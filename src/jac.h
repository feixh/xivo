// Jacobian management and cache.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "Eigen/Dense"
#include "core.h"
#include <array>

namespace xivo {

/** Storage for an out-of-state (MSCKF) measurement.
 *
 *  Used in two roles, which is why nothing is allocated by the constructor.
 *
 *  As *scratch* (`Feature::oos_scratch()`, one instance for the whole process):
 *  holds the stacked, un-marginalized `[Hf | Hx] dx = inn` while
 *  `Feature::ComputeOOSJacobianInternal` fills it one observation at a time, and
 *  is then consumed by `MarginalizeOOSPoint` or `ComputeInitJacobian`. Call
 *  `AllocateScratch`.
 *
 *  As a per-feature *result*: holds only the `rows - 3` marginalized rows that
 *  `Feature::Ho()` / `ro()` hand to the update, in an `Hx` sized to exactly
 *  those rows. `Hf` stays empty -- the whole point of the marginalization is
 *  that the 3D point is gone.
 *
 *  Splitting the two is worth about 300 MB of resident memory. A pooled
 *  `Feature` used to carry a full scratch buffer, and `MatX` is column-major, so
 *  the `Hx.block<2, kFullSize>(row, 0).setZero()` that writes one observation
 *  reaches into all 564 columns and therefore touches every 4 kB page of the
 *  406 kB allocation. Any feature that took the OOS path even once -- which,
 *  with `consistent_init`, is nearly every feature -- made its whole buffer
 *  resident. */
struct OOSJacobian {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MatX Hx; // ... w.r.t. state
  MatX Hf; // ... w.r.t. the free 3D point; scratch role only
  VecX inn;

  /** Rows for the most observations `SelectOOSObservations` can hand over: one
   *  per in-state group, times 2 for a stereo view that contributes 4 rows. */
  void AllocateScratch() {
    Hx.resize(2 * kMaxGroup, kFullSize);
    Hf.resize(2 * kMaxGroup, 3);
    inn.resize(2 * kMaxGroup);
  }

  /** Drop everything. Pooled objects must not retain this across a `Reset`. */
  void Release() {
    Hx.resize(0, 0);
    Hf.resize(0, 0);
    inn.resize(0);
  }
};

using OOSJacobianPtr = OOSJacobian *;

struct JacobianCache {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 Xc;     // 3D point in camera frame of the reference group
  Mat3 dXc_dx; // 3D point in reference camera frame w.r.t. local state

  Vec3 Xbr;
  Mat3 dXbr_dTbc, dXbr_dWbc;
  Mat3 dXbr_dXc;

  Vec3 Xs;                 // 3D point in spatial frame
  Mat3 dXs_dTsbr, dXs_dWsbr;   // w.r.t. body2spatial pose of the reference group
  Mat3 dXs_dXbr;            // 3D point in spatial frame w.r.t. Xc

  Vec3 Xb;
  Mat3 dXb_dTsb, dXb_dWsb;
  Mat3 dXb_dXs;

  Vec3 Xcn;      // 3D point in camera frame of the "new" (current) group
  Mat3 dXcn_dTbc, dXcn_dWbc; // w.r.t. cam2body alignment
  Mat3 dXcn_dXb;

  // Chain rule values
  Mat3 dXcn_dTsb, dXcn_dWsb;
  Mat3 dXcn_dTsbr, dXcn_dWsbr;
  Mat3 dXcn_dXs, dXcn_dx;

  Vec3 dXcn_dtd;                       // w.r.t. temporal offset
  Eigen::Matrix<number_t, 3, 9> dXcn_dCg; // w.r.t. gyroscope intrinsics
  Mat3 dXcn_dbg;

  Vec2 xcn;        // camera coordinates in the "new" group
  Mat23 dxcn_dXcn; // w.r.t. 3D point in camera frame

  Vec2 xp; // pixel coordinates
  Mat23 dxp_dXcn;
  Mat2 dxp_dxcn; // w.r.t. new camera coordinates
  Vec2 inn;      // innovation
};

} // namespace xivo
