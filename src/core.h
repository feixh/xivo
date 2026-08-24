// Core data structures include: 
// 1) timestamp types
// 2) representation of the nominal state (State)
// 3) layout of the error state (Index and other offsets)
// 4) status of components, etc.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <array>
#include <chrono>
#include <exception>
#include <iostream>
#include <list>
#include <memory>
#include <type_traits>

#include "sophus/se3.hpp"

#include "alias.h"
#include "camera_manager.h"
#include "helpers.h"
#include "rodrigues.h"
#include "utils.h"

namespace xivo {

////////////////////////////////////////
// TYPES FOR TIME
////////////////////////////////////////
using nanoseconds = std::chrono::nanoseconds;  // uint64_t
using seconds = std::chrono::duration<double>; // double
using timestamp_t = nanoseconds;

////////////////////////////////////////
// STATE DIMENSION
////////////////////////////////////////
// NOTE: in the implementation, the spatial frame is actually the world frame, which is arbitrary.
// Gravity of the form g=[0, 0, -9.8] is brought from the inertial frame to the spatial (world) frame .
// by the rotation matrix Rg, i.e., Rg * g is the gravity in the spatial (world) frame.
// Since the rotation around z-axis of the inertial frame is not observable, the z-component of Wg=Log(Rg)
// is not included in the model.
enum Index : int {
  Wsb = 0, // Wsb, rotation
  Tsb = 3, // Tsb, translation
  Vsb = 6, // vsb, velocity
  bg = 9,   // omega bias
  ba = 12,  // alpha bias
  Wbc = 15, // alignment rotation
  Tbc = 18, // alignment translation
  Wsg = 21,  // alignment of gravity from [0, 0, -9.8] to the spatial frame
#ifdef USE_ONLINE_TEMPORAL_CALIB
  td = Wsg + 2, // temporal offset
#endif

#ifdef USE_ONLINE_IMU_CALIB // USE_ONLINE_IMU_CALIB and USE_ONLINE_TEMPORAL_CALIB

#ifdef USE_ONLINE_TEMPORAL_CALIB
  Cg = td + 1, // gyro calibration, 9 numbers
#else
  Cg = Wsg + 2, // gyro calibration, 9 numbers
#endif
  Ca = Cg + 9, // accel calibration, 6 numbers
  End = Ca + 6,

#else

#ifdef USE_ONLINE_TEMPORAL_CALIB // USE_ONLINE_TEMPORAL_CALIB, but not USE_ONLINE_IMU_CALIB
  End = td + 1,
#else
  End = Wsg + 2
#endif

#endif

};

constexpr int kMotionSize = Index::End;

constexpr int kCameraBegin = kMotionSize;

#ifdef USE_ONLINE_CAMERA_CALIB
constexpr int kMaxCameraIntrinsics =
    9; // maximal possible number of intrinsic parameters
#else
constexpr int kMaxCameraIntrinsics =
    0; // maximal possible number of intrinsic parameters
#endif

constexpr int kGroupSize = 6;
constexpr int kFeatureSize = 3;

// By reducing the number of groups and features, we can trade off computational cost
// and accuracy
#ifdef EKF_MAX_FEATURES
constexpr int kMaxFeature = EKF_MAX_FEATURES;
#else
constexpr int kMaxFeature = 30;
#endif
#ifdef EKF_MAX_GROUPS
constexpr int kMaxGroup = EKF_MAX_GROUPS;
#else
constexpr int kMaxGroup = 15;
#endif

constexpr int kGroupBegin = kCameraBegin + kMaxCameraIntrinsics;
constexpr int kFeatureBegin = kGroupBegin + kGroupSize * kMaxGroup;
constexpr int kFullSize = kFeatureBegin + kFeatureSize * kMaxFeature;

////////////////////////////////////////
// SPARSITY OF A VISUAL MEASUREMENT
////////////////////////////////////////
/** A contiguous run of error-state columns. */
struct ColRun {
  int start;
  int len;
};

/** The column runs of a visual-measurement Jacobian that
 *  `Feature::ComputeJacobian` actually writes, other than the two that depend on
 *  which slots the feature and its reference group occupy.
 *
 *  A Jacobian row block is stored full width (`Eigen::Matrix<number_t, 2,
 *  kFullSize>`), but only a handful of its column blocks can ever be nonzero:
 *  a measurement of one feature says nothing about `Vsb`, `ba` or `Wsg`, nothing
 *  about any group but its own reference group, and nothing about any other
 *  feature. At the shipped capacity that is 25 columns of 564, so a dense
 *  `J * P * J^T` spends 96% of its arithmetic multiplying by structural zero.
 *
 *  Kept here beside `Index` and written with the same `#ifdef`s as
 *  `ComputeJacobian` so the two cannot drift apart; `unitTests_jacobians_stereo`
 *  pins the correspondence in both directions: everything outside these runs is
 *  exactly zero, and every column the finite-difference tests find live is
 *  inside one of them. Runs are merged where the layout makes them adjacent,
 *  which the `static_assert`s below enforce -- fewer, longer runs make the
 *  gather cheaper.
 *
 *  A run that is present but zero is deliberately *kept* rather than trimmed:
 *  including a zero column changes no result, whereas omitting a nonzero one
 *  silently corrupts the filter. That is why the camera-intrinsics run covers
 *  all `kMaxCameraIntrinsics` columns and not the `Camera::dim()` of them that
 *  are live.
 */
static_assert(Index::Tsb == Index::Wsb + 3, "Wsb and Tsb must be adjacent");
static_assert(Index::Tbc == Index::Wbc + 3, "Wbc and Tbc must be adjacent");

constexpr std::array<ColRun, 4
#ifdef USE_ONLINE_TEMPORAL_CALIB
                            + 1
#ifdef USE_ONLINE_IMU_CALIB
                            + 1
#endif
#endif
#ifdef USE_ONLINE_CAMERA_CALIB
                            + 1
#endif
                     >
    kJacSharedRuns{{
        {Index::Wsb, 6}, // Wsb and Tsb
        {Index::bg, 3},
        {Index::Wbc, 6}, // Wbc and Tbc
#ifdef USE_ONLINE_TEMPORAL_CALIB
        {Index::td, 1},
#ifdef USE_ONLINE_IMU_CALIB
        {Index::Cg, 9},
#endif
#endif
#ifdef USE_ONLINE_CAMERA_CALIB
        {kCameraBegin, kMaxCameraIntrinsics},
#endif
        // Sentinel, replaced per feature by its reference group's run.
        {kGroupBegin, kGroupSize},
    }};

/** Number of column runs in one measurement: the shared ones (the last of which
 *  is the group sentinel) plus the feature's own. */
constexpr int kJacRuns = static_cast<int>(kJacSharedRuns.size()) + 1;

constexpr int SumRunLengths() {
  int n = 0;
  for (const auto &r : kJacSharedRuns)
    n += r.len;
  return n + kFeatureSize;
}
/** Width of one measurement's compacted Jacobian: 25 at the shipped capacity. */
constexpr int kJacCols = SumRunLengths();

static_assert(kJacCols <= kFullSize, "compacted Jacobian cannot exceed the state");

/** A measurement's column runs, given the state slots its feature and reference
 *  group occupy (`Feature::sind()` and `Group::sind()`). */
inline void MeasurementRuns(int gsind, int fsind, ColRun (&runs)[kJacRuns]) {
  int n = 0;
  for (const auto &r : kJacSharedRuns)
    runs[n++] = r;
  runs[n - 1] = {kGroupBegin + kGroupSize * gsind, kGroupSize}; // the sentinel
  runs[n++] = {kFeatureBegin + kFeatureSize * fsind, kFeatureSize};
}

using JacCompact = Eigen::Matrix<number_t, 2, kJacCols>;
using CovCompact = Eigen::Matrix<number_t, kJacCols, kJacCols>;

/** Gathers the `kJacCols` columns named by `runs` out of a full-width row block.
 *  `dst` has as many rows as `src`. */
template <typename Derived, typename Dst>
inline void GatherCols(const Eigen::MatrixBase<Derived> &src,
                       const ColRun (&runs)[kJacRuns], Dst &dst) {
  int c = 0;
  for (const auto &r : runs) {
    dst.middleCols(c, r.len) = src.middleCols(r.start, r.len);
    c += r.len;
  }
}

/** Gathers the symmetric `kJacCols x kJacCols` submatrix of `P` on the rows and
 *  columns named by `runs`. */
template <typename Derived>
inline void GatherCov(const Eigen::MatrixBase<Derived> &P,
                      const ColRun (&runs)[kJacRuns], CovCompact &dst) {
  int ci = 0;
  for (const auto &ri : runs) {
    int cj = 0;
    for (const auto &rj : runs) {
      dst.block(ci, cj, ri.len, rj.len) =
          P.block(ri.start, rj.start, ri.len, rj.len);
      cj += rj.len;
    }
    ci += ri.len;
  }
}

/** Innovation covariance `J P J^T + R I` of one 2-row visual measurement,
 *  formed from the `kJacCols x kJacCols` slice of `P` that `J` can actually
 *  reach rather than from all of it. `gsind` and `fsind` are the state slots of
 *  the measurement's reference group and feature (`Group::sind()`,
 *  `Feature::sind()`).
 *
 *  Not bit-identical to the dense `J * P * J.transpose()`: compacting changes
 *  which nonzero products land in the same accumulation block inside Eigen's
 *  gemm, so the sums are reassociated. `unitTests_jacobians_stereo` checks the
 *  two agree to 1e-12 relative, and that a real `ComputeJacobian` output is exactly
 *  zero outside these runs -- which is what makes the two equal in the first
 *  place.
 *
 *  A free function rather than an `Estimator` member so that the test can hand
 *  it an arbitrary `P` and compare against the dense product. */
template <typename JDerived, typename PDerived>
inline Mat2 InnovationCov(const Eigen::MatrixBase<JDerived> &J,
                          const Eigen::MatrixBase<PDerived> &P, int gsind,
                          int fsind, number_t R) {
  ColRun runs[kJacRuns];
  MeasurementRuns(gsind, fsind, runs);
  JacCompact Jc;
  GatherCols(J, runs, Jc);
  CovCompact Pc;
  GatherCov(P, runs, Pc);
  Mat2 S = Jc * Pc * Jc.transpose();
  S(0, 0) += R;
  S(1, 1) += R;
  return S;
}

// frequency to project rotation matrices to SO3 to get rid of the accumulated numeric error
#ifdef ENFORCE_SO3_FREQ
constexpr int kEnforceSO3Freq = ENFORCE_SO3_FREQ;
#else
constexpr int kEnforceSO3Freq = 50;
#endif

////////////////////////////////////////
// STATE
////////////////////////////////////////
struct State {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // nominal state
  State(): counter{0} {}

  int counter;
  SO3 Rsb;       // body to spatial rotation
  Vec3 Tsb, Vsb; // body to spatial translation and velocity
  Vec3 bg, ba;   // gyro and accl bias

  SO3 Rbc;
  Vec3 Tbc;
  SO3 Rsg;  // gravity -> spatial

  // Declared unconditionally, but only assigned from the config under
  // USE_ONLINE_TEMPORAL_CALIB -- which the shipped build leaves undefined. The
  // constructor initialised only `counter`, so in the default build `td` held an
  // indeterminate value that was still copied out (Estimator::td(), and the
  // by-value argument to Feature::ComputeJacobian at update.cpp:29 and
  // manager.cpp:105) and printed by ~Estimator under `print_calibration`. It read
  // as zero in practice only because a freshly mapped heap page is zero.
  number_t td{0};

  using Tangent = Eigen::Matrix<number_t, kMotionSize, 1>;

  State &operator+=(const Tangent &dX) {

    SO3 dRsb = SO3_from_rotvec(dX.segment<3>(Index::Wsb));
    SO3 dRbc = SO3_from_rotvec(dX.segment<3>(Index::Wbc));
    Vec3 Wsg{dX(Index::Wsg), dX(Index::Wsg+1), 0.0};
    SO3 dRsg = SO3_from_rotvec(Wsg);

    Rsb *= dRsb;
    Tsb += dX.segment<3>(Index::Tsb);
    Vsb += dX.segment<3>(Index::Vsb);
    bg += dX.segment<3>(Index::bg);
    ba += dX.segment<3>(Index::ba);
    Rbc *= dRbc;
    Tbc += dX.segment<3>(Index::Tbc);
    Rsg *= dRsg;
#ifdef USE_ONLINE_TEMPORAL_CALIB
    td += dX(Index::td);
#endif

    if constexpr(kEnforceSO3Freq > 0) {
      if (++counter % kEnforceSO3Freq == 0) {
        Rsb.normalize();
        Rbc.normalize();
        auto Wsg = Rsg.log();
        Wsg(2) = 0.0;
        Rsg = SO3_from_rotvec(Wsg);
      }
    }

    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os, const State &s) {
    os << "\n=====\n";
    os << "Rsb=\n" << s.Rsb.matrix();
    os << "\nTsb=\n" << s.Tsb.transpose();
    os << "\nVsb=\n" << s.Vsb.transpose();
    os << "\nbg=\n" << s.bg.transpose();
    os << "\nba=\n" << s.ba.transpose();
    os << "\nRbc=\n" << s.Rbc.matrix();
    os << "\nTbc=\n" << s.Tbc.transpose();
    os << "\nRg=\n" << s.Rsg.matrix();
    os << "\n=====\n";
    return os;
  }
};

////////////////////////////////////////
// STATUS
////////////////////////////////////////
enum class TrackStatus : int {
  CREATED = 0,  // feature just been detected
  TRACKED = 1,  // feature being tracked well
  DROPPED = 2   // no longer in view
};
enum class FeatureStatus : int {
  CREATED = 0,
  INITIALIZING = 1,
  READY = 2,
  INSTATE = 3,
  REJECTED_BY_FILTER = 4,
  REJECTED_BY_TRACKER = 5,
  NULLREFED = 6,
  GAUGE = 7, // chosen to fix gauge freedom
};

enum class GroupStatus : int {
  CREATED = 0,  // newly created
  INSTATE = 1,  // instate
  FLOATING = 2, // floating
  GAUGE = 3     // chosen to fix gauge freedom
};

class Feature;
using FeaturePtr = Feature *;
class Group;
using GroupPtr = Group *;
class Tracker;
using TrackerPtr = Tracker *;
using Camera = CameraManager;
using CameraPtr = Camera *;
using Config = Json::Value;
class MemoryManager;
using MemoryManagerPtr = MemoryManager *;
class Mapper;
using MapperPtr = Mapper *;
class Estimator;
////////////////////////////////////////
// CUSTOM EXCEPTION
////////////////////////////////////////
struct NotImplemented : public std::exception {
  virtual char const *what() noexcept { return "NOT implemented"; }
};

struct Observation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GroupPtr g;
  Vec2 xp;
  /** The right camera's observation of the same feature at the same frame, when
   *  the stereo matcher found one. Recorded per (feature, group) edge by
   *  `Graph::AddFeatureToGroup`, unlike `Feature::xp_r()` which only holds the
   *  current frame: the out-of-state update revisits a whole track after the
   *  tracker has dropped it, so it needs the history. Left at `has_right =
   *  false` in monocular runs and for frames with no match. */
  bool has_right{false};
  Vec2 xp_r{Vec2::Zero()};
};

using Obs = Observation;

} // namespace xivo
