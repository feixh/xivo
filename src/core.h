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

////////////////////////////////////////
// THE OCCUPIED EXTENT OF THE STATE
////////////////////////////////////////
/** The error-state indices an update has to touch, as a list of contiguous runs.
 *
 *  The covariance is always the full `kFullSize` square, but a group or feature
 *  slot that is not occupied is *exactly uncorrelated* with everything else:
 *  `RemoveGroupFromState` and `RemoveFeatureFromState` zero the slot's whole row
 *  and column, `Feature::FillCovarianceBlock` does the same before writing its own
 *  block, and the propagation writes only the motion block and the two
 *  motion-to-structure correlation blocks. A vacant slot does keep a *variance* --
 *  `P_` is initialized to a scaled identity, so an untouched slot sits at 1 -- but
 *  it has no cross terms, and nothing about it is approximate.
 *
 *  Uncorrelated is all the update needs. With `L` the live set, `H` zero outside
 *  it (a measurement cannot reference an unoccupied slot) and `P(i, ·)` supported
 *  on `{i}` for `i` outside it, column `i` of `M = H P` is `H P e_i = P(i,i)
 *  H(:,i) = 0`; so `S = M H^T` is unchanged, `err_i = M(:,i)^T u` is zero, and the
 *  downdate `-W^T W` contributes nothing to any entry in row or column `i`. The
 *  vacant part of `P` comes out of the update exactly as it went in, which is what
 *  the dense form does too. Skipping it is a rearrangement, not an approximation.
 *
 *  It is worth skipping because the census (M0) found 7.3 of 45 group slots and 76
 *  of 90 feature slots occupied on TUM-VI: 47% of the 564 dimensions are inert,
 *  and the update's cost is quadratic in the dimension.
 *
 *  Both allocators take the *lowest* free slot (`AddGroupToState` and
 *  `AddFeatureToState` scan from 0), so the occupied slots are packed toward
 *  index 0 and the occupied region is described by two high-water marks rather
 *  than by a set of 135 bits. That is what keeps this to two runs -- the motion
 *  block and the group region are adjacent (`kGroupBegin == kCameraBegin +
 *  kMaxCameraIntrinsics`), so they open a single run, and the features follow
 *  after the unused group slots.
 *
 *  Runs are a covering, not a characterization: a run may contain vacant slots
 *  (a feature slot freed below the high-water mark) and including them costs
 *  arithmetic but changes no result. Excluding a live one would corrupt the
 *  filter, so this errs in the safe direction by construction.
 *
 *  Runs are ascending and disjoint; `MirrorLowerTriangle` relies on the order to
 *  know which block of a pair is below the diagonal. */
constexpr int kMaxStateRuns = 2;

struct StateRuns {
  ColRun runs[kMaxStateRuns];
  int nruns; ///< 1 or 2
  int dim;   ///< total length of the runs
};

/** The occupied extent, given one past the highest occupied group slot and one
 *  past the highest occupied feature slot (`Estimator::OccupiedState`). */
inline StateRuns OccupiedStateRuns(int groups_used, int features_used) {
  StateRuns s{};
  // Motion, plus the camera-intrinsics block if this build has one, is always
  // live and abuts the group region.
  s.runs[0] = {0, kGroupBegin + kGroupSize * groups_used};
  s.nruns = 1;
  const int fend = kFeatureBegin + kFeatureSize * features_used;
  if (features_used > 0) {
    if (s.runs[0].len == kFeatureBegin) {
      s.runs[0].len = fend; // every group slot is occupied: one run, not two
    } else {
      s.runs[1] = {kFeatureBegin, kFeatureSize * features_used};
      s.nruns = 2;
    }
  }
  s.dim = 0;
  for (int i = 0; i < s.nruns; ++i) {
    s.dim += s.runs[i].len;
  }
  return s;
}

/** The whole state as one run -- the extent that skips nothing. For callers with
 *  no slot bookkeeping, and for the tests, where it is the reference the
 *  occupied extent has to agree with. */
inline StateRuns WholeState() {
  StateRuns s{};
  s.runs[0] = {0, kFullSize};
  s.nruns = 1;
  s.dim = kFullSize;
  return s;
}

constexpr int kStructureSize = kFullSize - kMotionSize;
using MatMotion = Eigen::Matrix<number_t, kMotionSize, kMotionSize>;

/** Propagates the motion-to-structure correlation of `P` through one motion
 *  transition `F`: `P[0:24, 24:] <- F P[0:24, 24:]`, and the lower block to the
 *  transpose of the result.
 *
 *  The integrators used to do this inline, once per substep, rewriting both
 *  blocks -- 24x540, ~100 kB each -- ~30 times per image even though nothing
 *  reads them until the visual update. They now accumulate `F` across substeps
 *  and call this once per image instead; because the transition is linear,
 *  `F_n (... (F_1 P)) == (F_n ... F_1) P` exactly in algebra, and the 24x24
 *  products that replace them are free by comparison.
 *
 *  Mirroring the upper block into the lower one, rather than forming
 *  `P[24:, 0:24] F^T` on its own as the old code did, halves the work. It does
 *  not change the numbers: the two products sum the same terms in the same order
 *  inside Eigen's gemm, so the old form came out bit-for-bit symmetric as well
 *  (`unitTests_propagate_cov` pins that). The difference is that the mirror is
 *  symmetric by construction rather than by a property of the gemm kernel.
 *
 *  A free function rather than an `Estimator` member so the test can drive it
 *  with an arbitrary `P` and compare against the per-step reference. */
template <typename FDerived>
inline void ApplyMotionTransition(MatX &P, const Eigen::MatrixBase<FDerived> &F) {
  P.block<kMotionSize, kStructureSize>(0, kMotionSize) =
      F * P.block<kMotionSize, kStructureSize>(0, kMotionSize);
  P.block<kStructureSize, kMotionSize>(kMotionSize, 0) =
      P.block<kMotionSize, kStructureSize>(0, kMotionSize).transpose();
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
