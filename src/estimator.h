// Inertial-aided Visual Odometry estimator.
// Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <ostream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Eigen/Sparse"
#include "opencv2/core/core.hpp"
#include "json/json.h"

#include "component.h"
#include "core.h"
#include "ekf_update.h"
#include "graph.h"
#include "imu.h"
#include "tracker.h"
#include "visualize.h"
#include "mapper.h"

namespace xivo {

class Estimator;
using EstimatorPtr = Estimator*;
EstimatorPtr CreateSystem(const Json::Value &cfg);
EstimatorPtr CreateSystemTrackerOnly(const Json::Value &cfg);


namespace internal {
class Message {
public:
  Message(const timestamp_t &ts) : ts_{ts} {}
  const timestamp_t &ts() const { return ts_; }
  virtual ~Message() = default;
  virtual void Execute(EstimatorPtr) {}

protected:
  timestamp_t ts_;
};

class Visual : public Message {
public:
  Visual(const timestamp_t &ts, const cv::Mat &img) : Message{ts}, img_{img} {}
  void Execute(EstimatorPtr est);

private:
  cv::Mat img_;
};

/** A synchronized stereo pair. `img_` is the left/primary image -- the same one
 * `Visual` would carry -- so the two paths differ only by the extra right
 * image. */
class VisualStereo : public Message {
public:
  VisualStereo(const timestamp_t &ts, const cv::Mat &img, const cv::Mat &img_r)
      : Message{ts}, img_{img}, img_r_{img_r} {}
  void Execute(EstimatorPtr est);

private:
  cv::Mat img_, img_r_;
};

class VisualTrackerOnly : public Message {
public:
  VisualTrackerOnly(const timestamp_t &ts, const cv::Mat &img) : Message{ts}, img_{img} {}
  void Execute(EstimatorPtr est);

private:
  cv::Mat img_;
};


class VisualPointCloud : public Message {
public:
  VisualPointCloud(const timestamp_t &ts,
                   const VecXi &feature_ids,
                   const MatX3 &xp_and_depths) :
    Message{ts},
    feature_ids_{feature_ids},
    xp_and_depths_{xp_and_depths}
    {}
  void Execute(EstimatorPtr est);

private:
  VecXi feature_ids_;
  MatX3 xp_and_depths_;
};

class VisualPointCloudTrackerOnly : public Message {
public:
  VisualPointCloudTrackerOnly(const timestamp_t &ts,
                              const VecXi &feature_ids,
                              const MatX3 &xp_and_depths) :
    Message{ts},
    feature_ids_{feature_ids},
    xp_and_depths_{xp_and_depths}
    {}
  void Execute(EstimatorPtr est);

private:
  VecXi feature_ids_;
  MatX3 xp_and_depths_;
};

class Inertial : public Message {
public:
  Inertial(const timestamp_t &ts, const Vec3 &gyro, const Vec3 &accel)
      : Message{ts}, gyro_{gyro}, accel_{accel} {}
  void Execute(EstimatorPtr est);

private:
  Vec3 gyro_, accel_;
};

} // namespace internal


class Estimator : public Component<Estimator, State> {
  friend class internal::Visual;
  friend class internal::VisualStereo;
  friend class internal::VisualTrackerOnly;
  friend class internal::VisualPointCloud;
  friend class internal::VisualPointCloudTrackerOnly;
  friend class internal::Inertial;

public:
  static EstimatorPtr Create(const Json::Value &cfg);
  static EstimatorPtr instance();

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ~Estimator();

  void Run();
  // process inertial measurements
  void InertialMeas(const timestamp_t &ts, const Vec3 &gyro, const Vec3 &accel);
  // perform tracking/matching to generate tracks
  void VisualMeas(const timestamp_t &ts_raw, const cv::Mat &img);
  /** Same as `VisualMeas`, for a synchronized stereo pair.
   *
   * Requires a stereo rig to have been configured ("stereo": true); calling it
   * on a monocular system is fatal rather than a silent fall-back to the left
   * image alone, which would look like a merely-disappointing stereo result. */
  void VisualMeasStereo(const timestamp_t &ts_raw, const cv::Mat &img,
                        const cv::Mat &img_r);
  // perform tracking/matching for feature tracker only application
  void VisualMeasTrackerOnly(const timestamp_t &ts_raw, const cv::Mat &img);

  // Perform tracking in Point Cloud World (no images, only features)
  void VisualMeasPointCloud(const timestamp_t &ts,
                            const VecXi &feature_ids,
                            const MatX3 &xp_and_depths);
  void VisualMeasPointCloudTrackerOnly(const timestamp_t &ts,
                                       const VecXi &feature_ids,
                                       const MatX3 &xp_and_depths);


  /** Loop Closure Measurement Update - older features, newer group. */
  void CloseLoop();
  void CloseLoopInternal(GroupPtr g, std::vector<LCMatch>& matched_features);

  // accessors
  SE3 gbc() const { return SE3{X_.Rbc, X_.Tbc}; }
  SE3 gsb() const { return SE3{X_.Rsb, X_.Tsb}; }
  SE3 gsc() const { return gsb() * gbc(); }

  /** The body pose in the *gravity-aligned* spatial frame -- what a consumer of
   *  this filter should be given, and what every other VIO reports.
   *
   *  `gsb` is expressed in `S`, the frame in which `X_.Rsb` starts at identity:
   *  the body frame of the first IMU sample. `S` is not level. The filter knows
   *  by how much -- that is exactly what the two-degree-of-freedom `Wsg` state
   *  is: gravity in `S` is `Rsg * g_`, so `Rsg` maps the gravity-aligned frame
   *  `W` (z along `-g_`, i.e. up) into `S`, and `p_w = Rsg' p_s`. But nothing
   *  ever applied that rotation to the output, so every pose XIVO published was
   *  tilted by the rig's initial attitude relative to gravity.
   *
   *  On TUM-VI room1-room6 that tilt is 0.8-3.0 deg, and it lands in the
   *  orientation error undiminished: the standard evaluation aligns yaw and
   *  position only (`ov_eval ... posyaw`, and every other VIO benchmark, because
   *  yaw and position are the unobservable directions and roll/pitch are *not*),
   *  so a roll/pitch frame offset is not something an evaluator can remove. It
   *  was 60-100% of the reported orientation ATE on those sequences -- room2
   *  reported 3.23 deg against a 3.33 deg initial tilt.
   *
   *  `Rsg` is a state, so this uses the filter's current estimate of it, which
   *  is the causally-available best one. It converges well: against the mocap,
   *  the final `Rsg` is within 0.05 deg (room2) and 0.40 deg (room4) of true
   *  gravity, where the 20-sample accel average it starts from is off by 1.3 and
   *  2.6 deg. `Rsg` carries no yaw by construction (`State::operator+=` zeroes
   *  the third component of its tangent update and reprojects), so this is a
   *  pure levelling and cannot rotate the trajectory about the vertical.
   *
   *  Rotation and translation both, or the result would not be a pose in any
   *  frame. */
  SE3 gwb() const {
    if (!gravity_align_output_) {
      return gsb();
    }
    const SO3 Rws = X_.Rsg.inverse();
    return SE3{Rws * X_.Rsb, Rws * X_.Tsb};
  }
  SE3 gwc() const { return gwb() * gbc(); }
  const State& X() const { return X_; }
  const timestamp_t &ts() const { return curr_time_; }
  MatX P() const {
    MatX out = P_;
    // The motion-to-structure correlation is brought up to date lazily, once
    // per image, so between images the member is stale by a known transition.
    // Apply it to the copy rather than materializing it in `P_`, which would
    // make this accessor a mutation.
    ApplyMotionStructureCorrelation(out);
    return out;
  }
  MatX Pstate() const { return P_.block<9,9>(0,0); }
  MatX CameraCov() const {
#ifdef USE_ONLINE_CAMERA_CALIB
    return P_.block<kMaxCameraIntrinsics,kMaxCameraIntrinsics>(kCameraBegin,
      kCameraBegin);
#else
    // Eigen does not zero-initialise, so this used to return 648 bytes of
    // whatever was on the stack.
    Eigen::Matrix<number_t, 9, 9> all_zeros = Eigen::Matrix<number_t, 9, 9>::Zero();
    return all_zeros;
#endif
  }
  Vec3 Vsb() const { return X_.Vsb; }
  Vec3 bg() const { return X_.bg; }
  Vec3 ba() const { return X_.ba; }
  SO3 Rsg() const { return X_.Rsg; }
  number_t td() const { return X_.td; }
  Mat3 Ca() const { return imu_.Ca(); }
  Mat3 Cg() const { return imu_.Cg(); }
  bool MeasurementUpdateInitialized() const { return MeasurementUpdateInitialized_; }
  int gauge_group() const { return gauge_group_; }
  int num_instate_features() const { return instate_features_.size(); };
  int num_instate_groups() const {return instate_groups_.size(); };
  int num_mh_rejected() const { return num_mh_rejected_; };
  int num_oneptransac_rejected() const { return num_oneptransac_rejected_; };
  int num_tracker_outlier_rejected() const {
    return Tracker::instance()->num_rejected_outliers();
  };
  int num_tracker_failed_to_track() const {
    return Tracker::instance()->num_failed_to_track();
  };
  int num_tracker_new_detections() const {
    return Tracker::instance()->num_new_detections();
  };
  int num_stereo_frames() const {
    return Tracker::instance()->num_stereo_frames();
  };
  int num_stereo_matched() const {
    return Tracker::instance()->num_stereo_matched();
  };
  int num_stereo_attempted() const {
    return Tracker::instance()->num_stereo_attempted();
  };
  int num_stereo_rejected_klt() const {
    return Tracker::instance()->num_stereo_rejected_klt();
  };
  int num_stereo_rejected_epipolar() const {
    return Tracker::instance()->num_stereo_rejected_epipolar();
  };
  int num_stereo_rejected_circular() const {
    return Tracker::instance()->num_stereo_rejected_circular();
  };
  int num_stereo_rejected_disparity() const {
    return Tracker::instance()->num_stereo_rejected_disparity();
  };
  /** Cumulative outcomes of stereo depth seeding; see `StereoSeedDepth`. */
  int num_stereo_init_ok() const { return num_stereo_init_ok_; };
  int num_stereo_init_no_match() const { return num_stereo_init_no_match_; };
  int num_stereo_init_rejected() const { return num_stereo_init_rejected_; };
  int num_stereo_init_rej_degenerate() const {
    return num_stereo_init_rej_degenerate_;
  };
  int num_stereo_init_rej_gap() const { return num_stereo_init_rej_gap_; };
  int num_stereo_init_rej_range() const { return num_stereo_init_rej_range_; };
  int num_stereo_init_rej_std() const { return num_stereo_init_rej_std_; };
  /** Cumulative outcomes of the right-camera EKF rows; see
   * `GateStereoMeasurements`. */
  int num_stereo_upd_used() const { return num_stereo_upd_used_; };
  int num_stereo_upd_rej_geom() const { return num_stereo_upd_rej_geom_; };
  int num_stereo_upd_rej_mh() const { return num_stereo_upd_rej_mh_; };
  MatX3 InstateFeaturePositions(int n_output) const;
  MatX3 InstateFeaturePositions() const;
  MatX6 InstateFeatureCovs(int n_output) const;
  MatX6 InstateFeatureCovs() const;
  MatX3 InstateFeatureXc(int n_output) const;
  MatX3 InstateFeatureXc() const;
  MatX3 InstateFeaturexc() const;
  MatX3 InstateFeaturexc(int n_output) const;
  MatX2 InstateFeaturePreds() const;
  MatX2 InstateFeaturePreds(int n_output) const;
  MatX2 InstateFeatureMeas() const;
  MatX2 InstateFeatureMeas(int n_output) const;
  void InstateFeaturePositionsAndCovs(int max_output, int &npts,
    MatX3 &positions, MatX6 &covs, MatX2 &pixels, VecXi &feature_ids);
  VecXi InstateFeatureIDs(int n_output) const;
  VecXi InstateFeatureIDs() const;
  VecXi InstateFeatureSinds(int n_output) const;
  VecXi InstateFeatureSinds() const;
  VecXi InstateFeatureRefGroups(int n_output) const;
  VecXi InstateFeatureRefGroups() const;
  VecXi InstateGroupIDs() const;
  MatX7 InstateGroupPoses() const;
  MatX InstateGroupCovs() const;
  VecXi InstateGroupSinds() const;
  Mat3 InstateFeatureCov(FeaturePtr f) const;
  Mat6 InstateGroupCov(GroupPtr g) const;
  bool UsingLoopClosure() const;
  bool FeatureCovComparison(FeaturePtr f1, FeaturePtr f2) const;
  bool FeatureCovXYComparison(FeaturePtr f1, FeaturePtr f2) const;
  bool VisionInitialized() const { return vision_initialized_; };
  VecXi JustDroppedFeatureIDs() const;
  void ScaleInitVelocity(number_t scale) { X_.Vsb /= scale; };

  int OOS_update_min_observations() { return OOS_update_min_observations_; }

  // out-of-state (MSCKF) update statistics of the last update step
  int num_oos_candidates() const { return num_oos_candidates_; }
  int num_oos_used() const { return num_oos_used_; }
  int num_oos_rows() const { return num_oos_rows_; }
  int num_oos_gated() const { return num_oos_gated_; }
  int num_oos_bad_triangulation() const { return num_oos_bad_tri_; }

  // returns vector to information about tracked features per instance
  std::vector<std::tuple<int, Vec2, MatXf>> tracked_features();
  std::vector<std::tuple<int, Vec2>> tracked_features_no_descriptor();

  // depth option for simulation
  void InitWithSimDepths() { sim_initialize_depths_ = true; }
  
private:
  void UpdateState(const State::Tangent &dX) { X_ += dX; }

  /** Top-level function for state prediction and update when an IMU packet
   *  arrives */
  void InertialMeasInternal(const timestamp_t &ts, const Vec3 &gyro,
                            const Vec3 &accel);

  /** Top-level function for state prediction and update when an image
   *  packet arrives */
  void VisualMeasInternal(const timestamp_t &ts, const cv::Mat &img);

  /** Stereo counterpart of `VisualMeasInternal`. */
  void VisualMeasStereoInternal(const timestamp_t &ts, const cv::Mat &img,
                                const cv::Mat &img_r);

  /** Top-level function for update when an image packet arrives for
   * feature tracker*/
  void VisualMeasInternalTrackerOnly(const timestamp_t &ts, const cv::Mat &img);

  /** top-level function for update when we receive an update from point-cloud world */
  void VisualMeasPointCloudInternal(const timestamp_t &ts,
                                    const VecXi &feature_ids,
                                    const MatX3 &xp_and_depths);
  void VisualMeasPointCloudInternalTrackerOnly(
    const timestamp_t &ts, const VecXi &feature_ids, const MatX3 &xp_and_depths);

  // initialize gravity with initial stationary samples
  bool InitializeGravity();
  /** Integrates the State `X`. If parameter `visual_meas` is set to `false`, we
   *  update `slope_accel_` and `slope_gyro_`. If `visual_meas` is set to `true`, we
   *  use `slope_accel_` and `slope_gyro` to adjust the last IMU measurement. */
  void Propagate(bool visual_meas);
  /** The EKF measurement update on `P_`, `err_` from `H_`, `inn_`, `diagR_` and
   *  `meas_blocks_`. Takes the cheap symmetric-downdate form, and falls back to
   *  the Joseph form if the innovation covariance has no Cholesky factor; both
   *  live in ekf_update.h. */
  void MeasurementUpdate();
  /** The occupied extent of the error state: the rows and columns of `P_` that an
   *  update has to touch, as at most two contiguous runs (`StateRuns` in core.h).
   *  Everything outside them is exactly uncorrelated with everything else, so the
   *  update leaves it alone whether it visits it or not.
   *
   *  Both allocators take the lowest free slot, so this is a pair of high-water
   *  marks over `gsel_` and `fsel_` -- 135 bool tests, against the ~9 MFLOP of the
   *  update it shrinks. The premise is verified against `P_` and `H_` themselves
   *  inside `EkfUpdateDowndate`, in debug builds and under
   *  `-DXIVO_CHECK_OCCUPIED_STATE`.
   *
   *  With `ekf_update.exact_runs` it is instead the exact set of occupied slots
   *  (`OccupiedStateRunsExact`), which is ~5% smaller in dimension because the
   *  marks cover the vacant slots below them. */
  StateRuns OccupiedState() const;
  /** Predicts measurement (pixels) of features in input. */
  void Predict(std::list<FeaturePtr> &features);
  /** Compute the error-state dynamics Jacobian (the private member `Fdyn_`) at
   *  the given state and measurement. Only the nine rows that have dynamics are
   *  written, because the rest are structurally zero; see `kMotionDynSize`. The
   *  noise Jacobian `G` used to be built here as well and is not built at all
   *  any more -- `AddMotionNoiseCov` produces `G Qimu G'` directly. */
  void ComputeMotionJacobianAt(const State &X,
                               const Eigen::Matrix<number_t, 6, 1> &gyro_accel);
  // only need velocity as the slope for integration
  void ComposeMotion(State &X, const Vec3 &V,
                     const Eigen::Matrix<number_t, 6, 1> &gyro_accel, number_t dt);
  /** perform Fehlberg numerical integration */
  void Fehlberg(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  /** perform Prince-Dormand numerical integration */
  void PrinceDormand(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  /** Perform one-step in Prince-Dormand numerical integration and
   *  return max(slope), i.e., max(V, max(gyro), max(accel)) */
  number_t PrinceDormandStep(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  /** perform vanilla RK4 without step control */
  void RK4(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);
  /** perform one-step in RK4 integration (4 inner steps) */
  void RK4Step(const Vec3 &gyro0, const Vec3 &accel0, number_t dt);

  /** Records one integration step's transition against the motion-to-structure
   *  correlation blocks of `P_` *without touching them*.
   *
   *  `Fdt` is the dynamic rows of the step's transition *minus the identity*:
   *  the step transition is `I + [Fdt; 0]`, which is what the integrators have in
   *  hand (`FK * dt`) before they add the identity to it.
   *
   *  Those blocks are `P[0:24, 24:]` and its transpose -- 24x540, 100 kB each --
   *  and each integration step used to rewrite both. There are ~3 Prince-Dormand
   *  steps per IMU sample and ~10 IMU samples per image, so that is ~30 rewrites
   *  of 200 kB per image, while nothing reads the blocks until the visual update.
   *  Since the effect of n steps is the product `F_n ... F_1` applied once, the
   *  transition is accumulated here and applied by
   *  `FlushMotionStructureCorrelation` before anything can observe it. */
  void AccumulateMotionStructureCorrelation(const MatMotionDyn &Fdt);
  /** Applies what `AccumulateMotionStructureCorrelation` has recorded to `P_`
   *  and resets the accumulator. Called at the end of the propagation that
   *  precedes an image; a no-op if nothing is pending. */
  void FlushMotionStructureCorrelation();
  /** The pending transition applied to an arbitrary covariance, for the const
   *  accessors. No-op if nothing is pending. */
  void ApplyMotionStructureCorrelation(MatX &P) const;

  /** Top-level function for EKF update phase. Calls ProcessTracks, outlier
   *  rejection, EKF measurement update, and bookkeeps features and groups. */
  void UpdateStep(const timestamp_t &ts, std::list<FeaturePtr> &features);

  void ProcessTracks(const timestamp_t &ts, std::list<FeaturePtr> &features);
  void AdaptInitialDepth();
  void EnforceMaxGroupLifetime();
  void DiscardAffectedGroups();
  void SelectAndAddNewFeatures();
  void ZeroGaugeXYAddFeatures();
  void AddFeaturesWithInGroups();
  void AddGroupOfFeatures(int free_group_slots);
  /** Metric depth for a just-created feature from its stereo pair.
   *
   * Returns false -- and leaves the caller to use the monocular prior -- when
   * stereo seeding is disabled, when this feature has no right match on the
   * current frame, or when the triangulation fails any of its gates. The three
   * outcomes are counted separately so a poor seed rate can be attributed. */
  bool StereoSeedDepth(FeaturePtr f, number_t *z, number_t *std_z);
  void InitializeJustCreatedTracks(GroupPtr g,
                                   std::list<FeaturePtr> &tracks);
  void AssociateTrackedFeaturesWithGroup(GroupPtr g,
                                         std::list<FeaturePtr> &tracks);
  void OutlierRejection();
  void FindNewGaugeFeatures();

  /** Decide which right-camera observations enter this frame's EKF update.
   *
   * Runs over `in_current_ekf_update_` immediately before the update -- i.e. on
   * exactly the features whose rows will be assembled -- so it cannot be
   * bypassed by the configurations that skip `MHGating` or `OnePointRANSAC`.
   * Invalidates the right rows of any feature that fails, leaving the feature
   * itself (and its left measurement) untouched. */
  void GateStereoMeasurements();

  /** Computes measurement jacobians for all features in the EKF state. */
  void ComputeInstateJacobians();

  /** Function that contains logic for outlier rejection, filter EKF update, and
   *  filter MSCKF update. It will mark features for removal from the state, but
   *  does not do the actual removing and does not update the graph.
   *  `oos_rows` is the number of rows contributed by the out-of-state features
   *  in `oos_used_` (see `ComputeOOSMeasurements`), which are stacked below the
   *  in-state ones. */
  void FilterUpdate(int oos_rows = 0);

  /** Free up group state slots so that the pose window stays within its budget
   *  and at least `slots_needed` slots are available. Only groups that no
   *  in-state feature refers to can go. */
  void MaintainOOSPoseWindow(int slots_needed);

  /** Triangulates every out-of-state (dropped) feature collected by
   *  `ProcessTracks`, computes its marginalized MSCKF measurement and gates it.
   *  Fills `oos_used_` and returns the total number of measurement rows.
   *  Must run after all group management of this step, so that the state indices
   *  the Jacobians refer to are final. */
  int ComputeOOSMeasurements();

  /** Mahalanobis gate on the marginalized out-of-state measurement of `f`. */
  bool OOSGating(FeaturePtr f);

  /** Makes sure `oos_H_` / `oos_inn_` can hold `rows` rows, keeping the
   *  `num_oos_rows_` already stacked. */
  void ReserveOOSRows(int rows);

  /** Removes the out-of-state features used (or rejected) in this step from the
   *  graph and frees them. */
  void CleanupOOSFeatures();

  /** Main (simple) tool for outlier rejection */
  std::vector<FeaturePtr> MHGating();

  /** Outlier rejection on `Tracker` matches. Always occurs after MH-gating. */
  std::vector<FeaturePtr>
  OnePointRANSAC(const std::vector<FeaturePtr> &ic_matches);
  std::tuple<number_t, bool> HuberOnInnovation(const Vec2 &inn, number_t Rviz);

  void UpdateSystemClock(const timestamp_t &now);

  /** Checks that timestamp `now` (= timestamp of message currently processed) is
   *  at or later than timestamp of last processed message. */
  bool GoodTimestamp(const timestamp_t &now);

  // same as above, but the group list will be untouched
  void RemoveGroupFromState(GroupPtr g);
  void AddGroupToState(GroupPtr g);
  std::vector<FeaturePtr> FindNewOwnersForFeaturesOf(const GroupPtr g);
  void DiscardGroup(const GroupPtr g);
  void DiscardFeatures(const std::vector<FeaturePtr> &discards);
  void DestroyFeatures(const std::vector<FeaturePtr> &destroys);
  void SwitchRefGroup();
  GroupPtr FindNewRefGroup(std::vector<GroupPtr>& candidates);

  // same as above, but the feature list will be untouched
  void RemoveFeatureFromState(FeaturePtr f);
  void AddFeatureToState(FeaturePtr f);

  /** Give a feature that just took a state slot a covariance -- and a
   *  cross-covariance with the rest of the state -- consistent with the poses it
   *  was triangulated from.
   *
   *  The default is `Feature::FillCovarianceBlock`, which copies the depth
   *  sub-filter's 3x3 and zeroes every cross term. The sub-filter treats both the
   *  reference and the current pose as exact (see `Feature::SubfilterUpdate`), so
   *  that 3x3 is a *conditional* covariance and the zeros claim the feature is
   *  independent of the very group it is anchored to. Both make the filter
   *  over-confident about the geometry from the first frame a feature is used.
   *
   *  This instead does the standard delayed-initialization augmentation from the
   *  feature's stacked measurement over its in-state views (see
   *  `Feature::ComputeInitJacobian`):
   *
   *      P_ff = Hl^-1 (sigma^2 I + Hx P Hx') Hl^-T
   *      P_xf = -P Hx' Hl^-T
   *      x   += Hl^-1 res
   *
   *  Returns false and leaves the covariance untouched when it cannot be done
   *  (no parallax, anchor not in the state yet, too few in-state views), in which
   *  case the caller falls back to `FillCovarianceBlock`. */
  bool InitializeFeatureCovariance(FeaturePtr f);

  void AbsorbError(const VecX &err); // absorb error state into nominal state
  void AbsorbError();                // absorb error state into nominal state
  // helpers
  void PrintErrorStateNorm();
  void PrintErrorState();
  void PrintNominalState();

  void BackupState(std::unordered_set<FeaturePtr>& features,
                   std::unordered_set<GroupPtr>& groups);
  void RestoreState(std::unordered_set<FeaturePtr>& features,
                    std::unordered_set<GroupPtr>& groups);

  void FixFeatureXY(FeaturePtr f);
  /** Propagates a re-anchoring through the filter covariance: P <- S P S^T,
   *  where S is the identity except for this feature's three rows, which pick up
   *  `jac.dxn_dx` at the feature's own block and `jac.dxn_dref_{old,new}` at the
   *  two groups'. `scale` multiplies all three (i.e. inflates the feature block
   *  by scale^2), which is how `feature_owner_change_cov_factor` is applied. */
  void ReanchorFeatureCovariance(FeaturePtr f, GroupPtr old_ref, GroupPtr new_ref,
                                 const Feature::ReanchorJacobians &jac,
                                 number_t scale);

private:
  Estimator(const Json::Value &cfg);
  static std::unique_ptr<Estimator> instance_;

private:
  std::vector<FeaturePtr> instate_features_; ///< in-state features
  std::vector<FeaturePtr> oos_features_;     ///< out-of-state features
  std::vector<GroupPtr> instate_groups_;     ///< in-state groups

  // For feature and group management
  std::unordered_set<GroupPtr> affected_groups_; // lost a feature and might need to be tossed
  std::vector<GroupPtr> needs_new_gauge_features_;
  std::vector<FeaturePtr> new_features_;
  std::vector<FeaturePtr> inliers_;
  std::vector<FeaturePtr> in_current_ekf_update_;

  /** Index of the current gauge group. It is set to -1 when we lose the current
   *  gauge group while calling `UpdateStep`. */
  int gauge_group_;
  GroupPtr gauge_group_ptr_;

  /** Number of degrees of freedom fixed. 6 = "correct" if we we are estimating
   *  the direction of gravity. 4 = "correct" if we are pretty sure that we can
   *  get an accurate direction of gravity at initialization and just need a
   *  bit of wiggle room. (Corvis always fixes 4 no matter what.)
  */
  int group_degrees_fixed_;

  /** Number of features to hold as gauge features in each group.
   *  3 = "correct" */
  int num_gauge_xy_features_;

private:
  Config cfg_;        // this is just a reference of the global parameter server
  bool simulation_;   // estimator used in simulation or not
  bool use_canvas_;   // visualization or not
  bool print_timing_; // show timing info
  std::string integration_method_; ///< motion integration numerical scheme

  /** Whether or not to sue 1-pt RANSAC in outlier rejection. */
  bool use_1pt_RANSAC_;
  number_t ransac_thresh_, ransac_prob_, ransac_Chi2_;

  /** Whether or not to use MSCKF measurement update */
  bool use_OOS_;
  bool use_compression_;            // measurement compression
  number_t compression_trigger_ratio_; // use measurement compression, if the ratio
                                    // of columns/rows of measurement matrix is
                                    // above this level
  /** Minimum number of observations a feature needs by the time `Tracker` drops
   *  it in order to use it in a MSCKF update. */
  int OOS_update_min_observations_;
  OOSOptions oos_options_;          // out-of-state (MSCKF) update options
  bool use_depth_opt_;              // use depth optimization or not
  RefinementOptions refinement_options_; // depth refinement options
  SubfilterOptions subfilter_options_;   // depth-subfilter options
  bool triangulate_pre_subfilter_; // depth triangulation before depth subfilter
  TriangulateOptions triangulate_options_;
  AdaptiveInitialDepthOptions adaptive_initial_depth_options_;

  /** Minimum number of steps a feature is an outlier before it is removed */
  int remove_outlier_counter_;

  /** The current state estimate. Contains nominal state and calibrations, but no
   *  feature positions. */
  State X_;
  /** Backup of the current state estimate. Used with `BackupState` and
   *  `RestoreState` in 1-pt RANSAC calculations. */
  State X0_;
  /** Filter's error state: Contains both pose and feature positions. */
  VecX err_;
  /** Whether or not each group is in-state */
  std::array<bool, kMaxGroup> gsel_;
  /** Whether or not each feature is in-state */
  std::array<bool, kMaxFeature> fsel_;
  /** `ekf_update.exact_runs`: describe the live extent by the occupied slots
   *  themselves rather than by two high-water marks. */
  bool exact_state_runs_{false};
  /** `ekf_update.run_gap`: how many provably-zero dimensions the exact extent may
   *  absorb rather than split a run in two. */
  int state_run_gap_{kGroupSize};
  /** `ekf_update.fuse_passes`: form `H P` and `M H^T` in one pass over the
   *  destination per row block instead of one per column run of the Jacobian. */
  bool fuse_update_passes_{false};
  /** `ekf_update.chunks`: how many consecutive groups the update's rows are split
   *  into and applied one after another. 1 is the batch update. See
   *  `EkfUpdateDowndate` for why the sequence is the same update and what it buys. */
  int update_chunks_{1};
  /** Data and operators for IMU calibration variables `Ca` and `Cg` */
  IMU imu_;
  /** Current estimate of the gravity vector resolved in the reference frame. */
  Vec3 g_;

  // measurement noise

  /** The initial depth value given to new features when they are first created. It is
   *  updated at every frame to be (almost) equal to the median depth of all the
   *  features currently in the state. (i.e. `init_z = 0.01*init_z + 0.99*median_depth`) */
  number_t init_z_;
  /** Default subfilter covariance for each feature's (X/Z)-coordinate at initialization.
   *  (Subfilter covariance is initialized as a diagonal.) */
  number_t init_std_x_;
  /** Default subfilter covariance for each feature's (Y/Z)-coordinate at initialization.
   *  (Subfilter covariance is initialized as a diagonal.) */
  number_t init_std_y_;
  /** Default subfilter covariance for each feature's (1/Z)-coordinate at initialization.
   *  (Subfilter covariance is initialized as a diagonal.) */
  number_t init_std_z_;
  /** Default subfilter covariance for each feature's (X/Z)-coordinate at
   *  initialization when triangulation is poor.*/
  number_t init_std_x_badtri_;
  /** Default subfilter covariance for each feature's (Y/Z)-coordinate at
   *  initialization when triangulation is poor.*/
  number_t init_std_y_badtri_;
  /** Default subfilter covariance for each feature's (1/Z)-coordinate at
   *  initialization when triangulation is poor.*/
  number_t init_std_z_badtri_;

  ////////////////////////////////////////
  // Stereo depth initialization (M4)
  ////////////////////////////////////////
  /** Whether to seed a new feature's depth by triangulating its stereo pair
   *  instead of using `init_z_`. Off unless `stereo_init.enable` is set. */
  bool stereo_init_{false};
  /** Assumed left->right matching error, in pixels. Sets how tight the seeded
   *  log-depth covariance is; see `StereoRig::TriangulateFromPixels`. */
  number_t stereo_init_sigma_px_;
  /** Reject a triangulation whose two rays miss each other by more than this
   *  many metres. A large gap means the match is inconsistent with the rig
   *  geometry even if it passed the tracker's epipolar gate. */
  number_t stereo_init_max_gap_;
  /** Clamp on the seeded log-depth std. The floor stops a very close, very
   *  well-conditioned feature from being seeded so confidently that the filter
   *  cannot correct a calibration error; the ceiling means "no better than the
   *  monocular prior", at which point there is no reason to prefer stereo. */
  number_t stereo_init_min_std_z_;
  number_t stereo_init_max_std_z_;
  /** Let the two-frame temporal triangulation overwrite a stereo-seeded depth.
   *
   * Off by default, and that default is the whole point: `Feature::Triangulate`
   * rewrites `x_` without touching `P_`, so allowing it pairs a temporal depth
   * with the stereo's covariance. Exposed as a knob only so the choice stays
   * measurable without a rebuild; see notes-stereo/m4-stereo-depth-init.md. */
  bool stereo_init_allow_retriangulation_{false};
  /** Diagnostics: how often the stereo seed was used vs fell back. The four
   *  `rej_` counters partition `num_stereo_init_rejected_` by cause, which is
   *  what makes a low seed rate attributable rather than merely visible. */
  int num_stereo_init_ok_{0};
  int num_stereo_init_no_match_{0};
  int num_stereo_init_rejected_{0};
  int num_stereo_init_rej_degenerate_{0};
  int num_stereo_init_rej_gap_{0};
  int num_stereo_init_rej_range_{0};
  int num_stereo_init_rej_std_{0};

  ////////////////////////////////////////
  // Stereo EKF measurement update (M5)
  ////////////////////////////////////////
  /** Whether an in-state feature's right-camera observation contributes two
   *  extra rows to the EKF measurement. Off unless `stereo_update.enable`. */
  bool stereo_update_{false};
  /** Measurement variance of a right-camera pixel, as a multiple of `R_`.
   *
   *  Kept as a *ratio* rather than an absolute variance so that re-tuning `R_`
   *  keeps the two cameras' relative weighting intact. The default of 1 says
   *  the right observation is exactly as trustworthy as the left, which is the
   *  honest starting point for a hardware-synchronized pair of identical
   *  sensors tracked by the same KLT; a value > 1 discounts it for the extra
   *  error the left->right match adds on top of the temporal track. */
  number_t stereo_update_R_scale_{1.0};
  /** Threshold of the right-camera Mahalanobis gate, as a multiple of
   *  `MH_thresh_`. The gate is deliberately *separate* and 2-dof rather than
   *  folded into a 4-dof joint distance: the existing threshold is calibrated
   *  for 2 dof, and a bad right match should cost the feature its right
   *  measurement, not its place in the state. */
  number_t stereo_update_mh_scale_{1.0};
  /** Diagnostics, cumulative over the run: right measurements actually used,
   *  and those dropped by the geometric check inside
   *  `Feature::ComputeRightJacobian` or by the right MH gate. */
  int num_stereo_upd_used_{0};
  int num_stereo_upd_rej_geom_{0};
  int num_stereo_upd_rej_mh_{0};

  /** The minimum depth that a feature can be given when it is first
   *  created. (i.e. minimum value of `init_z_`) */
  number_t min_z_;
  /** The maximum depth that a feature can be given when it is first
   *  created. (i.e. maximum value of `init_z_`) */
  number_t max_z_;

  /** Error state dynamics Jacobian; used for the covariance update in the EKF's
   *  prediction step.
   *
   *  Nine rows, not twenty-four (`kMotionDynSize`), and dense rather than an
   *  `Eigen::SparseMatrix`. It used to hold ~57 nonzeros in a 24x24 sparse matrix
   *  that `ComputeMotionJacobianAt` rebuilt from `setZero()` plus ~57
   *  `coeffRef` calls -- i.e. ~57 insertions into a compressed structure -- seven
   *  times per integration step, ~30 steps per image; and every use of it was a
   *  product against a dense 24x24 anyway. At 9x24 the whole Jacobian is 1.7 kB
   *  and every product is fixed-size.
   *
   *  The *transition* `I + Fdyn_ dt` is no longer stored at all: it was written
   *  back over `F_` at the end of each step, and its only consumer
   *  (`AccumulateMotionStructureCorrelation`) now takes `Fdyn_ dt` and adds the
   *  identity implicitly. */
  MatMotionDyn Fdyn_;
  /** The product of every step transition since the motion-to-structure
   *  correlation blocks of `P_` were last brought up to date, and whether that
   *  product is anything other than the identity. See
   *  `AccumulateMotionStructureCorrelation`. Rows at and below `kMotionDynSize`
   *  are exactly rows of the identity, forever, which is what
   *  `ApplyMotionTransition` relies on. `Fcross_scratch_` exists only so the
   *  accumulating product does not alias its own destination. */
  MatMotion Fcross_;
  MatMotionDyn Fcross_scratch_;
  bool Fcross_pending_;
  /** Filter covariance. Size grows and shrinks with the number of tracked
   *  features. */
  MatX P_;
  /** Backup of filter covariance. Used with `BackupState` and `RestoreState`
   *  in 1-pt RANSAC */
  MatX P0_;
  /** Filter motion covariance. Size is `kMotionSize` x `kMotionSize` */
  MatX Qmodel_;
  /**
   * Filter IMU measurement covaraince, made up of four 3x3 blocks for a total
   * dimention of 12 x 12. The four blocks correspond to the gyro,
   * accelerometer, gyro bias, and accelerometer bias, measurements,
   * respectively. */
  MatX Qimu_;

  // for clamping signals
  bool clamp_signals_;
  Vec3 max_gyro_;
  Vec3 max_accel_;

  // for update

  /** Set to true once update has been initialized */
  bool MeasurementUpdateInitialized_;
  /** Filter measurement Jacobian */
  MatX H_;
  /** The row blocks of `H_` and their sparsity, filled alongside it. See
   *  `MeasBlock` in ekf_update.h: recording which slots each block belongs to is
   *  what lets `H P` be formed from the 25 columns per measurement that can be
   *  nonzero. */
  std::vector<MeasBlock> meas_blocks_;
  /** Filter innovation */
  VecX inn_;
  /** Diagonal of visual feature measurement covariance used in the filter.
   *  (We assume that the feature measurement covariance is diagonal, as opposed
   *  to just positive semi-definite.) Its size is 2*(number of in-state inliers) +
   *  (MSCKF measurement size) */
  VecX diagR_;
  /** The filter's (assumed) measurement covariance of every element of each MSCKF
   *  measurement. */
  number_t Roos_;
  /** The filter's (assumed) measurement covariance of x and y pixel measurement for
   *  in-state tracked features. */
  number_t R_;
  number_t Rtri_;           // UNUSED? measurement covariance, depth sub-filter
  number_t Rlc_;            // Loop closure measurement covariance
  number_t outlier_thresh_; // outlier threshold -- multipler of the measurement
                         // variance

  // MH (Mahalanobis) gating parameters
  bool use_MH_gating_;
  number_t MH_thresh_;           // MH threshold
  int min_required_inliers_;  // minimal inliers needed to perform update
  number_t MH_thresh_multipler_; // if not enough inliers, repeatedly multiple the
                              // MH_thresh by this amount
  /** How many *consecutive* MH-gate failures destroy an in-state feature. 1 is
   *  the original policy (destroy on the first failure). Larger values let a
   *  feature skip the update for a frame and stay in the state -- see
   *  `Feature::mh_strikes()`. */
  int MH_max_strikes_;
  /** In-state features that failed the MH gate this frame but were kept because
   *  they had strikes to spare. They are not in `inliers_`, so they contribute no
   *  rows to this update, but their state slots are still occupied. */
  int num_mh_deferred_{0};

  /** Consistent feature initialization; see `InitializeFeatureCovariance`. */
  bool consistent_init_{false};
  int consistent_init_min_views_{2};
  number_t consistent_init_R_{1.0};
  number_t consistent_init_max_var_{1e4};
  int num_consistent_init_{0}, num_consistent_init_failed_{0};
  /** Scratch for the compacted form of `InitializeFeatureCovariance`: the
   *  `kFullSize x runs.dim` slice of `P_` whose columns the init Jacobian can
   *  reach. A member, not a local, because that call happens ~6 times per frame
   *  and this is the one allocation in it that is not small; grown monotonically
   *  and never shrunk, so a run settles on one buffer of ~160 kB. Only meaningful
   *  inside that function. */
  MatX init_cov_Pcols_;

  // time
  timestamp_t last_imu_time_, curr_imu_time_; // time when the imu meas arrives
  timestamp_t last_vision_time_,
      curr_vision_time_;  // time when visual meas arrives
  timestamp_t curr_time_; // current system time
  timestamp_t last_time_; // last measurement time, either imu or visual

  // Zero-initialized, because `Propagate` reads all six before any of them is
  // necessarily written: the visual-measurement branch extrapolates
  // `last_{accel,gyro}_` along `slope_{accel,gyro}_`, and the slopes are only
  // ever assigned on the IMU branch. If the first measurement to reach the
  // filter is an image, that read used to be of whatever
  // -DEIGEN_INITIALIZE_MATRICES_BY_ZERO had left behind.
  Vec3 curr_accel_{Vec3::Zero()}, curr_gyro_{Vec3::Zero()}; // current gyro and accel measurement
  Vec3 last_accel_{Vec3::Zero()}, last_gyro_{Vec3::Zero()}; // accel & gyro measurement at last_time
  Vec3 slope_accel_{Vec3::Zero()}, slope_gyro_{Vec3::Zero()};

  bool gravity_initialized_, vision_initialized_;
  int imu_counter_, vision_counter_;
  int strict_criteria_timesteps_;

  // For simulation
  bool sim_initialize_depths_;
  std::unordered_map<int, number_t> ids_to_depths_;

  // How much to inflate covariance of features after a group ownership change
  number_t feature_owner_change_cov_factor_;

  // helpers
  int gravity_init_counter_;
  std::vector<Vec3> gravity_init_buf_; // buffer of accel measurements for
                                       // gravity initialization
  /** Gyro and timestamps alongside `gravity_init_buf_`, used only when
   *  `gravity_init_derotate_` is on. */
  std::vector<Vec3> gravity_init_gyro_buf_;
  std::vector<timestamp_t> gravity_init_time_buf_;
  /** Rotate each buffered accel sample into the body frame of the *last* sample
   *  before averaging, integrating the gyro to get the relative attitude.
   *
   * `InitializeGravity` calls its buffer "stationary accel samples", but on
   * TUM-VI's room sequences the rig is already turning at 0.11-0.32 rad/s when
   * the first sample lands. Averaging body-frame accelerations across a turn
   * smears the gravity direction by roughly |w| * window, which puts a hard
   * ceiling on how long the window can usefully be -- and a short window cannot
   * average away the carrier's own linear acceleration. De-rotating removes the
   * smearing, so the window can grow until the linear acceleration averages out.
   * Measured initial tilt error, mean over room1-room6: 1.47 deg as shipped
   * (20 samples, no de-rotation) against 0.73 deg de-rotated over 200.
   * See notes-stereo/m6-attitude-initialization.md. */
  bool gravity_init_derotate_{false};
  /** Publish poses in the gravity-aligned frame rather than in the initial body
   *  frame; see `gwb`. Defaults on -- it is the convention every consumer and
   *  every evaluator assumes -- and touches nothing the filter reads, so the
   *  estimate itself is bit-identical either way. */
  bool gravity_align_output_{true};
  // measurements buffer
  struct InternalBuffer
      : public std::vector<std::unique_ptr<internal::Message>> {
#ifdef MESSAGE_BUFFER_SIZE
    static constexpr int MAX_SIZE = MESSAGE_BUFFER_SIZE;
#else
    static constexpr int MAX_SIZE = 10;
#endif
    InternalBuffer() : initialized{false} {}
    std::mutex mtx;
    bool initialized;
  } buf_;
  bool async_run_; // if true, run in a separate thread
  void MaintainBuffer();

  own<std::thread *> worker_;
  /** Set by the destructor to break the worker's loop. Without it, `Run()`'s
   *  `for (;;)` never returned and `~Estimator`'s `worker_->join()` deadlocked,
   *  so destroying an estimator with `async_run: true` hung the process. */
  std::atomic<bool> stop_{false};

  /** Computes the running average time for dynamics propatagion, visual measurement
   *  processing, tracker, update tracker, jacobian, MH gating. (Those quantities
   *  overlap). */
  Timer timer_;

  /** Occupancy and measurement-size census, accumulated over the run and printed
   *  next to the timing block under `print_timing`.
   *
   *  The cost of the update is cubic in the *dimension* of the error state and
   *  linear in the number of measurement rows, and both are decided at run time
   *  by how many slots are actually occupied -- not by the compile-time capacity.
   *  `P_` is nonetheless always the full `kFullSize` square, with vacated slots
   *  zeroed rather than excluded, so knowing the gap between occupancy and
   *  capacity is what decides whether compacting the active set is worth
   *  anything. Counted rather than assumed, because the answer differs between
   *  the monocular and stereo settings. */
  struct Census {
    long frames{0};      ///< frames that reached the census point
    long updates{0};     ///< frames that ran an EKF update
    long feat_slots{0};  ///< occupied feature slots, summed over frames
    long group_slots{0}; ///< occupied group slots, summed over frames
    long update_feats{0};///< features in the update, summed over updates
    long rows{0};        ///< measurement rows, summed over updates
    long right_rows{0};  ///< of which right-camera rows
    long oos_rows{0};    ///< of which out-of-state rows
    /** Calls to `MeasurementUpdate` -- more than one per frame, since the
     *  1-pt RANSAC and loop-closure paths each run their own. */
    long live_updates{0};
    /** `OccupiedState().dim` summed over those calls: the dimension the update
     *  *actually* ran on, which is what the occupancy above buys once it is
     *  rounded up to whole runs. Strictly between `occupied-dim` and
     *  `kFullSize`. */
    long live_dim{0};
    long live_runs{0};   ///< `OccupiedState().nruns` summed over those calls
  } census_;
  void PrintCensus(std::ostream &os) const;

  std::unique_ptr<std::default_random_engine> rng_;


  /** Ids of features that were just dropped by the tracker or thrown out by
   * outlier rejection.
  */
  std::vector<int> just_dropped_feature_ids_;

  /** Keep track of features rejected by MH-Gating and One-Pt RANSAC. */
  int num_mh_rejected_;
  int num_oneptransac_rejected_;

  /** Length of the sliding pose window kept in the state for the out-of-state
   *  update (0 = no window, i.e. groups enter the state only as reference groups
   *  of features being promoted), and how often a pose is added to it (1 = every
   *  frame; k > 1 makes the same number of slots span k times as much time). */
  int oos_pose_window_{0};
  int oos_augment_every_{1};

  /** One accepted out-of-state measurement, copied out of the shared
   *  `Feature::oos_result()` as soon as it passes the gate.
   *
   *  The rows have to outlive the loop that computes them -- `FilterUpdate` stacks
   *  them below the in-state ones, and the in-state row count is not known until
   *  then -- but they do not have to outlive it inside the `Feature`. Holding them
   *  here instead is what lets the marginalized Jacobian be one shared buffer
   *  rather than one per pooled feature; see `Feature::oos_result()` and
   *  notes-oosfast/m4-memory.md. */
  struct OOSRowBlock {
    int row;      ///< first row in `oos_H_` / `oos_inn_`
    int rows;     ///< number of rows
    RunSet runs;  ///< the columns they can be nonzero in (empty when `oos_fast` is off)
  };
  std::vector<OOSRowBlock> oos_blocks_;
  /** The accepted rows, stacked. Grown geometrically and never shrunk; the row
   *  capacity settles at the largest number of out-of-state rows any one update
   *  produced, a few hundred kB. */
  MatX oos_H_;
  VecX oos_inn_;

  /** Out-of-state (MSCKF) update bookkeeping, per update step. */
  std::vector<FeaturePtr> oos_used_; ///< features actually used in the update
  int num_oos_candidates_{0};        ///< dropped out-of-state tracks
  int num_oos_used_{0};
  int num_oos_short_{0};   ///< rejected: too few in-state observations
  int num_oos_bad_tri_{0}; ///< rejected: triangulation gate
  int num_oos_gated_{0};   ///< rejected: Mahalanobis gate
  int num_oos_rows_{0};    ///< measurement rows contributed
  /** Same counters, accumulated over the whole run (for the run summary). */
  long total_oos_candidates_{0}, total_oos_used_{0}, total_oos_short_{0},
      total_oos_bad_tri_{0}, total_oos_gated_{0}, total_oos_rows_{0},
      total_oos_obs_{0};
  /** Of `total_oos_obs_` views, how many also contributed the right camera's
   *  two rows -- the one number that says whether stereo out-of-state rows are
   *  actually firing. */
  long total_oos_right_obs_{0};
  /** Observation coverage of the candidates: how many observations they have in
   *  total versus how many come from groups that are actually in the state (the
   *  only ones an OOS update can constrain). `oos_instate_view_hist_[k]` counts
   *  candidates with k in-state views, the last bin being a catch-all. */
  long total_oos_views_all_{0}, total_oos_views_instate_{0};
  std::array<long, 18> oos_instate_view_hist_{};
  /** Pose-window bookkeeping: groups evicted to keep it within budget, and
   *  frames whose pose could not be added because every slot was load bearing. */
  long num_oos_window_evictions_{0}, num_oos_window_starved_{0};
};

} // xivo
