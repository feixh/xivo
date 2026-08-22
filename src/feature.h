// The feature class.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <functional>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "component.h"
#include "core.h"
#include "jac.h"
#include "options.h"
#include "project.h"
#include "fastbrief.h"

namespace xivo {


/** Unordered map: group id -> observed pixel coordinates */
struct FeatureAdj : public std::unordered_map<int, Vec2> {
  void Add(const Observation &obs);
  void Remove(int id);
};


/** Track is a C++ <vector> containing all the (x,y) pixel detections found by
 *  the `Tracker` over a set of consecutive images paired with some metadata.
 */
class Track : public std::vector<Vec2, Eigen::aligned_allocator<Vec2>> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Track() : status_(TrackStatus::CREATED) {}
  Track(number_t x, number_t y) { Reset(x, y); }

  /** Deletes the entire history of tracks and starts a new vector. */
  void Reset(number_t x, number_t y) {
    clear();
    status_ = TrackStatus::CREATED;
    push_back(Vec2(x, y));
    // Features are recycled out of MemoryManager's pool (see Feature::Create,
    // which calls Reset on a slot that previously belonged to a different
    // feature), so this is the only chance to drop the previous tenant's data.
    // Leaving it behind meant a new track inherited the *previous* track's
    // descriptor history: `descriptor()` returns descriptors_.back(), so until
    // the first SetDescriptor call it handed out a descriptor belonging to an
    // unrelated feature, GetAllDescriptors()/GetAllDBoWDesc() mixed the two
    // tracks for the rest of the new feature's life, and the history grew for
    // the whole run instead of per track. Same for the keypoint.
    descriptors_.clear();
    keypoint_ = cv::KeyPoint();
  }

  TrackStatus status() const { return status_; }
  void SetStatus(TrackStatus status) { status_ = status; }
  /** Stores a *copy* of the descriptor.
   *
   *  Every caller passes `all_descriptors.row(i)`, which is a view sharing the
   *  per-frame descriptor matrix OpenCV filled in. Keeping the view keeps that
   *  whole matrix -- 110-270 keypoints of it -- alive for one 32-byte row, so
   *  holding on to a handful of rows pinned megabytes.
   */
  void SetDescriptor(const cv::Mat &descriptor) {
    descriptors_.push_back(descriptor.clone());
  }
  void SetKeypoint(const cv::KeyPoint &keypoint) { keypoint_ = keypoint; }
  const cv::KeyPoint &keypoint() const { return keypoint_; }
  cv::KeyPoint &keypoint() { return keypoint_; }
  bool has_descriptor() const { return !descriptors_.empty(); }
  const cv::Mat &descriptor() const {
    CHECK(!descriptors_.empty()) << "track has no descriptor";
    return descriptors_.back();
  }
  cv::Mat &descriptor() {
    CHECK(!descriptors_.empty()) << "track has no descriptor";
    return descriptors_.back();
  }
  const std::vector<cv::Mat>& GetAllDescriptors() { return descriptors_; }
  FastBrief::TDescriptor GetDBoWDesc();
  std::vector<FastBrief::TDescriptor> GetAllDBoWDesc();

protected:
  /** CREATED, TRACKED, REJECTED, or DROPPED */
  TrackStatus status_;

  /** OpenCV Keypoint from when this track was first detected in `Tracker::Detect()` */
  cv::KeyPoint keypoint_;

  /** Descriptor of all observations. */
  std::vector<cv::Mat> descriptors_;
};


/** All the data associated with a single tracked feature.
 *  Essentially, the `Track` class plus
 *  - estimate of current 3D position with respect to the global reference frame
 *  - subfiltering and triangulation functions to accurately estimate depth
 *  - Functions to compute Jacobians for the `Estimator` class's measurement update.
 */
class Feature : public Component<Feature, Vec3>, public Track {
  template<typename Feature> friend class CircBufWithHash;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static FeaturePtr Create(number_t x, number_t y);
  static FeaturePtr PointCloudWorldCreate(int fid, number_t x, number_t y);
  static void Deactivate(FeaturePtr f);
  static void Destroy(FeaturePtr f);

  /** Appends another point to vector of observations.
   *  Recall: (`Feature` << `Track` << `std::vector` */
  void UpdateTrack(number_t x, number_t y) { emplace_back(x, y); }
  /** Appends another point to vector of observations.
   *  Recall: (`Feature` << `Track` << `std::vector` */
  void UpdateTrack(const Vec2 &pt) { UpdateTrack(pt(0), pt(1)); }

  /** Returns whether or not the feature is currently in the filter's state */
  bool instate() const;
  // score of the potential goodness of being an instate feature
  // The higher, the better.
  number_t score() const;
  number_t outlier_counter() const { return outlier_counter_; }
  /**
   * Gets actual depth of feature from variable `x_` (calculation is different
   * depending on whether or not we're using an inverse-depth or log-depth
   * parameterization).
   * \todo Ensure depth is positive when using inverse-depth parameterization,
   *       which is guaranteed when using log-depth paramterization. */
  number_t z() const;
  const Vec3 &x() const { return x_; }
  const Mat3 &P() const { return P_; }
  Vec3 &x() { return x_; }
  Mat3 &P() { return P_; }
  void BackupState() { x0_ = x_; }
  void RestoreState() { x_ = x0_; }
  // get 3D coordinates in reference camera frame
  Vec3 Xc(Mat3 *dXc_dx = nullptr);
  // get 3D coordinates in spatial frame, cam2body alignment is required
  Vec3 Xs(const SE3 &gbc, Mat3 *dXs_dx = nullptr);
  const Vec3& Xs() const { return Xs_; }
  /** Changes the owner of the feature. Returns false if this results in a
   * negative depth. If change in ownership results in negative depth, no
   * changes in any members of this feature are made. Used when reference
   * group meets the maximum group lifetime and is removed from the state and
   * during loop closure. */
  /** Jacobians of a re-anchored feature's new local parameterization w.r.t. the
   *  error-state blocks it depends on. Re-anchoring is a change of state
   *  coordinates, so the filter covariance has to be pushed through the whole
   *  row -- not just the feature's own 3x3 block. `Feature` cannot reach
   *  `Estimator::P_`, so `ChangeOwner` hands these out instead. */
  struct ReanchorJacobians {
    Mat3 dxn_dx;         //< w.r.t. this feature's own (X/Z, Y/Z, log Z)
    Mat36 dxn_dref_old;  //< w.r.t. [Wsb, Tsb] of the outgoing reference group
    Mat36 dxn_dref_new;  //< w.r.t. [Wsb, Tsb] of the incoming reference group
  };
  bool ChangeOwner(GroupPtr nref, const SE3 &gbc,
                   ReanchorJacobians *jac_out = nullptr);
  const int LoopClosureMatch() { return lc_match_; }
  void SetLCMatch(int matched_feat_id) { lc_match_ = matched_feat_id; }

  /** Copy observations, descriptors from another feature, and adjust state
   * and covariance estimates. Used during loop closure. Returns true if
   * merge was successful. If merge is unsuccessful, then no changes to
   * are actually made to any private members. (`Successful' means that
   * coordiante change of other feature `f` doesn't result in a negative
   * depth estimate.) */
  bool Merge(FeaturePtr f, const SE3& gbc);

  // return (2M-3) as the dimension of the measurement
  /** Computes the Jacobian for the in-state (EKF) measurement model. */
  void ComputeJacobian(const Mat3 &Rsb, const Vec3 &Tsb, const Mat3 &Rbc,
                       const Vec3 &Tbc, const Vec3 &gyro, const Mat3 &Cg,
                       const Vec3 &bg, const Vec3 &Vsb, number_t td);

  void inflate_cov(number_t factor) { P_ *= factor; }

  /** Number of rows of the marginalized out-of-state measurement, i.e. of
   *  `ro()` and `Ho()`. */
  int oos_inn_size() const { return oos_jac_counter_; }
  /** Number of observations that went into the out-of-state measurement. */
  int oos_num_obs() const { return oos_num_obs_; }
  /** Mean per-view reprojection error (pixels) of the last call to
   *  `RefineOOSDepth`. */
  number_t oos_mean_reproj_err() const { return oos_mean_reproj_err_; }

  /** The observations of this feature that an out-of-state update can use:
   *  those made from groups that are in the state, oldest first, thinned down to
   *  `options.max_observations`. Idempotent, so the depth refinement and the
   *  Jacobian can be run on the very same rows. */
  std::vector<Obs> SelectOOSObservations(const std::vector<Obs> &obs,
                                         const OOSOptions &options) const;

  /** Gauss-Newton refinement of the 3D point over all of `views`, in the
   *  log-depth parameterization w.r.t. the reference camera. Returns false --
   *  and leaves `x_` at the best state found -- when the refinement diverges, or
   *  when the mean per-view reprojection error or the depth is out of bounds, in
   *  which case the feature must not be used for an update. */
  bool RefineOOSDepth(const SE3 &gbc, const std::vector<Obs> &views,
                      const OOSOptions &options);

  /** Computes the Jacobian for the out-of-state (MSCKF) measurement model, and
   *  marginalizes the 3D point out of it. Returns the number of rows of the
   *  resulting measurement (`2n - 3` for n observations), or 0 if the feature
   *  has too few observations from in-state groups. */
  int ComputeOOSJacobian(const std::vector<Obs> &obs, const Mat3 &Rbc,
                         const Vec3 &Tbc, const OOSOptions &options);
  /** Contains the equations used in `Feature::ComputeOOSJacobian` for each
   *  observation.
   *  \todo make the following private */
  void ComputeOOSJacobianInternal(const Obs &obs, const Mat3 &Rbc,
                                  const Vec3 &Tbc);
  /** Projects the first `rows` rows of the out-of-state Jacobian onto the left
   *  nullspace of `oos_.Hf`, which eliminates the 3D point. Returns the number
   *  of rows left, `rows - 3`. */
  int MarginalizeOOSPoint(int rows);

  /** Compute Jacobians for Loop Closure measurement update. */
  void ComputeLCJacobian(const Obs &obs, const Mat3 &Rbc, const Vec3 &Tbc,
                         int match_counter, MatX &H, VecX &inn);

  // fill-in the corresponding jacobian block
  // H: the big jacobian matrix of all measurements
  // offset: of the block in H
  void FillJacobianBlock(MatX &H, int offset);
  /** Same, for the right camera's two rows. Only call when
   * `right_jac_valid()`; otherwise `J_r_` is stale. */
  void FillRightJacobianBlock(MatX &H, int offset);
  // fill-in the corresponding covariance block when inserting the feature into state
  // P: the covariance matrix of the estimator
  void FillCovarianceBlock(MatX &P);

  bool TriangulationSuccessful() { return triangulation_successful_; }

  const Eigen::Matrix<number_t, 2, kFullSize> &J() const { return J_; }
  const Vec2 &inn() const { return inn_; }

  /** The right camera's two measurement rows, in the same error-state layout as
   * `J()`, and the matching innovation `xp_r() - predicted right pixel`.
   *
   * Only meaningful when `right_jac_valid()`, which `ComputeJacobian` sets from
   * scratch on every call: a feature with no right match this frame, or one
   * whose predicted right point falls behind camera 1, contributes nothing. */
  const Eigen::Matrix<number_t, 2, kFullSize> &Jr() const { return J_r_; }
  const Vec2 &inn_r() const { return inn_r_; }
  bool right_jac_valid() const { return right_jac_valid_; }
  /** Drop this frame's right measurement from the EKF update. Used by the
   * right-camera Mahalanobis gate, which rejects a bad *match* without
   * condemning the feature itself -- the left track may be perfectly good. */
  void InvalidateRightJacobian() { right_jac_valid_ = false; }

  /** Gets the last measurement (from the `Tracker`) of this feature */
  const Vec2 &xp() const { return back(); }
  /** Returns the last-computed predicted measurement
   *  (does not compute a new prediction) */
  const Vec2 &pred() const { return pred_; }
  /** Computes a new predicted measurement (in pixels) given transformations
   *  `gsb` and `gbc` */
  const Vec2 &Predict(const SE3 &gsb, const SE3 &gbc) {
    Vec3 Xc = (gsb * gbc).inverse() * this->Xs(gbc);
    pred_ = Camera::instance()->Project(project(Xc));
    return pred_;
  }
  /** Sets variable `pred_`, the last computed predicted measurement to (-1,-1),
   *  the default "invalid" value for a predicted measurement. */
  void ResetPred() { pred_ << -1, -1; }

  ////////////////////////////////////////
  // Stereo: the right camera's observation of this feature
  ////////////////////////////////////////
  /** Record a right-camera observation for the current frame, in *pixels*.
   *
   * Only the current frame's right observation is kept. Unlike the left track
   * (a full history in `Track`), the right observation is consumed by the
   * update at the timestamp it was made and never revisited, so keeping a
   * history would cost memory for nothing. */
  void SetRightObs(const Vec2 &xp_r) {
    xp_r_ = xp_r;
    has_right_ = true;
  }
  /** Forget the right observation. Called at the start of each stereo frame, so
   * `has_right()` always refers to the *current* frame and a stale match from a
   * previous frame can never be fed to the filter. */
  void ClearRightObs() { has_right_ = false; }
  bool has_right() const { return has_right_; }
  /** Right-camera pixel observation for the current frame; only meaningful when
   * `has_right()`. */
  const Vec2 &xp_r() const { return xp_r_; }

  /** True once this feature's depth was seeded by stereo triangulation.
   *
   * Consumed by `Estimator::ProcessTracks` to suppress the two-frame *temporal*
   * pre-subfilter triangulation. That path rewrites `x_` but leaves `P_`
   * untouched, so letting it run on a stereo-seeded feature would pair a depth
   * the stereo never vouched for with the tight covariance the stereo earned --
   * strictly worse than either estimate alone. The stereo baseline (101 mm,
   * known from calibration) also beats the inter-frame baseline of a slow
   * handheld rig at 20 Hz, so there is nothing to gain by overwriting. */
  void SetStereoSeeded() { stereo_seeded_ = true; }
  bool stereo_seeded() const { return stereo_seeded_; }

  ////////////////////////////////////////
  // OOS Jacobians accessors
  ////////////////////////////////////////
  VecX ro() const { return oos_.inn.head(oos_jac_counter_); }
  MatX Ho() const { return oos_.Hx.topRows(oos_jac_counter_); }

  void Initialize(number_t z0, const Vec3 &std_xyz);

  FeatureStatus status() const { return status_; }
  void SetStatus(FeatureStatus status) { status_ = status; }

  void SetTrackStatus(TrackStatus status) { Track::SetStatus(status); }
  TrackStatus track_status() const { return Track::status(); }

  int id() const { return id_; }
  int sind() const { return sind_; }
  void SetSind(int ind) { sind_ = ind; }

  int lifetime() const { return lifetime_; }
  void IncrementLifetime() { lifetime_++; }
  void ResetLifetime() { lifetime_ = 0; }

  GroupPtr ref() const { return ref_; }
  void SetRef(GroupPtr ref);
  void ResetRef(GroupPtr nref);

  // subfilter used for depth initialization
  void SubfilterUpdate(const SE3 &gsb, const SE3 &gbc,
                       const SubfilterOptions &options);
  bool RefineDepth(const SE3 &gbc, const std::vector<Obs> &obs,
                   const RefinementOptions &options);
  // triangulate the 3D point from the reference and another view
  void Triangulate(const SE3 &gsb, const SE3 &gbc,
                   const TriangulateOptions &options);

  void SetState(const Vec3 &x) { x_ = x; }
  void UpdateState(const Vec3 &dx) {
    x_ += dx;
    ClampLogDepth();
  }

  /** `x_(2)` is log-depth (see `unproject_logz`), so `Xc()` evaluates
   * `exp(x_(2))`. An EKF correction is not bounded, and a badly conditioned
   * update can push `x_(2)` far enough that `exp()` overflows to +/-inf. The
   * resulting `inf * 0` in the measurement Jacobian is NaN, which then spreads
   * through the whole filter state and eventually aborts inside Sophus when a
   * quaternion is normalized. Saturate the log-depth to a range that keeps
   * `exp()` (and its square, used when forming covariances) finite; a feature
   * that hits this bound is hopeless anyway and gets dropped by the usual
   * depth/innovation checks.
   */
  void ClampLogDepth() {
    // exp(kMaxLogDepth)^2 stays well inside the double range.
    constexpr number_t kMaxLogDepth = 80.0;
    if (!(x_(2) > -kMaxLogDepth)) {
      x_(2) = -kMaxLogDepth;
    } else if (!(x_(2) < kMaxLogDepth)) {
      x_(2) = kMaxLogDepth;
    }
  }

  /** Initial value of static variable `Feature::counter_`/smallest possible number
   *  used for feature IDs. Its purpose is so that feature IDs and group IDs never
   *  overlap.
   *  \todo Completely separate feature and group IDs so that we don't have
   *        problems if this SLAM code is ever run for a long, long time.  */
  static constexpr int counter0 = 10000;

private:
  Feature(const Feature &) = delete;
  /** default constructor used memory manager's pre-allocation */
  Feature() = default;
  /** Resets a `Feature` object. Calls `Track::Reset` */
  void Reset(number_t x, number_t y);

  /** The right camera's two rows of the measurement model. Called at the end of
   * `ComputeJacobian`, which must already have filled `cache_`: the whole
   * dependence on the state runs through `cache_.Xcn`, so the right rows reuse
   * the left camera's entire `dXcn_d*` chain and only differ in the final
   * projection. Sets `right_jac_valid_`. */
  void ComputeRightJacobian();
  /** Shared body of `FillJacobianBlock`/`FillRightJacobianBlock`: which blocks
   * of the (mostly zero) 2 x kFullSize row pair are live is a property of the
   * error-state layout, not of which camera made the measurement. */
  void FillJacobianBlockFrom(MatX &H, int offset,
                             const Eigen::Matrix<number_t, 2, kFullSize> &J);

private:
  /** Total number of features ever created (never decremented) +
   *  (static constexpr) `counter0`. Used for getting ID values of newly created
   *  features. `counter_` starts at `counter0` so that feature and group IDs
   *  do not overlap. */
  static int counter_;
  /** Feature ID. IDs are in order of creation. */
  int id_;
  /** Index of feature in Estimator's array of instate features. Not set until
   *  `status_` is `FeatureStatus::READY`. */
  int sind_;

  /** CREATED, INITIALIZING, READY, INSTATE, REJECTED_BY_FILTER, REJECTED_BY_TRACKER,
   *  or DROPPED */
  FeatureStatus status_;
  /** Pointer to feature's reference group: pose/instance in time where the feature
   *  is first observed. */
  GroupPtr ref_;

  /** Total number of timesteps (image frames) since the feature was first detected. */
  int lifetime_;

  /** Projected state: Let (X, Y, Z) be the coordinates of the feature in 3D
   *  space with respect to the current camera frame. Then, this variable
   *  contains the vector (X/Z, Y/Z, log(Z)) or (X/Z, Y/Z, 1/Z) when compiling
   *  with `#USE_INVDEPTH` */
  Vec3 x_;

  /** "Backup" of `Feature::x_` used in `Estimator::OnePointRANSAC` */
  Vec3 x0_;

  /** Subfilter (for estimating depth) covariance */
  Mat3 P_;

  /** Predicted pixel coordinates - computed right before the `Estimator` class's
   *  measurement update step in `Feature::Predict`. */
  Vec2 pred_;

  /** Right-camera pixel observation for the current frame, and whether one was
   * found. Reset every stereo frame; see `SetRightObs`. */
  Vec2 xp_r_;
  bool has_right_{false};
  /** Sticky for the lifetime of the feature, unlike `has_right_`. */
  bool stereo_seeded_{false};

  /** 3D coordinates of the feature with respect to the current camera frame. */
  Vec3 Xc_;

  /** 3D coordiantes of the feature with respect to the (current) global reference
   *  frame. */
  Vec3 Xs_;

  Eigen::Matrix<number_t, 2, kFullSize> J_;

  /** `xp` - predicted observation used in the filter for this particular feature. */
  Vec2 inn_;

  /** The right camera's counterparts of `J_`/`inn_`, and whether they were
   * computed for the current frame. Recomputed (or invalidated) on every
   * `ComputeJacobian`, so unlike `stereo_seeded_` these never persist. */
  Eigen::Matrix<number_t, 2, kFullSize> J_r_;
  Vec2 inn_r_;
  bool right_jac_valid_{false};

  /** Measurement model Jacobian with respect to the error state used in the filter. */
  Mat23 Hx_;

  // outlier rejection
  int init_counter_;
  bool inlier_;
  number_t outlier_counter_;

  /** Contains current intermediate variables used to compute the Jacobians in both the
   *  EKF and MSCKF measurement models. */
  static JacobianCache cache_;

  /** Current MSCKF measurement Jacobians (both Hf and Hx) and innovation */
  OOSJacobian oos_;

  /** While filling `oos_`: number of observations processed so far. Afterwards:
   *  number of rows of the marginalized MSCKF measurement. */
  int oos_jac_counter_;

  /** Number of observations that went into the MSCKF measurement. */
  int oos_num_obs_;

  /** Mean per-view reprojection error (pixels) after `RefineOOSDepth`. */
  number_t oos_mean_reproj_err_;

  /** id of a past feature this feature was loop-closed to. */
  int lc_match_;

  /** whether or not triangulation was successful */
  bool triangulation_successful_;

#ifdef APPROXIMATE_INIT_COVARIANCE
  // correlation block between local feature state (x) and group pose
  std::unordered_map<int, Eigen::Matrix<number_t, kFeatureSize, kGroupSize>> cov_;
  // correlation block between local feature state (x) and camera-body alignment (c),
  // and reference group (r)
  Eigen::Matrix<number_t, kFeatureSize, kGroupSize> cov_xc_, cov_xr_;
#endif


public:
  // simulation
  static int num_good_triangulations_;
  static int num_bad_triangulations_;

  struct {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3 Xs;
    Vec2 xp, xc;
    number_t z;
    int lifetime;
  } sim_;

};

} // namespace xivo
