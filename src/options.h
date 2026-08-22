// Options objects for various depth-related algorithms,
// and policies for feature selection, etc.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include "core.h"
#include <string>

namespace xivo {
// depth refinement options
struct RefinementOptions {
  RefinementOptions()
      : two_view{false}, use_hessian{false}, max_iters{5}, eps{1e-5}, damping{1e-3},
        max_res_norm{5.0} {}

  bool two_view;
  bool use_hessian; // overwrite feature covariance with inverse of Hessian from depth refinement
  int max_iters;      // maximal iterations to perform
  number_t eps;          // epsilon tolerance to stop optimization
  number_t damping;      // optional damping factor
  number_t max_res_norm; // maximal per observation residual norm
  number_t Rtri;  // measuremnt covariance for depth triangulation 
};

// depth subfilter options
struct SubfilterOptions {
  SubfilterOptions() : Rtri{3.5}, MH_thresh{5.991}, ready_steps{5} {}

  number_t Rtri;      // measurement covariance for triangulation
  number_t MH_thresh; // Mahalanobis gating threshold in depth-subfilter
  int ready_steps; // feature initialized with this amount of attempts is turned
                   // to ready status
};

// options for depth triangulation
struct TriangulateOptions {
  TriangulateOptions() : method{"linf_angular"}, zmin{0.05}, zmax{5.0}, max_theta_thresh{0.01}, beta_thesh{1e-8} {}

  std::string method;
  number_t zmin, zmax;
  number_t max_theta_thresh, beta_thesh; // thresholds for angular reprojection error and parallax error 
};

/** Options for the out-of-state (MSCKF) update. A feature that leaves the
 *  tracker without ever having been in the state is triangulated from all of its
 *  observations, and the resulting stack of reprojection residuals is projected
 *  onto the left nullspace of the Jacobian w.r.t. the 3D point, which
 *  marginalizes the point and leaves a constraint on the poses in the state. */
struct OOSOptions {
  OOSOptions()
      : min_observations{4}, max_observations{kMaxGroup}, refine{true},
        max_iters{10}, eps{1e-5}, Rtri{1.0}, max_mean_reproj_err{1.5},
        zmin{0.05}, zmax{50.0}, MH_thresh{0.0}, use_stereo{true},
        stereo_R_scale{1.0} {}

  int min_observations; // minimal number of observations from in-state groups
  int max_observations; // cap on the observations used, to bound the cost of
                        // the update (the track is thinned, not dropped)
  bool refine;          // Gauss-Newton refine the 3D point before marginalizing
  int max_iters;        // maximal iterations of the refinement
  number_t eps;         // convergence tolerance of the refinement
  number_t Rtri;        // measurement covariance used in the refinement
  number_t max_mean_reproj_err; // reject the feature if the mean per-view
                                // reprojection error (pixels) exceeds this
  number_t zmin, zmax;          // admissible depth range after refinement
  number_t MH_thresh; // Mahalanobis gate on the marginalized innovation,
                      // per degree of freedom; <= 0 disables it

  /** Use the right camera's observations of an out-of-state track as well,
   *  when a stereo rig is configured and the matcher found one at that frame.
   *  Each such view then contributes 4 rows instead of 2 before the 3D point is
   *  marginalized out, so an n-view track yields `4n - 3` rows rather than
   *  `2n - 3`. Ignored in monocular runs (`StereoRig::enabled()` false). */
  bool use_stereo;
  /** Variance of a right-camera row relative to a left-camera one, i.e. the
   *  same role `stereo_update.R_scale` plays for the in-state update. The
   *  out-of-state update feeds `Roos_` to the filter as a *scalar* (which is
   *  only valid because the marginalization uses an orthonormal nullspace
   *  basis), so instead of carrying a non-uniform R the right rows are whitened
   *  by `1 / sqrt(stereo_R_scale)` as they are written -- after which the
   *  stacked measurement really is isotropic and the existing algebra holds
   *  unchanged. 1.0 leaves the right rows alone. */
  number_t stereo_R_scale;
};

// Options for adaptive initial depth estimation
struct AdaptiveInitialDepthOptions {
  AdaptiveInitialDepthOptions() : median_weight{0.99}, min_feature_lifetime{5} {}
  number_t median_weight;
  int min_feature_lifetime;
};

struct Criteria {
  // how good is the feature to be an instate candidate
  static bool Candidate(FeaturePtr f);
  static bool CandidateStrict(FeaturePtr f);

  // Feature Comparison function: used to sort features
  // when selecting to move into the main state. Returns True when `f1` has
  // a higher (better) score than `f2`.
  static bool CandidateComparison(FeaturePtr f1, FeaturePtr f2);
};

} // namespace xivo
