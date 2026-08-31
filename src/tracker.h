// The feature tracking module;
// Multi-scale Lucas-Kanade tracker from OpenCV.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

#include <list>
#include <memory>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc.hpp" // cv::CLAHE, for the member below
#include "json/json.h"
#include "mapper.h"

#include "core.h"

namespace xivo {


enum TrackerType : int {
  LK = 0,
  MATCH = 1,
  POINTCLOUD = 2
};

class Tracker {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static TrackerPtr Create(const Json::Value &cfg);
  static TrackerPtr instance() { return instance_.get(); }

  /** `cv::buildOpticalFlowPyramid` with `tryReuseInputImage` forced off, so the
   *  result never aliases `image`. That argument defaults to *true*, and for an
   *  input that is a submatrix with enough margin OpenCV then sets
   *  `pyramid[0] = image` -- a view. `pyramid_` has to stay valid into the next
   *  frame while `img_` no longer owns its pixels (see `UpdateLK`), so the
   *  pyramid must own level 0 whatever the caller hands us.
   *
   *  Static and public so `unitTests_pyramid` can call it, and because it is a
   *  pure function of its arguments. */
  static void BuildOwnedPyramid(const cv::Mat &image,
                                std::vector<cv::Mat> &pyramid, int win_size,
                                int max_level);

  /** Matches features found on incoming image `img` to features in `features_`
   *  using LK-pyramid and detects a new set of features to be tracked.
   *  \todo Rescue features that would otherwise be dropped from tracker with newly
   *        detected features. */
  void UpdateLK(const cv::Mat &img);

  void UpdateMatch(const cv::Mat &img);

  void Update(const cv::Mat &img);

  /** Stereo counterpart of `Update`.
   *
   * Temporal tracking on the left image is bit-for-bit the same work `Update`
   * does; the right image is used only to attach a second observation to each
   * surviving left feature. As of M2 the right image is merely validated and
   * counted -- left->right matching arrives in M3 -- so a stereo run must
   * currently produce a trajectory identical to the monocular one. That
   * equality is the regression gate for the whole data path.
   */
  void UpdateStereo(const cv::Mat &img, const cv::Mat &img_r);

  /** Number of frames for which a right image was received. */
  int num_stereo_frames() const { return num_stereo_frames_; }

  /** Left features that got a right-camera match on the most recent frame. */
  int num_stereo_matched() const { return num_stereo_matched_; }
  /** Left features considered for a right match on the most recent frame. */
  int num_stereo_attempted() const { return num_stereo_attempted_; }
  /** Cumulative counts of why a candidate right match was thrown out.
   *  Separated because they diagnose different faults: a spike in epipolar
   *  rejections points at the rig calibration, whereas a spike in
   *  circular-consistency rejections points at repetitive texture. */
  int num_stereo_rejected_klt() const { return num_stereo_rejected_klt_; }
  int num_stereo_rejected_epipolar() const {
    return num_stereo_rejected_epipolar_;
  }
  int num_stereo_rejected_circular() const {
    return num_stereo_rejected_circular_;
  }
  int num_stereo_rejected_disparity() const {
    return num_stereo_rejected_disparity_;
  }

  void UpdatePointCloud(const VecXi &feature_ids, const MatX2 &xps);

  /** Called by function `CreateSystem` to force extraction of descriptors when
   * we want to use loop closure. */
  bool IsExtractingDescriptors() { return extract_descriptor_; };

  int num_rejected_outliers() const { return num_outliers_rejected_; };

  int num_failed_to_track() const { return num_failed_to_track_; };

  int num_new_detections() const { return num_new_detections_; };

public:
  std::list<FeaturePtr> features_;

private:
  Tracker(const Tracker &other) = delete;
  Tracker &operator=(const Tracker &other) = delete;

  Tracker(const Json::Value &cfg);
  static std::unique_ptr<Tracker> instance_;

  // variables
  bool differential_;
  bool initialized_;
  /** Rescale each image to [0, 255] before tracking. Read once at construction. */
  bool normalize_;
  Json::Value cfg_;
  int descriptor_distance_thresh_; // use this to verify feature tracking
  int max_pixel_displacement_;     // pixels shifted larger than this amount are
                                   // dropped
  TrackerType tracker_type_;
  bool do_outlier_rejection_;
  int outlier_rejection_method_;
  int outlier_rejection_maxiters_;
  number_t outlier_rejection_confidence_;
  number_t outlier_rejection_reproj_thresh_;
  int num_outliers_rejected_ = 0;
  /** Fundamental-matrix RANSAC on normalized bearings; see
   * `OutlierRejectionEpipolar`. Independent of `do_outlier_rejection_`. */
  bool epipolar_rejection_;
  /** RANSAC inlier band, in pixels; divided by the focal length before use so
   * that it means the same thing in normalized coordinates. */
  number_t epipolar_thresh_px_;
  number_t epipolar_confidence_;
  /** Fewer valid correspondences than this and the frame is left alone: a
   * 7-point fit on a handful of points rejects at random. */
  int epipolar_min_pts_;
  /** Bearings longer than this are excluded from the fit (not rejected). At
   * theta -> pi/2 the bearing diverges, so a single grazing track would
   * otherwise dominate the algebraic residual. */
  number_t epipolar_max_norm_;
  int num_epipolar_rejected_ = 0;
  int num_epipolar_frames_ = 0;
  int num_epipolar_total_rejected_ = 0;
  int num_failed_to_track_ = 0;
  int num_new_detections_ = 0;
  // All of these are cumulative over the whole run, deliberately: a mix of
  // per-frame and run-total counters makes any ratio computed from them wrong.
  int num_stereo_frames_ = 0;
  int num_stereo_matched_ = 0;
  int num_stereo_attempted_ = 0;
  int num_stereo_rejected_klt_ = 0;
  int num_stereo_rejected_epipolar_ = 0;
  int num_stereo_rejected_circular_ = 0;
  int num_stereo_rejected_disparity_ = 0;

  /** Match left features into the right image and attach the result to each
   * feature via `Feature::SetRightObs`. Called by `UpdateStereo` after the
   * left image's temporal tracking has finished, so it works on the features'
   * current-frame pixel locations. */
  void MatchStereo();


  // stereo matching params; see the config documentation in cfg/tumvi_stereo.json
  /** Reject a right match whose angular epipolar residual exceeds this, in
   * radians. `StereoRig::EpipolarResidual` returns the sine of the angular
   * miss, hence the unit. */
  number_t stereo_epipolar_thresh_;
  /** Reject a right match that does not track back to within this many pixels
   * of the original left point (circular / left-right consistency). */
  number_t stereo_circular_thresh_;
  /** Reject a match whose disparity is below this many pixels: too little
   * parallax to triangulate usefully, and near-zero disparity is also what a
   * KLT that simply failed to move looks like. */
  number_t stereo_min_disparity_;
  /** Reject a match displaced further than this many pixels from the left
   * point. With a 10 cm baseline nothing closer than ~0.2 m is plausible, so a
   * huge displacement means the KLT latched onto unrelated texture. */
  number_t stereo_max_disparity_;
  int stereo_win_size_;
  int stereo_max_level_;

  cv::Mat img_;
  /** Right image of the current stereo pair; empty on monocular runs. */
  cv::Mat img_r_;

  /** Last computed LK pyramid */
  std::vector<cv::Mat> pyramid_;
  /** Whether `pyramid_` was built from the current `img_`. True after
   *  `UpdateLK` has swapped in the new pyramid, false on the paths that return
   *  before the swap (no features left to track). `MatchStereo` reuses
   *  `pyramid_` as its left pyramid only when this holds. */
  bool pyramid_is_current_ = false;

  /** Number of rows in the input image. */
  int rows_;
  /** Number of columns in the input image. */
  int cols_;

  // for the geneirc feature2d interface, see the following openc document:
  // https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html
  cv::Ptr<cv::Feature2D> detector_, extractor_;
  bool extract_descriptor_;

  /**
   * A "helper" grayscale image that indicates where the feature detector is
   * allowed to find features. Features are only valid in places where the mask
   * is white. (Pixels in `mask_` are black or white.) The dimensions are
   * `rows_-2*margin_` x `cols_-2*margin_`. The purpose of `mask_` is to prevent
   * too many features in the same location and to prevent features from being
   * detected at the very edges of images.
   */
  cv::Mat mask_;
  /** The per-frame starting point for `mask_`: white everywhere a detection is
   *  admissible on *this camera*, black elsewhere. That is the `margin_` border
   *  plus, when `max_theta_` is finite, everything outside the camera model's
   *  usable field of view (see `BuildValidMask`). Built once, on the first
   *  frame, and copied over `mask_` at the start of every frame -- which is what
   *  the old `ResetMask(mask_(interior))` did, minus the FOV part. */
  cv::Mat valid_mask_;
  /** Largest half-angle from the optical axis, in radians, at which a detection
   *  is accepted. `M_PI` (the default) admits the whole image, reproducing the
   *  original behaviour exactly.
   *
   *  This is not cosmetic on a fisheye. A feature's state is
   *  `(X/Z, Y/Z, log Z)`, so a bearing at theta >= pi/2 has no representation at
   *  all, and `EquidistantCamera::UnProject` clamps theta to just under pi/2
   *  rather than failing -- turning such a pixel into a bearing with
   *  |(X/Z, Y/Z)| ~ 6.4e3. On the TUM-VI 512x512 intrinsics theta = pi/2 lands
   *  at r = 297 px from the principal point, so 7% of the image (the four
   *  corners) is in that regime and another 4% is above 85 deg, where
   *  tan(theta) > 11 and the parameterization is merely useless rather than
   *  meaningless. */
  number_t max_theta_;
  /** Number of pixels around a currently tracked feature where we shouldn't look
   *  for new features (so that we don't have two features for the same corner) */
  int mask_size_;
  int margin_;

  /** Sub-pixel refinement of new detections (`cv::cornerSubPix`). FAST reports
   *  integer coordinates, and a new detection's pixel *is* the anchor that
   *  defines two of the three components of the feature's state, so the +-0.5 px
   *  quantization there is a bias that persists for the whole track rather than
   *  noise that averages out. */
  bool subpix_refine_;
  int subpix_win_size_;
  int subpix_max_iter_;
  number_t subpix_eps_;

  /** Contrast equalization applied to the input image before tracking:
   *  `NONE` (default, original behaviour), `HISTOGRAM` (`cv::equalizeHist`) or
   *  `CLAHE`. */
  enum class HistogramMethod : int { NONE = 0, HISTOGRAM = 1, CLAHE = 2 };
  HistogramMethod histogram_method_;
  number_t clahe_clip_limit_;
  int clahe_grid_size_;
  cv::Ptr<cv::CLAHE> clahe_;
  /** Destinations for the equalized images. Members, not locals, because `img_`
   *  and `img_r_` are non-owning views that must stay valid for the frame. */
  cv::Mat img_eq_, img_r_eq_;
  /** Convert incoming frames to single-channel luminance. See `ToGray`. */
  bool grayscale_;
  cv::Mat img_gray_, img_r_gray_;

  /** Applies `histogram_method_` to `src`, writing into `dst`, and returns the
   *  image the rest of the frame should use. A no-op returns `src` itself so
   *  nothing is copied when equalization is off. */
  /** BGR -> luminance, into `dst`, when `grayscale_` is on and `src` needs it.
   * Returns a reference to whichever of the two now holds the image. */
  const cv::Mat &ToGray(const cv::Mat &src, cv::Mat &dst) const;

  const cv::Mat &Equalize(const cv::Mat &src, cv::Mat &dst);

  /** Fills `valid_mask_` for a `rows_` x `cols_` image of camera `cam_id`. */
  void BuildValidMask(int cam_id);

  /** Runs `cv::cornerSubPix` on `pts` in place, reverting any point that leaves
   *  the image or moves further than the search window (which means the corner
   *  was not really there). No-op if `subpix_refine_` is false. */
  void RefineSubPix(const cv::Mat &img, std::vector<cv::Point2f> &pts) const;

  // optical flow params
  int win_size_;
  int max_level_;
  int max_iter_;
  number_t eps_;

  // feature detector params
  int num_features_min_;
  int num_features_max_;

  // Matching newly detected tracks to tracks that were just dropped
  bool match_dropped_tracks_;
  /** If false, the KLT tracker ignores the filter's predicted measurement and
   * seeds each search from the previous pixel location instead. */
  bool use_prediction_;
  cv::Ptr<cv::BFMatcher> matcher_;

private:
  /** Detects new features. Entries of `newly_dropped_tracks` that get rescued
   * by descriptor matching are set to `nullptr`, so the caller can tell which
   * ones must still be marked DROPPED. Taken by reference for that reason. */
  void DetectLK(const cv::Mat &img, int num_to_add,
                std::vector<FeaturePtr> &newly_dropped_tracks,
                bool check_homography, cv::Mat H);

  /** An interface to OpenCV's `findHomography` that checks for outliers. */
  bool OutlierRejection(const std::vector<cv::Point2f> pts0,
                        const std::vector<cv::Point2f> pts1,
                        std::vector<uint8_t>& match_status,
                        cv::Mat& H);

  /** Two-view epipolar outlier rejection, after OpenVINS'
   * `TrackKLT::perform_matching`. Both point sets are unprojected to normalized
   * bearings first, because the pixel-space mapping of a fisheye is nonlinear
   * and a fundamental matrix does not exist in distorted coordinates. Zeroes
   * `match_status` for the rejected correspondences and returns how many.
   * Correspondences already at status 0, and those outside
   * `epipolar_max_norm_`, take no part in the fit and are left as they are. */
  int OutlierRejectionEpipolar(const std::vector<cv::Point2f> &pts0,
                               const std::vector<cv::Point2f> &pts1,
                               std::vector<uint8_t> &match_status, int cam_id);
};

// helpers

/** Called right before detecting a set of features on a new image. Makes all of
 *  `mask_` white. */
void ResetMask(cv::Mat mask);

/** Makes all the pixels in a `mask_size` x `mask_size` box centered at pixel `(x,y)`
 *  in `mask_` black. Called after each new detection is found. */
void MaskOut(cv::Mat mask, number_t x, number_t y, int mask_size = 15);

/** Checks whether or not `mask_` is white at pixel `(x,y)` and whether or not
 *  (x,y) is not too close to the edge of the image. */
bool MaskValid(const cv::Mat &mask, number_t x, number_t y);

/** Returns `true` if the distance between two descriptors,
 *  `descriptor_distance`, is less than `max_distance`. Also returns `true`
 *  if we are not doing a descriptor distance check (i.e.
 *  `max_distance = -1`). */
bool CheckDescriptorDistance(number_t descriptor_distance,
                             number_t max_distance);

/** Returns `true` if two keypoints are close-enough together (in Euclidean
 *  distance of pixel coordinates) */
bool CheckPixelDisplacement(const Vec2 kp1,
                            const Vec2 kp2,
                            const number_t max_displacement);

/** Same as above with different API, for convenience. */
bool CheckPixelDisplacement(const cv::KeyPoint kp1,
                            const Vec2 kp2,
                            const number_t max_displacement);

bool CheckHomography(cv::Point2f p0, cv::Point2f p1, cv::Mat H,
                     number_t reproj_threshold);

/** Assembles the descriptors of all the features in `fvec` into a single
 *  matrix. */
cv::Mat GetDescriptors(std::vector<FeaturePtr> fvec);


} // namespace xivo
