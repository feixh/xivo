// The feature tracking module;
// Multi-scale Lucas-Kanade tracker from OpenCV.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once

#include <list>
#include <memory>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
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

  /** Matches features found on incoming image `img` to features in `features_`
   *  using LK-pyramid and detects a new set of features to be tracked.
   *  \todo Rescue features that would otherwise be dropped from tracker with newly
   *        detected features. */
  void UpdateLK(const cv::Mat &img);

  void UpdateMatch(const cv::Mat &img);

  void Update(const cv::Mat &img);

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
  int num_failed_to_track_ = 0;
  int num_new_detections_ = 0;

  cv::Mat img_;

  /** Last computed LK pyramid */
  std::vector<cv::Mat> pyramid_;

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
  /** Number of pixels around a currently tracked feature where we shouldn't look
   *  for new features (so that we don't have two features for the same corner) */
  int mask_size_;
  int margin_;

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
