// The feature tracking module;
// Multi-scale Lucas-Kanade tracker from OpenCV.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include <fstream>
#include <algorithm>

#include "glog/logging.h"
#include "opencv2/video/video.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#include "camera_manager.h"
#include "feature.h"
#include "stereo.h"
#include "tracker.h"
#include "visualize.h"

namespace xivo {

auto sum_total = [](std::vector<uint8_t> vec) {
  int sum = 0;
  for (auto v: vec) {
    sum += v;
  }
  return sum;
};

auto num_zeros = [](std::vector<uint8_t> vec) {
  int num = 0;
  for (auto v: vec) {
    if (!v) {
      num++;
    }
  }
  return num;
};

cv::Ptr<cv::FeatureDetector> GetOpenCVDetectorDescriptor(
  std::string feature_type, Json::Value feature_cfg)
{
  if (feature_type == "FAST") {
    return cv::FastFeatureDetector::create(
      feature_cfg.get("threshold", 5).asInt(),
      feature_cfg.get("nonmaxSuppression", true).asBool());
  } else if (feature_type == "BRISK") {
    return cv::BRISK::create(
      feature_cfg.get("thresh", 5).asInt(),
      feature_cfg.get("octaves", 3).asInt(),
      feature_cfg.get("patternScale", 1.0).asFloat());
  } else if (feature_type == "ORB") {
    return cv::ORB::create(
      feature_cfg.get("nfeatures", 500).asInt(),
      feature_cfg.get("scaleFactor", 1.2).asFloat(),
      feature_cfg.get("nlevels", 4).asInt(),
      feature_cfg.get("edgeThreshold", 31).asInt(),
      feature_cfg.get("firstLevel", 0).asInt(),
      feature_cfg.get("WTA_K", 2).asInt(),
      static_cast<cv::ORB::ScoreType>(
        feature_cfg.get("scoreType", cv::ORB::HARRIS_SCORE).asInt()),
      feature_cfg.get("patchSize", 31).asInt(),
      feature_cfg.get("fastThreshold", 20).asInt());
  } else if (feature_type == "AGAST") {
    return cv::AgastFeatureDetector::create(
      feature_cfg.get("threshold", 10).asInt(),
      feature_cfg.get("nonmaxSuppression", true).asBool());
  } else if (feature_type == "GFTT") {
    return cv::GFTTDetector::create(
      feature_cfg.get("maxCorners", 1000).asInt(),
      feature_cfg.get("qualityLevel", 0.01).asDouble(),
      feature_cfg.get("minDistance", 1.0).asDouble(),
      feature_cfg.get("blockSize", 3).asInt(),
      feature_cfg.get("useHarrisDetector", false).asBool(),
      feature_cfg.get("k", 0.04).asDouble());
  } else if (feature_type == "SIFT") {
    return cv::SIFT::create(
      feature_cfg.get("nfeatures", 0).asInt(),
      feature_cfg.get("nOctaveLayers", 3).asInt(),
      feature_cfg.get("contrastThreshold", 0.04).asDouble(),
      feature_cfg.get("edgeThreshold", 10.0).asDouble(),
      feature_cfg.get("sigma", 1.6).asDouble());
  } else if (feature_type == "SURF") {
    return cv::xfeatures2d::SURF::create(
      feature_cfg.get("hessianThreshold", 100).asDouble(),
      feature_cfg.get("nOctaves", 4).asInt(),
      feature_cfg.get("nOctaveLayers", 3).asInt(),
      feature_cfg.get("extended", false).asBool(),
      feature_cfg.get("upright", false).asBool());
  } else if (feature_type == "BRIEF") {
    return cv::xfeatures2d::BriefDescriptorExtractor::create(
      feature_cfg.get("bytes", 64).asInt(),
      feature_cfg.get("use_orientation", false).asBool());
  } else if (feature_type == "FREAK") {
    return cv::xfeatures2d::FREAK::create(
      feature_cfg.get("orientationNormalized", true).asBool(),
      feature_cfg.get("scaleNormalized", true).asBool(),
      feature_cfg.get("patternScale", 22.0).asDouble(),
      feature_cfg.get("nOctaves", 4).asInt());
  } else {
    throw std::invalid_argument("unrecognized detector or descriptor type");
  }
}


std::unique_ptr<Tracker> Tracker::instance_ = nullptr;

TrackerPtr Tracker::Create(const Json::Value &cfg) {
  if (instance_ == nullptr) {
    instance_ = std::unique_ptr<Tracker>(new Tracker(cfg));
  } else {
    LOG(WARNING) << "tracker already created";
  }
  return instance_.get();
}

Tracker::Tracker(const Json::Value &cfg) : cfg_{cfg} {
  initialized_ = false;
  mask_size_ = cfg_.get("mask_size", 15).asInt();
  margin_ = cfg_.get("margin", 16).asInt();
  num_features_min_ = cfg_.get("num_features_min", 120).asInt();
  num_features_max_ = cfg_.get("num_features_max", 150).asInt();
  max_pixel_displacement_ = cfg_.get("max_pixel_displacement", 64).asInt();
  differential_ = cfg_.get("differential", true).asBool();
  // Was declared in every shipped config but read by nothing, so measurement
  // prediction was always on -- which closes a loop from the EKF state back
  // into the front end: a diverging filter biases the KLT initial guess toward
  // its own wrong prediction.
  use_prediction_ = cfg_.get("use_prediction", true).asBool();

  std::string tracker_type = cfg_.get("tracker_type", "LK").asString();
  if (tracker_type == "LK") {
    tracker_type_ = TrackerType::LK;
  } else if (tracker_type == "MATCH") {
    tracker_type_ = TrackerType::MATCH;
  } else if (tracker_type == "POINTCLOUD") {
    tracker_type_ = TrackerType::POINTCLOUD;
  } else {
    LOG(FATAL) << "Invalid tracker type";
  }

  do_outlier_rejection_ = cfg_.get("do_outlier_rejection", false).asBool();
  auto outlier_rejection_cfg = cfg_["outlier_rejection"];
  outlier_rejection_maxiters_ =
    outlier_rejection_cfg.get("RANSAC_max_iters", 2000).asInt();
  outlier_rejection_confidence_ =
    outlier_rejection_cfg.get("confidence", 0.995).asDouble();
  outlier_rejection_reproj_thresh_ =
    outlier_rejection_cfg.get("RANSAC_reproj_thresh", 3.0).asDouble();
  std::string outlier_rejection_method =
    outlier_rejection_cfg.get("method", "RANSAC").asString();
  if (outlier_rejection_method == "RANSAC") {
    outlier_rejection_method_ = cv::RANSAC;
  } else if (outlier_rejection_method == "LMEDS") {
    outlier_rejection_method_ = cv::LMEDS;
  } else if (outlier_rejection_method == "RHO") {
    outlier_rejection_method_ = cv::RHO;
  } else {
    LOG(FATAL) << "Invalid robust outlier rejection method " <<
      outlier_rejection_method;
  }


  // Read once, not once per frame: this used to be a string lookup into a
  // Json::Value on every image.
  normalize_ = cfg_.get("normalize", false).asBool();

  // Front-end quality knobs added by the position work. Every default here
  // reproduces the original behaviour bit-for-bit, so a config that does not
  // mention them is unaffected.
  {
    number_t max_theta_deg = cfg_.get("max_theta_deg", 180.0).asDouble();
    max_theta_ = max_theta_deg * M_PI / 180.0;
  }
  subpix_refine_ = cfg_.get("subpix_refine", false).asBool();
  auto subpix_cfg = cfg_["subpix"];
  subpix_win_size_ = subpix_cfg.get("win_size", 5).asInt();
  subpix_max_iter_ = subpix_cfg.get("max_iter", 20).asInt();
  subpix_eps_ = subpix_cfg.get("eps", 0.001).asDouble();

  grayscale_ = cfg_.get("grayscale", false).asBool();

  std::string hist = cfg_.get("histogram_method", "NONE").asString();
  if (hist == "NONE") {
    histogram_method_ = HistogramMethod::NONE;
  } else if (hist == "HISTOGRAM") {
    histogram_method_ = HistogramMethod::HISTOGRAM;
  } else if (hist == "CLAHE") {
    histogram_method_ = HistogramMethod::CLAHE;
  } else if (hist == "GAINMAP") {
    histogram_method_ = HistogramMethod::GAINMAP;
  } else {
    LOG(FATAL) << "Invalid tracker histogram_method " << hist;
  }
  clahe_clip_limit_ = cfg_.get("clahe_clip_limit", 10.0).asDouble();
  clahe_grid_size_ = cfg_.get("clahe_grid_size", 8).asInt();
  if (histogram_method_ == HistogramMethod::CLAHE) {
    clahe_ = cv::createCLAHE(clahe_clip_limit_,
                             cv::Size(clahe_grid_size_, clahe_grid_size_));
  }
  std::string eq_scope = cfg_.get("equalize_for", "ALL").asString();
  if (eq_scope == "ALL") {
    equalize_scope_ = EqualizeScope::ALL;
  } else if (eq_scope == "DETECT") {
    equalize_scope_ = EqualizeScope::DETECT;
  } else {
    LOG(FATAL) << "Invalid tracker equalize_for " << eq_scope
               << " (expected ALL or DETECT)";
  }
  auto gainmap_cfg = cfg_["gainmap"];
  gainmap_bins_ = std::max(2, gainmap_cfg.get("bins", 32).asInt());
  gainmap_smooth_ = std::max(0, gainmap_cfg.get("smooth", 2).asInt());
  gainmap_max_gain_ = gainmap_cfg.get("max_gain", 4.0).asDouble();
  // Both of these abort on a multi-channel input, and the driver hands us BGR
  // (see `ToGray`). Rather than leave a config that aborts on the first frame,
  // turn the conversion on for them.
  if (!grayscale_ &&
      (subpix_refine_ || histogram_method_ != HistogramMethod::NONE)) {
    grayscale_ = true;
    LOG(WARNING) << "tracker: forcing grayscale=true, required by "
                 << (subpix_refine_ ? "subpix_refine" : "histogram_method");
  }

  // Epipolar (fundamental-matrix) outlier rejection on *normalized* bearings.
  // Distinct from `do_outlier_rejection` above, which fits a homography to raw
  // distorted pixels -- a model that only holds for a plane or a pure rotation,
  // and whose residual is not even a metric distance under a 190-degree
  // fisheye. Defaults leave this off.
  auto epi_cfg = cfg_["epipolar_rejection"];
  epipolar_rejection_ = epi_cfg.get("enable", false).asBool();
  epipolar_thresh_px_ = epi_cfg.get("thresh_px", 2.0).asDouble();
  epipolar_confidence_ = epi_cfg.get("confidence", 0.999).asDouble();
  epipolar_min_pts_ = epi_cfg.get("min_points", 10).asInt();
  epipolar_max_norm_ = epi_cfg.get("max_bearing_norm", 10.0).asDouble();

  auto klt_cfg = cfg_["KLT"];
  win_size_ = klt_cfg.get("win_size", 15).asInt();
  max_level_ = klt_cfg.get("max_level", 4).asInt();
  max_iter_ = klt_cfg.get("max_iter", 15).asInt();
  eps_ = klt_cfg.get("eps", 0.01).asDouble();

  // Stereo left->right matching. Defaults are chosen for TUM-VI's 512x512
  // fisheye pair (~101 mm baseline, fx ~= 191 px); see
  // notes-stereo/m3-stereo-tracking.md for how each was picked.
  auto stereo_cfg = cfg_["stereo_matching"];
  stereo_epipolar_thresh_ =
      stereo_cfg.get("epipolar_thresh", 0.005).asDouble();
  stereo_circular_thresh_ =
      stereo_cfg.get("circular_thresh", 1.0).asDouble();
  stereo_min_disparity_ = stereo_cfg.get("min_disparity", 1.0).asDouble();
  stereo_max_disparity_ = stereo_cfg.get("max_disparity", 150.0).asDouble();
  // The right search reuses the temporal KLT window/levels unless overridden.
  stereo_win_size_ = stereo_cfg.get("win_size", win_size_).asInt();
  stereo_max_level_ = stereo_cfg.get("max_level", max_level_).asInt();
  stereo_back_track_ = stereo_cfg.get("back_track", true).asBool();
  stereo_seed_prev_disparity_ =
      stereo_cfg.get("seed_prev_disparity", false).asBool();
  stereo_max_disparity_jump_ =
      stereo_cfg.get("max_disparity_jump",
                     std::numeric_limits<number_t>::infinity())
          .asDouble();
  if (std::isfinite(stereo_max_disparity_jump_) &&
      !stereo_seed_prev_disparity_) {
    LOG(WARNING) << "tracker: stereo_matching.max_disparity_jump needs "
                    "seed_prev_disparity, which maintains the table it reads; "
                    "ignoring it";
    stereo_max_disparity_jump_ = std::numeric_limits<number_t>::infinity();
  }

  std::string detector_type = cfg_.get("detector", "FAST").asString();
  LOG(INFO) << "detector type=" << detector_type;
  if ((detector_type == "FAST") ||
      (detector_type == "BRISK") ||
      (detector_type == "ORB") ||
      (detector_type == "AGAST") ||
      (detector_type == "GFTT") ||
      (detector_type == "SIFT") ||
      (detector_type == "SURF")) {
    detector_ = GetOpenCVDetectorDescriptor(detector_type,
                                            cfg_[detector_type]);
    LOG(INFO) << "detector created";
  } else {
    LOG(FATAL) << "Invalid Feature Detector: " << detector_type;
  }

  descriptor_distance_thresh_ =
      cfg_.get("descriptor_distance_thresh", -1).asInt();
  extract_descriptor_ = cfg_.get("extract_descriptor", false).asBool() ||
                        descriptor_distance_thresh_ > -1;
  LOG(INFO) << "descriptor extraction "
            << (extract_descriptor_ ? "ENABLED" : "DISABLED");
  if ((tracker_type_ == TrackerType::MATCH) && !extract_descriptor_) {
    LOG(FATAL) << "Using a matcher-tracker requires extracting descriptors";
  }


  if (extract_descriptor_) {
    std::string descriptor_type = cfg_.get("descriptor", "BRIEF").asString();
    LOG(INFO) << "descriptor type=" << descriptor_type;
    if ((descriptor_type == "BRIEF") ||
        (descriptor_type == "BRISK") ||
        (descriptor_type == "ORB") ||
        (descriptor_type == "FREAK") ||
        (descriptor_type == "SIFT") ||
        (descriptor_type == "SURF")) {
      extractor_ = GetOpenCVDetectorDescriptor(descriptor_type,
                                               cfg_[descriptor_type]);
    } else {
      LOG(FATAL) << "Invalid feature descriptor: " << descriptor_type;
    }
  }

  // Rescuing dropped tracks (Only applicable to LK tracker)
  if (tracker_type_ == TrackerType::LK) {
    match_dropped_tracks_ = cfg_.get("match_dropped_tracks", false).asBool();
    if (match_dropped_tracks_ && !extract_descriptor_) {
      LOG(FATAL) << "must extract descriptors in order to match dropped tracks";
    }
    if (match_dropped_tracks_) {
      // The number of dropped tracks to match should not be that large, so
      // using Brute-Force matcher instead of FLANN-based matcher.
      matcher_ = cv::BFMatcher::create(extractor_->defaultNorm(), true);
    }
  } else if (tracker_type_ == TrackerType::MATCH) {
    matcher_ = cv::BFMatcher::create(extractor_->defaultNorm(), true);
  }
}


const cv::Mat &Tracker::ToGray(const cv::Mat &src, cv::Mat &dst) const {
  // Normally a no-op now: the file-path entry points decode with
  // IMREAD_GRAYSCALE (see `ReadImage` in pybind11/pyxivo.cpp), so frames arrive
  // single-channel and this returns `src` untouched. It still matters for the
  // numpy-buffer entry points, which pass whatever the caller hands over and are
  // deliberately left free to hand over HxWx3.
  //
  // Why the conversion has to happen somewhere: `equalizeHist` and `cornerSubPix`
  // both require a single channel, and tracking on BGR is not the same problem as
  // tracking on luminance. `FastFeatureDetector::detect` and
  // `calcOpticalFlowPyrLK` merely tolerate 3 channels, which is why a colour
  // frame used to run all the way through without complaint.
  if (!grayscale_ || src.channels() == 1) {
    return src;
  }
  cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
  return dst;
}

void Tracker::BuildGainMap(const cv::Mat &src, int cam, cv::Mat &gain) const {
  CHECK(src.type() == CV_8UC1) << "GAINMAP needs a single-channel 8-bit image";
  const int rows = src.rows, cols = src.cols;

  // The vignette is centred on the optical axis, not on the sensor, so use the
  // principal point when the camera model is available.
  number_t cx = 0.5 * cols, cy = 0.5 * rows;
  auto camera = Camera::instance(cam);
  if (camera != nullptr) {
    cx = camera->cx();
    cy = camera->cy();
  }

  // Mean intensity per radial bin. `rmax` is the largest radius any pixel can
  // have from this centre, so the last bin is never empty.
  const number_t rmax =
      std::sqrt(std::max(cx, number_t(cols) - cx) * std::max(cx, number_t(cols) - cx) +
                std::max(cy, number_t(rows) - cy) * std::max(cy, number_t(rows) - cy));
  const int NB = gainmap_bins_;
  std::vector<double> sum(NB, 0.0), cnt(NB, 0.0);
  const number_t inv = NB / (rmax > 0 ? rmax : 1);
  for (int r = 0; r < rows; ++r) {
    const uint8_t *s = src.ptr<uint8_t>(r);
    const number_t dy = r + number_t(0.5) - cy;
    for (int c = 0; c < cols; ++c) {
      const number_t dx = c + number_t(0.5) - cx;
      int b = static_cast<int>(std::sqrt(dx * dx + dy * dy) * inv);
      if (b >= NB) b = NB - 1;
      sum[b] += s[c];
      cnt[b] += 1.0;
    }
  }
  std::vector<double> mean(NB, 0.0);
  for (int b = 0; b < NB; ++b) {
    mean[b] = cnt[b] > 0 ? sum[b] / cnt[b] : 0.0;
  }
  // A single frame's profile is smooth in expectation but not in realization;
  // smoothing it keeps one dark annulus from turning into a ring in the gain.
  std::vector<double> prof(NB, 0.0);
  for (int b = 0; b < NB; ++b) {
    double acc = 0.0, w = 0.0;
    for (int k = -gainmap_smooth_; k <= gainmap_smooth_; ++k) {
      const int j = b + k;
      if (j < 0 || j >= NB || cnt[j] <= 0) continue;
      acc += mean[j];
      w += 1.0;
    }
    prof[b] = w > 0 ? acc / w : mean[b];
  }
  // Normalize to the *brightest* annulus, so every gain is >= 1 and the correction
  // only ever brightens. Normalizing to the centre instead would darken anything
  // brighter than the centre, throwing away contrast the detector already had.
  double ref = 0.0;
  for (int b = 0; b < NB; ++b) ref = std::max(ref, prof[b]);
  std::vector<double> g(NB, 1.0);
  if (ref > 1.0) {
    for (int b = 0; b < NB; ++b) {
      g[b] = prof[b] > 1.0 ? std::min<double>(gainmap_max_gain_, ref / prof[b])
                           : gainmap_max_gain_;
    }
  }

  // Per-pixel map, linearly interpolated between bin *centres* so there is no
  // visible step at a bin boundary -- a step would be an edge, and the detector
  // would find corners along it.
  gain.create(rows, cols, CV_16UC1);
  for (int r = 0; r < rows; ++r) {
    uint16_t *d = gain.ptr<uint16_t>(r);
    const number_t dy = r + number_t(0.5) - cy;
    for (int c = 0; c < cols; ++c) {
      const number_t dx = c + number_t(0.5) - cx;
      const number_t t = std::sqrt(dx * dx + dy * dy) * inv - number_t(0.5);
      int b0 = static_cast<int>(std::floor(t));
      number_t f = t - b0;
      if (b0 < 0) { b0 = 0; f = 0; }
      if (b0 >= NB - 1) { b0 = NB - 2; f = 1; }
      const double gv = g[b0] * (1.0 - f) + g[b0 + 1] * f;
      d[c] = static_cast<uint16_t>(
          std::lround(std::min(255.0, std::max(1.0, gv)) * 256.0));
    }
  }
  LOG(INFO) << "gain map for camera " << cam << ": centre (" << cx << ", " << cy
            << "), profile " << prof.front() << " -> " << prof.back()
            << ", gain " << g.front() << " -> " << g.back();
}


const cv::Mat &Tracker::Equalize(const cv::Mat &src, cv::Mat &dst, int cam) {
  switch (histogram_method_) {
  case HistogramMethod::HISTOGRAM:
    cv::equalizeHist(src, dst);
    return dst;
  case HistogramMethod::CLAHE:
    clahe_->apply(src, dst);
    return dst;
  case HistogramMethod::GAINMAP: {
    const int idx = (cam == 1) ? 1 : 0;
    cv::Mat &gain = gain_q8_[idx];
    if (gain.empty() || gain.rows != src.rows || gain.cols != src.cols) {
      BuildGainMap(src, idx, gain);
    }
    dst.create(src.size(), CV_8UC1);
    for (int r = 0; r < src.rows; ++r) {
      const uint8_t *s = src.ptr<uint8_t>(r);
      const uint16_t *g = gain.ptr<uint16_t>(r);
      uint8_t *d = dst.ptr<uint8_t>(r);
      for (int c = 0; c < src.cols; ++c) {
        const int v = (static_cast<int>(s[c]) * static_cast<int>(g[c])) >> 8;
        d[c] = v > 255 ? uint8_t(255) : static_cast<uint8_t>(v);
      }
    }
    return dst;
  }
  case HistogramMethod::NONE:
  default:
    return src;
  }
}


const cv::Mat &Tracker::DetectionImage() {
  if (equalize_scope_ != EqualizeScope::DETECT ||
      histogram_method_ == HistogramMethod::NONE) {
    return img_;
  }
  return Equalize(img_, img_det_, /*cam=*/0);
}


void Tracker::BuildValidMask(int cam_id) {
  valid_mask_ = cv::Mat(rows_, cols_, CV_8UC1);
  valid_mask_.setTo(0);
  cv::Mat interior = valid_mask_(
      cv::Rect(margin_, margin_, cols_ - 2 * margin_, rows_ - 2 * margin_));
  interior.setTo(255);

  if (!(max_theta_ < M_PI)) {
    return; // whole image admissible; identical to the original mask
  }

  auto cam = Camera::instance(cam_id);
  if (cam == nullptr) {
    LOG(WARNING) << "max_theta_deg is set but camera " << cam_id
                 << " does not exist; not applying a field-of-view mask";
    return;
  }
  // The test is done through UnProject rather than on a precomputed pixel
  // radius so that it holds for every camera model, including the ones whose
  // UnProject saturates instead of reporting failure: a saturated theta comes
  // back as a huge |xc| and is rejected here just the same.
  const number_t tan_max =
      max_theta_ < 0.5 * M_PI ? std::tan(max_theta_)
                              : std::numeric_limits<number_t>::infinity();
  int num_blocked = 0;
  for (int r = margin_; r < rows_ - margin_; ++r) {
    uint8_t *row = valid_mask_.ptr<uint8_t>(r);
    for (int c = margin_; c < cols_ - margin_; ++c) {
      Vec2 xc = cam->UnProject(
          Vec2{static_cast<number_t>(c) + number_t(0.5),
               static_cast<number_t>(r) + number_t(0.5)});
      if (!(xc.norm() <= tan_max) || !xc.allFinite()) {
        row[c] = 0;
        ++num_blocked;
      }
    }
  }
  LOG(INFO) << "field-of-view mask for camera " << cam_id << ": blocked "
            << num_blocked << " of " << (rows_ * cols_) << " pixels beyond "
            << (max_theta_ * 180.0 / M_PI) << " deg";
}


void Tracker::RefineSubPix(const cv::Mat &img,
                           std::vector<cv::Point2f> &pts) const {
  if (!subpix_refine_ || pts.empty()) {
    return;
  }
  std::vector<cv::Point2f> refined(pts);
  cv::cornerSubPix(
      img, refined, cv::Size(subpix_win_size_, subpix_win_size_),
      cv::Size(-1, -1),
      cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                       subpix_max_iter_, subpix_eps_));
  // cornerSubPix solves an unconstrained least-squares problem per point and
  // will happily walk a point off the corner it was seeded on, or out of the
  // image. Either means the refinement is not a better estimate of the same
  // corner, so keep the integer detection instead of following it.
  const float max_move = static_cast<float>(subpix_win_size_);
  for (size_t i = 0; i < pts.size(); ++i) {
    const float dx = refined[i].x - pts[i].x;
    const float dy = refined[i].y - pts[i].y;
    if (!std::isfinite(dx) || !std::isfinite(dy) ||
        std::sqrt(dx * dx + dy * dy) > max_move || refined[i].x < 0 ||
        refined[i].y < 0 || refined[i].x >= img.cols ||
        refined[i].y >= img.rows) {
      continue;
    }
    pts[i] = refined[i];
  }
}


void Tracker::DetectLK(const cv::Mat &img, int num_to_add,
                       std::vector<FeaturePtr> &newly_dropped_tracks,
                       bool check_homography, cv::Mat H)
{
  std::vector<cv::KeyPoint> kps;
  timer_.Tick("detect-fast");
  detector_->detect(img, kps, mask_);
  timer_.Tock("detect-fast");
  num_raw_detections_ += static_cast<long>(kps.size());
  ++num_detect_frames_;
  // sort
  timer_.Tick("detect-sort");
  std::sort(kps.begin(), kps.end(),
            [](const cv::KeyPoint &kp1, const cv::KeyPoint &kp2) {
              return kp1.response > kp2.response;
            });
  timer_.Tock("detect-sort");

  cv::Mat descriptors;
  if (extract_descriptor_) {
    descriptors.reserveBuffer(kps.size() * extractor_->descriptorSize());
    extractor_->compute(img, kps, descriptors);
  }

  // now every keypoint is equipped with a descriptor


  // match keypoints to old features - indices of these vectors correspond to
  // new features
  std::vector<bool> matched(kps.size(), false);
  std::vector<int> matchIdx(kps.size(), -1);

  // extract_descriptor_ is part of the condition because the rescue below
  // matches *descriptors*: with it off no track has one, and GetDescriptors
  // would read Track::descriptors_.back() on an empty vector.
  if (match_dropped_tracks_ && extract_descriptor_ &&
      (newly_dropped_tracks.size() > 0) &&
      (kps.size() > 0))
  {

    // Get matrix of old descriptors
    cv::Mat newly_dropped_descriptors = GetDescriptors(newly_dropped_tracks);

    // Attempt to rescue newly-dropped descriptors with brute-force feature
    // matching.
    // query = newly-dropped descriptors
    // train = just-found descriptors
    std::vector<std::vector<cv::DMatch>> matches;
    matcher_->knnMatch(newly_dropped_descriptors, descriptors, matches, 1,
                       cv::noArray(), true);
    for (int i=0; i<matches.size(); i++) {
      cv::DMatch D = matches[i][0];

      // Check that descriptor distance and pixel displacement are small
      // enough
      bool descriptor_distance_check_passed =
        CheckDescriptorDistance(D.distance, descriptor_distance_thresh_);
      bool pixel_displacement_check_passed =
        CheckPixelDisplacement(kps[D.trainIdx],
                               newly_dropped_tracks[D.queryIdx]->back(),
                               max_pixel_displacement_);
      
      // check reprojection error
      bool reprojection_error_check_passed;
      if (!check_homography) {
        reprojection_error_check_passed = true;
      } else {
        // `back()` (the last tracked pixel), not `keypoint().pt` -- the latter
        // is where the feature was *first* detected, possibly hundreds of
        // frames ago, and `H` is a frame-to-frame homography.
        const Vec2 &last_pos = newly_dropped_tracks[D.queryIdx]->back();
        reprojection_error_check_passed =
          CheckHomography(cv::Point2f(last_pos(0), last_pos(1)),
                          kps[D.trainIdx].pt, H,
                          outlier_rejection_reproj_thresh_);
      }

      if (descriptor_distance_check_passed &&
          pixel_displacement_check_passed &&
          reprojection_error_check_passed)
      {
        matched[D.trainIdx] = true;
        matchIdx[D.trainIdx] = D.queryIdx;
        int fid = newly_dropped_tracks[D.queryIdx]->id();
      }
    }
  }

  // Select keypoints. Selection is done on the integer detections and records
  // its decisions rather than acting on them, so that the sub-pixel refinement
  // below can run once on the (at most ~num_features_max) chosen points instead
  // of on the whole detection set, which is an order of magnitude larger. The
  // mask bookkeeping is unchanged -- it is still updated inside the loop, so a
  // later keypoint still sees the effect of an earlier acceptance.
  struct Selection {
    int kp_index;
    FeaturePtr rescued; // non-null iff this keypoint revives a dropped track
  };
  std::vector<Selection> selected;
  selected.reserve(std::max(num_to_add, 0));

  for (int i = 0; i < kps.size(); ++i) {
    const cv::KeyPoint &kp = kps[i];
    if (MaskValid(mask_, kp.pt.x, kp.pt.y)) {

      if (match_dropped_tracks_ && matched[i] &&
          newly_dropped_tracks[matchIdx[i]]) {
        int idx = matchIdx[i];
        selected.push_back({i, newly_dropped_tracks[idx]});
        // Tell the caller this one was rescued, so it does not get marked
        // DROPPED again. Clearing the slot rather than erasing it keeps the
        // `matchIdx` indices above valid.
        newly_dropped_tracks[idx] = nullptr;
        MaskOut(mask_, kp.pt.x, kp.pt.y, mask_size_);
        --num_to_add;
        continue;
      }

      // Didn't match to a previously-dropped track, so this becomes a new
      // feature.
      selected.push_back({i, nullptr});

      // mask out
      MaskOut(mask_, kp.pt.x, kp.pt.y, mask_size_);
      --num_to_add;
    }
    if (num_to_add <= 0 || kp.response < 5)
      break;
  }

  // Refine the selected detections to sub-pixel accuracy. FAST reports integer
  // coordinates, and for a *new* feature that pixel is the anchor from which
  // (X/Z, Y/Z) is initialized and against which every later observation of the
  // track is compared -- so the +-0.5 px quantization is a per-track bias, not
  // noise that averages away over the track. No-op unless `subpix_refine_`.
  std::vector<cv::Point2f> pts;
  pts.reserve(selected.size());
  for (const auto &s : selected) {
    pts.push_back(kps[s.kp_index].pt);
  }
  timer_.Tick("detect-subpix");
  RefineSubPix(img, pts);
  timer_.Tock("detect-subpix");

  for (size_t j = 0; j < selected.size(); ++j) {
    const int i = selected[j].kp_index;
    cv::KeyPoint kp = kps[i];
    kp.pt = pts[j];

    if (selected[j].rescued) {
      FeaturePtr f1 = selected[j].rescued;
      if (differential_) {
        f1->SetDescriptor(descriptors.row(i));
      }
      f1->UpdateTrack(kp.pt.x, kp.pt.y);
      f1->SetKeypoint(kp);
      f1->SetTrackStatus(TrackStatus::TRACKED);
      LOG(INFO) << "Potentially rescued dropped feature #" << f1->id();
      continue;
    }

    FeaturePtr f = Feature::Create(kp.pt.x, kp.pt.y);
    features_.push_back(f);
    num_new_detections_++;
    if (extract_descriptor_) {
      f->SetDescriptor(descriptors.row(i));
    }
    f->SetKeypoint(kp);
  }
}


void Tracker::Update(const cv::Mat &image) {
  if (tracker_type_ == TrackerType::LK) {
    UpdateLK(image);
  } else {
    UpdateMatch(image);
  }
}


void Tracker::UpdateStereo(const cv::Mat &image, const cv::Mat &image_r) {
  if (image_r.empty()) {
    LOG(FATAL) << "UpdateStereo called with an empty right image";
  }
  if (image_r.rows != image.rows || image_r.cols != image.cols) {
    LOG(FATAL) << "stereo images differ in size: left " << image.cols << "x"
               << image.rows << " vs right " << image_r.cols << "x"
               << image_r.rows;
  }
  // The right image gets exactly the same photometric treatment as the left:
  // left->right KLT compares patches across the two, so equalizing only one of
  // them would break the brightness-constancy assumption it rests on. Under
  // `DETECT` neither is equalized -- detection only ever happens on the left --
  // so the pair stays consistent and the right image's equalization, which no
  // detector ever reads, is not computed at all.
  timer_.Tick("equalize-right");
  const cv::Mat &gray_r = ToGray(image_r, img_r_gray_);
  img_r_ = equalize_scope_ == EqualizeScope::DETECT
               ? gray_r
               : Equalize(gray_r, img_r_eq_, /*cam=*/1);
  timer_.Tock("equalize-right");
  ++num_stereo_frames_;

  // Left-image temporal tracking is exactly the monocular path: the stereo run
  // reaches this point having done bit-identical work to a mono run.
  Update(image);

  // Then attach a right observation to whichever left features survived.
  timer_.Tick("stereo-match");
  MatchStereo();
  timer_.Tock("stereo-match");
}


void Tracker::MatchStereo() {
  // Every feature starts the frame with no right observation, so `has_right()`
  // can only ever mean "matched in the current frame". Without this, a feature
  // that matched at frame k and failed at k+1 would feed the filter a stale
  // observation paired with a fresh left one -- a large, silent inconsistency.
  for (auto f : features_) {
    f->ClearRightObs();
  }

  auto rig = StereoRig::instance();
  if (rig == nullptr) {
    return;
  }

  // Only features that were successfully tracked (or just created) this frame
  // have a meaningful current-frame pixel location.
  std::vector<FeaturePtr> vf;
  std::vector<cv::Point2f> pts_l;
  vf.reserve(features_.size());
  pts_l.reserve(features_.size());
  for (auto f : features_) {
    auto st = f->track_status();
    if (st != TrackStatus::TRACKED && st != TrackStatus::CREATED) {
      continue;
    }
    const Vec2 &xp = f->xp();
    vf.push_back(f);
    pts_l.emplace_back(xp(0), xp(1));
  }
  num_stereo_attempted_ += static_cast<int>(vf.size());
  if (vf.empty()) {
    return;
  }

  // The left pyramid is the one temporal tracking just built, provided the window
  // matches and it is at least as deep as the stereo search needs -- which it is
  // unless `stereo_matching` overrides them, since they default to the KLT values.
  // `pyramid_` holds the *current* frame's pyramid after `UpdateLK`'s swap, but
  // not on the paths that return before it, hence the flag rather than an
  // assumption.
  //
  // A *deeper* left pyramid than `stereo_max_level_` is fine and is the point of
  // the `<=`: `calcOpticalFlowPyrLK` clamps its `maxLevel` down to the shallower
  // of the two pyramids it is handed (`lkpyramid.cpp`, the `levels1 < maxLevel`
  // and `levels2 < maxLevel` tests), so a shallow right pyramid plus an explicit
  // `stereo_max_level_` is what actually decides how many levels get searched.
  // Without the `<=` a config that shortens only the stereo search would pay for a
  // second full left pyramid to throw most of it away.
  const bool reuse_left = pyramid_is_current_ &&
                          stereo_win_size_ == win_size_ &&
                          stereo_max_level_ <= max_level_;
  timer_.Tick("stereo-pyramid");
  std::vector<cv::Mat> pyr_l_own, pyr_r;
  if (!reuse_left) {
    cv::buildOpticalFlowPyramid(img_, pyr_l_own,
                                cv::Size(stereo_win_size_, stereo_win_size_),
                                stereo_max_level_);
  }
  const std::vector<cv::Mat> &pyr_l = reuse_left ? pyramid_ : pyr_l_own;
  cv::buildOpticalFlowPyramid(img_r_, pyr_r,
                              cv::Size(stereo_win_size_, stereo_win_size_),
                              stereo_max_level_);
  timer_.Tock("stereo-pyramid");

  cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                            max_iter_, eps_);

  // Left -> right. Unseeded, the best guess available is the left location itself,
  // since the disparity depends on the unknown depth. With
  // `seed_prev_disparity`, an already-matched feature is seeded with the offset
  // that worked last frame, which is a far better guess -- disparity is smooth in
  // depth and depth is smooth in time -- and is what makes a shallow
  // `stereo_max_level_` safe.
  std::vector<cv::Point2f> pts_r;
  // Parallel to `vf`, so the jump gate below does not repeat the hash lookup.
  std::vector<cv::Point2f> prev_d;
  std::vector<uint8_t> have_prev;
  int lr_flags = 0;
  if (stereo_seed_prev_disparity_) {
    pts_r.reserve(vf.size());
    prev_d.resize(vf.size(), cv::Point2f(0.f, 0.f));
    have_prev.assign(vf.size(), 0);
    for (size_t i = 0; i < vf.size(); ++i) {
      auto it = stereo_prev_disparity_.find(vf[i]->id());
      if (it == stereo_prev_disparity_.end()) {
        pts_r.push_back(pts_l[i]);
      } else {
        prev_d[i] = it->second;
        have_prev[i] = 1;
        pts_r.emplace_back(pts_l[i].x + it->second.x, pts_l[i].y + it->second.y);
      }
    }
    lr_flags = cv::OPTFLOW_USE_INITIAL_FLOW;
  }
  std::vector<uint8_t> status_lr;
  // `cv::noArray()` for the error: nothing here reads it, and asking for it makes
  // OpenCV run an extra full-window photometric pass per point at level 0. It is
  // not merely unused -- see the note in `UpdateLK` for why dropping it leaves
  // `status` unchanged for every point this function can accept.
  timer_.Tick("stereo-klt");
  cv::calcOpticalFlowPyrLK(pyr_l, pyr_r, pts_l, pts_r, status_lr, cv::noArray(),
                           cv::Size(stereo_win_size_, stereo_win_size_),
                           stereo_max_level_, criteria, lr_flags);
  timer_.Tock("stereo-klt");

  // Right -> left, for the circular-consistency check. Running it on the whole
  // batch is cheaper than filtering first and re-entering OpenCV, and the
  // rejected entries are simply ignored below.
  std::vector<cv::Point2f> pts_l_back;
  std::vector<uint8_t> status_rl;
  if (stereo_back_track_) {
    cv::calcOpticalFlowPyrLK(pyr_r, pyr_l, pts_r, pts_l_back, status_rl,
                             cv::noArray(),
                             cv::Size(stereo_win_size_, stereo_win_size_),
                             stereo_max_level_, criteria);
  }

  auto cam_l = Camera::instance(0);
  auto cam_r = Camera::instance(1);

  // Rebuilt from scratch each frame, so a disparity can only ever be one frame
  // old and the table cannot outlive the features it describes.
  std::unordered_map<int, cv::Point2f> disparity_next;
  if (stereo_seed_prev_disparity_) {
    disparity_next.reserve(vf.size() * 2);
  }

  for (size_t i = 0; i < vf.size(); ++i) {
    if (!status_lr[i] || (stereo_back_track_ && !status_rl[i])) {
      ++num_stereo_rejected_klt_;
      continue;
    }
    // Outside the right image (KLT will happily report a point past the border).
    if (pts_r[i].x < 0 || pts_r[i].y < 0 || pts_r[i].x >= img_r_.cols ||
        pts_r[i].y >= img_r_.rows) {
      ++num_stereo_rejected_klt_;
      continue;
    }

    const number_t dx = pts_r[i].x - pts_l[i].x;
    const number_t dy = pts_r[i].y - pts_l[i].y;
    const number_t disparity = std::sqrt(dx * dx + dy * dy);
    if (disparity < stereo_min_disparity_ ||
        disparity > stereo_max_disparity_) {
      ++num_stereo_rejected_disparity_;
      continue;
    }

    // Temporal disparity consistency: a real feature's disparity cannot jump.
    if (std::isfinite(stereo_max_disparity_jump_) && have_prev[i]) {
      const number_t jx = dx - prev_d[i].x;
      const number_t jy = dy - prev_d[i].y;
      if (std::sqrt(jx * jx + jy * jy) > stereo_max_disparity_jump_) {
        // Counted as a circular rejection: the two gates do the same job (reject a
        // left-right correspondence that is not self-consistent) and the shipped
        // configs enable exactly one of them, so keeping one counter keeps the
        // printed diagnostic comparable across the two.
        ++num_stereo_rejected_circular_;
        continue;
      }
    }

    // Circular consistency: the round trip must land back where it started.
    // This is the single most effective filter against repeated texture, since
    // an aliased match is usually not symmetric.
    if (stereo_back_track_) {
      const number_t bx = pts_l_back[i].x - pts_l[i].x;
      const number_t by = pts_l_back[i].y - pts_l[i].y;
      if (std::sqrt(bx * bx + by * by) > stereo_circular_thresh_) {
        ++num_stereo_rejected_circular_;
        continue;
      }
    }

    // Epipolar check on unprojected bearings. Doing it in normalized
    // coordinates (not pixels) is what makes a single threshold valid across
    // the whole fisheye field.
    Vec2 xc_l = cam_l->UnProject(Vec2{pts_l[i].x, pts_l[i].y});
    Vec2 xc_r = cam_r->UnProject(Vec2{pts_r[i].x, pts_r[i].y});
    if (rig->EpipolarResidual(xc_l, xc_r) > stereo_epipolar_thresh_) {
      ++num_stereo_rejected_epipolar_;
      continue;
    }

    vf[i]->SetRightObs(Vec2{pts_r[i].x, pts_r[i].y});
    ++num_stereo_matched_;
    if (stereo_seed_prev_disparity_) {
      disparity_next.emplace(vf[i]->id(), cv::Point2f(dx, dy));
    }
  }

  if (stereo_seed_prev_disparity_) {
    stereo_prev_disparity_.swap(disparity_next);
  }
}


void Tracker::UpdateMatch(const cv::Mat &image) {
  // This path detects on every frame, so `DETECT` has nothing to defer and the
  // two scopes coincide.
  const cv::Mat &src = Equalize(ToGray(image, img_gray_), img_eq_);
  img_ = src.clone();
  if (normalize_) {
    cv::normalize(src, img_, 0, 255, cv::NORM_MINMAX);
  }
  // This path builds no pyramid, so anything still in `pyramid_` belongs to an
  // older frame and `MatchStereo` must not reuse it.
  pyramid_is_current_ = false;

  // detect features in the new image
  std::vector<cv::KeyPoint> new_kps;
  detector_->detect(img_, new_kps, cv::noArray());
  // sort
  std::sort(new_kps.begin(), new_kps.end(),
            [](const cv::KeyPoint &kp1, const cv::KeyPoint &kp2) {
              return kp1.response > kp2.response;
            });

  cv::Mat new_descriptors;
  new_descriptors.reserveBuffer(new_kps.size() * extractor_->descriptorSize());
  extractor_->compute(img_, new_kps, new_descriptors);

  std::vector<FeaturePtr> feature_vec{features_.begin(), features_.end()};

  std::vector<bool> new_kp_matched(new_kps.size(), false);
  std::vector<bool> existing_feature_matched(feature_vec.size(), false);

  // if initialized, then match descriptors to existing features
  if (initialized_) {

    cv::Mat existing_descriptors = GetDescriptors(feature_vec);

    // query descriptors = existing kps/descriptors
    // train descriptors = new kps/descriptors
    std::vector<std::vector<cv::DMatch>> matches;
    matcher_->knnMatch(existing_descriptors, new_descriptors, matches, 1,
                       cv::noArray(), true);

    // Check matches for descriptor distance, pixel displacement
    // outlier rejection -- mark status of each one
    std::vector<uint8_t> match_status(matches.size(), 0);

    for (int i=0; i<matches.size(); i++) {
      cv::DMatch D = matches[i][0];

      // Check that descriptor distance and pixel displacement are small
      // enough
      bool descriptor_distance_check_passed =
        CheckDescriptorDistance(D.distance, descriptor_distance_thresh_);
      bool pixel_displacement_check_passed =
        CheckPixelDisplacement(new_kps[D.trainIdx],
                               feature_vec[D.queryIdx]->back(),
                               max_pixel_displacement_);

      match_status[i] = uint8_t(descriptor_distance_check_passed &&
                                pixel_displacement_check_passed);
    }

   num_failed_to_track_ = feature_vec.size() - matches.size() + num_zeros(match_status);

    // Outlier rejection
    if (do_outlier_rejection_) {
      std::vector<cv::Point2f> pts0;
      std::vector<cv::Point2f> pts1;
      for (int i = 0; i < matches.size(); i++) {
        cv::DMatch D = matches[i][0];
        // `back()`, not `keypoint().pt` -- findHomography must see the
        // previous-frame pixel, not where the feature was first detected.
        const Vec2 &last_pos = feature_vec[D.queryIdx]->back();
        pts0.emplace_back(last_pos(0), last_pos(1));
        pts1.push_back(new_kps[D.trainIdx].pt);
      }
      cv::Mat H;
      OutlierRejection(pts0, pts1, match_status, H);
    }

    // After outlier rejection, mark match status of old and new features and
    // update existing tracks
    for (int i=0; i<matches.size(); i++) {
      if (match_status[i]) {
        cv::DMatch D = matches[i][0];
        new_kp_matched[D.trainIdx] = true;
        existing_feature_matched[D.queryIdx] = true;

        FeaturePtr f = feature_vec[D.queryIdx];
        cv::KeyPoint kp = new_kps[D.trainIdx];
        f->UpdateTrack(Vec2{kp.pt.x, kp.pt.y});
        if (differential_) {
          f->SetDescriptor(new_descriptors.row(D.trainIdx));
        }
        f->SetTrackStatus(TrackStatus::TRACKED);
      }
    }
  }


  // Drop features that weren't matched to a new point
  int num_features_dropped = 0;
  for (int i=0; i<feature_vec.size(); i++) {
    if (!existing_feature_matched[i]) {
      feature_vec[i]->SetTrackStatus(TrackStatus::DROPPED);
      num_features_dropped += 1;
    }
  }

  // Turn rest of detected tracks into a new feature
  int num_to_create = num_features_max_ - feature_vec.size()
    + num_features_dropped;
  num_new_detections_ = 0;
  for (int i=0; i<new_kps.size(); i++) {
    if (num_to_create <= 0) {
      break;
    }

    if (!new_kp_matched[i]) {
      FeaturePtr f = Feature::Create(new_kps[i].pt.x, new_kps[i].pt.y);
      f->SetDescriptor(new_descriptors.row(i));
      f->SetKeypoint(new_kps[i]);
      features_.push_back(f);
      num_new_detections_++;
      num_to_create -= 1;
    }
  }

  initialized_ = true;
}


void Tracker::BuildOwnedPyramid(const cv::Mat &image,
                                std::vector<cv::Mat> &pyramid, int win_size,
                                int max_level) {
  // The middle three arguments are OpenCV's own defaults, spelled out only
  // because the last one cannot be reached without them. See the declaration for
  // why it is forced off.
  cv::buildOpticalFlowPyramid(image, pyramid, cv::Size(win_size, win_size),
                              max_level, /*withDerivatives=*/true,
                              cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT,
                              /*tryReuseInputImage=*/false);
}


void Tracker::UpdateLK(const cv::Mat &image) {
  // `img_` is read only within this call and the `MatchStereo` that follows it,
  // and the caller owns `image` for the duration of the measurement, so the
  // frame does not need its own copy. The old code cloned unconditionally --
  // an allocation plus a 256 kB copy per image on TUM-VI -- and then overwrote
  // the clone when normalizing, so the copy was dead either way.
  //
  // The cost of not copying is that `img_` no longer owns its pixels, which is
  // why `pyramid_` has to (see `BuildOwnedPyramid`).
  // Contrast equalization, when enabled and scoped to `ALL`, happens before
  // everything else: the KLT, the detector and the descriptors then all see the
  // same pixels. Under `DETECT` it is deferred to `DetectionImage()`, so that
  // tracking runs on the raw frame and the equalization is not computed at all on
  // a frame that does not detect.
  timer_.Tick("equalize-left");
  const cv::Mat &gray = ToGray(image, img_gray_);
  const cv::Mat &src = equalize_scope_ == EqualizeScope::DETECT
                           ? gray
                           : Equalize(gray, img_eq_);
  if (normalize_) {
    cv::normalize(src, img_, 0, 255, cv::NORM_MINMAX);
  } else {
    img_ = src;
  }
  timer_.Tock("equalize-left");
  pyramid_is_current_ = false;

  if (!initialized_) {
    rows_ = img_.rows;
    cols_ = img_.cols;
    // `valid_mask_` folds in both the border margin and the field-of-view test;
    // it is what every frame's `mask_` starts from.
    BuildValidMask(0);
    valid_mask_.copyTo(mask_);

    // build image pyramid
    BuildOwnedPyramid(img_, pyramid_, win_size_, max_level_);
    pyramid_is_current_ = true;
    // detect an initial set of features (nothing to rescue on the first frame)
    std::vector<FeaturePtr> no_dropped_tracks;
    DetectLK(DetectionImage(), num_features_max_, no_dropped_tracks, false,
             cv::Mat());
    initialized_ = true;
    return;
  }
  // reset mask
  valid_mask_.copyTo(mask_);

  // build new pyramid
  std::vector<cv::Mat> pyramid;
  timer_.Tick("pyramid");
  BuildOwnedPyramid(img_, pyramid, win_size_, max_level_);
  timer_.Tock("pyramid");

  // prepare for optical flow
  cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                            max_iter_, eps_);

  std::vector<cv::Point2f> pts0, pts1;
  std::vector<uint8_t> status;

  pts0.reserve(features_.size());
  pts1.reserve(pts0.size());

  for (auto f : features_) {
    const Vec2 &pt(f->xp());
    pts0.emplace_back(pt[0], pt[1]);

    // fill in predicted locations
    auto pred = f->pred();
    const bool have_pred = (pred(0) != -1) && (pred(1) != -1);
    if (use_prediction_ && have_pred) {
      pts1.emplace_back(pred(0), pred(1));
    } else {
      pts1.emplace_back(pt[0], pt[1]);
    }
    if (have_pred) {
      f->ResetPred(); // consume it either way, so it cannot go stale
    }
  }

  if (pts0.size() == 0) {
    initialized_ = false;
    return;
  }

  // No error output. The tracking error was never read, and requesting it is not
  // free: OpenCV's `LKTrackerInvoker` then runs a second pass over the whole
  // 15x15 window of every point at level 0 just to accumulate the photometric
  // residual (`lkpyramid.cpp`, the `err && level == 0` block).
  //
  // That block also contains the one place where `err` feeds back into `status`:
  // it clears `status` if the tracked point ends up outside the image *by more
  // than half a window* (`inextPoint.x < -winSize.width || >= J.cols`, i.e.
  // x < -8 or x >= 519 at win 15). Dropping it cannot change any outcome here,
  // because every consumer of `status` in this file already applies a strictly
  // tighter bound: `MaskValid` below rejects anything outside [0, cols), and
  // `MatchStereo` rejects `pts_r` outside the right image and `pts_l_back`
  // farther than `circular_thresh` (1 px) from where it started.
  timer_.Tick("klt");
  cv::calcOpticalFlowPyrLK(pyramid_, pyramid, pts0, pts1, status,
                           cv::noArray(),
                           cv::Size(win_size_, win_size_), max_level_, criteria,
                           cv::OPTFLOW_USE_INITIAL_FLOW);
  timer_.Tock("klt");

  std::vector<cv::KeyPoint> kps;
  cv::Mat descriptors;
  if (extract_descriptor_) {
    std::vector<FeaturePtr> vf{features_.begin(), features_.end()};
    kps.reserve(vf.size());
    descriptors.reserveBuffer(vf.size() * extractor_->descriptorSize());
    for (int i = 0; i < vf.size(); ++i) {
      auto f = vf[i];
      cv::KeyPoint kp =
          f->keypoint(); // preserve all the properties of the initial keypoint
      kp.pt.x = pts1[i].x; // with updated pixel location
      kp.pt.y = pts1[i].y;
      kp.class_id = i;
      kps.push_back(kp);
    }
    extractor_->compute(img_, kps, descriptors);

    // `compute` above compacts `kps` -- BRIEF drops every keypoint within
    // PATCH_SIZE/2 + KERNEL_SIZE/2 = 28 px of the border -- so `i` is no longer
    // the index of the feature in `features_`/`status`/`pts1`. The original
    // index was stashed in `class_id`; use it for `status` too, not just for
    // `vf`. Indexing `status` with `i` dropped an arbitrary innocent track and
    // let the mismatched one through.
    for (int i = 0; i < kps.size(); ++i) {
      const int fi = kps[i].class_id;
      auto f = vf[fi];
      if (descriptor_distance_thresh_ != -1) {
        int dist = cv::norm(f->descriptor(), descriptors.row(i),
                            extractor_->defaultNorm());
        if (dist > descriptor_distance_thresh_) {
          status[fi] = 0; // enforce to be dropped
        } else {
          if (differential_) {
            f->SetDescriptor(descriptors.row(i));
          }
        }
      } else {
        if (differential_) {
          f->SetDescriptor(descriptors.row(i));
        }
      }
    }
  }

  // iterate through features and mark bad ones
  int num_valid_features = 0;
  int i = 0;

  for (auto it = features_.begin(); it != features_.end(); ++it, ++i) {
    FeaturePtr f(*it);

    Vec2 last_pos(f->xp());
    if (status[i]) {
      if (MaskValid(mask_, pts1[i].x, pts1[i].y) &&
          (last_pos - Vec2{pts1[i].x, pts1[i].y}).norm() <
              max_pixel_displacement_) {
        // update track status
        f->SetTrackStatus(TrackStatus::TRACKED);
        f->UpdateTrack(pts1[i].x, pts1[i].y);
        MaskOut(mask_, pts1[i].x, pts1[i].y, mask_size_);
        ++num_valid_features;
      } else {
        // failed to extract descriptors or invalid mask
        status[i] = 0;
      }
    }
  }

  num_new_detections_ = 0;
  num_failed_to_track_ =  num_zeros(status);

  cv::Mat H;
  bool outlier_rejection_success = false;
  if (do_outlier_rejection_) {
    outlier_rejection_success = OutlierRejection(pts0, pts1, status, H);
    num_valid_features -= num_outliers_rejected_;
  }

  if (epipolar_rejection_) {
    // After the mask/displacement pass and after the homography pass, so it only
    // ever sees correspondences the cheaper tests already accepted -- and so the
    // valid-feature budget below is reduced by the union, not double-counted.
    num_epipolar_rejected_ =
        OutlierRejectionEpipolar(pts0, pts1, status, 0);
    num_valid_features -= num_epipolar_rejected_;
    num_failed_to_track_ = num_zeros(status);
  }

  // Mark newly dropped tracks for possible rescue
  std::vector<FeaturePtr> newly_dropped_tracks;
  i = 0;
  for (auto it = features_.begin(); it != features_.end(); ++it, ++i) {
    if (!status[i]) {
      FeaturePtr f(*it);
      newly_dropped_tracks.push_back(f);
    }
  }

  // detect a new set of features
  // this can rescue dropped featuers by matching them to newly detected ones
  if (num_valid_features < num_features_min_) {
    bool check_homography = do_outlier_rejection_ && outlier_rejection_success;
    timer_.Tick("detect-total");
    DetectLK(DetectionImage(), num_features_max_ - num_valid_features,
             newly_dropped_tracks, check_homography, H);
    timer_.Tock("detect-total");
  }

  // Mark every track that DetectLK did not rescue (rescued slots were set to
  // nullptr) as dropped. Dropped features get deleted later in
  // Estimator::ProcessTracks().
  for (auto f: newly_dropped_tracks) {
    if (f) {
      f->SetTrackStatus(TrackStatus::DROPPED);
    }
  }

  // swap buffers ...
  std::swap(pyramid, pyramid_);
  pyramid_is_current_ = true;

}


void Tracker::UpdatePointCloud(const VecXi &feature_ids, const MatX2 &xps)
{
  // Turn input into a hash table for measurements.
  // unmarked points become new features at the end of this function
  std::unordered_map<int, Vec2> measurements;
  std::unordered_map<int, bool> measurement_marked;
  for (int i = 0; i < feature_ids.size(); i++) {
    measurements[feature_ids[i]] = xps.row(i);
    measurement_marked[feature_ids[i]] = false;
  }

  // Save data for possible outlier rejection
  std::vector<cv::Point2f> pts0;
  std::vector<cv::Point2f> pts1;

  // status of existing tracks
  int i = 0;
  int num_dropped = 0;
  std::vector<uint8_t> status(features_.size(), 0);
  for (auto it = features_.begin(); it != features_.end(); ++it, ++i) {
    FeaturePtr f{*it};
    bool existing_feature_seen = (measurements.count(f->id()) > 0);
    if (existing_feature_seen) {
      // distance between current and last point
      bool close_enough =
        CheckPixelDisplacement(measurements[f->id()], f->xp(),
                               max_pixel_displacement_);
      if (close_enough) {
        pts0.push_back(cv::Point2f(f->back()[0], f->back()[1]));
        pts1.push_back(cv::Point2f(measurements[f->id()][0],
                                   measurements[f->id()][1]));
        status[i] = 1;
        f->push_back(measurements[f->id()]);
        f->SetTrackStatus(TrackStatus::TRACKED);
        measurement_marked[f->id()] = true;
      } else {
        status[i] = 0;
        f->SetTrackStatus(TrackStatus::DROPPED);
        num_dropped++;
      }
    } else {
      status[i] = 0;
      f->SetTrackStatus(TrackStatus::DROPPED);
      num_dropped++;
    }
  }

  // Outlier Rejection
  if (do_outlier_rejection_) {
    cv::Mat H;
    OutlierRejection(pts0, pts1, status, H);
  }

  // Create new tracks
  int num_to_add = num_features_max_ - features_.size()
    + num_dropped + num_outliers_rejected_;
  for (i = 0; i < feature_ids.size(); i++) {
    if (num_to_add <= 0) {
      break;
    }

    int fid = feature_ids[i];
    if (!measurement_marked[fid]) {
      Vec2 xp = measurements[fid];
      FeaturePtr f = Feature::PointCloudWorldCreate(fid, xp(0), xp(1));
      f->SetKeypoint(cv::KeyPoint(xp(0), xp(1), 0.0));
      features_.push_back(f);
      num_to_add--; // only a *new* feature consumes the budget
    }
  }
}


bool Tracker::OutlierRejection(const std::vector<cv::Point2f> pts0,
                               const std::vector<cv::Point2f> pts1,
                               std::vector<uint8_t>& match_status,
                               cv::Mat& H)
{
  CHECK(pts0.size() == pts1.size());

  // Reset before any early return: this is a member that persists across
  // frames, and callers subtract it from their valid-feature count
  // unconditionally, so a stale value corrupted the next frame's budget.
  num_outliers_rejected_ = 0;

  // Check that we have at least 4 valid points
  if (sum_total(match_status) < 4) {
    return false;
  }

  // Remove all points that are already marked as rejected
  std::vector<cv::Point2f> pts0_valid;
  std::vector<cv::Point2f> pts1_valid;
  std::vector<int> idx_map; // maps input idx to _valid idx
  int cnt = 0;
  for (int i=0; i<pts0.size(); i++) {
    if (match_status[i] != 0) {
      pts0_valid.push_back(pts0[i]);
      pts1_valid.push_back(pts1[i]);
      idx_map.push_back(cnt);
      cnt++;
    } else {
      idx_map.push_back(-1);
    }
  }

  // Call OpenCV
  cv::Mat inlier_outlier_mask(1, pts0_valid.size(), CV_8UC1);
  H = cv::findHomography(
    pts0_valid, pts1_valid, outlier_rejection_method_,
    outlier_rejection_reproj_thresh_, inlier_outlier_mask,
    outlier_rejection_maxiters_, outlier_rejection_confidence_);

  // When the registrator cannot fit a model, OpenCV releases `H` and replaces
  // the mask with an empty one (the pre-allocation above is reallocated away).
  // Reporting success then made the callers multiply by an empty `H` and index
  // into a null mask -- `cv::Mat::at` only checks bounds under CV_DbgAssert, so
  // in a release build that is a straight segfault.
  if (H.empty() || inlier_outlier_mask.total() != pts0_valid.size()) {
    LOG(WARNING) << "findHomography failed to fit a model; skipping outlier "
                    "rejection for this frame";
    H.release();
    return false;
  }

  // record number of rejected outliers
  num_outliers_rejected_ = num_zeros(inlier_outlier_mask);

  // Mark outliers in `match_status`
  for (int i=0; i<pts0.size(); i++) {
    if ((match_status[i] != 0) && (idx_map[i] > -1)) {
      if (inlier_outlier_mask.at<uchar>(idx_map[i]) == 0) {
        match_status[i] = 0;
      }
    }
  }

  return true;
}


int Tracker::OutlierRejectionEpipolar(const std::vector<cv::Point2f> &pts0,
                                      const std::vector<cv::Point2f> &pts1,
                                      std::vector<uint8_t> &match_status,
                                      int cam_id) {
  CHECK(pts0.size() == pts1.size());
  CHECK(match_status.size() == pts0.size());

  auto cam = Camera::instance(cam_id);
  if (cam == nullptr) {
    return 0;
  }

  // Unproject the surviving correspondences. `idx` maps back into the caller's
  // indexing, which is parallel to `features_`.
  std::vector<cv::Point2f> n0, n1;
  std::vector<int> idx;
  n0.reserve(pts0.size());
  n1.reserve(pts0.size());
  idx.reserve(pts0.size());
  for (int i = 0; i < static_cast<int>(pts0.size()); ++i) {
    if (!match_status[i]) {
      continue;
    }
    Vec2 x0 = cam->UnProject(Vec2{static_cast<number_t>(pts0[i].x),
                                  static_cast<number_t>(pts0[i].y)});
    Vec2 x1 = cam->UnProject(Vec2{static_cast<number_t>(pts1[i].x),
                                  static_cast<number_t>(pts1[i].y)});
    if (!x0.allFinite() || !x1.allFinite() ||
        x0.norm() > epipolar_max_norm_ || x1.norm() > epipolar_max_norm_) {
      continue;
    }
    n0.emplace_back(x0(0), x0(1));
    n1.emplace_back(x1(0), x1(1));
    idx.push_back(i);
  }

  if (static_cast<int>(n0.size()) < epipolar_min_pts_) {
    return 0;
  }

  // The threshold is quoted in pixels for legibility; the fit is in normalized
  // coordinates, so divide by the larger focal length (the conservative choice:
  // it makes the band narrower along the better-resolved axis).
  const number_t f = std::max(cam->fx(), cam->fy());
  if (!(f > 0)) {
    return 0;
  }

  cv::Mat mask;
  cv::Mat F = cv::findFundamentalMat(n0, n1, cv::FM_RANSAC,
                                     epipolar_thresh_px_ / f,
                                     epipolar_confidence_, mask);
  // findFundamentalMat returns an empty matrix (and an empty or, with the
  // 7-point solver, a multi-row mask) when it cannot fit a model. Treat that as
  // "no information this frame" rather than rejecting everything.
  if (F.empty() || mask.total() != n0.size()) {
    return 0;
  }

  int rejected = 0;
  for (int k = 0; k < static_cast<int>(idx.size()); ++k) {
    if (mask.at<uint8_t>(k) == 0) {
      match_status[idx[k]] = 0;
      ++rejected;
    }
  }
  ++num_epipolar_frames_;
  num_epipolar_total_rejected_ += rejected;
  return rejected;
}


////////////////////////////////////////
// helpers
////////////////////////////////////////
void ResetMask(cv::Mat mask) { mask.setTo(255); }

void MaskOut(cv::Mat mask, number_t x, number_t y, int mask_size) {
  // Not `static`: that froze half_size at the first call's mask_size and
  // silently ignored the parameter (and the header's default argument) forever
  // after -- also a data race on first use when async_run is on.
  const int half_size = (mask_size >> 1);
  cv::rectangle(mask, cv::Point2d(x - half_size, y - half_size),
                cv::Point2d(x + half_size, y + half_size), cv::Scalar(0), -1);
}

bool MaskValid(const cv::Mat &mask, number_t x, number_t y) {
  int col = static_cast<int>(x);
  int row = static_cast<int>(y);
  if (col < 0 || col >= mask.cols || row < 0 || row >= mask.rows)
    return false;
  return static_cast<bool>(mask.at<uint8_t>(row, col));
}


cv::Mat GetDescriptors(std::vector<FeaturePtr> fvec)
{
  // fvec[0] was indexed unconditionally, and descriptor() is
  // descriptors_.back(); neither is safe without these two checks.
  if (fvec.empty() || !fvec[0]->has_descriptor()) {
    return cv::Mat();
  }

  int d_size = fvec[0]->descriptor().cols;
  int d_type = fvec[0]->descriptor().type();

  cv::Mat descriptors(fvec.size(), d_size, d_type);
  int i = 0;
  for (auto f: fvec) {
    f->descriptor().copyTo(descriptors.row(i));
    i++;
  }
  return descriptors;
}

bool CheckDescriptorDistance(number_t descriptor_distance,
                             number_t max_distance)
{
  if (max_distance > 0) {
    return (descriptor_distance < max_distance);
  } else {
    return true;
  }
}

bool CheckPixelDisplacement(const Vec2 kp1,
                            const Vec2 kp2,
                            const number_t max_displacement)
{
  return ((kp1 - kp2).norm() < max_displacement);
}

bool CheckPixelDisplacement(const cv::KeyPoint kp1,
                            const Vec2 kp2,
                            const number_t max_displacement)
{
  return CheckPixelDisplacement(Vec2{kp1.pt.x, kp1.pt.y},
                                kp2,
                                max_displacement);
}


bool CheckHomography(cv::Point2f p0,
                     cv::Point2f p1,
                     cv::Mat H,
                     number_t reproj_threshold)
{
  if (H.empty()) {
    return false;
  }
  cv::Mat p0_h(cv::Vec3d(p0.x, p0.y, 1.0), true);
  cv::Mat p1_h(cv::Vec3d(p1.x, p1.y, 1.0), true);
  // The reprojection has to be compared against p1 -- the old code computed
  // `Hp0` and then measured ||p0 - p1||, so `H` had no effect at all and this
  // degenerated into a (much tighter) duplicate of the pixel-displacement
  // check. The homogeneous coordinate also has to be divided out.
  cv::Mat Hp0 = H * p0_h;
  const double w = Hp0.at<double>(2);
  if (std::abs(w) < 1e-12) {
    return false;
  }
  Hp0 /= w;
  number_t dist = cv::norm(Hp0, p1_h, cv::NORM_L2);
  return (dist < reproj_threshold);
}



} // namespace xivo
