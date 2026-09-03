#include "init_detect.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "camera_manager.h"

namespace xivo {

Mat3 FitRotationWahba(const std::vector<Vec3> &u1, const std::vector<Vec3> &u2,
                      int iters, number_t huber_rad) {
  const size_t n = std::min(u1.size(), u2.size());
  Mat3 R = Mat3::Identity();
  if (n < 3)
    return R;

  std::vector<number_t> w(n, 1.0);
  for (int it = 0; it < std::max(1, iters); ++it) {
    Mat3 M = Mat3::Zero();
    for (size_t i = 0; i < n; ++i)
      M += w[i] * u2[i] * u1[i].transpose();

    Eigen::JacobiSVD<Mat3> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    Mat3 V = svd.matrixV();
    // The unconstrained minimiser of |u2 - R u1|^2 over 3x3 matrices is U V',
    // which can be a reflection. Flipping the sign of the least-significant
    // singular direction is the closest rotation to it.
    Vec3 d{1.0, 1.0, (U * V.transpose()).determinant() < 0 ? -1.0 : 1.0};
    R = U * d.asDiagonal() * V.transpose();

    if (it + 1 == std::max(1, iters))
      break;
    for (size_t i = 0; i < n; ++i) {
      const number_t c =
          std::max<number_t>(-1.0, std::min<number_t>(1.0, (R * u1[i]).dot(u2[i])));
      const number_t ang = std::acos(c);
      w[i] = ang < huber_rad ? 1.0 : huber_rad / std::max(ang, number_t(1e-12));
    }
  }
  return R;
}

MotionDetector::MotionDetector() : MotionDetector(Options{}) {}

MotionDetector::MotionDetector(const Options &opt) : opt_(opt) {}

void MotionDetector::Reset() {
  imu_.clear();
  flow_.clear();
  prev_img_.release();
  prev_pts_.clear();
  have_prev_ = false;
}

void MotionDetector::AddImu(number_t t, const Vec3 &gyro, const Vec3 &accel) {
  imu_.push_back(ImuSample{t, gyro, accel});
  // Bound the buffer: nothing beyond the horizon can influence the verdict, and
  // the detector must not become a memory leak if a caller keeps feeding it.
  while (!imu_.empty() && t - imu_.front().t > opt_.horizon_sec + opt_.window_sec)
    imu_.pop_front();
}

number_t MotionDetector::ImuSpan() const {
  return imu_.size() < 2 ? 0 : imu_.back().t - imu_.front().t;
}

void MotionDetector::AddImage(number_t t, const cv::Mat &gray) {
  cv::Mat img;
  if (gray.channels() == 1)
    img = gray;
  else
    cv::cvtColor(gray, img, cv::COLOR_BGR2GRAY);

  auto *cam = CameraManager::instance(opt_.cam_id);

  if (have_prev_ && prev_pts_.size() >= static_cast<size_t>(opt_.min_tracks) &&
      cam != nullptr) {
    std::vector<cv::Point2f> nxt, back;
    std::vector<uchar> st, st2;
    std::vector<float> err;
    const cv::Size win(21, 21);
    const auto crit = cv::TermCriteria(
        cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01);
    cv::calcOpticalFlowPyrLK(prev_img_, img, prev_pts_, nxt, st, err, win, 3,
                             crit);
    cv::calcOpticalFlowPyrLK(img, prev_img_, nxt, back, st2, err, win, 3, crit);

    std::vector<Vec3> u1, u2;
    std::vector<cv::Point2f> kept2;
    u1.reserve(prev_pts_.size());
    u2.reserve(prev_pts_.size());
    kept2.reserve(prev_pts_.size());
    for (size_t i = 0; i < prev_pts_.size(); ++i) {
      if (!st[i] || !st2[i])
        continue;
      // Forward-backward consistency. A track that does not come home is
      // rejected outright: its residual under the rotation fit would be
      // indistinguishable from parallax, and parallax is the whole signal.
      const number_t fb = cv::norm(back[i] - prev_pts_[i]);
      if (fb > opt_.fb_thresh)
        continue;
      Vec2 n1 = cam->UnProject(Vec2{prev_pts_[i].x, prev_pts_[i].y});
      Vec2 n2 = cam->UnProject(Vec2{nxt[i].x, nxt[i].y});
      u1.push_back(Vec3{n1(0), n1(1), 1.0}.normalized());
      u2.push_back(Vec3{n2(0), n2(1), 1.0}.normalized());
      kept2.push_back(nxt[i]);
    }

    if (u1.size() >= static_cast<size_t>(opt_.min_tracks)) {
      const Mat3 R_fit = FitRotationWahba(u1, u2, 4, opt_.huber_rad);

      // Score in pixels, by pushing the rotated bearing back through the real
      // camera model. Scaling an angular residual by the focal length would be
      // wrong by a factor of several toward the edge of a wide-angle image, and
      // this threshold has to mean the same thing on both datasets.
      std::vector<number_t> res;
      res.reserve(u1.size());
      for (size_t i = 0; i < u1.size(); ++i) {
        const Vec3 r = R_fit * u1[i];
        if (r(2) < 0.05)
          continue; // projection is not well posed near the horizon
        const Vec2 px = cam->Project(Vec2{r(0) / r(2), r(1) / r(2)});
        res.push_back((px - Vec2{kept2[i].x, kept2[i].y}).norm());
      }
      if (res.size() >= static_cast<size_t>(opt_.min_tracks)) {
        std::nth_element(res.begin(), res.begin() + res.size() / 2, res.end());
        const number_t med = res[res.size() / 2];

        // Free byproduct: how far the gyro's integrated rotation is from the
        // one the images actually support. While the rig is still, that
        // difference *is* the gyro bias, so it seeds the bundle adjustment with
        // something better than zero.
        number_t bias = 0;
        Mat3 R_gyro = Mat3::Identity();
        int used = 0;
        for (size_t k = 1; k < imu_.size(); ++k) {
          if (imu_[k].t <= prev_t_ || imu_[k].t > t)
            continue;
          const number_t dt = imu_[k].t - imu_[k - 1].t;
          R_gyro = R_gyro *
                   SO3::exp(0.5 * (imu_[k].gyro + imu_[k - 1].gyro) * dt).matrix();
          ++used;
        }
        if (used > 0 && t > prev_t_) {
          // Compare in the camera frame. `R_fit` maps frame-1 bearings into
          // frame-2 coordinates, so it is R_{c2<-c1}; the gyro integral above is
          // R_{b1<-b2}, so it has to be conjugated by the extrinsics before the
          // two can be multiplied. Skipping that step is not a small error: the
          // conjugation preserves the gyro rotation's own angle but not the angle
          // of the product, so with TUM-VI's camera flip it reports 0.3 rad/s
          // where the truth is 0.02.
          const Mat3 dR = opt_.Rbc.transpose() * R_gyro * opt_.Rbc * R_fit;
          const number_t c = std::max<number_t>(
              -1.0, std::min<number_t>(1.0, (dR.trace() - 1.0) / 2.0));
          bias = std::acos(c) / (t - prev_t_);
        }
        flow_.push_back(FlowSample{t, med, bias});
      }
    }
  }

  // Re-detect every frame rather than maintaining long tracks: the statistic is
  // frame-to-frame flow, and re-detecting keeps the sample count up over a
  // window that is only ten to forty frames long.
  prev_img_ = img.clone();
  prev_t_ = t;
  prev_pts_.clear();
  cv::goodFeaturesToTrack(img, prev_pts_, opt_.max_tracks, 0.01, 12.0);
  have_prev_ = true;
}

number_t MotionDetector::MinWindowAccelSd() const {
  number_t best = -1;
  const size_t n = imu_.size();
  size_t i = 0;
  while (i < n && imu_[i].t - imu_.front().t <= opt_.horizon_sec) {
    size_t j = i;
    while (j < n && imu_[j].t - imu_[i].t < opt_.window_sec)
      ++j;
    const size_t m = j - i;
    if (m >= static_cast<size_t>(opt_.min_window_samples)) {
      Vec3 mean = Vec3::Zero();
      for (size_t k = i; k < j; ++k)
        mean += imu_[k].accel;
      mean /= static_cast<number_t>(m);
      number_t ss = 0;
      for (size_t k = i; k < j; ++k)
        ss += (imu_[k].accel - mean).squaredNorm();
      // Pooled over the three axes -- the same statistic OpenVINS'
      // StaticInitializer compares against init_imu_thresh.
      const number_t sd = std::sqrt(ss / static_cast<number_t>(m));
      if (best < 0 || sd < best)
        best = sd;
    }
    i += std::max<size_t>(1, m / 4); // slide by a quarter window
  }
  return best;
}

number_t MotionDetector::MinWindowFlow() const {
  if (flow_.empty())
    return -1;
  number_t best = -1;
  for (size_t i = 0; i < flow_.size(); ++i) {
    size_t j = i;
    number_t sum = 0;
    while (j < flow_.size() && flow_[j].t - flow_[i].t < opt_.window_sec) {
      sum += flow_[j].px;
      ++j;
    }
    const int m = static_cast<int>(j - i);
    // At least a few frame pairs, so one bad KLT step cannot decide the verdict.
    if (m < 3)
      continue;
    const number_t avg = sum / m;
    if (best < 0 || avg < best)
      best = avg;
  }
  // Fewer than three pairs anywhere: fall back to the median of what there is
  // rather than declaring no opinion, since a short window is still evidence.
  if (best < 0) {
    std::vector<number_t> v;
    v.reserve(flow_.size());
    for (const auto &f : flow_)
      v.push_back(f.px);
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    best = v[v.size() / 2];
  }
  return best;
}

bool MotionDetector::Ready() const {
  if (imu_.size() < static_cast<size_t>(opt_.min_window_samples))
    return false;
  // Enough of a window for both cues to have been minimised over more than one
  // placement, or the horizon is up and we decide with what we have.
  return ImuSpan() >= opt_.horizon_sec ||
         (ImuSpan() >= 2 * opt_.window_sec && flow_.size() >= 6);
}

MotionVerdict MotionDetector::Classify() const {
  MotionVerdict v;
  v.imu_samples = static_cast<int>(imu_.size());
  v.frame_pairs = static_cast<int>(flow_.size());
  v.accel_sd = MinWindowAccelSd();
  v.flow_px = MinWindowFlow();
  if (!flow_.empty()) {
    std::vector<number_t> b;
    b.reserve(flow_.size());
    for (const auto &f : flow_)
      b.push_back(f.bias);
    std::nth_element(b.begin(), b.begin() + b.size() / 2, b.end());
    v.gyro_bias_hint = b[b.size() / 2];
  }

  if (!Ready() || v.accel_sd < 0) {
    v.kind = MotionVerdict::kUndecided;
    return v;
  }

  // Each cue is consulted only where it is sound, rather than combined.
  //
  // The visual cue decides whenever it has an opinion, because it is the only
  // one that responds to *translation*. The accelerometer is the fallback for
  // when tracking has too few survivors to fit a rotation to -- not a second
  // trigger, and not a veto.
  //
  // Both alternatives are worse, for opposite reasons:
  //
  //   * `imu && flow` would reintroduce the blind spot this detector exists to
  //     remove. An accelerometer cannot see constant velocity at all, so
  //     requiring it to agree lets a steady glide through as static.
  //   * `imu || flow` produces false positives on a rig that is stationary but
  //     steadily rotating. Gravity then sweeps through the body frame, and at
  //     0.3 rad/s that alone is about 0.42 m/s^2 of accelerometer sd -- past any
  //     threshold that MH_02's 0.497 has to clear. TUM-VI room4 averages
  //     0.32 rad/s at init, so this is ordinary handheld behaviour, and the cost
  //     is a bundle adjustment on a window with no parallax in it.
  //
  // Deferring to vision costs nothing in blindness, because the visual cue sees
  // constant velocity, which is the only thing the accelerometer cue was needed
  // for. On the 17 sequences measured, the visual cue alone classifies every one
  // correctly with a 6.8x margin, against 1.93x for the accelerometer.
  if (v.flow_px >= 0)
    v.kind = v.flow_px > opt_.flow_thresh ? MotionVerdict::kDynamic
                                          : MotionVerdict::kStatic;
  else
    v.kind = v.accel_sd > opt_.imu_thresh ? MotionVerdict::kDynamic
                                          : MotionVerdict::kStatic;
  return v;
}

} // namespace xivo
