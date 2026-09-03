#include "init_window.h"

#include <algorithm>
#include <unordered_map>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "camera_manager.h"

namespace xivo {

InitWindowTracker::InitWindowTracker() : InitWindowTracker(Options{}) {}

InitWindowTracker::InitWindowTracker(const Options &opt) : opt_(opt) {}

void InitWindowTracker::Reset() {
  frame_t_.clear();
  frame_obs_.clear();
  imu_.clear();
  prev_img_.release();
  prev_pts_.clear();
  prev_ids_.clear();
  next_id_ = 0;
  have_prev_ = false;
}

void InitWindowTracker::AddImu(number_t t, const Vec3 &gyro,
                               const Vec3 &accel) {
  imu_.push_back(InitImu{t, gyro, accel});
}

void InitWindowTracker::AddImage(number_t t, const cv::Mat &gray) {
  if (Full())
    return;
  if (!frame_t_.empty() && t - frame_t_.back() < opt_.frame_gap)
    return;
  // An image ahead of the IMU stream cannot be preintegrated to, and one behind
  // its start has no interval either. Both would otherwise be silently
  // extrapolated by `InterpolateImu`'s clamping, which is the right behaviour
  // inside an interval and the wrong one at its edge.
  if (imu_.empty() || t < imu_.front().t)
    return;

  cv::Mat img;
  if (gray.channels() == 1)
    img = gray;
  else
    cv::cvtColor(gray, img, cv::COLOR_BGR2GRAY);

  auto *cam = CameraManager::instance(opt_.cam_id);
  if (cam == nullptr)
    return;

  std::vector<cv::Point2f> pts;
  std::vector<int> ids;

  if (have_prev_ && !prev_pts_.empty()) {
    std::vector<cv::Point2f> nxt, back;
    std::vector<uchar> st, st2;
    std::vector<float> err;
    const cv::Size win(21, 21);
    const auto crit = cv::TermCriteria(
        cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01);
    cv::calcOpticalFlowPyrLK(prev_img_, img, prev_pts_, nxt, st, err, win, 3,
                             crit);
    cv::calcOpticalFlowPyrLK(img, prev_img_, nxt, back, st2, err, win, 3, crit);
    for (size_t i = 0; i < prev_pts_.size(); ++i) {
      if (!st[i] || !st2[i])
        continue;
      if (cv::norm(back[i] - prev_pts_[i]) > opt_.fb_thresh)
        continue;
      if (nxt[i].x < 1 || nxt[i].y < 1 || nxt[i].x > img.cols - 2 ||
          nxt[i].y > img.rows - 2)
        continue;
      pts.push_back(nxt[i]);
      ids.push_back(prev_ids_[i]);
    }
  }

  // Top up, masking out the neighbourhood of every survivor. Without the mask a
  // corner already under track picks up a second id, and the two rows are then
  // treated as independent evidence when they are the same measurement twice.
  const int want = opt_.max_tracks - static_cast<int>(pts.size());
  if (want > 0 &&
      (!have_prev_ || static_cast<int>(pts.size()) < opt_.topup_below)) {
    cv::Mat mask(img.size(), CV_8U, cv::Scalar(255));
    for (const auto &p : pts)
      cv::circle(mask, p, static_cast<int>(opt_.min_feature_dist), 0, -1);
    std::vector<cv::Point2f> fresh;
    cv::goodFeaturesToTrack(img, fresh, want, opt_.quality_level,
                            opt_.min_feature_dist, mask);
    for (const auto &p : fresh) {
      pts.push_back(p);
      ids.push_back(next_id_++);
    }
  }

  std::vector<Ob> obs;
  obs.reserve(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    const Vec2 px{pts[i].x, pts[i].y};
    obs.push_back(Ob{ids[i], cam->UnProject(px), px});
  }

  frame_t_.push_back(t);
  frame_obs_.push_back(std::move(obs));
  prev_img_ = img.clone();
  prev_pts_ = std::move(pts);
  prev_ids_ = std::move(ids);
  have_prev_ = true;
}

bool InitWindowTracker::Build(const Vec3 &bg, const Vec3 &ba,
                              int min_track_frames, InitProblem *prob) const {
  if (prob == nullptr)
    return false;
  *prob = InitProblem{};
  if (frame_t_.size() < 2 || imu_.empty())
    return false;
  // Preintegration needs the IMU to cover the whole window; without the far end
  // the last frames would be integrated against a clamped constant sample.
  if (imu_.back().t < frame_t_.back())
    return false;

  // A track's observations are one per frame by construction, so its count *is*
  // its frame count.
  std::unordered_map<int, int> count;
  for (const auto &f : frame_obs_)
    for (const auto &o : f)
      ++count[o.id];
  std::unordered_map<int, int> index;
  for (const auto &f : frame_obs_)
    for (const auto &o : f)
      if (count[o.id] >= min_track_frames && index.find(o.id) == index.end())
        index.emplace(o.id, static_cast<int>(index.size()));
  if (index.size() < 4)
    return false;

  InitCamera cam{opt_.Rbc, opt_.Tbc, 1};
  if (auto *c = CameraManager::instance(opt_.cam_id))
    cam.focal = c->GetFocalLength();
  prob->cams.push_back(cam);
  prob->gravity = opt_.gravity;
  prob->num_tracks = static_cast<int>(index.size());
  prob->frames.resize(frame_t_.size());
  for (size_t k = 0; k < frame_t_.size(); ++k) {
    prob->frames[k].t = frame_t_[k];
    prob->frames[k].pre =
        Preintegrate(imu_, frame_t_.front(), frame_t_[k], bg, ba);
    if (k > 0)
      prob->frames[k].pre_prev =
          Preintegrate(imu_, frame_t_[k - 1], frame_t_[k], bg, ba);
  }
  for (size_t k = 0; k < frame_obs_.size(); ++k) {
    for (const auto &o : frame_obs_[k]) {
      const auto it = index.find(o.id);
      if (it == index.end())
        continue;
      InitObservation io;
      io.frame = static_cast<int>(k);
      io.track = it->second;
      io.cam = 0;
      io.xn = o.xn;
      prob->obs.push_back(io);
    }
  }
  return true;
}

Vec3 InitWindowTracker::GravityFromAccelMean(const Vec3 &bg,
                                             const Vec3 &ba) const {
  const Vec3 kDown{0, 0, -1};
  if (frame_t_.size() < 2 || imu_.empty())
    return kDown * opt_.gravity;
  const number_t t0 = frame_t_.front(), t1 = frame_t_.back();
  Vec3 acc = Vec3::Zero();
  int n = 0;
  for (const auto &s : imu_) {
    if (s.t < t0 || s.t > t1)
      continue;
    acc += Preintegrate(imu_, t0, s.t, bg, ba).R * (s.accel - ba);
    ++n;
  }
  if (n == 0)
    return kDown * opt_.gravity;
  const Vec3 g = -acc / n;
  const number_t nrm = g.norm();
  return nrm > 0 ? Vec3(g * (opt_.gravity / nrm)) : Vec3(kDown * opt_.gravity);
}

} // namespace xivo
