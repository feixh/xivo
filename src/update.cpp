// The update step.
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <unordered_set>

#include "glog/logging.h"

#ifdef USE_GPERFTOOLS
#include "gperftools/profiler.h"
#endif

#include "estimator.h"
#include "feature.h"
#include "geometry.h"
#include "group.h"
#include "tracker.h"
#include "graph.h"

namespace xivo {


void Estimator::ComputeInstateJacobians() {
  timer_.Tick("jacobian");
  for (auto f : instate_features_) {
    f->ComputeJacobian(X_.Rsb.matrix(), X_.Tsb, X_.Rbc.matrix(), X_.Tbc,
                       last_gyro_, imu_.Cg(), X_.bg, X_.Vsb, X_.td);
  }
  timer_.Tock("jacobian");

}


void Estimator::FindNewGaugeFeatures() {
  // find new gauge features (includes newly added groups and groups that lost
  // an existing gauge feature)
  for (auto g: needs_new_gauge_features_) {
    std::vector<FeaturePtr> new_gauge_feats =
      Graph::instance()->FindNewGaugeFeatures(g);
    for (auto f: new_gauge_feats) {
      FixFeatureXY(f);
    }
  }

  needs_new_gauge_features_.clear();
}


std::vector<FeaturePtr> Estimator::MHGating() {

  timer_.Tick("MH-gating");

  std::vector<FeaturePtr> inliers; // individually compatible matches
  std::vector<number_t> dist, inlier_dist; // MH distance of features & inlier features
  num_mh_rejected_ = 0;
  std::vector<FeaturePtr> to_destroy;

  // Compute Mahalanobis distance
  for (auto f: instate_features_) {
    const auto &J = f->J();
    const auto &res = f->inn();

    // Mahalanobis gating
    Mat2 S = J * P_ * J.transpose();
    S(0, 0) += R_;
    S(1, 1) += R_;
    number_t mh_dist = res.dot(S.llt().solve(res));
    dist.push_back(mh_dist);
  }

  // The actual gating
  number_t mh_thresh = MH_thresh_;
  // A non-finite Mahalanobis distance (S singular, or NaN in J/inn) never
  // compares less than any threshold, so relaxing `mh_thresh` can never admit
  // it as an inlier. If enough features are affected that `min_required_inliers_`
  // is unreachable, the loop below spins forever. Bound the relaxation: once
  // the threshold can no longer grow (or has grown past every finite distance)
  // there is nothing left to gain, so stop and let the caller proceed with
  // whatever inliers were found.
  while (inliers.size() < min_required_inliers_) {
    // reset states
    for (auto f : instate_features_) {
      if (f->status() != FeatureStatus::GAUGE) {
        f->SetStatus(FeatureStatus::INSTATE);
      }
    }
    inliers.clear();
    to_destroy.clear();
    // mark inliers
    for (int i = 0; i < instate_features_.size(); ++i) {
      auto f = instate_features_[i];
      if (dist[i] < mh_thresh) {
        inliers.push_back(f);
      } else {
        num_mh_rejected_++;
        to_destroy.push_back(f);
        LOG(INFO) << "feature #" << f->id() << " rejected by MH-gating";
      }
    }
    if (inliers.size() >= min_required_inliers_) {
      break;
    }
    // relax the threshold
    number_t relaxed = mh_thresh * MH_thresh_multipler_;
    if (!std::isfinite(relaxed) || relaxed <= mh_thresh) {
      LOG(WARNING) << "MH-gating could not reach " << min_required_inliers_
                   << " inliers (" << inliers.size() << " found of "
                   << instate_features_.size()
                   << " in-state features); giving up on relaxing the threshold";
      break;
    }
    mh_thresh = relaxed;
    num_mh_rejected_ = 0;
  }

#ifndef NDEBUG
  CHECK(inliers.size() + to_destroy.size() == instate_features_.size());
#endif

  timer_.Tock("MH-gating");
  LOG(INFO) << "MH rejected " << num_mh_rejected_ << " features";

  for (auto f: to_destroy) {
    if (f->status() == FeatureStatus::GAUGE) {
      needs_new_gauge_features_.push_back(f->ref());
      LOG(INFO) << "Group # " << f->ref()->id() << " just lost a gauge feature rejected by MH-gating";
    }
    f->SetStatus(FeatureStatus::REJECTED_BY_FILTER);
    affected_groups_.insert(f->ref());
  }
  DestroyFeatures(to_destroy);

  return inliers;
}



void Estimator::FilterUpdate() {

#ifdef USE_GPERFTOOLS
  ProfilerStart(__PRETTY_FUNCTION__);
#endif

  timer_.Tick("update");

 
  int total_size = 2 * in_current_ekf_update_.size();
  H_.setZero(total_size, err_.size());
  inn_.setZero(total_size);
  diagR_.resize(total_size);

  for (int i = 0; i < in_current_ekf_update_.size(); ++i) {
    in_current_ekf_update_[i]->FillJacobianBlock(H_, 2 * i);
    inn_.segment<2>(2 * i) = in_current_ekf_update_[i]->inn();
    diagR_.segment<2>(2 * i) << R_, R_;
  }

  timer_.Tick("actual-update");
  UpdateJosephForm();
  timer_.Tock("actual-update");

  // absorb error
  AbsorbError();
  timer_.Tock("update");

  LOG(INFO) << "Error state absorbed";

#ifdef USE_GPERFTOOLS
  ProfilerStop();
#endif
}


void Estimator::CloseLoop() {
#ifdef USE_MAPPER
  std::vector<FeaturePtr> instate_features =
    Graph::instance()->GetInstateFeatures();
  std::vector<LCMatch> matches;
  if (instate_features.size() > 0) {
    matches = Mapper::instance()->DetectLoopClosures(instate_features, gbc());
  }

  if (matches.size() > 0) {
    CloseLoopInternal(Graph::instance()->LastAddedGroup(), matches);
  }
#endif
}

void Estimator::CloseLoopInternal(GroupPtr g, std::vector<LCMatch>& matched_features) {
#ifdef USE_MAPPER
  Graph& graph{*Graph::instance()};

  int num_matches = matched_features.size();

  // H and R matrices
  int total_size = 2 * matched_features.size();
  H_.setZero(total_size, err_.size());
  diagR_.resize(total_size);
  inn_.setZero(total_size);

  // Compute feature Jacobians (fill in H)
  for (int i=0; i<num_matches; i++) {
    FeaturePtr new_feature = matched_features[i].first;
    FeaturePtr old_feature = matched_features[i].second;

    Observation obs = graph.GetObservationOf(new_feature, g);
    old_feature->ComputeLCJacobian(obs, X_.Rbc, X_.Tbc, err_, i, H_, inn_);

    // Fill in R
    diagR_.segment<2>(2*i) << Rlc_, Rlc_;

    // Print out stuffs
    //std::cout << "Comparing new (#" << new_feature->id() << ") to old (#" << old_feature->id() << ")" << std::endl;
    //std::cout << "new Xs: " << new_feature->Xs().transpose() << std::endl;
    //std::cout << "old Xs: " << old_feature->Xs().transpose() << std::endl;
  }

  //std::cout << "LC innovation: " << inn_.transpose() << std::endl;

  // Update Group list
  instate_groups_.clear();
  instate_groups_ = Graph::instance()->GetInstateGroups();

  // Measurement Update
  UpdateJosephForm();
  AbsorbError();
#endif
}


std::vector<FeaturePtr>
Estimator::OnePointRANSAC(const std::vector<FeaturePtr> &mh_inliers) {
  if (mh_inliers.empty())
    return mh_inliers;
  // Reference:
  // https://www.doc.ic.ac.uk/~ajd/Publications/civera_etal_jfr2010.pdf
  int n_hyp = 1000;
  std::vector<bool> selected(mh_inliers.size(), false);
  int selected_counter = 0;
  std::uniform_int_distribution<int> distribution(0, mh_inliers.size() - 1);

  // find those involved in update step
  std::unordered_set<FeaturePtr> active_features;
  std::unordered_set<GroupPtr> active_groups;
  for (auto f : mh_inliers) {
    active_features.insert(f);
  }
  for (auto f : mh_inliers) {
    active_groups.insert(f->ref());
  }

  /* We've already done the EKF prediction step and measurement prediction.
  So this step just looks for the maximal set of low-innovation inliers.
  */
  std::unordered_set<FeaturePtr> max_inliers, inliers;
  for (int i = 0; i < n_hyp && selected_counter < selected.size(); ++i) {
    int k = distribution(*rng_);
    while (selected[k]) {
      k = distribution(*rng_);
    }
    selected[k] = true;
    ++selected_counter;

    inliers.clear();
    for (auto f : mh_inliers) {
      auto res = f->xp() - f->Predict(gsb(), gbc());
      if (res.norm() < ransac_thresh_) {
        inliers.insert(f);
      }
    }
    if (inliers.size() > max_inliers.size()) {
      max_inliers = inliers;
      number_t eps = max_inliers.size() / float(mh_inliers.size());
      n_hyp = int(log(1 - ransac_prob_) / log(1-eps)) + 1; // RANSAC minimum number of trials
    }
  }
  auto str = StrFormat("#hyp tested=%d: li_inliers/mh_inliers=%d/%d",
                             n_hyp, max_inliers.size(), mh_inliers.size());

  LOG(INFO) << str;

  // If everything is a low-innovation inlier, we don't need to do anything more.
  if (max_inliers.size() == mh_inliers.size()) {
    return mh_inliers;
  }

  // Save which features are inliers.
  std::vector<bool> is_low_innovation_inlier;
  std::unordered_set<GroupPtr> groups_with_low_inn_inlier;
  for (int i=0; i<mh_inliers.size(); i++) {
    if (max_inliers.count(mh_inliers[i])) {
      is_low_innovation_inlier.push_back(true);
      groups_with_low_inn_inlier.insert(mh_inliers[i]->ref());
    }
    else {
      is_low_innovation_inlier.push_back(false);
    }
  }

  // back up state and covariance.
  BackupState(active_features, active_groups);


  // STEP 2: EKF update using only low-innovation inlier measurements.
  if (!max_inliers.empty()) {
    int size = err_.size();

    // Find a new temporary reference group if the gauge group pointer doesn't
    // contain a high-inlier feature
    if (groups_with_low_inn_inlier.count(gauge_group_ptr_) == 0) {
      LOG(INFO) << "One-Pt RANSAC using temporary new reference group";
      std::vector<GroupPtr> candidates;
      candidates.insert(candidates.end(), groups_with_low_inn_inlier.begin(),
                        groups_with_low_inn_inlier.end());
      GroupPtr tmpref = FindNewRefGroup(candidates);
      int offset = kGroupBegin + kGroupSize * tmpref->sind();
      P_.block(offset, 0, kGroupSize, size).setZero();
      P_.block(0, offset, size, kGroupSize).setZero();
    }

    // Zero out features and groups that aren't in the low-inlier set
    for (int i=0; i<mh_inliers.size(); i++) {
      if (!is_low_innovation_inlier[i]) {
        int offset = kFeatureBegin + kFeatureSize * mh_inliers[i]->sind();
        P_.block(offset, 0, kFeatureSize, size).setZero();
        P_.block(0, offset, size, kFeatureSize).setZero();
      }
    }
    for (auto g: active_groups) {
      if (groups_with_low_inn_inlier.count(g) == 0) {
        int offset = kGroupBegin + kGroupSize * g->sind();
        P_.block(offset, 0, kGroupSize, size).setZero();
        P_.block(0, offset, size, kGroupSize).setZero();
      }
    }

    // low innovation update
    H_.setZero(2 * max_inliers.size(), size);
    inn_.setZero(2 * max_inliers.size());
    diagR_.resize(2 * max_inliers.size());
    int f_cnt = 0;
    for (int i = 0; i < mh_inliers.size(); ++i) {
      if (is_low_innovation_inlier[i]) {
        H_.block(2 * f_cnt, 0, 2, size) = mh_inliers[i]->J();
        inn_.segment<2>(2 * f_cnt) = mh_inliers[i]->inn();
        diagR_.segment<2>(2 * f_cnt) << R_, R_;
        f_cnt++;
      }
    }
    UpdateJosephForm();
    AbsorbError();
  }

  if (max_inliers.size() < mh_inliers.size()) {
    // rescue high-innovation measurements
    std::vector<FeaturePtr> hi_inliers; // high-innovation inlier set
    std::vector<FeaturePtr> to_destroy;
    
    num_oneptransac_rejected_ = 0;

    for (int i = 0; i < mh_inliers.size(); ++i) {
      if (!is_low_innovation_inlier[i]) {
        // potentially a high-innovation inlier
        auto f = mh_inliers[i];

        f->ComputeJacobian(X_.Rsb.matrix(), X_.Tsb, X_.Rbc.matrix(), X_.Tbc, last_gyro_,
                           imu_.Cg(), X_.bg, X_.Vsb, X_.td);
        auto J = f->J();
        auto res = f->inn();

        Mat2 S = J * P_ * J.transpose();
        S(0, 0) += R_;
        S(1, 1) += R_;
        if (res.dot(S.llt().solve(res)) < ransac_Chi2_) {
          hi_inliers.push_back(f);
        } else {
          if (f->status() == FeatureStatus::GAUGE) {
            needs_new_gauge_features_.push_back(f->ref());
            LOG(INFO) << "Group # " << f->ref()->id() << " just lost a guage feature rejected by one-pt ransac";
          }
          f->SetStatus(FeatureStatus::REJECTED_BY_FILTER);
          to_destroy.push_back(f);
          num_oneptransac_rejected_++;
          affected_groups_.insert(f->ref());
          LOG(INFO) << "feature #" << f->id() << " rejected by one-pt ransac";
        }
      }
    }

    DestroyFeatures(to_destroy);
    // active_features was snapshotted before the update and is walked again
    // below (RestoreState + ComputeJacobian). A destroyed feature has had
    // RemoveFeatureFromState set its sind to -1 (estimator.cpp), and
    // ComputeJacobian indexes Jacobian blocks off sind(), so leaving it in the
    // set writes at a negative offset.
    for (auto f : to_destroy) {
      active_features.erase(f);
    }

    if (!hi_inliers.empty()) {
      max_inliers.insert(hi_inliers.begin(), hi_inliers.end());
      LOG(INFO) << "rescued " << hi_inliers.size() << " high-innovation inliers"
                << std::endl;
    }

  }

  // restore state (need to re-compute jacobians at original state)
  RestoreState(active_features, active_groups);
  for (auto f : active_features) {
    f->ComputeJacobian(X_.Rsb.matrix(), X_.Tsb, X_.Rbc.matrix(), X_.Tbc, last_gyro_, imu_.Cg(),
                       X_.bg, X_.Vsb, X_.td);
  }

  // create a vector for output
  std::vector<FeaturePtr> output;
  output.insert(output.end(), max_inliers.begin(), max_inliers.end());
  return output;
}

} // namespace xivo
