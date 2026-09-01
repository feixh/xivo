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
    const auto &res = f->inn();

    // Mahalanobis gating. The 25 columns of `J` that can be nonzero reach only a
    // 25x25 slice of `P_`; the dense product read all 2.5 MB of it once per
    // in-state feature, ~90 times a frame, which made this bandwidth-bound and
    // the third-largest cost in the system.
    Mat2 S = InnovationCov(f->J(), P_, f->ref()->sind(), f->sind(), R_);
    number_t mh_dist = res.dot(S.llt().solve(res));
    dist.push_back(mh_dist);
  }

  // Pick the threshold first, then apply it once. The relaxation policy is
  // unchanged -- start at `MH_thresh_` and multiply until `min_required_inliers_`
  // features pass -- but it no longer has to rebuild the inlier/reject lists and
  // rewrite every feature's status on each attempt.
  //
  // A non-finite Mahalanobis distance (S singular, or NaN in J/inn) never
  // compares less than any threshold, so relaxing `mh_thresh` can never admit
  // it as an inlier. If enough features are affected that `min_required_inliers_`
  // is unreachable, the search below would not terminate. Bound the relaxation:
  // once the threshold can no longer grow there is nothing left to gain, so stop
  // and let the caller proceed with whatever inliers were found.
  number_t mh_thresh = MH_thresh_;
  auto count_inliers = [&dist](number_t thresh) {
    return std::count_if(dist.begin(), dist.end(),
                         [thresh](number_t d) { return d < thresh; });
  };
  while (static_cast<size_t>(count_inliers(mh_thresh)) <
         min_required_inliers_) {
    number_t relaxed = mh_thresh * MH_thresh_multipler_;
    if (!std::isfinite(relaxed) || relaxed <= mh_thresh) {
      LOG(WARNING) << "MH-gating could not reach " << min_required_inliers_
                   << " inliers (" << count_inliers(mh_thresh) << " found of "
                   << instate_features_.size()
                   << " in-state features); giving up on relaxing the threshold";
      break;
    }
    mh_thresh = relaxed;
  }

  // reset states
  for (auto f : instate_features_) {
    if (f->status() != FeatureStatus::GAUGE) {
      f->SetStatus(FeatureStatus::INSTATE);
    }
  }

  // Apply the threshold. A feature that fails is not necessarily an outlier:
  // with a 95% gate and ~90 in-state features a consistent filter rejects four
  // or five *good* features every frame, and destroying them is the dominant
  // reason the state runs below capacity (76 of 90 slots occupied on TUM-VI
  // room1). `MH_max_strikes` > 1 lets a feature sit out this update and stay in
  // the state; only a run of consecutive failures destroys it. One strike
  // reproduces the original behaviour exactly.
  num_mh_deferred_ = 0;
  for (int i = 0; i < instate_features_.size(); ++i) {
    auto f = instate_features_[i];
    if (dist[i] < mh_thresh) {
      f->ClearMHStrikes();
      inliers.push_back(f);
      continue;
    }
    num_mh_rejected_++;
    if (f->AddMHStrike() >= MH_max_strikes_) {
      to_destroy.push_back(f);
      LOG(INFO) << "feature #" << f->id() << " rejected by MH-gating";
    } else {
      ++num_mh_deferred_;
      LOG(INFO) << "feature #" << f->id() << " failed MH-gating ("
                << f->mh_strikes() << " of " << MH_max_strikes_
                << " strikes); skipping it for this update";
    }
  }

#ifndef NDEBUG
  CHECK(inliers.size() + to_destroy.size() + num_mh_deferred_ ==
        instate_features_.size());
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



void Estimator::GateStereoMeasurements() {
  if (!stereo_update_) {
    // The right rows are computed unconditionally by `ComputeRightJacobian`
    // whenever a right match exists, so they must be explicitly disowned here or
    // a `stereo_update.enable = false` run would silently still use them.
    for (auto f : in_current_ekf_update_) {
      f->InvalidateRightJacobian();
    }
    return;
  }

  const number_t Rr = R_ * stereo_update_R_scale_;
  const number_t thresh = MH_thresh_ * stereo_update_mh_scale_;

  for (auto f : in_current_ekf_update_) {
    if (!f->has_right()) {
      continue;
    }
    if (!f->right_jac_valid()) {
      // Had a match, but the geometry was unusable (point behind camera 1).
      ++num_stereo_upd_rej_geom_;
      continue;
    }

    // A 2-dof Mahalanobis gate on the right measurement alone, using the same
    // threshold family as `MHGating`. Separate from the left gate on purpose:
    // the left track and the left->right match fail independently, and a wrong
    // right match should cost only its two rows.
    const Vec2 &res = f->inn_r();
    // The right rows have the same column structure as the left: the same motion
    // states, the same reference group, the same feature. `Rc1c0` is a constant,
    // not a state.
    Mat2 S = InnovationCov(f->Jr(), P_, f->ref()->sind(), f->sind(), Rr);
    number_t mh_dist = res.dot(S.llt().solve(res));
    // A non-finite distance never compares less than the threshold, so this
    // also catches a singular S.
    if (!(mh_dist < thresh)) {
      f->InvalidateRightJacobian();
      ++num_stereo_upd_rej_mh_;
      LOG(INFO) << "feature #" << f->id()
                << ": right observation rejected by MH-gating";
      continue;
    }
    ++num_stereo_upd_used_;
  }
}


void Estimator::ReserveOOSRows(int rows) {
  if (oos_H_.rows() >= rows && oos_H_.cols() == err_.size()) {
    return;
  }
  // Geometric, and preserving what is already stacked -- `MatX::resize` does not.
  // In practice this fires a handful of times in the first seconds of a run and
  // then never again.
  const int keep = std::min<int>(num_oos_rows_, oos_H_.rows());
  MatX old = oos_H_.topRows(keep);
  VecX old_inn = oos_inn_.head(keep);
  oos_H_.setZero(std::max(rows, 2 * static_cast<int>(oos_H_.rows())), err_.size());
  oos_inn_.setZero(oos_H_.rows());
  if (keep > 0 && old.cols() == oos_H_.cols()) {
    oos_H_.topRows(keep) = old;
    oos_inn_.head(keep) = old_inn;
  }
}

int Estimator::ComputeOOSMeasurements() {
  timer_.Tick("oos-jacobian");

  Graph &graph{*Graph::instance()};

  oos_used_.clear();
  oos_blocks_.clear();
  num_oos_candidates_ = oos_features_.size();
  num_oos_used_ = 0;
  num_oos_short_ = 0;
  num_oos_bad_tri_ = 0;
  num_oos_gated_ = 0;
  num_oos_rows_ = 0;

  const int min_obs = std::max(2, oos_options_.min_observations);

  for (auto f : oos_features_) {
    // Group management of this step (`DiscardAffectedGroups`) may have taken the
    // feature out of the graph -- its reference group went away and no new owner
    // could be found -- in which case there is nothing left to work with.
    if (!graph.HasFeature(f)) {
      continue;
    }
    auto all_obs = graph.GetObservationsOf(f);
    auto views = f->SelectOOSObservations(all_obs, oos_options_);
    total_oos_views_all_ += all_obs.size();
    total_oos_views_instate_ += views.size();
    oos_instate_view_hist_[std::min<size_t>(views.size(),
                                            oos_instate_view_hist_.size() - 1)]++;
    if (static_cast<int>(views.size()) < min_obs) {
      ++num_oos_short_;
      continue;
    }
    // Re-triangulate from all the views at once. The depth carried over from the
    // sub-filter is a two-view estimate and is not accurate enough: a wrong 3D
    // point yields a consistent but wrong constraint on the whole window of
    // poses the feature was seen from.
    if (oos_options_.refine && !f->RefineOOSDepth(gbc(), views, oos_options_)) {
      ++num_oos_bad_tri_;
      continue;
    }
    int rows = f->ComputeOOSJacobian(views, X_.Rbc.matrix(), X_.Tbc,
                                     oos_options_);
    if (rows <= 0) {
      ++num_oos_short_;
      continue;
    }
    if (!OOSGating(f)) {
      ++num_oos_gated_;
      continue;
    }
    // Out of the shared buffer and into ours, now, while these rows are still this
    // feature's: the next candidate overwrites them. This copy is what pays for
    // `Feature::oos_result()` being one buffer instead of ~800.
    ReserveOOSRows(num_oos_rows_ + rows);
    // `rows` is what `ComputeOOSJacobian` just returned, so the map is exactly that
    // tall and needs no slicing.
    oos_H_.middleRows(num_oos_rows_, rows) = f->oos_Hx();
    oos_inn_.segment(num_oos_rows_, rows) = f->oos_inn().head(rows);
    oos_blocks_.push_back({num_oos_rows_, rows, f->oos_runs()});

    oos_used_.push_back(f);
    num_oos_rows_ += rows;
    total_oos_obs_ += f->oos_num_obs();
    total_oos_right_obs_ += f->oos_num_right_obs();
  }
  num_oos_used_ = oos_used_.size();

  total_oos_candidates_ += num_oos_candidates_;
  total_oos_used_ += num_oos_used_;
  total_oos_short_ += num_oos_short_;
  total_oos_bad_tri_ += num_oos_bad_tri_;
  total_oos_gated_ += num_oos_gated_;
  total_oos_rows_ += num_oos_rows_;

  VLOG(1) << "OOS: " << num_oos_used_ << "/" << num_oos_candidates_
          << " features used, " << num_oos_rows_ << " rows (short="
          << num_oos_short_ << ", bad_tri=" << num_oos_bad_tri_ << ", gated="
          << num_oos_gated_ << ")";

  timer_.Tock("oos-jacobian");
  return num_oos_rows_;
}

bool Estimator::OOSGating(FeaturePtr f) {
  if (oos_options_.MH_thresh <= 0) {
    return true;
  }
  const int n = f->oos_inn_size();
  // Named, not `const auto H = f->oos_Hx().topRows(n)`: `oos_Hx()` returns a map by
  // value, and slicing the temporary would leave the block holding a reference to a
  // dead object. The map is already exactly `n` rows tall, so there is nothing to
  // slice anyway.
  const Eigen::Map<MatX> H = f->oos_Hx();
  const auto r = f->oos_inn().head(n);

  // `H` is `n x kFullSize` but structurally nonzero in only `Wbc`/`Tbc` and the
  // pose block of each observing group -- 36 columns of 564 for a 5-view track.
  // Formed densely, the gate reads all 2.5 MB of `P_` for every candidate and
  // spends 94% of its arithmetic on zero. See `RunSet` in `core.h`; the compacted
  // product is the same matrix, up to gemm reassociation.
  const RunSet &rs = f->oos_runs();
  MatX S;
  if (rs.nruns > 0) {
    MatX Hc(n, rs.dim);
    GatherRunCols(H, rs, Hc);
    MatX Pc(rs.dim, rs.dim);
    GatherRunCov(P_, rs, Pc);
    S.noalias() = Hc * Pc * Hc.transpose();
  } else {
    S.noalias() = H * P_ * H.transpose();
  }
  S.diagonal().array() += Roos_;
  number_t mh_dist = r.dot(S.llt().solve(r));
  // Normalized per degree of freedom, so that the threshold does not depend on
  // the length of the track (an in-state measurement has 2 dofs, an OOS one
  // 2n-3).
  if (!std::isfinite(mh_dist) || mh_dist > oos_options_.MH_thresh * n) {
    LOG(INFO) << "OOS feature #" << f->id() << " rejected by MH-gating, d="
              << mh_dist / std::max(n, 1);
    return false;
  }
  return true;
}

void Estimator::CleanupOOSFeatures() {
  Graph &graph{*Graph::instance()};
  for (auto f : oos_features_) {
    // The marginalized measurement has been consumed by the update above, and this
    // is the last point at which the feature is reachable. Drop its claim on the
    // shared `Feature::oos_result()` here rather than at the pooled slot's next
    // `Reset`: `CircBufWithHash` searches slots circularly, so a recycled slot can
    // be handed out long after this and a stale `oos_inn_size()` would make `Ho()`
    // return another feature's rows. Nothing reads the rows again, so this changes
    // no arithmetic -- though it does move heap addresses, which is not the same as
    // leaving the output bit-identical; see notes-oosfast/m5-shared-oos-buffer.md.
    f->ReleaseOOS();
    if (graph.HasFeature(f)) {
      graph.RemoveFeature(f);
      Feature::Destroy(f);
    }
    // else: already removed from the graph and deactivated while discarding
    // groups; the memory manager will recycle the slot.
  }
  oos_features_.clear();
  oos_used_.clear();
}

void Estimator::PrintCensus(std::ostream &os) const {
  const auto &c = census_;
  const auto per = [](long n, long d) { return d ? n / double(d) : 0.0; };
  // The error-state dimension implied by the *occupied* slots, against the
  // compile-time capacity that `P_` is actually sized to. The gap is what an
  // active-set compaction would recover.
  const double dim = kMotionSize + kMaxCameraIntrinsics +
                     kGroupSize * per(c.group_slots, c.frames) +
                     kFeatureSize * per(c.feat_slots, c.frames);
  os << "[census]frames:" << c.frames << " updates:" << c.updates
     << " feature-slots:" << per(c.feat_slots, c.frames) << "/" << kMaxFeature
     << " group-slots:" << per(c.group_slots, c.frames) << "/" << kMaxGroup
     << " occupied-dim:" << dim << "/" << kFullSize
     << " live-dim:" << per(c.live_dim, c.live_updates) << "/" << kFullSize
     << " live-runs:" << per(c.live_runs, c.live_updates)
     << " update-features:" << per(c.update_feats, c.updates)
     << " rows:" << per(c.rows, c.updates)
     << " (right:" << per(c.right_rows, c.updates)
     << " oos:" << per(c.oos_rows, c.updates) << ")";
  if (consistent_init_) {
    os << " consistent-init:" << num_consistent_init_ << "/"
       << (num_consistent_init_ + num_consistent_init_failed_);
  }
  os << "\n";
}

void Estimator::FilterUpdate(int oos_rows) {

#ifdef USE_GPERFTOOLS
  ProfilerStart(__PRETTY_FUNCTION__);
#endif

  timer_.Tick("update");

  timer_.Tick("stereo-gating");
  GateStereoMeasurements();
  timer_.Tock("stereo-gating");

  // Each in-state feature contributes two rows for the left camera and, when it
  // was matched into the right image and survived gating, two more for the
  // right. The measurement height is therefore data-dependent -- hence the
  // running `row` cursor below rather than the old fixed `2 * i` stride.
  // The out-of-state rows are appended after all of the in-state ones.
  const number_t Rr = R_ * stereo_update_R_scale_;
  int instate_size = 2 * in_current_ekf_update_.size();
  for (auto f : in_current_ekf_update_) {
    if (f->right_jac_valid()) {
      instate_size += 2;
    }
  }
  int total_size = instate_size + oos_rows;

  ++census_.updates;
  census_.update_feats += in_current_ekf_update_.size();
  census_.rows += total_size;
  census_.right_rows += instate_size - 2 * in_current_ekf_update_.size();
  census_.oos_rows += oos_rows;

  H_.setZero(total_size, err_.size());
  inn_.setZero(total_size);
  diagR_.resize(total_size);
  // The sparsity of each row block, recorded as it is written: a feature's rows
  // (both cameras') are nonzero only in the motion columns, its reference
  // group's slot and its own, which is what the update exploits. Nothing here
  // depends on it being right except the speed of `H P` -- but `MeasBlock` is
  // also what says the block *is* dense when it is (the out-of-state rows
  // below), so it has to be filled for every row of `H_`.
  meas_blocks_.clear();

  int row = 0;
  for (int i = 0; i < in_current_ekf_update_.size(); ++i) {
    auto f = in_current_ekf_update_[i];
    const int gsind = f->ref()->sind();
    f->FillJacobianBlock(H_, row);
    inn_.segment<2>(row) = f->inn();
    diagR_.segment<2>(row) << R_, R_;
    meas_blocks_.push_back({row, 2, gsind, f->sind()});
    row += 2;
    if (f->right_jac_valid()) {
      f->FillRightJacobianBlock(H_, row);
      inn_.segment<2>(row) = f->inn_r();
      diagR_.segment<2>(row) << Rr, Rr;
      meas_blocks_.push_back({row, 2, gsind, f->sind()});
      row += 2;
    }
  }
#ifndef NDEBUG
  CHECK(row == instate_size);
#endif

  // Out-of-state measurements below the in-state ones. Their noise is isotropic
  // with variance Roos_ *because* the point was marginalized out with an
  // orthonormal basis of the left nullspace of Hf (see
  // Feature::MarginalizeOOSPoint) -- the diagonal diagR_ would otherwise be
  // wrong.
  int offset = instate_size;
  for (const auto &b : oos_blocks_) {
    H_.block(offset, 0, b.rows, err_.size()) = oos_H_.middleRows(b.row, b.rows);
    inn_.segment(offset, b.rows) = oos_inn_.segment(b.row, b.rows);
    diagR_.segment(offset, b.rows).setConstant(Roos_);
    // Not of the fixed in-state shape: the marginalized rows span every group the
    // track was observed from, up to `oos_options_.max_observations` of them, and
    // the left-nullspace projection mixes them. Which groups those are is still
    // known, and passing it lets the update skip the rest of the state; `nruns ==
    // 0` when `oos_fast.enable` is off, and then it is null and the block is
    // treated as fully dense as before. The pointer is into `oos_blocks_`, which is
    // not touched again until the next `ComputeOOSMeasurements`.
    meas_blocks_.push_back(
        {offset, b.rows, -1, -1, b.runs.nruns > 0 ? &b.runs : nullptr});
    offset += b.rows;
  }
  CHECK_EQ(offset, total_size);

  timer_.Tick("actual-update");
  MeasurementUpdate();
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
  // `ComputeLCJacobian` writes the loop-closed group's columns as well as the
  // matched feature's, i.e. two groups per row block, so these rows do not have
  // the shape `MeasBlock` describes; treated as dense.
  meas_blocks_.clear();
  meas_blocks_.push_back({0, total_size, -1, -1});

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
  MeasurementUpdate();
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
    meas_blocks_.clear();
    int f_cnt = 0;
    for (int i = 0; i < mh_inliers.size(); ++i) {
      if (is_low_innovation_inlier[i]) {
        auto f = mh_inliers[i];
        H_.block(2 * f_cnt, 0, 2, size) = f->J();
        inn_.segment<2>(2 * f_cnt) = f->inn();
        diagR_.segment<2>(2 * f_cnt) << R_, R_;
        meas_blocks_.push_back({2 * f_cnt, 2, f->ref()->sind(), f->sind()});
        f_cnt++;
      }
    }
    MeasurementUpdate();
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
        auto res = f->inn();

        Mat2 S = InnovationCov(f->J(), P_, f->ref()->sind(), f->sind(), R_);
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
