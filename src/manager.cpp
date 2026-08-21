#include <algorithm>
#include <iomanip>
#include <iostream>
#include <unordered_set>

#include "glog/logging.h"

#include "estimator.h"
#include "feature.h"
#include "geometry.h"
#include "group.h"
#include "tracker.h"
#include "mapper.h"
#include "camera_manager.h"

namespace xivo {

void Estimator::UpdateStep(const timestamp_t &ts,
                           std::list<FeaturePtr> &tracks) {

  // Data structures for bookkeeping features and groups as we add and remove
  // them from the state.
  instate_features_.clear();
  instate_groups_.clear();
  affected_groups_.clear();
  new_features_.clear();
  inliers_.clear();
  in_current_ekf_update_.clear();

  // only used for data collection.
  just_dropped_feature_ids_.clear();

  // retrieve the visibility graph
  Graph& graph{*Graph::instance()};

  // increment lifetime of all features and groups
  for (auto f : graph.GetFeatures()) {
    f->IncrementLifetime();
  }
  for (auto g : graph.GetGroups()) {
    g->IncrementLifetime();
  }

  // Based on feature's TrackStatus and FeatureStatus, will delete feature,
  // update subfilter, etc.
  ProcessTracks(ts, tracks);
  instate_features_ = graph.GetInstateFeatures();

#ifndef NDEBUG
  auto sum = [](std::array<bool, kMaxFeature> ff) {
    int cnt = 0;
    for (auto d: ff) {
      if (d)
        cnt += 1;
    }
    return cnt;
  };
  CHECK(sum(fsel_) == instate_features_.size())
    << "bookkeeping error in processing of tracks";
#endif

  // Potentially add new features to the EKF state.
  if (instate_features_.size() < kMaxFeature) {
    SelectAndAddNewFeatures();
  }
#ifndef NDEBUG
  CHECK(sum(fsel_) == instate_features_.size())
    << "bookkeeping error in adding new features";
#endif

  // Compute Jacobians of all instate features, including those just added
  ComputeInstateJacobians();

  // Perform outlier rejection and EKF update with instate features.
  // This edits the vector `inliers_`.
  if (!instate_features_.empty()) {
    MakePtrVectorUnique(instate_features_);
    OutlierRejection();
  }
#ifndef NDEBUG
  CHECK(sum(fsel_) == inliers_.size())
    << "bookkeeping error in outlier rejection";
#endif
  // We need to remove floating groups (with no instate features) and
  // floating features (not instate and reference group is floating)
  DiscardAffectedGroups();
  FindNewGaugeFeatures();

  // Final step before update: make sure NULLREFED features aren't used
  // in EKF update
  for (auto f: inliers_) {
    if (f->instate()) {
      in_current_ekf_update_.push_back(f);
    }
  }

#ifndef NDEBUG
  CHECK(sum(fsel_) == in_current_ekf_update_.size())
    << "bookkeeping error in removing floating groups";
#endif

  if (!in_current_ekf_update_.empty()) {
    instate_groups_ = graph.GetInstateGroups();
    FilterUpdate();
  }

  // Make accessors work.
  instate_features_ = in_current_ekf_update_;

  MeasurementUpdateInitialized_ = true;

  // Post-update feature management
  // For instate features rejected by the filter,
  // 1) remove the fetaure from features_ and free state & covariance
  // 2) detach the feature from the reference group
  // 3) remove the group if it lost all the instate features


  // Create a new group for this pose. Initialize with the newly updated
  // value of Rsb and Tsb
  GroupPtr g = Group::Create(X_.Rsb, X_.Tsb);
  graph.AddGroup(g);

  // reassemble the tracker's feature list with newly created features and
  // currently tracked features
  tracks.clear();
  InitializeJustCreatedTracks(g, tracks);
  AssociateTrackedFeaturesWithGroup(g, tracks);

  // adapt initial depth to average depth of features currently visible
  AdaptInitialDepth();

  // remove old non-reference groups
  EnforceMaxGroupLifetime();

  // std::cout << "#groups=" << graph.GetGroups().size() << std::endl;
  // check & clean graph
  // graph.SanityCheck();
  // // remove isolated groups
  // auto empty_groups = graph.GetGroupsIf([this](GroupPtr g)->bool {
  //     return graph.GetGroupAdj(g).empty(); });
  // LOG(INFO) << "#empty groups=" << empty_groups.size();
  // graph.RemoveGroups(empty_groups);
  // for (auto g : empty_groups) {
  //   CHECK(!g->instate());
  //   Group::Delete(g);
  // }

  // Update Visualization
  if (use_canvas_) {
    for (auto f : tracks) {
      Canvas::instance()->Draw(f);
    }
    Canvas::instance()->OverlayStateInfo(
      X_, imu_.State(), CameraManager::instance()->GetIntrinsics()
    );
  }

  static int print_counter{0};
  if (print_timing_ && ++print_counter % 50 == 0) {
    std::cout << print_counter << std::endl;
    std::cout << timer_;
  }

  // Save the frame (only if set to true in json file)
  Canvas::instance()->SaveFrame();
}



void Estimator::ProcessTracks(const timestamp_t &ts,
                              std::list<FeaturePtr> &tracks) 
{
  Graph& graph{*Graph::instance()};

  for (auto it = tracks.begin(); it != tracks.end();) {
    auto f = *it;

    // Track just created, must not included in the graph yet
    if (f->track_status() == TrackStatus::CREATED) {
      new_features_.push_back(f);
      it = tracks.erase(it);
    }

    // Track is in the EKF state and just dropped by the tracker
    else if (f->instate() && f->track_status() == TrackStatus::DROPPED) {
#ifdef USE_MAPPER
      Mapper::instance()->AddFeature(f, graph.GetFeatureAdj(f), gbc());
#endif
      just_dropped_feature_ids_.push_back(f->id());
      graph.RemoveFeature(f);

      LOG(INFO) << "Tracker rejected feature #" << f->id();
      if (f->status() == FeatureStatus::GAUGE) {
        needs_new_gauge_features_.push_back(f->ref());
        LOG(INFO) << "Group # " << f->ref()->id() << " just lost a gauge feature rejected by tracker.";
      }
      RemoveFeatureFromState(f);
      affected_groups_.insert(f->ref());

      Feature::Deactivate(f);
      it = tracks.erase(it);
    }

    // Track is not in the EKF state and just dropped by tracker
    else if (!f->instate() && f->track_status() == TrackStatus::DROPPED) {
      just_dropped_feature_ids_.push_back(f->id());

      graph.RemoveFeature(f);
      Feature::Destroy(f);
      it = tracks.erase(it);
    }

    // instate feature being tracked -- use in measurement update later on
    else if (f->instate() && f->track_status() == TrackStatus::TRACKED) {
      ++it;
    }

    // Track is an "initializing" feature that has been tracked - update the
    // Subfilter. Feature will be removed if Mahalanobis gating in the
    // subfilter determines that it is an outlier.
    else {
#ifndef NDEBUG
      CHECK(f->track_status() == TrackStatus::TRACKED);
      CHECK(!f->instate());
#endif
      // perform triangulation if we've observed the feature exactly twice
      // so far 
      if (triangulate_pre_subfilter_ && f->size() == 2) {
        f->Triangulate(gsb(), gbc(), triangulate_options_);
      }

      // run depth subfilter to improve depth ...
      f->SubfilterUpdate(gsb(), gbc(), subfilter_options_);

      // Mark feature as outlier if its total MH distance (calculated using
      // subfilter covariance) is too high
      if (f->outlier_counter() > remove_outlier_counter_) {
        graph.RemoveFeature(f);
        Feature::Destroy(f);
        it = tracks.erase(it);
      } else {
        ++it;
      }

    } // end track status

  } // end for loop

} // end ProcessTracks




void Estimator::AdaptInitialDepth() {
  Graph& graph{*Graph::instance()};
  auto depth_features = graph.GetFeaturesIf([this](FeaturePtr f) -> bool {
    return f->instate() ||
           (f->status() == FeatureStatus::READY &&
            f->lifetime() > adaptive_initial_depth_options_.min_feature_lifetime);
  });
  if (!depth_features.empty()) {
    std::vector<number_t> depth(depth_features.size());
    std::transform(depth_features.begin(), depth_features.end(), depth.begin(),
                   [](FeaturePtr f) { return f->z(); });
    number_t median_depth = depth[depth.size() >> 1];

    if (median_depth < min_z_ || median_depth > max_z_) {
      VLOG(0) << "Median depth out of bounds: " << median_depth;
      VLOG(0) << "Reuse the old one: " << init_z_;
    } else {
      number_t beta = adaptive_initial_depth_options_.median_weight;
      init_z_ = (1.0-beta) * init_z_ + beta * median_depth;
      VLOG(0) << "Update aptive initial depth: " << init_z_;
    }
  }

}



void Estimator::EnforceMaxGroupLifetime() {
  Graph& graph{*Graph::instance()};
  auto all_groups = graph.GetGroups();
  int max_group_lifetime = cfg_.get("max_group_lifetime", 1).asInt();
  for (auto g : all_groups) {
    if (g->lifetime() > max_group_lifetime) {
      const auto &adj = graph.GetGroupAdj(g);
      if (std::none_of(adj.begin(), adj.end(), [&graph, g](int fid) {
            return graph.GetFeature(fid)->ref() == g;
          })) {
        // for groups which have no reference features, they cannot be instate
        // anyway
#ifndef NDEBUG
        CHECK(!g->instate());
#endif

#ifdef USE_MAPPER
        Mapper::instance()->AddGroup(g, graph.GetGroupAdj(g));
#endif
        graph.RemoveGroup(g);
        Group::Deactivate(g);
      }
    }
  }
}



void Estimator::DiscardAffectedGroups() {
  Graph& graph{*Graph::instance()};
  // `affected_groups_` is an unordered_set keyed by pointer, so its iteration
  // order depends on the hash of the addresses and therefore on ASLR. That
  // would be harmless for a read-only loop, but this one *mutates the graph*:
  // FindNewOwnersForFeaturesOf reassigns feature ownership, so discarding one
  // group changes whether the next group still meets the instate-feature
  // threshold below. Iterating in id order makes the outcome reproducible.
  // See notes-stereo/m3a-determinism.md.
  std::vector<GroupPtr> affected(affected_groups_.begin(),
                                 affected_groups_.end());
  std::sort(affected.begin(), affected.end(),
            [](GroupPtr a, GroupPtr b) { return a->id() < b->id(); });
  for (auto g : affected) {
    std::vector<FeaturePtr> instate_features_of_g = graph.GetFeaturesIf(
      [g](FeaturePtr f) { return (f->ref() == g) && (f->instate()); }
    );
    int num_instate_features_of_g = instate_features_of_g.size();
    if ((num_instate_features_of_g < num_gauge_xy_features_) ||
        ((num_gauge_xy_features_ == 0) && (num_instate_features_of_g == 0))) {
      std::vector<FeaturePtr> nullrefs = FindNewOwnersForFeaturesOf(g);
      DiscardFeatures(nullrefs);
      for (auto f: nullrefs) {
        f->SetStatus(FeatureStatus::NULLREFED);
      }
      DiscardGroup(g);
    }
  }
  affected_groups_.clear();
}



void Estimator::SelectAndAddNewFeatures() {
  Graph& graph{*Graph::instance()};

  // First, try to add features that are already owned by an existing group.
  // Then, try to add an entire new group at once.

  int free_group_slots = std::count(gsel_.begin(), gsel_.end(), false);
  int free_feature_slots = kMaxFeature - instate_features_.size();

  if (num_gauge_xy_features_ == 0) {
    ZeroGaugeXYAddFeatures();
  }
  else if (free_feature_slots < num_gauge_xy_features_) {
    AddFeaturesWithInGroups();
  }
  else if (free_group_slots == 0) {
    AddFeaturesWithInGroups();
  }
  else {
    AddGroupOfFeatures(free_group_slots);
    AddFeaturesWithInGroups();
  }

}


void Estimator::AddFeaturesWithInGroups() {
  Graph& graph{*Graph::instance()};

  // choose the candidates
  auto vision_counter_criterion =
    vision_counter_ < strict_criteria_timesteps_ ? Criteria::Candidate
                                                  : Criteria::CandidateStrict;
  auto criterion = vision_counter_criterion;
  auto ref_group_is_instate = [](FeaturePtr f) { return f->ref()->instate(); };
  auto candidates = graph.GetFeaturesIf(
      [criterion, ref_group_is_instate](FeaturePtr f) {
        return (criterion(f) && ref_group_is_instate(f));
      }
    );

  // Sort candidates by metric (default: DepthUncertainty)
  MakePtrVectorUnique(candidates);
  std::sort(candidates.begin(), candidates.end(),
      Criteria::CandidateComparison);

  // For depth refinement
  std::vector<FeaturePtr> bad_features;

  for (auto it = candidates.begin();
       it != candidates.end() && instate_features_.size() < kMaxFeature;
       ++it) {

    auto f = *it;

    if (use_depth_opt_) {
      auto obs = graph.GetObservationsOf(f);
      if (obs.size() > 1) {
        if (!f->RefineDepth(gbc(), obs, refinement_options_)) {
          bad_features.push_back(f);
          continue;
        }
      }
      else if (obs.size() == 0) {
        LOG(ERROR) << "A feature with no observations should not be a candidate";
      }
    }

    instate_features_.push_back(f);
    AddFeatureToState(f); // insert f to state vector and covariance

  }
  DestroyFeatures(bad_features);
}


void Estimator::ZeroGaugeXYAddFeatures() {
  Graph& graph{*Graph::instance()};

  int free_slots = std::count(gsel_.begin(), gsel_.end(), false);

  // choose the instate-candidate criterion
  auto criterion =
    vision_counter_ < strict_criteria_timesteps_ ? Criteria::Candidate
                                                  : Criteria::CandidateStrict;
  auto candidates = graph.GetFeaturesIf(criterion);

  MakePtrVectorUnique(candidates);
  std::sort(candidates.begin(), candidates.end(),
      Criteria::CandidateComparison);

  std::vector<FeaturePtr> bad_features;

  for (auto it = candidates.begin();
        it != candidates.end() && instate_features_.size() < kMaxFeature;
        ++it) {

    auto f = *it;

    if (use_depth_opt_) {
      auto obs = graph.GetObservationsOf(f);
      if (obs.size() > 1) {
        if (!f->RefineDepth(gbc(), obs, refinement_options_)) {
          bad_features.push_back(f);
          continue;
        }
      }
      else if (obs.size() == 0) {
        LOG(ERROR) << "A feature with no observations should not be a candidate";
      }
    }

    if (!f->ref()->instate() && free_slots <= 0) {
      // If we turn this feature to instate, its reference group should
      // also be instate, which out-number the available group slots ...
      continue;
    }

    instate_features_.push_back(f);
    AddFeatureToState(f); // insert f to state vector and covariance
    if (!f->ref()->instate()) {
#ifndef NDEBUG
      CHECK(graph.HasGroup(f->ref()));
      CHECK(graph.GetGroupAdj(f->ref()).count(f->id()));
      CHECK(graph.GetFeatureAdj(f).count(f->ref()->id()));
#endif
      // need to add reference group to state if it's not yet instate
      AddGroupToState(f->ref());
      needs_new_gauge_features_.push_back(f->ref());
      // use up one more free slot
      --free_slots;
    }
  }
  DestroyFeatures(bad_features);
}


void Estimator::AddGroupOfFeatures(int free_group_slots) {
  Graph& graph{*Graph::instance()};

#ifndef NDEBUG
  CHECK(instate_features_.size() <= (kMaxFeature - num_gauge_xy_features_));
  CHECK(free_group_slots > 0);
#endif

  // total number of features we need to add
  int num_features_to_add = kMaxFeature - instate_features_.size();

  // Get all candidate groups
  auto candidates = graph.GetInstateGroupCandidates(num_gauge_xy_features_);

  // Sort groups by the number of features that they own (none of these
  // features should be instate). Comparison function returns True when g1 is
  // better than g2
  // Ties on the feature count are broken by id for the same reason
  // Criteria::CandidateComparison does it: the input order here comes from a
  // pointer-value sort, so without a deterministic final tie-break the outcome
  // depends on heap addresses. Small integer counts tie constantly.
  auto comp_fun = [&graph](GroupPtr g1, GroupPtr g2) {
    int nf1 = graph.NumFeatureCandidatesOwnedBy(g1);
    int nf2 = graph.NumFeatureCandidatesOwnedBy(g2);
    if (nf1 != nf2) {
      return nf1 > nf2;
    }
    return g1->id() < g2->id();
  };
  std::sort(candidates.begin(), candidates.end(), comp_fun);

  // For each group add all the features.
  for (auto it = candidates.begin(); it != candidates.end(); ++it) {
    auto g = *it;

    // Add group of features
    std::vector<FeaturePtr> features_of_group = graph.GetFeatureCandidatesOwnedBy(g);
    std::sort(features_of_group.begin(), features_of_group.end(),
              Criteria::CandidateComparison);

    // case 1: we're using depth optimization -- which means that we'll only add
    // the group if enough features optimize well.
    if (use_depth_opt_) {
      std::vector<FeaturePtr> good_features;
      std::vector<FeaturePtr> bad_features;
      for (auto f: features_of_group) {
        auto obs = graph.GetObservationsOf(f);
        if (obs.size() > 1) {
          if (!f->RefineDepth(gbc(), obs, refinement_options_)) {
            bad_features.push_back(f);
            continue;
          } else {
            good_features.push_back(f);
          }
        }
        else {
          LOG(ERROR) << "A feature with no observations should not be a candidate";
        }
      }
      if (good_features.size() >= num_gauge_xy_features_) {
        int num_features_added = 0;
        for (auto f: good_features) {
          AddFeatureToState(f);
          instate_features_.push_back(f);
          num_features_to_add--;
          num_features_added++;
          if (num_features_to_add == 0) {
            break;
          }
        }
        LOG(INFO) << "Added " << num_features_added << " features from group " << g->id();
      }
      else {
        DestroyFeatures(bad_features);
        affected_groups_.insert(g);
      }
    }

    // No depth optimization = the simple case.
    else {
      int num_features_added = 0;
      for (auto f: features_of_group) {
        AddFeatureToState(f);
        instate_features_.push_back(f);
        num_features_added++;
        num_features_to_add--;
        if (num_features_to_add == 0) {
          break;
        }
      }
      LOG(INFO) << "Added " << num_features_added << " features from group " << g->id();
    }

    // Add group
    AddGroupToState(g);
    needs_new_gauge_features_.push_back(g);
    LOG(INFO) << "group #" << g->id() << " added to EKF state" << std::endl;

    // Check whether or not we're done
    free_group_slots--;
    if ((num_features_to_add < num_gauge_xy_features_) || (free_group_slots==0)) {
      break;
    }
  }
}



void Estimator::InitializeJustCreatedTracks(GroupPtr g,
                                            std::list<FeaturePtr> &tracks)
{
  Graph& graph{*Graph::instance()};

  for (auto f : new_features_) {
    // distinguish two cases:
    // 1) feature is truely just created
    // 2) feature just lost its reference
#ifndef NDEBUG
    CHECK(f->track_status() == TrackStatus::CREATED &&
          f->status() == FeatureStatus::CREATED);
    CHECK(f->ref() == nullptr);
#endif
    f->SetRef(g);
    if (triangulate_pre_subfilter_ && !f->TriangulationSuccessful()) {
      f->Initialize(init_z_, {init_std_x_badtri_, init_std_y_badtri_, init_std_z_badtri_});
    } else if (sim_initialize_depths_) {
      f->Initialize(ids_to_depths_[f->id()], {init_std_x_, init_std_y_, init_std_z_});
    } else {
      f->Initialize(init_z_, {init_std_x_, init_std_y_, init_std_z_});
    }
    //std::cout << "feature id: " << f->id() << ", Xc" << f->Xc().transpose() << std::endl;

    graph.AddFeature(f);
    graph.AddFeatureToGroup(f, g);
    graph.AddGroupToFeature(g, f);

    // put back the detected feature
    tracks.push_back(f);
  }

}



void Estimator::AssociateTrackedFeaturesWithGroup(GroupPtr g,
                                                  std::list<FeaturePtr> &tracks)
{
  Graph& graph{*Graph::instance()};

  auto tracked_features = graph.GetFeaturesIf([](FeaturePtr f) -> bool {
    return f->track_status() == TrackStatus::TRACKED;
  });
  for (auto f : tracked_features) {
#ifndef NDEBUG
    CHECK(f->ref() != nullptr);
#endif

    // attach the new group to all the features being tracked
    graph.AddFeatureToGroup(f, g);
    graph.AddGroupToFeature(g, f);

    // put back the tracked feature
    tracks.push_back(f);
  }

}



void Estimator::OutlierRejection() {
  Graph& graph{*Graph::instance()};

  // Call outlier rejection algorithms
  if (use_MH_gating_ && instate_features_.size() > min_required_inliers_) {
    inliers_ = MHGating();
  } else {
    inliers_.resize(instate_features_.size());
    std::copy(instate_features_.begin(), instate_features_.end(),
              inliers_.begin());
  }
  if (use_1pt_RANSAC_) {

    // Since One-Pt RANSAC is a global measurement update, we need to make sure it's observable -- remove features and groups that are unobservable.
    DiscardAffectedGroups();
    FindNewGaugeFeatures();
    std::vector<FeaturePtr> inliers_backup = inliers_;
    inliers_.clear();
    for (auto f: inliers_backup) {
      if (f->instate()) {
        inliers_.push_back(f);
      }
    }

    inliers_ = OnePointRANSAC(inliers_);
  }

  // Remove rejected features from the state
  auto rejected_features = graph.GetFeaturesIf([](FeaturePtr f) -> bool {
    return f->status() == FeatureStatus::REJECTED_BY_FILTER;
  });
  if (use_canvas_) {
    for (auto f : rejected_features) {
      Canvas::instance()->Draw(f);
    }
  }
  LOG(INFO) << "Removed " << rejected_features.size() << " rejected features";
  for (auto f : rejected_features) {
#ifndef NDEBUG
    CHECK(f->ref() != nullptr);
#endif
    affected_groups_.insert(f->ref());
  }
  graph.RemoveFeatures(rejected_features);
  for (auto f : rejected_features) {
    RemoveFeatureFromState(f);
    Feature::Destroy(f);
  }

}



} // namespace xivo
