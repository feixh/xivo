#include <algorithm>

#include "graph.h"
#include "estimator.h"
#include "feature.h"
#include "group.h"
#include "param.h"
#include "geometry.h"


namespace xivo {

std::unique_ptr<Graph> Graph::instance_ = nullptr;

Graph* Graph::Create() {
  if (instance_ == nullptr) {
    instance_ = std::unique_ptr<Graph>(new Graph);
  }
  return instance_.get();
}

Graph* Graph::instance() {
  if (instance_ == nullptr) {
    LOG(WARNING) << "Graph not created yet! Creating one ...";
    Graph::Create();
  }
  return instance_.get();
}


void Graph::RemoveFeature(const FeaturePtr f) {
  GraphBase::RemoveFeature(f);

  // Removes feature from `gauge_features_` if it is a gauge feature.
  gauge_features_[f->ref()].erase(f);

  LOG(INFO) << "feature #" << f->id() << " removed from Graph";
}

void Graph::RemoveFeatures(const std::vector<FeaturePtr> &features) {
  for (auto f : features) {
    RemoveFeature(f);
  }
}

void Graph::RemoveGroup(const GroupPtr g) {
  GraphBase::RemoveGroup(g);
  gauge_features_.erase(g);
  LOG(INFO) << "group #" << g->id() << " removed from Graph";
}

void Graph::RemoveGroups(const std::vector<GroupPtr> &groups) {
  for (auto g : groups) {
    RemoveGroup(g);
  }
}

void Graph::AddFeature(FeaturePtr f) {
  GraphBase::AddFeature(f);
  LOG(INFO) << "feature #" << f->id() << " added to graph";
}

void Graph::AddGroup(GroupPtr g) {
  GraphBase::AddGroup(g);
  gauge_features_[g] = {};
  last_added_group_ = g;
  LOG(INFO) << "group #" << g->id() << " added to graph";
}

void Graph::AddGroupToFeature(GroupPtr g, FeaturePtr f) {

  int gid = g->id();
  int fid = f->id();

  CHECK(HasFeature(f)) << "feature #" << fid << " not exists";
  CHECK(HasGroup(g)) << "group #" << gid << " not exists";

  feature_adj_.at(fid).Add({g, f->xp()});
  LOG(INFO) << "group #" << gid << " added to feature #" << fid;
}

void Graph::AddFeatureToGroup(FeaturePtr f, GroupPtr g) {

  int fid = f->id();
  int gid = g->id();

  CHECK(HasFeature(f)) << "feature #" << fid << " not exists";
  CHECK(HasGroup(g)) << "group #" << gid << " not exists";

  group_adj_[gid].Add(fid);
  LOG(INFO) << "feature #" << fid << " added to group #" << gid;
}


std::vector<FeaturePtr> Graph::GetInstateFeatures() {
  return GetFeaturesIf([](FeaturePtr f) -> bool {
    return f->instate();
  });
}

std::vector<GroupPtr> Graph::GetInstateGroups() {
  return GetGroupsIf([](GroupPtr g) -> bool {
    return g->instate();
  });
}

std::vector<FeaturePtr> Graph::GetGaugeFeatureCandidates(GroupPtr owner) {
  return GetFeaturesIf([owner](FeaturePtr f) -> bool {
    return ((f->status() == FeatureStatus::INSTATE) &&
            (f->ref() == owner));
  });
}


std::vector<FeaturePtr> Graph::GetFeaturesOwnedBy(GroupPtr g) {
  return GetFeaturesIf([g](FeaturePtr f) -> bool {
    return (f->ref() == g);
  });
}


std::vector<FeaturePtr> Graph::GetFeatureCandidatesOwnedBy(GroupPtr g) {
  return GetFeaturesIf([g](FeaturePtr f) -> bool {
    return ((f->ref() == g) && (f->status() == FeatureStatus::READY));
  });
}


int Graph::NumFeatureCandidatesOwnedBy(GroupPtr g) {
  return GetFeatureCandidatesOwnedBy(g).size();
}


int Graph::NumFeaturesOwnedBy(GroupPtr g) {
  return GetFeaturesOwnedBy(g).size();
}


std::vector<GroupPtr> Graph::GetInstateGroupCandidates(int degrees_fixed) {
  return GetGroupsIf(
    [degrees_fixed, this](GroupPtr g) -> bool {
      return ((g->status() == GroupStatus::CREATED) &&
              (NumFeatureCandidatesOwnedBy(g) >= degrees_fixed));
    }
  );
}


void Graph::SanityCheck() {
  for (auto p : features_) {
    int fid = p.first;
    auto f = p.second;
    CHECK(feature_adj_.count(fid));
    for (auto obs : feature_adj_.at(fid)) {
      CHECK(groups_.count(obs.first));
    }
    CHECK(f->ref());
    CHECK(groups_.count(f->ref()->id()));
  }

  for (auto p : groups_) {
    int gid = p.first;
    auto g = p.second;
    CHECK(group_adj_.count(gid));
    for (auto fid : group_adj_.at(gid)) {
      CHECK(features_.count(fid));
    }
  }
  LOG(INFO) << "#graph.features=" << features_.size()
            << " ;#graph.groups=" << groups_.size();
}

std::vector<FeaturePtr> Graph::TransferFeatureOwnership(GroupPtr g,
                                                        const SE3 &gbc,
                                                        number_t cov_factor) {

  CHECK(HasGroup(g));

  std::vector<FeaturePtr> failed;
  int gid = g->id();

  for (int fid : group_adj_.at(gid)) {
    CHECK(HasFeature(fid));

    auto f = features_.at(fid);
    if (f->ref() == g) {
      // transfer ownership
      auto nref = FindNewOwner(f);
      if (nref) {
        bool success = f->ChangeOwner(nref, gbc);
        f->inflate_cov(cov_factor);

        if (success) {
          LOG(INFO) << "feature #" << fid << " transfered from group #" << gid
                    << " to group #" << nref->id();
        }
        else {
          LOG(WARNING) << "Graph::TransferFeatureOwnership: " <<
            "negative depth; mark feature #" << fid << " as failed";
          //f->ResetRef(nullptr);
          failed.push_back(f);
        }
      } else {
        // Note: back before we were saving old groups and features for loop
        // closure, reseting the feature group made sense. Now, we can keep the
        // both feature and the discarded group around in the Mapper.
        //f->ResetRef(nullptr);
        failed.push_back(f);
        LOG(WARNING) << "failed to find new owner for feature #" << fid;
      }
    }
  }
  return failed;
}

GroupPtr Graph::FindNewOwner(FeaturePtr f) {
  int fid = f->id();
  CHECK(features_.count(fid));
  CHECK(feature_adj_.count(fid));
  auto old_gid = f->ref()->id();
  for (const auto &obs : feature_adj_.at(fid)) {
    int gid = obs.first;
    GroupPtr g = groups_.at(gid);
    if ((gid != old_gid) && (g->status() == GroupStatus::INSTATE)) {
      // TODO: can have fancy measure on which group should be the best to be
      // the new owner
      // For now, just pick the first one met.
      return g;
    }
  }
  return nullptr;
}

void Graph::CleanIsolatedGroups() {
  std::vector<GroupPtr> islands;
  ;
  for (auto p : groups_) {
    if (group_adj_.at(p.first).empty()) {
      islands.push_back(p.second);
    }
  }
  LOG(INFO) << "removing " << islands.size() << " isolated groups" << std::endl;
  RemoveGroups(islands);
  for (auto g : islands) {
    Group::Deactivate(g);
  }
}

void Graph::CleanIsolatedFeatures() {
  std::vector<FeaturePtr> islands;
  ;
  for (auto p : features_) {
    if (feature_adj_.at(p.first).empty()) {
      islands.push_back(p.second);
    }
  }
  LOG(INFO) << "removing " << islands.size() << " isolated features"
            << std::endl;
  RemoveFeatures(islands);
  for (auto f : islands) {
    Feature::Deactivate(f);
  }
}

void Graph::CleanIsolatedNodes() {
  CleanIsolatedFeatures();
  CleanIsolatedGroups();
}


std::vector<FeaturePtr> Graph::FindNewGaugeFeatures(GroupPtr g) {
  // Get parameters
  ParameterServer& P{*ParameterServer::instance()};
  int directions_to_fix = P.get("num_gauge_xy_features", 3).asInt();
  number_t collinear_cross_prod_thresh =
    P.get("collinear_cross_prod_thresh", 1e-3).asDouble();

  // Estimator Pointer
  EstimatorPtr est = Estimator::instance();

  int num_gauge_features = gauge_features_[g].size();
  int num_to_find = directions_to_fix - num_gauge_features;

#ifndef NDEBUG
  CHECK(num_to_find >= 0);
  CHECK(num_to_find <= directions_to_fix);
  CHECK(directions_to_fix <= 3);
#endif

  // Candidates that could be gauge features
  std::vector<FeaturePtr> candidates = GetGaugeFeatureCandidates(g);
  //std::sort(candidates.begin(), candidates.end(),
  //          [est](FeaturePtr f1, FeaturePtr f2) -> bool {
  //            return est->FeatureCovXYComparison(f1, f2);
  //          });
  std::vector<FeaturePtr> candidates_backup = candidates;

  // Lambda function that checks whether or not features are collinear
  auto collinear_check = [](std::unordered_set<FeaturePtr> U,
                            number_t collinear_thresh) {
    // `U` is keyed by pointer, so iterating it directly hands
    // PointsAreCollinear its points in an order that depends on heap addresses
    // -- and the collinearity verdict is a thresholded cross product, which is
    // not invariant to which point plays the role of the base. Ordering by id
    // makes the test reproducible. See notes-stereo/m3a-determinism.md.
    std::vector<FeaturePtr> ordered(U.begin(), U.end());
    std::sort(ordered.begin(), ordered.end(),
              [](FeaturePtr a, FeaturePtr b) { return a->id() < b->id(); });
    std::vector<Vec3> gauge_xy_features_Xc;
    for (auto f: ordered) {
      gauge_xy_features_Xc.push_back(f->Xc());
    }
    return PointsAreCollinear(gauge_xy_features_Xc, collinear_thresh);
  };

  // Lambda function that adds features to guage list
  auto fill_slots = [this, num_to_find](GroupPtr g, std::vector<FeaturePtr> C)
  {
    std::vector<FeaturePtr> new_gauge_feats;
    for (int i=0; i<std::min(num_to_find, int(C.size())); i++) {
      gauge_features_[g].insert(C[i]);
      new_gauge_feats.push_back(C[i]);
    }
    return new_gauge_feats;
  };

  if (candidates.size() < num_to_find) {
    LOG(WARNING) << "[Graph::SwitchGaugeXYFeatures]: not enough instate features owned by group " << g->id();
  }


  std::vector<FeaturePtr> new_gauge_features_for_g;
  if ((candidates.size() == 0) || (num_to_find == 0)) {
    // do nothing
  } else if (candidates.size() <= num_to_find) {
    new_gauge_features_for_g = fill_slots(g, candidates);
  } else {
    // Try up to 10 times to find a set of features that are not collinear.
    std::unordered_set<FeaturePtr> gauge_features_backup = gauge_features_[g];
    for (int NT = 0; NT<10; NT++) {
      gauge_features_[g] = gauge_features_backup;
      new_gauge_features_for_g = fill_slots(g, candidates);
      if (gauge_features_[g].size() >= 3) {
        if (collinear_check(gauge_features_[g], collinear_cross_prod_thresh)) {
          std::shuffle(candidates.begin(), candidates.end(), *random_generator);
          if (NT==9) {
            LOG(WARNING) << "Did not find a set of non-collinear features. defaulting to using those with smallest covariance";
            gauge_features_[g] = gauge_features_backup;
            fill_slots(g, candidates_backup);
          }
        } else {
          break;
        }
      }
    }
  }

  for (auto f: new_gauge_features_for_g) {
    f->SetStatus(FeatureStatus::GAUGE);
    LOG(INFO) << "Feature " << f->id() << " is now a gauge feature for Group " << g->id();
  }

#ifndef NDEBUG
  CHECK(new_gauge_features_for_g.size() <= num_to_find);
#endif

  return new_gauge_features_for_g;
}


} // namespace xivo