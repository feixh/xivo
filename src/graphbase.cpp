#include "graphbase.h"


namespace xivo {

void GraphBase::AddFeature(const FeaturePtr f) {
  int fid = f->id();
  CHECK(!features_.count(fid)) << "feature #" << fid << " already exists";
  features_[fid] = f;
  feature_adj_[fid] = {};
}

void GraphBase::AddGroup(const GroupPtr g) {
  int gid = g->id();
  CHECK(!groups_.count(gid)) << "group #" << gid << " already exists";
  groups_[gid] = g;
  group_adj_[gid] = {};
}

void GraphBase::RemoveFeature(const FeaturePtr f) {
  CHECK(HasFeature(f)) << "feature #" << f->id() << " not exists";
  int fid = f->id();
  features_.erase(fid);
  for (const auto &obs : feature_adj_.at(fid)) {
    group_adj_.at(obs.first).Remove(fid);
  }
  feature_adj_.erase(fid);
}

void GraphBase::RemoveFeatures(const std::vector<FeaturePtr> &features) {
  for (auto f : features) {
    RemoveFeature(f);
  }
}

void GraphBase::RemoveGroup(const GroupPtr g) {
  CHECK(HasGroup(g)) << "group #" << g->id() << " not exists";
  int gid = g->id();
  groups_.erase(gid);
  for (auto fid : group_adj_.at(gid)) {
    auto f = features_.at(fid);
    // need to transfer ownership of the feature first
    if (f->ref() == g) {
      LOG(FATAL) << "removing group #" << gid << " but feature #" << fid
                 << " refers to it";
    }
    feature_adj_.at(fid).Remove(gid);
  }
  group_adj_.erase(gid);
}

void GraphBase::RemoveGroups(const std::vector<GroupPtr> &groups) {
  for (auto g : groups) {
    RemoveGroup(g);
  }
}

bool GraphBase::HasGroup(GroupPtr g) const {
  return HasGroup(g->id());
}

bool GraphBase::HasGroup(int gid) const {
  return groups_.count(gid) && group_adj_.count(gid);
}

bool GraphBase::HasFeature(FeaturePtr f) const {
  return HasFeature(f->id());
}

bool GraphBase::HasFeature(int fid) const {
  return features_.count(fid) && feature_adj_.count(fid);
}

FeaturePtr GraphBase::GetFeature(int fid) const {
  return features_.at(fid);
}

std::vector<FeaturePtr> GraphBase::GetFeatures() const {
  std::vector<FeaturePtr> out;
  out.reserve(features_.size());
  for (auto p : features_) {
    out.push_back(p.second);
  }
  return out;
}

GroupPtr GraphBase::GetGroup(int gid) const {
  return groups_.at(gid);
}

std::vector<GroupPtr> GraphBase::GetGroups() const {
  std::vector<GroupPtr> out;
  out.reserve(groups_.size());
  for (auto p : groups_) {
    out.push_back(p.second);
  }
  return out;
}

std::vector<FeaturePtr> GraphBase::GetFeaturesOf(GroupPtr g) const {
  std::vector<FeaturePtr> out;
  for (int fid : group_adj_.at(g->id())) {
    out.push_back(features_.at(fid));
  }
  return out;
}

std::vector<GroupPtr> GraphBase::GetGroupsOf(FeaturePtr f) const {
  std::vector<GroupPtr> out;
  for (const auto &obs : feature_adj_.at(f->id())) {
    out.push_back(groups_.at(obs.first));
  }
  return out;
}

const FeatureAdj &GraphBase::GetFeatureAdj(FeaturePtr f) const {
  return feature_adj_.at(f->id());
}

const GroupAdj &GraphBase::GetGroupAdj(GroupPtr g) const {
  return group_adj_.at(g->id());
}

std::vector<FeaturePtr>
GraphBase::GetFeaturesIf(std::function<bool(FeaturePtr)> pred) const {
  std::vector<FeaturePtr> out;
  for (auto p : features_) {
    if (pred(p.second)) {
      out.push_back(p.second);
    }
  }
  return out;
}

std::vector<GroupPtr>
GraphBase::GetGroupsIf(std::function<bool(GroupPtr)> pred) const {
  std::vector<GroupPtr> out;
  for (auto p : groups_) {
    if (pred(p.second)) {
      out.push_back(p.second);
    }
  }
  return out;
}

std::vector<Observation> GraphBase::GetObservationsOf(FeaturePtr f) const {
  const FeatureAdj &adj = feature_adj_.at(f->id());
  std::vector<Observation> out;
  for (const auto &obs : adj) {
    Observation o{groups_.at(obs.first), obs.second};
    auto it = adj.right.find(obs.first);
    if (it != adj.right.end()) {
      o.has_right = true;
      o.xp_r = it->second;
    }
    out.push_back(o);
  }
  return out;
}

Observation GraphBase::GetObservationOf(FeaturePtr f, GroupPtr g) const {
  const FeatureAdj &adj = feature_adj_.at(f->id());
  for (const auto &obs : adj) {
    if (obs.first == g->id()) {
      Observation ret;
      ret.g = g;
      ret.xp = obs.second;
      auto it = adj.right.find(obs.first);
      if (it != adj.right.end()) {
        ret.has_right = true;
        ret.xp_r = it->second;
      }
      return ret;
    }
  }

  LOG(FATAL) << "GraphBase: could not find observation of feature " <<
    f->id() << " from group " << g->id();
}


}