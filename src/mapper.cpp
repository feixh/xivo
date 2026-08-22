#include "mapper.h"
#include "feature.h"
#include "group.h"

#ifdef USE_G2O
#include "optimizer_adapters.h"
#endif


namespace cvl {


/**
     * @brief apply a fast version of outs =P*ins and then project outs
     * @param pose
     * @param ins
     * @param outs
     */
template<class T>
void apply_and_project(const Pose<T>& pose,
                       const std::vector<Vector3D>& ins,
                       std::vector<Vector2D>& outs){
    outs.resize(ins.size());
    Vector4<T> xh;
    Matrix4x4<T> M=pose.get4x4(); // much faster...
    for(unsigned int i=0;i<ins.size();++i){
        xh=ins[i].homogeneous();
        xh=M*xh;
        outs[i]=Vector2<T>(xh[0]/xh[2], xh[1]/xh[2]);
    }
}

}


// for debug printing
template<class T>
void PrintRotMat(T mat) {
  for (int i=0; i<3; i++) {
    std::cout << mat(i,0) << ", " << mat(i,1) << ", " << mat(i,2) << std::endl;
  }
}

template<class T>
void PrintT(T v) {
  std::cout << v(0,0) << ", " << v(1,0) << ", " << v(2,0) << std::endl;
}

namespace xivo {


std::unique_ptr<Mapper> Mapper::instance_{nullptr};

Mapper* Mapper::instance() {  return instance_.get(); }


std::unique_ptr<cvl::PnpParams> GetRANSACParams(const Json::Value &cfg) {
  auto params = std::make_unique<cvl::PnpParams>();
  params->min_iterations = cfg.get("min_iter", 1).asInt();
  params->max_iterations = cfg.get("max_iter", 100).asInt();
  params->threshold = cfg.get("threshold", 0.02).asDouble();
  params->min_probability = cfg.get("probability", 0.98).asDouble();
  params->early_exit = cfg.get("early_exit", false).asBool();
  params->early_exit_inlier_ratio = cfg.get("early_exit_inlier_ratio", 0.85).asDouble();
  params->early_exit_min_iterations = cfg.get("early_exit_min_iter", 5).asInt();

  // This was a formula in Corvis for translating a threshold in pixels to
  // a threshold for RANSAC.
  // TODO (ST): Figure out where this formula came from
  //double threshold = cfg.get("threshold", 0.02).asDouble();
  //params->threshold = 1.0 - cos(atan(sqrt( 2.0 )* threshold /300.0));
  return params;
}


void GetPnPInput(const std::vector<LCMatch> &matches,
                 const SE3 &gbc,
                 std::vector<cvl::Vector3D> &xs,
                 std::vector<cvl::Vector2D> &yns)
{
  xs.resize(matches.size());
  yns.resize(matches.size());
  for (int i=0; i<matches.size(); i++) {
    LCMatch m = matches[i];
    FeaturePtr ekf_feat = m.first;
    FeaturePtr map_feat = m.second;

    Vec2 y_px = ekf_feat->back();
    Vec2 y = Camera::instance()->UnProject(y_px);
    Vec3 X = map_feat->Xs();
    cvl::Vector2D last_obs(y(0), y(1)); 
    cvl::Vector3D map_pos(X(0), X(1), X(2));

    xs[i] = map_pos;
    yns[i]= last_obs;
  }
}


void GetInlierMatches(std::vector<LCMatch> &matches,
                      std::vector<cvl::Vector3D> &Xs,
                      std::vector<cvl::Vector2D> &yns,
                      cvl::PoseD ransac_soln,
                      double tol,
                      std::vector<LCMatch> &ransac_matches)
{
  // Transform all Xs to camera frame using ransac solution
  std::vector<cvl::Vector3D> Xc;
  cvl::apply(ransac_soln, Xs, Xc);

  // compare points to projected points
  std::vector<cvl::Vector2D> yns_soln;
  cvl::apply_and_project(ransac_soln, Xs, yns_soln);
  for (int i=0; i<Xs.size(); i++) {
    cvl::Vector2D diff = yns_soln[i] - yns[i];
    double dist = diff.norm();
    if (dist < tol) {
      //std::cout << "diff inlier: " << diff[0] << ", " << diff[1] << std::endl;
      //std::cout << "dist inlier: " << dist << std::endl;
      ransac_matches.push_back(matches[i]);
    }
    //else {
      //std::cout << "diff outlier: " << diff[0] << ", " << diff[1] << std::endl;
      //std::cout << "dist outlier: " << dist << std::endl;
    //}
  }
}


Mapper::Mapper(const Json::Value &cfg) {

  use_loop_closure_ = cfg.get("detectLoopClosures", false).asBool();
  std::string vocab_file =
    cfg.get("vocabulary", "cfg/ukbench10K_FASTBRIEF32.yml.gz").asString();

  uplevel_word_search_ = cfg.get("uplevel_word_search", 0).asInt();
  nn_dist_thresh_ = cfg.get("nn_dist_thresh", 20.0).asDouble();
  voc_ = std::make_unique<FastBriefVocabulary>(vocab_file);

  merge_features_ = cfg.get("merge_features", true).asBool();
  feature_merge_cov_factor_ = cfg.get("feature_merge_cov_factor", 1.0).asDouble();

  ransac_params_ = GetRANSACParams(cfg["RANSAC"]);
}


MapperPtr Mapper::Create(const Json::Value &cfg) {
  if (instance_ == nullptr) {
    instance_ = std::unique_ptr<Mapper>(new Mapper(cfg));
  }
  return instance_.get();
}


Mapper::~Mapper() {}


void Mapper::AddFeature(FeaturePtr f, const FeatureAdj& f_obs, const SE3 &gbc) {

#ifdef USE_G2O
  adapter::AddFeature(f);
#endif

  int fid = f->id();
  CHECK(!features_.count(fid)) << "feature #" << fid << " already in mapper";

  // Merge observations of feature
  bool merge_success = false;
  features_mtx.lock();
  // Case 1: feature not in map. Just add it.
  int matched_map_feat_id = f->LoopClosureMatch();
  if (matched_map_feat_id == -1) {
    features_[fid] = f;
    feature_adj_[fid] = f_obs;
    merge_success = true;
  }
  // Case 2: feature has a loop closure. Need to merge observations, tracks,
  // and state estimates
  // Note (ST): Corvis didn't merge estimates at all, but it was marked as a
  // todo item
  else {
    FeaturePtr matched_feat = features_[matched_map_feat_id];
    if (merge_features_) {
      f->inflate_cov(feature_merge_cov_factor_);
      merge_success = matched_feat->Merge(f, gbc);
      for (auto p: f_obs) {
        feature_adj_[matched_map_feat_id].insert({p.first, p.second});
      }
    } else {
      merge_success = true;
    }
  }
  features_mtx.unlock();

  if (!merge_success) {
    LOG(WARNING) << "Mapper: Failed to merge feature #" << fid
      << " into feature #" << matched_map_feat_id;
    //return;
  }

  // Add all feature descriptors to invese index
  FeaturePtr fptr = (matched_map_feat_id == -1) ? f : features_[matched_map_feat_id];
  std::vector<FastBrief::TDescriptor> all_descriptors = f->GetAllDBoWDesc();
  for (auto desc: all_descriptors) {
    DBoW2::WordId wid = voc_->transform(desc);
    UpdateInverseIndex(wid, fptr);
  }

  // If the feature has been merged with a previous feature, then we will
  // destroy it and its slot in the MemoryManager will become uninitialized.
  if (matched_map_feat_id > -1) {
    Feature::Destroy(f);
    //std::cout << "feature #" << fid << " merged with feature " <<
    //  matched_map_feat_id << std::endl;
    LOG(INFO) << "feature #" << fid << " merged with feature " <<
      matched_map_feat_id;
  }
  else {
    LOG(INFO) << "feature #" << fid << " added to mapper";
    //std::cout << "feature #" << fid << " added to mapper" << std::endl;
  }
}


void Mapper::AddGroup(GroupPtr g, const GroupAdj& g_features) {

#ifdef USE_G2O
  adapter::AddGroup(g);
#endif

  int gid = g->id();
  CHECK(!groups_.count(gid)) << "group #" << gid << " already in mapper";

  groups_mtx.lock();
  groups_[gid] = g;
  group_adj_[gid] = g_features;
  groups_mtx.unlock();

  LOG(INFO) << "group #" << gid << " added to mapper-graph";
}


void Mapper::RemoveFeature(const FeaturePtr f) {
  CHECK(HasFeature(f)) << "feature #" << f->id() << " not in mapper-graph";

  int fid = f->id();

  features_mtx.lock();
  features_.erase(fid);
  for (const auto &obs : feature_adj_.at(fid)) {
    if (HasGroup(obs.first)) {
      group_adj_.at(obs.first).Remove(fid);
    }
  }
  feature_adj_.erase(fid);
  features_mtx.unlock();

  LOG(INFO) << "feature #" << fid << " removed from mapper-graph";
}

void Mapper::RemoveGroup(const GroupPtr g) {
  CHECK(HasGroup(g)) << "group #" << g->id() << " not in mapper-graph";

  int gid = g->id();

  groups_mtx.lock();
  groups_.erase(gid);
  for (auto fid : group_adj_.at(gid)) {
    if (HasFeature(fid)) {
      auto f = features_.at(fid);

      // features should always get removed before removing the group
      if (f->ref() == g) {
        LOG(FATAL) << "mapper removing group #" << gid << " but feature #" << fid
                  << " refers to it";
      }

      feature_adj_.at(fid).Remove(gid);
    }
  }
  group_adj_.erase(gid);
  groups_mtx.unlock();

  LOG(INFO) << "group #" << gid << " removed from mapper-graph";
}


std::vector<FeaturePtr> Mapper::GetFeaturesOf(GroupPtr g) const {
  std::vector<FeaturePtr> out;
  for (int fid : group_adj_.at(g->id())) {
    if (HasFeature(fid)) {
      out.push_back(features_.at(fid));
    }
  }
  return out;
}


std::vector<GroupPtr> Mapper::GetGroupsOf(FeaturePtr f) const {
  std::vector<GroupPtr> out;
  for (const auto &obs : feature_adj_.at(f->id())) {
    if (HasGroup(obs.first)) {
      out.push_back(groups_.at(obs.first));
    }
  }
  return out;
}


void Mapper::UpdateInverseIndex(const DBoW2::WordId &word_id, FeaturePtr f) {
  if (InvIndex_.count(word_id) > 0) {
    InvIndex_[word_id].insert(f);
  }
  else {
    InvIndex_.insert({word_id, {f}});
  }
}

std::unordered_set<FeaturePtr> Mapper::GetLoopClosureCandidates(
  const DBoW2::WordId& word_id)
{
  const DBoW2::NodeId &node_id(voc_->getParentNode(word_id, uplevel_word_search_));
  std::vector<DBoW2::WordId> words_under_node;
  voc_->getWordsFromNode(node_id, words_under_node);

  std::unordered_set<FeaturePtr> ret;
  for (auto w: words_under_node) {
    std::unordered_set<FeaturePtr> features_in_word = InvIndex_[w];
    ret.merge(features_in_word);
  }
  return ret;
}


std::vector<LCMatch> Mapper::DetectLoopClosures(const std::vector<FeaturePtr>& instate_features, const SE3 &gbc)
{

  std::vector<LCMatch> matches;
  std::vector<LCMatch> ransac_matches;

  for (auto f: instate_features) {
    FastBrief::TDescriptor desc = f->GetDBoWDesc();

    // Convert descriptor to word and find other features that match to the
    // same word
    const DBoW2::WordId& word_id(voc_->transform(desc));
    std::unordered_set<FeaturePtr> other_matches =
      GetLoopClosureCandidates(word_id);

    // Only use the matches that close enough to the descriptor
    double distance = nn_dist_thresh_;
    FeaturePtr best_match = nullptr;
    for (auto f1: other_matches) {
      FastBrief::TDescriptor desc1 = f1->GetDBoWDesc();
      double d = FastBrief::distance(desc, desc1);
      if (d < distance) {
        distance = d;
        best_match = f1;
      }
    }

    if (best_match != nullptr) {
      LOG(INFO) << "Mapper: matched feature " << f->id() << " to feature "
        << best_match->id() << std::endl;
      matches.push_back(LCMatch(f, best_match));
      f->SetLCMatch(best_match->id());
    }
  }

  // If number of matches is at least 5, check with P3P RANSAC. Otherwise, don't
  // return anything. Technically, we only need 4, but then the probability that
  // RANSAC fits an outlier is too high.
  if (matches.size() >= 5) {
    //std::cout << std::endl;
    //std::cout << "Mapper: matched " << matches.size() << " features" << std::endl;
    LOG(INFO) << "Mapper: matched " << matches.size() << " features" << std::endl;

    // debug printing -- this block is useful for adjusting parameters
    // `nn_dist_thresh` and `RANSAC.threshold` in configuration files.
    // If the filter is working as intended, then the position of matches in
    // the spatial frame should be "pretty close".
    /*
    for (int j=0; j< matches.size(); j++) {
      LCMatch m = matches[j];
      std::cout << "Match " << j << std::endl;
      PrintT(m.first->Xs(gbc));
      PrintT(m.second->Xs(gbc));
    }
    */

    std::vector<cvl::Vector3D> Xs;
    std::vector<cvl::Vector2D> yns;
    GetPnPInput(matches, gbc, Xs, yns);
    cvl::PoseD camera_pose = cvl::pnp_ransac(Xs, yns, *ransac_params_);
    GetInlierMatches(matches, Xs, yns, camera_pose, ransac_params_->threshold,
                     ransac_matches);

    LOG(INFO) << "Mapper: RANSAC kept " << ransac_matches.size() << " matches" << std::endl;
    // more debug printing to make sure that RANSAC is working.
    /*
    std::cout << "Mapper: RANSAC kept " << ransac_matches.size() << " matches" << std::endl;
    for (int j=0; j< ransac_matches.size(); j++) {
      LCMatch m = ransac_matches[j];
      std::cout << "Match " << j << std::endl;
      PrintT(m.first->Xs());
      PrintT(m.second->Xs());
    }
    */
  }

  // If RANSAC only kept 4 matches, it might be by chance, so let's get rid
  // of them.
  if (ransac_matches.size() <= 4) {
    ransac_matches.clear();
  }

  return ransac_matches;
}


}