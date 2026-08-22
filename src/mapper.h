// The Mapper module.
// Author: Stephanie Tsuei (stephanietsuei@ucla.edu)
#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "json/json.h"
#include "DBoW2/DBoW2.h"
#include "pnp_ransac.h"

#include "core.h"
#include "feature.h"
#include "group.h"
#include "graphbase.h"
#include "fastbrief.h"

namespace xivo {

/** Singleton class in charge of loop closures and pose-graph optimization.
 *  It manages relationships for all features and groups formerly in the
 *  filter. It also contains functions to search for loop closures and runs
 *  pose-graph optimization.
 */

using FastBriefVocabulary =
  DBoW2::TemplatedVocabulary<FastBrief::TDescriptor, FastBrief>;

using LCMatch = std::pair<FeaturePtr, FeaturePtr>;


// Helper functions for interfacing with Lambdatwist PnP RANSAC
std::unique_ptr<cvl::PnpParams> GetRANSACParams(const Json::Value &cfg);
void GetPnPInput(std::vector<LCMatch> &matches,
                 std::vector<cvl::Vector3D> &xs,
                 std::vector<cvl::Vector2D> &yns);
std::vector<LCMatch> GetInlierMatches(std::vector<LCMatch> &matches,
                                      std::vector<cvl::Vector3D> &xs,
                                      std::vector<cvl::Vector2D> &yns,
                                      cvl::PoseD soln);


class Mapper : public GraphBase {

public:
  ~Mapper();
  static MapperPtr Create(const Json::Value &cfg);
  static MapperPtr instance();

  void AddFeature(FeaturePtr f, const FeatureAdj& feature_adj, const SE3 &gbc);
  void AddGroup(GroupPtr f, const GroupAdj& group_adj);
  void RemoveFeature(const FeaturePtr f);
  void RemoveGroup(const GroupPtr g);

  std::vector<FeaturePtr> GetFeaturesOf(GroupPtr g) const;
  std::vector<GroupPtr> GetGroupsOf(FeaturePtr f) const;

  std::mutex features_mtx;
  std::mutex groups_mtx;

  bool UseLoopClosure() const { return use_loop_closure_; }
  std::vector<LCMatch> DetectLoopClosures(
    const std::vector<FeaturePtr>& instate_features, const SE3 &gbc);

private:
  Mapper() = default;
  Mapper(const Json::Value &cfg);
  Mapper(const Mapper &) = delete;
  Mapper &operator=(const Mapper &) = delete;

  static std::unique_ptr<Mapper> instance_;

  // Loop closure variables
  bool use_loop_closure_;
  bool merge_features_;
  double feature_merge_cov_factor_;
  int uplevel_word_search_;
  double nn_dist_thresh_;
  std::unique_ptr<FastBriefVocabulary> voc_;

  /** Maps DBoW2 words to a set of features that map to the same word in the
   *  vocabulary. */
  std::unordered_map<DBoW2::WordId, std::unordered_set<FeaturePtr>> InvIndex_;

  // Functions related to loop closure
  std::unordered_set<FeaturePtr> GetLoopClosureCandidates(const DBoW2::WordId& word_id);
  void UpdateInverseIndex(const DBoW2::WordId &word_id, FeaturePtr f);

  /** Solves the P3P problem for outlier rejection of loop closure matches.
   *  Contains an interface to the lambdatwist P3P solver.
   *  Deletes all matches that aren't within tolerance*/
  void RANSAC(std::vector<LCMatch> &match_list);

  // RANSAC parameters
  std::unique_ptr<cvl::PnpParams> ransac_params_;
};



}