// Bundle Adjustment/Pose Graph Optimization module.
// Run in the background, trigger every ... seconds?
// Reference:
//  g2o/examples/bal/bal_demo.cpp
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#pragma once
#include <memory>

#include "json/json.h"

#include "optimizer_types.h"

namespace xivo {


class Optimizer;
using OptimizerPtr = Optimizer*;


class Optimizer {
public:
  ~Optimizer();
  static OptimizerPtr Create(const Json::Value &cfg);
  static OptimizerPtr instance();
  void Solve(int iters=1);

  void AddFeature(const FeatureAdapter &f, const VectorObsAdapterG &obs);
  void AddGroup(const GroupAdapter &g, const VectorObsAdapterF &obs);

private:
  Optimizer() = delete;
  Optimizer(const Optimizer &) = delete;
  Optimizer &operator=(const Optimizer &) = delete;

  Optimizer(const Json::Value &cfg);

  /** g2o vertex index for a XIVO feature or group ID.
   *
   *  g2o requires vertex indices to be unique across the whole graph, and both
   *  features and groups become vertices here. That used to be arranged by
   *  keeping every group ID below `Feature::counter0` and every feature ID above
   *  it -- which capped a run at 10000 images, because XIVO creates one group per
   *  image. Interleaving instead (groups even, features odd) removes the cap: the
   *  two kinds cannot collide whatever their values, so group IDs are free to
   *  count as high as a run needs.
   *
   *  Only this function is allowed to know about the encoding; `fvertices_` and
   *  `gvertices_` stay keyed by the raw XIVO IDs, as does every caller. */
  static int VertexId(int id, bool is_feature) {
    return 2 * id + (is_feature ? 1 : 0);
  }

  FeatureVertex* CreateFeatureVertex(const FeatureAdapter &f);
  GroupVertex* CreateGroupVertex(const GroupAdapter &g);
  Edge* CreateEdge(FeatureVertex *fv, GroupVertex *gv, const Vec2 &xp, const Mat2 &IM);

  // the instance class memeber
  static std::unique_ptr<Optimizer> instance_;

  // flags
  bool verbose_;
  std::string solver_type_;
  bool use_robust_kernel_;
  bool initialized_;
  
  // graph structure: features & groups as vertices
  std::unordered_map<int, FeatureVertex*> fvertices_;
  std::unordered_map<int, GroupVertex*> gvertices_;

  // g2o variables
  g2o::SparseOptimizer optimizer_;
};

} // namespace xivo
