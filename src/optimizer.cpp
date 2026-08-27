#include <iostream>

// 3rdparty
#include "glog/logging.h"

// xivo
#include "optimizer.h"
#include "utils.h"

namespace xivo {

std::unique_ptr<Optimizer> Optimizer::instance_ = nullptr;

OptimizerPtr Optimizer::Create(const Json::Value &cfg) {
  if (instance_) {
    LOG(WARNING) << 
      "Optimizer instance already created! Returning existing one ...";
  } else {
    instance_ = std::unique_ptr<Optimizer>(new Optimizer(cfg));
  }
  return instance_.get();
} 

OptimizerPtr Optimizer::instance() {
  CHECK(instance_ != nullptr);
  return instance_.get();
}

Optimizer::~Optimizer() {
}

Optimizer::Optimizer(const Json::Value &cfg) 
  : verbose_{false}, solver_type_{"cholmod"}, use_robust_kernel_{false}, initialized_{false}

{
  // setup flags
  verbose_ = cfg.get("verbose", true).asBool();
  solver_type_ = cfg.get("solver", "cholmod").asString();
  use_robust_kernel_ = cfg.get("use_robust_kernel", true).asBool();

  // _6_3: poses are parametrized by 6-dim vectors and landmarks by 3-dim vectors
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> solver;
  if (solver_type_ == "cholmod") {
    solver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
  } else if (solver_type_ == "csparse") {
    solver = g2o::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
  } else if (solver_type_ == "dense") {
    solver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
  } else {
    // default to cholmod
    LOG(WARNING) << "unknown linear solver type; default to cholmod";
    solver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
  }


  optimizer_.setVerbose(verbose_);

  auto algorithm = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(std::move(solver)));

  optimizer_.setAlgorithm(algorithm);
}


FeatureVertex* Optimizer::CreateFeatureVertex(const FeatureAdapter &f) {
  auto fv = new FeatureVertex();
  fv->setId(VertexId(f.id, true));
  fv->setMarginalized(true);
  fv->setEstimate(f.Xs);
  fvertices_[f.id] = fv;
  optimizer_.addVertex(fv);
  return fv;
}

GroupVertex* Optimizer::CreateGroupVertex(const GroupAdapter &g) {
  auto gv = new GroupVertex();
  gv->setId(VertexId(g.id, false));
  gv->setEstimate(g.gsb);
  // FIXME (xfei): to fix gauge freedom
  // gv->setFixed(true);  
  gvertices_[g.id] = gv;
  optimizer_.addVertex(gv);
  return gv;
}

Edge* Optimizer::CreateEdge(FeatureVertex *fv, GroupVertex *gv, const Vec2 &xp, const Mat2 &IM) {
  auto e = new Edge();
  e->setVertex(0, fv);
  e->setVertex(1, gv);
  e->setMeasurement(xp);
  if (IM.isZero(0)) {
    e->setInformation(Mat2::Identity());
  } else {
    e->setInformation(IM);  // IM = Information Matrix
  }

  if (use_robust_kernel_) {
    auto rk = new g2o::RobustKernelHuber();
    e->setRobustKernel(rk);
  }
  optimizer_.addEdge(e);
  return e;
}

void Optimizer::AddFeature(const FeatureAdapter &f, const VectorObsAdapterG &obs) {
  // CHECK(!fvertices_.count(f->id()) << "Feature #" << f->id() << " already in optimization graph";
  if (!fvertices_.count(f.id)) {
    // feature vertex not exist, create one
    CreateFeatureVertex(f);
  }
  auto fv = fvertices_.at(f.id);

  for (auto [g, xp, IM]: obs) {
    if (!gvertices_.count(g.id)) {
      // group vertex not exist, create one
      CreateGroupVertex(g);
    }
    auto gv = gvertices_.at(g.id);
    // FIXME (xfei): make sure no duplicate edges are added
    CreateEdge(fv, gv, xp, IM);
  }
}

void Optimizer::AddGroup(const GroupAdapter &g, const VectorObsAdapterF &obs) {
  if (!gvertices_.count(g.id)) {
    CreateGroupVertex(g);
  }
  auto gv = gvertices_.at(g.id);

  for (auto [f, xp, IM]: obs) {
    if (!fvertices_.count(f.id)) {
      CreateFeatureVertex(f);
    }
    auto fv = fvertices_.at(f.id);
    // FIXME (xfei): make sure no duplicate edges are added
    CreateEdge(fv, gv, xp, IM);
  }
}

void Optimizer::Solve(int max_iters) {
  if (!initialized_) {
    optimizer_.initializeOptimization();
    initialized_ = true;
  }

  // for (auto fid : fvertices_) {
  //   Vec3 optimizer_.vertex(fid)->estimate();
  // }
  optimizer_.setVerbose(verbose_);

  // compute error once before solving
  optimizer_.initializeOptimization();
  optimizer_.computeActiveErrors();

  int num_active_edges = optimizer_.activeEdges().size();
  number_t init_average_chi2 = optimizer_.chi2() / num_active_edges;

  optimizer_.optimize(max_iters);

  number_t average_chi2 = optimizer_.chi2() / num_active_edges;
  std::cout << StrFormat("average chi2: %0.2f -> %0.2f\n", init_average_chi2, average_chi2);
}


} // namespace xivo
