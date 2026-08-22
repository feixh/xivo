// factory method to create a system
// Author: Xiaohan Fei (feixh@cs.ucla.edu)
#include "param.h"
#include "camera_manager.h"
#include "stereo.h"
#include "mm.h"
#include "tracker.h"
#include "graph.h"
#include "estimator.h"
#include "mapper.h"

#ifdef USE_G2O
#include "optimizer.h"
#endif

namespace xivo {

namespace {

/** Fail at startup when the build and the config disagree about capacity, or the
 * memory pools cannot cover the caps that will draw on them.
 *
 * The pools in `mm.h` are pre-allocated and fixed; `GetItem` calls `LOG(FATAL)`
 * when every slot is active. That abort lands wherever the sequence happens to
 * need one more feature -- typically minutes into a run, from inside the tracker
 * -- and says nothing about which config number caused it. Both caps are known
 * before the first frame, so check them here instead.
 *
 * `num_features_max` is the tracker's steady-state target, so it is a hard lower
 * bound on the feature pool. Slots are also held by features the tracker has
 * dropped but the estimator has not destroyed yet, which is why the bound is
 * necessary and not sufficient; measured peak usage on TUM-VI room1-room6 is
 * roughly 1.7x `num_features_max`, and a pool at 1.67x did abort. Hence the
 * warning at 2x, which is also about the ratio the upstream configs ship
 * (200 features for a tracker cap of 60).
 */
void CheckMemoryPools(const Json::Value &cfg, const Json::Value &tracker_cfg) {
  auto mm = MemoryManager::instance();

  // A config can state the EKF capacity it was tuned for. Capacity is a
  // compile-time constant (EKF_MAX_FEATURES/EKF_MAX_GROUPS), so a config and a
  // binary can silently disagree: cfg/tumvi_stereo.json run against a 30-feature
  // build loses half its accuracy and reports nothing unusual, because the
  // tracker still finds 180 features and the filter quietly drops most of them.
  // Absent these keys nothing is checked, so upstream configs are unaffected.
  const int need_f = cfg.get("require_ekf_max_features", 0).asInt();
  if (need_f > kMaxFeature) {
    LOG(FATAL) << "config requires EKF_MAX_FEATURES >= " << need_f
               << " but this binary was built with " << kMaxFeature
               << "; rebuild with -DEKF_MAX_FEATURES=" << need_f;
  }
  const int need_g = cfg.get("require_ekf_max_groups", 0).asInt();
  if (need_g > kMaxGroup) {
    LOG(FATAL) << "config requires EKF_MAX_GROUPS >= " << need_g
               << " but this binary was built with " << kMaxGroup
               << "; rebuild with -DEKF_MAX_GROUPS=" << need_g;
  }
  // Defaults must track tracker.cpp's, or the check reasons about a cap the
  // tracker will not use.
  const int tracker_max = tracker_cfg.get("num_features_max", 150).asInt();

  if (mm->max_features() < tracker_max) {
    LOG(FATAL) << "memory.max_features (" << mm->max_features()
               << ") is below tracker_cfg.num_features_max (" << tracker_max
               << "); the tracker alone would exhaust the pool";
  }
  if (mm->max_features() < kMaxFeature) {
    LOG(FATAL) << "memory.max_features (" << mm->max_features()
               << ") is below the EKF's feature capacity (" << kMaxFeature
               << ", set by EKF_MAX_FEATURES at build time)";
  }
  if (mm->max_groups() < kMaxGroup) {
    LOG(FATAL) << "memory.max_groups (" << mm->max_groups()
               << ") is below the EKF's group capacity (" << kMaxGroup
               << ", set by EKF_MAX_GROUPS at build time)";
  }
  if (mm->max_features() < 2 * tracker_max) {
    // ERROR, not WARNING: glog's default stderrthreshold is ERROR, and an
    // advisory the user cannot see does not prevent the abort it is warning
    // about.
    LOG(ERROR) << "memory.max_features (" << mm->max_features()
               << ") is under 2x tracker_cfg.num_features_max (" << tracker_max
               << "); runs may abort on pool exhaustion";
  }
}

} // namespace

EstimatorPtr CreateSystem(const Json::Value &cfg) {
  static bool system_created{false};

  if (system_created) {
    return Estimator::instance();
  }

  // Initialize paramter server
  ParameterServer::Create(cfg);
  LOG(INFO) << "Parameter server created";

  // Load camera parameters
  auto cam_cfg = cfg["camera_cfg"].isString()
                     ? LoadJson(cfg["camera_cfg"].asString())
                     : cfg["camera_cfg"];
  Camera::Create(cam_cfg, 0);
  LOG(INFO) << "Camera created";

  // Stereo: a second camera plus the fixed rig geometry. Absent these keys the
  // system stays monocular and every stereo code path is skipped.
  if (cfg.get("stereo", false).asBool()) {
    auto cam1_cfg = cfg["camera1_cfg"].isString()
                        ? LoadJson(cfg["camera1_cfg"].asString())
                        : cfg["camera1_cfg"];
    if (cam1_cfg.isNull()) {
      LOG(FATAL) << "\"stereo\": true requires a \"camera1_cfg\" block";
    }
    Camera::Create(cam1_cfg, 1);
    LOG(INFO) << "Camera 1 created";

    auto rig_cfg = cfg["stereo_cfg"].isString()
                       ? LoadJson(cfg["stereo_cfg"].asString())
                       : cfg["stereo_cfg"];
    if (rig_cfg.isNull()) {
      LOG(FATAL) << "\"stereo\": true requires a \"stereo_cfg\" block";
    }
    StereoRig::Create(rig_cfg);
    LOG(INFO) << "Stereo rig created, baseline="
              << StereoRig::instance()->baseline() << " m";
  }

  // Initialize memory manager
  MemoryManager::Create(cfg["memory"].get("max_features", 256).asInt(),
                        cfg["memory"].get("max_groups", 128).asInt());
  LOG(INFO) << "Memory management unit created";

  // Initialize tracker
  auto tracker_cfg = cfg["tracker_cfg"].isString()
                         ? LoadJson(cfg["tracker_cfg"].asString())
                         : cfg["tracker_cfg"];
  Tracker::Create(tracker_cfg);
  LOG(INFO) << "Tracker created";

  CheckMemoryPools(cfg, tracker_cfg);

  // Initialize the visibility graph
  Graph::Create();
  LOG(INFO) << "Visibility graph created";

  // Initialize the Mapper
#ifdef USE_MAPPER
  auto mapper_cfg = cfg["mapper_cfg"].isString()
                        ? LoadJson(cfg["mapper_cfg"].asString())
                        : cfg["mapper_cfg"];
  Mapper::Create(mapper_cfg);
  LOG(INFO) << "Mapper created";
#endif

#ifdef USE_G2O
  // Initialize the optimizer
  Optimizer::Create(cfg["optimizer"]);
  LOG(INFO) << "Optimizer created";
#endif

  // Initialize the estimator
  Estimator::Create(cfg);
  LOG(INFO) << "Estimator created";

  // Sanity check -- if we are extracting loop closures, then make sure that
  // the Tracker is extracting descriptors
#ifdef USE_MAPPER
  if (Mapper::instance()->UseLoopClosure() &&
      !Tracker::instance()->IsExtractingDescriptors()) {
    LOG(FATAL) << "Loop closure requires descriptor extraction. Go edit the .cfg file";
  }
#endif

  system_created = true;

  return Estimator::instance();
}


EstimatorPtr CreateSystemTrackerOnly(const Json::Value &cfg) {
  static bool system_created{false};

  if (system_created) {
    return Estimator::instance();
  }

  // Initialize paramter server
  ParameterServer::Create(cfg);
  LOG(INFO) << "Parameter server created";

  // Load camera parameters
  auto cam_cfg = cfg["camera_cfg"].isString()
                     ? LoadJson(cfg["camera_cfg"].asString())
                     : cfg["camera_cfg"];
  Camera::Create(cam_cfg);
  LOG(INFO) << "Camera created";

  // // Initialize memory manager
  MemoryManager::Create(cfg["memory"].get("max_features", 256).asInt(),
                        cfg["memory"].get("max_groups", 128).asInt());
  LOG(INFO) << "Memory management unit created";

  // Initialize tracker
  auto tracker_cfg = cfg["tracker_cfg"].isString()
                         ? LoadJson(cfg["tracker_cfg"].asString())
                         : cfg["tracker_cfg"];
  Tracker::Create(tracker_cfg);
  LOG(INFO) << "Tracker created";

  CheckMemoryPools(cfg, tracker_cfg);

  // Initialize the estimator
  Estimator::Create(cfg);
  LOG(INFO) << "Estimator created";

  system_created = true;

  return Estimator::instance();
}


} // namespace xivo
