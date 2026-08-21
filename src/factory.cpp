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

  // Initialize the estimator
  Estimator::Create(cfg);
  LOG(INFO) << "Estimator created";

  system_created = true;

  return Estimator::instance();
}

 
} // namespace xivo
