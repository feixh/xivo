// Startup validation of the memory-manager pool sizes, M6.
//
// Run from the repository root: the config path below is relative to it.
//
// The pools in `mm.h` are pre-allocated and fixed, and `GetItem` calls
// `LOG(FATAL)` once every slot is active. Before `CheckMemoryPools` existed, an
// undersized `memory.max_features` surfaced as an abort inside
// `Tracker::UpdateLK` at whatever point in the sequence the pool happened to
// fill -- on TUM-VI room1, ten minutes in, with a trajectory file already half
// written and nothing in the message about which config number was at fault.
// These tests pin the up-front diagnosis.
//
// They are death tests because the failure mode being checked *is* the abort:
// asserting on a return value would test a different function. gtest runs each
// death-test body in a forked child, so the `CreateSystem` singleton is fresh in
// every one of them, and the parent must therefore never build a system itself.
#include <gtest/gtest.h>

#include "estimator.h"
#include "utils.h"

using namespace xivo;

namespace {

const char *kCfg = "cfg/tumvi_stereo.json";

Json::Value LoadCfg() {
  auto cfg = LoadJson(kCfg);
  EXPECT_FALSE(cfg.isNull())
      << "could not load " << kCfg << "; run tests from the repo root";
  return cfg;
}

} // namespace

TEST(MemoryPools, ShippedConfigPasses) {
  // Guards against the check itself being wrong: the config this branch ships
  // must survive it, or every run fails at startup.
  auto cfg = LoadCfg();
  ASSERT_GE(cfg["memory"]["max_features"].asInt(),
            2 * cfg["tracker_cfg"]["num_features_max"].asInt())
      << "shipped pool is under the 2x margin CheckMemoryPools advises";
  ASSERT_GE(cfg["memory"]["max_features"].asInt(), kMaxFeature);
  ASSERT_GE(cfg["memory"]["max_groups"].asInt(), kMaxGroup);
}

TEST(MemoryPools, FeaturePoolBelowTrackerCapIsFatal) {
  auto cfg = LoadCfg();
  // One slot short of what the tracker alone will ask for.
  const int tracker_max = cfg["tracker_cfg"]["num_features_max"].asInt();
  cfg["memory"]["max_features"] = tracker_max - 1;
  EXPECT_DEATH(CreateSystem(cfg), "below tracker_cfg.num_features_max");
}

TEST(MemoryPools, FeaturePoolBelowEkfCapacityIsFatal) {
  auto cfg = LoadCfg();
  // Small enough to trip the EKF-capacity check, and small enough that the
  // tracker check would also fire -- so this pins the *order*: the tracker cap
  // is reported first because it is the number the user is more likely to have
  // just edited. Drop the tracker cap out of the way to reach the EKF check.
  cfg["tracker_cfg"]["num_features_max"] = 1;
  cfg["tracker_cfg"]["num_features_min"] = 1;
  cfg["memory"]["max_features"] = kMaxFeature - 1;
  EXPECT_DEATH(CreateSystem(cfg), "below the EKF's feature capacity");
}

TEST(MemoryPools, GroupPoolBelowEkfCapacityIsFatal) {
  auto cfg = LoadCfg();
  cfg["memory"]["max_groups"] = kMaxGroup - 1;
  EXPECT_DEATH(CreateSystem(cfg), "below the EKF's group capacity");
}

TEST(MemoryPools, ConfigDeclaringMoreCapacityThanTheBuildHasIsFatal) {
  // The silent-underperformance case: a config tuned for 90 in-state features run
  // against a 30-feature binary tracks all 180 features and quietly discards most
  // of them, reporting nothing but worse ATE.
  auto cfg = LoadCfg();
  cfg["require_ekf_max_features"] = kMaxFeature + 1;
  EXPECT_DEATH(CreateSystem(cfg), "requires EKF_MAX_FEATURES >=");

  auto cfg2 = LoadCfg();
  cfg2["require_ekf_max_groups"] = kMaxGroup + 1;
  EXPECT_DEATH(CreateSystem(cfg2), "requires EKF_MAX_GROUPS >=");
}

TEST(MemoryPools, ShippedCapacityRequirementIsSatisfiedByThisBuild) {
  // Pairs with the CMakeLists defaults: if someone lowers EKF_MAX_FEATURES below
  // what cfg/tumvi_stereo.json asks for, this fails at `make test` rather than at
  // the end of a six-sequence evaluation.
  auto cfg = LoadCfg();
  EXPECT_GE(kMaxFeature, cfg.get("require_ekf_max_features", 0).asInt());
  EXPECT_GE(kMaxGroup, cfg.get("require_ekf_max_groups", 0).asInt());
}

TEST(MemoryPools, ThinMarginIsReportedButNotFatal) {
  // Between 1x and 2x the tracker cap the run usually completes, so this must
  // stay an advisory. Checked in a forked child anyway, since it still builds a
  // system: the exit status must be 0 and the message must reach *stderr*
  // (glog's default stderrthreshold is ERROR, so a LOG(WARNING) would be
  // invisible, which was the original bug in this advisory).
  auto cfg = LoadCfg();
  const int tracker_max = cfg["tracker_cfg"]["num_features_max"].asInt();
  cfg["memory"]["max_features"] = 2 * tracker_max - 1;
  EXPECT_EXIT(
      {
        CreateSystem(cfg);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "under 2x tracker_cfg.num_features_max");
}
