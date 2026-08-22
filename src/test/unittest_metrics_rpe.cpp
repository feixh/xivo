// Regression tests for the out-of-range read in ComputeRPE (M4 / L3-6).
//
// The pairing loop checks `it_est < est.end()` at the top, then increments
// `it_est` inside the body and dereferences it a few lines later. When the
// matched estimate is the *last* one, that dereference is one past the end of
// the vector.
//
// The value ComputeRPE returns is the same either way (the following search loop
// is empty, so the pair is skipped), which is exactly why this went unnoticed:
// the defect is the read itself. `EndOfEstimateDoesNotReadPastTheEnd` therefore
// only *fails* under a sanitizer -- it is a heap-buffer-overflow in the ASan
// build (scripts/mem/build.sh asan) and a silent bad read otherwise. The
// companion test pins the normal path so that the added bounds check cannot
// quietly stop the pairing early.
#include <gtest/gtest.h>

#include "metrics.h"

using namespace xivo;

namespace {

constexpr uint64_t kSecond = 1000000000ULL;

msg::Pose PoseAt(double t_sec, double x) {
  SE3 g;
  g.translation() = Vec3{x, 0, 0};
  return msg::Pose{timestamp_t{static_cast<uint64_t>(t_sec * kSecond)}, g};
}

} // namespace

// The last estimate matches a ground-truth interval, so the body runs with
// `it_est` on the final element and increments it to end().
TEST(MetricsRPE, EndOfEstimateDoesNotReadPastTheEnd) {
  std::vector<msg::Pose> gt{PoseAt(0.0, 0.0), PoseAt(1.0, 1.0),
                            PoseAt(2.0, 2.0)};
  std::vector<msg::Pose> est{PoseAt(0.0, 0.0)};

  number_t rpe_pos, rpe_rot;
  std::tie(rpe_pos, rpe_rot) = ComputeRPE(est, gt, 1.0, 0.005);

  // No pair can be completed from a single estimate, so RPE is undefined.
  EXPECT_EQ(-1, rpe_pos);
  EXPECT_EQ(-1, rpe_rot);
}

// ... and the guard must not cut the pairing short when there is more estimate
// left: a trajectory compared against itself has zero relative pose error.
TEST(MetricsRPE, IdenticalTrajectoriesHaveZeroError) {
  std::vector<msg::Pose> traj;
  for (int i = 0; i <= 8; ++i) {
    traj.push_back(PoseAt(0.5 * i, 0.25 * i));
  }

  number_t rpe_pos, rpe_rot;
  std::tie(rpe_pos, rpe_rot) = ComputeRPE(traj, traj, 1.0, 0.005);

  ASSERT_NE(-1, rpe_pos) << "no pose pairs were found at all";
  EXPECT_NEAR(0.0, rpe_pos, 1e-9);
  EXPECT_NEAR(0.0, rpe_rot, 1e-9);
}
