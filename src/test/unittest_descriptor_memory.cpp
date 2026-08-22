// Regression tests for the unbounded descriptor growth fixed in M3.
//
// Two defects multiplied each other:
//   * Track::Reset cleared the pixel history but not descriptors_, so a pooled
//     Feature slot inherited every descriptor its previous tenants ever had, and
//     the count grew linearly with run length (9,059 retained descriptors at the
//     end of TUM-VI room1).
//   * SetDescriptor stored `all_descriptors.row(i)`, a view sharing the whole
//     per-frame descriptor matrix, so each retained 32-byte row pinned 3.5-8.6 kB.
//
// Neither is visible to LeakSanitizer -- the memory is reachable from a
// singleton and is freed at exit -- so it has to be tested directly.
#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "feature.h"
#include "mm.h"

using namespace xivo;

namespace {

/** A stand-in for what BriefDescriptorExtractorImpl::compute produces: one
 *  32-byte BRIEF descriptor per detected keypoint, in a single matrix. */
cv::Mat MakeDescriptorBlock(int num_keypoints) {
  cv::Mat block(num_keypoints, 32, CV_8U);
  for (int i = 0; i < num_keypoints; ++i) {
    block.row(i).setTo(static_cast<uint8_t>(i));
  }
  return block;
}

} // namespace

class DescriptorMemory : public ::testing::Test {
protected:
  static constexpr int kMaxFeatures = 4;
  static constexpr int kMaxGroups = 2;

  void SetUp() override {
    // The MemoryManager is a process-wide singleton and cannot be re-created,
    // so every test in this binary shares one small pool. A small pool is the
    // point: slots are recycled after four features.
    MemoryManager::Create(kMaxFeatures, kMaxGroups);
  }
};

// Track::Reset is how a pooled slot is handed to its next tenant, and is the
// only chance to drop the previous tenant's descriptors.
TEST_F(DescriptorMemory, ResetDropsDescriptors) {
  cv::Mat block = MakeDescriptorBlock(8);

  Track track(1.0, 2.0);
  track.SetDescriptor(block.row(0));
  track.SetDescriptor(block.row(1));
  ASSERT_EQ(2u, track.GetAllDescriptors().size());

  track.Reset(3.0, 4.0);
  EXPECT_TRUE(track.GetAllDescriptors().empty());
  EXPECT_FALSE(track.has_descriptor());
}

// SetDescriptor has to copy: its callers all pass a row *view* of the per-frame
// matrix, and retaining the view retains the matrix.
TEST_F(DescriptorMemory, SetDescriptorDoesNotAliasTheSourceMatrix) {
  cv::Mat block = MakeDescriptorBlock(64);

  Track track(1.0, 2.0);
  track.SetDescriptor(block.row(7));

  const cv::Mat &stored = track.descriptor();
  ASSERT_EQ(32u, stored.total() * stored.elemSize());

  // Different allocation, and -- the part that actually bounds the memory --
  // not a reference to the parent matrix's buffer.
  EXPECT_NE(block.data, stored.data);
  EXPECT_NE(block.u, stored.u);

  // The values are the ones that were handed over...
  EXPECT_EQ(7, stored.at<uint8_t>(0, 0));
  // ...and they do not follow later writes to the source.
  block.row(7).setTo(99);
  EXPECT_EQ(7, stored.at<uint8_t>(0, 0));
}

// The end-to-end shape of the leak: the tracker creates a feature, gives it one
// descriptor, and eventually drops it; the slot is then handed to a new feature.
// Retention must not depend on how many times that has happened.
TEST_F(DescriptorMemory, PoolSlotRetentionIsBoundedAcrossRecycles) {
  cv::Mat block = MakeDescriptorBlock(200);

  // Many times the pool size, so every slot is recycled repeatedly.
  const int kCycles = 25 * kMaxFeatures;
  for (int i = 0; i < kCycles; ++i) {
    FeaturePtr f = Feature::Create(10.0, 20.0);
    f->SetDescriptor(block.row(i % block.rows));

    EXPECT_EQ(1u, f->GetAllDescriptors().size())
        << "slot handed out on cycle " << i << " still holds a previous "
        << "tenant's descriptors";
    EXPECT_NE(block.u, f->descriptor().u)
        << "descriptor on cycle " << i << " pins the whole source matrix";

    Feature::Deactivate(f);
  }
}
