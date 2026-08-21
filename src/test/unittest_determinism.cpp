// Regression tests for run-to-run determinism of feature/group selection.
//
// Background: XIVO promotes features into the EKF state by sorting candidates
// and truncating at kMaxFeature. Candidate lists reach the sort via
// MakePtrVectorUnique, which sorts by *pointer value*, so if the comparison
// function leaves ties unresolved the winner is decided by heap addresses. Ties
// are the common case rather than a rare one -- every freshly initialized
// candidate carries the same initial depth variance, so they all compare equal.
//
// The symptom was a trajectory that changed roughly 1 run in 20 on TUM-VI room3
// (ATE 0.1549 vs 0.1703) with no change to inputs, config, or seed. Disabling
// ASLR (`setarch -R`) made it disappear. See notes-stereo/m3a-determinism.md.
//
// The property these tests pin is stronger than "the comparator is consistent":
// it is that the *result of sorting is independent of the input order*. That is
// what makes the pipeline immune to the address-ordered input it is handed.
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#define private public

#include "feature.h"
#include "mm.h"
#include "options.h"
#include "param.h"

using namespace xivo;

namespace {

class DeterminismTest : public ::testing::Test {
protected:
  void SetUp() override {
    MemoryManager::Create(256, 128);
    auto cfg = LoadJson("src/test/camera_configs.json");
    Camera::Create(cfg["perfect_pinhole"]);

    // CandidateComparison reads its score type from the ParameterServer.
    Json::Value params;
    params["comparison_score_type"] = "DepthUncertainty";
    ParameterServer::Create(params);
  }

  /** A set of features that are deliberately *tied* under the comparator:
   * same status and the same initial covariance, which is exactly the state
   * every newly initialized candidate is in. */
  std::vector<FeaturePtr> MakeTiedFeatures(int n) {
    std::vector<FeaturePtr> fs;
    for (int i = 0; i < n; ++i) {
      // Different pixel locations, but Initialize is given identical depth and
      // uncertainty, so P_(2,2) -- the score -- is identical across all of them.
      auto f = Feature::Create(20.0 + 3.0 * i, 30.0 + 2.0 * i);
      f->Initialize(2.5, Vec3{1.0, 1.0, 1.0});
      f->SetStatus(FeatureStatus::READY);
      fs.push_back(f);
    }
    return fs;
  }
};

TEST_F(DeterminismTest, ComparatorIsAntisymmetricOnTiedFeatures) {
  auto fs = MakeTiedFeatures(8);
  for (size_t i = 0; i < fs.size(); ++i) {
    for (size_t j = 0; j < fs.size(); ++j) {
      bool ij = Criteria::CandidateComparison(fs[i], fs[j]);
      bool ji = Criteria::CandidateComparison(fs[j], fs[i]);
      if (i == j) {
        // Irreflexive: a strict ordering never places an element before itself.
        EXPECT_FALSE(ij) << "feature " << fs[i]->id() << " precedes itself";
      } else {
        // Exactly one direction holds. Before the fix both were false for tied
        // features, which is a valid strict weak ordering but leaves the sort
        // free to return either order -- and it did, based on address.
        EXPECT_TRUE(ij != ji)
            << "features " << fs[i]->id() << " and " << fs[j]->id()
            << " are mutually incomparable: ties are unresolved";
      }
    }
  }
}

TEST_F(DeterminismTest, TiedFeaturesHaveIdenticalScores) {
  // Guards the premise of the test above: if Initialize ever stopped producing
  // identical covariances the antisymmetry test would pass trivially and stop
  // testing anything.
  auto fs = MakeTiedFeatures(4);
  for (size_t i = 1; i < fs.size(); ++i) {
    EXPECT_DOUBLE_EQ((fs[0]->P())(2, 2), (fs[i]->P())(2, 2));
    EXPECT_EQ(fs[0]->status(), fs[i]->status());
  }
}

TEST_F(DeterminismTest, SortResultDoesNotDependOnInputOrder) {
  auto fs = MakeTiedFeatures(12);

  // The reference order: sort once, record the resulting ids.
  std::vector<FeaturePtr> reference = fs;
  std::sort(reference.begin(), reference.end(), Criteria::CandidateComparison);
  std::vector<int> expected;
  for (auto f : reference) {
    expected.push_back(f->id());
  }

  // Now sort many different permutations of the same set. A total order gives
  // the same answer every time; a partial one lets the input order leak through.
  std::mt19937 rng(0);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<FeaturePtr> shuffled = fs;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    std::sort(shuffled.begin(), shuffled.end(), Criteria::CandidateComparison);

    std::vector<int> got;
    for (auto f : shuffled) {
      got.push_back(f->id());
    }
    ASSERT_EQ(expected, got) << "sort result depends on input order (trial "
                             << trial << ")";
  }
}

TEST_F(DeterminismTest, TruncationKeepsTheSameFeaturesRegardlessOfInputOrder) {
  // The selection path does not just sort, it truncates at kMaxFeature. This is
  // where a non-total order actually changes behavior: a different tied feature
  // gets promoted into the EKF state.
  auto fs = MakeTiedFeatures(12);
  const int kKeep = 5;

  std::mt19937 rng(7);
  std::vector<int> first;
  for (int trial = 0; trial < 100; ++trial) {
    std::vector<FeaturePtr> shuffled = fs;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    std::sort(shuffled.begin(), shuffled.end(), Criteria::CandidateComparison);

    std::vector<int> kept;
    for (int i = 0; i < kKeep; ++i) {
      kept.push_back(shuffled[i]->id());
    }
    if (trial == 0) {
      first = kept;
    } else {
      ASSERT_EQ(first, kept) << "the promoted subset depends on input order";
    }
  }
}

TEST_F(DeterminismTest, ScoreOutranksIdTieBreak) {
  // The id tie-break must be the *last* resort: a genuinely better-scoring
  // feature has to win regardless of its id. Otherwise the fix for
  // nondeterminism would have quietly replaced the selection policy with
  // "oldest first".
  auto fs = MakeTiedFeatures(2);
  ASSERT_LT(fs[0]->id(), fs[1]->id());

  // Make the higher-id feature strictly more certain in depth (a better score).
  fs[1]->P_(2, 2) = 0.5 * (fs[0]->P())(2, 2);

  EXPECT_TRUE(Criteria::CandidateComparison(fs[1], fs[0]))
      << "better score did not win";
  EXPECT_FALSE(Criteria::CandidateComparison(fs[0], fs[1]));
}

} // namespace
