// Tests for the stereo data path (M2): timestamp pairing in DataLoader and the
// partner-directory derivation.
//
// These run against the real TUM-VI room1 sequence, which is the data the system
// is evaluated on, so they also assert the dataset itself is intact. The path is
// relative to the repository root (where the other tests are run from) and the
// tests skip themselves if the dataset is not present, so the suite still passes
// on a checkout without data.
#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <set>

#include "core.h"
#include "loader.h"
#include "message_types.h"

using namespace xivo;

namespace {

const char *kRoot = "../data/tumvi";

std::string SeqDir(const std::string &seq) {
  return std::string(kRoot) + "/dataset-" + seq + "_512_16/mav0";
}

bool HaveDataset(const std::string &seq) {
  std::ifstream f(SeqDir(seq) + "/cam0/data.csv");
  return f.good();
}

// Timestamps listed in an ASL data.csv.
std::vector<int64_t> CsvTimestamps(const std::string &dir) {
  std::vector<int64_t> ts;
  std::ifstream is(dir + "/data.csv");
  std::string line;
  std::getline(is, line); // header
  while (is >> line) {
    if (line.front() == '#') continue;
    ts.push_back(std::stoll(line.substr(0, line.find(','))));
  }
  return ts;
}

} // namespace

TEST(StereoPairDir, SwapsTheCameraComponent) {
  EXPECT_EQ(StereoPairDir("/data/room1/mav0/cam0/", 0, 1),
            "/data/room1/mav0/cam1/");
  EXPECT_EQ(StereoPairDir("/data/room1/mav0/cam1/", 1, 0),
            "/data/room1/mav0/cam0/");

  // Only the last occurrence is swapped: a dataset root may legitimately
  // contain "cam0" in its own name, and rewriting that would point at a
  // directory that does not exist.
  EXPECT_EQ(StereoPairDir("/mnt/cam0_dumps/room1/mav0/cam0/", 0, 1),
            "/mnt/cam0_dumps/room1/mav0/cam1/");
}

TEST(StereoLoader, PairsAllRoom1FramesAndInterleavesIMU) {
  if (!HaveDataset("room1")) {
    GTEST_SKIP() << "TUM-VI room1 not present under " << kRoot;
  }
  const std::string cam0 = SeqDir("room1") + "/cam0";
  const std::string cam1 = SeqDir("room1") + "/cam1";
  const std::string imu = SeqDir("room1") + "/imu0";

  auto left_ts = CsvTimestamps(cam0);
  auto right_ts = CsvTimestamps(cam1);
  auto imu_ts = CsvTimestamps(imu);
  // The premise of the whole stereo design: this rig is hardware-triggered, so
  // every left frame has an exactly-equal right frame. If this ever fails, the
  // loader's zero-tolerance pairing is the wrong policy and interpolation would
  // be needed.
  ASSERT_EQ(left_ts.size(), right_ts.size());
  EXPECT_EQ(left_ts, right_ts);
  ASSERT_EQ(left_ts.size(), 2821u);

  DataLoader loader{cam0, cam1, imu};

  int n_stereo = 0, n_imu = 0, n_mono = 0;
  std::vector<int64_t> seen_ts;
  for (int i = 0; i < loader.size(); ++i) {
    auto *m = loader.Get(i);
    if (auto *s = dynamic_cast<msg::StereoImage *>(m)) {
      ++n_stereo;
      seen_ts.push_back(s->ts_.count());
      // Both paths must exist and must differ: reading the same file twice
      // would give zero disparity everywhere and quietly disable stereo.
      EXPECT_NE(s->image_path_, s->image_path_r_);
      EXPECT_NE(s->image_path_.find("cam0"), std::string::npos);
      EXPECT_NE(s->image_path_r_.find("cam1"), std::string::npos);
      // The two frames of a pair are named after the same timestamp.
      auto base = [](const std::string &p) {
        return p.substr(p.rfind('/') + 1);
      };
      EXPECT_EQ(base(s->image_path_), base(s->image_path_r_));
    } else if (dynamic_cast<msg::IMU *>(m)) {
      ++n_imu;
    } else if (dynamic_cast<msg::Image *>(m)) {
      // A stereo loader must never emit a bare monocular frame; if it did,
      // the estimator would silently process half the pairs as mono.
      ++n_mono;
    }
  }

  EXPECT_EQ(n_mono, 0);
  EXPECT_EQ(n_stereo, 2821);
  EXPECT_EQ(n_imu, static_cast<int>(imu_ts.size()));
  EXPECT_EQ(loader.size(), n_stereo + n_imu);

  // Entries come out in ascending time, since the estimator's buffer assumes it.
  EXPECT_TRUE(std::is_sorted(seen_ts.begin(), seen_ts.end()));
  std::vector<int64_t> sorted_left = left_ts;
  std::sort(sorted_left.begin(), sorted_left.end());
  EXPECT_EQ(seen_ts, sorted_left);
}

TEST(StereoLoader, MismatchedDirectoriesYieldNoPairs) {
  if (!HaveDataset("room1") || !HaveDataset("room2")) {
    GTEST_SKIP() << "TUM-VI room1/room2 not present under " << kRoot;
  }
  // room1's cam0 against room2's cam1: two different recordings, so no
  // timestamp should match. The loader treats zero pairs as fatal, so assert on
  // the timestamp sets directly rather than provoking the LOG(FATAL) -- the
  // point being that the pairing is genuinely driven by timestamps and would
  // not blindly zip two equal-length directories together.
  auto a = CsvTimestamps(SeqDir("room1") + "/cam0");
  auto b = CsvTimestamps(SeqDir("room2") + "/cam1");
  std::set<int64_t> sa(a.begin(), a.end());
  int overlap = 0;
  for (int64_t t : b) {
    if (sa.count(t)) ++overlap;
  }
  EXPECT_EQ(overlap, 0);
}
