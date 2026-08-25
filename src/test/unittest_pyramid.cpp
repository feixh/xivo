// Tests for `Tracker::BuildOwnedPyramid`, the wrapper that forces
// `cv::buildOpticalFlowPyramid`'s `tryReuseInputImage` off.
//
// The wrapper exists for one reason: M4 stopped cloning the input image, so
// `img_` is now a header onto the caller's pixels and `pyramid_` has to survive
// into the next frame. If level 0 of `pyramid_` were a *view* of `img_`, it would
// be a view of a buffer the caller is free to overwrite. OpenCV takes that view
// silently, and only for submatrix inputs with enough margin -- which is exactly
// what a camera driver handing out an ROI into a larger frame buffer produces.
// So the aliasing is (a) invisible in the common non-submatrix case and (b)
// catastrophic in the case it does fire. Both halves are pinned here.
#include <vector>

#include "gtest/gtest.h"
#include "opencv2/video/video.hpp"

#include "tracker.h"

using namespace xivo;

namespace {

constexpr int kWin = 15;
constexpr int kLevels = 4;
constexpr int kRows = 128;
constexpr int kCols = 160;

/** Deterministic, non-constant content -- a constant image would make the
 *  content comparisons below pass for the wrong reason. */
void Fill(cv::Mat &m, int salt) {
  for (int i = 0; i < m.rows; ++i) {
    for (int j = 0; j < m.cols; ++j) {
      m.at<uint8_t>(i, j) =
          static_cast<uint8_t>((i * 7 + j * 13 + i * j + salt) & 0xff);
    }
  }
}

/** An ROI inset far enough on every side that OpenCV's reuse test accepts it:
 *  it requires `winSize` of margin around the ROI within the parent buffer. */
cv::Mat SubmatrixWithMargin(cv::Mat &whole) {
  const int pad = kWin + 1;
  whole.create(kRows + 2 * pad, kCols + 2 * pad, CV_8U);
  Fill(whole, 0);
  cv::Mat sub = whole(cv::Rect(pad, pad, kCols, kRows));
  EXPECT_TRUE(sub.isSubmatrix());
  return sub;
}

bool SharesDataWith(const cv::Mat &a, const cv::Mat &b) {
  return a.data == b.data;
}

}  // namespace

/** The property the front end depends on. */
TEST(Pyramid, OwnedPyramidDoesNotAliasASubmatrixInput) {
  cv::Mat whole;
  cv::Mat sub = SubmatrixWithMargin(whole);

  std::vector<cv::Mat> pyr;
  Tracker::BuildOwnedPyramid(sub, pyr, kWin, kLevels);
  ASSERT_FALSE(pyr.empty());
  EXPECT_FALSE(SharesDataWith(pyr[0], sub));

  // The consequence, stated the way the bug would present: the pyramid keeps its
  // own copy of the pixels even after the caller's buffer is reused for the next
  // frame.
  const cv::Mat before = pyr[0].clone();
  Fill(whole, 99);
  EXPECT_EQ(0.0, cv::norm(pyr[0], before, cv::NORM_INF));
}

/** The OpenCV behaviour that makes the flag necessary. If this ever starts
 *  failing, `tryReuseInputImage` no longer aliases and the wrapper is redundant
 *  rather than wrong -- which is worth knowing either way. */
TEST(Pyramid, TheDefaultBuildDoesAliasASubmatrixInput) {
  cv::Mat whole;
  cv::Mat sub = SubmatrixWithMargin(whole);

  std::vector<cv::Mat> pyr;
  cv::buildOpticalFlowPyramid(sub, pyr, cv::Size(kWin, kWin), kLevels);
  ASSERT_FALSE(pyr.empty());
  EXPECT_TRUE(SharesDataWith(pyr[0], sub));

  const cv::Mat before = pyr[0].clone();
  Fill(whole, 99);
  EXPECT_LT(0.0, cv::norm(pyr[0], before, cv::NORM_INF));
}

/** Forcing the flag off changes ownership and nothing else: every level, and
 *  every derivative level, is identical to what the default call produces. This
 *  is what says the end-to-end trajectories must be unchanged. */
TEST(Pyramid, OwnedPyramidHasTheSameContentAsTheDefault) {
  cv::Mat whole;
  cv::Mat sub = SubmatrixWithMargin(whole);

  std::vector<cv::Mat> owned, deflt;
  Tracker::BuildOwnedPyramid(sub, owned, kWin, kLevels);
  cv::buildOpticalFlowPyramid(sub, deflt, cv::Size(kWin, kWin), kLevels);

  ASSERT_EQ(deflt.size(), owned.size());
  // withDerivatives defaults to true, so the vector interleaves images and
  // gradients: 2 entries per level. Fewer levels than asked for is normal --
  // OpenCV stops halving once a level would be smaller than the window and
  // returns the level it reached -- so this is `<=`, and the equality with
  // `deflt` above is what pins the two calls to agree.
  ASSERT_EQ(0u, owned.size() % 2);
  ASSERT_LE(owned.size(), 2u * (kLevels + 1));
  ASSERT_LT(0u, owned.size());
  for (size_t i = 0; i < owned.size(); ++i) {
    ASSERT_EQ(deflt[i].size(), owned[i].size()) << "level entry " << i;
    ASSERT_EQ(deflt[i].type(), owned[i].type()) << "level entry " << i;
    EXPECT_EQ(0.0, cv::norm(owned[i], deflt[i], cv::NORM_INF))
        << "level entry " << i;
  }
}

/** A full frame is not a submatrix, so OpenCV copies level 0 regardless and the
 *  flag is a no-op. Recorded because it is why the aliasing never showed up on
 *  TUM-VI, where the images come from `cv::imread`. */
TEST(Pyramid, ANonSubmatrixInputIsCopiedEitherWay) {
  cv::Mat img(kRows, kCols, CV_8U);
  Fill(img, 5);
  ASSERT_FALSE(img.isSubmatrix());

  std::vector<cv::Mat> owned, deflt;
  Tracker::BuildOwnedPyramid(img, owned, kWin, kLevels);
  cv::buildOpticalFlowPyramid(img, deflt, cv::Size(kWin, kWin), kLevels);

  EXPECT_FALSE(SharesDataWith(owned[0], img));
  EXPECT_FALSE(SharesDataWith(deflt[0], img));
  EXPECT_EQ(0.0, cv::norm(owned[0], deflt[0], cv::NORM_INF));
}

/** The window and level count are arguments now, not members, because the stereo
 *  matcher may use its own. A wrapper that ignored them and used the KLT defaults
 *  would pass every test above, so pin them here.
 *
 *  The level count is visible in the vector's length. The window is not visible
 *  in any pixel -- each level's ROI is the same size and holds the same values
 *  whatever the padding, since BORDER_REFLECT_101 gives the same value at a given
 *  distance from the edge regardless of how far it extends. What it sets is the
 *  *margin* around each level inside its parent buffer, which is what
 *  `calcOpticalFlowPyrLK` reads when a window straddles the image boundary. So
 *  the witness is `locateROI`, not the content. */
TEST(Pyramid, TheWindowAndLevelArgumentsAreHonoured) {
  cv::Mat img(kRows, kCols, CV_8U);
  Fill(img, 5);

  std::vector<cv::Mat> two_levels;
  Tracker::BuildOwnedPyramid(img, two_levels, 9, 2);
  ASSERT_EQ(2 * (2 + 1), static_cast<int>(two_levels.size()));

  std::vector<cv::Mat> reference;
  cv::buildOpticalFlowPyramid(img, reference, cv::Size(9, 9), 2);
  ASSERT_EQ(reference.size(), two_levels.size());
  for (size_t i = 0; i < two_levels.size(); ++i) {
    EXPECT_EQ(0.0, cv::norm(two_levels[i], reference[i], cv::NORM_INF))
        << "level entry " << i;
  }

  for (int win : {9, 21}) {
    std::vector<cv::Mat> pyr;
    Tracker::BuildOwnedPyramid(img, pyr, win, 2);
    for (int level = 0; level <= 2; ++level) {
      cv::Size whole;
      cv::Point ofs;
      pyr[level * 2].locateROI(whole, ofs);
      EXPECT_EQ(win, ofs.x) << "win " << win << " level " << level;
      EXPECT_EQ(win, ofs.y) << "win " << win << " level " << level;
    }
  }
}
