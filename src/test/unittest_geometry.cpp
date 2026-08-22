// Regression tests for `PointsAreCollinear`, whose only caller
// (`Graph::FindNewGaugeFeatures`) decides which features fix the gauge of a
// group. The original implementation compared an *area* against a fixed
// threshold, read `pts[1]` unguarded, and gave a verdict that depended on which
// point happened to land at index 0 -- and the caller builds the vector by
// iterating an `unordered_set<FeaturePtr>`, i.e. in heap-address order. That
// last property made runs of two different binaries on identical input diverge.
#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "geometry.h"

using namespace xivo;

namespace {

constexpr number_t kThresh = 1e-3;

std::vector<Vec3> Collinear() {
  // Exactly on a line, in an arbitrary direction, not through the origin.
  Vec3 p0{0.3, -1.2, 4.0};
  Vec3 d{0.5, 0.2, -0.1};
  return {p0, p0 + 1.7 * d, p0 + 4.1 * d, p0 - 2.3 * d};
}

std::vector<Vec3> Spread() {
  return {Vec3{0.3, -1.2, 4.0}, Vec3{0.8, -1.0, 3.9}, Vec3{0.1, -0.4, 4.2},
          Vec3{0.6, -0.9, 3.5}};
}

} // namespace

TEST(Geometry, CollinearDetectsALine) {
  EXPECT_TRUE(PointsAreCollinear(Collinear(), kThresh));
}

TEST(Geometry, CollinearRejectsASpreadSet) {
  EXPECT_FALSE(PointsAreCollinear(Spread(), kThresh));
}

TEST(Geometry, CollinearHandlesDegenerateSizes) {
  // Used to read pts[1] out of bounds. Fewer than three points cannot span a
  // plane, so they are collinear by definition.
  EXPECT_TRUE(PointsAreCollinear({}, kThresh));
  EXPECT_TRUE(PointsAreCollinear({Vec3{1, 2, 3}}, kThresh));
  EXPECT_TRUE(PointsAreCollinear({Vec3{1, 2, 3}, Vec3{4, 5, 6}}, kThresh));
}

TEST(Geometry, CollinearIsIndependentOfPointOrder) {
  // The caller's input order is the iteration order of an unordered_set of
  // pointers, so a permutation-dependent verdict is a reproducibility bug.
  // Both sets are checked: the old code was order-dependent in both directions
  // (a duplicated pts[0]/pts[1] pair made a spread set look collinear, and
  // starting from an outlier made a line look spread).
  for (const auto &base : {Collinear(), Spread()}) {
    std::vector<Vec3> pts = base;
    const bool expected = PointsAreCollinear(pts, kThresh);
    // All 24 permutations of the four points.
    std::vector<int> idx{0, 1, 2, 3};
    std::sort(idx.begin(), idx.end());
    int n = 0;
    do {
      std::vector<Vec3> permuted;
      for (int i : idx) {
        permuted.push_back(base[i]);
      }
      EXPECT_EQ(PointsAreCollinear(permuted, kThresh), expected)
          << "permutation " << n << " disagrees";
      ++n;
    } while (std::next_permutation(idx.begin(), idx.end()));
    EXPECT_EQ(n, 24);
  }
}

TEST(Geometry, CollinearIsIndependentOfScale) {
  // The old test quantity was |(p1-p0) x (pi-p0)|, an area: it grows with the
  // square of the distance between the points, so the same geometry gave
  // different answers at different ranges. A gauge-feature triple 10 m from the
  // camera is no more or less collinear than the same triple at 1 m.
  const std::vector<Vec3> spread = Spread();
  const std::vector<Vec3> line = Collinear();
  for (number_t s : {0.01, 0.1, 1.0, 10.0, 100.0}) {
    std::vector<Vec3> scaled_spread, scaled_line;
    for (const auto &p : spread) {
      scaled_spread.push_back(s * p);
    }
    for (const auto &p : line) {
      scaled_line.push_back(s * p);
    }
    EXPECT_FALSE(PointsAreCollinear(scaled_spread, kThresh)) << "scale " << s;
    EXPECT_TRUE(PointsAreCollinear(scaled_line, kThresh)) << "scale " << s;
  }
}

TEST(Geometry, CollinearIsNotFooledByACloseLeadingPair) {
  // v1 = pts[1] - pts[0] was the reference direction. Make it tiny: the cross
  // products |v1 x vi| all collapse below any fixed threshold even though the
  // points obviously span a plane. The old code called this collinear, which
  // means the retry-and-shuffle loop in FindNewGaugeFeatures kept rejecting
  // perfectly good gauge triples.
  std::vector<Vec3> pts = {Vec3{0.30000, -1.20000, 4.0},
                           Vec3{0.30001, -1.20000, 4.0},
                           Vec3{0.10000, -0.40000, 4.2},
                           Vec3{0.60000, -0.90000, 3.5}};
  EXPECT_FALSE(PointsAreCollinear(pts, kThresh));
}

TEST(Geometry, CollinearHandlesCoincidentPoints) {
  std::vector<Vec3> pts(4, Vec3{0.3, -1.2, 4.0});
  EXPECT_TRUE(PointsAreCollinear(pts, kThresh));
}
