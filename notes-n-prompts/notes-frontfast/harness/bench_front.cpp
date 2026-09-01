// Microbenchmark for the per-frame image work the position branch turned on, and
// for the cheaper substitutes considered in its place.
//
// Stages timed, all on real TUM-VI 512x512 16-bit grayscale PNGs decoded with
// IMREAD_GRAYSCALE (which is what pybind11/pyxivo.cpp does after the efficiency
// branch's M1):
//
//   decode                cv::imdecode, the floor both systems pay
//   clahe                 cv::createCLAHE(10, 8x8)->apply   -- what the config does now
//   equalizeHist          global equalization, the cheap alternative
//   gainmap               a precomputed per-pixel fixed-point multiply (one pass)
//   gainmap_lut           a precomputed per-tile 256-entry LUT gather
//   pyramid L4 / L5       cv::buildOpticalFlowPyramid, win 15
//   klt                   cv::calcOpticalFlowPyrLK, 180 pts, win 15, L5, 30 iters
//   klt L4 / iters 15     the same at the cheaper settings
//   fast                  cv::FastFeatureDetector(20, nms) + the response sort
//   cornerSubPix          win 5, 20 iters, on 45 and on 180 points
//
// Build:
//   g++ -O2 -march=native -std=c++17 -I<opencv_install>/include/opencv4 \
//       bench_front.cpp -L<opencv_install>/lib -lopencv_core -lopencv_imgproc \
//       -lopencv_imgcodecs -lopencv_video -lopencv_features2d -o bench_front
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include "opencv2/opencv.hpp"

using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// --- the two candidate replacements for CLAHE --------------------------------

// A static per-pixel gain, in Q8 fixed point, applied in one pass. This is the
// "the vignette is a property of the lens, not the scene" idea: estimate the
// radial falloff once, then multiply.
static void ApplyGainQ8(const cv::Mat &src, const cv::Mat &gainQ8, cv::Mat &dst) {
  dst.create(src.size(), CV_8UC1);
  const int n = src.cols;
  for (int r = 0; r < src.rows; ++r) {
    const uint8_t *s = src.ptr<uint8_t>(r);
    const uint16_t *g = gainQ8.ptr<uint16_t>(r);
    uint8_t *d = dst.ptr<uint8_t>(r);
    for (int c = 0; c < n; ++c) {
      int v = (int(s[c]) * int(g[c])) >> 8;
      d[c] = v > 255 ? 255 : uint8_t(v);
    }
  }
}

// Per-tile LUT gather: G x G tiles, each with its own 256-entry table, no
// interpolation between tiles. This is the shape of CLAHE's *apply* step with
// the histogram/clip/CDF work hoisted out of the frame.
static void ApplyTileLUT(const cv::Mat &src, const cv::Mat &lut /*G*G x 256*/,
                         int G, cv::Mat &dst) {
  dst.create(src.size(), CV_8UC1);
  const int th = (src.rows + G - 1) / G, tw = (src.cols + G - 1) / G;
  for (int r = 0; r < src.rows; ++r) {
    const uint8_t *s = src.ptr<uint8_t>(r);
    uint8_t *d = dst.ptr<uint8_t>(r);
    const int ty = std::min(r / th, G - 1);
    for (int c = 0; c < src.cols; ++c) {
      const int tx = std::min(c / tw, G - 1);
      d[c] = lut.ptr<uint8_t>(ty * G + tx)[s[c]];
    }
  }
}

int main(int argc, char **argv) {
  cv::setNumThreads(1);
  std::string dir = argv[1];
  int n = argc > 2 ? atoi(argv[2]) : 200;
  std::vector<cv::String> files;
  cv::glob(dir + "/*.png", files, false);
  std::sort(files.begin(), files.end());
  if ((int)files.size() > n) files.resize(n);
  const int N = (int)files.size();
  printf("%d images from %s\n", N, dir.c_str());

  // Decode once, keep them: every later stage is timed on in-memory images.
  std::vector<cv::Mat> imgs(N);
  double t_decode = 0;
  for (int i = 0; i < N; ++i) {
    std::vector<uchar> buf;
    FILE *f = fopen(files[i].c_str(), "rb");
    fseek(f, 0, SEEK_END);
    buf.resize(ftell(f));
    fseek(f, 0, SEEK_SET);
    if (fread(buf.data(), 1, buf.size(), f) != buf.size()) return 2;
    fclose(f);
    auto a = clk::now();
    imgs[i] = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
    t_decode += ms(a, clk::now());
  }
  printf("%-24s %8.3f ms/img\n", "decode(GRAYSCALE)", t_decode / N);
  const int rows = imgs[0].rows, cols = imgs[0].cols;

  // ---- CLAHE and the alternatives ----
  auto clahe = cv::createCLAHE(10.0, cv::Size(8, 8));
  std::vector<cv::Mat> eq(N);
  {
    cv::Mat dst;
    double t = 0;
    for (int i = 0; i < N; ++i) {
      auto a = clk::now();
      clahe->apply(imgs[i], dst);
      t += ms(a, clk::now());
      eq[i] = dst.clone();
    }
    printf("%-24s %8.3f ms/img\n", "CLAHE(10,8x8)", t / N);
  }
  {
    cv::Mat dst;
    double t = 0;
    for (int i = 0; i < N; ++i) {
      auto a = clk::now();
      cv::equalizeHist(imgs[i], dst);
      t += ms(a, clk::now());
    }
    printf("%-24s %8.3f ms/img\n", "equalizeHist", t / N);
  }
  // Build a radial gain map from the mean image of this batch: gain(r) so that
  // the mean intensity is flat in radius. Exactly the offline estimate the
  // runtime would load.
  cv::Mat gainQ8(rows, cols, CV_16UC1);
  {
    cv::Mat acc(rows, cols, CV_32FC1, 0.0f);
    for (int i = 0; i < N; ++i) {
      cv::Mat f;
      imgs[i].convertTo(f, CV_32F);
      acc += f;
    }
    acc /= float(N);
    const double cy = rows * 0.5, cx = cols * 0.5;
    const int NB = 64;
    std::vector<double> sum(NB, 0), cnt(NB, 0);
    const double rmax = std::sqrt(cx * cx + cy * cy);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) {
        double rr = std::hypot(r - cy, c - cx);
        int b = std::min(NB - 1, int(NB * rr / rmax));
        sum[b] += acc.at<float>(r, c);
        cnt[b] += 1;
      }
    double ref = sum[0] / std::max(1.0, cnt[0]);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) {
        double rr = std::hypot(r - cy, c - cx);
        int b = std::min(NB - 1, int(NB * rr / rmax));
        double m = sum[b] / std::max(1.0, cnt[b]);
        double g = m > 1.0 ? ref / m : 1.0;
        gainQ8.at<uint16_t>(r, c) = uint16_t(std::min(4095.0, g * 256.0));
      }
  }
  {
    cv::Mat dst;
    double t = 0;
    for (int i = 0; i < N; ++i) {
      auto a = clk::now();
      ApplyGainQ8(imgs[i], gainQ8, dst);
      t += ms(a, clk::now());
    }
    printf("%-24s %8.3f ms/img\n", "gainmap Q8 (1 pass)", t / N);
  }
  {
    cv::Mat lut(64, 256, CV_8UC1);
    cv::randu(lut, 0, 255);
    cv::Mat dst;
    double t = 0;
    for (int i = 0; i < N; ++i) {
      auto a = clk::now();
      ApplyTileLUT(imgs[i], lut, 8, dst);
      t += ms(a, clk::now());
    }
    printf("%-24s %8.3f ms/img\n", "tile-LUT 8x8 gather", t / N);
  }
  {
    // CLAHE on a half-resolution image, LUTs implicitly upsampled by applying
    // the resulting mapping to the full image is not expressible with OpenCV's
    // CLAHE, so just time the downsampled apply + the resize, as a lower bound.
    cv::Mat small, dst;
    double t = 0;
    for (int i = 0; i < N; ++i) {
      auto a = clk::now();
      cv::resize(imgs[i], small, cv::Size(cols / 2, rows / 2), 0, 0, cv::INTER_AREA);
      clahe->apply(small, dst);
      t += ms(a, clk::now());
    }
    printf("%-24s %8.3f ms/img\n", "CLAHE on half res", t / N);
  }

  // ---- pyramids ----
  for (int lvl : {3, 4, 5}) {
    std::vector<cv::Mat> pyr;
    double t = 0;
    for (int i = 0; i < N; ++i) {
      auto a = clk::now();
      cv::buildOpticalFlowPyramid(eq[i], pyr, cv::Size(15, 15), lvl, true,
                                  cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT,
                                  false);
      t += ms(a, clk::now());
    }
    printf("%-24s %8.3f ms/img  (L%d)\n", "pyramid win15", t / N, lvl);
  }

  // ---- FAST + sort ----
  auto fast = cv::FastFeatureDetector::create(20, true);
  {
    cv::Mat mask(rows, cols, CV_8UC1, cv::Scalar(255));
    for (int pass = 0; pass < 2; ++pass) {
      const std::vector<cv::Mat> &src = pass ? eq : imgs;
      double t = 0, tsort = 0;
      long total = 0;
      std::vector<cv::KeyPoint> kps;
      for (int i = 0; i < N; ++i) {
        auto a = clk::now();
        fast->detect(src[i], kps, mask);
        auto b = clk::now();
        std::sort(kps.begin(), kps.end(),
                  [](const cv::KeyPoint &p, const cv::KeyPoint &q) {
                    return p.response > q.response;
                  });
        auto c = clk::now();
        t += ms(a, b);
        tsort += ms(b, c);
        total += (long)kps.size();
      }
      printf("%-24s %8.3f ms/img  sort %6.3f  kps/img %.0f  (%s)\n",
             "FAST(20)+sort", t / N, tsort / N, double(total) / N,
             pass ? "CLAHE" : "raw");
    }
  }

  // ---- KLT ----
  {
    std::vector<cv::Mat> p0, p1;
    cv::buildOpticalFlowPyramid(eq[0], p0, cv::Size(15, 15), 5, true,
                                cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
    std::vector<cv::Point2f> pts0;
    cv::RNG rng(7);
    for (int k = 0; k < 180; ++k)
      pts0.emplace_back(rng.uniform(24, cols - 24), rng.uniform(24, rows - 24));
    for (auto cfgv : std::vector<std::pair<int, int>>{{5, 30}, {5, 15}, {4, 30}, {3, 30}}) {
      const int L = cfgv.first, it = cfgv.second;
      double t = 0;
      std::vector<cv::Point2f> a0 = pts0, a1;
      std::vector<uint8_t> st;
      cv::TermCriteria crit(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, it, 0.01);
      for (int i = 1; i < N; ++i) {
        cv::buildOpticalFlowPyramid(eq[i - 1], p0, cv::Size(15, 15), L, true,
                                    cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
        cv::buildOpticalFlowPyramid(eq[i], p1, cv::Size(15, 15), L, true,
                                    cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
        a1 = a0;
        auto a = clk::now();
        cv::calcOpticalFlowPyrLK(p0, p1, a0, a1, st, cv::noArray(),
                                 cv::Size(15, 15), L, crit,
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        t += ms(a, clk::now());
      }
      printf("%-24s %8.3f ms/img  (L%d, %d iters, 180 pts)\n", "KLT", t / (N - 1), L, it);
    }
  }

  // ---- cornerSubPix ----
  for (int np : {45, 180}) {
    for (int win : {5, 3}) {
      for (int it : {20, 5}) {
        std::vector<cv::Point2f> pts;
        cv::RNG rng(3);
        for (int k = 0; k < np; ++k)
          pts.emplace_back(rng.uniform(24, cols - 24), rng.uniform(24, rows - 24));
        double t = 0;
        for (int i = 0; i < N; ++i) {
          std::vector<cv::Point2f> p = pts;
          auto a = clk::now();
          cv::cornerSubPix(eq[i], p, cv::Size(win, win), cv::Size(-1, -1),
                           cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                            it, 0.001));
          t += ms(a, clk::now());
        }
        printf("%-24s %8.3f ms/img  (%d pts, win %d, %d iters)\n", "cornerSubPix",
               t / N, np, win, it);
      }
    }
  }
  return 0;
}
