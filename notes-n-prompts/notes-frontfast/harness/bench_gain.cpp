// Why the first GAINMAP form lost: what a radial gain does to the image, and to
// FAST's keypoint supply, compared with CLAHE and with raw.
//
// Two forms of the correction are compared:
//   multiply  dst = clamp(src * g(r))
//   affine    dst = clamp((src - m(r)) * g(r) + ref)
// where m(r) is the measured radial mean profile, ref its maximum, and
// g(r) = ref / m(r) clamped to max_gain. FAST only ever looks at *differences*
// of intensity, so both forms scale what it sees by the same g(r); they differ
// only in how much of the range they spend, which is what decides saturation.
//
// Build: see bench_front.cpp.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "opencv2/opencv.hpp"

using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char **argv) {
  cv::setNumThreads(1);
  std::string dir = argv[1];
  int n = argc > 2 ? atoi(argv[2]) : 200;
  double max_gain = argc > 3 ? atof(argv[3]) : 4.0;
  int NB = argc > 4 ? atoi(argv[4]) : 32;
  int SM = argc > 5 ? atoi(argv[5]) : 2;
  std::vector<cv::String> files;
  cv::glob(dir + "/*.png", files, false);
  std::sort(files.begin(), files.end());
  if ((int)files.size() > n) files.resize(n);
  const int N = (int)files.size();
  std::vector<cv::Mat> imgs(N);
  for (int i = 0; i < N; ++i) imgs[i] = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
  const int rows = imgs[0].rows, cols = imgs[0].cols;
  // TUM-VI 512x512 cam0 intrinsics (dso/camchain), the principal point the
  // runtime would use.
  const double cx = 255.2372, cy = 256.4818;

  // --- radial profile from the FIRST frame only, exactly as BuildGainMap does
  const double rmax = std::hypot(std::max(cx, cols - cx), std::max(cy, rows - cy));
  std::vector<double> sum(NB, 0), cnt(NB, 0);
  const double inv = NB / rmax;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      int b = std::min(NB - 1, int(std::hypot(c + 0.5 - cx, r + 0.5 - cy) * inv));
      sum[b] += imgs[0].at<uint8_t>(r, c);
      cnt[b] += 1;
    }
  std::vector<double> mean(NB), prof(NB);
  for (int b = 0; b < NB; ++b) mean[b] = cnt[b] > 0 ? sum[b] / cnt[b] : 0;
  for (int b = 0; b < NB; ++b) {
    double a = 0, w = 0;
    for (int k = -SM; k <= SM; ++k) {
      int j = b + k;
      if (j < 0 || j >= NB || cnt[j] <= 0) continue;
      a += mean[j];
      w += 1;
    }
    prof[b] = w > 0 ? a / w : mean[b];
  }
  double ref = 0;
  for (double p : prof) ref = std::max(ref, p);
  std::vector<double> g(NB, 1.0);
  for (int b = 0; b < NB; ++b)
    g[b] = prof[b] > 1 ? std::min(max_gain, ref / prof[b]) : max_gain;
  printf("bins=%d smooth=%d max_gain=%.2f  ref=%.1f\n", NB, SM, max_gain, ref);
  printf("profile:");
  for (int b = 0; b < NB; b += NB / 8) printf(" %.1f", prof[b]);
  printf("  (last %.1f)\n", prof[NB - 1]);
  printf("gain   :");
  for (int b = 0; b < NB; b += NB / 8) printf(" %.2f", g[b]);
  printf("  (last %.2f)\n", g[NB - 1]);

  // per-pixel maps, interpolated between bin centres
  cv::Mat gp(rows, cols, CV_32FC1), mp(rows, cols, CV_32FC1);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      double t = std::hypot(c + 0.5 - cx, r + 0.5 - cy) * inv - 0.5;
      int b0 = int(std::floor(t));
      double f = t - b0;
      if (b0 < 0) { b0 = 0; f = 0; }
      if (b0 >= NB - 1) { b0 = NB - 2; f = 1; }
      gp.at<float>(r, c) = float(g[b0] * (1 - f) + g[b0 + 1] * f);
      mp.at<float>(r, c) = float(prof[b0] * (1 - f) + prof[b0 + 1] * f);
    }

  auto fast = cv::FastFeatureDetector::create(argc > 6 ? atoi(argv[6]) : 20, true);
  auto clahe = cv::createCLAHE(10.0, cv::Size(8, 8));
  cv::Mat mask(rows, cols, CV_8UC1, cv::Scalar(255));
  std::vector<cv::KeyPoint> kps;

  struct Res { long kps; double sat, resp, t; };
  auto score = [&](const char *name, int form) {
    Res R{0, 0, 0, 0};
    long npx = 0;
    cv::Mat dst(rows, cols, CV_8UC1);
    for (int i = 0; i < N; ++i) {
      auto a = clk::now();
      if (form == 0) {
        dst = imgs[i];
      } else if (form == 1) {
        clahe->apply(imgs[i], dst);
      } else {
        for (int r = 0; r < rows; ++r) {
          const uint8_t *s = imgs[i].ptr<uint8_t>(r);
          const float *gg = gp.ptr<float>(r), *mm = mp.ptr<float>(r);
          uint8_t *d = dst.ptr<uint8_t>(r);
          for (int c = 0; c < cols; ++c) {
            double v = form == 2 ? s[c] * gg[c] : (s[c] - mm[c]) * gg[c] + ref;
            d[c] = uint8_t(std::min(255.0, std::max(0.0, v)));
          }
        }
      }
      R.t += ms(a, clk::now());
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
          if (dst.at<uint8_t>(r, c) >= 255) R.sat += 1;
          npx += 1;
        }
      fast->detect(dst, kps, mask);
      R.kps += (long)kps.size();
      for (auto &k : kps) R.resp += k.response;
    }
    printf("%-12s kps/img %7.0f  mean-resp %5.1f  saturated %6.3f%%  apply %6.3f ms\n",
           name, double(R.kps) / N, R.kps ? R.resp / R.kps : 0,
           100.0 * R.sat / npx, R.t / N);
  };
  score("raw", 0);
  score("CLAHE", 1);
  score("gain-multiply", 2);
  score("gain-affine", 3);
  return 0;
}
