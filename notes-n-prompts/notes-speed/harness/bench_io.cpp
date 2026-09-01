// Microbenchmark for the image front end: PNG decode, pyramid build and KLT, at
// one channel and at three. XIVO's pybind layer calls cv::imread with default
// flags, which on TUM-VI's 16-bit grayscale PNGs yields an 8UC3 image, so every
// stage downstream runs on three identical copies of the same plane.
//
// Build:
//   g++ -O2 -std=c++17 -I<opencv_install>/include/opencv4 bench_io.cpp \
//       -L<opencv_install>/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
//       -lopencv_video -lopencv_features2d -o bench_io
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char **argv) {
  cv::setNumThreads(1);
  std::string dir = argv[1];
  int n = argc > 2 ? atoi(argv[2]) : 200;
  std::vector<cv::String> files;
  cv::glob(dir + "/*.png", files, false);
  std::sort(files.begin(), files.end());
  if ((int)files.size() > n) files.resize(n);
  printf("%zu images\n", files.size());

  for (int flag : {cv::IMREAD_COLOR, cv::IMREAD_GRAYSCALE}) {
    const char *tag = flag == cv::IMREAD_COLOR ? "COLOR (8UC3)" : "GRAY  (8UC1)";
    // ---- decode
    auto t0 = clk::now();
    std::vector<cv::Mat> imgs;
    for (auto &f : files) imgs.push_back(cv::imread(f, flag));
    auto t1 = clk::now();
    // ---- pyramid (KLT settings from cfg/eff_*.json: win 15, levels 5)
    std::vector<std::vector<cv::Mat>> pyrs(imgs.size());
    for (size_t i = 0; i < imgs.size(); ++i) {
      cv::buildOpticalFlowPyramid(imgs[i], pyrs[i], cv::Size(15, 15), 5, true);
    }
    auto t2 = clk::now();
    // ---- KLT, 180 points, consecutive frames
    std::vector<cv::Point2f> p0;
    for (int i = 0; i < 180; ++i)
      p0.emplace_back(40 + (i % 15) * 28.0f, 40 + (i / 15) * 36.0f);
    double klt = 0;
    for (size_t i = 1; i < imgs.size(); ++i) {
      std::vector<cv::Point2f> p1;
      std::vector<uchar> st;
      std::vector<float> er;
      auto a = clk::now();
      cv::calcOpticalFlowPyrLK(pyrs[i - 1], pyrs[i], p0, p1, st, er,
                               cv::Size(15, 15), 5,
                               cv::TermCriteria(cv::TermCriteria::COUNT +
                                                    cv::TermCriteria::EPS,
                                                30, 0.01));
      klt += ms(a, clk::now());
    }
    auto t3 = clk::now();
    // ---- FAST detect
    auto det = cv::FastFeatureDetector::create(20, true);
    auto b = clk::now();
    for (auto &im : imgs) {
      std::vector<cv::KeyPoint> kps;
      det->detect(im, kps);
    }
    auto t4 = clk::now();
    printf("%s  decode %6.3f  pyramid %6.3f  klt %6.3f  fast %6.3f  ms/image\n",
           tag, ms(t0, t1) / imgs.size(), ms(t1, t2) / imgs.size(),
           klt / (imgs.size() - 1), ms(b, t4) / imgs.size());
  }
  return 0;
}
