// Is PNG decode -- 2.8 ms/image, the largest single item in either system's
// per-frame budget on TUM-VI -- reducible without changing a single output byte?
//
// The TUM-VI 512_16 images are 16-bit non-interlaced grayscale PNGs (IHDR
// bit_depth 16, color_type 0), ~279 kB each. cv::imdecode(IMREAD_GRAYSCALE)
// already asks libpng for the cheapest possible conversion (grfmt_png.cpp:234
// calls png_set_strip_16 when the destination is CV_8U), so the cost is
// inflate + unfilter + strip, in libpng and zlib.
//
// This times three things on the same files:
//   A  cv::imdecode(IMREAD_GRAYSCALE)          -- what both systems do today
//   B  cv::imdecode(IMREAD_UNCHANGED)          -- same minus the 16->8 strip
//   C  libdeflate + hand-written unfilter+strip -- the candidate
//
// C is checked byte-for-byte against A on every image; the reported time is
// meaningless unless "mismatch 0" is printed.
//
// Build:
//   g++ -O2 -march=native -std=c++17 -I<opencv_install>/include/opencv4 \
//       bench_decode.cpp -L<opencv_install>/lib -lopencv_core -lopencv_imgcodecs \
//       -ldeflate -lz -o bench_decode
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>

#include <libdeflate.h>
#include <zlib.h>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

static uint32_t be32(const uint8_t *p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) |
         uint32_t(p[3]);
}

// --- the candidate fast path ------------------------------------------------
struct FastPng {
  libdeflate_decompressor *dec = nullptr;
  std::vector<uint8_t> idat;   // concatenated zlib stream
  std::vector<uint8_t> raw;    // inflated filtered rows
  FastPng() { dec = libdeflate_alloc_decompressor(); }
  ~FastPng() { libdeflate_free_decompressor(dec); }

  // Returns false when the file is not the shape this path handles, in which
  // case the caller must fall back to cv::imdecode.
  bool Decode(const uint8_t *p, size_t n, cv::Mat &out) {
    if (n < 8 || memcmp(p, "\x89PNG\r\n\x1a\n", 8) != 0) return false;
    size_t i = 8;
    int w = 0, h = 0, bd = 0, ct = 0;
    idat.clear();
    while (i + 8 <= n) {
      const uint32_t len = be32(p + i);
      const char *tag = (const char *)(p + i + 4);
      const uint8_t *data = p + i + 8;
      if (i + 12 + len > n) return false;
      if (!memcmp(tag, "IHDR", 4)) {
        if (len != 13) return false;
        w = int(be32(data));
        h = int(be32(data + 4));
        bd = data[8];
        ct = data[9];
        if (data[10] != 0 || data[11] != 0 || data[12] != 0) return false;  // deflate/adaptive/no interlace
        if (bd != 16 || ct != 0) return false;                              // 16-bit gray only
      } else if (!memcmp(tag, "IDAT", 4)) {
        idat.insert(idat.end(), data, data + len);
      } else if (!memcmp(tag, "IEND", 4)) {
        break;
      } else if (!memcmp(tag, "tRNS", 4) || !memcmp(tag, "gAMA", 4) ||
                 !memcmp(tag, "sBIT", 4) || !memcmp(tag, "cHRM", 4) ||
                 !memcmp(tag, "iCCP", 4) || !memcmp(tag, "sRGB", 4)) {
        // Colour/transparency chunks would change what libpng produces; refuse.
        return false;
      }
      i += 12 + len;
    }
    if (w <= 0 || h <= 0 || idat.empty()) return false;

    const size_t bpp = 2;                    // 16-bit gray
    const size_t stride = size_t(w) * bpp;   // bytes per unfiltered row
    const size_t need = size_t(h) * (stride + 1);
    raw.resize(need);
    size_t got = 0;
    if (libdeflate_zlib_decompress(dec, idat.data(), idat.size(), raw.data(),
                                   need, &got) != LIBDEFLATE_SUCCESS ||
        got != need) {
      return false;
    }

    out.create(h, w, CV_8UC1);
    // Unfilter in place, row by row, and strip 16->8 as each row completes.
    // PNG stores 16-bit samples big-endian and png_set_strip_16 keeps the high
    // byte, so the 8-bit output is every even byte of the unfiltered row.
    std::vector<uint8_t> zero(stride, 0);
    uint8_t *prior = zero.data();
    for (int y = 0; y < h; ++y) {
      uint8_t *row = raw.data() + size_t(y) * (stride + 1);
      const int ft = row[0];
      uint8_t *cur = row + 1;
      switch (ft) {
        case 0:
          break;
        case 1:
          for (size_t x = bpp; x < stride; ++x) cur[x] = uint8_t(cur[x] + cur[x - bpp]);
          break;
        case 2:
          for (size_t x = 0; x < stride; ++x) cur[x] = uint8_t(cur[x] + prior[x]);
          break;
        case 3:
          for (size_t x = 0; x < bpp; ++x) cur[x] = uint8_t(cur[x] + (prior[x] >> 1));
          for (size_t x = bpp; x < stride; ++x)
            cur[x] = uint8_t(cur[x] + ((int(cur[x - bpp]) + int(prior[x])) >> 1));
          break;
        case 4:
          for (size_t x = 0; x < bpp; ++x) cur[x] = uint8_t(cur[x] + prior[x]);
          for (size_t x = bpp; x < stride; ++x) {
            const int a = cur[x - bpp], b = prior[x], c = prior[x - bpp];
            const int pp = a + b - c;
            const int pa = std::abs(pp - a), pb = std::abs(pp - b), pc = std::abs(pp - c);
            const int pred = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
            cur[x] = uint8_t(cur[x] + pred);
          }
          break;
        default:
          return false;
      }
      uint8_t *d = out.ptr<uint8_t>(y);
      for (int x = 0; x < w; ++x) d[x] = cur[2 * x];
      prior = cur;
    }
    return true;
  }
};

int main(int argc, char **argv) {
  cv::setNumThreads(1);
  std::string dir = argv[1];
  int n = argc > 2 ? atoi(argv[2]) : 300;
  std::vector<cv::String> files;
  cv::glob(dir + "/*.png", files, false);
  std::sort(files.begin(), files.end());
  if ((int)files.size() > n) files.resize(n);
  const int N = (int)files.size();

  std::vector<std::vector<uint8_t>> bufs(N);
  for (int i = 0; i < N; ++i) {
    FILE *f = fopen(files[i].c_str(), "rb");
    fseek(f, 0, SEEK_END);
    bufs[i].resize(ftell(f));
    fseek(f, 0, SEEK_SET);
    if (fread(bufs[i].data(), 1, bufs[i].size(), f) != bufs[i].size()) return 2;
    fclose(f);
  }
  printf("%d images, mean %.0f kB\n", N,
         std::accumulate(bufs.begin(), bufs.end(), 0.0,
                         [](double a, const std::vector<uint8_t> &b) {
                           return a + b.size();
                         }) / N / 1024.0);

  std::vector<cv::Mat> ref(N);
  double tA = 0;
  for (int i = 0; i < N; ++i) {
    auto a = clk::now();
    ref[i] = cv::imdecode(bufs[i], cv::IMREAD_GRAYSCALE);
    tA += ms(a, clk::now());
  }
  printf("%-34s %7.3f ms/img\n", "A cv::imdecode GRAYSCALE", tA / N);

  double tB = 0;
  for (int i = 0; i < N; ++i) {
    auto a = clk::now();
    cv::Mat m = cv::imdecode(bufs[i], cv::IMREAD_UNCHANGED);
    tB += ms(a, clk::now());
  }
  printf("%-34s %7.3f ms/img\n", "B cv::imdecode UNCHANGED(16u)", tB / N);

  FastPng fp;
  cv::Mat out;
  int bad = 0, mismatch = 0;
  double tC = 0;
  for (int i = 0; i < N; ++i) {
    auto a = clk::now();
    bool ok = fp.Decode(bufs[i].data(), bufs[i].size(), out);
    tC += ms(a, clk::now());
    if (!ok) { ++bad; continue; }
    if (out.size() != ref[i].size() ||
        std::memcmp(out.data, ref[i].data, out.total()) != 0)
      ++mismatch;
  }
  printf("%-34s %7.3f ms/img   unsupported %d  mismatch %d\n",
         "C libdeflate+unfilter+strip", tC / N, bad, mismatch);

  // How much of C is inflate alone?
  double tI = 0;
  {
    libdeflate_decompressor *d = libdeflate_alloc_decompressor();
    std::vector<uint8_t> raw(512 * (1024 + 1) + 4096);
    for (int i = 0; i < N; ++i) {
      // re-extract IDAT cheaply outside the timer
      std::vector<uint8_t> idat;
      size_t k = 8;
      const uint8_t *p = bufs[i].data();
      size_t nn = bufs[i].size();
      int h = 0, w = 0;
      while (k + 8 <= nn) {
        uint32_t len = be32(p + k);
        if (!memcmp(p + k + 4, "IHDR", 4)) { w = be32(p + k + 8); h = be32(p + k + 12); }
        if (!memcmp(p + k + 4, "IDAT", 4)) idat.insert(idat.end(), p + k + 8, p + k + 8 + len);
        if (!memcmp(p + k + 4, "IEND", 4)) break;
        k += 12 + len;
      }
      size_t need = size_t(h) * (size_t(w) * 2 + 1);
      raw.resize(need);
      size_t got = 0;
      auto a = clk::now();
      libdeflate_zlib_decompress(d, idat.data(), idat.size(), raw.data(), need, &got);
      tI += ms(a, clk::now());
    }
    libdeflate_free_decompressor(d);
  }
  printf("%-34s %7.3f ms/img\n", "   of which inflate (libdeflate)", tI / N);

  // The same inflate with stock zlib, to attribute the win between "a faster
  // inflate" and "less work around it".
  double tZ = 0;
  {
    std::vector<uint8_t> raw;
    for (int i = 0; i < N; ++i) {
      std::vector<uint8_t> idat;
      size_t k = 8;
      const uint8_t *p = bufs[i].data();
      size_t nn = bufs[i].size();
      int h = 0, w = 0;
      while (k + 8 <= nn) {
        uint32_t len = be32(p + k);
        if (!memcmp(p + k + 4, "IHDR", 4)) { w = be32(p + k + 8); h = be32(p + k + 12); }
        if (!memcmp(p + k + 4, "IDAT", 4)) idat.insert(idat.end(), p + k + 8, p + k + 8 + len);
        if (!memcmp(p + k + 4, "IEND", 4)) break;
        k += 12 + len;
      }
      uLongf need = uLongf(size_t(h) * (size_t(w) * 2 + 1));
      raw.resize(need);
      auto a = clk::now();
      uncompress(raw.data(), &need, idat.data(), uLong(idat.size()));
      tZ += ms(a, clk::now());
    }
  }
  printf("%-34s %7.3f ms/img\n", "   same inflate, stock zlib", tZ / N);
  return 0;
}
