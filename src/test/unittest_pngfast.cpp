// The fast grayscale-PNG path must be byte-identical to
// cv::imdecode(IMREAD_GRAYSCALE), and must decline everything it cannot
// reproduce. See src/pngfast.h for why it exists.
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "pngfast.h"

namespace xivo {
namespace {

std::vector<uint8_t> EncodePng(const cv::Mat &img, int compression = 3) {
  std::vector<uint8_t> buf;
  const std::vector<int> params{cv::IMWRITE_PNG_COMPRESSION, compression};
  EXPECT_TRUE(cv::imencode(".png", img, buf, params));
  return buf;
}

void ExpectSameAsOpenCV(const cv::Mat &img, int compression = 3) {
  const std::vector<uint8_t> buf = EncodePng(img, compression);
  const cv::Mat ref = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(ref.empty());
  cv::Mat fast;
  ASSERT_TRUE(DecodeGrayPng(buf.data(), buf.size(), fast));
  ASSERT_EQ(ref.rows, fast.rows);
  ASSERT_EQ(ref.cols, fast.cols);
  ASSERT_EQ(CV_8UC1, fast.type());
  EXPECT_EQ(0, cv::countNonZero(ref != fast));
}

// Content chosen to push libpng's adaptive filter selection through all five
// filter types: a flat field takes None/Up, a horizontal ramp takes Sub, a
// diagonal ramp takes Average/Paeth, and noise takes whatever is least bad.
cv::Mat Flat(int depth) {
  cv::Mat m(37, 53, depth == 16 ? CV_16UC1 : CV_8UC1,
            cv::Scalar(depth == 16 ? 4242 : 42));
  return m;
}

cv::Mat Ramp(int depth, bool diagonal) {
  cv::Mat m(64, 96, depth == 16 ? CV_16UC1 : CV_8UC1);
  for (int r = 0; r < m.rows; ++r) {
    for (int c = 0; c < m.cols; ++c) {
      const int v = diagonal ? (r * 7 + c * 13) : (c * 11);
      if (depth == 16) {
        m.at<uint16_t>(r, c) = static_cast<uint16_t>(v * 331);
      } else {
        m.at<uint8_t>(r, c) = static_cast<uint8_t>(v);
      }
    }
  }
  return m;
}

cv::Mat Noise(int depth, int seed) {
  cv::Mat m(65, 65, depth == 16 ? CV_16UC1 : CV_8UC1);
  cv::RNG rng(seed);
  rng.fill(m, cv::RNG::UNIFORM, cv::Scalar(0),
           cv::Scalar(depth == 16 ? 65535 : 256));
  return m;
}

// A one-pixel image and a single-row image: the unfilter loops special-case the
// first `bpp` bytes and the first row, so both edges need a case.
cv::Mat Tiny(int depth, int rows, int cols) {
  cv::Mat m(rows, cols, depth == 16 ? CV_16UC1 : CV_8UC1);
  cv::RNG rng(9);
  rng.fill(m, cv::RNG::UNIFORM, cv::Scalar(0),
           cv::Scalar(depth == 16 ? 65535 : 256));
  return m;
}

TEST(PngFast, Gray16MatchesOpenCV) {
  ExpectSameAsOpenCV(Flat(16));
  ExpectSameAsOpenCV(Ramp(16, false));
  ExpectSameAsOpenCV(Ramp(16, true));
  ExpectSameAsOpenCV(Noise(16, 1));
  ExpectSameAsOpenCV(Noise(16, 2));
}

TEST(PngFast, Gray8MatchesOpenCV) {
  ExpectSameAsOpenCV(Flat(8));
  ExpectSameAsOpenCV(Ramp(8, false));
  ExpectSameAsOpenCV(Ramp(8, true));
  ExpectSameAsOpenCV(Noise(8, 3));
}

TEST(PngFast, EdgeShapes) {
  for (int depth : {8, 16}) {
    ExpectSameAsOpenCV(Tiny(depth, 1, 1));
    ExpectSameAsOpenCV(Tiny(depth, 1, 17));
    ExpectSameAsOpenCV(Tiny(depth, 17, 1));
    ExpectSameAsOpenCV(Tiny(depth, 2, 2));
  }
}

TEST(PngFast, EveryCompressionLevelMatches) {
  // The compression level changes which filters libpng picks per row, so this
  // is the cheapest way to cover the filter switch.
  for (int z = 0; z <= 9; ++z) {
    ExpectSameAsOpenCV(Ramp(16, true), z);
    ExpectSameAsOpenCV(Noise(16, 4 + z), z);
  }
}

TEST(PngFast, DeclinesWhatItCannotReproduce) {
  cv::Mat fast;

  // Colour: libpng would run a gray conversion.
  cv::Mat colour(32, 32, CV_8UC3);
  cv::RNG(11).fill(colour, cv::RNG::UNIFORM, cv::Scalar(0, 0, 0),
                   cv::Scalar(256, 256, 256));
  std::vector<uint8_t> buf = EncodePng(colour);
  EXPECT_FALSE(DecodeGrayPng(buf.data(), buf.size(), fast));

  // Not a PNG at all.
  cv::Mat gray = Noise(8, 12);
  ASSERT_TRUE(cv::imencode(".jpg", gray, buf));
  EXPECT_FALSE(DecodeGrayPng(buf.data(), buf.size(), fast));

  // Truncated PNG: must not read past the buffer, and must not invent pixels.
  // Declining is the normal answer; the one case that legitimately succeeds is
  // a cut inside IEND's trailing CRC, after every row has already arrived, and
  // libpng accepts that file too -- so the contract is "decline, or agree with
  // OpenCV", not "always decline".
  buf = EncodePng(gray);
  for (size_t n : {size_t(0), size_t(7), size_t(8), size_t(20),
                   buf.size() / 2, buf.size() - 1}) {
    if (DecodeGrayPng(buf.data(), n, fast)) {
      const cv::Mat ref = cv::imdecode(cv::Mat(1, static_cast<int>(n), CV_8U,
                                               const_cast<uint8_t *>(buf.data())),
                                       cv::IMREAD_GRAYSCALE);
      ASSERT_FALSE(ref.empty()) << "accepted a file OpenCV rejects, size " << n;
      EXPECT_EQ(0, cv::countNonZero(ref != fast)) << "size " << n;
    }
  }

  // Corrupt filter byte inside the first IDAT is caught by the inflate or the
  // filter switch, never by an out-of-range write.
  buf = EncodePng(gray, 0);
  ASSERT_GT(buf.size(), 60u);
  buf[buf.size() / 2] ^= 0xff;
  DecodeGrayPng(buf.data(), buf.size(), fast); // must not crash either way
}

TEST(PngFast, ReadGrayImageIsIdenticalWithAndWithoutTheFastPath) {
  // Round-trips through a real file, which is what the drivers call.
  const cv::Mat img = Noise(16, 21);
  const std::string path = std::string(::testing::TempDir()) + "pngfast.png";
  ASSERT_TRUE(cv::imwrite(path, img));

  const bool was_on = FastPngDecodeEnabled();
  SetFastPngDecode(false);
  const cv::Mat slow = ReadGrayImage(path);
  SetFastPngDecode(true);
  const long before = NumFastPngDecoded();
  const cv::Mat fast = ReadGrayImage(path);
  EXPECT_EQ(before + 1, NumFastPngDecoded()) << "fast path was not taken";
  SetFastPngDecode(was_on);

  ASSERT_FALSE(slow.empty());
  ASSERT_FALSE(fast.empty());
  EXPECT_EQ(0, cv::countNonZero(slow != fast));
}

} // namespace
} // namespace xivo
