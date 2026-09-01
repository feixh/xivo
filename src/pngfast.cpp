// Copyright 2024 The XIVO Authors. All rights reserved.
#include "pngfast.h"

#include <cstring>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <zlib.h>
#ifdef XIVO_HAVE_LIBDEFLATE
#include <libdeflate.h>
#endif

#include "glog/logging.h"
#include "opencv2/imgcodecs.hpp"

namespace xivo {

namespace {

bool fast_png_enabled = false;
long num_fast = 0;
long num_fallback = 0;

uint32_t BE32(const uint8_t *p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
         (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

/** Inflates a whole zlib stream whose uncompressed size is known exactly.
 *
 *  Knowing the size is what makes the fast route possible: libpng cannot know
 *  it (it streams into row callbacks), but a PNG's unfiltered size is fixed by
 *  IHDR, so the entire IDAT can go through one call with no window
 *  bookkeeping. */
bool InflateExact(const uint8_t *src, size_t src_size, uint8_t *dst,
                  size_t dst_size) {
#ifdef XIVO_HAVE_LIBDEFLATE
  // One decompressor per thread, kept alive across frames: allocating it costs
  // more than decoding a row.
  static thread_local libdeflate_decompressor *dec = nullptr;
  if (dec == nullptr) {
    dec = libdeflate_alloc_decompressor();
    if (dec == nullptr) {
      return false;
    }
  }
  size_t got = 0;
  return libdeflate_zlib_decompress(dec, src, src_size, dst, dst_size, &got) ==
             LIBDEFLATE_SUCCESS &&
         got == dst_size;
#else
  uLongf got = static_cast<uLongf>(dst_size);
  return uncompress(dst, &got, src, static_cast<uLong>(src_size)) == Z_OK &&
         got == static_cast<uLongf>(dst_size);
#endif
}

/** Undoes one row of PNG filtering in place.
 *
 *  `cur` is the row's `stride` filtered bytes, `prior` the previous row already
 *  unfiltered (all zeros for the first row), `bpp` the byte distance to the
 *  pixel on the left. Straight out of RFC 2083 section 6; the arithmetic is
 *  modulo 256, which is what the uint8_t truncation gives. */
bool UnfilterRow(int filter, uint8_t *cur, const uint8_t *prior, size_t stride,
                 size_t bpp) {
  switch (filter) {
    case 0: // None
      break;
    case 1: // Sub
      for (size_t x = bpp; x < stride; ++x) {
        cur[x] = static_cast<uint8_t>(cur[x] + cur[x - bpp]);
      }
      break;
    case 2: // Up
      for (size_t x = 0; x < stride; ++x) {
        cur[x] = static_cast<uint8_t>(cur[x] + prior[x]);
      }
      break;
    case 3: // Average
      for (size_t x = 0; x < bpp; ++x) {
        cur[x] = static_cast<uint8_t>(cur[x] + (prior[x] >> 1));
      }
      for (size_t x = bpp; x < stride; ++x) {
        cur[x] = static_cast<uint8_t>(
            cur[x] + ((int(cur[x - bpp]) + int(prior[x])) >> 1));
      }
      break;
    case 4: // Paeth
      for (size_t x = 0; x < bpp; ++x) {
        cur[x] = static_cast<uint8_t>(cur[x] + prior[x]);
      }
      for (size_t x = bpp; x < stride; ++x) {
        const int a = cur[x - bpp], b = prior[x], c = prior[x - bpp];
        const int p = a + b - c;
        const int pa = p > a ? p - a : a - p;
        const int pb = p > b ? p - b : b - p;
        const int pc = p > c ? p - c : c - p;
        const int pred = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
        cur[x] = static_cast<uint8_t>(cur[x] + pred);
      }
      break;
    default:
      return false; // corrupt file; let OpenCV report it
  }
  return true;
}

} // namespace

bool DecodeGrayPng(const uint8_t *data, size_t size, cv::Mat &out) {
  static const uint8_t kMagic[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  if (size < 8 || std::memcmp(data, kMagic, 8) != 0) {
    return false;
  }

  // Pass 1: read IHDR and gather IDAT. The chunks are walked rather than
  // assumed in order, because the spec permits any number of IDATs and
  // arbitrary ancillary chunks between them.
  int w = 0, h = 0, bit_depth = 0, color_type = 0;
  bool saw_iend = false;
  thread_local std::vector<uint8_t> idat;
  idat.clear();
  size_t i = 8;
  while (i + 12 <= size) {
    const uint32_t len = BE32(data + i);
    if (len > size || i + 12 + len > size) {
      return false;
    }
    const uint8_t *tag = data + i + 4;
    const uint8_t *body = data + i + 8;
    if (!std::memcmp(tag, "IHDR", 4)) {
      if (len != 13) {
        return false;
      }
      w = static_cast<int>(BE32(body));
      h = static_cast<int>(BE32(body + 4));
      bit_depth = body[8];
      color_type = body[9];
      // compression 0, filter method 0, interlace 0: anything else needs
      // libpng.
      if (body[10] != 0 || body[11] != 0 || body[12] != 0) {
        return false;
      }
      // Grayscale, 8 or 16 bits. Lower depths need png_set_expand_gray_1_2_4,
      // and every other colour type makes libpng run a transform.
      if (color_type != 0 || (bit_depth != 8 && bit_depth != 16)) {
        return false;
      }
    } else if (!std::memcmp(tag, "IDAT", 4)) {
      idat.insert(idat.end(), body, body + len);
    } else if (!std::memcmp(tag, "IEND", 4)) {
      saw_iend = true;
      break;
    } else if (!std::memcmp(tag, "tRNS", 4) || !std::memcmp(tag, "gAMA", 4) ||
               !std::memcmp(tag, "sBIT", 4) || !std::memcmp(tag, "cHRM", 4) ||
               !std::memcmp(tag, "iCCP", 4) || !std::memcmp(tag, "sRGB", 4)) {
      // Any of these can change what libpng emits for the same pixels.
      return false;
    }
    i += 12 + len;
  }
  // No IEND means the walk ran off the end of a truncated file. libpng refuses
  // such a file ("PNG input buffer is incomplete") even when every row has
  // already arrived, so accepting it here would be the one case where this path
  // disagrees with cv::imdecode -- by returning an image where OpenCV returns
  // an empty Mat. unitTests_pngfast pins that.
  if (w <= 0 || h <= 0 || idat.empty() || !saw_iend) {
    return false;
  }
  // Guard the multiplications below on 32-bit size_t and against a hostile IHDR.
  if (static_cast<uint64_t>(w) * h > (1ull << 28)) {
    return false;
  }

  const size_t bpp = bit_depth == 16 ? 2 : 1;
  const size_t stride = static_cast<size_t>(w) * bpp;
  const size_t raw_size = static_cast<size_t>(h) * (stride + 1);
  thread_local std::vector<uint8_t> raw;
  raw.resize(raw_size);
  if (!InflateExact(idat.data(), idat.size(), raw.data(), raw_size)) {
    return false;
  }

  out.create(h, w, CV_8UC1);
  // `prior` for row 0 is an implicit all-zero row.
  thread_local std::vector<uint8_t> zero;
  zero.assign(stride, 0);
  const uint8_t *prior = zero.data();
  for (int y = 0; y < h; ++y) {
    uint8_t *row = raw.data() + static_cast<size_t>(y) * (stride + 1);
    uint8_t *cur = row + 1;
    if (!UnfilterRow(row[0], cur, prior, stride, bpp)) {
      return false;
    }
    uint8_t *dst = out.ptr<uint8_t>(y);
    if (bit_depth == 16) {
      // PNG stores 16-bit samples big-endian and png_set_strip_16 keeps the
      // high byte, i.e. every even byte of the row.
      for (int x = 0; x < w; ++x) {
        dst[x] = cur[2 * x];
      }
    } else {
      std::memcpy(dst, cur, stride);
    }
    prior = cur;
  }
  return true;
}

cv::Mat ReadGrayImage(const std::string &path) {
  thread_local std::vector<uchar> buf;
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return cv::imread(path, cv::IMREAD_GRAYSCALE); // let OpenCV log the error
  }
  struct stat st;
  if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0) {
    ::close(fd);
    return cv::imread(path, cv::IMREAD_GRAYSCALE);
  }
  buf.resize(static_cast<size_t>(st.st_size));
  size_t off = 0;
  while (off < buf.size()) {
    const ssize_t n = ::read(fd, buf.data() + off, buf.size() - off);
    if (n <= 0) {
      break;
    }
    off += static_cast<size_t>(n);
  }
  ::close(fd);
  if (off != buf.size()) {
    return cv::imread(path, cv::IMREAD_GRAYSCALE);
  }
  if (fast_png_enabled) {
    cv::Mat fast;
    if (DecodeGrayPng(buf.data(), buf.size(), fast)) {
      ++num_fast;
      return fast;
    }
    ++num_fallback;
  }
  const cv::Mat raw(1, static_cast<int>(buf.size()), CV_8U, buf.data());
  return cv::imdecode(raw, cv::IMREAD_GRAYSCALE);
}

void SetFastPngDecode(bool enabled) {
  if (enabled == fast_png_enabled) {
    return;
  }
  fast_png_enabled = enabled;
#ifndef XIVO_HAVE_LIBDEFLATE
  if (enabled) {
    LOG(WARNING) << "fast_png_decode is on but XIVO was built without "
                    "libdeflate; falling back to zlib's inflate, which keeps "
                    "about half of the speedup";
  }
#endif
}

bool FastPngDecodeEnabled() { return fast_png_enabled; }
long NumFastPngDecoded() { return num_fast; }
long NumFastPngFallback() { return num_fallback; }

} // namespace xivo
