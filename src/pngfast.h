// Copyright 2024 The XIVO Authors. All rights reserved.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "opencv2/core/core.hpp"

namespace xivo {

/** Decodes a grayscale PNG from memory into an 8-bit single-channel image,
 *  producing exactly the bytes `cv::imdecode(buf, IMREAD_GRAYSCALE)` produces,
 *  and returns true. Returns false -- having touched nothing -- for any file
 *  this path does not handle, in which case the caller must fall back to
 *  OpenCV.
 *
 *  Why this exists. On TUM-VI, PNG decode is the single largest item in either
 *  system's per-frame budget: 2.81 ms per image, measured on room1/cam0 (see
 *  notes-frontfast/harness/bench_decode.cpp). That is 21% of XIVO's monocular
 *  frame and 21% of its stereo frame, and OpenVINS pays it too -- its own
 *  `run_euroc_folder` reports `wall_imread_s=15.92` for 5642 images on room1
 *  stereo, i.e. 2.82 ms/image, 40% of its 14.08 ms end-to-end frame. It is not
 *  a cost either estimator can be blamed for and not one a live camera pays,
 *  but it is inside the measured throughput of both.
 *
 *  The sequences are 512x512 non-interlaced 16-bit grayscale PNGs (IHDR
 *  bit_depth 16, colour_type 0), ~282 kB each. `cv::imdecode` already asks
 *  libpng for the cheapest conversion available -- `grfmt_png.cpp:234` calls
 *  `png_set_strip_16` when the destination is CV_8U -- so there is no flag left
 *  to change. What remains is the cost of inflate plus libpng's per-row
 *  transform machinery, and both are reducible:
 *
 *    cv::imdecode(IMREAD_GRAYSCALE)                 2.809 ms
 *    this path                                      1.422 ms
 *      of which inflate (libdeflate)                1.118 ms
 *      the same inflate with stock zlib             1.884 ms
 *
 *  So the 1.39 ms saved splits about evenly: 0.77 ms from libdeflate's
 *  whole-buffer inflate over zlib's streaming one, and 0.62 ms from doing the
 *  unfilter and the 16->8 strip in one fused pass over each row instead of
 *  through libpng's row callbacks and transform list. Without libdeflate at
 *  build time this file falls back to `zlib`'s `uncompress` and still saves the
 *  second half.
 *
 *  Bit-identity is not an argument, it is checked: `bench_decode` compares
 *  every decoded image against `cv::imdecode` byte for byte (0 mismatches over
 *  300 images), and `unitTests_pngfast` re-checks it in-tree on a set of
 *  encoded fixtures covering all five PNG filter types at both supported bit
 *  depths.
 *
 *  Deliberately narrow. Anything that would make libpng apply a transform --
 *  a palette, an alpha channel, colour, a bit depth below 8, interlacing, a
 *  tRNS/gAMA/sRGB chunk -- is refused rather than approximated, because the
 *  point of the path is that it cannot disagree with OpenCV. */
bool DecodeGrayPng(const uint8_t *data, size_t size, cv::Mat &out);

/** Reads and decodes one image file as 8-bit single channel.
 *
 *  Equivalent to `cv::imread(path, IMREAD_GRAYSCALE)` for every input, and
 *  bit-identical to it; faster for the grayscale PNGs `DecodeGrayPng` accepts
 *  when `SetFastPngDecode(true)` has been called. The file is pulled into
 *  memory with one `read(2)` rather than handed to libpng as a `FILE*`: libpng
 *  asks for a few bytes at a time, so glibc's 4 kB buffer turns one 282 kB PNG
 *  into ~70 `read` syscalls plus a separate open/read/close for the format
 *  sniff. The buffer is `thread_local` and reused, so steady-state frames cost
 *  no allocation. */
cv::Mat ReadGrayImage(const std::string &path);

/** Enables the fast path in `ReadGrayImage`. Off by default, so a build that
 *  merges this file without a config key behaves exactly as before. The switch
 *  is process-wide rather than a member because decoding happens in the drivers
 *  (`pybind11/pyxivo.cpp`, `src/app/vio.cpp`), which read the image before any
 *  estimator object is involved; each of them sets it from its own config. */
void SetFastPngDecode(bool enabled);
bool FastPngDecodeEnabled();

/** How many images `ReadGrayImage` decoded on the fast path, and how many fell
 *  back to OpenCV. Exposed so a run can assert the fast path was actually taken
 *  rather than silently declined for the whole sequence. */
long NumFastPngDecoded();
long NumFastPngFallback();

} // namespace xivo
