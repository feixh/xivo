# M2 — Fast grayscale PNG decode: 2.81 ms → 1.42 ms per image, bit-identical

`fast_png_decode` (top level, default `false`). Files: `src/pngfast.{h,cpp}`,
`src/test/unittest_pngfast.cpp`, wired into `pybind11/pyxivo.cpp`,
`src/app/vio.cpp`, `src/app/feature_tracker_only.cpp`.

## Why decode is fair game and why it is worth this much

`cv::imread(path, IMREAD_GRAYSCALE)` on a TUM-VI 512_16 image costs **2.81
ms**. That is 21% of XIVO's 13.5 ms mono frame and 21% of its 26.3 ms stereo
frame (two images), and it is the largest single line item in either. OpenVINS
pays the same thing at the same rate — `wall_imread_s=15.920` for 2821 stereo
pairs in `experiments/results/ov_fps_onecore/stereo/room1_r0/stats.txt` is 2.82
ms/image — so this is 40% of the 71.3 FPS stereo target and 32% of the 114.6 FPS
mono target. Both harnesses feed from files, so it is inside both measured
numbers, and neither system's estimator is involved.

It is also the only stage in the whole image path that can be made faster with
**zero** change to any downstream byte, which after M1's three negative results
is the property that matters most: everything else I could make cheaper changed
what the tracker saw and lost more in `process-tracks` than it saved.

## Why OpenCV's 2.81 ms is not tight

The TUM-VI images are 512x512, colour type 0 (grayscale), **bit depth 16**,
non-interlaced, ~282 kB. OpenCV's `grfmt_png.cpp` already asks for the cheapest
possible conversion — at line 234 it calls `png_set_strip_16`, and for a
grayscale source with `IMREAD_GRAYSCALE` it adds no other transform — so the
2.81 ms is not OpenCV doing something silly. It is:

1. **zlib's streaming `inflate`.** The single biggest piece. libpng cannot know
   the output size (it hands rows to a callback as they arrive), so it inflates
   into a sliding window with per-call bookkeeping.
2. **libpng's row machinery** — `png_read_row` per row, the transform list
   walked per row, the progressive-read state kept between rows.

A PNG's *unfiltered* size, though, is fixed exactly by IHDR: `h * (w*bpp + 1)`
bytes. So the whole IDAT stream can go through **one** whole-buffer inflate with
the destination pre-sized, and the five RFC 2083 row filters plus the 16→8 strip
are a few lines each.

## Measured (`harness/bench_decode.cpp`, room1/cam0, 300 images, one core)

| | ms/image |
|---|---|
| A `cv::imdecode(IMREAD_GRAYSCALE)` | 2.809 |
| B `cv::imdecode(IMREAD_UNCHANGED)` (16u out) | 2.566 |
| C this path (libdeflate + unfilter + fused strip) | **1.422** |
| inflate alone, libdeflate | 1.118 |
| the same inflate, stock zlib | 1.884 |
| C vs A pixel mismatches over 300 images | **0** |

So of the 1.39 ms saved, ~0.77 ms is libdeflate beating zlib's inflate and ~0.62
ms is not running libpng's per-row machinery. Note B: asking OpenCV for the
16-bit image and stripping it myself would save only 0.24 ms — the strip is not
the cost, the inflate is.

libdeflate is optional. `CMakeLists.txt` probes for it and defines
`XIVO_HAVE_LIBDEFLATE`; without it `InflateExact` uses zlib's `uncompress`,
which still gets ~0.9 ms of the 1.39 (whole-buffer, no libpng), and
`SetFastPngDecode(true)` logs a warning saying so. cmake prints
`-- libdeflate: /usr/lib/x86_64-linux-gnu/libdeflate.so` on this box.

## What it refuses

Everything it cannot reproduce byte-for-byte, falling back to `cv::imdecode`:
colour type != 0, bit depth other than 8 or 16, interlaced, non-zero compression
or filter method, a `tRNS`/`gAMA`/`sBIT`/`cHRM`/`iCCP`/`sRGB` chunk (any of which
can change what libpng emits for the same pixel bytes), a missing IEND, a
too-large IHDR, a failed inflate, an out-of-range filter byte, and anything that
is not a PNG. Chunks are walked rather than assumed in order, because the spec
allows any number of IDATs with ancillary chunks interleaved.

The missing-IEND rule is the one non-obvious case, and a test found it: a file
truncated one byte inside IEND's trailing CRC still contains every row, so an
earlier version of this decoder happily returned the image — while libpng
rejects it with "PNG input buffer is incomplete" and OpenCV returns an empty
Mat. That would have been the single input where the fast path *disagreed* with
`cv::imdecode`, in the direction of being more permissive. Requiring `saw_iend`
closes it.

## How the identity is established

Not by assertion — by test and by end-to-end comparison.

`unitTests_pngfast` (new ctest entry `PngFast`, so the build is now **22/22**,
was 21/21) has 6 tests: 8- and 16-bit against `cv::imdecode` on flat, ramp,
diagonal-ramp and noise images; the 1x1 / 1x17 / 17x1 / 2x2 edge shapes that
exercise the first-`bpp`-bytes and first-row special cases in the unfilter
loops; **every compression level 0..9**, which is the cheap way to sweep
libpng's adaptive per-row filter choice through all five filters; the refusal
list above; and a round trip through a real file via `ReadGrayImage`, asserting
`NumFastPngDecoded()` incremented so the test cannot silently pass by taking the
slow path.

End to end, with `XIVO_DUMP_PRECISE=1`, `tumvi_room1_cam0` with and without
`fast_png_decode` is **byte-identical under `cmp`, mono and stereo both**. That
is the real claim: turning this key on cannot change accuracy, so none of the
accuracy contract is at risk from it.

## Paired throughput, room1, same window

| arm | mono FPS | ratio | stereo FPS | ratio |
|---|---|---|---|---|
| base (merged `auto` cfg) | 74.28 | — | 38.05 | — |
| `fast_png_decode=true` | 81.16 | **1.093** | 41.58 | **1.093** |

Both modes gain the same 9.3%, which is the signature of a per-image saving: 1.39
ms of 13.5 mono, 2.78 ms of 26.3 stereo. On the authoritative one-core baselines
that is 83.1 → 90.8 mono and 41.1 → 44.9 stereo, from a change that provably
alters nothing.

## Where the shared floor now sits

| | OpenVINS | XIVO before | XIVO after |
|---|---|---|---|
| decode, mono frame | 2.82 ms | 2.81 | **1.42** |
| decode, stereo frame | 5.64 ms | 5.62 | **2.84** |

XIVO now enters the race with a 2.8 ms/frame structural advantage on stereo that
OpenVINS does not have, which is a third of the gap between 41.1 and 71.3 FPS.
