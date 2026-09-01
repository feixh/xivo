# M4 — the front end: one image copy and one pyramid too many

Commit: `M4: stop copying the image, and build the left pyramid once per stereo
frame` (`48a5f54`).

## What was there

Two redundant copies per frame, on the stereo path.

**The image clone.** `UpdateLK` opened with:

```c++
img_ = image.clone();
if (cfg_.get("normalize", false).asBool()) {
  cv::normalize(image, img_, 0, 255, cv::NORM_MINMAX);
}
```

The clone is an allocation plus a 256 kB memcpy per image (512x512 CV_8U on
TUM-VI). It is dead in both branches: when `normalize` is on, `cv::normalize`
overwrites every pixel of it from `image`; when it is off — which is what every
shipped config sets — nothing writes to `img_` at all. Every use of `img_` in
`UpdateLK`, `MatchStereo` and `DetectLK` is a read through a `const cv::Mat &` or
an `InputArray`, and `img_` is private with no accessor, so no code outside the
tracker can see it either.

The `cfg_.get("normalize", ...)` on the third line is a string lookup into a
`Json::Value` on every image, in a function that otherwise does no config reads.

**The duplicate left pyramid.** `MatchStereo` began:

```c++
std::vector<cv::Mat> pyr_l, pyr_r;
cv::buildOpticalFlowPyramid(img_, pyr_l,
                            cv::Size(stereo_win_size_, stereo_win_size_),
                            stereo_max_level_);
```

`stereo_win_size_` and `stereo_max_level_` *default to* `win_size_` and
`max_level_` (M3-stereo made them overridable but nothing overrides them), and
`MatchStereo` runs immediately after `Update`, which ends with
`std::swap(pyramid, pyramid_)`. So on every stereo frame this rebuilt, pixel for
pixel, the pyramid sitting in `pyramid_`: 5 levels of image plus 5 of Scharr
derivatives, ~1.4 MB written, from the same source with the same parameters.

## The change

`img_` becomes a header onto the caller's pixels (`img_ = image`), and
`MatchStereo` takes its left pyramid by reference from `pyramid_`:

```c++
const bool reuse_left = pyramid_is_current_ &&
                        stereo_win_size_ == win_size_ &&
                        stereo_max_level_ == max_level_;
std::vector<cv::Mat> pyr_l_own, pyr_r;
if (!reuse_left) { cv::buildOpticalFlowPyramid(img_, pyr_l_own, ...); }
const std::vector<cv::Mat> &pyr_l = reuse_left ? pyramid_ : pyr_l_own;
```

The flag, rather than an assumption, because `pyramid_` is *not* always the
current frame's: `UpdateLK`'s first-frame branch returns early (it sets the flag
after building), and `UpdateMatch` — the `MATCH`/`POINTCLOUD` tracker types —
builds no pyramid at all, so whatever is in `pyramid_` there belongs to an older
frame or to nothing. Both paths clear it. Getting this wrong would feed
`calcOpticalFlowPyrLK` last frame's left image against this frame's right one,
which is a plausible-looking disparity field rather than an obvious failure.

The lifetime argument for dropping the clone: `image` is owned by the caller for
the whole of `Visual::Execute` / `VisualMeasStereoInternal`, `img_` is read only
within `UpdateLK` and the `MatchStereo` that follows it in that same dispatch, and
there is no yield in between. The window in which the caller's buffer must stay
valid is *the same window as before* — the old code's clone also happened inside
`UpdateLK`, i.e. at dispatch time, not when the message was queued — it just
extends to the end of the tracker's work instead of ending at the first
statement.

`UpdateMatch` keeps its clone. It is not on the measured path, and its
`detector_`/`extractor_` route is not one this milestone characterized.

## Why the pyramid has to own level 0

This is the part that is not merely a deletion. `cv::buildOpticalFlowPyramid`
takes a `tryReuseInputImage` argument that **defaults to true**, and under it
OpenCV does:

```c++
if (tryReuseInputImage && img.isSubmatrix() && (pyrBorder & BORDER_ISOLATED) == 0) {
  img.locateROI(wholeSize, ofs);
  if (ofs.x >= winSize.width && ofs.y >= winSize.height && ...) {
    pyramid.getMatRef(0) = img;   // a view, not a copy
    lvl0IsSet = true;
  }
}
```

`pyramid_` has to survive into the *next* frame — that is what the temporal KLT
call reads. As long as `img_` was a clone, level 0 aliasing it was harmless:
`img_`'s buffer was refcounted and the next frame's clone allocated a fresh one.
Now `img_` is a view of a buffer the caller may reuse or free the moment the
measurement returns, so an aliased level 0 would be a dangling read one frame
later. The same applies on the `normalize` path for a different reason: it now
writes into `img_`'s existing buffer in place each frame (`cv::normalize`'s
`create` is a no-op at matching size) rather than into a fresh clone.

Hence `Tracker::BuildOwnedPyramid`, which is the same call with
`tryReuseInputImage=false` and the three intervening defaults spelled out because
the flag cannot be reached without them.

The condition never fires on TUM-VI: images come from `cv::imread` and are not
submatrices, so OpenCV copies level 0 regardless. It fires for a camera driver
that hands out an ROI into a larger frame buffer, which is the normal shape of a
zero-copy capture path — i.e. exactly the deployment where the rest of this
milestone matters most. Left as-is it would have been a use-after-free that
depends on the capture layer, invisible on the dataset the code is tested with.

## Tests

`unitTests_pyramid` (new target, 5 cases). The helper is a public static, so the
test calls it without constructing the tracker singleton.

| test | what it pins |
| --- | --- |
| `OwnedPyramidDoesNotAliasASubmatrixInput` | level 0 does not share `data` with a submatrix input, and refilling the parent buffer afterwards leaves the pyramid unchanged |
| `TheDefaultBuildDoesAliasASubmatrixInput` | the OpenCV behaviour above, asserted directly — if this ever fails the wrapper is redundant, which is worth learning from a test rather than from a profile |
| `OwnedPyramidHasTheSameContentAsTheDefault` | every level and every derivative level is identical (`NORM_INF == 0`) to the default call's, so forcing the flag off changes ownership and nothing else |
| `ANonSubmatrixInputIsCopiedEitherWay` | why the aliasing never showed up here |
| `TheWindowAndLevelArgumentsAreHonoured` | the level count in the vector length, and the window via `locateROI` on each level |

The last one needed a detour worth recording: the window size is invisible in
every pixel the pyramid returns. Each level's ROI has the same dimensions and the
same values whatever the padding, because `BORDER_REFLECT_101` gives the same
value at a given distance from the edge no matter how far it extends. What
`winSize` sets is the *margin* around each level inside its parent buffer, which
is what `calcOpticalFlowPyrLK` reads when a window straddles the image boundary.
So the witness is `locateROI`, not `cv::norm` — a content comparison would have
passed against a wrapper that ignored the argument entirely.

Second detour: `buildOpticalFlowPyramid` returns *fewer* levels than asked for
once halving would take a level below the window, so an exact
`2 * (maxLevel + 1)` count assertion is wrong in general (it fails at 128 rows /
`maxLevel` 4). The count is pinned as a bound and the agreement with the default
call carries the rest.

21/21 targets pass under `ctest`.

## Speed

Same sweep as M3 (`sweeps/m3m4.log`), M4 from the frozen worktree `xivo-effm4`
(`48a5f54`):

| arm | seq | wall (s) | **FPS** | track (ms) | process_tracks (ms) | visual_meas (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| m3_mono | room1 | 36.8 | 76.61 | 3.76 | 3.93 | 7.88 |
| **m4_mono** | room1 | 36.7 | **76.93** | 3.76 | 3.91 | 7.86 |
| m3_stereo | room1 | 85.8 | 32.89 | 13.42 | 8.27 | 21.89 |
| **m4_stereo** | room1 | 76.6 | **36.84** | **10.42** | 8.02 | 18.64 |
| m3_mono | room6 | 34.0 | 77.55 | 3.49 | 4.05 | 7.74 |
| **m4_mono** | room6 | 33.9 | **77.84** | 3.44 | 4.06 | 7.69 |
| m3_stereo | room6 | 81.8 | 32.21 | 13.57 | 8.78 | 22.54 |
| **m4_stereo** | room6 | 70.8 | **37.21** | **9.84** | 8.38 | 18.42 |

| | vs. baseline | vs. M3 |
| --- | --- | --- |
| mono | **3.68x** | 1.004x |
| stereo | **2.99x** | 1.14x |

The whole of it lands in `track`, which is the timer that brackets
`Tracker::Update` and `MatchStereo`: **13.4 -> 10.4 ms** on room1 and **13.6 -> 9.8
ms** on room6, a 3.0 and 3.7 ms saving against a 3.26 and 4.18 ms drop in frame
time. Nothing else moved, which is the right shape for a change that deletes work
and touches no arithmetic.

Mono is flat to within a repeat's noise (76.61 -> 76.93, and `track` 3.76 -> 3.76),
and that is the honest reading rather than a disappointment: the only thing this
milestone removes on the monocular path is the 256 kB `clone`, which at ~20 us
against a 13.0 ms frame is 0.15% and below what this harness resolves. The mono
arm was run to check that nothing *regressed* — the clone was load-bearing or it
was not — not to show a speedup.

Stereo/mono FPS ratio moves 0.42 (M3) to 0.48. Undoing part of what M2 did to that
ratio: the second camera's share of the remaining work is what M4 cut.

One number for scale. The duplicate pyramid was ~1.4 MB written per stereo frame
(4 levels of 512x512 plus their Scharr derivatives, halving) and removing it is
worth ~3.5 ms — i.e. it was running at ~0.4 GB/s effective, so the cost was the
Scharr convolutions and the pyrDown filtering, not the stores. That is why reuse
beats any amount of care about allocation.

## Accuracy

8-member ensembles from the frozen worktree `xivo-effm4` (`48a5f54`), 6 rooms each:

| | ATE | RPE_rot | RPE_tra | RPE_rot_i | RPE_tra_i |
| --- | --- | --- | --- | --- | --- |
| m3_mono | 0.0786 | 0.6205 | 0.0226 | 0.5126 | 0.0222 |
| **m4_mono** | **0.0786** | 0.6205 | 0.0226 | 0.5126 | 0.0222 |
| m3_stereo | 0.0549 | 0.6208 | 0.0139 | 0.5128 | 0.0132 |
| **m4_stereo** | **0.0549** | 0.6208 | 0.0139 | 0.5128 | 0.0132 |

**All 96 runs are byte-for-byte identical to M3's** (2 settings x 8 members x 6
rooms, `filecmp` on `m<k>/tumvi_<seq>_cam0`), and the divergence set against the
*baseline* is unchanged from M3's — the same 4 mono room3 runs and the same one
stereo room1 run, no new ones.

This is the outcome the milestone should have, and unlike M1's version of the same
result it needs no argument about gate margins: nothing here touches a floating
point operation. The image is not copied, so the same bytes are read; the pyramid
is not rebuilt, so the same pyramid is used —
`OwnedPyramidHasTheSameContentAsTheDefault` is what pins that the reused one is
pixel-identical to the one `MatchStereo` used to build for itself. The identity is
therefore a *check on the reasoning*, not evidence about accuracy: if any of the 96
had differed, the reuse would have been wrong (a stale pyramid, a mismatched
window) rather than merely differently rounded.

Which is also why the mono ensemble is worth having despite the change being
stereo-shaped. The dropped clone is on the monocular path too, and a monocular
divergence would have meant something was writing to `img_` — the assumption the
whole milestone rests on.
