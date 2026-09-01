# M3 — front end and configuration

Commit `1c9e5a8`. Sixteen defects in `tracker.cpp`/`tracker.h`,
`options.cpp`, `camera_manager.cpp`, `fastbrief.cpp`, plus a config
migration.

## The one that changed the numbers: `use_prediction`

`tracker_cfg.use_prediction` appeared in 23 shipped configs. `grep -rn
use_prediction .` matched **only** the JSON files — no C++, no Python, no
ROS node ever read it. `Estimator::VisualMeasInternal` calls
`Predict(tracker->features_)` unconditionally (`estimator.cpp:1128`), and
`UpdateLK` seeds `cv::OPTFLOW_USE_INITIAL_FLOW` from `f->pred()`. So the
EKF's predicted pixel was always the KLT initial guess, in every config,
including the 22 that asked for it to be off.

That is a closed loop from filter state back into the front end — a
diverging filter biases the tracker toward its own wrong prediction — which
is exactly the failure mode the setting exists to prevent. So it is a real
bug and the key has to be honored.

But honoring it as written is a large behaviour change, and it went the
wrong way. Measured on room1–room6 (`results/bugfix/m3_frontend/`):

| seq | prediction ON (as always ran) | prediction OFF (config as written) |
|---|---|---|
| room1 | 0.1269 | 0.0808 |
| room2 | 0.0664 | 0.0842 |
| room3 | 0.1368 | 0.1616 |
| room4 | 0.1035 | **1.0443** |
| room5 | 0.1309 | 0.1127 |
| room6 | 0.0599 | 0.0644 |
| mean | **0.1041** | 0.2580 |

room4 diverges outright without prediction. TUM-VI room sequences are
hand-held and fast; a 64-px displacement bound around the *previous* pixel
is not a good enough KLT initialisation on its own.

**Resolution:** honor the key (that is the bug fix) and set every config
that said `false` to `true` (that is the behaviour-preserving migration —
`true` is what the code has always done and what the sweeps were tuned
against). 26 config files changed. The knob is now real and can be swept;
it just defaults to the setting that works.

## Everything else in M3 is bit-identical under this config

With `use_prediction: true`, M3's six-sequence output is **byte-identical**
to M2's, per sequence, not just in the mean. That is the useful negative
result: it independently confirms that all fifteen remaining fixes are on
paths `sweep_dlt_nodesc` never reaches, which is also why the test suite
never caught them. `extract_descriptor`, `match_dropped_tracks` and
`do_outlier_rejection` are all `false` in this config, and those three
flags gate almost all of it.

It also proves the `CandidateComparison` tie-break (below) never changes an
outcome here — the pointer order it replaced already agreed with id order
in every tie that occurred.

## The defects

**`match_dropped_tracks` was net-harmful, not inert.** `DetectLK` took
`std::vector<FeaturePtr> newly_dropped_tracks` **by value** and never
erased from it, but the caller's loop was written on the assumption that it
did:

```cpp
// "Mark all features that are still in newly_dropped_tracks_ ... as dropped"
for (auto f: newly_dropped_tracks) { f->SetTrackStatus(TrackStatus::DROPPED); }
```

So every feature the rescue branch had just marked `TRACKED` was re-marked
`DROPPED` and destroyed by `ProcessTracks`. And the rescue branch had
already run `MaskOut` on that corner and spent a detection slot — so
turning the flag on converted "a new feature is created here" into
"nothing is created here". The gate is very permissive in the live config
(`descriptor_distance_thresh == -1` makes the descriptor check return
`true` unconditionally, `max_pixel_displacement` is 64), so it fires often.

Fixed by taking the vector by reference and clearing rescued slots to
`nullptr` — *not* erasing, because `matchIdx` holds indices into that
vector — then skipping nulls in the caller. Rescued features also now get
`SetKeypoint`, since `keypoint_` was otherwise left at the original
detection pixel.

**BRIEF's border compaction desynchronised `status[]`.** This one is worth
recording in full because the code was *half* aware of it:

```cpp
kp.class_id = i;                       // original index, stashed deliberately
extractor_->compute(img_, kps, descriptors);   // ERASES keypoints near border
for (int i = 0; i < kps.size(); ++i) {
  auto f = vf[kps[i].class_id];        // correct -- uses class_id
  ...
  status[i] = 0;                       // WRONG -- post-compaction index
```

`opencv_contrib/modules/xfeatures2d/src/brief.cpp:288` runs
`KeyPointsFilter::runByImageBorder(keypoints, image.size(), PATCH_SIZE/2 +
KERNEL_SIZE/2)` with `PATCH_SIZE=48, KERNEL_SIZE=9` — it silently erases
every keypoint within **28 px** of the border. `margin` is 8 in every
shipped config, so the compaction always happens. With `kps[3]` removed,
the surviving `class_id`s are `[0,1,2,4,5,6,7,8,9]`; at `i == 3` the code
tests feature `vf[4]`'s descriptor and, on failure, kills track 3. Every
`status[i]` in that loop is now `status[kps[i].class_id]`.

(Secondary, not fixed because it needs a design decision: features in the
8–28 px band pass `MaskValid` but never appear after `compute`, so they are
silently exempt from the descriptor check and keep a stale descriptor.)

**`CheckHomography` never used the homography.**

```cpp
cv::Mat Hp0 = H * p0_h;                        // computed
number_t dist = cv::norm(p0_h, p1_h, NORM_L2); // ...and discarded
```

The homogeneous 1s cancel, so `dist` is `‖p0 − p1‖` — a raw pixel
displacement, with `H` having no effect at all. What is documented as a
reprojection check was a duplicate of the pixel-displacement check two
lines above, at 3 px instead of 64 px, i.e. 21× tighter. Now dehomogenises
`Hp0` and compares against `p1`, and returns false on an empty `H`.

**`findHomography` failure was reported as success.** When the registrator
cannot fit a model, OpenCV releases `H` and replaces the output mask with
an empty one — the pre-allocation on the previous line is reallocated away.
`OutlierRejection` then read `inlier_outlier_mask.at<uchar>(idx)` through a
null `data` pointer. `cv::Mat::at` only bounds-checks under `CV_DbgAssert`,
and this is a release build, so that is a straight segfault. It also
returned `true`, so `check_homography` became `true` and `CheckHomography`
multiplied by the empty `H`. Now checked, with `H.release()` and `return
false`.

**Two stale-pixel reads.** `keypoint_` is written *only* at feature
creation — the sole `SetKeypoint` calls are for a new LK feature, a new
MATCH feature, and a point-cloud feature. Neither `UpdateLK` nor
`UpdateMatch` refreshes it. So `f->keypoint().pt` is where the feature was
first detected, possibly hundreds of frames ago, while `f->back()` is the
current pixel. Lines 391 and 406 of the same function used the two
different quantities for the same feature — which is what makes it
unambiguous rather than a judgement call. At line 406 that meant
`findHomography` was being fit between first-detection pixels and current
pixels, i.e. not a frame-to-frame homography at all.

**Stale `num_outliers_rejected_`.** A member that persists across frames,
written only on `OutlierRejection`'s success path, but subtracted from
`num_valid_features` unconditionally by the caller. On the `< 4 valid
points` early return the caller used the *previous* frame's count, which
can drive `num_valid_features` negative. Now reset at function entry,
before any early return.

**`CandidateComparison` computed three score types and used none of them.**
All three branches wrote `score1`/`score2`; the return statement compared
`f1->score()`, which is hard-wired to `-P_(2,2)`. So
`"CovarianceDiagNorm"` and `"CovarianceDiagNormPlusOutlierCount"` were
unreachable, and the invalid-type branch left both uninitialised. This
comparator decides which candidates are promoted into the EKF state.

Fixed to use them, with a final tie-break on `f->id()`: ties are common
(every freshly-initialised candidate carries the identical initial depth
variance) and the caller sorts a vector that `MakePtrVectorUnique` ordered
by **pointer value**, so which of several tied candidates got promoted
depended on the heap layout.

**`fl_` had two different formulas in one class.** Constructor: `0.5 *
sqrt(fx² + fy²)`. `UpdateState`: `sqrt(0.5 * (fx² + fy²))`. These differ by
1/√2 — 135.04 vs 190.98 for TUM-VI cam0, whose actual focal length is
190.98. So the constructor's value was 41 % too small, *and* `fl_` would
jump by ×1.414 on the first autocalibration update. Only the RMS form is
dimensionally a focal length. Note this fix is **inert on its own**: its
only consumer scales `init_std_x_/y_`, which are dead code because the
badtri branch is always taken — it becomes observable only together with
the M4 `_badtri` normalisation.

**Four small ones.** `MaskOut` cached `half_size` in a function-local
`static`, freezing it at the first call's `mask_size` and silently ignoring
the parameter — and the header's default argument — for the life of the
process (also a first-use data race under `async_run`). `bool
outlier_rejection_success;` was read uninitialised as the **left** operand
of `&&`, where short-circuiting cannot save it. `LOG(INFO) << "..." <<
extract_descriptor_ ? "ENABLED" : "DISABLED"` parses as `(stream) ? ... :
...` because `<<` binds tighter than `?:`, so it logged `1`/`0` and threw
both literals away. `UpdatePointCloud` decremented its detection budget
outside the `if (!measurement_marked[fid])` guard, spending slots on
already-matched measurements.

**`FastBrief::meanValue`** did `memset(&mean, 0, ...)` where `mean` is a
`uint64_t*&` — that is the address of the caller's *pointer*, so it
clobbered the pointer plus 24 bytes past it on the stack and left the
freshly `new`'d buffer uninitialised (and leaked). It also used `i &
((i<<6)-1)` as a bit index, which is not one: `i=64` gives 64, a shift
count of 64 on a `uint64_t` (UB). Should be `i & 63`. DBoW2 vocabulary
*training* only, unreachable as configured.

## Dead config keys are systematic

Three separate instances found so far: `use_prediction` (read by nothing),
`comparison_score_type` (read, result discarded), and — found in M4 —
`feature_owner_change_cov_factor`, which the estimator looks up under the
key `filter_owner_change_cov_factor` that no config defines. All three are
plumbed through the sweep infrastructure. Anyone who swept them measured
noise and concluded the parameter did not matter.

Cross-checking every key in the live config against an actual reader is now
part of the audit. The rest of `tracker_cfg` checks out: all 30-odd
remaining keys are read where expected, no typo'd variants, no type
mismatches, and the one degrees→radians conversion in reach
(`triangulation.max_theta_thresh`) is done correctly at parse time.
