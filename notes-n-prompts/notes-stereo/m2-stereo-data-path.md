# M2 — stereo data path

Goal: get the right image from disk all the way to the tracker, through both
entry points (the python binding and the C++ `vio` app), without changing a
single number in the output.

That last clause is the design of this milestone. `Tracker::UpdateStereo` in M2
validates the right image, counts it, and then calls `Update(img)` — the ordinary
monocular path. Nothing consumes the right pixels yet. So the M2 acceptance test
is not "does it look reasonable", it is **the stereo trajectory must be
byte-identical to the monocular one**. Any difference at this stage is a bug in
plumbing, and finding it now is far cheaper than untangling it from a tracking or
Jacobian change later.

## What changed

### `msg::StereoImage` does not inherit from `msg::Image`

`src/message_types.h`. The obvious design — add a `image_path_r_` field to
`msg::Image` — was rejected. Consumers dispatch with
`dynamic_cast<msg::Image *>`, so an inheriting `StereoImage` would match that
cast and a consumer not yet updated for stereo would silently process the left
image alone. Making it a sibling turns that class of bug into a compile-time or
`LOG(FATAL) << "Invalid entry type"` failure instead of a quiet 50% loss of
information.

`unittest_stereo_loader.cpp` asserts this directly: `n_mono == 0` over the whole
of room1.

### `DataLoader(image_dir, image_dir_r, imu_dir)`

`src/loader.{h,cpp}`. Builds a `std::map<timestamp_t, path>` per camera and pairs
them on **exact** timestamp equality — no tolerance window.

Justification for zero tolerance: TUM-VI's cameras share a hardware trigger, and
`unittest_stereo_loader` verifies the two timestamp vectors for room1 are
*element-wise equal* (2821 each). Given that, a tolerance would not fix any real
data; it would only hide the case where the two directories come from different
recordings. That case is a real risk — this dataset ships as per-sequence
tarballs — so it is better to see zero pairs and a `LOG(FATAL)` than to see a
plausible-looking run built on mismatched frames.

Unpaired frames on either side are dropped and counted in a `LOG(WARNING)`;
zero pairs is fatal with a message naming both directories.

### `StereoPairDir(image_dir, cam_id, cam_id_r)`

Derives the partner directory by rewriting the `camN` component of the primary
one, rather than re-deriving a path from root/sequence. That keeps it impossible
for the two paths to be constructed by different rules and drift apart.

Uses `rfind`, not `find`: a dataset root can itself contain `cam0` (e.g.
`/mnt/cam0_dumps/...`), and rewriting that occurrence would point at a directory
that does not exist. There is a test for exactly this.

### `Estimator::VisualMeasStereo` / `VisualMeasStereoInternal`

`src/estimator.{h,cpp}` plus `internal::VisualStereo`.

`VisualMeasStereoInternal` mirrors `VisualMeasInternal` line for line, with one
substitution: `tracker->UpdateStereo(img, img_r)` in place of
`tracker->Update(img)`. Propagation, `Predict`, `UpdateStep` and the reference
group switch are the *same code*. This is deliberate — it means any future
divergence between the mono and stereo trajectories has to originate in tracking
or in what the update step does with right observations, and cannot come from a
subtly different measurement loop.

Only the left image goes to the `Canvas`: the canvas and everything reading it is
in left-camera pixel coordinates.

`VisualMeasStereo` on a system without a rig is `LOG(FATAL)`, not a fall-back to
the left image. A silent fall-back would present itself as a stereo run that
merely failed to improve — the most expensive possible failure mode for M6
tuning, because it looks like a tuning problem.

The `td` (temporal calibration) correction is applied once for the pair, which is
correct precisely because the rig is hardware-triggered.

### `Tracker::UpdateStereo`

`src/tracker.{h,cpp}`. Currently: reject an empty right image, reject a
size mismatch between the two (both fatal), stash `img_r_`, bump
`num_stereo_frames_`, then run the ordinary `Update(img)`.

The size check is cheap and catches a real scenario — pointing the right path at
a different resolution export of the same sequence, which would otherwise produce
garbage matches in M3 rather than an error.

### Entry points

- **`pybind11/pyxivo.cpp`**: `VisualMeasStereo(ts, left_path, right_path)`, plus
  `num_stereo_frames()` and `StereoEnabled()` accessors so a test can confirm
  from python that the stereo path really ran.
- **`scripts/pyxivo.py`**: `is_stereo_cfg()` reads `"stereo"` out of the
  estimator config. **There is deliberately no `-stereo` command-line flag** —
  the config is the single source of truth, so a config and an invocation cannot
  disagree, and the existing `run_eval.sh` needs no changes at all to run stereo.
  The right-image map is built by *checking each right file exists*, so a partial
  dataset shows up as counted dropped frames rather than a `KeyError` 20 minutes
  into a run.
- **`src/app/vio.cpp`**: reads the estimator config first to decide which
  `DataLoader` constructor to use, then dispatches on `StereoImage` before
  `Image`.

## Testing

### New `src/test/unittest_stereo_loader.cpp` — 3 tests, all pass

Runs against the real room1 data (and skips itself if the dataset is absent, so
the suite still passes on a bare checkout).

`PairsAllRoom1FramesAndInterleavesIMU` is the substantive one. It asserts:
- left and right timestamp vectors are **element-wise equal**, 2821 each — this
  is the premise the zero-tolerance pairing rests on, now checked rather than
  assumed;
- 2821 `StereoImage` entries, **0 bare `Image` entries**;
- IMU count matches `imu0/data.csv` exactly, and `size() == n_stereo + n_imu`, so
  nothing is dropped or duplicated in the merge;
- `image_path_ != image_path_r_` (reading one file twice would give zero
  disparity everywhere and silently disable stereo — a failure that would be
  very hard to diagnose from M4's results), and the two basenames are equal;
- entries are in ascending timestamp order, matching the sorted left timestamps.

`MismatchedDirectoriesYieldNoPairs` checks room1/cam0 against room2/cam1 and
confirms zero timestamp overlap, i.e. the pairing is genuinely timestamp-driven
and would not blind-zip two equal-length directories.

`SwapsTheCameraComponent` covers `StereoPairDir`, including the
`cam0`-in-the-root case.

### End-to-end: identical to mono, both entry points

| path | result |
|---|---|
| `run_eval.sh tumvi_stereo … room1` | ATE 0.133641, RPE_rot 0.529545 — **byte-identical** trajectory to `results/m1_registry/tumvi_room1_cam0` |
| `bin/vio` with `sweep_dlt_nodesc` (mono) vs `tumvi_stereo` | 30943-line state dumps **byte-identical** |

The stereo path really did execute — this is not a fall-back masquerading as a
pass. Directly from python:

```
StereoEnabled: True
num_stereo_frames: 37      (over the first 40 frames)
```

and the loader printed `stereo: 2821 pairs from 2821 left / 2821 right frames`.
37 rather than 40 is expected: the first few frames precede vision
initialization, which also matches the 2818 trajectory rows out of 2821 frames.

Trap avoided while checking the `vio` app: my first comparison used
`cfg/vio_tumvi.json` (which points at `tumvi_cam0.json`) against the stereo
config (derived from `sweep_dlt_nodesc.json`), and they differed at line 70. That
was two different base configs, not a stereo bug. Re-ran with `sweep_dlt_nodesc`
as the mono side and the outputs were identical. Worth recording because "the
comparison differed" is exactly the moment one is tempted to start debugging the
wrong thing.

### Full suite

`unitTests_stereo` 8/8, `unitTests_stereo_loader` 3/3, `unitTests_Jacobians`
13/13, equi 3/3, atan 6/6, pinhole 3/3, radtan 3/3. Two pre-existing failures
(`Triangulation.Angular_Reprojection_Error`,
`NumericalLinearAlgebra.SlowAndFastGivensMatch`) unchanged since before M0.

### Build note

`unitTests_stereo_loader` needs `jsoncpp` listed *again* after `${deps}`.
`common` appears after `jsoncpp` in the `deps` list and pulls in `Json::Value`
symbols, which a single-pass static link cannot resolve backwards. The other test
targets happen not to hit this because they do not pull in `xapp`.

## Status

M2 complete. Next: M3, left→right matching in `UpdateStereo` — at which point the
byte-identical gate necessarily stops holding, and the notes need to switch from
"identical" to a measured comparison. Before writing M3 I should record what the
*expected* right-match rate is, so a low rate is recognized as a bug rather than
absorbed as a tuning parameter.
