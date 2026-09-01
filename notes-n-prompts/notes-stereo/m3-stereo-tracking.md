# M3 — left→right stereo matching

Goal: give every surviving left feature a right-camera pixel observation, so
M4 can triangulate a metric depth and M5 can add a second measurement row. M3
*produces* those observations and nothing reads them yet, which keeps the
regression gate from M1/M2 intact: a stereo run must still yield a trajectory
byte-identical to the monocular one.

## What was added

`Feature` gained `SetRightObs` / `ClearRightObs` / `has_right()` / `xp_r()` and
two members (`xp_r_`, `has_right_`). Both are cleared in `Feature::Reset`,
which matters because features are recycled from the `MemoryManager` pool: a
fresh feature must not inherit the right observation of whichever feature
previously occupied its slot. That is the kind of bug that would show up as a
handful of impossible depths thousands of frames in, so it is cheaper to
prevent than to find.

`Tracker::UpdateStereo` now runs `Update(image)` — bit-for-bit the monocular
path — and then `MatchStereo()`.

`MatchStereo()`:
1. Clears every feature's right observation first, so `has_right()` can only
   ever mean "matched on the current frame". Without this a feature that
   matched at frame *k* and failed at *k+1* would pair a stale right
   observation with a fresh left one — a silent, large inconsistency.
2. Collects features with `track_status()` in {TRACKED, CREATED}; only those
   have a meaningful current-frame pixel location.
3. Builds both pyramids and runs KLT left→right, then right→left on the whole
   batch for a circular-consistency check.
4. Applies four gates, each with its own counter.

## Why bidirectional KLT rather than epipolar search

The textbook stereo approach rectifies and searches along a scanline. These are
512×512 fisheye images with ~190° FOV; rectification would either crop most of
the field or introduce heavy resampling, and XIVO's whole design keeps images
unrectified and pushes the distortion into the camera model. Running the same
KLT the temporal tracker already uses costs two extra `calcOpticalFlowPyrLK`
calls per frame, needs no new machinery, and leaves the epipolar geometry as a
*validation* step instead of a search constraint. Measured cost: stereo runs
take roughly 1.4× the monocular wall clock, which is not a limiting factor here.

Deliberately **no** `OPTFLOW_USE_INITIAL_FLOW`: disparity depends on the
feature's unknown depth, so the left pixel location is already the best
available seed, and supplying a wrong hint would bias the search.

## The four gates and how the thresholds were chosen

Config lives in `tracker_cfg.stereo_matching` in `cfg/tumvi_stereo.json` so a
sweep can move them without a rebuild.

| gate | threshold | reasoning |
|---|---|---|
| KLT status / in-bounds | — | OpenCV happily reports points past the border; those are not matches. |
| disparity | 1.0 – 150.0 px | Below ~1 px there is no usable parallax, and near-zero disparity is also exactly what a KLT that simply failed to move looks like, so the floor catches two faults at once. The ceiling comes from the geometry: 101.09 mm baseline at f≈190 px means 150 px of disparity implies a point ~0.13 m from the camera, closer than anything in a room sequence. A larger displacement means the KLT latched onto unrelated texture. |
| circular | 1.0 px | The round trip must land back within a pixel. This is the strongest filter against repetitive texture, because an aliased match is usually not symmetric. |
| epipolar | 0.005 rad | `StereoRig::EpipolarResidual` returns the sine of the angular miss between the right bearing and the epipolar plane — the M1 unit test pins that unit by asserting a 1e-3 offset produces a 1e-3 residual. 5 mrad is ~1 px at the centre of this lens, loose enough not to fight calibration error and tight enough to reject a match that is off the epipolar line. |

The gates are evaluated cheapest-first (status, then bounds, then disparity,
then round-trip, then the two unprojections needed for the epipolar test).

Counters are kept separately rather than as one "rejected" total because they
diagnose different faults: a spike in epipolar rejections indicts the rig
calibration, a spike in circular rejections indicts repetitive texture, and a
spike in disparity rejections indicts the extrinsics' sign or scale.

## Measured match rate — room1, first 600 frames

Recorded *before* interpreting the number, per the note at the end of the M2
write-up, so that a low rate would be recognized as a bug rather than absorbed
as a tuning parameter. Expectation for a hardware-synchronized, well-calibrated
101 mm rig on textured indoor scenes: well above 90%.

```
frames with attempts: 597
attempted: 32057   matched: 31347   rate: 97.8%
rejected  klt=190  disparity=0  circular=347  epipolar=173
```

97.8% is consistent with a correct rig. Reading the breakdown:

- **disparity = 0** is the most informative entry. Not one match out of 32k
  fell outside [1, 150] px. That independently confirms the baseline magnitude
  and the direction of `T_c1c0`: had the extrinsic been inverted or scaled, a
  large fraction of matches would have piled up against one of the two bounds.
- **circular (347) > epipolar (173)** is the expected ordering. The rejections
  are dominated by texture ambiguity, not geometry — the opposite ordering
  would have pointed at the calibration.
- **klt = 190** is ordinary tracking failure at image borders and on
  low-texture patches.

The remaining 2.2% is not a concern: these are per-frame rejections of
individual observations, and M5 simply omits the right row for a feature with
no match rather than dropping the feature.

## Regression gate

The point of M3 producing but not consuming right observations is that the
trajectory must not move. All six rooms, `cfg/tumvi_stereo.json` vs
`cfg/sweep_dlt_nodesc.json`, seed 0:

```
room1 IDENTICAL 9c799411a2      room4 IDENTICAL 784d8c72a0
room2 IDENTICAL 0b6bca1736      room5 IDENTICAL 42a80cfdb1
room3 IDENTICAL 35f1491608      room6 IDENTICAL 0d476b1f0b
```

Byte-identical, not merely equal to 6 decimal places in the metrics. Any
difference here would have been a bug in the left-tracking path — `MatchStereo`
writes only state that nothing reads.

Establishing this gate cost one afternoon and immediately earned it back: the
first 6-room stereo run *did* differ on room3, and because the gate is
byte-identity rather than "the ATE looks about the same", it was obvious that
something needed explaining rather than absorbing. It turned out not to be a
stereo bug at all — see [m3a-determinism.md](m3a-determinism.md).

## Verification

- `unitTests_stereo` (8 tests) and `unitTests_stereo_loader` (3 tests) still pass.
- New `unitTests_determinism` (5 tests) — see the companion note.
- Match rate and rejection breakdown as above.
- Byte-identical trajectories on all six rooms.

Baseline going into M4 (this is a *re-measured* baseline; the determinism fix
changed which features get promoted, so the M0/M1 numbers are no longer the
reference):

```
seq      ATE        RPE_rot    RPE_tra
room1    0.107525   0.529762   0.022311
room2    0.080113   0.722767   0.027844
room3    0.143678   0.732361   0.041235
room4    0.096501   0.636700   0.023102
room5    0.109758   0.573526   0.029526
room6    0.077238   0.527510   0.027653
mean over 6 seq:  ATE=0.1025  RPE_rot=0.6204  RPE_tra=0.0286
```

## Next

M4 consumes `xp_r()`: triangulate at first observation to get a metric depth and
a much tighter depth variance, falling back to `initial_z` when `has_right()` is
false. That is the first milestone where the trajectory is *expected* to change,
so the gate switches from byte-identity to a measured comparison — against the
noise floor established in the companion note (per-room ±0.013 m, mean ±0.005 m).
