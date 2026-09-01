# M1 — multi-camera foundation

Goal: be able to *hold* two cameras and the rigid transform between them, with
zero change to monocular behaviour. No image is read from cam1 yet (that is M2)
and nothing enters the filter (M5).

## What changed

### 1. `CameraManager` singleton → indexed registry

`src/camera_manager.{h,cpp}`. Was:

```cpp
static CameraManager *Create(const Json::Value &cfg);
static CameraManager *instance() { return instance_.get(); }
static std::unique_ptr<CameraManager> instance_;
```

Now a `std::vector<std::unique_ptr<CameraManager>>` indexed by camera id, with
`Create(cfg, cam_id = 0)` and `instance(cam_id = 0)`.

The defaulted argument is the whole trick: all **34** existing `Camera::instance()`
call sites compile and mean exactly what they meant before (camera 0), so this is
a pure extension rather than a migration. `instance()` returns `nullptr` for an
unpopulated or out-of-range slot instead of growing the vector, because it is
called from `const` contexts and on hot paths.

`Create` on an already-populated slot is a no-op returning the existing camera.
That preserves the old `Create`-is-idempotent behaviour, which the codebase does
rely on: several unit tests call `Camera::Create` per-test.

### 2. `StereoRig` — the fixed left→right leg

New `src/stereo.{h,cpp}`. Holds `gc0c1` / `gc1c0` and the derived `Rc1c0`,
`Tc1c0`, and essential matrix `E = [Tc1c0]_x Rc1c0`.

**Design decision: the stereo extrinsics are NOT in the EKF state.** Adding them
would mean extending `Index` in `core.h`, which moves `kMotionSize`, `kFullSize`,
`kCameraBegin`, `kGroupBegin` and every covariance block keyed off them — a large
blast radius through `update.cpp`, `estimator.cpp` and `graph.cpp` for a quantity
TUM-VI calibrates to ~1e-3. The body-to-camera alignment `gbc` stays in the state
exactly as before, and refers to camera 0. If residuals later show the rig
calibration is the limiting error, this is the first thing to revisit — but it
should be *measured* first, not assumed.

Three primitives on top of the geometry:

- `ToCam1(Xc0)` — one matrix-vector product. This is what makes the stereo EKF
  update cheap: see the Jacobian note below.
- `Triangulate(xc0, xc1, *Xc0, *gap)` — midpoint of the two rays' closest
  approach. Chosen over DLT because it degrades gracefully at low parallax and
  because the ray gap it writes out is a *free, physically meaningful* quality
  gate (metres of disagreement between the two rays). Rejects near-parallel rays
  via `det > 1e-12 * b00 * b11` (a genuine angular test, not an absolute one) and
  rejects points behind either camera.
- `EpipolarResidual(xc0, xc1)` — `|b1 · (E b0)| / |E b0|` on *normalized bearings*.

  The normalization matters. The raw algebraic residual `b1' E b0` has units that
  drift with image position, so a fixed threshold on it means different things in
  different parts of the frame. Dividing by `|E b0|` turns it into the **sine of
  the angular miss**, so a threshold is quotable in radians. And it must work on
  bearings, not pixels: TUM-VI's cameras are 512×512 fisheyes where epipolar
  "lines" are curves in pixel space, so any pixel-space epipolar test would be
  wrong at the edges — precisely where the wide FoV is supposed to be helping.

Loud failure on an implausible baseline (`< 1 mm` or `> 1 m` → `LOG(FATAL)`).
A rig transform composed in the wrong direction, or in mm instead of m, otherwise
produces plausible-looking but silently wrong depths, which would be very hard to
distinguish from a tracking bug two milestones later.

### 3. `factory.cpp` wiring

Everything sits behind `cfg["stereo"] == true`. Absent that key the code path is
not merely inactive but unreachable, which is what lets the mono run act as a
regression gate for the rest of the project. `"stereo": true` with a missing
`camera1_cfg` or `stereo_cfg` is a `LOG(FATAL)`, not a silent fallback to mono —
a silent fallback would let a broken stereo config masquerade as a mediocre
stereo result.

### 4. `scripts/make_stereo_cfg.py` — config generated, not transcribed

Converts `dso/camchain.yaml` + a base mono config into a stereo config. The point
is traceability: hand-transcribing 8 intrinsics + 8 distortion coefficients + a
4×4 extrinsic is exactly the sort of thing that yields a config that is 99%
correct and quietly bad.

Verification that the conversion is faithful: the generated `camera_cfg` for cam0
is **identical, digit for digit**, to the hand-written `camera_cfg` already in
`cfg/sweep_dlt_nodesc.json` (fx=190.97847715128717, etc.). The converter is
therefore reproducing a known-good block, and its cam1 output comes from the same
code path.

kalibr convention check: `cam1.T_cn_cnm1` maps cam(n−1) → cam(n) = cam0 → cam1,
which is `StereoRig`'s `T_c1c0` — same direction, no inversion needed. The
resulting baseline is **101.09 mm**, matching TUM-VI's published ~10 cm.

`camchain.yaml` is byte-identical (md5 `63c2259677130737d7ea9ac595f49e88`) across
all six room sequences, so one generated `cfg/tumvi_stereo.json` serves all of
room1–room6. No per-sequence calibration handling is needed.

Detour worth recording: the first version of `parse_camchain()` was a hand-rolled
regex YAML reader, written to avoid a dependency. It failed on the multi-line
`T_cn_cnm1:` block (`cam1.T_cn_cnm1 missing or malformed`). Replaced with
`yaml.safe_load` (PyYAML installed into `dependencies/venv`). Writing a YAML
parser to save one `pip install` was a bad trade.

## Testing

New `src/test/unittest_stereo.cpp`, 8 tests, all passing. It runs against the
*actual generated* `cfg/tumvi_stereo.json`, so it doubles as a test of the
converter's output.

| Test | What it would catch |
|---|---|
| `BothCamerasLoadAndAreDistinct` | registry regressions; `instance()` no longer meaning cam0; slot 1 silently aliasing slot 0 (asserts the intrinsics differ by >1e-3); out-of-range slots must return null, not UB |
| `CreateIsIdempotentPerSlot` | a second `Create` swapping the camera out from under held pointers |
| `ProjectUnprojectRoundTripBothCameras` | 200 samples per camera over the usable fisheye field, round-trip < 1e-9 — the plan's stated M1 criterion |
| `BaselineAndExtrinsicsMatchCalibration` | baseline 0.10109 ± 1e-4; `gc0c1 * gc1c0 == I`; cached `Rc1c0`/`Tc1c0` in sync with `gc1c0`; `Rc1c0 ∈ SO(3)`; `ToCam1` agrees with applying `gc1c0`; >99% of the baseline along x |
| `TriangulateRecoversKnownDepth` | 500 random points, depth 0.5–20 m: exact recovery to 1e-9 **in metres**, ray gap 0. This is the metric-scale property stereo buys over the monocular `initial_z: 2.5` prior |
| `TriangulateRejectsDegenerateAndBehindCamera` | identical bearings (zero parallax) and behind-camera intersections must return false, not a garbage point |
| `EpipolarResidualZeroForTrueCorrespondence` | 200 true correspondences → residual < 1e-12; catches a wrong `E` or a flipped rig direction |
| `EpipolarResidualGrowsWithOffEpipolarError` | monotonic in the off-epipolar offset, **and** a 1e-3 normalized offset gives 1e-3 ± 2e-4 rad — pins the "residual is a sine, in radians" property that M3's gating threshold will depend on |

That last assertion is the one that makes the M3 threshold tunable rather than
magic: it establishes the units empirically instead of by comment.

### Mono regression gate — passed

Full 6-sequence run, `cfg/sweep_dlt_nodesc.json`, `XIVO_RANDOM_SEED=0`,
output in `results/m1_registry/`:

```
seq      ATE        RPE_rot    RPE_tra
room1    0.133641   0.529545   0.022903
room2    0.068441   0.723542   0.025861
room3    0.154850   0.732075   0.037016
room4    0.091062   0.636677   0.022796
room5    0.099227   0.575374   0.031542
room6    0.063883   0.525690   0.021193

mean over 6 seq:  ATE=0.1019  RPE_rot=0.6205  RPE_tra=0.0269
```

All six `tumvi_roomN_cam0` trajectory files are **byte-identical** to
`results/m0_jacfix/`. (The `tumvi_roomN_bench` files differ, but only because
that file is an append-only run header and the m0 directory accumulated two
runs — not a trajectory difference.)

Other unit tests: `unitTests_Jacobians` 13/13, equi 3/3, atan 6/6, pinhole 3/3,
radtan 3/3. The two pre-existing failures (`Triangulation.Angular_Reprojection_Error`,
`NumericalLinearAlgebra.SlowAndFastGivensMatch`) are unchanged from before M0.

## Note for M5, recorded now while it is fresh

`Feature::ComputeJacobian` builds `cache_.dXcn_d{Wsb,Tsb,Wbc,Tbc,Wsbr,Tsbr,x}`
and only applies `dxp_dXcn` at the very end. Since `Xc1 = Rc1c0 * Xc0 + Tc1c0`
with `Rc1c0` *constant*, the right camera's Jacobians are

```
dXc1_d(state) = Rc1c0 * dXc0_d(state)
J_r           = dxp_r/dXc1 * Rc1c0 * cache_.dXcn_d(state)
```

i.e. the entire existing derivative chain is reused with one fixed 3×3 multiply.
The stereo update needs **no new derivative math** — only a second projection and
a taller `H`. This is the single most important structural fact of the project.

## Status

M1 complete. Committed as `M1: multi-camera registry and fixed stereo rig
geometry`. Next: M2, the stereo data path (loader + `VisualMeasStereo` through
the estimator and the pybind11 binding).
