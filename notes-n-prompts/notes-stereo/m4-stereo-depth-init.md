# M4 — stereo depth initialization

Seed a new feature's depth by triangulating its left/right pair at the frame it
is created, instead of starting from the `initial_z` prior. This is the first
milestone that consumes the right observations M3 produces, and therefore the
first where the trajectory is *expected* to move: the gate changes from
byte-identity to a measured comparison.

Headline: **mean ATE 0.1025 → 0.0801 m (−21.8%)**, non-worse in 6/6 rooms;
mean RPE_tra 0.0286 → 0.0208 (−27%); RPE_rot unchanged (0.6204 → 0.6197), as
expected — a depth prior carries no rotation information.

## A measurement-protocol trap, found first

The first 6-room comparison looked like a regression: room1 ATE 0.1155 (mono)
→ 0.1324 (stereo). It was not. `run_eval.sh` scores ATE with

```
evaluate_ate.py --max_difference 0.001 ...
```

and I had been calling `evaluate_ate.py` with its default `--max_difference
0.02`. The 1 ms window associates only the ~25% of frames whose image timestamp
happens to land within 1 ms of a ground-truth sample (images 20 Hz, GT 120 Hz),
so the two protocols score *different subsets*:

| seq | ATE @0.001 (n≈550) | ATE @0.02 (n≈2600) |
|---|---|---|
| room1 mono | 0.107525 (n=720) | 0.115544 (n=2771) |
| room1 stereo | 0.088956 | 0.132400 |

The `off` control — `stereo_init.enable=false`, which must reproduce mono
exactly — is what caught it: its RPE matched the recorded baseline to six
digits while its ATE did not, which can only mean the trajectory was identical
and the *metric* differed. At `--max_difference 0.001` the `off` arm reproduces
the recorded baseline exactly in all six rooms (0.107525, 0.080113, 0.143678,
0.096501, 0.109758, 0.077238), so the M3 byte-identity gate held throughout.

Consequences adopted for the rest of the project:

- **The headline number uses `--max_difference 0.001`**, because that is what
  the README's monocular baseline and the exit criteria were produced with.
  Anything else is not comparable to the target.
- **Both are tracked** (`mATE001` and `mATE02` in every sweep table below). The
  official protocol throws away 75% of the poses, so it is the noisier of the
  two; a tuning decision is only accepted here if it improves both. Where they
  disagree in sign, the disagreement is reported rather than resolved by
  picking the flattering one.

## Two real bugs in the first implementation

**1. The temporal pre-subfilter triangulation silently clobbered the seed.**
`Estimator::ProcessTracks` calls `Feature::Triangulate` when a feature reaches
its second observation. That function rewrites `x_` — bearing *and* log-depth —
but leaves `P_` untouched. So a stereo-seeded feature had its metric depth
replaced by a two-frame temporal triangulation (baseline: whatever the rig moved
in 50 ms) while keeping the tight covariance the stereo had earned. Worst of
both: a depth the stereo never vouched for, asserted with the stereo's
confidence.

Fixed with a sticky `Feature::stereo_seeded()` flag that suppresses that path.
The suppression is exposed as `stereo_init.allow_retriangulation` (default
false) purely so the choice stays measurable; measured on all six rooms:

| arm | room1 | room2 | room3 | room4 | room5 | room6 | mATE001 | mATE02 |
|---|---|---|---|---|---|---|---|---|
| suppressed (default) | 0.0890 | 0.0814 | 0.1409 | 0.0627 | 0.0959 | 0.0612 | **0.0885** | **0.1170** |
| allowed | 0.1493 | 0.0952 | 0.1759 | 0.0996 | 0.0839 | 0.0774 | 0.1136 | 0.1528 |

Better in 5/6 rooms, so the reasoning and the measurement agree.

**2. The seeded depth was measured at the wrong point.** `StereoRig::Triangulate`
returns the *midpoint* of the two rays' closest approach. `Feature::Initialize`
pairs the depth it is given with the left bearing `UnProject(back())`. When the
rays do not exactly intersect — i.e. always, once there is any match or
calibration error — the midpoint lies off the left ray, so the seeded
(bearing, depth) pair named a slightly different 3D point than the one that was
triangulated. `TriangulateFromPixels` now projects the result onto the left ray,
keeping all three components coherent. `StereoInit.ReturnedPointLiesOnTheLeftRay`
covers it, and fails when the projection is removed.

Neither bug is large in isolation; both were found by asking *why* a number
moved rather than accepting it.

## What was ruled out along the way

Worse ATE with *better* RPE_tra is the signature of a map-scale error, so that
was checked before anything else: the Umeyama scale factor between estimate and
ground truth is 0.9958–1.0027 across all arms, and `ATE_scaled == ATE_rigid` to
four decimals. Not a scale error — and, incidentally, an independent
confirmation that the 101.09 mm baseline and the calibration are metrically
right to ~0.4%, from a completely different direction than M3's disparity
argument.

Feature-selection bias was the other candidate: `Feature::score()` returns
`-P_(2,2)`, and a stereo-derived log-depth variance is a monotone function of
depth, so seeding could have turned candidate selection into "always prefer the
nearest feature". Ruled out by measurement: `min_std_z = 0.3` clamps nearly
every seed to an identical variance (and therefore an identical score, falling
through to the id tie-break), and it changed the mean ATE by 0.0000 relative to
the unclamped run.

## Uncertainty propagation

The textbook form `sigma_z = z² sigma_d / (f b)` needs a single focal length. On
a ~190° equidistant fisheye the effective focal length varies substantially
across the field, so a constant `f` would understate uncertainty at the
periphery — exactly where matches are worst. `TriangulateFromPixels` instead
perturbs the right pixel by `sigma_px` and re-triangulates:

```
log_depth_std = |log z(xp1 + sigma_px * x̂) − log z(xp1)|
```

The perturbation is along **x** because the TUM-VI baseline is 99.9% along x
(asserted in the M1 rig test), so x is the disparity direction. Perturbing along
y would mostly slide the point along the epipolar line and understate the depth
uncertainty. If the *perturbed* pair triangulates degenerately while the nominal
one did not, the feature sits at the edge of usable parallax and the function
reports failure rather than a fabricated uncertainty.

`StereoInit.LogDepthStdGrowsWithDepthAndMatchesClosedForm` checks this against
the closed form in the image centre, where the fisheye is near-pinhole and the
formula does apply (15% tolerance), and checks monotonicity in depth.

## Threshold selection

All six rooms, seed 0, ASLR off. Noise floor from `m3a-determinism.md`: per-room
±0.013 m, 6-room mean ±0.005 m.

### `sigma_px` — the only knob that matters

| sigma_px | room1 | room2 | room3 | room4 | room5 | room6 | mATE001 | mATE02 | mRPEtra |
|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.1137 | 0.0598 | 0.1083 | 0.0526 | 0.1151 | 0.0551 | 0.0841 | 0.1031 | 0.0201 |
| **0.15** | 0.1046 | 0.0627 | 0.1007 | 0.0577 | 0.1078 | 0.0474 | **0.0801** | **0.0998** | 0.0208 |
| 0.25 | 0.1187 | 0.0840 | 0.1136 | 0.0545 | 0.0998 | 0.0317 | 0.0837 | 0.1098 | 0.0209 |
| 0.35 | 0.0939 | 0.0846 | 0.1325 | 0.0598 | 0.1015 | 0.0524 | 0.0874 | 0.1168 | 0.0219 |
| 0.50 | 0.0890 | 0.0814 | 0.1409 | 0.0627 | 0.0959 | 0.0612 | 0.0885 | 0.1170 | 0.0230 |
| 1.00 | 0.1182 | 0.0698 | 0.1515 | 0.1008 | 0.0955 | 0.1110 | 0.1078 | 0.1320 | 0.0270 |

There is a genuine interior optimum near 0.15 (0.10 is worse, so this is not
just "tighter is better"), and the ordering 0.15 < 0.25 < 0.35 < 0.50 < 1.00 is
monotone on `mATE001`, `mATE02` *and* `mRPEtra` — three metrics agreeing is more
than the noise floor would produce by chance. The 0.15-vs-0.50 gap (0.0084 on
mATE001) is ~1.7× the mean noise, so it is real but not large; 0.10/0.15/0.25
are mutually indistinguishable.

**Honest caveat.** 0.15 px is *tighter* than any defensible estimate of true
sub-pixel KLT match error (~0.3–0.5 px on 512×512). So `sigma_px` is not
functioning purely as a calibrated sensor-noise figure here; it is also acting
as a knob on how far the filter is willing to trust a metrically-correct seed
against a very weak monocular alternative (log-depth std 1.0, a factor of e).
Two readings are consistent with the data — the numerical propagation
overstates uncertainty somewhere, or the EKF simply benefits from over-trusting
a correct depth — and ATE alone cannot separate them. Flagged for M6, which
should re-tune `sigma_px` *jointly* with the M5 right-camera measurement noise,
since M5 introduces a second, continuous channel for the same geometry and will
likely move this optimum.

### The other three gates do essentially nothing

| arm | change | mATE001 | mATE02 | verdict |
|---|---|---|---|---|
| default | — | 0.0885 | 0.1170 | — |
| `max_gap` 0.10 → 0.30 | looser | 0.0885 | 0.1170 | byte-identical: never fires |
| `max_gap` 0.10 → 0.02 | tighter | 0.0972 | 0.1216 | worse |
| `min_std_z` 0.01 → 0.0 | no floor | 0.0924 | 0.1232 | slightly worse |
| `max_std_z` 1.0 → 0.30 | tighter | 0.1023 | 0.1359 | clearly worse |

(These were run at `sigma_px = 0.5`, the default at the time, so compare them to
the 0.0885 row rather than to the final 0.0801.)

- **`max_gap = 0.10` is inert.** Raising it to 0.30 gives a byte-identical
  trajectory, and the per-room counter confirms it: `gap=0` rejections in all
  six rooms. It is kept as insurance against a miscalibrated rig — a wrong
  extrinsic would make it fire loudly — not as a live filter. Tightening it to
  0.02 starts discarding good seeds and costs 0.009.
- **`max_std_z = 1.0`** means "no better than the monocular prior, so there is
  no reason to prefer stereo". Tightening it to 0.30 throws away usable far
  features and costs 0.014.
- **`min_std_z = 0.01`** (1% depth) binds only for very near, very
  well-conditioned features. Removing the floor is mildly worse, consistent with
  it doing its intended job of stopping one over-confident seed from locking in
  a calibration error.

Sweep configs were generated as deltas on `cfg/tumvi_stereo.json` and are not
checked in (they are trivially regenerable):

```python
b = json.loads(re.sub(r'(?m)//.*$', '', open('cfg/tumvi_stereo.json').read()))
c = copy.deepcopy(b); c['stereo_init'].update({'sigma_px': 0.15})
json.dump(c, open('cfg/m4_sig015.json', 'w'), indent=2)
```

## Where the seeds come from, and where they go

Per-room counters, shipped config. ~78% of newly created features get a stereo
seed; the rest fall back to the monocular prior, which is a graceful
degradation, not a failure.

| seq | new features | seeded | no right match | degenerate | out of range | std too big |
|---|---|---|---|---|---|---|
| room1 | 10143 | 7926 (78.1%) | 1658 | 537 | 21 | 1 |
| room2 | 8787 | 7001 (79.7%) | 1393 | 384 | 9 | 0 |
| room3 | 10222 | 8021 (78.5%) | 1619 | 567 | 14 | 1 |
| room4 | 7929 | 5815 (73.3%) | 1608 | 485 | 21 | 0 |
| room5 | 11575 | 9235 (79.8%) | 1749 | 563 | 28 | 0 |
| room6 | 4107 | 3271 (79.6%) | 613 | 201 | 22 | 0 |

Two things worth noting:

- **The dominant loss is "no right match" (~16%), not any of my gates.** That is
  ~3× the whole-run per-observation no-match rate (4–6%, see the `stereo:` lines:
  room1 141476/148139 = 95.5% matched). Newly created features are freshly
  detected corners in regions the tracker had no coverage of, and they have
  survived no temporal consistency check yet, so they are systematically the
  hardest to match. Raising the seed rate is therefore a *matching* problem, not
  a gating one — a candidate for M6.
- **Of my gates, only "degenerate" (~5%) does real work**, and it is mostly the
  perturbed-pair check described above rejecting features whose disparity is
  barely above the tracker's 1 px floor. Those are precisely the seeds whose
  depth is least trustworthy.

A counter bug was fixed while reading these numbers: `num_stereo_matched_` and
`num_stereo_attempted_` were reset every frame while the four `rejected_`
counters accumulated, so any rate computed across them mixed a per-frame
numerator with a run-total denominator. All seven are now cumulative. (The
97.8% figure in `m3-stereo-tracking.md` was summed externally per frame and is
unaffected.)

## Result

Shipped config `cfg/tumvi_stereo.json`, seed 0, ASLR off, ATE at
`--max_difference 0.001`:

| seq | mono ATE | M4 ATE | Δ | mono RPE_tra | M4 RPE_tra |
|---|---|---|---|---|---|
| room1 | 0.107525 | 0.104566 | −0.003 | 0.022311 | 0.019673 |
| room2 | 0.080113 | 0.062660 | −0.017 | 0.027844 | 0.017833 |
| room3 | 0.143678 | 0.100661 | −0.043 | 0.041235 | 0.023602 |
| room4 | 0.096501 | 0.057737 | −0.039 | 0.023102 | 0.019998 |
| room5 | 0.109758 | 0.107785 | −0.002 | 0.029526 | 0.025724 |
| room6 | 0.077238 | 0.047389 | −0.030 | 0.027653 | 0.018163 |
| **mean** | **0.1025** | **0.0801** | **−0.022** | **0.0286** | **0.0208** |

RPE_rot: 0.6204 → 0.6197, i.e. unchanged. The exit criterion of RPE_rot < 0.5°
is untouched by M4 and remains entirely M5's problem — rotation is constrained
by *where* features appear across the field, not by how well their depth is
known, so only a continuous second-camera measurement can move it.

Distance to target: mean ATE 0.0801 against < 0.06 m. M5 (the four-row stereo
EKF update) is the milestone that should close it.

## Tests

`src/test/unittest_stereo.cpp`, 5 new tests (13 total in the binary, all pass):

- `RecoversKnownDepthFromPixelObservations` — metric depth to 1e-6 relative,
  round-tripped through the real fisheye projections, 300+ samples.
- `ReturnedPointLiesOnTheLeftRay` — the bug-2 guard. Deliberately rounds the
  right pixel to integers so the rays genuinely miss, which is the only regime
  where midpoint and left-ray differ. **Verified to fail** when
  `X = onto_left_ray(X)` is removed.
- `LogDepthStdGrowsWithDepthAndMatchesClosedForm` — monotone in depth, and
  within 15% of `z σ_d / (f b)` at the image centre.
- `RejectsZeroDisparityAndBehindCamera` — identical pixels (point at infinity)
  and wrong-sign disparity both return false rather than a fabricated depth.
- `GapIsReportedForAnInconsistentMatch` — `gap` ≈ 0 for a true correspondence
  and > 1e-3 for one displaced 5 px across the epipolar line.

Full suite from the repository root: all pass except the two failures that
predate M0 (`NumericalLinearAlgebra.SlowAndFastGivensMatch`,
`Triangulation.Angular_Reprojection_Error`).

Note that `ctest` from `build/` reports 9/10 failures — every test that loads a
config does so by a path relative to the repository root. Pre-existing harness
quirk; run the binaries from the root (`for t in bin/unitTests_*; do $t; done`).
