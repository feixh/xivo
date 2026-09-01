# M4 — memory-safety faults (leak class L3)

Scope: the L3 rows of the register in `m1-leak-register.md` that were still open
after M2 and M3. L3-1 (the `memset(&mean, ...)` in `FastBrief`) landed in M2
together with the L1-3 leak in the same function; L3-5 (`descriptor()` /
`GetDescriptors` on an empty `descriptors_`) had to land in M3 because clearing
`descriptors_` in `Track::Reset` is what turns it from stale-but-valid data into
UB. The remaining eight are here.

None of the eight is on the mono+IMU `bin/vio` path with the shipped configs —
that is why the release trajectories are bit-identical before and after (see
"Verification" below). They are reachable from the python bindings, from
`use_1pt_RANSAC: true`, from `tracker_type: MATCH`, from `USE_MAPPER`, and from
`USE_ONLINE_CAMERA_CALIB`. The register calls each one's reachability out
individually; this note is about the fixes.

## The eight defects

### L3-2 — `std::max` where `std::min` is meant (9 sites)

`src/estimator_accessors.cpp:16,51,83,115,149,183,217,250,282`, the `int n_output`
overloads of `InstateFeature{Sinds,RefGroups,IDs,Positions,Xc,xc,Preds,Meas,Covs}`.

Each sizes its output matrix `npts = std::max(instate_features.size(), n_output)`
and then fills it with a loop that stops at *both* `n_output` and the end of
`instate_features` — i.e. at the `min`. Eigen does not zero-initialise, so asking
for more features than are in state returns uninitialised heap to numpy: with 40
in-state features, `InstateFeatureCovs(200)` handed back 160 rows × 6 doubles =
7,680 bytes of whatever was in that block. The correct form is already in the
same file, in the no-argument overloads (`:318` pre-fix numbering).

Fix: `std::min` in all nine, with one comment at the first site explaining why
(the other eight are a one-token change each and a comment apiece would be
noise).

This is the one L3 defect with a *silent wrong answer* as its symptom rather than
a crash, and no sanitizer catches it: the memory is validly allocated, just never
written. It is caught by reading the fill loop, and the register's cross-check is
the no-arg overloads three lines away.

### L3-3 — non-strict comparators passed to `std::sort`

`src/estimator.cpp:1456-1470`, `FeatureCovComparison` and
`FeatureCovXYComparison`, both `return score1 <= score2`.

`std::sort` requires a strict weak ordering; `<=` makes `comp(a, a)` true.
libstdc++'s `__unguarded_partition` relies on the comparator to stop its inner
loops, so a run of equal scores lets the pivot loop walk past either end of the
range — an out-of-bounds read, and a swap of whatever is there.

Fix: `<` in both, with a comment naming the precondition.

Contained: both are members of `Estimator`, and the only callers are the nine
accessors above (the XY one appears once more in `graph.cpp:294`, in a
commented-out line). Equal scores are not hypothetical here — two features that
share a reference group and were initialised in the same frame can have
bit-identical covariance norms.

### L3-4 — the numpy-buffer `VisualMeas` overloads

`pybind11/pyxivo.cpp`, both buffer overloads (`VisualMeas` and
`VisualMeasTrackerOnly`). Two independent faults in the same four lines:

```cpp
int size_row = info.strides[0];
int num_col = size_row / info.strides[1] / info.itemsize;
int num_row = info.size / size_row;
cv::Mat image(num_row, num_col, CV_8UC3, info.ptr);   // wrong shape, and a view
```

(a) *Shape from the strides, channels hard-coded.* For an `(H, W, 3)` uint8 array
the arithmetic happens to come out right. For a 2-D grayscale `(H, W)` array
`strides = (W, 1)`, `itemsize = 1`, so `num_col = W` and `num_row = H` — but the
`CV_8UC3` type means the `Mat` spans `3*H*W` bytes over an `H*W`-byte buffer.

(b) *A view, not a copy.* The `cv::Mat` does not own `info.ptr`, and
`Estimator::VisualMeas` does not process the frame it is handed: it appends a
message and `MaintainBuffer` processes `buf_.front()`, the oldest of ≥10 buffered
messages (`MESSAGE_BUFFER_SIZE` defaults to 10, `src/estimator.h:556-567`). By
the time the pixels are read, python may have dropped its reference to the array.

Fix: one helper, `CloneImageFromBuffer` (`pyxivo.cpp:35-51`), used at both call
sites. It takes the geometry from `info.shape`/`info.ndim`, rejects anything that
is not 2-D or 3-D with 1–4 channels with a `std::runtime_error` (which pybind
turns into a python exception), and returns `borrowed.clone()`.

Note that (b) is *not* reliably sanitizer-visible: numpy's freed block is
normally handed straight back out for the next array, so the stale read lands in
valid memory and merely returns the wrong pixels. That is why the fix is a clone
rather than a test assertion — see the reproduction below for what is and is not
observable.

### L3-6 — `ComputeRPE` dereferences past the end

`src/metrics.cpp:66-74`. The pairing loop checks `it_est < est.end()` at the top,
then does `auto gY = (it_est++)->g_;` and dereferences `it_est` five lines later
to compute `desire`. When the matched estimate is the last one, that read is one
past the end of the vector. The same holds for `it_gt`.

Fix: re-check both iterators after the increments and `break` — the value
returned is unchanged (with the iterator at `end()` the following search loop is
empty and `continue` skips the pair anyway), so this is purely about the read.

This is the only L3 defect with a real regression test, because it is the only one
where a small hand-built input reaches the faulting line: see
`src/test/unittest_metrics_rpe.cpp` and "Reproductions" below.

### L3-7 — `CameraCov()` returns uninitialised stack

`src/estimator.h:162-168`, the `#else` branch taken when
`USE_ONLINE_CAMERA_CALIB` is undefined (which is the shipped configuration):

```cpp
Eigen::Matrix<number_t, 9, 9> all_zeros;   // despite the name
return all_zeros;
```

648 bytes of indeterminate stack. Fix: `= Eigen::Matrix<number_t, 9, 9>::Zero()`,
which is what the name already promised.

### L3-8 — destroyed features left in the `OnePointRANSAC` snapshot

`src/update.cpp`. `active_features` is snapshotted from `mh_inliers` at `:245-249`,
the high-innovation rescue destroys some of those features at `:392`
(`DestroyFeatures(to_destroy)`), and `:411-415` then walks the *snapshot* to
`RestoreState` and re-`ComputeJacobian`. `RemoveFeatureFromState` has meanwhile
set the destroyed feature's `sind` to `-1` (`src/estimator.cpp:762`), and
`ComputeJacobian` indexes its Jacobian block off `sind()` — so a destroyed
feature writes at a negative offset into the state.

Fix: erase `to_destroy` from `active_features` right after the `DestroyFeatures`
call (`update.cpp:393-400`).

The other `DestroyFeatures` call site, in `MHGating` (`update.cpp:133`), needs no
such fix: its `to_destroy` set is built and consumed inside that function, and
the inlier vector it returns never contains a destroyed feature.

### L3-9 — `Graph::last_added_group_`

`src/graph.h:126`. Never initialised by the constructor, and never cleared by
`Graph::RemoveGroup`. Read before the first `AddGroup` it is an indeterminate
pointer; read after the last-added group has been discarded it points at a
deactivated pool slot, which by then may belong to a different group.

Fix: `GroupPtr last_added_group_{nullptr};` plus a doc comment
(`graph.h:123-129`), and clear it in `RemoveGroup` when it is the group being
removed (`graph.cpp:49-53`).

### L3-10 — `gauge_group_ptr_` outlives its group

`src/estimator.cpp:1318-1328`, `DiscardGroup`. When the discarded group is the
gauge group the id is cleared (`gauge_group_ = -1`) but the pointer is not, and
the group is then removed from the graph and deactivated — its pool slot is
available for the next group. The pointer's only reader is the identity
comparison `groups_with_low_inn_inlier.count(gauge_group_ptr_)` at
`update.cpp:312`, so nothing is dereferenced; the fault is that the comparison
silently starts asking about the slot's *next* tenant, which decides whether
`OnePointRANSAC` picks a temporary reference group.

Fix: `gauge_group_ptr_ = nullptr;` alongside `gauge_group_ = -1;`.

## Reproductions

Two of the eight can be made to fail on demand. The rest are argued from the
code; where a sanitizer cannot see a defect this note says so rather than
implying coverage.

### L3-6 under ASan

`src/test/unittest_metrics_rpe.cpp` is new, registered as the ctest `MetricsRPE`
target. `EndOfEstimateDoesNotReadPastTheEnd` builds three ground-truth poses at
0/1/2 s and a single estimate at 0 s, so the loop body runs with `it_est` on the
final element. With the fix removed from `metrics.cpp` and the test rebuilt in
`build-asan`:

```
[ RUN      ] MetricsRPE.EndOfEstimateDoesNotReadPastTheEnd
==2399526==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x79c19dfe0088
READ of size 8 at 0x79c19dfe0088 thread T0
    #0 ... std::chrono::operator+ ... /usr/include/c++/15/bits/chrono.h:706
    #1 ... xivo::ComputeRPE(...) .../src/metrics.cpp:74
    #2 ... MetricsRPE_EndOfEstimateDoesNotReadPastTheEnd_Test::TestBody() .../unittest_metrics_rpe.cpp:41
```

With the fix in place it passes, in both the release and the ASan build. Because
`ComputeRPE` returns the same numbers either way, this test can only *fail* under
a sanitizer — the header comment in the test file says so, so that nobody later
concludes from a green release run that the check is exercised. The companion
test `IdenticalTrajectoriesHaveZeroError` (nine poses at 0.5 s spacing compared
against themselves, RPE must be 0 and must not be `-1`) is the guard against the
new `break` cutting the pairing short; that one does fail without a sanitizer.

### L3-4 from python under ASan

`scripts/mem/pybind_buffer_check.{sh,py}` are new. The script drives the buffer
overload of `VisualMeas` with 60 frames and 10 IMU samples each, deletes the
numpy array immediately after handing it over, and allocates three throwaway
arrays per frame so that a freed buffer is reused before the frame is processed.
Run against `out-asan` with `libasan.so` and `libstdc++.so` preloaded (the
wrapper explains why both are needed).

| layout | old code | fixed |
| --- | --- | --- |
| `(512, 512)` grayscale | `AddressSanitizer: SEGV on unknown address ... in icv_k0_ownsCopy_8u_inv` (libopencv_core) | `OK gray: detections=25 instate=0` |
| `(512, 512, 3)` colour | `OK color: detections=44 instate=0` | `OK color: detections=44 instate=0` |

So fault (a) is a hard, reproducible crash for a grayscale frame — which is what
a user feeding an already-mono camera would naturally pass. Fault (b) produced no
sanitizer report in either layout, exactly as predicted above: the freed numpy
block is recycled by the next allocation, so the estimator reads valid memory
holding the wrong pixels. The colour row is therefore evidence that the clone did
not break the working path, not evidence about the aliasing.

### Not reproducible by construction

* **L3-2** — uninitialised-but-allocated memory. Invisible to ASan; MSan would
  see it, but MSan needs an instrumented libstdc++/OpenCV/Eigen stack, which is
  out of proportion to a one-token fix that the no-arg overloads in the same file
  already show the correct form of.
* **L3-3** — needs a run of exactly-equal covariance norms *and* a libstdc++
  partition layout that walks off the end. Reasoned from the standard's
  precondition and libstdc++'s `__unguarded_partition`.
* **L3-7** — reading uninitialised stack; same MSan argument as L3-2. The only
  caller is in `EstimatorProcess`, which is never instantiated in this build.
* **L3-8, L3-9, L3-10** — all three need a config the shipped ones do not set
  (`use_1pt_RANSAC`, `USE_MAPPER`) and, for L3-9/L3-10, a pool slot reuse in the
  same frame as the read. Reasoned from the code paths cited above.

## Verification

Build: `scripts/mem/build.sh release` and `scripts/mem/build.sh asan` both exit
0 with no new warnings.

Unit tests, release build (`ctest` plus the gtest binaries): **37 pass, 2 fail**.
The two failures are `NumericalLinearAlgebra.SlowAndFastGivensMatch` and
`Triangulation.Angular_Reprojection_Error`, both pre-existing on `auto` and
recorded in `m0-baseline.md`.

Unit tests, ASan build (`ctest` in `build-asan`): 7 of 9 targets pass, the same
two fail, and **zero `AddressSanitizer` reports** in the whole log — the two
failures are gtest numeric assertions, not sanitizer findings.

End-to-end, release build, both mono+IMU configs over the six TUM-VI room
sequences:

| config | ATE (mean over 6) | RPE_rot | RPE_tra | vs M0 baseline |
| --- | --- | --- | --- | --- |
| `tumvi_cam0` | 0.1409 | 0.6219 | 0.0352 | identical=20 differ=6 |
| `sweep_dlt_nodesc` | 0.1267 | 0.6226 | 0.0364 | identical=20 differ=6 |

The six differing files in each case are the `run_room*.log` files, and the only
difference in them is the output directory embedded in the three command lines
the harness echoes (`.../m0_baseline_default` vs `.../m4_default`) — they carry no
timings or other run-dependent content, so `differ=6` here means "no difference
at all". (An earlier draft of this note said the logs differ in wall-clock
timings; they do not contain any.) Every trajectory and metrics file is
byte-identical to the M0 baseline, which is the expected result: none of the eight
defects is on the `bin/vio` mono+IMU path with these configs.

End-to-end under ASan/LSan (`scripts/mem/leakcheck.sh`), whole sequences, with
the M4 fixes in place:

| run | entries | exit | sanitizer output |
| --- | --- | --- | --- |
| `room1`, `vio_tumvi` | 30,943 | 0 | none (`report.txt` empty) |
| `room4`, `vio_tumvi_nodesc` | 24,440 | 0 | none (`report.txt` empty) |

Exit 0 matters specifically: ASan returns 23 when LSan reports a leak, so a clean
exit is the leak gate as well as the memory-safety one. The remaining four
sequences and the second config are M5's full matrix; these two are here to show
M4 did not introduce a fault of its own on the path the release evaluation
exercises.

## Tooling added

* `scripts/mem/pybind_buffer_check.py` / `.sh` — the L3-4 reproduction, kept
  because it is the only check that exercises the numpy-buffer entry point at
  all; the shipped python scripts all pass a *path* and bind the `imread`
  overload instead.
* `src/test/unittest_metrics_rpe.cpp` + the `MetricsRPE` ctest target.

## What M4 does not claim

* No leak-report change. All eight are memory-safety faults, not leaks; the LSan
  output is the same before and after. The leak work is M2 (definite leaks) and
  M3 (unbounded growth).
* No behavioural change on the shipped mono+IMU configs — and that is asserted
  from the bit-identical trajectories, not from reading the code alone.
* The fixes to L3-2 do change what the python `InstateFeature*(n_output)`
  accessors return when `n_output` exceeds the number of in-state features: the
  returned matrix now has `min` rows instead of `max`, i.e. it is shorter and
  contains no garbage rows. Any caller that assumed a fixed `n_output` row count
  sees a shorter matrix. No caller in this repository does — `savers.py` and
  `pyxivo` demos use the no-argument overloads, which always returned `min`.
