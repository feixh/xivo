# M1 — leak census and register

Four independent methods were used, because no single one sees all three leak
classes on this codebase (see `m0-baseline-and-tooling.md` for why plain LSan is
blind here):

| method | what it can see | what it produced |
|---|---|---|
| ASan + LSan on `out-asan/bin/vio`, 6 rooms x 2 configs, full sequences | L1 definite leaks, L3 faults | nothing at all (see below) |
| valgrind massif, `--threshold=0.05`, snapshot diffing (`scripts/mem/massif_profile.sh`, `scripts/mem/massif_diff.py`) | L2 growth, attributed to a source line | the two descriptor sites and `traj_est` |
| direct pool census (temporary instrumentation in `~CircBufWithHash`, reverted) | L2 growth inside pooled objects, exactly counted | 9,059 retained descriptors, 2.72 MB pinned, at end of room1 |
| static ownership audit, three sub-agents over disjoint file sets | L1/L3 in code the two configs never execute | 12 further defects, all off the mono+IMU path |

## What ASan/LSan reports: nothing

`scripts/mem/leakcheck.sh` on all six rooms with both configs (12 full sequences,
`XIVO_RANDOM_SEED=0`) exits 0 with an empty sanitizer report every time — 12/12
`exit=0`, 12/12 "no sanitizer output" — and the unit suite under ASan is equally
clean. That is the M0 finding restated: every
long-lived allocation is owned by a `static std::unique_ptr` singleton whose
destructor runs *before* LSan's atexit check, so by the time LSan looks there is
nothing left to report. The classic leak criterion is already satisfied on
`auto`; it is not the interesting criterion here.

Reports: `/tmp/mem_m1/lsan/{vio_tumvi,vio_tumvi_nodesc}_room{1..6}/report.txt`.

## Heap growth, measured

massif on `out-valgrind/bin/vio`, room1, 8000 entries, `--time-unit=B`
`--threshold=0.05`, diffing two detailed snapshots from the steady-state part of
the run (default config: snapshot 20 -> 34, spanning 7.8 GB of allocation
traffic). Both steps are committed as tooling —
`scripts/mem/massif_profile.sh room1 /tmp/m.out vio_tumvi 8000` then
`scripts/mem/massif_diff.py /tmp/m.out 20 34`:

```
delta      site
+174,712   cv::fastMalloc <- BriefDescriptorExtractorImpl::compute
           <- Tracker::DetectLK (tracker.cpp:236)        349,936 -> 524,648 B
 +72,000   operator new <- vector<cv::Mat>::_M_realloc_append
           <- Track::SetDescriptor (feature.h:46) <- DetectLK (tracker.cpp:320)
                                                          144,192 -> 216,192 B
+202,554   in 326 places, all below 0.05% (diffuse; small-allocation churn)
```

96% of the attributable growth is the descriptor path. Separately, `traj_est`
(`vio.cpp:109`) appears as a doubling series — 49 kB, 98 kB, 393 kB, 786 kB by
entry 8000 — so it does not show up in a *between-snapshots* diff, but it is the
single largest accumulating block in the process.

Same experiment with `cfg/vio_tumvi_nodesc.json` (`extract_descriptor: false`):
heap is flat, 26.7 MB -> 25.7 MB across the run; the second-half diff is 171 kB
of diffuse churn plus 24 kB of `Track` vector capacity (`UpdateTrack`,
feature.h:86) which is a plateau, not a trend. So the whole descriptor component
disappears with the best-performing config, exactly as the M0 RSS slopes
suggested.

Raw profiles: `/tmp/mem_m1/massif_{default,fine,nodesc}.out`.

## The pool census

A temporary print in `CircBufWithHash<Feature>::~CircBufWithHash` (added,
measured, reverted — `git checkout src/mm.cpp`) counted what the 200 pooled
`Feature` objects were still holding at exit, on room1, default config:

```
entries  retained descriptors  max per slot  distinct pinned cv::Mat buffers  pinned bytes
  2,000                   151             1                              6        39,872
  8,000                 1,744            18                             86       572,864
 20,000                 5,379            48                            296     1,832,416
 whole                  9,059            73                            490     2,717,792
```

Linear in run length, with no upper bound. `vio_tumvi_nodesc` gives 0 in every
column. An independent measurement by the pool audit — gdb breakpoint on
`MemoryManager::~MemoryManager`, walking the pool through the gdb Python API —
reproduced the same numbers to the byte (9,059 headers, 1,225,728 B of headers,
490 pinned parents, 2,717,792 B pinned, 3.94 MB total; ~435 B per
`Feature::Create`, ~1.4 kB per image), and 0 under the no-descriptor config. The `Track` base vector's capacity, by contrast, saturates
(15,669 -> 25,936 -> 28,704 -> 33,024 slots total, ~0.5 MB), which is a plateau
and not a leak.

Two facts explain the numbers together:

* `Track::Reset()` (`src/feature.h:41-44`) clears the *pixel* history but never
  `descriptors_`, so a recycled pool slot inherits every descriptor its previous
  tenants ever had.
* `Tracker` stores `descriptors.row(i)`, an OpenCV *view*. Retaining one 32-byte
  row keeps the whole per-detection descriptor block (110-270 keypoints,
  3.5-8.6 kB) alive — a 100-250x amplification. Hence 9,059 rows pinning
  2.7 MB across only 490 distinct buffers.

# The register

`repro` column: `both` = both mono+IMU configs, `default` = `cfg/tumvi_cam0.json`
only, `vio` = the `bin/vio` app (the pyxivo harness does not have it), `off-path`
= real defect in compiled code that neither config executes.

## L2 — unbounded growth (this is the actual memory problem)

| # | site | repro | evidence | fix | milestone |
|---|---|---|---|---|---|
| L2-1 | `src/feature.h:41-44` `Track::Reset` does not clear `descriptors_` | default | pool census: 9,059 descriptors at exit, linear in run length; massif +72 kB/half-run at `feature.h:46` | `descriptors_.clear()` in `Reset` | M3 |
| L2-2 | `src/tracker.cpp:305,320,425,453,558,563` store `descriptors.row(i)` views | default | 490 pinned buffers = 2.72 MB at exit; massif +175 kB/half-run at `DetectLK (tracker.cpp:236)` | `.clone()` the row | M3 |
| L2-3 | `src/app/vio.cpp:65,109` `traj_est` grows 96 B per dataset entry and is never read | vio | massif `vio.cpp:109`: 49 -> 98 -> 393 -> 786 kB by entry 8000; ~3.1 MB live at the end of room1, ~4.7 MB transient across the last realloc | delete both lines | M3 |

L2-1 and L2-2 are one defect with two multipliers: L2-1 makes the count
unbounded, L2-2 makes each count expensive. Both are needed.

## L1 — definite leaks (unmatched `new`), all off the evaluated path

| # | site | repro | evidence | fix | milestone |
|---|---|---|---|---|---|
| L1-1 | `src/viewer.cpp:62,88` `SetHandler(new pangolin::Handler3D(...))` | off-path (`visualize:false`, `viewer_cfg:''`) | `pangolin::View` holds a raw non-owning `Handler* handler` (`thirdparty/Pangolin/include/pangolin/display/view.h:135,220`) and `~Viewer` deletes only the render states and the texture | own the handlers in `unique_ptr` members | M2 |
| L1-2 | `src/mapper.cpp:155` `Mapper::~Mapper() {}` leaks `voc_` (`:138`) and `ransac_params_` (`:143`, from `GetRANSACParams`, `:57-58`) | off-path (`USE_MAPPER` undefined) | ctor `new`s both, dtor is empty | `unique_ptr` members | M2 |
| L1-3 | `src/fastbrief.cpp:22` `mean = new uint64_t[4]`, never freed by any DBoW2 caller | off-path | out-parameter is a raw `uint64_t*&`; nothing owns it | see L3-1: same edit | M2/M4 |
| L1-4 | `src/fastbrief.cpp:110` `FastBrief::fromString` `a = new uint64_t[4]` | off-path | one leak per vocabulary node on load; `TDescriptor` is a raw pointer stored by value in DBoW2's node array | same ownership change | M2 |
| L1-5 | `common/utils.cpp:127` `auto writer = builder.newStreamWriter()` | off-path (`SaveJson` has no caller) | jsoncpp's `newStreamWriter` returns an owning raw pointer (`thirdparty/jsoncpp/include/json/writer.h:129`) | wrap in `unique_ptr` | M2 |

## L3 — memory-safety faults

| # | site | repro | evidence | fix | milestone |
|---|---|---|---|---|---|
| L3-1 | `src/fastbrief.cpp:23` `memset(&mean, 0, 32)` | off-path | `&mean` is the address of the reference, i.e. the caller's 8-byte pointer variable: 32 bytes written over it, and the pointer is set to null, so `mean[i>>6] \|= ...` at `:45` is a null write | `memset(mean, 0, ...)` | M4 |
| L3-2 | `src/estimator_accessors.cpp:16,48,80,112,146,180,214,247,279` `std::max` where `std::min` is meant | off-path (only the `int n_output` pybind overloads; the shipped savers use the no-arg ones) | the matrix is sized `max(size, n_output)` but the fill loop stops at `min(size, n_output)`, and Eigen does not zero-initialise, so uninitialised heap is copied into numpy. The correct form is in the same file at `:318` | `std::min` in all nine | M4 |
| L3-3 | `src/estimator.cpp:1454,1461` `return score1 <= score2` | off-path (same accessors) | non-strict comparator breaks `std::sort`'s strict-weak-ordering precondition; libstdc++'s unguarded partition can walk before `begin()` on ties | `<` | M4 |
| L3-4 | `pybind11/pyxivo.cpp:106,144` `cv::Mat image(num_row, num_col, CV_8UC3, info.ptr)` | off-path (the shipped scripts pass a *path*, binding the `imread` overload) | two faults: (a) channel count assumed 3 and geometry derived from strides, so a 2-D grayscale array is read ~2x past its end; (b) the `Mat` does not own `info.ptr`, and `VisualMeas` does **not** process the frame — `MaintainBuffer` executes `buf_.front()`, the *earliest* of >=10 buffered messages (`src/estimator.cpp:923-941`, and the `!async_run_` branch behaves the same way), so the frame is read after the numpy buffer may be gone | derive shape/channels from `info.shape`/`info.ndim`, and `.clone()` before enqueueing | M4 |
| L3-5 | `src/feature.h:50-51` `descriptor()` returns `descriptors_.back()` unguarded; `src/tracker.cpp:781-782` also indexes `fvec[0]` unguarded | off-path | `back()` on an empty vector is UB, and under `sweep_dlt_nodesc` `descriptors_` is provably always empty (census: 0 headers). `tracker.cpp:253` is guarded by `:246-249`; `:370` (`UpdateMatch`, needs `tracker_type: MATCH`) is not; `estimator_accessors.cpp:679` (`tracked_features()`, reached from `savers.py:291` in tracker-dump mode) is not. **Interacts with L2-1:** today a recycled slot may still hold a previous tenant's descriptor, so `back()` returns stale-but-valid data; once L2-1 clears the vector the same call becomes UB. They must be fixed together | guard both: early-return an empty `cv::Mat` from `GetDescriptors`, `CHECK(!descriptors_.empty())` in `descriptor()` | M3+M4 |
| L3-6 | `src/metrics.cpp:66-74` `ComputeRPE` | off-path (`app/legacy/evaluate.cpp` is not built) | `it_est` is incremented at `:68` and dereferenced at `:74` without re-checking `end()` | re-check after the increment | M4 |
| L3-7 | `src/estimator.h:165-166` `CameraCov()` returns a default-constructed `Matrix<double,9,9>` | off-path (`USE_ONLINE_CAMERA_CALIB` undefined; only caller is in the never-instantiated `EstimatorProcess`) | despite the name `all_zeros` is uninitialised; 648 B of indeterminate data returned | `::Zero()` | M4 |
| L3-8 | `src/update.cpp:403-407` iterates the `active_features` snapshot taken at `:247-249` *after* `DestroyFeatures(to_destroy)` at `:392` | off-path (`use_1pt_RANSAC: false`) | `RemoveFeatureFromState` sets `SetSind(-1)` (`src/estimator.cpp:762`), and `ComputeJacobian` indexes Jacobian blocks off `sind()`, so a destroyed feature writes at a negative offset | erase `to_destroy` from `active_features` after `:392` | M4 |
| L3-9 | `src/graph.h:126` `last_added_group_` is never initialised in the `Graph` ctor and never cleared by `Graph::RemoveGroup` (`src/graph.cpp:46-50`) | off-path (its only reader, `Estimator::CloseLoop` at `src/update.cpp:186`, is inside `USE_MAPPER`) | reading it before the first `AddGroup` is an uninitialised-pointer read; after a discard it points at a deactivated slot | `{nullptr}` initialiser + clear in `RemoveGroup` | M4 |
| L3-10 | `src/estimator.cpp:1319-1323` `DiscardGroup` clears `gauge_group_` but leaves `gauge_group_ptr_` | off-path (read only at `src/update.cpp:312`, inside `OnePointRANSAC`) | the pointer survives `Graph::RemoveGroup` + `Group::Deactivate`; only ever compared by identity, never dereferenced | `gauge_group_ptr_ = nullptr` alongside | M4 |

## Teardown defects — recorded, deliberately not "fixed" by draining

| # | site | repro | note |
|---|---|---|---|
| T-1 | `common/process.h:36-42` + `:44-56` | off-path (`Start()`'s only caller is the unbuilt `app/legacy/vio.cpp`) | the worker body is `for(;;)` with no stop flag, so `~Process()`'s `join()` can never return. Fixing needs an `atomic<bool>` stop flag; the queue's up-to-1000 pending `unique_ptr<EstimatorMessage>` (each owning a `cv::Mat`) are never destroyed. |
| T-2 | `src/estimator.cpp:94-97` + `:419-437` | off-path (`async_run: false` in both configs) | identical shape in live code: `Run()`'s lambda never returns, so `~Estimator()` would hang. Also `buf_` has no capacity bound in async mode — producers never block or drop (`:952-955`), so it grows for the whole run at ~786 kB per image message. |
| T-3 | `src/estimator.cpp:933-940` | both | the last <=10 buffered measurements are never executed and are freed only when the `Estimator` singleton dies. Memory is *bounded* (10 messages) and *is* freed, so this is a correctness bug, not a leak — and draining the buffer would process ten more measurements and change every trajectory. Out of scope for a memory task; flagged in the report instead. |

## Growth that is by design or plateaus — not fixed

* The `MemoryManager` pool itself: `max_features=200` x `sizeof(Feature)` +
  `max_groups=100`, all `new`ed up front in `CircBufWithHash`'s constructor and
  deleted in its destructor. massif attributes 9.75 MB to
  `Feature::Feature() <- OOSJacobian (jac.h:13)` and it is constant for the whole
  run. Pre-allocation is the point of the class.
* `Track` (the `vector<Vec2>` base), `FeatureAdj`, `GroupAdj`: `clear()` keeps
  capacity, so each pool slot's capacity rises to the worst track/graph size it
  ever held and then stops. Measured plateau ~0.5 MB total.
* `Viewer::trace_` (`viewer.cpp:133`): one `Vec3` per pose, needed to draw the
  trace. Unbounded in principle (17 MB/hour when driven from `pyxivo`'s
  `InertialMeas` at 200 Hz) but only with `-use_viewer`, and it is the feature's
  own data.
* `Estimator::ids_to_depths_` (`estimator.h:545`, filled at `:1166`) and
  `Mapper::InvIndex_` (`mapper.h:84`): both genuinely unbounded, both behind
  `simulation: true` / `USE_MAPPER`. Recorded, not fixed — fixing them means
  choosing a retention policy, which is a design change.
* `DataLoader::entries_` (2.7 MB for room1): a load-once index, constant.

## Non-memory defects found on the way — recorded, out of scope

`src/tracker.cpp:549-555` (`status[i]` indexed post-`compute()` while the feature
comes from `vf[kps[i].class_id]`; BRIEF's border filter shrinks `kps`, so the
wrong feature is dropped — dead while `descriptor_distance_thresh == -1`);
`src/tracker.cpp:604-626` (rescued tracks are not erased from
`newly_dropped_tracks`, so they get re-dropped); `src/tracker.cpp:765`
(`static int half_size` caches the first mask size forever);
`src/tracker.cpp:736-748` (`inlier_outlier_mask` read when `findHomography`
fails; dead while `do_outlier_rejection: false`);
`src/fastbrief.cpp:37,45` (`i & ((i<<6)-1)` where `i & 63` is meant — a UB shift
count for `i >= 1`); `src/mm.cpp:115-121` (`DestroyItem` decrements
`num_slots_initialized_`, so `GetItem`'s first branch can hand back a destroyed
slot without the `RemoveFromMapper` call the other branch makes — mapper-only);
`src/estimator.cpp:1319-1333` (`DiscardGroup` clears `gauge_group_` but leaves
`gauge_group_ptr_` pointing at a slot that can be recycled; only ever compared,
never dereferenced, and `use_1pt_RANSAC: false`);
`src/visualize.h:22` (`static void Delete()` declared, never defined or called).

## Two sub-agent claims that had to be resolved

* The threading agent called the `pyxivo` ndarray overload a use-after-free via
  the deferred `buf_`; the front-end agent called it safe because `VisualMeas` is
  synchronous. Reading `MaintainBuffer` settles it in favour of the first:
  even with `async_run: false` the just-pushed message goes into a heap and
  `buf_.front()` — a *different*, earlier message — is what executes. The frame
  is retained for >=10 messages. L3-4 stands.
* The front-end agent first estimated ~20 MB for the descriptor leak and then
  corrected itself to single-digit MB after measuring. The census above agrees
  with the corrected figure (2.7 MB pinned + 1.2 MB of `cv::Mat` headers on
  room1), and that is what M3 has to remove.
