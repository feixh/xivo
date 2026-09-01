# Report — find and fix memory leaks with ASan/LSan (XIVO, monocular + IMU)

Branch **`auto-memory`** in the xivo package, worktree
`/home/ubuntu/workspace/auto-slam-engineer/xivo-memory`, branched from `auto`
@ `888511d`. Six commits, HEAD `631df4b`. 22 source files changed
(+416/-73) plus 14 new/changed files of tooling under `scripts/mem/` (+1023).
Detailed notes in `notes-n-prompts/notes-memory/`, one document per milestone.

---

## Summary

**18 memory defects found, catalogued and fixed** — 5 definite leaks (L1),
3 unbounded-growth defects (L2) and 10 memory-safety faults (L3). The register
(`notes-memory/m1-leak-register.md`) gives every one a `file:line`, a
reachability verdict, the evidence it was found by, and the fix.

**The headline finding is methodological: AddressSanitizer and LeakSanitizer
report absolutely nothing on this codebase — before *or* after the fixes — and
that is expected rather than good news.** 12 full TUM-VI sequences × 2 mono+IMU
configs under ASan+LSan exit 0 with an empty report on the `auto` branch, and
they still do at HEAD. Two reasons, both established in M0 and re-confirmed with
a `use_globals=0` census: every long-lived XIVO object hangs off a
`static std::unique_ptr` singleton, so LSan calls it reachable by definition;
and those singletons' destructors run *before* LSan's atexit check, so the
containers that grew all run are already freed when LSan looks.

**The real memory problem was therefore invisible to the tool the task names,
and it was on the evaluated path.** A pooled `Feature` slot inherited every
descriptor its previous tenants had ever held (`Track::Reset` cleared the pixel
history but not `descriptors_`), and each retained 32-byte descriptor was an
OpenCV *view* pinning the whole per-frame BRIEF matrix. Measured by a direct pool
census at the end of room1: **9,059 retained descriptors, worst slot 73, 2.72 MB
pinned across 490 buffers, growing linearly with run length and with no upper
bound.** A third defect, `traj_est` in `vio.cpp`, accumulated 96 bytes per
dataset entry and was never read: ~3.1 MB live, ~4.7 MB across its final
realloc. All three are gone.

| metric | `auto` baseline | `auto-memory` HEAD |
|---|---|---|
| retained descriptors at exit, whole room1 | 9,059 (max 73/slot), 2.72 MB pinned | **200 (max 1/slot), 6,400 B pinned** |
| retention vs run length (2k → 8k → 30,943 entries) | 151 → 1,744 → 9,059 (linear) | 151 → 200 → **200** (saturates) |
| peak RSS, room1 whole, default cfg | 86.0 MB | **78.8 MB** |
| peak RSS, room1 whole, nodesc cfg | 80.0 MB | **76.5 MB** |
| peak RSS vs run length (5k entries → whole sequence) | — | 78.6 / 78.8 / 80.4 / 78.8 MB (**≤1.8 MB over a 6× range**) |
| massif total allocation traffic, 8k entries | 42.98 GB | **42.62 GB (-0.8%)** |
| massif steady-state heap | 26.4-26.9 MB | **25.0-25.2 MB** |
| LSan-reported leaks, 6 rooms × 2 cfgs | 0 | 0 |
| ASan findings, 6 rooms × 2 cfgs, ~170k entries instrumented | 0 | **0** |
| mean ATE over 6 rooms (default / nodesc) | 0.1409 / 0.1267 | **0.1409 / 0.1267** |
| trajectory files | — | **byte-identical**, 20/20 per config |
| wall clock, room1 whole (default / nodesc) | 32.02 s / 30.46 s | 32.18 s / 30.79 s (**+0.5% / +1.1%**, inside noise) |

**Both exit criteria are met, with the honest qualification that "free of memory
leaks" cannot be established by LSan alone here and is argued from four
independent metrics instead** (§5).

**Accuracy does not move at all — every trajectory file is byte-identical to the
baseline — and that is a result, not a lucky escape.** It is argued in §4.3 from
the code before it was measured: `descriptor()` returns `descriptors_.back()`,
which is the current tenant's freshest descriptor whether or not dead tenants'
entries are still underneath it; and cloning a descriptor changes its address,
not its 32 bytes. The independent check is stronger than a metric comparison: a
binary built from the `auto` branch and the delivered binary produce identical
30,943-line trajectory files.

**Three defects that a memory task should not fix are recorded and left alone**
(§6): two worker threads whose bodies are `for(;;)` so that `join()` in the
destructor can never return, and a message buffer whose last ≤10 measurements
are never executed. Draining any of them changes the trajectory, which is a
correctness change, not a leak fix.

---

## 1. What was asked, and what was done

| Requirement | Status |
|---|---|
| Assume monocular camera + IMU | Held. Both shipped mono+IMU configs are evaluated at every milestone; off-path defects are fixed but each one's reachability is stated |
| Use tools like ASan/LSan to find leaks and fix them | ASan+LSan on `bin/vio` (6 rooms × 2 cfgs), on the `pyxivo` binding, and on the unit suite; plus valgrind massif, a direct pool census and a static ownership audit, because LSan alone sees none of the L2 class (§2) |
| Plan first, split into milestones, written to the notes dir | `notes-n-prompts/plan-memory.md` |
| Each milestone sufficiently tested; e2e evaluation when appropriate | Unit suite at every milestone (36 → 41 cases, 5 added); full 6-room × 2-config e2e at M0, M2, M3, M4, M5 |
| One git commit per milestone | 6 commits on `auto-memory`; M1 and M5 are tooling + notes, M6 is this file |
| `report-memory.md` in the notes dir | this file |
| Detailed notes under `notes-memory/` | 6 documents, one per milestone |
| Sub-agents allowed; worktrees allowed; deliver on `auto-memory` | 3 sub-agents for the parallel static audit (M1); one scratch worktree (`xivo-base`) for the baseline measurements, since removed; everything delivered on `auto-memory` |
| Worktree `xivo-memory` created from `auto` | done |
| Out of scope: new algorithmic features | Held. No new feature; the only new code is 5 unit-test cases and `scripts/mem/` tooling |
| Out of scope: tuning / improving performance | Held. No config default changed, no estimator parameter touched. `cfg/vio_tumvi_nodesc.json` was added, but it is the `bin/vio` wrapper for an *already existing* estimator config, added so the app and the python harness exercise the same settings |

Milestones, as delivered:

| | commit | content |
|---|---|---|
| M0 | `49e3bc8` | baseline (accuracy, RSS, unit tests) + the four instrumented build variants and the leak/RSS tooling |
| M1 | `e81dbdb` | heap-attribution tooling (massif profile + snapshot diff) and the 18-entry leak register |
| M2 | `fbee91e` | L1-1 … L1-5, the five definite leaks (and L3-1, same function as L1-3) |
| M3 | `2feb2eb` | L2-1 … L2-3, the unbounded growth (and L3-5, which L2-1's fix creates) |
| M4 | `3bd17c4` | the remaining eight L3 memory-safety faults |
| M5 | `631df4b` | the leak and no-regression gate: 12-run leak matrix, python-path gate, A/B throughput harness |
| M6 | this file | report |

**Deviation from the plan.** The plan's milestone structure survived intact.
Two things inside it did not:

* The plan's M3 gate was *"RSS slope over a full room is flat within noise."*
  That statistic turned out to be unusable at the size of these defects — a
  least-squares fit over a trace that swings ±4 MB, ranging from -1,417 to +259
  kB/s on runs that all end at the same RSS, and fitting a *positive* slope to a
  trace that ends 2.2 MB below where the window started. The gate was replaced
  with two statistics that are load-independent: the pool census, and peak RSS
  across a 6× run-length scan. `m3-unbounded-growth.md` §4 shows the numbers
  behind the rejection.
* The plan proposed a checked-in suppression file so that "zero leaks" means
  "zero XIVO leaks". That works for `bin/vio` (`scripts/mem/lsan.supp`, one
  justification per entry), but not for the python path: the interpreter's
  leaked blocks have `_PyEval_EvalFrameDefault` in their stacks, LSan
  suppressions match *any* frame, and every call into `pyxivo` goes through that
  frame too — so the suppression that silences CPython would silence XIVO. The
  python gate is by *attribution* instead: it reports the total and fails if any
  leaked block was allocated under `src/`, `common/` or `pybind11/`.

---

## 2. How the defects were found — and why LSan found none of them

Four methods, because no single one sees all three classes on this codebase:

| method | what it can see | what it produced |
|---|---|---|
| ASan+LSan on `out-asan/bin/vio`, 6 rooms × 2 cfgs, whole sequences | L1 definite leaks, L3 faults | **nothing at all** |
| valgrind massif, `--threshold=0.05`, snapshot diffing | L2 growth, attributed to a source line | the two descriptor sites and `traj_est` |
| direct pool census in `~CircBufWithHash` (temporary instrumentation in M1; a committed `pool_census.sh` afterwards) | L2 growth inside pooled objects, exactly counted | 9,059 retained descriptors, 2.72 MB pinned |
| static ownership audit, 3 sub-agents over disjoint file sets | L1/L3 in code the two configs never execute | 12 further defects |

The LSan blindness is the load-bearing fact of this task, so it was proved
rather than asserted. `REACHABLE=1` in `leakcheck.sh` sets
`use_globals=0:use_stacks=0:use_tls=0:use_registers=0`, which makes LSan report
everything still allocated. A 500-entry census and an 8,000-entry census came
back **identical** — same 161 kB in 1,052 allocations — while massif showed the
heap growing over the same span. That is the direct evidence for reason 2: by the
time LSan looks, the singleton destructors have already released the growth.

So the exit criterion "free of memory leaks" was decomposed into three
measurable statements, all of which are gated in M5:

1. zero LSan-reported leaks — the classic criterion, which `auto` already passed;
2. zero ASan memory-safety errors;
3. no growth in the heap while the filter runs on a bounded state.

Only (1) is what LSan measures. (3) is where the actual bug was.

### Two bugs in the project's own python leak-checker

`scripts/mem/leakcheck_py.sh` exists to check the path the evaluation harness
actually uses (the `pyxivo` extension, not `bin/vio`). It had two defects that
meant **it had never checked that path**:

* `scripts/pyxivo.py` does `sys.path.insert(0, 'lib')`, which beats
  `PYTHONPATH` — so exporting `PYTHONPATH=out-asan/lib` and running from the
  source root imported the **release** binding. It now runs in a scratch
  directory whose `lib` symlinks to `out-asan/lib`.
* Only `libasan.so` was preloaded. The `Estimator` constructor's jsoncpp parse
  throws, and ASan then aborts with
  `CHECK failed: asan_interceptors.cpp:470 "((real___cxa_throw)) != (0)"`.
  `libstdc++.so` must be preloaded next to `libasan.so`. Every previous
  invocation of the script died inside `Estimator::Estimator` before processing
  a frame.

Both are fixed in M5, and the python path is now genuinely gated.

---

## 3. The defects

Reachability column: `default` = `cfg/tumvi_cam0.json` only, `vio` = the
`bin/vio` app, `off-path` = real defect in compiled code that neither mono+IMU
config executes.

### 3.1 L2 — unbounded growth (the actual memory problem; M3)

| # | site | reach | evidence | fix |
|---|---|---|---|---|
| L2-1 | `src/feature.h:41-44` — `Track::Reset` does not clear `descriptors_` | default | census: 9,059 descriptors at exit, linear in run length; massif +72 kB per half-run at `SetDescriptor` | `descriptors_.clear()` in `Reset` |
| L2-2 | `src/tracker.cpp` (6 sites) store `descriptors.row(i)`, an OpenCV *view* | default | 490 pinned parent buffers = 2.72 MB; massif +175 kB per half-run at `DetectLK` | `.clone()` the row in `SetDescriptor` |
| L2-3 | `src/app/vio.cpp:65,109` — `traj_est` grows 96 B per dataset entry and is never read | vio | massif: 49 → 98 → 393 → 786 kB by entry 8,000; ~3.1 MB live, ~4.7 MB transient | delete both lines |

L2-1 set the count and L2-2 set the price — a retained 32-byte row keeps
110-270 keypoints × 32 B alive, a 100-250× amplification — which is why 9,059
rows pinned 2.72 MB across only 490 distinct buffers.

Fixing L2-1 *created* a fault: once `Reset` empties `descriptors_`, a `Track` can
genuinely have none, so `descriptor()`'s unguarded `descriptors_.back()` becomes
UB where it previously returned stale-but-valid data from a dead tenant. That is
L3-5, and it had to land in the same commit. `descriptor()` now `CHECK`s,
`has_descriptor()` was added, and the three unguarded callers
(`Tracker::GetDescriptors`, the dropped-track rescue, `tracked_features`) were
made honest. A third-party config with `match_dropped_tracks` on and
`extract_descriptor` off used to run straight into that UB; it now takes a
defined path.

### 3.2 L1 — definite leaks (M2)

All five are unmatched `new`s, all off the mono+IMU path — which is exactly why
LSan never reported them: the code is behind a build flag, behind a config flag
both configs turn off, or has no caller at all. They are real, so they are fixed
rather than suppressed, and **the fix in every case is to give the allocation an
owner, never to add a matching `delete`.**

| # | site | reach | what leaked | fix |
|---|---|---|---|---|
| L1-1 | `src/viewer.cpp:62,88` `SetHandler(new Handler3D)` | off-path (`visualize:false`) | `pangolin::View` stores a non-owning `Handler*`; `~Viewer` freed the render states but not the handlers | `unique_ptr` members + `SetHandler(nullptr)` on each view before teardown, because the views outlive `Viewer` in pangolin's global registry |
| L1-2 | `src/mapper.cpp:138,143`, empty `~Mapper` | off-path (`USE_MAPPER` undefined) | a 21,110-node `FastBriefVocabulary` and a `cvl::PnpParams` | `unique_ptr` members; `GetRANSACParams` returns `unique_ptr` |
| L1-3 | `src/fastbrief.cpp:22` `mean = new uint64_t[4]` | off-path | one array per cluster during training, nothing owns it | see below |
| L1-4 | `src/fastbrief.cpp:110` `FastBrief::fromString` | off-path | 32 B per vocabulary node on load — 675 kB per `Mapper` construction with the shipped vocabulary | see below |
| L1-5 | `common/utils.cpp:127` `auto writer = builder.newStreamWriter()` | off-path (`SaveJson` has no caller) | jsoncpp returns an owning raw pointer | `unique_ptr<Json::StreamWriter>` |

L1-3, L1-4 and L3-1 are one root cause: `typedef uint64_t *TDescriptor`. DBoW2's
`TemplatedVocabulary` stores one `TDescriptor` **by value** per node and never
frees it — for a generic value type there is nothing to free — so every
descriptor DBoW2 ever constructs leaks. The fix is to make the type a value
type, `std::array<uint64_t, 4>`: nothing can leak, both `new`s disappear, and
`memset` becomes `fill(0)`. Every use inside DBoW2 was already value-semantic, so
the template did not need touching. Two XIVO call sites did: `GetDBoWDesc` /
`GetAllDBoWDesc` used to *cast* a pooled `Feature`'s `cv::Mat` data to
`uint64_t*` and hand the vocabulary a borrowed pointer into memory a recycled
slot will overwrite — a latent dangling read, now a 32-byte copy.

### 3.3 L3 — memory-safety faults (M4, plus L3-1 in M2 and L3-5 in M3)

| # | site | reach | fault | fix |
|---|---|---|---|---|
| L3-1 | `src/fastbrief.cpp:23` `memset(&mean, 0, 32)` | off-path | `&mean` is the address of the *reference* — 32 bytes written over the caller's 8-byte pointer, nulling it, so `mean[i>>6] \|= …` is a null write | value type (§3.2) |
| L3-2 | `src/estimator_accessors.cpp`, 9 sites | off-path (the `n_output` pybind overloads) | matrix sized `max(size, n_output)` but filled to `min(…)`; Eigen does not zero-initialise, so uninitialised heap is copied into numpy. The correct form is in the same file at `:318` | `std::min` in all nine |
| L3-3 | `src/estimator.cpp:1454,1461` | off-path | `return score1 <= score2` breaks `std::sort`'s strict-weak-ordering precondition; libstdc++'s unguarded partition can walk before `begin()` on ties | `<` |
| L3-4 | `pybind11/pyxivo.cpp:106,144` numpy-buffer `VisualMeas` | off-path (shipped scripts pass a path) | (a) channel count assumed 3 and geometry derived from strides, so a 2-D grayscale array is read ~2× past its end; (b) the `Mat` does not own `info.ptr` and `MaintainBuffer` executes `buf_.front()` — an *earlier* message — so the frame is read ≥10 messages later, after the numpy buffer may be gone | derive shape/channels from `info.shape`/`info.ndim`; `.clone()` before enqueueing |
| L3-5 | `src/feature.h:50-51`, `src/tracker.cpp:781-782` | off-path (until L2-1 is fixed) | `back()` on an empty vector; interacts with L2-1 as described in §3.1 | `CHECK` + `has_descriptor()`, honest callers |
| L3-6 | `src/metrics.cpp:66-74` `ComputeRPE` | off-path (`evaluate.cpp` not built) | `it_est` incremented then dereferenced without re-checking `end()` | re-check; regression test `MetricsRPE` |
| L3-7 | `src/estimator.h:165-166` `CameraCov()` | off-path (`USE_ONLINE_CAMERA_CALIB`) | despite the name `all_zeros`, 648 B of indeterminate data returned | `::Zero()` |
| L3-8 | `src/update.cpp:403-407` | off-path (`use_1pt_RANSAC:false`) | iterates an `active_features` snapshot taken *before* `DestroyFeatures`; `sind()` is then -1 and `ComputeJacobian` indexes at a negative offset | erase the destroyed set from the snapshot |
| L3-9 | `src/graph.h:126` `last_added_group_` | off-path (`USE_MAPPER`) | never initialised, never cleared by `RemoveGroup` | `{nullptr}` + clear in `RemoveGroup` |
| L3-10 | `src/estimator.cpp:1319-1323` `gauge_group_ptr_` | off-path (`use_1pt_RANSAC:false`) | survives `RemoveGroup` + `Deactivate`; only ever compared by identity | null it alongside `gauge_group_` |

Two sub-agent claims had to be adjudicated rather than accepted. The threading
agent called L3-4 a use-after-free; the front-end agent called it safe because
`VisualMeas` is synchronous. Reading `MaintainBuffer` settles it for the first:
even with `async_run: false` the just-pushed message goes into a heap and
`buf_.front()` — a different, earlier message — is what executes. And the
front-end agent's first estimate of the descriptor leak (~20 MB) was corrected to
single-digit MB after measurement; the census agrees with the corrected figure.

---

## 4. Evidence

### 4.1 The leak gate

| check | result |
|---|---|
| `leakcheck_matrix.sh` — 6 rooms × 2 cfgs, whole sequences, `out-asan/bin/vio` | **12/12 exit 0, `report.txt` empty**, ~170,000 instrumented dataset entries |
| `leakcheck_py.sh` — the `pyxivo` path, room1, both cfgs | 628,973 B in 531 objects, **0 frames in XIVO sources** |
| `pybind_buffer_check.sh` — the L3-4 reproduction, grayscale and colour layouts | pass |
| unit suite under ASan (`ctest` + gtest binaries) | 39 pass, 2 pre-existing failures, **0 sanitizer findings** |

Exit status is the gate, not the log text: ASan exits **23** when LSan reports a
leak, so a 0 exit *is* the leak criterion, and an empty `report.txt` says there
was no other finding (overflow, UAF) either.

The 628,973 B on the python path is 100% CPython and numpy module
initialisation — `PyType_GenericAlloc`, `PyDict_Copy`,
`_multiarray_umath_exec`, `PyUFunc_AddLoop`, the cython `__pyx_pymod_exec_*`
initialisers. The cross-check that it is start-up state and not XIVO's is that
the byte count is *identical* for two configs that make the filter behave
differently. It is deliberately not suppressed (§1, deviation 2); the gate is
attribution.

### 4.2 Bounded memory — the class LSan cannot see

Three independent metrics, since the defect class is invisible to the sanitizer:

**Pool census** (`pool_census.sh`, what the 200 pooled `Feature`s still hold when
`~MemoryManager` runs, room1):

| entries | retained descriptors | max per slot | pinned bytes |
|---|---|---|---|
| 2,000 | 151 | 1 | 4,832 |
| 8,000 | 200 | 1 | 6,400 |
| whole (30,943) | **200** | **1** | **6,400** |
| whole, `nodesc` | 0 | 0 | 0 |

Saturating at exactly one descriptor per slot, and not moving between 8,000 and
30,943 entries, is what "bounded" means in the metric the defect was found in
(pre-fix: 9,059 / 73 / 2,717,792). The 2,000-entry row reproduces M1's 151
exactly, which is the evidence that the pre- and post-fix columns are the same
measurement.

**massif against an independently built `auto` baseline.** M1↔M5 profiles of the
same tree appeared to say the fixes traded retention for allocator churn (43.1 GB
of traffic post-fix against M1's 32.4 GB). That comparison is invalid — M1's
profiles predate M3's fix to `massif_profile.sh`'s arguments and its pinned
snapshot flags, so they are not diffable. The valid comparison is same-tool,
same-flags, two binaries, so M5 built one: `git worktree add xivo-base auto`,
with the compile flags verified byte-identical
(`git diff auto..auto-memory -- CMakeLists.txt`). room1, 8,000 entries:

| | `auto` | HEAD |
|---|---|---|
| total allocation traffic | 42.98 GB | **42.62 GB (-0.8%)** |
| peak total heap | 33.6 MB | 32.2 MB |
| steady-state heap | 26.4-26.9 MB | 25.0-25.2 MB |

So the `.clone()` costs no extra traffic — slightly less, consistent with
`traj_est`'s geometric reallocations being gone — and **the churn concern is
retracted with a measurement.** Per-site attribution over the same span:
`traj_est` +393 kB → site gone; `SetDescriptor` +189 kB → net 0;
`DetectLK`→BRIEF→`fastMalloc` +414 kB → not retained. The one site that still
grows is the `Track` pixel-history vectors (+80 kB), bounded by the number of
live tracks rather than by run length — the growth-by-design the register
recorded.

**Peak RSS does not depend on run length** (`rss_profile.sh`, `/proc` sampling):
78.6 / 78.8 / 80.4 / 78.8 MB at 5k / 10k / 20k / all entries (default), and
75.2 / 75.5 / 76.1 / 76.5 MB (nodesc). A 6× longer run costs ≤1.8 MB, inside the
±4 MB transient band. Against M0 the peak is down 6-7 MB (default) and 3-4 MB
(nodesc) — matching L2-3's 3.1 MB plus the 2.7 MB of descriptors, and L2-3 alone
for the config that extracts no descriptors.

Measurement caveat found here and worth carrying: **`/usr/bin/time -v`'s
"Maximum resident set size" reads 61 MB for a run whose peak the kernel itself
puts at 80 MB** (`VmHWM`, which agrees with the sampled `VmRSS` peak to 0.5 MB).
It is usable as a relative A/B number and not as a process peak; every absolute
figure above comes from `/proc`.

### 4.3 No accuracy regression — twice over

**(a) Against the M0 baseline**, both configs, all six rooms, `XIVO_RANDOM_SEED=0`:

| config | ATE (mean over 6) | RPE_rot | RPE_tra | vs M0 |
|---|---|---|---|---|
| `tumvi_cam0` | 0.1409 | 0.6219 | 0.0352 | identical=20 differ=6 |
| `sweep_dlt_nodesc` | 0.1267 | 0.6226 | 0.0364 | identical=20 differ=6 |

Per sequence, `tumvi_cam0`: 0.1117 / 0.1106 / 0.2298 / 0.1111 / 0.2113 / 0.0709.
`sweep_dlt_nodesc`: 0.1219 / 0.1199 / 0.2165 / 0.0879 / 0.1240 / 0.0897. The six
differing files per config are the `run_room*.log` harness logs, and the only
difference in them is the output directory in the command lines the harness
echoes — they contain no timings. **Every trajectory and every metrics file is
byte-identical.**

**(b) Against a binary built from `auto`** — the stronger form, since it does not
depend on M0's artifacts being what they claim: `xivo-base/bin/vio` (built from
`auto`, same flags) and `xivo-memory/bin/vio` on room1 with `cfg/vio_tumvi.json`
produce byte-identical 30,943-line trajectory files.

Determinism is what makes this gate possible at all: with `XIVO_RANDOM_SEED=0`,
re-running a sequence reproduces its trajectory byte for byte, so the
no-regression criterion is "identical output", not "ATE within noise".

### 4.4 No throughput regression

`throughput_ab.sh` alternates the two binaries A,B,A,B… over whole room1,
4 reps each, on an idle machine:

| config | build | wall (mean of 4) | min-max | user CPU |
|---|---|---|---|---|
| `vio_tumvi` | `auto` | 32.02 s | 31.67-32.54 | 108.94 s |
| `vio_tumvi` | HEAD | 32.18 s (+0.5%) | 31.79-32.74 | 109.14 s |
| `vio_tumvi_nodesc` | `auto` | 30.46 s | 30.21-30.80 | 100.44 s |
| `vio_tumvi_nodesc` | HEAD | 30.79 s (+1.1%) | 29.85-31.47 | 100.68 s |

Both deltas are smaller than each build's own min-max spread (0.9-1.6 s), and
the CPU-time means agree to 0.2%. `vio` runs at >1000% CPU because of OpenCV's
internal threading, so single-run user/sys readings swing ~15% — an early
single-shot pair suggested +34% system time, which four alternating reps show to
be noise. Minor page faults swing 27k-574k on nominally identical runs and no
conclusion is drawn from them.

### 4.5 Tests

| build | ctest targets | gtest cases | sanitizer findings |
|---|---|---|---|
| release | 7/9 pass | 39 pass, 2 fail | n/a |
| ASan | 7/9 pass | 39 pass, 2 fail | **0** |

The two failures — `NumericalLinearAlgebra.SlowAndFastGivensMatch` and
`Triangulation.Angular_Reprojection_Error` — are pre-existing on `auto` and
recorded in the M0 notes. The pass count is up from M0's 34 because M3 added
`DescriptorMemory` (3 cases) and M4 added `MetricsRPE` (2).

`DescriptorMemory` is the regression test for the milestone that matters, and it
tests the invariant directly without a sanitizer or the dataset:
`Reset` empties `descriptors_`; a stored descriptor does not share the source
matrix's `UMatData` (`stored.u != block.u`); and a 4-slot pool run through 100
create/`SetDescriptor`/`Deactivate` cycles retains exactly one descriptor per
slot (pre-fix it reaches 25). Verified the only way that means anything: with the
two fixes reverted all three fail, restored 3/3 pass.

Along the way, `add_test` had no `WORKING_DIRECTORY`, so 7 of 8 registered tests
failed under `ctest` for everyone — they open fixtures by relative path. Fixed in
M3, which is what makes the M5 test gate possible.

---

## 5. Did the exit criteria get met?

**"The code is free of memory leaks."** Yes, under the three-part definition of
§2, and this is the claim that needs its limits stated rather than a bare yes:

* Zero LSan-reported leaks over 12 whole sequences × 2 configs and the unit
  suite, and zero on the python path after attribution. **But** a clean LSan
  report reads the same on `auto` as it does here, so on its own it is not
  evidence of anything (§2).
* Zero ASan findings over ~170,000 instrumented dataset entries.
* Retention is bounded: 200 descriptors and 6,400 pinned bytes regardless of run
  length, flat peak RSS across a 6× range, and every growth site massif
  attributed in M1 either gone or net zero.
* All 18 register entries are fixed, including the 15 that neither mono+IMU
  config reaches. Those are argued from ownership analysis and reachability, not
  from a sanitizer trace, because no run can produce one.

Not claimed: that L3-2 and L3-7 (reads of uninitialised memory) are verified by a
sanitizer — they are MSan-class, and MSan needs an instrumented
libstdc++/OpenCV/Eigen stack; nor that L3-4's aliasing fault is
sanitizer-visible, since the freed numpy block is recycled and the symptom is
wrong pixels rather than a report. Its grayscale over-read *is* reproducible and
was reproduced. Nor that the python path leaks zero bytes: it leaks 628,973 B of
interpreter and numpy state, none of it XIVO's, and a future python will change
that number.

**"Performance … does not regress."** Yes, on both readings:

* Accuracy: byte-identical trajectories, twice over (§4.3). This is the strongest
  form the gate can take.
* Throughput: +0.5% / +1.1% wall clock, inside each build's own run-to-run
  spread, with CPU-time means agreeing to 0.2% (§4.4). Measured on one sequence
  on one machine — it is a ±2% statement, not a benchmark.
* Memory: peak RSS is 6-7 MB (default) and 3-4 MB (nodesc) *lower* than the
  baseline, and total allocation traffic is 0.8% lower.

---

## 6. What was deliberately not fixed

Recorded in the register with reasons, because fixing them is either a design
change or a behavioural change, and both are out of scope for a memory task:

* **T-1 / T-2 — worker threads that cannot be joined.** `Process::Run`'s body and
  `Estimator`'s async lambda are `for(;;)` with no stop condition, so
  `~Process()` / `~Estimator()` would *hang* on `join()` rather than leak. The
  queued messages (up to 1,000 `unique_ptr<EstimatorMessage>`, each owning a
  `cv::Mat`) are then never destroyed. Both are off-path today (`async_run:
  false` in both configs; `Process::Start`'s only caller is an unbuilt legacy
  app). Fixing needs an `atomic<bool>` stop flag and a drain policy — new
  behaviour, not an owner for an allocation.
* **T-3 — the last ≤10 buffered measurements are never executed.** Memory is
  bounded and *is* freed, so it is a correctness bug, not a leak. Draining the
  buffer would process ten more measurements and change every trajectory.
* **Growth that is by design or plateaus.** The `MemoryManager` pool itself
  (200 features × `sizeof(Feature)` + 100 groups, `new`ed up front by
  `CircBufWithHash` and freed in its destructor — pre-allocation is the point of
  the class); `Track` / `FeatureAdj` / `GroupAdj` capacity, which rises to the
  worst size a slot ever held and then stops (~0.5 MB total); `Viewer::trace_`,
  which is the feature's own data; `Estimator::ids_to_depths_` and
  `Mapper::InvIndex_`, genuinely unbounded but behind `simulation: true` /
  `USE_MAPPER` — fixing them means choosing a retention policy, which is a design
  decision; `DataLoader::entries_`, a load-once 2.7 MB index.
* **Eight non-memory defects found on the way** are listed in the register
  (`m1-leak-register.md`, "Non-memory defects found on the way") — a wrong-index
  drop in the descriptor-distance path, re-dropped rescued tracks, a
  `static int half_size` that caches the first mask size forever, an
  `inlier_outlier_mask` read after `findHomography` fails, a UB shift count in
  `fastbrief.cpp`, a `DestroyItem`/`GetItem` slot-accounting asymmetry in
  `mm.cpp`, and a declared-but-undefined `Delete()`. All dead under the shipped
  configs; all out of scope here.

---

## 7. If this continued

* **MSan.** L3-2 and L3-7 are uninitialised-memory reads that no available tool
  here confirms. An MSan build needs an instrumented libstdc++, OpenCV and Eigen;
  that is a day of dependency work and would close the one evidence gap in the
  L3 register.
* **The threading lifecycle (T-1/T-2).** A stop flag plus a documented drain
  policy would make `async_run: true` a supported mode instead of a mode whose
  teardown hangs. It needs its own accuracy gate, since draining changes output.
* **A CI gate.** `leakcheck_matrix.sh` and `pool_census.sh` are the two checks
  that would have caught this defect class at the commit that introduced it. The
  census is cheap (one short run); the matrix is not (ASan is ~30× slower), so a
  nightly matrix and a per-commit census is the sensible split.
* **The RSS-slope statistic should be retired from the project's vocabulary.**
  It is quoted in the M0 baseline and it cannot resolve a 2.7 MB defect against a
  ±4 MB transient band. Peak-vs-run-length is the statistic that works.

---

## 8. Artifacts

| what | where |
|---|---|
| delivered branch | `auto-memory` in the xivo package (worktree `xivo-memory`), 6 commits, HEAD `631df4b` |
| plan | `notes-n-prompts/plan-memory.md` |
| milestone notes | `notes-n-prompts/notes-memory/m{0..5}-*.md` |
| leak register (18 entries + the not-fixed lists) | `notes-memory/m1-leak-register.md` |
| build variants | `scripts/mem/build.sh {release,asan,asan-ub,valgrind}` → `bin/`, `out-asan/`, `out-valgrind/` |
| leak checks | `scripts/mem/leakcheck.sh`, `leakcheck_matrix.sh`, `leakcheck_py.sh`, `lsan.supp`, `leak_summary.py` |
| growth measurement | `scripts/mem/massif_profile.sh`, `massif_diff.py`, `pool_census.{sh,py}`, `rss_profile.sh` |
| throughput A/B | `scripts/mem/throughput_ab.sh` |
| L3-4 reproduction | `scripts/mem/pybind_buffer_check.{sh,py}` |
| regression tests | `src/test/unittest_descriptor_memory.cpp` (`DescriptorMemory`), `src/test/unittest_metrics_rpe.cpp` (`MetricsRPE`) |
| e2e results | `results/memory/m0_baseline_{default,dltnodesc}`, `m2_*`, `m3_*`, `m4_*`, `m5v_{default,dltnodesc}` |

`results/m5_final/` at the workspace root is **not** from this task — its copied
config differs from the tree's in `use_prediction: true`, i.e. it is a tuning
experiment from another line of work. None of the numbers above come from it.
