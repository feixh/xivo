# Plan — find and fix memory leaks with ASan/LSan (mono + IMU)

Scope: the monocular-camera + IMU path of the `xivo` package, delivered on branch
`auto-memory` in worktree `/home/ubuntu/workspace/auto-slam-engineer/xivo-memory`
(branched from `auto`, commit `888511d`).

Out of scope (per the requirements): new algorithmic features, and tuning the
system for accuracy. The exit criteria are *no leaks* and *no ATE/RPE
regression* — accuracy must stay put, not improve.

## What "memory leak" means here

Three distinct defect classes, because they need different detectors and get
different fixes. All three are in scope: a run that ends with a clean LSan
report can still grow without bound while it runs.

| Class | Symptom | Detector |
|---|---|---|
| **L1 definite leak** | heap block allocated, never freed, no live pointer at exit | LeakSanitizer (`detect_leaks=1`) |
| **L2 unbounded growth / still-reachable** | container or pool slot that accumulates for the whole run and is only released at process teardown (LSan stays silent) | RSS-vs-time sampling + per-object growth assertions |
| **L3 memory-safety fault** | invalid read/write, use-after-free, double-free, UB such as `back()` on an empty vector | AddressSanitizer, plus `_GLIBCXX_ASSERTIONS` |

XIVO deliberately pre-allocates `Feature`/`Group` in a fixed pool
(`MemoryManager` + `CircBufWithHash`), so classic per-frame `new` leaks are
unlikely in the filter core; the interesting cases are L2 (recycled objects that
are not fully reset) and L1 in the singleton/thirdparty glue. The plan therefore
does **not** rely on LSan alone.

## Detection strategy

1. **ASan+LSan instrumented build** of the whole tree
   (`-fsanitize=address -fno-omit-frame-pointer -O1 -g`), producing
   * `bin/vio` — the standalone C++ mono+IMU app. This is the primary leak
     target: a real `main()` that runs a full TUM-VI room, so LSan's report is
     about XIVO and not about a Python interpreter.
   * `lib/pyxivo*.so` — the binding the evaluation harness actually uses. Run
     under `LD_PRELOAD=libasan.so` with a suppression file for CPython's own
     one-time allocations, so the leak numbers cover the path that produces
     RESULTS.md.
   * the seven `unitTests_*` binaries — cheap, fast ASan coverage of the
     numeric core and the camera models.
2. **Heap-growth profiling.** Sample `/proc/self/statm` RSS (and
   `mallinfo2` arena/uordblks) every N frames of a full room and fit a slope.
   A flat line after the pool warms up is the pass condition; a positive slope
   localises an L2 leak even when LSan is silent. Runs on the *uninstrumented*
   Release build so the numbers are the ones a user would see.
3. **Static ownership audit.** ~15 kLOC over `src/` + `common/` is small enough
   to read exhaustively. Split by subsystem and audited in parallel by
   sub-agents, each reporting `file:line`, the owning/leaking party, whether the
   mono+IMU path reaches it, and a minimal fix. Every finding is re-verified by
   hand (and, where possible, by a sanitizer run) before any code changes.
   Targeted questions the audit must answer: every `new` without a matching
   `delete`; every `Reset()` against its object's full member list (does
   recycling clear everything?); every singleton's teardown path; every raw
   pointer handed across a thread boundary.
4. **Valgrind memcheck** (if installable) on a short slice, as an independent
   second opinion on ASan's findings — different instrumentation, different
   blind spots.

## Milestones

Each milestone is one git commit on `auto-memory`, with notes in
`notes-n-prompts/notes-memory/`.

| # | Milestone | Deliverable | Gate |
|---|---|---|---|
| M0 | Baseline + harness | worktree built; 6-room ATE/RPE and peak-RSS baseline for the default (`tumvi_cam0`) and best (`sweep_dlt_nodesc`) mono configs; `tools/` scripts for the ASan build, the leak run and the RSS profile | baseline ATE reproduces the `auto` branch numbers in RESULTS.md |
| M1 | Leak census | ASan/LSan builds green-to-run; LSan reports for `vio` (all 6 rooms), `pyxivo`, and the unit tests; RSS-growth traces; audited leak register | every entry has `file:line`, a class (L1/L2/L3), evidence (sanitizer output or a reasoned static argument), and a proposed fix |
| M2 | Fix L1 definite leaks | raw `new` without `delete`, singleton/thirdparty teardown, `FastBrief` descriptor arrays, `Mapper` vocabulary + RANSAC params, thread lifecycle (`Process`/`Estimator` workers that can never be joined) | LSan report for `vio` drops to zero XIVO-attributed leaks; unit tests still pass |
| M3 | Fix L2 unbounded growth | objects recycled through `MemoryManager` that are not fully reset (`Track::descriptors_` is the known one), plus any accumulating graph/tracker container found in M1 | RSS slope over a full room is flat within noise; a new unit test asserts the invariant directly |
| M4 | Fix L3 memory-safety faults | ASan-reported invalid accesses and UB on the mono+IMU path | ASan clean across all 6 rooms; regression test per fault |
| M5 | Verification + regression | full re-run: ASan/LSan on 6 rooms, unit suite under ASan, RSS profile, and a Release e2e on both configs compared against M0 | zero leaks; ATE/RPE per sequence identical (or within determinism noise) to M0 |
| M6 | Report | `notes-n-prompts/report-memory.md` | — |

## Subsystem split for the audit (M1)

| Area | Files |
|---|---|
| A Pool & lifecycle | `mm.{h,cpp}`, `feature.{h,cpp}`, `group.{h,cpp}`, `graph.{h,cpp}`, `graphbase.cpp`, `component.h` |
| B Estimator & threading | `estimator.cpp`, `estimator_process.cpp`, `estimator_accessors.cpp`, `common/process.h`, `common/ProducerConsumerQueue.h` |
| C Front end | `tracker.{h,cpp}`, `manager.cpp`, `fastbrief.{h,cpp}`, `visualize.cpp` |
| D Singletons & I/O | `factory.cpp`, `param.cpp`, `camera_manager.cpp`, `loader.cpp`, `options.cpp`, `publisher.cpp`, `graphwriter.cpp`, `viewer.cpp` |
| E Mapper / optional paths | `mapper.{h,cpp}`, `optimizer*` , `oos.cpp`, `mm.cpp` mapper hooks (compiled-out today, but `fastbrief.cpp` and `mapper.cpp` are in `libxest`) |

## Known-suspicious sites (to confirm with evidence, not assume)

Spotted while sizing the task; each must be proven with a sanitizer run or a
line-by-line argument before it is called a bug.

1. `Track::Reset()` clears the point history but not `descriptors_`
   (`src/feature.h:38-42` vs `:64`) — a pooled `Feature` slot would accumulate
   one `cv::Mat` per observation *for the whole process lifetime*, across every
   reuse of the slot. Class L2.
2. `FastBrief::meanValue` does `mean = new uint64_t[4]; memset(&mean, ...)` —
   memsets the pointer variable, not the array, and overwrites the caller's
   pointer each call (`src/fastbrief.cpp:22-23`). Class L1 + L3.
   `FastBrief::fromString` (`:110`) has the same `new` without an owner.
3. `Mapper::~Mapper()` is empty while the constructor `new`s a
   `FastBriefVocabulary` and a `cvl::PnpParams` (`src/mapper.cpp:138,155,58`).
   Class L1 (mapper is `#ifdef USE_MAPPER`-gated today — confirm reachability).
4. `Process::~Process()` and `Estimator::~Estimator()` `join()` a worker whose
   body is `for(;;)` with no exit condition (`common/process.h:36-40`,
   `src/estimator.cpp:95-98`, `:420`) — teardown hangs rather than leaks, and
   the queued messages are then never released. Reachable only with
   `async_run: true`; confirm what the mono configs set.
5. `Viewer` `new`s Pangolin handlers that nothing deletes
   (`src/viewer.cpp:62,88`); `Graph` owns `random_device`/`mt19937` by raw
   pointer (`src/graph.h:109-118`).
6. `Track::descriptor()` returns `descriptors_.back()` with no emptiness check
   (`src/feature.h:50-51`) — UB if a descriptor was never set. Class L3.

## Risks and how they are handled

- **A leak fix that changes numbers.** Clearing state that was previously
  carried over between pool reuses can change the trajectory. Every fix is
  followed by a Release e2e; if ATE moves, the fix is re-examined and the
  behavioural difference is explained in the notes before it is kept. Accuracy
  tuning stays out of scope either way.
- **Determinism.** `XIVO_RANDOM_SEED=0` is mandatory for any comparison;
  without it `Graph()` seeds `mt19937` from `std::random_device` and every run
  differs.
- **ASan changes the allocator.** Timing- or address-dependent behaviour can
  differ from Release, so any *numeric* claim is made on the Release build, and
  the sanitizer builds are used only to find defects.
- **Third-party noise.** OpenCV, Pangolin, glog, jsoncpp and CPython all have
  one-time allocations that LSan will list. These go into a checked-in
  suppression file with a one-line justification each, so "zero leaks" means
  zero *XIVO* leaks and the suppressions stay auditable.
- **Pool semantics are a feature, not a leak.** The 256 `Feature` / 128 `Group`
  slots are allocated up front and freed at teardown by design. The fix for a
  pool problem is correct recycling, not per-frame `new`/`delete`.
