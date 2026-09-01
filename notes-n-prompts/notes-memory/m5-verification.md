# M5 — verification and no-regression

M5 adds no fixes. Its job is to decide whether the two exit criteria hold at
HEAD (`3bd17c4`, the M4 commit) for the mono+IMU setting:

1. *the code is free of memory leaks* — and, because LSan is blind to most of
   what was actually wrong here, that has to be argued in more than one metric;
2. *performance does not regress* — both accuracy and throughput.

Everything below was measured from a release build made from the committed
tree (`scripts/mem/build.sh release`, no compiler warnings; only the
pre-existing `cmake_minimum_required` deprecation notes), and an ASan build from
the same tree (`scripts/mem/build.sh asan`).

## 1. Leak gate — ASan/LSan over the whole evaluation matrix

`scripts/mem/leakcheck_matrix.sh` (new here) runs `leakcheck.sh` for all six
TUM-VI rooms × both mono+IMU configs, whole sequences, `bin/vio` from
`out-asan`:

```
cfg                seq     exit  report    findings
vio_tumvi          room1..room6   0    0 bytes   0
vio_tumvi_nodesc   room1..room6   0    0 bytes   0
all runs clean: exit 0, no sanitizer findings
```

12/12 runs: exit status 0 and an empty `report.txt`. Exit status is the point —
ASan exits **23** when LSan reports a leak, so a 0 exit is itself the leak gate,
and `report.txt` being zero bytes says there was no other sanitizer finding
(overflow, UAF) over ~170,000 dataset entries of instrumented execution.

Pitfall worth recording, because it made the first matrix run look like a
failure: `grep -c` prints `0` **and** exits 1 when there is no match, so the
obvious `$(grep -c … || echo 0)` appends a second line and every clean run reads
as non-zero. The script takes `| tail -1` instead; the comment in it says why.

## 2. Leak gate — the python path, and two bugs in the tool that checked it

The evaluation harness does not run `bin/vio`; it runs `scripts/pyxivo.py`
against the `pyxivo` extension module. `scripts/mem/leakcheck_py.sh` exists for
that path since M0, and it had two defects that meant **it had never actually
checked it**:

* `scripts/pyxivo.py` does `sys.path.insert(0, 'lib')`, which beats
  `PYTHONPATH`. The script exported `PYTHONPATH=out-asan/lib` and then ran from
  the source root, so python imported the *release* binding. (The ASan runtime
  was still preloaded, so allocations were tracked — the check was not
  meaningless, just not instrumented.) It now runs in a scratch directory whose
  `lib` is a symlink to `out-asan/lib` and whose other entries symlink back into
  the tree.
* Only `libasan.so` was preloaded. The `Estimator` constructor parses its config
  with jsoncpp, which *throws* (`Json::throwLogicError` from `asDouble` on a key
  read with the wrong type), and ASan then aborts:
  `CHECK failed: asan_interceptors.cpp:470 "((real___cxa_throw)) != (0)"`.
  `libstdc++.so` has to be preloaded next to `libasan.so`. So every previous
  invocation of this script died in `Estimator::Estimator` before processing a
  frame.

With both fixed, room1 completes under ASan/LSan for both configs:

| cfg | LSan total | attribution |
| --- | --- | --- |
| `tumvi_cam0` | 628,973 B in 531 objects | 100% CPython + numpy; **0 frames in XIVO sources** |
| `sweep_dlt_nodesc` | 628,973 B in 531 objects | same |

`scripts/mem/leak_summary.py` puts all of it under `/usr/bin/python3.14`
(`PyType_GenericAlloc`, `PyDict_Copy`, `PyMem_Malloc`, `_PyEval_EvalFrameDefault`)
and numpy's module init (`_multiarray_umath_exec`, `PyUFunc_AddLoop`,
`PyArray_AddCastingImplementation_FromSpec`, the cython `__pyx_pymod_exec_*`
initialisers). The byte count being *identical* for two different configs is the
cross-check: it is interpreter and module state, fixed at start-up, independent
of what the filter did.

These are deliberately **not** suppressed. LSan suppressions match any frame in
the stack, and the largest group's stack goes through
`_PyEval_EvalFrameDefault` — which every call into `pyxivo` also goes through,
so suppressing it would mask XIVO's own leaks too. The script instead prints the
attribution and gates on "no leaked block was allocated in `src/`, `common/` or
`pybind11/`", which is the question that matters.

`scripts/mem/pybind_buffer_check.sh` (the M4 L3-4 reproduction) also passes at
HEAD under ASan for both the grayscale and colour numpy-buffer layouts.

## 3. The LSan-blind class, metric 1: the pool census

`scripts/mem/pool_census.sh` at HEAD, room1, counting what the 200 pooled
`Feature`s still hold when `~MemoryManager` runs:

| entries | retained descriptors | max per slot | pinned bytes |
|---|---|---|---|
| 2,000 | 151 | 1 | 4,832 |
| 8,000 | 200 | 1 | 6,400 |
| whole (30,943) | 200 | 1 | 6,400 |
| whole, `nodesc` | 0 | 0 | 0 |

Saturating at exactly one descriptor per slot, and not moving between 8,000
entries and 30,943, is what bounded means in the metric M1 found the defect in
(pre-fix: 9,059 headers, max 73 per slot, 2,717,792 pinned bytes over the whole
sequence — `m3-unbounded-growth.md` has the full table). The 2,000-entry row
reproduces M1's 151 exactly, which is the evidence that the two columns are the
same measurement.

## 4. The LSan-blind class, metric 2: massif against an independently built baseline

M1's and M3's massif profiles were compared across milestones of the *same*
tree. That leaves one thing unchecked: whether the fixes trade retention for
allocator churn. The M1→M5 numbers appeared to say they do — 43.1 GB of total
allocation traffic post-fix against M1's 32.4 GB for the same 8,000-entry run —
which would be a throughput risk even though accuracy is bit-identical.

That comparison turns out not to be valid: M1's profiles were taken with a
hand-written command line, before M3 fixed `massif_profile.sh`'s arguments and
pinned its snapshot flags, so pre-M3 files are not diffable against post-M3 ones.
The comparison that *is* valid is same-tool, same-flags, two binaries — so M5
built one.

**The baseline build.** `git worktree add xivo-base auto` (`888511d`), then:
`thirdparty/` symlinked to the delivery tree's already-built thirdparty (a fresh
worktree has the pristine sources but none of the headers/libs `build_all.sh`
installs, so glog/Eigen are simply missing); `cfg/vio_tumvi_nodesc.json` copied
in (added on the memory branch in M0); and, for the massif run only, the
`-max_entries` flag back-ported by hand — it is an M0 addition, and without it
the baseline cannot be profiled over the same 8,000 entries. The back-port is
four lines that bound the loop; `traj_est` (the L2-3 defect) is left in place.
`git diff auto..auto-memory -- CMakeLists.txt` confirms M0 kept the compile flags
byte-identical (`-O0 -std=c++17 … -funroll-loops`, `XIVO_ARCH_FLAGS` defaulting
to `-mtune=native -march=native`), so the two trees compile the same way; the
valgrind variant switches both to `-march=x86-64-v3` for the same reason
(`massif_profile.sh` documents the AVX-512 SIGILL).

room1, `vio_tumvi`, `-max_entries 8000`, `--time-unit=B`:

| | `auto` baseline | HEAD |
| --- | --- | --- |
| total allocation traffic | 42.98 GB | 42.62 GB (**-0.8%**) |
| peak total heap | 33.6 MB | 32.2 MB |
| steady-state heap (non-peak snapshots) | 26.4-26.9 MB | 25.0-25.2 MB |

So the fixes cost *no* extra allocation traffic — slightly less, consistent with
`traj_est`'s geometric reallocations being gone — and the earlier 43.1-vs-32.4
reading was an artifact of comparing across tool versions. This is also the
measurement that answers the question I raised while writing M4's notes, and it
answers it against the baseline rather than against an older profile of the same
tree.

Attribution over the same span (`massif_diff.py <early> <late>`), the growing
sites:

| site | baseline | HEAD |
| --- | --- | --- |
| `main (vio.cpp:110)` — `traj_est` | +393,216 B (393 kB → 786 kB) | site does not exist |
| `SetDescriptor (feature.h)` | +189,312 B | +3,744 / -3,744 B (net 0) |
| `DetectLK` → Brief `compute` → `fastMalloc` | +414,352 B | not retained |
| whole-heap drift over the span | -4.57 MB (dominated by a transient pyramid peak in the early snapshot) | +0.12 MB |

The one site that still grows at HEAD is `UpdateTrack (feature.h:109)` — the
`Track` pixel-history vectors, +80,544 B — which is the growth-by-design the
register recorded: bounded by the number of live tracks, not by run length.

## 5. Peak RSS does not depend on run length

`scripts/mem/rss_profile.sh` at HEAD, room1, `-max_entries` scan:

| entries | `vio_tumvi` peak | `vio_tumvi_nodesc` peak |
|---|---|---|
| 5,000 | 78.6 MB | 75.2 MB |
| 10,000 | 78.8 MB | 75.5 MB |
| 20,000 | 80.4 MB | 76.1 MB |
| whole (30,943) | 78.8 MB | 76.5 MB |

A 6× longer run costs ≤1.8 MB, inside the ±4 MB transient band that M3 measured
for this statistic. Against M0 (86.0 MB default, 80.0 MB nodesc) the peak is
down 6-7 MB and 3-4 MB respectively. The second-half RSS *slope* is still not a
usable statistic (0.0 to +190.9 kB/s across these runs, on traces that all end at
the same value); M3's note explains why, and M5 does not gate on it.

Measurement caveat found here: `/usr/bin/time -v`'s "Maximum resident set size"
reports **61 MB** for a run whose peak the kernel itself puts at **80 MB**
(`VmHWM` read from `/proc/<pid>/status` at exit, which agrees with the sampled
`VmRSS` peak to 0.5 MB). It is usable as a relative A/B number and is not usable
as the process's peak; `throughput_ab.sh`'s header says so, and every absolute
peak in these notes comes from `/proc`.

## 6. Accuracy — no regression, twice over

**(a) Against the M0 baseline.** Rebuilt from the committed tree, both mono+IMU
configs, all six rooms, `XIVO_RANDOM_SEED=0`
(`results/memory/m5v_{default,dltnodesc}`):

| config | ATE (mean over 6) | RPE_rot | RPE_tra | vs M0 |
| --- | --- | --- | --- | --- |
| `tumvi_cam0` | 0.1409 | 0.6219 | 0.0352 | identical=20 differ=6 |
| `sweep_dlt_nodesc` | 0.1267 | 0.6226 | 0.0364 | identical=20 differ=6 |

Per sequence, `tumvi_cam0`: 0.111701 / 0.110572 / 0.229807 / 0.111078 /
0.211323 / 0.070892. `sweep_dlt_nodesc`: 0.121896 / 0.119918 / 0.216530 /
0.087888 / 0.123986 / 0.089733.

The six differing files per config are the `run_room*.log` harness logs, and the
only difference in them is the output directory in the three command lines the
harness echoes; they contain no timings. Every trajectory and every metrics file
is byte-identical to `results/memory/m0_baseline_*`.

**(b) Against a binary built from the `auto` branch.** The stronger form, since
it does not depend on M0's artifacts being what they claim: `xivo-base/bin/vio`
(built from `auto`, same flags) and `xivo-memory/bin/vio` on room1 with
`cfg/vio_tumvi.json` produce **byte-identical** 30,943-line trajectory files.

## 7. Throughput — no regression

`scripts/mem/throughput_ab.sh` (new here) alternates the two binaries A,B,A,B…
over whole room1, 4 reps each, on an otherwise idle machine:

| config | build | wall (mean of 4) | wall (min-max) | user CPU (mean) |
| --- | --- | --- | --- | --- |
| `vio_tumvi` | `auto` | 32.02 s | 31.67-32.54 | 108.94 s |
| `vio_tumvi` | HEAD | 32.18 s (**+0.5%**) | 31.79-32.74 | 109.14 s |
| `vio_tumvi_nodesc` | `auto` | 30.46 s | 30.21-30.80 | 100.44 s |
| `vio_tumvi_nodesc` | HEAD | 30.79 s (**+1.1%**) | 29.85-31.47 | 100.68 s |

Both deltas are smaller than each build's own min-max spread (0.9-1.6 s), and
the CPU-time means agree to 0.2%. `vio` runs at >1000% CPU because of OpenCV's
internal threading, so single-run user/sys readings swing by ~15% — an early
single-shot pair suggested +34% system time, which four alternating reps show to
be noise. Minor page faults are the noisiest column of all (27k-574k on
nominally identical runs) and no conclusion is drawn from them.

## 8. Unit tests

| build | ctest targets | gtest cases | sanitizer findings |
| --- | --- | --- | --- |
| release | 7/9 pass | 39 pass, 2 fail | n/a |
| ASan | 7/9 pass | 39 pass, 2 fail | **0** |

The two failures are `NumericalLinearAlgebra.SlowAndFastGivensMatch` and
`Triangulation.Angular_Reprojection_Error`, both pre-existing on `auto` and
recorded in `m0-baseline-and-tooling.md`. The case count is up from M0's 37
because M3 added `DescriptorMemory` (3 cases, minus the one that was already
counted) and M4 added `MetricsRPE` (2 cases).

## Tooling added in M5

* `scripts/mem/leakcheck_matrix.sh` — the 12-run leak gate.
* `scripts/mem/throughput_ab.sh` — the interleaved A/B timing harness.
* `scripts/mem/leakcheck_py.sh` — fixed as described in §2 (this is the change
  that turned it from a script that aborted in the `Estimator` constructor into
  the python-path gate).

## Housekeeping

* `xivo-base` is a scratch worktree, used only for §4/§6b/§7, and removed after
  the measurements. Its local edits (symlinked `thirdparty`, the `-max_entries`
  back-port, the arch-flag swap for the valgrind variant) were never committed
  and never touched the delivery tree.
* `results/m5_final/` at the workspace root is **not** from this task. Its copied
  config differs from the tree's `sweep_dlt_nodesc.json` in `use_prediction:
  true`, i.e. it is a tuning experiment from another line of work (mean ATE
  0.1051). It is left alone; tuning is explicitly out of scope here, and none of
  the numbers above come from it.

## What M5 does not claim

* **Not** that LSan proves the absence of leaks. The whole L2 class (unbounded
  growth reachable from a live singleton) is invisible to it; that class is
  gated by §3, §4 and §5 instead. §1 reads the same at M0 as it does here — the
  L1 leaks M2 fixed are all off the mono+IMU path (behind `USE_MAPPER`, the
  viewer, or a config flag both configs turn off), and the L2 leaks M3 fixed
  were on it but stayed reachable to the end — which is precisely why a clean
  §1 cannot be the only evidence.
* **Not** that the python path leaks zero bytes — it leaks 628,973 B of
  interpreter and numpy state at exit, none of it XIVO's. A future python or
  numpy version will change that number.
* **Not** that L3-2 and L3-7 (reads of uninitialised memory) are verified by a
  sanitizer. They are MSan-class, and MSan needs an instrumented
  libstdc++/OpenCV/Eigen stack; M4's notes argue them from the code and from the
  correct form of the same computation elsewhere in the same file.
* **Not** that the L3-4 aliasing fault is sanitizer-visible; the freed numpy
  block is recycled, so the symptom is wrong pixels, not a report. The
  grayscale over-read *is* reproducible and was reproduced (M4 §Reproductions).
* Throughput is measured on one sequence (room1, whole, both configs) on one
  machine. It is a ±2% statement, not a benchmark.
