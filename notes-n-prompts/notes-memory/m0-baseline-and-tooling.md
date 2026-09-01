# M0 — baseline and memory tooling

Worktree `xivo-memory`, branch `auto-memory`, from `auto` @ `888511d`.
`thirdparty/` build outputs were rsynced from the already-built `xivo` worktree
(same commit content, verified: `git status` clean afterwards) instead of
rebuilding ceres/Pangolin/DBoW2 from scratch.

## Accuracy baseline (Release, mono + IMU, `XIVO_RANDOM_SEED=0`)

Six TUM-VI rooms, cam0, via `run_eval_memory.sh` -> `run_and_eval_pyxivo.py`.

```
cfg=tumvi_cam0 (default; extract_descriptor=true)     results/memory/m0_baseline_default
seq      ATE       RPE_rot    RPE_tra
room1    0.111701  0.529697   0.024243
room2    0.110572  0.722711   0.026848
room3    0.229807  0.734710   0.062014
room4    0.111078  0.636277   0.027727
room5    0.211323  0.575169   0.044561
room6    0.070892  0.532696   0.025585
mean     0.1409    0.6219     0.0352

cfg=sweep_dlt_nodesc (best; extract_descriptor=false) results/memory/m0_baseline_dltnodesc
seq      ATE       RPE_rot    RPE_tra
room1    0.121896  0.530619   0.023907
room2    0.119918  0.725508   0.042083
room3    0.216530  0.736194   0.062585
room4    0.087888  0.635901   0.022834
room5    0.123986  0.575496   0.033299
room6    0.089733  0.531867   0.033810
mean     0.1267    0.6226     0.0364
```

Both configs are run because they exercise different code: the default extracts
a BRIEF descriptor per observation and matches dropped tracks, the best one does
neither. Any leak in the descriptor path only shows up under the first.

These numbers are 0.005-0.017 m per sequence away from the table in RESULTS.md
(e.g. `dlt_nodesc` room5 0.124 here vs 0.107 there) even though the commit and
the seed match. RESULTS.md was produced in the `xivo` worktree, which carries
uncommitted local changes on `auto-stereo`; the honest baseline for *this* task
is the one measured here, and M5 compares against it, not against RESULTS.md.

**Determinism is exact.** Re-running room1 with the same seed produced a
byte-identical `tumvi_room1_cam0` trajectory file. So the M5 no-regression gate
can be the strongest possible one: identical trajectories, not "ATE within
noise".

## Unit tests

`out-asan/bin/unitTests_*` under ASan+LSan: 34 of 36 pass, no sanitizer error of
any kind. The two failures are pre-existing numerical ones on `auto`, unrelated
to memory and out of scope here:
`NumericalLinearAlgebra.SlowAndFastGivensMatch`,
`Triangulation.Angular_Reprojection_Error`.

## Memory baseline

`scripts/mem/rss_profile.sh` samples `VmRSS` every 250 ms while `bin/vio` plays
a whole sequence, then fits a slope over the second half of the run (the first
half contains start-up and the feature pool warming up, which is legitimate).

```
room1, cfg vio_tumvi (descriptors on):   start 26.8 MB  peak 86.0 MB  end 86.0 MB
                                         slope 132.8 kB/s  (+2.35 MB over the window)
room1, cfg vio_tumvi_nodesc (off):       start  9.9 MB  peak 80.0 MB  end 78.5 MB
                                         slope  48.6 kB/s  (+0.77 MB over the window)
```

Traces: `results/memory/rss_room1_{default,nodesc}.csv`. Sampled shape with
descriptors on: 79.3 MB at 2.7 s rising monotonically to 86.0 MB at 30 s. That
is steady growth over a fixed-size state (the filter holds at most 200 features
and 100 groups), so it is a leak, not a working set. Turning descriptor
extraction off removes about two thirds of the slope, which points at the
per-observation descriptor path — to be confirmed in M1, not assumed.

## Tooling added (`scripts/mem/`)

| file | what it does |
|---|---|
| `build.sh` | builds `release` / `asan` / `asan-ub` / `valgrind` variants out-of-tree; needed two small CMake hooks, `XIVO_SANITIZE` and `XIVO_OUTPUT_DIR`, so an instrumented build lands in `out-asan/` instead of overwriting `bin/` and `lib/` |
| `leakcheck.sh` | runs the instrumented `vio` on a sequence under ASan+LSan and extracts the report |
| `leakcheck_py.sh` | same for the `pyxivo` binding (needs `LD_PRELOAD=libasan.so`, since a non-instrumented python dlopens the instrumented module) |
| `lsan.supp` | suppressions, third-party one-time allocations only, one justification per entry |
| `rss_profile.sh` | RSS sampling + growth slope |
| `leak_summary.py` | aggregates a LSan report per allocating source line, and diffs two reports to expose sites whose byte count grows with run length |

Also added: `-max_entries N` to `bin/vio` (short runs under a sanitizer, which
is otherwise 30x slower), `cfg/vio_tumvi_nodesc.json` (the vio-app wrapper for
the best estimator config), and `XIVO_ARCH_FLAGS` in CMake because `-march=native`
emits AVX-512 that valgrind cannot execute — massif/memcheck builds target
`x86-64-v3` instead.

## The methodological finding of M0

**Plain LSan reports nothing on this codebase, and that is expected, not good
news.** Two independent runs (500 and 4000 dataset entries) both end with an
empty report. Two reasons:

1. LSan treats anything reachable from a global as live. Every long-lived XIVO
   object hangs off a `static std::unique_ptr` singleton (`MemoryManager`,
   `Graph`, `Tracker`, `Estimator`, `Camera`, `ParameterServer`), so memory that
   accumulates inside them is "reachable" by definition.
2. Those singletons' destructors run *before* LSan's atexit check (ASan
   registers its handler first, so it runs last), which frees the whole feature
   pool — including the containers that grew all run — before anything is
   counted.

So a run-length diff of a *census* is needed: `REACHABLE=1` in `leakcheck.sh`
sets `use_globals=0:use_stacks=0:use_tls=0:use_registers=0`, which reports
everything still allocated. Comparing a 500-entry census with an 8000-entry one
(`leak_summary.py a b`) showed every site identical — same 161 kB in 1052
allocations — which confirms reason 2: at the moment LSan looks, the growth has
already been freed. Attribution therefore has to happen *during* the run, which
is what M1 uses valgrind/massif for.

Consequence for the exit criterion: "free of memory leaks" here has to mean all
three of
* zero LSan-reported leaks (the classic criterion),
* zero ASan memory-safety errors, and
* no growth in the heap while the filter runs on a bounded state.

Only the first is what LSan alone measures, and this codebase already passes it.
