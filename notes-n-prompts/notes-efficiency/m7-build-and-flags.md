# M7 — the build: what the flags actually were, and the define that was holding the filter up

Commit: `M7: make the build's two implicit choices explicit, and fix the five
reads that one of them was masking` (`7a9f583`).

Every earlier milestone changed the code. This one changes only how it is
compiled, which makes it the milestone where the *measurement* is the whole
deliverable: three of the four things examined here turned out to cost nothing,
and the fourth turned out to cost 310 MB and to be load-bearing for five genuine
bugs.

## What the compile line really was

Not a guess — read out of the generated makefile of a configured build tree
(`build/src/CMakeFiles/xest.dir/flags.make`):

```
-O0 -std=c++17 -Wno-narrowing -Wno-register -fPIC -g -mtune=native -march=native -funroll-loops -O3 -DNDEBUG
```

Two things in there are worth recording because both look like bugs and only one
is.

**`-O0 ... -O3` is a no-op, not a de-optimization.** `CMakeLists.txt:66` opened
`CMAKE_CXX_FLAGS` with `-O0`. CMake appends `CMAKE_CXX_FLAGS_<CONFIG>` *after*
`CMAKE_CXX_FLAGS` on the command line, and gcc lets the last `-O` win, so the
Release build was `-O3` all along. The `-O0` did nothing but mislead anyone
reading the file — including, for a while, me. Removed.

**`-march=native` was already set**, at `CMakeLists.txt:70`. This matters because
"maybe it isn't vectorizing" is the first thing one wants to blame when a dense
EKF runs at 3% of machine peak (it was M6's sparse-machinery overhead, not the
instruction set). The flag lives in a `CACHE` variable precisely so it can be
turned *down*: valgrind on this host cannot execute the AVX-512 that
`-march=native` emits, so the heap-profiling build in M0 configured with
`-march=x86-64-v3`. That is now spelled out in a comment instead of being folk
knowledge.

**`set(CMAKE_BUILD_TYPE "Release")` at line 77 is a plain `set`, not a cache
entry**, so it overrides `-DCMAKE_BUILD_TYPE=...` from the command line. Every
build recipe in this repository passes that flag and every one of them is a
no-op. Left as-is (changing it would silently change what `build_vg` and any
`Debug` tree compile to) but commented, because the alternative is someone
eventually "fixing" a Debug build that was never possible.

## Two implicit choices, made explicit

```cmake
set(XIVO_EIGEN_INIT "none" CACHE STRING "how Eigen fills uninitialized dense objects: zero, nan, none")
option(XIVO_LTO "compile and link with -flto" OFF)
```

Both were introduced at their *pre-existing* values — `zero` and `OFF` — so that
adding the knobs was a no-op, and `XIVO_EIGEN_INIT`'s default was flipped to
`none` only after the measurement below. The point of the knobs is that a flag
experiment then has *exactly one variable* in it. `harness/build_variant.sh` builds a named variant of one unchanged source
tree into `build<suffix>/` and `lib<suffix>/` via `XIVO_OUTPUT_SUFFIX`, and
`fps_one.sh` grew a fifth argument that points `XIVO_LIB` at `lib<suffix>`. So
`zero`, `none`, `lto` and `ltonone` are four builds of one commit, timed
interleaved in one batch. No rebuild between arms, no source difference to argue
about.

## `EIGEN_INITIALIZE_MATRICES_BY_ZERO`: 310 MB, and five bugs

### What the define does

Eigen's `PlainObjectBase.h`:

```c++
#ifdef EIGEN_INITIALIZE_MATRICES_BY_ZERO
# define EIGEN_INITIALIZE_COEFFS
# define EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED for(int i=0;i<base().size();++i) coeffRef(i)=Scalar(0);
```

Two things follow that are easy to miss. First, it is a **scalar store loop**,
not a `memset` — Eigen writes one coefficient at a time. Second, defining
`EIGEN_INITIALIZE_COEFFS` also turns on the refill inside `resize()`:

```c++
#ifdef EIGEN_INITIALIZE_COEFFS
  ...
  if(size_changed) EIGEN_INITIALIZE_COEFFS_IF_THAT_OPTION_IS_ENABLED
```

That second half is where the memory goes.

### Where the 310 MB is

`OOSJacobian` (`src/jac.h`) sizes itself in its constructor:

```c++
OOSJacobian() {
  Hx.resize(2 * kMaxGroup, kFullSize);   // 90 x 564
  Hf.resize(2 * kMaxGroup, 3);
  inn.resize(2 * kMaxGroup);
}
```

Every `Feature` owns one (`feature.h:483`), and the memory manager preallocates
`max_features: 800` of them. `90 * 564 * 8 = 406,080` bytes each, so:

```
800 * 406,080 B = 324,864,000 B = 309.8 MiB
```

Measured RSS difference between the `zero` and `none` builds, same run:
`453,720 - 136,264 = 317,456 kB = 310.0 MiB`. The two agree to a rounding error,
which is as clean an attribution as this kind of thing gets.

The mechanism is that `resize()`'s refill *touches every page*. Without it,
`resize` only asks the allocator for address space and the pages fault in on
first write — and for most of those 800 features first write never comes. The
state holds at most `kMaxFeature = 90` features, only a subset of those are ever
out-of-state, and a feature that does take the OOS path writes
`2 * oos_num_obs` rows of the 90. So the define was faulting in, and zeroing one
double at a time, ~310 MB of buffer that the program overwhelmingly does not
read or write.

Note what this is *not*: it is not a per-frame cost, and it is not why the filter
was slow. It happens once, at pool construction. It shows up in RSS, not in FPS.

### Whether it was load-bearing: it was, in five places

A build that is bit-identical without the define proves nothing, because a fresh
page from the kernel reads as zero anyway — so `none` and `zero` agree by
accident on any freshly-faulted buffer. The probe has to make an unwritten read
*visible*, which is `-DEIGEN_INITIALIZE_MATRICES_BY_NAN`: same machinery, fill
value `quiet_NaN`.

Under `nan` the monocular trajectory diverged from `zero`'s at **frame 4, by
4.2e-3**, and the run eventually produced garbage. Five distinct sites, each
found and fixed:

| # | site | what was read before being written |
| --- | --- | --- |
| 1 | `estimator.cpp`, `Mat3 Ka;` | only `.diagonal()` assigned from `Cas`; the off-diagonal was garbage |
| 2 | `estimator.cpp`, `Mat3 Kg;` | same shape, from `Cgs` |
| 3 | `estimator.cpp`, `Vec3 Wsg;` | gravity rotation has 2 DOF, only `head<2>()` configured; the third component is structurally zero |
| 4 | `estimator.h`, `slope_accel_` / `slope_gyro_` (and `curr_`/`last_`) | `Propagate` extrapolates `last_* + slope_* * dt` on the *visual* branch, but the slopes are only assigned on the IMU branch — so the first measurement being an image reads all six |
| 5 | `helpers.cpp`, `Mat34 P1;` in `DirectLinearTransformSVD` | `[I \| 0]`: the identity block is written, the fourth column never is, and `A.row(0)`/`A.row(1)` read `P1.row(2)` in full |

(1) and (2) are the visible ones: `IMU::IMU` has
`CHECK(Ca(1,0)==0 && Ca(2,0)==0 && Ca(2,1)==0)` at `imu.cpp:24`, which fires
immediately. (3) feeds `SO3::exp` a NaN. All three were fixed with an explicit
`Mat3::Zero()` / `Vec3::Zero()` initializer and a comment saying *why* the
zeroing is structural rather than defensive.

(4) took the most work to locate and is the most interesting: it is an ordering
bug that the define had been papering over since the code was written. The
trajectory-poisoning path is
`Propagate(visual_meas=true)` → `PrinceDormand` → `PrinceDormandStep` →
`ComposeMotion` (`estimator.cpp:876`) → `Sophus::SO3::expAndTheta`. Nothing in
the filter guarantees an IMU sample arrives before the first image, and on this
dataset one does not.

(5) is the one that no conventional tool can see, and it is the transferable
finding of this milestone.

### The tooling result: three tools, and where each is blind

I ran all of them against the same unfixed source:

| tool | verdict on the unfixed code | why |
| --- | --- | --- |
| valgrind memcheck, 250 frames | **0 errors** | memcheck does not re-poison stack memory when a frame pops, so a reused stack slot reads as *defined*. `Mat34 P1` is a stack object. |
| `MALLOC_PERTURB_=42` | **bit-identical trajectory** | heap only. Same blind spot. |
| `-DEIGEN_INITIALIZE_MATRICES_BY_NAN` | **diverges at frame 4** | poisons every dense Eigen object wherever it lives |
| `nan` + an `FE_INVALID` trap | **SIGFPE at the exact instruction** | gcc emits signalling `comisd` for `<` / `>`, so any comparison against a quiet NaN traps |

The trap is an `LD_PRELOAD` shim, three lines, and it is what turned "the
trajectory is wrong somewhere upstream" into a backtrace:

```c
#define _GNU_SOURCE
#include <fenv.h>
__attribute__((constructor)) static void enable(void) { feenableexcept(FE_INVALID); }
```

Must be linked `-lm`, and must be injected through gdb's `set env` rather than
the ambient environment — `LD_PRELOAD` applies to every child process, and it
breaks `sed`.

Two practical notes for anyone repeating this. `pybind11_add_module` **strips**
the module, so gdb in `lib_nan/pyxivo*.so` reports `?? ()` with no line numbers;
debug the unstripped `bin_nan/vio` driver instead. And `vio` wants a *wrapper*
config with an `estimator_cfg` key, not the estimator config itself:

```json
{"estimator_cfg": "cfg/eff_mono.json", "verbose": false, "visualize": false}
```

I also lost time to a self-inflicted version of the same problem: I resolved a
`gdb` offset from a pre-fix build against a post-fix object file, and `addr2line`
duly pointed at an innocent line in `estimator.cpp`, which I then disproved with
a standalone reproduction. Rebuild before you `addr2line`.

### Verification after the fixes

- **Four full NaN-trap runs** — mono and stereo, room1 and room6 — exit 0. No NaN
  reaches a comparison anywhere in four complete sequences. This is the strong
  statement: not "the answer is the same", but "no dense object is read before it
  is written".
- **Bit-identity census** against the `zero` build, `mode eval -dump`, `cmp` on
  `tumvi_<seq>_cam0`. First mono+stereo x room1/room6:

  ```
  none:    4/4 identical      nan: 4/4 identical
  lto:     4/4 identical      ltonone: 4/4 identical
  ```

  then widened to mono+stereo x rooms 2-5:

  ```
  nan:  8/8 identical
  none: 7/8 identical
  ```

  `nan` being identical to `zero` everywhere — 12/12 — is the load-bearing
  entry. It means the fill *value* is unobservable, which is exactly what "no read
  before write" cashes out to, and it is a far stronger statement than `none`
  agreeing with `zero` (fresh pages read as zero, so those two can agree by
  accident).
- 21/21 `ctest` targets under `nan`, and 21/21 under `none`.
- **M6 -> M7 continuity:** the `zero` build of this commit is **4/4
  bit-identical** to the frozen `xivo-effm6` worktree (mono+stereo x room1/room6).
  So all five source fixes are exact no-ops under the old fill setting, and the
  milestone's numerical content is entirely in the flag.

### The one divergence, and why it is not a bug

`none` differs from `zero` on **stereo room3**, and only there. The trajectories
are bit-identical for 548 of 2818 rows, then separate: 1.54e-4 at the first
differing row, growing to 2.0 m at the worst point and 5.9 cm at the end.

That shape is diagnostic. A read-before-write would show up as a difference from
the first frames, and it would have shown up under `nan` — which is 12/12
identical. What a long identical prefix followed by a sudden macroscopic split
means here is a **gate flip**: some comparison in the accept/reject machinery
(`MH_thresh`, depth validity, triangulation cheirality) landed on opposite sides
of its threshold, a different set of features survived, and from there the two
runs are different trajectories. The dumped pose has 6 decimals, so last-bit
differences in the covariance and in feature depths accumulate invisibly for
hundreds of frames before one of them reaches a threshold.

The last bits differ because **omitting the fill loop is a codegen change**, not
just a memory-traffic change. Removing a loop from a constructor changes the
inliner's budget, which changes where gcc contracts `a*b+c` into an FMA, which
changes rounding. This is also why `nan` is bit-identical to `zero` while `none`
is not: `nan` and `zero` emit the *same shape* of fill loop and differ only in the
stored constant, whereas `none` emits no loop at all.

So `none` is not validated by identity — it is validated by ensemble, below,
which is the same standard M1 and M5 were held to when they flipped a run.

## Speed

Four builds of one commit, timed interleaved in a single batch
(`sweeps/m7_flags.log`, 2 repeats, `-mode runOnly`, single-threaded and
seed-pinned). Load average sat at 1.00–1.13 for all 32 runs and the two repeats
of each cell agree to within 0.03 s in 13 of the 16 cells, which is what makes
sub-1% differences readable at all here. FPS = 2821 images (room1) / 2636
(room6) over wall clock.

| arm | seq | wall (s) | **FPS** | RSS (MB) | visual_meas (ms) | track (ms) | update (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero (= M6) | room1 mono | 28.735 | 98.17 | 443.1 | 6.433 | 3.749 | 1.858 |
| **none** | room1 mono | 28.460 | **99.12** | **133.1** | 6.419 | 3.743 | 1.872 |
| lto | room1 mono | 28.630 | 98.53 | 442.9 | 6.408 | 3.740 | 1.863 |
| lto+none | room1 mono | 28.395 | 99.35 | 133.6 | 6.401 | 3.744 | 1.857 |
| zero (= M6) | room6 mono | 26.320 | 100.15 | 441.5 | | | |
| **none** | room6 mono | 26.100 | **101.00** | **132.1** | | | |
| lto | room6 mono | 26.300 | 100.23 | 440.0 | | | |
| lto+none | room6 mono | 25.955 | 101.56 | 132.0 | | | |
| zero (= M6) | room1 stereo | 64.470 | 43.76 | 447.5 | 15.800 | 10.398 | 4.419 |
| **none** | room1 stereo | 64.160 | **43.97** | **138.2** | 15.79 | 10.39 | 4.42 |
| lto | room1 stereo | 64.420 | 43.79 | 447.4 | | | |
| lto+none | room1 stereo | 64.155 | 43.97 | 138.1 | | | |
| zero (= M6) | room6 stereo | 60.075 | 43.88 | 447.9 | | | |
| **none** | room6 stereo | 59.880 | **44.02** | **139.2** | 15.778 | 9.863 | 4.957 |
| lto | room6 stereo | 59.990 | 43.94 | 458.8 | | | |
| lto+none | room6 stereo | 59.690 | 44.16 | 139.8 | | | |

Three readings, in decreasing order of how much they matter.

**Memory: 3.3x.** 443 -> 133 MB monocular, 448 -> 138 MB stereo. This is the
result of the milestone. It is not an FPS number and it does not appear in the
metric this task is graded on, which is exactly why it is worth stating plainly:
the filter's resident set was 70% zero-fill of a buffer it does not use.

**Wall clock: ~0.25 s, once.** The saving is 0.275 / 0.220 s on mono room1 /
room6 and 0.310 / 0.195 s on stereo — *constant in absolute terms*, independent
of frame count and of whether the second camera is on. That is the signature of a
one-time cost, and it matches the arithmetic: 324.9 MB of scalar stores into cold
pages in ~0.25 s is ~1.3 GB/s, right for `coeffRef(i)=Scalar(0)` plus a page
fault every 4 kB. Consistent with it, every per-frame timer is flat (`visual_meas`
6.433 -> 6.419, `track` 3.749 -> 3.743, `update` 1.858 -> 1.872 — noise, and not
all in the same direction). So the honest statement is *not* "the define cost 1%
of frame time"; it is "the define cost a quarter of a second of startup, which a
2821-frame run amortizes into a 0.96% FPS gain and a 30-minute run would amortize
into nothing."

On the cumulative figure: chaining M6's 4.72x / 3.54x by the within-batch ratios
here (x1.0097 mono, x1.0048 stereo) gives **4.76x** and **3.56x**. That is now
seven chained ratios from seven batches, which is a worse estimator than it looks
— M8 should put the baseline and the final build in a single batch and quote that
instead.

**LTO: nothing.** +0.37% / +0.08% mono and +0.08% / +0.14% stereo over `zero`;
+0.23% / +0.56% and -0.01% / +0.32% for `lto+none` over `none`. One of the eight
comparisons is negative, which is the tell that this is at the noise floor rather
than a small real gain. The reason is structural and was foreseeable: the hot
code is templated Eigen living in headers, so it is already visible to the
compiler at every call site and `-flto` has nothing left to inline across a
translation-unit boundary. The remaining cross-TU calls — `Estimator` into
`Tracker`, `Feature`, `helpers` — are a handful per frame around milliseconds of
arithmetic.

### Symbol visibility: also nothing, but for a reason worth knowing

The plan's third flag idea was `-fvisibility=hidden` /
`-fvisibility-inlines-hidden` / `-fno-semantic-interposition`, on the theory that
in a `-fPIC` shared object every call to an exported symbol goes through the PLT
and cannot be inlined. The premise checked out — the delivered module exports
**472 `xivo::` symbols** and has **822 PLT slots** — and the flags do what they
claim: 472 -> **0** exported `xivo::` symbols, 822 -> **415** slots, and the
module shrinks 2,274,600 -> 2,193,512 bytes (-3.6%).

The speed (`sweeps/m7_vis.log`, against `none` in the same batch):

| seq | none (s) | vis (s) | delta |
| --- | --- | --- | --- |
| room1 mono | 28.555 | 28.360 | +0.68% |
| room6 mono | 26.090 | 26.100 | -0.04% |
| room1 stereo | 64.285 | 64.235 | +0.08% |
| room6 stereo | 60.010 | 59.715 | +0.49% |

Noise, and negative in one cell — the same verdict as LTO, and consistent with
it: LTO subsumes cross-TU inlining and gave nothing, so the subset of that win
which hidden visibility unlocks could not have been large. **Not adopted**, so
that the delivered binary carries one fewer unvalidated codegen change for zero
measured gain. The variant is left buildable (`build_variant.sh xivo-efficiency
_vis -DCMAKE_CXX_FLAGS="-fvisibility=hidden -fvisibility-inlines-hidden
-fno-semantic-interposition"`).

Worth flagging separately from efficiency, since it is a real defect this
measurement happened to surface: a Python extension module exporting 472
`xivo::` symbols at default visibility is a symbol-collision hazard for any other
extension in the same interpreter that links its own Eigen, Sophus or glog. That
is a packaging bug, not a performance one, and it is out of scope here.

## Accuracy

8-member ensembles from the frozen worktree `xivo-effm7` (`7a9f583`), 6 rooms
each, both settings — the full protocol, because the stereo room3 flip means
identity is not available as a shortcut here.

| | ATE | ATE@0.02 | RPE_rot | RPE_tra | RPE_rot_i | RPE_tra_i |
| --- | --- | --- | --- | --- | --- | --- |
| m6_mono | 0.0797 ± 0.0063 | 0.0957 | 0.6205 | 0.0227 | 0.5126 | 0.0222 |
| **m7_mono** | **0.0786 ± 0.0049** | **0.0945** | 0.6205 | 0.0226 | 0.5126 | 0.0222 |
| m6_stereo | 0.0549 ± 0.0033 | 0.0631 | 0.6208 | 0.0139 | 0.5128 | 0.0132 |
| **m7_stereo** | **0.0549 ± 0.0033** | **0.0630** | 0.6208 | 0.0139 | 0.5128 | 0.0132 |

Per sequence, every room is identical to M6 to four decimals **except room3**:

| seq | mono M6 -> M7 | stereo M6 -> M7 |
| --- | --- | --- |
| room1, room2, room4, room5, room6 | unchanged | unchanged |
| room3 | 0.1045 -> **0.0983** | 0.0804 -> 0.0805 |
| room3 @0.02 | 0.1488 -> **0.1412** | 0.0962 -> 0.0957 |

And the per-run census, which localizes it exactly:

```
mono   m7 vs m6: 46/48 identical   differ: m1/room3, m4/room3
mono   m7 vs m5: 46/48 identical   differ: m1/room3, m4/room3
stereo m7 vs m6: 47/48 identical   differ: m0/room3
stereo m7 vs m5: 48/48 identical
```

Three runs out of 96 changed, all three on room3 — which is the sequence with the
largest ensemble spread in both settings (mono sd 0.0151, the biggest of the six)
and the one that has absorbed every flip since M3. The stereo entry is neat:
`m7 vs m5` is **48/48**, so what M7 did on stereo was un-flip the single run M6
had flipped, putting stereo back on M5's exact trajectories. And `m0` is the
`Vsb_x = 0` member, i.e. literally the unperturbed run whose divergence I found in
the bit-identity census — the two observations are the same event seen twice.

Direction of the change: mono ATE **improves** 0.0797 -> 0.0786 and ATE@0.02
0.0957 -> 0.0945; stereo is flat at 0.0549 / 0.0630. Both mono moves are ~0.2 sd
and mean nothing — the correct reading is *unchanged*, not *better*. The
requirement was not to degrade, and nothing degraded in any of the ten reported
statistics across both settings.

## What is delivered

| knob | default | why |
| --- | --- | --- |
| `XIVO_EIGEN_INIT` | **`none`** | 310 MB and 0.25 s for nothing. Bit-identical output; the five reads it was masking are fixed. |
| `XIVO_LTO` | **`OFF`** | Unmeasurable gain, real cost in build time, and it perturbs inlining and therefore FMA contraction. Kept as a knob, not a default. |

`XIVO_EIGEN_INIT=zero` and `=nan` both stay available, and `nan` is the one to
reach for the next time a trajectory goes strange: it is now a *sharp* tool, in
the sense that it produces bit-identical output on a healthy tree, so any
difference at all is a finding.

The evidence the `none` default rests on, in order of strength:

1. **No dense Eigen object is read before it is written.** Four full NaN-trap runs
   (mono/stereo x room1/room6) with `FE_INVALID` armed, exit 0; and the `nan`
   build is bit-identical to `zero` on all 12 sequence/setting pairs, so the fill
   value is unobservable.
2. **Accuracy holds at ensemble level.** 8 members x 6 rooms x 2 settings; ATE,
   ATE@0.02 and all four RPE statistics unchanged or marginally better; 93 of 96
   runs bit-identical, the other three all on room3.
3. 21/21 `ctest` targets under `nan`, 21/21 under `none`, 21/21 in the frozen
   `xivo-effm7` delivery build.

What that does *not* cover is a code path none of the six sequences and none of
the 21 test targets reach. That residual is why the define is a cache variable
with three values rather than a deletion.

## Packaging: `-march=native`

Not changed, and worth writing down why. `-march=native` is correct for this
task — every number in this report is from one machine, and the point of the
exercise is how fast the filter *can* run. It is wrong for a distributed binary:
the AVX-512 that Zen 4 accepts here will `SIGILL` on anything older, which is
also the concrete reason the flag is already a `CACHE STRING` and not baked in.
A packaging build should configure `-DXIVO_ARCH_FLAGS="-march=x86-64-v3"` (AVX2 +
FMA, ~2013 and later), which is where essentially all of the vectorization
benefit already is: as established in M6, Zen 4 splits a 512-bit FMA into two
256-bit halves anyway, so the peak is 16 flops/cycle either way. Recording the
recipe rather than measuring it, because "how fast is it on a CPU I do not have"
is not a question this harness can answer.
