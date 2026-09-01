# M4 -- stop the driver paying for memory the estimator never asked for

M1-M3 were throughput milestones and dragged peak RSS down as a side effect
(131.7 -> 122.5 MB mono, 137.1 -> 127.5 MB stereo) because a grayscale frame,
its pyramid and the KLT scratch are all a third of the colour size. That still
left XIVO 37% above OpenVINS' 89.2 / 95.7 MB, so this milestone attacks memory
directly. Both remaining items turned out to be in the Python driver, not in the
estimator.

## Where the 122.5 MB actually was

`/proc/<pid>/smaps`, aggregated per mapping, sampled at 20 Hz over a pinned mono
room3 run and reported at the peak (VmHWM 127.3 MB, sampled Rss 124.3 MB):

```
[heap]                        34.62 MB      libgfortran            1.41 MB
[anon]                        33.93 MB      libpangolin            0.73 MB
libopencv_imgproc              9.41 MB      libm                   0.69 MB
libopencv_core                 7.71 MB      libopencv_features2d   0.60 MB
python3.14                     5.71 MB      libGLEW                0.56 MB
numpy _multiarray_umath        5.17 MB      libGLdispatch          0.53 MB
libscipy_openblas64            5.00 MB      libopencv_imgcodecs    0.53 MB
libcrypto                      4.05 MB
pyxivo...so                    1.85 MB
libstdc++                      1.84 MB
libc                           1.58 MB
```

and a phase-by-phase VmRSS probe (`XIVO_LIB` import, then `Estimator`, then the
data load, then frames) of the same run:

| phase | VmRSS | delta |
|---|---|---|
| interpreter + stdlib | 13.4 MB | |
| `import pyxivo` (pulls in all of OpenCV) | 37.9 MB | +24.5 |
| `Estimator` constructor (the pools) | 61.4 MB (hwm 64.4) | +23.5 |
| **build the `data` list** | **79.1 MB** | **+17.7** |
| first 10 frames (pyramids, KLT scratch) | 90.3 MB | +11.2 |
| steady state, 2000 frames | 94.5 MB | +4.2 |

Two things stand out. numpy + scipy_openblas + libgfortran + libcrypto is
~15.6 MB of *library* that a throughput run never calls, and the `data` list is
17.7 MB -- bigger than any buffer XIVO itself allocates, and bigger than the
whole covariance (564x564x8 = 2.5 MB).

## 1. `savers` is imported lazily (-15.6 MB)

`scripts/pyxivo.py` did `import savers` at module scope. `savers` imports numpy
and transforms3d (which imports scipy), so every run paid for numpy's
`_multiarray_umath`, `libscipy_openblas64`, `libgfortran`, and `libcrypto`
(pulled in by `hashlib`, which `random` imports, which numpy imports).

`-mode runOnly` -- what the scored timing pass uses, and what a deployment would
use -- never constructs a saver. The import moved into `main()` behind
`if args.mode != 'runOnly'`, which is the same predicate the two existing
`args.mode != 'runOnly'` guards in the run loop already use. Nothing else in the
file touches numpy. Standalone `/usr/bin/time -f %M` of the three import states
confirms the size of it: bare `python3.14` 7.76 MB, `+numpy` 33.04 MB,
`+numpy +pyxivo` 56.95 MB.

`-mode eval` behaviour is unchanged, which the bit-identity check below exercises
directly -- it runs in `eval` mode, so a broken lazy import would show up as a
missing trajectory file rather than a matching md5.

## 2. IMU samples live in `array.array`, not in Python objects (-15.6 MB)

The driver used to append both streams to one list and sort it:

```python
data.append((ts, p))                    # image
data.append((ts, (w, t)))               # IMU, w and t are 3-element lists
data.sort(key=lambda tup: tup[0])
```

A room sequence has ~2.8k images but ~28k IMU samples (200 Hz). Each boxed
sample is an outer tuple (56 B) + an inner tuple (56) + two lists (80 each) +
six `float` objects (24 each) + the timestamp `int` (32) + a list slot (8) =
~460 B, so the IMU half of that list alone was ~13 MB, and 17.7 MB with the
allocator's rounding and fragmentation. It is live for the whole run, so it is
peak RSS, not a transient.

Now: `frames` is a list of `(ts, path)` (2.8k entries, 0.7 MB) and the IMU stream
is `imu_ts = array.array('q')` plus a flat `imu_v = array.array('d')`, six
doubles per sample -- 56 B per sample, 1.6 MB total. `readlines()` on the IMU csv
also became a plain file iteration, which removes a 3 MB transient list of
strings.

**Order is preserved exactly**, which is the whole correctness argument. The old
single `data.sort` was a *stable* sort of `[images..., imu...]`, so:

- images sorted by timestamp among themselves (they are unique);
- IMU samples in file order among themselves at equal timestamps;
- at a timestamp shared by a frame and a sample, the **image first**, because
  every image was appended before any IMU sample.

The run loop is now a two-pointer merge whose image-side test is `<=`, which
reproduces all three. The IMU side is consumed in file order; if a data file were
ever non-monotonic the arrays are permuted once up front by a stable
`sorted(range(n), key=imu_ts.__getitem__)`, so the loop itself stays
branch-free. Values are unchanged: `float()` yields the same IEEE double and
`array('d')` stores it without conversion.

## Result

Peak RSS, `-mode runOnly` room3, pinned to one cpu (so glibc creates a single
malloc arena -- the unpinned number is ~10 MB higher and meaningless), measured
against the `auto` worktree in the same window:

| | `auto` | after M3 | after M4 | M4 / `auto` | OpenVINS | target |
|---|---|---|---|---|---|---|
| mono | 135.7 MB | 97.0 MB* | **81.4 MB** | **0.600x** | 89.2 MB | <= 89 |
| stereo | 140.5 MB | 103.4 MB* | **88.3 MB** | **0.628x** | 95.7 MB | <= 96 |

*the "after M3" column is with the lazy `savers` import already applied; M3 as
committed was 122.5 / 127.5 MB.

**Both memory targets are met, and both now beat OpenVINS.** The post-M4 smaps
census is 30.2 MB `[heap]` + 28.7 MB `[anon]` + 17.1 MB of OpenCV text + 5.2 MB
interpreter; numpy, scipy, gfortran and libcrypto are gone entirely.

Throughput is unaffected within noise -- the driver was 0.34% of one-core wall
clock to begin with (see `split-estimator-vs-io.md`). `array.array` indexing boxes
a fresh `float` per read, six per IMU sample, which is paid back by not building
28k tuples and lists; the paired final timing shows no change outside run-to-run
scatter.

## Accuracy

Exact by construction: this milestone moves *no* arithmetic. It changes when a
Python module is imported and what container holds the IMU samples. `float()`
yields the same IEEE double and `array('d')` stores it without conversion, so the
estimator receives the identical bits in the identical order (the two-pointer
merge argument above).

Same md5 as M1/M2/M3 on all four checks (`-mode eval`, md5 of
`dump/tumvi_<seq>_cam0`):

    mono   room1 b5185115fb76d44726cc6ee861ad6e73
    mono   room3 ee2210f7a5093e5baf852cbb22ab09ca
    stereo room1 6dc11ae2a241e8f5296690a708ba1dd0
    stereo room3 129c2d3f7fd637389847346455ea67c3

Note that this md5 is a 1e-6 m check, not a bit-identity check -- that file has
six decimals (`XIVO_DUMP_PRECISE=1` makes it exact). See the CORRECTION in
`m2-batched-sparse-products.md`; it is the mistake that made this caveat
necessary. The stronger evidence for M4 is the same bisection: a build carrying
M4 with only `src/ekf_update.cpp` reverted to M1 reproduced M1's md5 exactly on
mono room5 member 2, a run that M2's reassociation *did* move.

For the record, the corresponding `auto` @ 9e3ec06 md5s are
`9670c4cbc5e359fa67b4adb250bc092a`, `f44de02c1ec8a75a5a04d9786a436dbc`,
`8d06caeaacb00ea6cd29d673c2413393`, `6f3c60422206593ca8590b7d8f283f86`. Two
milestones deliberately change floating-point results -- M1 (grayscale decode
changes rounding inside `LKTrackerInvoker`) and M2 (the batched products are a
reassociation) -- so the branch's accuracy proof is the full `--jitter 6`
ensemble in `speed_acc_final`, tabulated in `summary.md`. M3 and M4 are exact,
so that one ensemble covers the whole branch.

## Considered and not done

- **The `Estimator` constructor's +23.5 MB.** This is the memory pools:
  `CircBufWithHash` does `new T()` for all 800 `Feature` and 300 `Group` slots up
  front. The 800 `Feature`s land in a single 312.6 MB mmap (each carries an
  `OOSJacobian::Hx` of `2*kMaxGroup x kFullSize` = 397 kB) of which only 5.7 MB
  is resident, because `XIVO_EIGEN_INIT=none` means the big members are never
  touched -- the resident part is the ~7 kB of small members per feature spread
  across a couple of pages. Making `Hx` conditional on `use_OOS` would recover a
  few MB of that and 312 MB of address space, but it is a layout change in
  `src/jac.h`/`src/feature.h`, i.e. exactly the files the accuracy agents are
  editing, and the memory targets are already met. Recorded, not taken.
- **The 11 MB fully-resident anon region** that appears during the first frames.
  OpenCV KLT/pyramid scratch, allocated by `cv::fastMalloc` and retained. Reducing
  it means changing pyramid levels or window size, which is capacity.
- **Unlinking Pangolin/GLEW/GLdispatch/GL/X11** when no viewer is requested:
  2.5 MB, and it would mean either a build option that changes the module's
  exported surface or `dlopen`ing the viewer. Poor value for the risk.
- **`libopencv_imgproc` 8.6 MB / `libopencv_core` 6.8 MB resident** are
  file-backed clean text pages (of 24.2 and 11.2 MB mapped). They count in RSS
  but are reclaimable and shared, and OpenVINS links the same libraries.
