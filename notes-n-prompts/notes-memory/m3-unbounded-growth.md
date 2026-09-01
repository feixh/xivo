# M3 — unbounded growth (L2-1, L2-2, L2-3)

This is the milestone that fixes the part of the memory problem that is actually
on the mono+IMU evaluation path. None of it is a LeakSanitizer finding, and none
of it ever will be: every byte here is reachable from a live singleton until
`exit()` and is then released, which is precisely the case LSan cannot see (see
`m1-leak-register.md`, "What ASan/LSan reports: nothing").

## The three defects

### L2-1 — a pooled `Feature` inherited its predecessors' descriptors

`Feature`s are never destroyed; `MemoryManager` pre-allocates `max_features`
(200) of them and hands the same slots out over and over, and `Track::Reset` is
what a slot's next tenant gets instead of a constructor. `Reset` cleared the
pixel history (`clear()`, the `std::vector<Vec2>` base) but not `descriptors_`,
so the descriptor vector was never emptied for the life of the process. Each
tenant appended to whatever the previous tenants had left there.

```cpp
void Track::Reset(number_t x, number_t y) {
  clear();
  descriptors_.clear();          // <-- added
  status_ = TrackStatus::CREATED;
  push_back(Vec2(x, y));
}
```

`src/feature.h:40-49`. This is what made retention grow linearly with run
length: 151 descriptors held after 2,000 dataset entries, 9,059 after the whole
of room1, with the worst single slot holding 73.

### L2-2 — each retained descriptor pinned a whole per-frame matrix

`SetDescriptor` stored the `cv::Mat` it was handed, and every caller hands it
`all_descriptors.row(i)` — a *view* that shares the parent's `UMatData`. Keeping
a 32-byte row therefore kept the entire per-frame BRIEF output alive: 110-270
keypoints × 32 B, a 100-250× amplification. `SetDescriptor` now clones
(`src/feature.h:53-62`), which is a 32-byte copy per detection.

L2-1 set the count, L2-2 set the price; either alone is much smaller than the
product, which is why the register listed them as one defect with two
multipliers.

### L2-3 — `vio.cpp` accumulated a trajectory nothing read

`main` kept `std::vector<msg::Pose> traj_est` and pushed one 96-byte entry per
*dataset entry* — IMU samples included, ~20× the image rate — and never read the
vector. ~3.1 MB by the end of room1, and ~4.7 MB counting the old buffer during
the final reallocation. The pose was already being streamed to `ostream` on the
next line, so the accumulator is simply gone (`src/app/vio.cpp:107-116` carries
the note).

## Fallout from fixing L2-1: three unguarded `back()` calls

Once `Reset` empties `descriptors_`, a `Track` can genuinely have no descriptor,
so `descriptor()` (which returns `descriptors_.back()`) becomes UB where it was
previously returning stale-but-valid data from a previous tenant. That is L3-5 in
the register, and it had to be fixed in the same commit. `descriptor()` now
`CHECK`s, `has_descriptor()` was added, and the three callers were made honest:

* `src/tracker.cpp:781-782` — `GetDescriptors` indexed `fvec[0]` with no size
  check and no descriptor check; it now returns an empty `cv::Mat`.
* `src/tracker.cpp` (dropped-track rescue) — the guard gained
  `extract_descriptor_`. With descriptor extraction off, *no* track has a
  descriptor, so the rescue path was matching garbage; it is now skipped.
* `src/estimator_accessors.cpp` (`tracked_features`) — leaves the returned
  descriptor matrix empty instead of reading `back()` on an empty vector.

Neither evaluated config can reach the new `CHECK`s: `vio_tumvi` has
`extract_descriptor: true` and every track gets a descriptor at detection, and
`vio_tumvi_nodesc` never calls these paths at all. A third-party config with
`match_dropped_tracks` on and `extract_descriptor` off used to run into UB there;
it now takes a defined path.

## Evidence

### 1. The pool census, before and after

M1 measured this defect by counting what the 200 pooled `Feature`s still held at
exit. That measurement is now a committed tool rather than a throwaway patch:
`scripts/mem/pool_census.sh` stops the process in `MemoryManager::~MemoryManager`
under gdb and runs `scripts/mem/pool_census.py` there, walking `slots_` and every
`descriptors_` entry's `cv::Mat::u`. No instrumentation, no special build — the
release build carries the debug info.

room1, `cfg/vio_tumvi.json`, retained `cv::Mat` headers / max per slot / distinct
pinned buffers / pinned bytes:

| entries | pre-fix (M1) | post-fix |
|---|---|---|
|  2,000 | 151 / 1 / 6 / 39,872 | 151 / 1 / 151 / **4,832** |
|  8,000 | 1,744 / 18 / 86 / 572,864 | 200 / 1 / 200 / **6,400** |
| 20,000 | 5,379 / 48 / 296 / 1,832,416 | 200 / 1 / 200 / **6,400** |
| whole  | 9,059 / 73 / 490 / 2,717,792 | 200 / 1 / 200 / **6,400** |

Post-fix the census saturates at exactly one descriptor per slot — 200 headers,
200 buffers of 32 B — and stops moving after the pool has been filled once. That
is what bounded looks like in the metric the defect was found in: 425× less
pinned memory at the end of the sequence, and flat in run length instead of
linear. Note also that pre-fix each pinned buffer was ~5.5 kB (490 buffers,
2.7 MB) because it was a whole per-frame matrix; post-fix a buffer *is* the
descriptor (6,400 / 200 = 32 B), which is L2-2 measured directly.

`vio_tumvi_nodesc` reads 0 in every column before and after, as expected: it
extracts no descriptors.

The `Track` base vector is unaffected and still plateaus (528,384 B of point
capacity at the end of room1, ~0.5 MB) — the "growth by design" the register
recorded.

### 2. Heap attribution: the growth sites are flat

`scripts/mem/massif_profile.sh room1 <out> vio_tumvi 8000` before and after, same
flags, same seed, `-max_entries 8000`. Bytes retained at the two descriptor sites,
per detailed snapshot:

| snapshot | pre-fix `SetDescriptor` | pre-fix `DetectLK (tracker.cpp:236)` | post-fix `SetDescriptor` | post-fix `DetectLK` |
|---|---|---|---|---|
|  9/12 |  39,552 | 128,088 | 60,800 | 0 |
| 11/17 |  52,992 | 159,936 | 60,800 | 0 |
| 14/22 |  78,720 | 227,592 | 60,800 | 0 |
| 20/26 | 144,192 | 357,592 | 60,800 | (1,314,964 — see below) |
| 24/30 | 172,032 | 418,112 | 60,800 | 0 |
| 28    | 199,296 | 481,176 | — | — |
| 34    | 216,192 | 532,304 | 60,800 | 0 |

Pre-fix both sites climb monotonically — 748 kB between them at 8,000 entries, on
the way to the 2.7 MB the M1 pool census measured over the full sequence. Post-fix
the descriptor store is **dead flat at 60,800 B** (≈200 slots × one cloned 32-byte
`cv::Mat` with its header and allocator padding) and the per-frame matrices are no
longer retained at all.

The 1,314,964 B under `DetectLK` in post-fix snapshot 26 is not retention: 26 is a
*peak* snapshot (32.62 MB) taken inside `BriefDescriptorExtractorImpl::compute`,
and the bytes are that call's own scratch — `cv::integral` (1,052,748 B) and
`cv::cvtColorBGR2Gray` (262,216 B), both freed before the call returns. It appears
in exactly one snapshot; the pre-fix numbers appear in every one.

Total heap at the same snapshot index also drops: 26.37 → 25.15 MB.

### 3. A deterministic regression test

`src/test/unittest_descriptor_memory.cpp` (ctest: `DescriptorMemory`) tests the
three things that were wrong, without a sanitizer and without the dataset:

* `ResetDropsDescriptors` — `Reset` empties `descriptors_`.
* `SetDescriptorDoesNotAliasTheSourceMatrix` — asserts `stored.u != block.u`,
  i.e. the stored descriptor does not share the parent's `UMatData`. `cv::Mat::u`
  being public is what makes the pinning testable directly rather than by
  measuring bytes.
* `PoolSlotRetentionIsBoundedAcrossRecycles` — a 4-slot pool, 100
  create/`SetDescriptor`/`Deactivate` cycles, asserting exactly one retained
  descriptor per slot every time. Pre-fix this reaches 25.

Checked as a regression test the only way that means anything: with the
`descriptors_.clear()` and `.clone()` reverted, all three tests fail; restored,
3/3 pass.

### 4. Resident-set growth is bounded in run length

The RSS slope the M0 baseline used as its growth gate turns out to be
noise-dominated at the size of these defects, and it is worth writing down why
rather than quoting it. Six full room1 runs of the fixed build:

```
default  slope over 2nd half =  0.7 /  8.9 /  2.7 kB/s     peak 79.6 / 80.3 / 79.4 MB
nodesc   slope over 2nd half = 39.6 / 59.2 / 28.9 kB/s     peak 76.1 / 76.2 / 76.1 MB
```

and across the length scan below the same statistic ranges from **-1417 to +259
kB/s** on runs that all end at the same RSS. It is a least-squares fit over a
trace that swings ±4 MB with OpenCV's per-frame buffers and glibc arena
behaviour, and it is not monotone: the nodesc trace ends *below* its own second-half
start (-2.2 MB) while fitting a *positive* slope. So the post-M3 nodesc reading of
64.2 kB/s against an M0 reading of 48.6 kB/s is a difference well inside one
statistic's own spread, not a regression. A ~2.7 MB retention simply cannot be
resolved against a ±4 MB transient band — which is why M1 attributed the growth
with massif instead.

The load-independent statement is that peak RSS does not depend on how long the
process runs. `-max_entries` 5,000 → 10,000 → 20,000 → whole sequence:

| entries | default peak | nodesc peak |
|---|---|---|
|  5,000 | 77.9 MB | 76.1 MB |
| 10,000 | 78.2 MB | 75.8 MB |
| 20,000 | 79.8 MB | 76.4 MB |
| all    | 79.1 MB | 75.7 MB |

A 4× longer run costs ~1 MB, inside the run-to-run spread of the measurement
itself. That is what "bounded" looks like.

Where RSS *is* usable is the difference in peak against M0, because it is larger
than the band: default 86.0 → 79.6-80.3 MB (-6 MB, matching L2-3's 3.1 MB plus
the 2.7 MB of descriptors), nodesc 80.0 → 76.1 MB (-4 MB, L2-3 alone; nodesc
extracts no descriptors, so L2-1/L2-2 cost it nothing).

### 5. No behavioural change

* Unit suite: **37 pass, 2 fail** — `NumericalLinearAlgebra.SlowAndFastGivensMatch`
  and `Triangulation.Angular_Reprojection_Error`, the same two that fail on `auto`.
  The 3 new passes are `DescriptorMemory`.
* End-to-end, 6 rooms × 2 configs, `XIVO_RANDOM_SEED=0`: trajectory files
  **bit-identical** to `results/memory/m0_baseline_{default,dltnodesc}`, 18/18 per
  config. Mean ATE 0.1409 (default) / 0.1267 (dlt+nodesc), unchanged.

That is the expected result and worth arguing rather than just measuring:
`descriptor()` returns `descriptors_.back()`, and `back()` is always the current
tenant's freshest descriptor whether or not the previous tenants' entries are
still underneath it. The only reader of the *whole* vector is
`GetAllDescriptors`, which only the (unbuilt) mapper calls. Cloning changes the
address of the descriptor, not its 32 bytes. So the filter sees identical input
and produces identical output.

## Tooling fixed along the way

* `scripts/mem/massif_profile.sh` was committed in M1 with the sequence baked
  into `-root` and no `-seq`, which is not what `GetDirs` (`src/loader.cpp:120`)
  expects — the M1 profiles were taken with a hand-written command line, and the
  script as committed aborted in `DataLoader`. It now passes `-root <parent>
  -seq <seq>`, creates the output directory, and its snapshot flags are pinned to
  the values the M0/M1 baselines used so new profiles remain diffable against
  them.
* `add_test` had no `WORKING_DIRECTORY`, so 7 of the 8 registered tests failed
  under `ctest` for everyone: they open fixtures by relative path (e.g.
  `src/test/camera_configs.json`) and only work when run from the source root.
  With `WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}` added, `ctest` now reports the two
  genuine pre-existing failures and nothing else, which is what M5 needs to be
  able to gate on.

## What M3 does *not* claim

* No LSan report changes, again — nothing here was ever in an LSan report.
* The pre-fix column of the census table is M1's measurement, taken with a
  temporary print in `~CircBufWithHash` (and an independent gdb cross-check that
  agreed to the byte). The post-fix column comes from the committed
  `pool_census.sh`, whose 2,000-entry reading reproduces M1's 151 exactly, so the
  two columns are the same metric — but the pre-fix numbers themselves were not
  re-measured with the new tool.
* `descriptors_` is still a `std::vector<cv::Mat>` that a single long-lived track
  appends to once per frame; that is bounded by the track's lifetime, not by the
  run, and is the "growth by design" the register recorded. It is not touched
  here.
