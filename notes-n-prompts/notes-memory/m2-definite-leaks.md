# M2 — the five definite leaks (L1-1 … L1-5)

All five are unmatched `new`s. None of them is on the mono+IMU evaluation path,
which is exactly why LSan never reported them: the code they live in is either
behind a build flag, behind a config flag that both configs turn off, or has no
caller at all. They are still real, so they are fixed rather than suppressed.

The fix in every case is to give the allocation an owner, never to add a matching
`delete`. A `delete` in a destructor is the bug one refactor away from coming
back; a `unique_ptr` member cannot leak even if someone adds an early return.

## L1-1 — the Pangolin view handlers (`src/viewer.cpp`)

`View::SetHandler(Handler*)` stores a **non-owning** pointer
(`thirdparty/Pangolin/include/pangolin/display/view.h:220`), so both
`SetHandler(new pangolin::Handler3D(...))` calls leaked. `~Viewer` deleted the
two render states and the texture but not the handlers.

The subtlety is destruction order. The `View` objects are *not* owned by
`Viewer` — they live in pangolin's global display registry (`pangolin::Display`
looks them up by name) and are destroyed at process exit, i.e. after `~Viewer`.
So handlers owned by `Viewer` leave the views holding dangling `Handler*`s. Every
input dispatch site in `src/display/display.cpp` (:476, :486, :531, :548, :595)
null-checks `View::handler` before calling it, so `~Viewer` now clears the
handler on each view before dropping it:

```cpp
pangolin::Display("image").SetHandler(nullptr);
if (!tracker_only_) pangolin::Display("cam").SetHandler(nullptr);
```

`camera_state_`, `image_state_` and `texture_` became `unique_ptr` members in the
same change, which is what empties the destructor of its three `delete`s.
`GlTexture` is destroyed a little later than before (member teardown, after the
body) — it holds a GL texture name and `~GlTexture` calls `glDeleteTextures`
without a bound context either way, so nothing changes there.

## L1-2 — `Mapper`'s vocabulary and RANSAC parameters (`src/mapper.cpp`)

`Mapper::Mapper` `new`s a `FastBriefVocabulary` (:138 — the shipped vocabulary
is 21,110 nodes, and the whole object leaked, not just the descriptors of L1-4)
and a `cvl::PnpParams` (:143, via `GetRANSACParams`);
`Mapper::~Mapper() {}` was empty. Both are `unique_ptr` members now, and
`GetRANSACParams` returns `unique_ptr<cvl::PnpParams>` so the ownership is
visible at the call site instead of being a convention. `*ransac_params_` and
`ransac_params_->threshold` at :394-395 read the same either way.

`mapper.cpp` is compiled unconditionally (`src/CMakeLists.txt:107`) even though
`USE_MAPPER` gates who *calls* it, so this fix is compile-verified in the
default build.

## L1-3 / L1-4 / L3-1 — `FastBrief::TDescriptor` (`src/fastbrief.{h,cpp}`)

These three entries are one root cause: `typedef uint64_t *TDescriptor`.

DBoW2's `TemplatedVocabulary` stores one `TDescriptor` **by value** in every node
(`TemplatedVocabulary.h:284`) and never frees it — for a generic value type there
is nothing to free. With a pointer typedef, every descriptor DBoW2 ever
constructs is leaked:

* `fromString` (`fastbrief.cpp:110`) `new`ed a 4-word array per node while
  loading a vocabulary — 32 B × node count. The shipped
  `cfg/ukbench10K_FASTBRIEF32.yml.gz` has 21,110 nodes (counted), so 675 kB
  leaked on every `Mapper` construction,
* `meanValue` (`:22`) `new`ed one per cluster while training,
* and `meanValue` then did `memset(&mean, 0, 32)` — `&mean` is the address of
  the *reference*, i.e. of the caller's 8-byte pointer variable. That wrote 32
  bytes over an 8-byte object and set the pointer it had just been handed to
  null, so the `mean[i >> 6] |= ...` at `:45` was a write through a null
  pointer. (L3-1.)

So the fix is to make `TDescriptor` a value type:

```cpp
typedef std::array<uint64_t, BRIEF_BYTES / sizeof(uint64_t)> TDescriptor;
```

Nothing can leak, `memset` becomes `mean.fill(0)`, and both `new`s disappear.
Every use in DBoW2 is already value-semantic — `clusters[c]` assignment (:714),
`m_nodes.back().descriptor = clusters[i]` (:788), `F::distance` (:1236, :1241),
`F::toString` (:1420), `F::fromString` (:1488), `getWord` (:1034) — so the
template did not need touching.

Two call sites in XIVO did. `Track::GetDBoWDesc` / `GetAllDBoWDesc`
(`src/feature.cpp`) used to *cast* `descriptors_.back().data` to `uint64_t *`,
which handed DBoW2 a pointer into a pooled `Feature`'s `cv::Mat` — a borrowed
pointer into memory the vocabulary keeps for its whole life, and the same
`cv::Mat` a recycled pool slot will overwrite. They now copy the 32 bytes out
(`ToDBoWDesc`), which also stops reading `uint64_t`s out of a `CV_8U` row that
has no alignment guarantee. `CHECK_EQ` on the row size asserts the 32-byte
assumption that was previously implicit in the cast.

`Feature::Merge` (`feature.cpp:224`) pushes another feature's descriptors into
its own vector; those are now owned `cv::Mat`s, so the refcount keeps them alive
after the merged feature is recycled. That was a latent dangling read before.

## L1-5 — `SaveJson`'s stream writer (`common/utils.cpp:127`)

`Json::StreamWriterBuilder::newStreamWriter()` returns an owning raw pointer
(`thirdparty/jsoncpp/include/json/writer.h:129`); `auto writer = ...` leaked it.
Now a `unique_ptr<Json::StreamWriter>`. `SaveJson` has no caller in the tree, so
this is a leak waiting for its first user.

## Verification

* **Compiles clean.** `scripts/mem/build.sh release`. `viewer.cpp` (target
  `xapp`), `mapper.cpp` and `fastbrief.cpp`/`feature.cpp` (target `xest`) are all
  in the default build, so every edit is compiler-checked. The one warning in the
  changed files' compile output (`-Wstringop-overread` from
  `avxintrin.h` inside `Eigen::CompleteOrthogonalDecomposition::_solve_impl`) is
  pre-existing on `auto` and unrelated.
* **Unit tests: 34 pass, 2 fail** — `NumericalLinearAlgebra.SlowAndFastGivensMatch`
  and `Triangulation.Angular_Reprojection_Error`, byte for byte the M0 baseline's
  two pre-existing failures.
* **End-to-end, 6 rooms × 2 configs, `XIVO_RANDOM_SEED=0`:** ATE/RPE identical
  to the M0 baseline to all printed digits, and the trajectory files are
  **bit-identical** — 18/18 files per config match
  `results/memory/m0_baseline_{default,dltnodesc}`. That is the expected result:
  M2 touches only code the evaluated path does not execute. Results in
  `results/memory/m2_{default,dltnodesc}`.

```
cfg=tumvi_cam0        mean ATE=0.1409  RPE_rot=0.6219  RPE_tra=0.0352
cfg=sweep_dlt_nodesc  mean ATE=0.1267  RPE_rot=0.6226  RPE_tra=0.0364
```

## What M2 does *not* claim

No LSan report changes as a result of this milestone, and none should: none of
these five sites is reached by either config, so they were absent from the M1
reports and are absent now. The evidence for L1-1 … L1-5 is ownership analysis of
compiled code plus, for L1-3/L1-4, the reachability argument above — not a
sanitizer trace. The measurable part of the memory problem is L2, which is M3.
