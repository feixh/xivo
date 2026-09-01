# M6 — the OOS scratch buffer: 302 MB, and a lesson about column-major

Accuracy-neutral by construction (byte-identical trajectories). Included because
turning on `use_OOS` + `consistent_init` tripled peak RSS, and that would have
been an unacceptable price to hand to the merge.

| config | peak RSS mono | peak RSS stereo |
|---|---|---|
| pristine baseline | 137.9 | 143.3 |
| final config, before this fix | 444.4 | 459.5 |
| final config, after | **141.2** | **177.2** |

So the whole feature set now costs **+3.3 MB mono / +33.9 MB stereo**, instead of
+306 MB / +316 MB. (The stereo residual is real and expected: twice the
observations per group means twice the marginalized rows to keep.)

## The bug

`struct OOSJacobian` (`src/jac.h`) was a plain member of `Feature`:

```cpp
class Feature {
  ...
  OOSJacobian oos_;   // Hx is kMaxFeature x kFullSize
};
```

`Hx` was sized `kMaxFeature x kFullSize` = 90 x 564 doubles = **406 kB**, and
`Feature` objects come from a `MemoryManager` pool of 1024. So the pool alone is
406 MB of *address space* — which is free until it is touched.

What touched it is `Feature::ComputeOOSJacobianInternal`, which writes one
observation at a time:

```cpp
oos_.Hx.block<2, kFullSize>(row, 0).setZero();
```

**`Eigen::MatX` is column-major.** Writing 2 rows across all 564 columns means
writing 2 doubles at stride `90*8 = 720 B`, 564 times — i.e. reaching into every
single 4 kB page of the 406 kB allocation. One observation faults in the entire
buffer.

Before this work the OOS path was inert (`OOS.pose_window` defaults to 0, `m1`),
so no feature ever ran it and the 406 MB stayed untouched: the bug was latent in
`auto` and was only exposed by turning `use_OOS` on. `consistent_init` then made
it worse, because `ComputeInitJacobian` reuses the same stacking routine, so now
*nearly every* feature — not just the dropped ones — touches its buffer once and
never releases it.

This is why the growth looked so strange while I was measuring it: RSS scaled with
the number of *distinct pool slots that had ever held an OOS or promoted feature*,
which saturates at the pool size a few hundred frames in, and is independent of
how many features are live at any moment.

## The fix

The buffer has two entirely different lifetimes, so it is now two objects.

* **Scratch** — the un-marginalized `[Hf | Hx] dx = inn` stack. Filled and consumed
  inside a single `ComputeOOSJacobian` / `ComputeInitJacobian` call, never read
  afterwards. One instance for the whole process, `Feature::oos_scratch()`; the
  measurement model is single-threaded by construction, the same reason
  `JacobianCache cache_` is already a static.
* **Result** — the `rows - 3` rows that survive the nullspace projection, which is
  what `Ho()` / `ro()` hand to the update. Stays per-feature (the update loop reads
  them after the per-feature loop finishes, so they cannot be shared), but Eigen
  now sizes it to exactly those rows:

```cpp
oos_.Hx = A.transpose() * s.Hx.topRows(rows);
oos_.inn = A.transpose() * s.inn.head(rows);
```

Assigning between two *different* objects also removes the aliasing temporaries the
old in-place version needed. `Feature::Reset` calls `oos_.Release()` so a recycled
pool slot does not keep the previous occupant's rows resident.

`OOSJacobian` now allocates nothing in its constructor — `AllocateScratch()` is
explicit — which is what makes a 1024-slot pool cheap.

## Verification that it is a no-op

1. All 21 `ctest` suites pass.
2. Dumped trajectories for room1 / room3 / room5, both modes, are `cmp`-identical
   to the pre-fix binary.
3. Every accuracy cell of a full `--jitter 6` both-modes run is unchanged to 4
   decimals (`position_memfix` vs `position_final`).
4. FPS unchanged: 52.2 -> 52.5 mono, 28.0 -> 28.0 stereo (within run-to-run
   contention noise).

## Test-fixture fallout, worth knowing

`unittest_oos_update.cpp` and `unittest_oos_stereo.cpp` used `#define private
public` and filled `f_->oos_.Hf` / `.Hx` directly to set up
`MarginalizationAnnihilatesPointJacobian`. After the split that member is the
*empty* result buffer, and since the tests build with `NDEBUG` Eigen's bounds
asserts are compiled out — so instead of failing an assertion the fixture wrote
past the end of a 0x0 matrix and the binary **segfaulted (rc 139)**. The fill phase
now targets `Feature::oos_scratch()`; reads of the marginalized *result* are
unchanged. A reminder that `#define private public` fixtures are coupled to
storage layout, not just to the API.
