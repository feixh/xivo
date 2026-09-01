# M7 — the update's allocations and copies

> **NOT MERGED.** It was merged as `839f45e` and then reverted; `auto` stands at
> `b565b25`. Everything below is correct as measured, but §10 supersedes the word
> "shipped" wherever it appears: the change is bit-identical and 0.022 ms/frame faster
> on stereo, and it reproducibly costs 1.6 MB of stereo peak RSS, which turns 13 of 14
> reported metrics into 12. The work lives on `auto-covscratch` @ `48884e5`.

Branch `auto-covscratch` off `auto` @ `b565b25`. Baseline worktree `xivo-oosbase`,
detached at the same `b565b25`. All measurement on cpus 96-155.

## 0. The verdict first

The ask was the last **0.22 ms/frame** of stereo (14.245 ms today against the 14.025 ms
that 71.3 FPS needs), and the hypothesis was that `Eigen::LLT<MatX> llt(Sc)`'s copy plus
the per-update allocation churn was worth 0.1-0.2 ms of it.

**It is worth 0.026-0.033 ms/frame — about 13% of the ask.** Everything in this
milestone is implemented, bit-identical, tested and shipped, because it is free and
provably exact. But it does not close the gap, and per the round's own instruction I
stopped there rather than reaching for the front end, for OOS tuning, or for an
approximation. §6 accounts for where the remaining ~0.19 ms lives and why none of the
routes to it are worth their price.

Two predictions were wrong and both are recorded below: the copy's cost (§4) and my own
page-fault argument for reusing the scratch (§5).

## 1. What was actually in the loop

Per stereo update, at `chunks: 4`:

| per-update cost | size | why it was there |
|---|---|---|
| `Eigen::LLT<MatX>`'s private copy of `S_c` | 4 x ~370 kB | `LLT(const EigenBase&)` copies, then factorizes the copy |
| one heap allocation per chunk for that copy | 4 | size changes chunk to chunk, so glibc cannot always recycle |
| `MatX M(rows, P.cols())` | 3.7 MB | constructed and destroyed per update |
| `MatX S(kmax, kmax)` | ~370 kB | ditto |
| `VecX u = inn.segment(r0, k)` | 4 x ~1.7 kB | constructed per chunk |
| `ZeroOutsideRuns(M, live)` | ~0.7 MB of stores | see §3 — dead work |

Three keys, each defaulting to the old behaviour:

* **`ekf_update.inplace_llt`** — factorize `S_c` where it already sits.
* **`ekf_update.reuse_scratch`** — hold `M`, `S`, `u` across updates in an
  `UpdateScratch` that grows monotonically.
* **`ekf_update.zero_unused`** — `false` stops the dead zeroing of §3.

## 2. Why this is bit-identical, and what nearly made it not

The load-bearing fact is that **layout, not the API, decides the last bit**.

`Eigen::LLT<Eigen::Ref<MatX>>` factorizes in place, but only if it binds the
`LLT(EigenBase<InputType>&)` constructor, which needs a **named non-const lvalue**. An
rvalue block binds the copying constructor instead, which then fails to compile for a
`Ref` MatrixType — that is the first thing that happened, and it is worth knowing
because the failure mode of getting it slightly wrong is a silent copy, not an error.

The bigger trap: I first wrote this as `S.topLeftCorner(k, k)` factorized in place, `S`
being a `kmax`-square buffer. A standalone probe (`/tmp/llttest/t.cpp`) compared that
against `Eigen::LLT<MatX>` on the same input at k in {13, 64, 97, 215, 360}:

* in-place factorization of a **strided block** of a larger matrix: **not** bit-identical
  (relative error ~1e-17 at k = 13, 97, 215, 360; k = 64 matched by coincidence, its
  columns being 512-byte aligned);
* in-place factorization of a **tight `Map<MatX>` into the front of a flat buffer**:
  **bit-identical at every size**.

Eigen's kernels pick a load path from the alignment of a column, and the alignment of
column *j* of a strided block depends on the outer stride. So the whole design follows
from that one measurement: `M`, `S` and `u` are tight maps onto flat `VecX` storage
grown to a high-water mark, never `topRows()`/`topLeftCorner()` views of an over-sized
matrix. That is also why `UpdateScratch` holds `VecX` and not `MatX` — `MatX::resize`
reallocates whenever the *total* size changes and `rows` changes nearly every update, so
a `MatX` member would reallocate about as often as a local does.

The helper signatures went from `MatX &` to **non-const `Eigen::Ref<MatX>`** deliberately:
a non-const `Ref` will not bind a mismatched layout at all, so a future stride mistake is
a compile error. (`const Ref<const MatX>&` would have silently evaluated into a temporary
— the opposite of what is wanted here.)

## 3. A third item, found by auditing rather than measuring

`ZeroOutsideRuns(M, live)` inside `EkfUpdateDowndate` has been **dead since M3**. Every
read of `M` below it is at columns in `JacFixedSpan()`, `GroupRun(gsind)`,
`FeatureRun(fsind)`, or `DenseSumRuns`' output; the first is inside the motion block,
which is always the head of live run 0 (`OccupiedStateRunsExact` sets
`s.runs[0] = {0, kGroupBegin}` unconditionally), and the rest are inside `live` by the
function's own precondition. The triangular solve, the `err` product and the downdate all
walk `live`. So the gap columns are written every update and never looked at: ~0.7 MB of
stores per stereo update.

It stays behind a key, defaulting to doing the work, because "provably never read" is
exactly the kind of claim that should be switchable. It is **proved rather than argued**:
`NothingOutsideTheLiveExtentOfHPIsEverRead` sizes the scratch, fills all of it with NaN,
runs the update with `zero_unused` off, and asserts both that `P` has no NaN and that it
is bit-identical to the reference. `MeasurementTimesCov` (whose only caller is the test,
and which promises a *full* `H P`) still zeros unconditionally.

## 4. Measured

`run_xivo_reference.sh --timing --no-score`, `CPU_BASE=96 CPU_SPAN=60`, one core per run,
`setarch -R`, serial, all pools pinned to 1 thread, 6 sequences (room1-6), mean of the
per-sequence ms/frame. `base1`/`base2`/`base3` are three passes of `xivo-oosbase` @
`b565b25` spread across the session; their spread is the noise floor.

| stereo | FPS | ms/frame | Δ ms | ratio | peak RSS max / mean |
|---|---|---|---|---|---|
| base1 | 70.13 | 14.260 | — | 1.0000 | 95.9 / 91.8 |
| base2 | 70.14 | 14.258 | −0.001 | 1.0001 | 95.9 / 91.8 |
| base3 | 70.16 | 14.254 | −0.006 | 1.0004 | 95.9 / 91.8 |
| v0 — new code, all three keys off | 70.16 | 14.253 | −0.007 | 1.0004 | 95.5 / 91.8 |
| v1 — `inplace_llt` only | 70.26 | 14.233 | **−0.026** | 1.0018 | 95.6 / 92.4 |
| v2 — `reuse_scratch` only | 70.17 | 14.252 | −0.008 | 1.0005 | 96.5 / **94.1** |
| v3 — both | 70.24 | 14.238 | −0.022 | 1.0015 | 94.9 / 93.5 |
| v4 — both + `zero_unused` off | 70.26 | 14.233 | −0.027 | 1.0019 | 96.1 / 93.6 |
| **v5 — shipped: `inplace_llt`, `zero_unused` off, reuse off** | **70.30** | **14.226** | **−0.033** | **1.0023** | 95.5 / 91.2 |

Mono, same protocol, five passes so that the shipped row has an error bar of its own:

| mono | FPS | ms/frame | Δ ms | ratio | peak RSS max / mean |
|---|---|---|---|---|---|
| base1 / base2 / base4 / base5 | 123.49 / 123.50 / 123.47 / 123.49 | 8.099 / 8.098 / 8.101 / 8.099 | — | 1.0000 | 85.6 / 83.9 (identical in all four) |
| v5-ship, three passes | 123.64 / 123.59 / 123.57 | 8.089 / 8.093 / 8.094 | **−0.007 mean** | **1.0008** | 87.7, 85.5, 86.6 / 84.0, 83.9, 84.3 |

`base1` reproduces the coordinator's idle-box re-measurement (mono 123.4, stereo 70.2,
mono peak RSS 85.4/83.5) so the two protocols agree. `v5-ship` ran immediately before
`base3`; against that adjacent pass the stereo delta is **−0.028 ms**, which is the number
I would quote. The three stereo baselines drift downward by 0.006 ms across 70 minutes,
and the four mono baselines agree to 0.003 ms, so anything below ~0.01 ms is not a
measurement — which is why mono's row is quoted as a mean of three.

**Read the rows, not just the total:**

* **`inplace_llt` is the only piece that measures.** −0.026 ms on its own, and v4 and v5
  do not beat it by more than the noise floor.
* **The copy was predicted at 0.1-0.2 ms/update and delivered 0.026 ms/frame** — and at
  ~1.0 updates/frame (room5: 2841 updates, 2847 frames) those units are interchangeable,
  so the prediction was 4-8x high. 1.5 MB/update at the ~10 GB/s a single core gets from
  L3 is 0.15 ms *if the copy is the only thing happening*; it is not — it is interleaved
  with a factorization that already has the data in cache, and it overlaps.
* **`reuse_scratch` buys nothing and costs memory.** −0.008 ms alone, no better than
  `inplace_llt` alone when combined, and it raises mean stereo peak RSS from 91.8 to
  94.1 MB. Ships **off**. See §5 for why my reasoning for it was wrong.
* **`zero_unused` off is worth ~0.005 ms**, i.e. inside the noise. Ships **on**
  (i.e. the zeroing off) anyway, because the work is provably dead and the test proves it.

## 5. The wrong prediction about page faults

I argued `reuse_scratch` would pay for itself in *minor faults*: 3.7 MB of `M` written
every update is ~900 fresh pages if the allocator hands back new ones. It does not.
glibc's per-arena free lists recycle a block of nearly the same size immediately, and the
sizes here change by a few rows update to update, so the pages were already warm — there
were almost no faults to remove. Reuse then *adds* cost: the buffers are held at the
high-water mark of the widest update for the life of the process instead of being freed,
which is the +2.3 MB of mean peak RSS in the v2 row.

Corollary worth keeping: on this workload, "allocation churn" is not a per-update cost
worth removing. `ReserveOOSRows`' 2x growth was on the same list and is ruled out by the
same argument plus its own comment — it doubles from an initial size, so it fires ~10
times in the first seconds of a run and never again: ~50 MB of copying against ~2900
frames, i.e. unmeasurable. Its 1.4-2.0 MB of unwritten-but-resident pages is the only
real part, and the memory target is already met. Not touched.

### Peak RSS is arena noise here, in both directions

Within the harness protocol, mono max-over-sequences peak RSS reads 87.7, 85.5, 86.6 MB
over three shipped passes against a baseline that reads 85.6 in all four of its passes,
while the shipped mean stays at 83.9-84.3 against the baseline's 83.9. So the effect is
**+1.0 MB on the max with ±1.1 MB of scatter, and nothing on the mean** — and what moves
is *which sequence* holds the maximum (room1 84.0, down from 85.6; room6 87.7, up from
85.4). That is fragmentation, not a buffer. The control that settles it — mono room6,
three repeats each, direct invocation:

| | base | shipped |
|---|---|---|
| default allocator | 85.2, 85.2, 85.2 | 82.2, 82.2, 82.2 |
| `MALLOC_MMAP_THRESHOLD_=65536` | 80.0, 80.0 | **80.0, 80.0** |

Forcing the large blocks to `mmap` makes peak RSS reflect the live buffer set, and the
two builds are then **exactly equal** — so the live set is unchanged and every difference
either way is glibc arena top. Note also that the sign of the default-allocator
difference flips between the harness (+2.3 MB on room6) and a direct run (−3.0 MB on the
same sequence); the only thing that changed is the thread-pool environment. Do not read
±3 MB of single-sequence peak RSS on this codebase as a result.

## 6. The chunk re-sweep, and where the remaining 0.19 ms is

Chunking's per-chunk overhead just went down, so the optimum could have moved. Stereo,
same protocol, all at the v4 keys:

| C | ms/frame | Δ vs base1 | ratio |
|---|---|---|---|
| 3 | 14.310 | +0.051 | 0.9964 |
| **4 (shipped)** | 14.233 | −0.027 | 1.0019 |
| 5 | **14.215** | **−0.044** | 1.0031 |
| 6 | 14.224 | −0.036 | 1.0025 |

C = 5 is 0.017 ms better than C = 4, consistently across five of six sequences. **I did
not take it.** Changing `chunks` is a different factorization, not a reassociation, so it
is not bit-identical; it would need its own accuracy ensemble, and it would invalidate the
72-run ensemble that was running against `chunks: 4` while this was measured. 0.017 ms is
8% of the ask — not worth spending that. It is recorded here as available, at that price.

The rest of the account, at `chunks: 4`, ~1.0 updates/frame, so ms/update ≈ ms/frame:

| phase | ms/update | headroom, honestly |
|---|---|---|
| `P -= W^T W` | 1.590 | 47 GFLOP/s, at the core's limit. The occupancy census caps dimension-cutting at 10%, and m6 measured it at 2%. ~0.03 ms, already taken. |
| `W = L^-1 M` (trsm) | 0.689 | 41 GFLOP/s, same. Falls as 1/C; that is the C = 5 row above. |
| `M = H P` | 0.681 | already block-sparse; `fuse_passes` got 0.023 and the destination is L3-resident. |
| `S = M H^T` | 0.269 | same shape of argument. |
| mirror | 0.194 | **the one real remaining item — see below.** |
| `LLT(S_c)` | 0.095 | 1/C^2; already small. |
| `err` re-prediction | 0.031 | noise. |

**The mirror** is the only phase whose cost is structural rather than machine-limited: it
exists solely because the next chunk forms `H_c P` off *both* triangles of `P`. Reading
only the lower triangle — transposing the above-diagonal run-blocks — would leave just the
diagonal run-blocks needing a per-chunk mirror, worth roughly half of 0.194, so **~0.09
ms/frame**, 40% of the ask. It transposes gemm operands, so it is **not** bit-identical
and would need the full 72-run accuracy ensemble to ship.

Adding up every exact, in-scope, filter-side item that remains: 0.09 (mirror) + 0.017
(C = 5) + the 0.03 already banked here ≈ 0.14 ms against a 0.22 ms ask, and two thirds of
that is a numerics change requiring an ensemble. **There is no route to 0.22 ms on the
filter side that is both exact and bit-identical.** Reporting that is the point of this
milestone.

## 7. Correctness

`ctest` 22/22; `unitTests_ekf_update` 22 → **23** cases.

* `ReusedScratchAndInPlaceFactorizationAreBitIdentical` — `chunks` in {1, 3, 4} x `fuse`
  in {false, true} x `inplace_llt` in {false, true}, three consecutive updates through
  one scratch, one of which is a differently-shaped update (4 groups / 30 features) so
  that the scratch is exercised above and below its high-water mark. Asserts **bit
  identity** (`memcmp`) of `P` and `err` against a fresh-allocation reference, not a
  tolerance. This is what pins §2's claim.
* `ScratchGrowsMonotonicallyAndIsNotResetBetweenUpdates` — the high-water mark survives a
  narrower update, and `scopy` is never allocated at all when `inplace_llt` is on.
* `NothingOutsideTheLiveExtentOfHPIsEverRead` — the NaN-poison proof of §3.

`ChunkedEqualsTheBatchUpdate`, the gate against the independent dense Joseph form, still
passes to the same 1e-9.

**Twelve real runs, byte-identical — twice over.** 6 sequences x {mono, stereo},
`XIVO_DUMP_PRECISE=1` (so 17 significant digits, round-trippable), against
`xivo-oosbase` @ `b565b25`: **all 12 md5s match, on both the raw `*_cam0` dump and
`traj.txt`.** Run once with `reuse_scratch` on (`scr-bitid-*`) and again at the shipped
keys with it off (`scr-bitid2-*`); the 24 hashes are the same 12 values, so both key sets
reproduce `b565b25` exactly. No accuracy ensemble was needed and no accuracy margin was
spent — provably zero, over ~2850 frames per sequence, on a filter chaotic enough that a
single flipped last bit diverges visibly.

That chain also covers the one change that is *not* behind a key: `S_c` is now a tight
`k x k` map where it used to be `S.topLeftCorner(k, k)` of a `kmax`-square matrix, so
`CovTimesMeasurementTRange` writes to a destination with a different outer stride. The
unit test pins all eight key combinations to each other bit-for-bit, and the 12-run check
pins one of them to `b565b25` byte-for-byte, so key-off is byte-identical to `b565b25`
too. Worth knowing that this rests on measurement, not on an argument — gemm destination
stride is not guaranteed to be irrelevant, it just is here.

## 8. Shipped

```jsonc
  "ekf_update": { "exact_runs": true, "run_gap": 3, "fuse_passes": true,
                  "chunks": 4,              // 3 in eff_mono.json
                  "inplace_llt": true, "reuse_scratch": false,
                  "zero_unused": false },
```

`reuse_scratch` ships **off**: it measured nothing and cost ~2.3 MB of mean peak RSS. It
stays in the tree as a key because it is the A/B that establishes §5, and because on a
machine with a less friendly allocator it might yet win.

Result directories, all under `experiments/openvins/results/`:

* `scr-bitid-xivo-{oosbase,oosfast}-{mono,stereo}` and `scr-bitid2-...` — the two 12-run
  byte-identity checks (`reuse_scratch` on and off respectively).
* `scr-t-{base1,base2,base3,base4,base5,v0-none,v1-inplace,v2-scratch,v3-both,v4-nozero,v5-ship,v5b-ship,v5c-ship,v4-c3,v4-c5,v4-c6}-{mono,stereo}`
  — timing. `--no-score`, so there is no `summary.csv`; read `stats.txt` per run dir.
  `base1`-`base5` are `xivo-oosbase`; `v5*-ship` is the shipped configuration; `v4-c*` is
  the chunk re-sweep (stereo only).

Reproduce: `experiments/openvins/run_xivo_reference.sh --worktree xivo-oosfast --mode
stereo --timing --no-score --out <dir>` with `CPU_BASE`/`CPU_SPAN` set, and the same
against `--worktree xivo-oosbase` back to back; drop `--timing` and export
`XIVO_DUMP_PRECISE=1` for the byte-identity check.

## 9. Two measurement-hygiene notes

* **A rebuild in the middle of a sweep invalidates it.** The first attempt at §4 was
  contaminated that way (a `libxivo` rebuild for a new test landed 20 minutes into the
  sweep); it was discarded and re-run clean. Afterwards a comment-only source edit was
  made deliberately *without* rebuilding, to keep the live sweep valid.
* **A build script that takes its target as `"$@"` will silently do nothing if you forget
  the argument.** That is what happened here: `scratch_build.sh` printed `BUILD_DONE`
  having built nothing, so the whole of §4 ran on a binary one comment behind the source.
  Cleared rather than assumed: the old `ekf_update.cpp.o` was kept, the tree rebuilt, and
  `objdump -d` compared. The only difference in the disassembly is one immediate,
  `$0x27a` → `$0x279` — a `LOG` line number shifted by the deleted comment line. So the
  measurements stand, and the final binary was re-checked byte-identical on all 12 runs
  (`scr-bitid3-*`) too. Compare artifact timestamps against source timestamps before
  believing a sweep.

---

## 10. Coordinator's verification, and a correction to §4 and §5

Merged as `839f45e`, then reverted -- see the end of this section. `ctest` 22/22 on the
merged tree; `unitTests_ekf_update` 23 cases.
`cfg/eff_{mono,stereo}.json` both parse with the seven `ekf_update` keys under the object
`estimator.cpp` reads. Byte-identity re-checked independently of the agent's runs: room3
and room5 x {mono, stereo}, `XIVO_DUMP_PRECISE=1`, dumped from the *same* worktree before
and after the merge so the build flags cannot differ -- **8 of 8 files identical**, on both
`tumvi_<seq>_cam0` and `traj.txt`, with a mono-vs-stereo control confirming the comparison
can fail. So the accuracy table needs no re-running, and `final_acc_final5` carries over
to `839f45e` exactly.

### The timing win is real, but a single pass cannot see it

My first attempt compared a fresh pass on the merged tree against `final_fps_final5_*`,
taken about two hours earlier on the same binary, and got **+0.020 ms/frame -- the wrong
sign**. The box drifts more between sessions than this effect is large: the same
`b565b25` binary read 14.247 ms/frame in the earlier session and 14.285 in the later one,
a 0.038 ms shift against a 0.022 ms effect.

Three alternated A/B pairs on core 0 (`xivo-oosbase` @ `b565b25` vs `xivo` @ `839f45e`,
A/B/A/B/A/B, 6 sequences each) settle it:

| | A = `b565b25` | B = `839f45e` | paired delta |
|---|---|---|---|
| stereo ms/frame | 14.285 (sd 0.002) | **14.263** (sd 0.002) | −0.026, −0.020, −0.018; mean **−0.022** |
| mono ms/frame | 8.107 (sd 0.003) | 8.110 (sd 0.002) | +0.002, +0.007, +0.001; **flat** |

So §4's stereo figure is confirmed in sign and order, at **−0.022 rather than −0.033 ms**,
and mono is flat rather than −0.007 (it was inside the noise floor either way). **Never
quote a timing delta of this size from unpaired passes** -- interleave them.

### The peak-RSS regression is real, reproducible, and an allocator artifact

The same A/B, stereo max-over-sequences peak RSS: **A = 94.6 MB in all three passes,
B = 96.2 MB in all three.** Perfectly deterministic, and it crosses the OpenVINS figure
(95.5). Per sequence, B is *lower* on room2/4/6 (−1.8/−1.6/−1.2) and higher on room1/3/5
(+0.8/+0.2/**+1.6**): B's *mean* is 0.3 MB better, but room5 -- the sequence with the
widest update -- becomes the new maximum.

**It is not caused by any of the three keys.** room5 stereo, direct invocation, core 0:

| `inplace_llt`, `reuse_scratch`, `zero_unused` | peak RSS |
|---|---|
| off, off, on (= `b565b25` behaviour, bit-identical to it) | 96.0 |
| on, off, on | 96.0 |
| off, off, off | 94.9 |
| **on, off, off (shipped)** | **96.0** |
| on, on, off | 95.8 |

The key-off row already reads 96.0, so what moved arena top is the *restructuring* of the
temporaries -- four `VecX` members of a local struct in place of two `MatX` locals -- and
not any change in behaviour. A key that changes no arithmetic swings the same number by
1.1 MB, which is the size of the effect being attributed.

The live buffer set went the other way. Forcing large blocks to `mmap` so that peak RSS
reflects live pages rather than arena top, room5 stereo:

| | `b565b25` behaviour | shipped |
|---|---|---|
| `MALLOC_MMAP_THRESHOLD_=65536` | 88.1 MB | **86.6 MB** |

**The shipped code holds 1.5 MB less, and reads 1.6 MB more.** (That mode is also 28%
slower -- 52.0 s vs 40.5 s on room5 -- so it is a diagnostic, not a shipping option.)

### Why it was NOT merged in the end

I first read the regression as arena noise, on the strength of cross-session drift: mono
max-over-sequences peak RSS reads 85.4 MB in one session and 86.7 in another for
`b565b25`, and 86.8 then 85.3 for `839f45e` -- **±1.5 MB, both directions, same binary.**
On that basis stereo peak RSS looked like a tie either way and the merge looked free.

Then I measured the thing that framing depended on, and it did not hold. Three one-core
passes of OpenVINS, and four of each XIVO commit:

| | pass-to-pass spread | value |
|---|---|---|
| OpenVINS stereo peak RSS | **0.2 MB** (95.3, 95.3, 95.5) | 95.4 |
| `b565b25` stereo peak RSS | **0.1 MB** (94.6 x3, 94.7) | 94.65 |
| `839f45e` stereo peak RSS | **0.0 MB** (96.2 x4) | 96.2 |

*Stereo* peak RSS is reproducible per commit to 0.1-0.2 MB on both systems. The ±1.5 MB
drift is a mono phenomenon, and using it to excuse a reproducible stereo regression would
have been reasoning from the wrong sequence. So the honest reading is the uncomfortable
one: **`b565b25` reproducibly clears OpenVINS on stereo peak RSS by 0.75 MB and `839f45e`
reproducibly misses it by 0.8 MB.**

And arena fragmentation is not an instrumentation artifact -- the process really does hold
those pages, so 96.2 MB is a real cost of the shipped binary even though its live buffer
set is 1.5 MB smaller. `ru_maxrss` reverses the true ordering here, but it reverses it
about something real.

The trade, priced against the 3-pass OpenVINS means (stereo 14.057 ms, 95.4 MB):

| | `b565b25` | `839f45e` |
|---|---|---|
| stereo throughput | 0.984x | 0.986x -- **still a miss** |
| stereo peak RSS | **0.992x** | 1.009x -- **a miss** |

So merging costs a metric XIVO wins in order to gain 0.022 ms/frame (0.15% of the stereo
frame) on a metric that stays lost either way: **13 of 14 reported metrics become 12 of
14.** `auto` was therefore reset back to `b565b25`, and the merge commit `839f45e`
discarded. The work is preserved on `auto-covscratch` @ `48884e5`, and it is genuinely
better code -- faster, smaller live set, three new tests, one real numerical trap
documented. It is not shipped because the scorecard it produces is worse.

What would make it shippable, in order of directness: an allocator policy that stops arena
top from absorbing the freed blocks (`mallopt`/`M_MMAP_THRESHOLD` gets the live set but
costs 28% of stereo throughput as measured above, so not that one); or any change that
closes the remaining 0.23 ms of stereo throughput, at which point 0.022 ms stops being
immaterial and the memory metric can be argued on its merits.

What is *not* available is a variant that gets the win without the shift: the saving comes
from removing `Eigen::LLT`'s copy, which necessarily changes the sequence of allocation
sizes, and tuning against glibc arena placement is not a measurement worth optimising.
