# M5 — one marginalized-Jacobian buffer instead of ~800, and what it taught me about "bit-identical"

No config key (see the last section for why not, and what I checked instead).

## The change

`Feature::oos_` — an `OOSJacobian` per pooled feature, up to 257 kB of `Hx` — is
gone. In its place:

* `Feature::oos_result()`, one `static OOSJacobian` sized `2*kMaxGroup x kFullSize`
  (406 kB, once), holding whichever feature most recently called
  `ComputeOOSJacobian`;
* `Feature::oos_result_Hx(rows)`, an `Eigen::Map<MatX>` over its storage with exactly
  the `rows x kFullSize` layout a freshly allocated matrix would have;
* `Estimator::oos_H_` / `oos_inn_` / `oos_blocks_`, the estimator's own stacked copy,
  grown geometrically by `ReserveOOSRows`;
* `Feature::oos_runs_`, a `RunSet` (384 B) which *is* still per feature, because
  `FilterUpdate` hands it to the covariance update long after the loop that built it.

The argument for the old per-feature buffer was that `FilterUpdate` reads `Ho()`/`ro()`
after the loop that computes them, so the rows must survive the loop. They must — but
not *in the feature*. `ComputeOOSMeasurements` now copies an accepted measurement's
rows into `oos_H_` the moment the gate passes, so no feature's rows are read after the
next feature is processed. That copy is the entire price: `rows x 564 x 8` bytes per
accepted feature per frame, ~5 accepted features a frame, i.e. under 1% of the update.

Three lifetime hazards, all real, all handled:

* `Ho()`/`oos_Hx()` are views into shared scratch and are valid only until the next
  `ComputeOOSJacobian`. Documented at both accessors.
* A recycled pool slot must not inherit `oos_jac_counter_`, or `Ho()` hands out another
  feature's rows. Cleared in `Feature::Reset` and in `ReleaseOOS()`.
* `oos_Hx()` returns a `Map` **by value**, so `const auto H = f->oos_Hx().topRows(n)`
  binds a `Block` to a temporary that dies at the end of the full expression. That is
  UB and I wrote it; `Estimator::OOSGating` now names the map instead. The map is
  already exactly `n` rows tall, so there was nothing to slice.

`OOSJacobian` also lost the `RunSet runs` member M1 had added to it — the run set that
actually gets used lives in `Feature::oos_runs_` and in `Estimator::oos_blocks_`, so
`src/jac.h` is back to its HEAD contents.

## What it bought — and the branch total

This is the last milestone, so the final paired numbers for the whole branch live here.
There is no separate `summary.md`: the environment I ran in refuses to create one, so
this section is it.

Everything is **paired**: candidate and baseline interleaved in one script, one core
(`taskset -c 0`), `setarch -R`, all thread pools at 1, on an otherwise idle box (load
0.04 at launch). The baseline pass reads 83.02 FPS mono, matching the 83.1 FPS quoted
for the merged tree, so these absolutes are on the same scale as the targets.

| mono, room1-6 | baseline | branch | delta |
| --- | --- | --- | --- |
| end-to-end FPS, mean of 6 | 83.02 | **101.75** | **1.228x** (geo 1.227) |
| ms/frame | 12.076 | **9.830** | **−2.246** |
| peak RSS, max / mean | 103.8 / 93.5 MB | **96.1 / 87.2 MB** | −7.7 / −6.3 MB |
| ATE@0.02, 6-member ensemble | 0.05662 | 0.05662 | **+0.00000** |

| stereo, room1-6 | baseline | branch | delta |
| --- | --- | --- | --- |
| end-to-end FPS, mean of 6 | 41.06 | **46.17** | **1.125x** (geo 1.125) |
| ms/frame | 24.364 | **21.658** | **−2.706** |
| peak RSS, max / mean | 150.8 / 127.6 MB | **142.8 / 110.3 MB** | −8.0 / −17.3 MB |
| ATE@0.02, 6-member ensemble | 0.04716 | 0.04716 | **+0.00000** |

Per-sequence FPS ratios: mono `1.239 1.245 1.276 1.208 1.269 1.132`, stereo
`1.137 1.123 1.139 1.119 1.151 1.081`. Lowest in both modes is room6, the sequence with
the fewest out-of-state features — the win tracks where it should.

Estimator timers, ms/frame, mean over room1-6, `base/cand`:

| timer | mono | stereo |
| --- | --- | --- |
| `visual-meas` (whole visual path) | 8.958 / **6.698** | 18.491 / **15.761** |
| ⤷ `track` (front end — *not this branch*) | 3.464 / 3.443 | 9.366 / 9.328 |
| ⤷ `process-tracks` | 5.407 / **3.168** | 9.034 / **6.343** |
| ⤷⤷ `jacobian` | 0.079 / 0.074 | 0.133 / 0.126 |
| ⤷⤷ `MH-gating` | 0.093 / 0.100 | 0.098 / 0.111 |
| ⤷⤷ `oos-jacobian` (incl. the OOS gate) | 0.327 / **0.044** | 0.563 / **0.083** |
| ⤷⤷ `update` | 2.504 / **2.270** | 5.757 / **5.232** |
| ⤷⤷ untimed remainder of `process-tracks` | 2.404 / **0.680** | 2.483 / **0.791** |
| `propagation` | 0.032 / 0.032 | 0.033 / 0.032 |

The three rows that matter:

* **the untimed remainder, −1.72 ms mono / −1.69 ms stereo** — `MarginalizeOOSPoint`
  plus `InitializeFeatureCovariance`, neither of which has a timer of its own. Both used
  to form `x * kFullSize` products. `consistent-init` fires 17511 times in a mono room1
  run; nobody had costed it. **M1 and M2 are the largest part of the win.**
* **`oos-jacobian`, 7.4x mono / 6.8x stereo** — mostly `OOSGating` no longer reading all
  2.54 MB of `P_` for a 9-row gate (M1).
* **`update`, −0.234 / −0.525 ms** — run-aware `MeasBlock` for the out-of-state blocks,
  which previously paid for the live extent, 488.5 of 564 columns (M3).

`MH-gating` is 0.007 ms/frame *slower* in both modes, unexplained; it is 0.07% of the
frame and I stopped looking.

**Accuracy, 72 paired runs** (6 seqs x 6 `--jitter` members x 2 modes x 2 trees, no
divergences). The contract's preferred tier is "within 1 ensemble sd"; the result is far
stronger — every scored metric agrees **run-for-run** to the scorer's printed precision,
largest single-run difference 1e-6 m (`ate_0001`) and 1e-3 deg (orientation ATE), each
one unit in the last printed digit.

| metric | mono base / cand | mono ens. sd | stereo base / cand | stereo ens. sd |
| --- | --- | --- | --- | --- |
| ATE@0.02 [m] | 0.05662 / 0.05662 | 0.0056 | 0.04716 / 0.04716 | 0.0051 |
| ATE@0.001 [m] | 0.04732 / 0.04732 | 0.0033 | 0.03925 / 0.03925 | 0.0036 |
| posyaw [m] | 0.05839 / 0.05839 | 0.0056 | 0.04869 / 0.04869 | 0.0052 |
| posyaw [deg] | 0.91042 / 0.91039 | 0.095 | 0.88442 / 0.88442 | 0.092 |
| RPE 8 m [m] | 0.02628 / 0.02628 | 0.0021 | 0.02083 / 0.02083 | 0.0010 |
| RPE 8 m [deg] | 0.51378 / 0.51378 | 0.016 | 0.51539 / 0.51539 | 0.015 |

So the second tier (trade accuracy, stay ahead of OpenVINS) never had to be used. For
the record the branch is ahead of the OpenVINS baseline on every metric in both modes
(mono 0.0566 vs 0.0611 m, 0.910 vs 1.42 deg, 0.0263 vs 0.030 m RPE; stereo 0.0472 vs
0.068 m, 0.884 vs 1.44 deg, 0.0208 vs 0.027 m) — but that accuracy is inherited from the
merged tree, not produced here.

The final `[census]` line is **byte-identical** between the trees in both modes — same
feature slots (84.5729/90), same groups (30.9525/45), same live extent (488.52/564),
same out-of-state rows (`oos:11.5191`), same promotions (17511/17743). No gate,
threshold or slot decision differs anywhere.

**Against the targets, and what is left.** Mono needs 114.6 FPS: the branch reaches
101.75, so 1.126x remains. Stereo needs 71.3 FPS: 46.17, so 1.54x remains, and stereo
cannot be closed from the estimator — `track` alone is 9.33 ms of a 21.66 ms frame and
stereo decode is ~5.9 ms, so a *free* estimator would still leave stereo near 64 FPS.
Of the 9.83 ms mono frame, 3.13 ms is decode/IMU outside `visual-meas` and 3.44 ms is
`track` (both the front-end agent's), leaving 3.20 ms of estimator, of which **`update`
is 2.27 ms — now the largest single estimator item in both modes** (5.23 ms stereo).
`EkfUpdateDowndate` still touches the whole live block of `P_`; making the covariance
update itself run-aware the way the measurement products now are is the obvious next
step, and it is a bigger and riskier change than anything on this branch. Mono peak RSS
mean is now under target (87.2 vs 88.1 MB) but the max is 96.1 MB; stereo is 142.8 vs
95.5 MB, and that overshoot is no longer the OOS path (406 kB total after this
milestone) — it needs a fresh probe against the image pyramids and the mapper.

**What didn't work:** measurement compression (QR the stacked `H` to `n` rows) is a
no-op for XIVO's shape — ~181 rows mono / 360 stereo against 564 columns, so `m < n` and
there is nothing to compress; I did not build it. One contiguous run per OOS feature is
the wrong shape (non-adjacent group slots), hence `RunSet`. `MeasurementUpdate` was
already live-extent aware, so M3's headroom was 488.5/564, not 564/25. glibc allocator
tunables buy 4.4 MB for 32% of the throughput. Retuning OOS parameters down was ruled
out and the probe agrees it is the worse trade. And the three probes below.

Reproduce with:

```bash
cd /home/ubuntu/workspace/auto-slam-engineer
H=experiments/openvins; R=experiments/results
for t in cand:xivo-oosfast base:xivo-oosbase; do for m in mono stereo; do
  CPU_BASE=0 CPU_SPAN=60 $H/run_xivo_reference.sh --worktree ${t##*:} --mode $m \
      --jitter 6 --out $R/oosfast_m5_${t%%:*}_$m ; done ; done
for m in mono stereo; do for t in cand:xivo-oosfast base:xivo-oosbase; do
  CPU_BASE=0 CPU_SPAN=60 $H/run_xivo_reference.sh --worktree ${t##*:} --mode $m \
      --timing --no-score --out $R/oosfast_m5_t${t%%:*}_$m ; done ; done
```

## "Bit-identical with the key off" is not a property this tree has

The brief asks that merging the code without the config key be bit-identical. I could
not deliver that, and while trying to find out why I found something more useful.

The chase, in order, all on TUM-VI room1 mono with `XIVO_DUMP_PRECISE=1` (17
significant figures — a `%f` dump only proves 1e-6 m and is not a bit-identity check):

| run | md5 of `dump/tumvi_room1_cam0` |
| --- | --- |
| A: `xivo-oosbase` (HEAD, c0e7f62), stock cfg | `6ea1cd77ecaede37eb0e8c7768149fc0` |
| B: `xivo-oosfast`, **the same cfg file** (so `oos_fast` absent ⇒ off) | `c5f8e7670907e834f32a435a1a7cafde` |
| C: `xivo-oosfast`, `oos_fast.enable: true` | `51e29101828bbc0e47f37189644960d3` |

A ≠ B, i.e. the key-off path is not bit-identical. Max position difference over the
2818 poses: **3.396e-14 m** (B), 6.549e-14 m (C). Every `[census]` line is
byte-identical between all three, and so is the whole OOS statistics block
(`candidates=14324 used=5017 too_short=4891 bad_triangulation=4415 gated=1`,
`rows=32297`), so no gate, threshold or slot decision differs anywhere — it is
rounding, not behaviour.

Four hypotheses, tested and killed:

1. *The gemm destination's outer stride* (the shared buffer is 90 rows tall, the old
   per-feature matrix was `out_rows`). Introduced `oos_result_Hx()` to give the map the
   exact old layout. **md5 did not move at all** — Eigen's gemm is insensitive to the
   destination stride here.
2. *Writing the product straight into the shared buffer* rather than into a fresh
   matrix. Reverted to a local `MatX`/`VecX` followed by a copy. **md5 unchanged.**
3. *Statement order in `ComputeInitJacobian`* (M2 moved the `Hx` assembly after the QR).
   Restored HEAD's order for the key-off branch. **md5 unchanged.**
4. *Codegen*: the tree is built `-O3 -march=native -funroll-loops -flto=auto`, so an
   unrelated edit could perturb FMA contraction. Added a `volatile int` dead store to
   `InitializeFeatureCovariance`. **md5 unchanged** — so the tree is not gratuitously
   codegen-sensitive either.

Then the decisive test. Turn off **both** `use_OOS` and `consistent_init`, so that
essentially none of this branch's code runs, and the two trees *still* disagree
(2.824e-14 m). What is left that differs? Only object sizes and therefore heap
addresses. So I perturbed the allocator instead of the source:

| room1 mono, `use_OOS: false`, `consistent_init: false` | md5 | max diff vs first row |
| --- | --- | --- |
| `xivo-oosbase` (HEAD) | `6758ec25779ce1d0d98ac80974d34eb0` | — |
| `xivo-oosbase`, **same binary**, `MALLOC_TOP_PAD_=1048576` | `6758ec25779ce1d0d98ac80974d34eb0` | 0 |
| `xivo-oosbase`, **same binary**, `MALLOC_MMAP_THRESHOLD_=16384` | `e21b3a605e9e65f76333ff48ef8586c4` | **4.312e-12 m** |
| `xivo-oosfast` (this branch, both features off) | `fd2ff770af9f2d136603546462aff92f` | 2.824e-14 m |

**XIVO at HEAD is a function of its heap layout.** One environment variable, no source
change, no config change, ASLR off, one core, one thread — and the trajectory moves by
4.3e-12 m, which is **130x larger than the largest deviation my branch produces with
its key off**. Any change that alters an object's size or an allocation's order lands
inside that noise, so md5 equality is not an available acceptance criterion for memory
work in this tree, and a passing md5 check would have been luck rather than evidence.

I did not chase the leak to its source; the candidates are the pointer-keyed
`std::unordered_set<FeaturePtr>` / `<GroupPtr>` containers (`gauge_features_` in
`src/graph.h`, the sets in `Estimator::OnePointRANSAC` — that one is off in
`eff_mono.json` — and `Mapper::InvIndex_`), whose iteration order is the objects' heap
order, and OpenCV's alignment-dependent SIMD paths in the tracker, which would move
the measured pixel coordinates themselves in the last bit. `src/geometry.cpp:177`
records that a previous agent already found and fixed one instance of exactly this
(`PointsAreCollinear` fed from an `unordered_set`), so the class of bug is known here.
Fixing the rest is a determinism task, not an efficiency one, and it would change the
numbers, so I left it alone and wrote it down.

**What I claim instead of bit-identity**, for both the key-off and the key-on path:

* every `[census]` line and the whole OOS statistics block are byte-identical to HEAD;
* max trajectory deviation 3.4e-14 m (key off) / 6.5e-14 m (key on) over room1;
* both are 60-130x below the tree's own allocator-layout noise (4.3e-12 m), and 5
  orders of magnitude below the single-run ATE scatter of ±0.007 m that the accuracy
  protocol already has to average over;
* the 72-run accuracy ensemble in "What it bought" above is the real check, and it
  agrees run-for-run to the scorer's printed precision.
