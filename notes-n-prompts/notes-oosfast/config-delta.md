# Config delta — branches `auto-oosfast`, `auto-covrun` and `auto-covscratch`

Filter/estimator side of task 3, three rounds. The front end, the stereo matcher and the
image-decode path belong to another agent and were not touched in any of them.

| round | branch | based on | milestones |
| --- | --- | --- | --- |
| 1 | `auto-oosfast` | pre-merge `auto` | m1-m5; merged as `b0d7ec5` |
| 2 | `auto-covrun` | `auto` @ `017c4a4` | m6; merged as `b565b25` |
| 3 | `auto-covscratch` | `auto` @ `b565b25` | m7; one commit, `48884e5` |

**This file is the index; there is no `summary.md`, because the environment I run in
refuses to create one under that name.** Reading order: this file, then m1 (the
sparsity shape), m2, m3, m4 (the memory probe), m5, m6, m7.

Final paired throughput / RSS / accuracy tables live per round, in **the "What it
bought" section of m5-shared-oos-buffer.md** (round 1), **m6 §7** (round 2) and **m7 §4**
(round 3). Do not compare their absolutes to each other: round 1's baseline predates the
front-end agent's merge, so mono baseline FPS is 83.0 there, 120.1 in m6 and 123.5 in m7.

Round 1 was mono **1.228x** / stereo **1.125x** with peak RSS −7.7 / −8.0 MB max and
ATE unchanged to five decimals. Round 2 is mono **1.028x** / stereo **1.096x**
(−1.375 ms/frame) with stereo peak RSS **−10.5 MB** max, and 415 of 432 printed
accuracy values identical run for run. Round 3 is mono **1.0012x** / stereo **1.0023x**
(−0.033 ms/frame) and is **byte-identical on all 12 real runs** — it spends no accuracy
because it changes no arithmetic. No accuracy margin was spent in any round.

**Round 3 did not reach its ask and says so.** It was scoped to the last 0.22 ms/frame of
stereo, on the hypothesis that `Eigen::LLT`'s copy and the per-update allocation churn
were worth 0.1-0.2 ms of it. Measured: 0.033 ms, ~13%. m7 §6 accounts for the remaining
~0.19 ms and shows there is no route to it that is both exact and bit-identical.

Eight keys total across the three rounds, plus one environment variable. **Nothing
existing was retuned**: no OOS parameter, no threshold, no noise, no `subpix_refine`.
Three things a reviewer should know before touching any of it are the `run_gap` warning,
the chunk-drop behaviour change, and the tight-map requirement behind `inplace_llt`, all
flagged in their sections below.

Ideas costed and rejected, named here so nobody re-derives them: tuning
`OOS.pose_window` / `max_observations` down (ruled out by the contract, and the storage
fixes beat it — m4); compressing the measurement by QR the way OpenVINS does for its
MSCKF residual (impossible, `m` 360 < `n_live` 491 and `H` is full row rank — m6 §2);
one measurement block at a time (exact, makes the factorizations free, dead on
bandwidth at ~173 MB/update — m6 §2); `ReserveOOSRows`' 2x growth (fires ~10 times per
run, so ~50 MB of copying against ~2900 frames — unmeasurable as time, and its 1.4-2.0 MB
of resident pages is not needed now the memory target is met — m7 §5); reusing the
update's scratch across updates (implemented, measured at nothing, costs 2.3 MB — m7 §5).
Two priced-but-not-taken items remain, both in **m7 §6**: `chunks: 5` (−0.017 ms, but not
bit-identical, so it needs its own ensemble) and reading only the lower triangle of `P` in
`H_c P` so the per-chunk mirror shrinks (~0.09 ms, transposes gemm operands, likewise not
bit-identical).

## `oos_fast.enable` — new, default `false`

```jsonc
  // Column-sparse form of the out-of-state and feature-init products: same
  // matrices, formed over the ~36 error-state columns an out-of-state or
  // feature-init Jacobian can reach instead of all 564.
  "oos_fast": { "enable": true },
```

* Parsed in `Estimator::Estimator` into `OOSOptions::fast_sparse`
  (`cfg_["oos_fast"].get("enable", false)`), so a config that does not mention the key
  gets the previous code path.
* Its own top-level section rather than a member of `OOS`, because it also governs
  `consistent_init`'s promotion products (`Estimator::InitializeFeatureCovariance`),
  which are not part of the OOS update.
* Set to `true` in `cfg/eff_mono.json` and `cfg/eff_stereo.json` (line 42, immediately
  before the `OOS` block). **`sed 's/"enable": true/"enable": false/'` on these files
  is a trap**: it also flips `consistent_init.enable` at line 223 and produces a 7e-2 m
  "divergence" that is not one. Use `sed '42s/...'`.
* What it turns on, per milestone note: M1 (scratch clear + `OOSGating`), M2
  (`ComputeInitJacobian` + `InitializeFeatureCovariance`), M3 (the `MeasBlock::runs`
  the stacked update gets for out-of-state blocks).
* Off, the only remaining difference from HEAD is storage lifetime (M4/M5), which has
  no key. See m5-shared-oos-buffer.md for why a config key would not have bought
  bit-identity anyway: HEAD's own output moves by 4.3e-12 m under a pure allocator
  tunable, 130x more than this branch's key-off deviation.

## `ekf_update.*` — new section, branches `auto-covrun` (4 keys) and `auto-covscratch` (3 more), every key defaults to the old behaviour

Seven keys, all under one new top-level `"ekf_update"` object, parsed in
`Estimator::Estimator` just before the `fej` block. A config that does not mention the
section takes the previous code path in all seven. Rationale and measurements in
`m6-chunked-update.md` and `m7-update-scratch.md`; the short version is that only
`chunks` and `inplace_llt` matter, and `chunks` matters ~50x more.

| key | default | shipped mono | shipped stereo | worth |
| --- | --- | --- | --- | --- |
| `chunks` | `1` (batch) | `3` | `4` | **−1.32 ms/update stereo, −0.24 mono; and 11.3 MB → 0.7 MB of stereo peak RSS** |
| `exact_runs` | `false` | `true` | `true` | part of −0.093 / −0.034 ms |
| `run_gap` | `6` (`kGroupSize`) | `3` | `3` | the larger part of that; see below |
| `fuse_passes` | `false` | `true` | `true` | −0.023 / −0.020 ms |
| `inplace_llt` | `false` | `true` | `true` | −0.026 ms/frame stereo, −0.003 mono; ~0.4 MB. **Bit-identical** — see the warning below |
| `reuse_scratch` | `false` | `false` | `false` | nothing (−0.008 ms), and +2.3 MB of mean peak RSS. **Deliberately off**; kept only as the A/B behind that claim |
| `zero_unused` | `true` (do the zeroing) | `false` | `false` | ~0.005 ms, i.e. inside noise; removes ~0.7 MB of provably dead stores per stereo update |

* **`chunks`** applies the update's rows in that many consecutive groups, each against
  the covariance the previous ones left and with its innovation re-predicted. This is
  the *same* update, not an approximation — the information form is additive in the
  chunks whenever `R` is block diagonal across them (it is diagonal) and `H` is held
  at one linearization point (it is). Clamped to `[1, kMaxUpdateChunks]` = `[1, 16]`
  on read, and `SplitChunks` refuses to open a chunk below 48 rows, so a thin update
  silently runs as one chunk however the key is set.
* **`run_gap`** is easy to get wrong and is the one number here I would not change
  without re-measuring. It is how many provably-zero dimensions the exact live extent
  may absorb rather than split a run in two. **At the default 6 it absorbs every
  single-slot hole and `exact_runs` saves almost nothing**; at 0 it splits into ~8
  runs, and the extra gemm and mirror calls cost *more* than the dimensions they save
  (the downdate goes 1.581 → 1.644 ms). 3 measures best in both modes.
* **A behaviour change on an unreached path.** With `chunks > 1`, a *later* chunk whose
  `S_c` has no Cholesky factor is dropped with a `LOG(WARNING)` instead of triggering
  the whole-batch Joseph fallback, because `P` already carries the earlier chunks. A
  first-chunk failure still returns false with `P` untouched. Neither the new path nor
  the fallback it replaces has ever fired on TUM-VI.
* **`inplace_llt` has a precondition that is not visible at the call site.** It
  factorizes each chunk's `S` in place with `Eigen::LLT<Eigen::Ref<MatX>>`, and it is
  bit-identical *only because `S` is a tight `Map<MatX>` onto the front of a flat buffer*.
  Factorizing a strided **block** of a larger matrix in place is **not** bit-identical —
  the alignment of a column decides which load path Eigen's kernel takes; measured at
  ~1e-17 relative for k in {13, 97, 215, 360}. So do not refactor `sc.s` into a
  `kmax`-square `MatX` with `topLeftCorner(k, k)` views, and do not give `M` a
  `topRows()` view either. The helpers take non-const `Eigen::Ref<MatX>` precisely so
  that getting this wrong is a compile error rather than a silent copy. (`LLT<Ref<...>>`
  also needs a *named non-const lvalue* to bind its in-place constructor; an rvalue block
  binds the copying one, which for a `Ref` fails to compile rather than copying quietly.)
* **`zero_unused: false` rests on a proof, not an argument.** The columns of `H P` outside
  the live extent are written and never read — every read below is at `JacFixedSpan()`
  (inside the motion block, which is always the head of live run 0), a group run, a
  feature run, or `DenseSumRuns`' output. `NothingOutsideTheLiveExtentOfHPIsEverRead`
  poisons the whole scratch with NaN and checks the answer does not move.
* **One unconditional change, not behind a key.** `S_c` is now a tight `k x k` map where
  it used to be `S.topLeftCorner(k, k)`, so `CovTimesMeasurementTRange` writes to a
  destination with a different outer stride. `unitTests_ekf_update` pins all eight key
  combinations to each other bit-for-bit and the 12-run trajectory check pins one of them
  to `b565b25` byte-for-byte, so key-off is byte-identical too — but that rests on
  measurement, not on a guarantee about gemm destination strides. m7 §7.
* Turning the section off is a genuine A/B: `"ekf_update": {}` (or deleting it) gives
  the `017c4a4` code path exactly, since all seven keys gate at the read site.

## `XIVO_DUMP_PRECISE=1` — new environment variable, `scripts/savers.py`

`EvalModeSaver.onResultsReady` wrote `fmt='%f'` — six decimals, 1 um. An md5 match on
that dump proves agreement only to 1e-6 m, which is useless for comparing two builds
of an EKF; a previous agent published a false "bit-identical" claim that way, and I
nearly repeated it. With `XIVO_DUMP_PRECISE=1` the dump is `%.17g`, i.e.
round-trippable, and an md5 comparison means what it says. Default unchanged, so the
TUM scoring scripts see exactly the same files as before.

## Not changed, on purpose

| key | why |
| --- | --- |
| `OOS.pose_window`, `min_observations`, `max_observations` | tuning these down is the wrong lever and was explicitly ruled out; the memory probe in m4 shows the storage fix beats all three anyway |
| `subpix_refine` | 0.06 ms for −0.0062 m ATE; nobody should touch it |
| `consistent_init.*` | M2 makes its promotion products cheaper, not weaker; `consistent-init:17511/17743` is unchanged |
| `use_OOS` | stays `true` |
