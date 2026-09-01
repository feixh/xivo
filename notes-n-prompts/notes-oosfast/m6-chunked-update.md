# M6 — a run-aware, chunked covariance update

Branch `auto-covrun` off `auto` @ `017c4a4`. Baseline worktree `xivo-oosbase`,
detached at the same `017c4a4`.

The ask was 1.60 ms/frame out of stereo and ~10 MB out of stereo peak RSS. This
milestone is the answer to both, and they turn out to be the same change.

## 1. Where the update's time actually goes

Nothing here was worth guessing at. `perf record` does not work on this box
(`perf_event_paranoid`), so I put a temporary phase probe inside
`EkfUpdateDowndate` — seven `steady_clock` reads per update, printed from a static
destructor — and ran room5. 2841 updates, ms/update, at `017c4a4` with every
`ekf_update` key off:

| phase | stereo | | mono | |
|---|---|---|---|---|
| `M = H P` | 0.681 | 14.5% | 0.542 | 24.8% |
| `S = M H^T` | 0.269 | 5.7% | 0.088 | 4.0% |
| `LLT(S)` | 0.515 | 10.9% | 0.107 | 4.9% |
| `W = L^-1 M` (trsm) | **1.551** | **32.9%** | **0.536** | **24.5%** |
| `err` | 0.031 | 0.7% | 0.015 | 0.7% |
| `P -= W^T W` | **1.602** | **34.0%** | **0.836** | **38.3%** |
| mirror | 0.064 | 1.4% | 0.059 | 2.7% |
| **total** | **4.713** | | **2.183** | |

The flop model that explains it, with `m` the update's rows and `n` the live
dimension (stereo `m` 360 average / 860 widest, `n` 491):

    trsm      = m^2 n / 2      LLT = m^3 / 6      downdate = n^2 m / 2

The trsm runs at 41 GFLOP/s and the downdate at 47. **Both are already at the
machine's single-core limit**, so no amount of rearranging them helps. That kills
the two obvious ideas at the outset:

- **Fewer dimensions.** The census says `occupied-dim` 466 against `live-dim` 491
  — a 5% ratio. Even a perfect description of the occupied set caps the downdate
  saving at ~10% and the trsm at ~5%. Measured below: it is worth 2%.
- **Fewer passes over the destination.** I predicted `H P`'s four read-modify-write
  passes over a 1.4 MB destination were bandwidth-bound and that fusing them would
  buy 0.55 ms. **Measured: 0.023 ms.** The destination is L3-resident, L3 bandwidth
  is not the constraint, and my traffic model was simply wrong. Recorded here
  because it was the load-bearing assumption of the original plan.

What the model *does* say is that only `m` is a free variable. `n` is the state,
`m` is the measurement — and the measurement can be split.

## 2. Sequential chunks: the same update, not an approximation

Apply the rows in `C` consecutive groups, each against the covariance the previous
groups left behind and with its innovation re-predicted as `r_c -= H_c * err_so_far`.
This is exactly the batch update, for two reasons that both have to hold:

    P+^-1     = P^-1 + sum_c H_c^T R_c^-1 H_c
    P+^-1 err = sum_c H_c^T R_c^-1 r_c

Both right-hand sides are *sums over the chunks*, so processing them one at a time
gives the same left-hand side, provided

1. `R` is block diagonal across the chunks — here `R` is diagonal, so it always is;
2. `H` is held at one linearization point — nothing is re-evaluated between chunks,
   which is why `MeasurementUpdate` still builds `H_` once.

The cost model, per update:

| | batch | `C` chunks |
|---|---|---|
| `LLT` | `m^3/6` | `m^3/6C^2` |
| trsm | `m^2 n/2` | `m^2 n/2C` |
| downdate | `n^2 m/2` | `n^2 m/2` (unchanged) |
| mirror | 1 | `C` |
| `S` + Eigen's copy of its factor | `2 m^2` doubles | `2 (m/C)^2` doubles |

The downdate is untouched because its MAC count is linear in `m`. The mirror is the
new cost: the next chunk forms `H_c P` off both triangles of `P`, so each chunk has
to mirror. That last row is the one I did not expect to matter and which turned out
to be the whole memory story — see §5.

It is a *different factorization*, not a reassociation: `C` Choleskys of `m/C` rows
in place of one of `m`. So it is not bit-identical, and unlike the `live`
restriction there is no argument that it should be. It is better conditioned, since
each `S_c` is a smaller matrix, but the tests below only claim agreement.

### What was rejected on the way

- **One block at a time** (`C` = number of features, ~90). Exact by the same
  argument, and it makes `LLT` and trsm free. Dead on bandwidth: 90 read-modify-write
  passes over the live triangle of `P` is ~173 MB/update, about 8.6 ms. The mirror
  row of the table above is the same effect, which is why `C` has an optimum rather
  than being "as large as possible".
- **Compressing the measurement** (QR down to `n` rows, as OpenVINS does for its
  MSCKF residual). Impossible here: `m` 360 < `n` 491, `H` is already full row rank,
  there is nothing to compress.
- **Tuning `OOS.pose_window` / `max_observations` down**, which is what actually
  sets `m`. Explicitly off the table, and rightly.

## 3. Measured: the chunk sweep

room5, one core, `setarch -R`, phase probe active, ms/update. All rows have
`exact_runs` + `run_gap: 3` + `fuse_passes` on, so the `C` column is the marginal
effect of chunking alone.

| C | stereo total | trsm | LLT | downdate | mirror | mono total |
|---|---|---|---|---|---|---|
| 1 | 4.618 | 1.494 | 0.528 | 1.581 | 0.058 | 2.111 |
| 2 | 3.728 | 0.960 | 0.192 | 1.581 | 0.105 | 1.963 |
| 3 | 3.478 | 0.781 | 0.125 | 1.580 | 0.149 | **1.941** |
| **4** | **3.392** | 0.689 | 0.095 | 1.590 | 0.194 | 1.977 |
| 6 | 3.388 | 0.598 | 0.068 | 1.617 | 0.278 | 2.068 |
| 8 | 3.446 | 0.552 | 0.057 | 1.642 | 0.362 | 2.169 |

Everything the model predicted: trsm falls as `1/C`, `LLT` as `1/C^2`, the downdate
is flat, the mirror grows linearly, and the sum has a minimum. Stereo's is flat
between 4 and 6 (0.005 ms apart, noise); mono's is at 3, earlier because mono's
trsm is a third the size so the linear mirror cost catches it sooner.

**Shipped: stereo `chunks: 4`, mono `chunks: 3`.** Against the all-keys-off
baseline of 4.713 / 2.183 that is **−1.32 ms/update stereo and −0.24 mono**, i.e.
28% and 11% of the update.

`SplitChunks` also refuses to open a chunk below `kMinChunkRows` = 48 rows. What a
split saves is quadratic and cubic in the chunk's rows; what it costs — one more
mirror — does not depend on them at all. From the table the two cross at about 85
rows for the first split, so 48 rows per chunk keeps the benefit on a full update
and turns chunking off on a thin one rather than paying `C` mirrors for a
factorization that was already free.

## 4. The two small keys

Kept, but they are rounding on the above.

- `exact_runs` + `run_gap: 3`: **−0.093 ms/update stereo, −0.034 mono.** `run_gap`
  matters more than `exact_runs` does. The gap tolerance is how many provably-zero
  dimensions a run may absorb rather than split in two; at the default 6 it absorbs
  every single-slot hole and saves essentially nothing (live-dim 488 → 484), and at
  0 it splits into 7.8 runs whose extra gemm and mirror calls cost *more* than the
  39 dimensions they save (downdate 1.581 → 1.644). 3 is the optimum: live-dim 462,
  4.5 runs.
- `fuse_passes`: **−0.023 ms/update stereo, −0.020 mono.** See §1 — this is the
  measurement that showed my bandwidth model was wrong. Kept because it is free and
  not negative, not because it earned its keep.

## 5. The memory, which was the other half of the ask

A parallel probe (`xivo-oosbase` only) attributed the stereo peak-RSS overshoot by
forcing `MALLOC_MMAP_THRESHOLD_=65536` and reconstructing the live mapping set from
an `mmap` strace. It found the gap is **not** the image pyramids (forcing a second
full left pyramid every frame costs +0.2 MB, i.e. noise) and **not** the mapper
(`USE_MAPPER` is not defined). It is four dense matrices that coexist inside one
EKF update, at the widest update of the run:

| buffer | mono | stereo | note |
|---|---|---|---|
| `H_`, `rows x 564` | 2.04 | 3.70 | required |
| `M = H P`, `rows x 564` | 2.04 | 3.70 | required |
| `S`, `rows^2` | 1.71 | **5.64** | |
| Eigen's copy of `S` inside `LLT` | 1.71 | **5.64** | |

room5's widest update is 473 rows mono and **860 rows stereo** — the sizes are
exactly `page_round(rows^2 * 8)`, which is how the identification was made. Stereo
is twice as tall because a stereo view contributes 4 equations to a marginalized
out-of-state track instead of 2, so ~500 of those 860 rows are out-of-state.

`S` is quadratic in `m`. **Chunking makes it quadratic in `m/C`.** With `C` = 4 the
pair falls from 11.3 MB to about 0.7 MB, because a chunk only ever holds the
*diagonal* block of `S` that it owns — the off-diagonal blocks are the
cross-covariances between chunks, and sequential processing folds those into `P`
instead of forming them. So `S` is allocated at the widest chunk, not the widest
update, and `CovTimesMeasurementTRange` indexes its columns relative to the chunk's
first row. This is why the two halves of the ask are one change.

The probe also identified `Eigen::LLT<MatX> llt(S)` as a pure duplication that
`LLT<Ref<MatX>>` would remove outright. That is still true, and still worth doing,
but with `S` down to `(m/C)^2` it is now worth ~0.4 MB instead of 5.6, so it is not
in this milestone. Two other items it found and I did not act on: the `2 *
oos_H_.rows()` doubling in `ReserveOOSRows` leaves ~1.4-2.0 MB unwritten-but-resident
in room1 stereo, and running the out-of-state rows as their own update would cap the
widest buffer set at `max(360, 500)^2` instead of `860^2` — but that one moves the
linearization point of the second block, so it changes the estimate and is a design
decision, not a memory fix.

## 6. Correctness

`unitTests_ekf_update` gained four cases; `ctest` is 22/22.

- `ChunkedEqualsTheBatchUpdate` — at the real dimensions (564 states, 76 features,
  a 12-row dense out-of-state block), for `C` in {2, 3, 4, 8, 16} and both settings
  of `fuse`, against **both** the batch downdate and the independent dense Joseph
  form, on `P` and on `err`, to 1e-9 relative. Plus, for every `C`: exact symmetry
  of `P`, positive variances on every live slot, and a vacant tail that is
  bit-for-bit untouched. This is the test that matters — a subtle error in a
  covariance update would not show up as a divergence, so it is checked against a
  reference that shares no code with it.
- `ChunkingASingleBlockIsTheBatchUpdate` — `C` above the block count, and an update
  whose rows are one indivisible block, both fall back to one chunk. Bit-identical
  there, since there was nothing to split.
- `ChunkedRefusesAnIndefiniteInnovationCovariance` — a first-chunk failure leaves
  `P` untouched, which is what lets the caller still fall back to Joseph over the
  whole batch.
- `ExactRunsCoverEveryOccupiedSlot` — 40 random occupancy patterns with holes, four
  gap tolerances; checks that the runs are a *superset* of the occupied slots
  (which is all correctness needs), ascending, disjoint, in bounds, `dim`
  consistent, and never worse than the high-water cover of the same occupancy.
- `ExactRunsGiveTheSameAnswerAsTheHighWaterCover` — the two descriptions of the same
  set, through the update, to 1e-13.

**The one behaviour change on a path that has never fired.** If a *later* chunk's
`S_c` has no Cholesky factor, `P` already carries the earlier chunks and cannot be
rolled back, so the chunk is dropped with a `LOG(WARNING)` and the update returns
true. Dropping measurements leaves the filter consistent — it is the same as their
having been gated out — but it is not the same as the batch Joseph fallback. The
alternative was a 2.5 MB snapshot of `P` per update, which would have spent a fifth
of the memory saving on a path that has never been reached on TUM-VI. Flagged
rather than hidden.

## 7. Numbers

See `config-delta.md` for the config keys.

### Throughput and peak RSS

Paired, same conditions, both worktrees built the same way and run back to back:
`run_xivo_reference.sh --timing --no-score`, `CPU_BASE=0 CPU_SPAN=60`, one core per
run, `setarch -R`, serial, all pools pinned to 1 thread. 6 sequences (room1-6),
`stats.txt` per run. `base` is `xivo-oosbase` detached at `017c4a4`; `chunked` is this
branch with the shipped keys.

| | base | chunked | ratio / delta |
|---|---|---|---|
| mono FPS (mean of 6) | 120.1 | 123.4 | **1.028** |
| mono ms/frame | 8.325 | 8.102 | −0.223 |
| mono peak RSS, max / mean (MB) | 89.6 / 86.4 | 88.0 / 84.8 | −1.6 / −1.7 |
| stereo FPS (mean of 6) | 64.0 | 70.1 | **1.096** |
| stereo ms/frame | 15.633 | 14.258 | **−1.375** |
| stereo peak RSS, max / mean (MB) | 106.4 / 101.5 | **95.9 / 92.5** | **−10.5 / −9.0** |

Per-sequence stereo ratios are 1.103 / 1.091 / 1.081 / 1.100 / 1.095 / 1.109 — a
0.028 spread, so the mean is not carried by one sequence. The `base` column
reproduces the merge-time authoritative numbers (mono 120.3, stereo 64.0, stereo peak
RSS 105.2 max / 101.4 mean) to within run-to-run noise, which is what licenses
applying these ratios to them: mono **123.7 FPS**, stereo **70.1 FPS** (0.984x of the
71.3 target, up from 0.90x), stereo peak RSS **94.7 max / 92.4 mean** against the
95.5 target.

Against the ask — 1.60 ms/frame and ~10 MB — this is 1.375 ms (86%) and 10.5 MB
(met). The 0.225 ms shortfall is the difference between the per-update saving
(−1.32 ms of a 4.71 ms update on room5) and the per-frame saving, i.e. updates do not
happen on every frame.

### Accuracy

72 paired runs: `--jitter 6`, 6 sequences, both modes, both worktrees. **415 of the
432 printed metric values are identical run for run.** Mono is identical on all 216.
Stereo differs on 3 of its 36 runs, all of them room6 — the sequence already known to
be chaotic on this codebase, where single-run ATE scatters by ~0.007 m for reasons
that have nothing to do with this change.

Ensemble means (mean ± sd over the 6 per-sequence means), with the OpenVINS floor
this may not cross:

| metric | base | chunked | delta | floor |
|---|---|---|---|---|
| mono ate_002 | 0.0555 ±0.0212 | 0.0555 ±0.0212 | 0 | 0.0621 |
| mono ate_ori | 0.8788 ±0.3217 | 0.8788 | 0 | 1.5742 |
| mono rpe8_pos | 0.0265 ±0.0059 | 0.0265 | 0 | 0.0308 |
| mono rpe8_ori | 0.5131 ±0.0725 | 0.5131 | 0 | 0.6445 |
| stereo ate_002 | 0.0492 ±0.0133 | 0.0490 ±0.0135 | −0.00014 | 0.0677 |
| stereo ate_ori | 0.8924 ±0.3058 | 0.8921 | −0.00022 | 1.4440 |
| stereo rpe8_pos | 0.0216 ±0.0026 | 0.0215 | −0.00006 | 0.0265 |
| stereo rpe8_ori | 0.5156 ±0.0763 | 0.5161 | +0.00056 | 0.5837 |

Stereo improves on four of five metrics and regresses on one by 0.7% of that metric's
ensemble sd, which is not a measurement. **No accuracy margin was spent.** That is
the expected outcome given §2 and §6: it is the same update, and the only reason the
numbers are not bit-identical is that `C` small Choleskys are not the same
floating-point operations as one large one.

Result directories: `experiments/openvins/results/covrun-acc-xivo-{oosbase,oosfast}-{mono,stereo}`
(accuracy) and `covrun-t-xivo-{oosbase,oosfast}-{mono,stereo}` (timing; no
`summary.csv`, since `--no-score` skips scoring — read `stats.txt` per run dir).
