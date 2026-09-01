# M4 -- where low precision can go, and the reformulation that pays for itself

M3 ended with bf16 collapsing the filter and one number explaining it: an update
moves the covariance by ~5e-3 relative, and a bf16 product is wrong by ~3e-3.
The obvious repair is to put the low precision on the *correction* instead of on
the covariance, so that the same 3e-3 applies to a quantity 200x smaller. This
milestone does that, and then finds out that it is not sufficient -- for a
reason that turns out to be a hard limit on bf16 in an EKF, not a tuning
problem.

The efficiency win is real and lands anyway: **4.1x fewer multiply-adds in the
update, with the trajectory unchanged to 1e-11 m.**

## The short form

```
P+ = P - K (H P),   symmetrized     "short"      (kernel_precision.covariance_form)
P+ = (I - KH) P (I - KH)^T + K R K^T             "joseph"  (default)
```

The two are the same expression given the optimal gain. Expanding the Joseph
form gives `P - K(HP) - (HP)^T K^T + K S K^T` with `S = H P H^T + R`, and
`K = P H^T S^-1` makes `K S K^T = K (H P)`, so two of the three correction terms
cancel. XIVO's gain comes from `S.ldlt().solve(HP)`, i.e. it *is* the optimal
gain, so the identity applies here.

| | Joseph | short |
| --- | --- | --- |
| products | `K H` (n m n), `A P` (n^3), `(AP) A^T` (n^3), `K R K^T` (n m n) | `K (H P)` (n m n) |
| multiply-adds at n = 564, m = 180 | 5.49e8 | 1.33e8 |
| symmetric by construction | no (two products) | no (one product) |
| stays PSD under a suboptimal gain | yes | no |

The last row is the reason the Joseph form exists, and the reason `joseph`
remains the default. What it buys here is measurable, and it is nothing:

`XIVO_DIAG_UPDATE=1` computes both forms every 40th update and reports
`|joseph - short| / |dP|` -- the disagreement measured against the size of the
update itself, which is the only scale on which it matters. Over mono room1:

```
|form-short|/|dP| = 5.5e-16 ... 2.6e-11    (median ~ 5e-12)
```

Twelve digits of agreement relative to the correction, i.e. fp64 roundoff. And
end to end, at fp64, with `XIVO_DUMP_PRECISE=1` so the dumped trajectory is
round-trippable rather than the usual six decimals:

| mode | max position difference vs the shipped Joseph path, over 2771 poses |
| --- | --- |
| mono room1 | **1.4e-11 m** |
| stereo room1 | **3.3e-11 m** |

Not one gating decision flips in either mode over 130 s. (For comparison, in the
same measurement the `f64` and `batch` arms are *exactly* bit-identical to the
shipped path, so the 1e-11 is genuinely the short form's own roundoff and not an
artefact of the harness.)

`mindiag` of P is 0 at every frame in both forms -- the gauge-fixed states carry
a hard zero variance and neither form perturbs it, because `P`'s zero rows make
`K`'s corresponding rows exactly zero, and zero survives every rounding.

## Why bf16 still fails, and why nothing fixes it

The short form does exactly what was intended to the *average* error. Measured
in the filter (`DIAGK`, which computes `K (H P)` at all three precisions and
compares against fp64):

| | Frobenius relative error | worst elementwise relative error |
| --- | --- | --- |
| f32 | 3e-8 .. 4e-7 | 1e-5 .. **4e-2** |
| bf16 | 8e-4 .. 3e-3 | 1.8 .. **660** |

The norm column is what a synthetic benchmark reports, and it is exactly the
3e-3 the M3 kernel table predicted. The second column is the one that decides
whether the filter survives: **individual entries of the bf16 correction are
wrong by factors of 100 to 600.**

Both columns are consistent with a single mechanism. The error in `C(i,j) = sum_k
A(i,k) B(k,j)` is ~eps times the sum of the *absolute* partial products, while
`C(i,j)` itself is their signed sum. `K (H P)` is a projection-like operator:
its small entries are differences of much larger terms. So the error is set by
the row/column scale and the small entries -- which belong to the *well-converged*
states, the ones whose variance the filter is relying on being small -- are
swamped. An EKF covariance whose diagonal spans six orders of magnitude cannot
be updated by a product with 8 significand bits, whatever the product is of.

Nothing in the arrangement fixes this, and the arithmetic says so:

* **Diagonal scaling / shared exponents (MX-style blocks) cannot help.** Scaling
  the inner index by `d_k` -- `A(i,k)/d_k` and `d_k B(k,j)` -- leaves every
  partial product unchanged, so it leaves the error unchanged. Scaling a row of
  A divides every term of that row equally, so it leaves the *relative* error
  unchanged. The problem is cancellation in the sum, and no diagonal
  transformation touches cancellation.
* **Split (hi + lo) bf16 is dominated by fp32 on this hardware.** Two bf16
  products give ~16 significand bits against fp32's 24, and measured on the
  Joseph shape they cost 2 x 4.11 = 8.22 ms against fp32's 7.23 ms. Strictly
  worse on both axes. This is a property of Zen 4's 1.33x bf16:fp32 issue ratio
  (312.1 vs 234.2 GFLOP/s, M0) -- on hardware where bf16 is 8x fp32 (AMX, or a
  GPU tensor core) the same scheme would win, and this conclusion would flip.
* **Iterative refinement** needs a second product for the residual, so it lands
  in the same place as the split.

So fp32 is the floor for any product whose output has to keep its small entries,
and that is a hardware fact on this machine rather than a limitation of the
kernel.

## fp32 is below the floor too, in mono

fp32 on the correction product passes every single-run check (the arm table
below) and passes the stereo ensembles cleanly. It fails the mono ensembles, on
one sequence:

| arm | mono | stereo |
| --- | --- | --- |
| `short_f32c_gbf16` (f32 correction, fp64 gain) | **2/36 diverge**, both room3 | 36/36 clean |
| `short_f32_gbf16` (f32 correction *and* gain) | **4/36 diverge**, all room3 | 36/36 clean |

Two isolations say what this is and what it is not.

**It is the correction product.** `short_jf32` narrows nothing but `K (H P)` --
gain path and gating both fp64 -- and on mono room3 it loses the *same two
members* (m0, m4) with the same magnitudes. So bf16 in the gating sweep is not
implicated; one fp32 matrix product is sufficient on its own.

**It is not chaos.** The natural suspicion is that room3 sits on a cliff and any
last-bit change tips it, in which case fp32 would be innocent and the baseline
equally fragile. Measured: fp64 `short` on room3 with the initial velocity
perturbed by `k * 1e-3` and `k * 1e-2` m/s -- up to 5e-2 m/s, four to five
orders of magnitude larger than the 1e-6 the ensembles use -- gives 12/12 clean
runs, ATE 0.067 .. 0.091. A physical perturbation 50000x the ensemble's is
harmless; fp32 rounding inside the covariance is not. (`ens.sh` grew `SEQS` and
`VSCALE` for these two tests.)

**It happens at initialization.** The diverging runs are not slow drifts:

| run | first \|T\| > 2 m | first > 1000 m | max \|T\| |
| --- | --- | --- | --- | --- |
| `short` m0 (fp64) | 27.5 s | never | 2.8 m |
| `short_f32c_gbf16` m0 | **0.8 s** | 12.4 s | 9.4e4 m |
| `short_f32c_gbf16` m4 | 2.7 s | 9.6 s | 2.2e4 m |
| `short_f32_gbf16` m4 | 3.3 s | 22.8 s | 3.2e3 m |

Everything blows up in the first one to four seconds, which is consistent with
the mechanism above and explains the mono/stereo split: at init the covariance
spans its widest range of scales and the filter has the fewest measurements to
recover from a bad correction. Stereo doubles the measurement count per frame
and never gets into that state. This is also why room1 smoke tests, and the
single-run arm table below, cleared arms the ensembles reject -- room1 has an
easy start.

fp32 in the covariance is therefore out of the deliverable. It would need an
fp64 warmup, i.e. a second precision knob whose threshold is fitted to one
sequence of one dataset, to buy ~1.2 ms/frame on top of a change that already
saves ~22. That trade is not worth making.

## Where bf16 does belong

The gating sweeps. `J_i P J_i^T` is a 2x2 whose two eigenvalues are dominated by
the largest terms, feeding a chi-square threshold; a 1e-3 error on a Mahalanobis
distance changes an accept/reject decision only for a measurement sitting on the
threshold. The output has no dynamic range to lose. Measured: `gate_bf16`
(bf16 gating, everything else fp64) gives mono room1 ATE@0.001 = 0.069831
against the baseline's 0.071380, well inside the +-0.0034 ensemble sd, and gets
the sweep from 2.36 to 1.17 ms.

That is the rule the arms below confirm: **narrow what is discarded at the end
of the frame; keep what is integrated.** A gating statistic is discarded. The
covariance is integrated, and its small entries are the whole point of keeping it.

## Arms, mono room1, single runs

Single runs, so read only the order of magnitude -- the six-member sd is
+-0.0034 and the ensembles in `m5-*.md` are what decide between the survivors.

| arm | form | joseph / innovation / gating | ATE@0.001 | verdict |
| --- | --- | --- | --- | --- |
| baseline | joseph | f64 f64 f64, unbatched | 0.071383 | reference |
| `batch` | joseph | f64 f64 f64 | 0.071383 | bit-exact |
| `short` | short | f64 f64 f64 | 0.071383 | 1.4e-11 m |
| `short_gbf16` | short | f64 f64 **bf16** | 0.069831 | **the deliverable** |
| `short_jf32` | short | **f32** f64 f64 | 0.058745 | **rejected** (room3, 2/6) |
| `short_f32c_gbf16` | short | **f32** f64 **bf16** | -- | **rejected by ensemble** |
| `short_f32_gbf16` | short | **f32 f32 bf16** | 0.063836 | **rejected by ensemble** |
| `short_f32` | short | **f32 f32 f32** | 0.067513 | same product, rejected |
| `short_jbf16` | short | **bf16** f64 f64 | 2405.8 | **diverges** |
| `short_bf16c` | short | **bf16** f64 **bf16** | 11415.7 | **diverges** |
| `short_bf16` | short | **bf16 bf16 bf16** | 102183.2 | **diverges** |
| `jos_bf16` | joseph | **bf16** f64 bf16 | *aborts* | **diverges** |
| `inn_bf16` | joseph | f64 **bf16** bf16 | 161973.1 | **diverges** |
| `bf16` | joseph | **bf16 bf16 bf16** | 5137691.4 | **diverges** |

Every arm containing a narrowed *covariance* or *gain* product at bf16 diverges.
The short form does not change that -- `short_jbf16` (2405 m) fails just as
`jos_bf16` does, only less spectacularly, which is the 200x showing up as three
fewer orders of magnitude of nonsense rather than as survival.

Read this table as a *screen*, not as a verdict. Single runs on room1 rank the
f32 arms as the best in the table, and the ensembles then reject all of them --
which is the whole reason the ensemble harness exists.

## What survives: `short_gbf16`

`covariance_form = short`, `batch_gating = true`, gating sweeps at bf16,
covariance and gain products at fp64. Six members x six rooms, both modes,
against the shipped baseline:

| | ATE@0.001 | ATE@0.02 | RPE_tra | RPE_rot | diverged |
| --- | --- | --- | --- | --- | --- |
| mono baseline | 0.0686 +- 0.0034 | 0.0852 | 0.0213 | 0.6203 | 0/36 |
| mono `short` | 0.0681 (t -0.25) | 0.0863 | 0.0215 | 0.6207 | 0/36 |
| mono `short_gbf16` | 0.0704 (t **+1.07**) | 0.0866 (+0.50) | 0.0215 (+0.37) | 0.6204 (+0.75) | 0/36 |
| stereo baseline | 0.0453 +- 0.0031 | 0.0591 | 0.0132 | 0.6215 | 0/36 |
| stereo `short` | 0.0448 (t -0.30) | 0.0579 | 0.0132 | 0.6215 | 0/36 |
| stereo `short_gbf16` | 0.0437 (t **-1.36**) | 0.0539 (-3.91) | 0.0133 (+0.57) | 0.6215 (+0.05) | 0/36 |

Mono is +0.0018 (0.5 sd), stereo is -0.0017; the two have opposite signs and
neither is resolvable, which is what "no degradation" looks like when the
measurement is honest about its own noise. The stereo ATE@0.02 t of -3.91 is an
improvement, and not one to claim: it is the same chaotic reshuffling of accepted
features, sampled favourably.

## Cost of the pieces, mono, from the M3 kernel table

| | f64 | with the change |
| --- | --- | --- |
| gating sweep, per-feature -> batched (fp64) | 9.88 | 2.36 |
| batched sweep, fp64 -> bf16 | 2.36 | 1.17 |
| update, Joseph -> short (fp64) | ~19.0 | ~4.9 |
| short update, fp64 -> f32 on `K (H P)` and the gain path | ~4.9 | ~2.8 (**rejected**) |

Reading down the column: of the ~26 ms/frame the baseline spends on dense
covariance algebra, the two *algebraic* changes (batching, short form) remove
~22, bf16 on the gating sweep removes ~1.2 more, and the last row is the one the
ensembles took away. The headline of this branch is a reformulation; bf16's own
contribution is the gating sweep. That is the opposite of what the milestone plan
assumed, and it is what the measurements say.
