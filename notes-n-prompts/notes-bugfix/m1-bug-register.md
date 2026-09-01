# M1 — bug register

Consolidated from five independent audits (four sub-agents on disjoint
subsystems, plus my own differential/reproducibility work in M0). Every
entry below was **re-verified by reading the code myself** before it was
acted on; a handful of sub-agent claims did not survive that and are
listed at the bottom.

"Live" means: reached on a monocular TUM-VI run with
`cfg/sweep_dlt_nodesc.json` (the config the shipped results were produced
with). "Dormant" means the code is wrong but a config flag keeps it
unreachable — still fixed, because the exit criterion is *the code is free
of bugs*, not *this config works*.

Entries #1–#55 came out of the M1 audits. #56–#64 were found while *fixing*
the earlier ones and are appended in discovery order rather than by severity;
#64 is the most consequential single finding in the project and is documented
in `m6-numerics-and-plumbing.md` §7. Liveness verdicts for #35, #39, #46 and
#52 were corrected in M6 after empirical testing — the notes column records
the corrected value, and the reasoning is in that same §6.

## Severity ranking

| # | defect | file | live? | milestone |
|---|---|---|---|---|
| 1 | `FillJacobianBlock` writes both ref-group blocks to `goff` | `feature.cpp:688` | **live** | M2 |
| 2 | `AdaptInitialDepth` "median" of an unsorted vector | `manager.cpp:271` | **live** | M2 |
| 3 | `use_prediction` declared in 23 configs, read by nothing | `tracker.cpp` | **live** | M3 |
| 4 | `match_dropped_tracks` re-drops every rescued track (by-value param) | `tracker.cpp:222,624` | dormant | M3 |
| 5 | `status[]` indexed post-BRIEF-compaction → wrong track killed | `tracker.cpp:556` | dormant | M3 |
| 6 | `CheckHomography` throws away the reprojection it computes | `tracker.cpp:827` | dormant | M3 |
| 7 | `findHomography` failure reported as success → null-mask deref | `tracker.cpp:754` | dormant | M3 |
| 8 | `keypoint().pt` (first detection) used as previous-frame pixel | `tracker.cpp:280,406` | dormant | M3 |
| 9 | `num_outliers_rejected_` stale across frames on early return | `tracker.cpp:715` | dormant | M3 |
| 10 | `CandidateComparison` discards `score1`/`score2`; pointer-order ties | `options.cpp:60` | **live** | M3 |
| 11 | `fl_` constructor formula is 1/√2 of `UpdateState`'s | `camera_manager.cpp:56` | **live** | M3 |
| 12 | `MaskOut` caches `half_size` in a function-local `static` | `tracker.cpp:765` | dormant | M3 |
| 13 | uninitialised `bool` read as left operand of `&&` | `tracker.cpp:597` | **live** (UB) | M3 |
| 14 | `<<` binds tighter than `?:` in a startup log | `tracker.cpp:181` | **live** (log) | M3 |
| 15 | `UpdatePointCloud` budget decremented outside the guard | `tracker.cpp:702` | dormant | M3 |
| 16 | `FastBrief::meanValue`: `memset(&mean,…)`, bogus bit index | `fastbrief.cpp:22,35` | dormant | M3 |
| 17 | DLT-SVD divides by `V(3,3)`≈0, returns `true` → NaN depth "successful" | `helpers.cpp:126` | **live** | M4 |
| 18 | `Triangulate` range test lets NaN through into the success branch | `feature.cpp:755` | **live** | M4 |
| 19 | unclamped `acos` → NaN silently *disables* the reprojection gate | `helpers.cpp:346` | dormant | M4 |
| 20 | `dV_dWsg` uses `Rsb` where the right-perturbation convention needs `Rsg` | `estimator.cpp:647` | **live** | M4 |
| 21 | `initial_std_x/y_badtri` never converted from pixels | `estimator.cpp:356` | **live** | M4 |
| 22 | `feature_owner_change_cov_factor` read under a key no config defines | `estimator.cpp:373` | **live** | M4 |
| 23 | `LinfAngular` never normalises the plane normal | `helpers.cpp:300` | dormant | M4 |
| 24 | DLT-avg inverts a singular 2×2 without checking | `helpers.cpp:147` | **live** (alt) | M4 |
| 25 | in-state features re-anchored *after* their Jacobians are computed | `manager.cpp:72,86,104` | **live** | M5 |
| 26 | `ChangeOwner` transforms only the dead local `P_`, not the filter block | `feature.cpp:235` | **live** | M5 |
| 27 | `inflate_cov` scales the dead local `P_`, and runs on the failure path | `graph.cpp:191` | **live** | M5 |
| 28 | `FindNewGaugeFeatures` NT==9 fallback desyncs `gauge_features_` from the returned list | `graph.cpp:341` | **live** | M5 |
| 29 | re-anchored gauge features keep GAUGE + zeroed x,y under a new owner | `graph.cpp:186` | **live** | M5 |
| 30 | `SetStatus(NULLREFED)` written after `DiscardFeatures` released the slot | `manager.cpp:326` | **live** | M5 |
| 31 | `PointsAreCollinear` unnormalised + order-dependent → non-reproducible | `geometry.cpp:162` | **live** | M5 |
| 32 | `FeatureCovComparison` returns `<=`, used as a `std::sort` comparator | `estimator.cpp:1454` | live (accessors) | M6 |
| 33 | `Givens()` rotates only `Hf.cols()` columns of `Hx` | `helpers.cpp:73` | dormant | M6 |
| 34 | `SlowGivens()` non-orthonormal basis, ignores `effective_rows` | `helpers.cpp:13` | dormant | M6 |
| 35 | `dA_dAu()` returns uninitialised memory; `dAB_dA/dB` flatten output rows column-major while everything else is row-major | `rodrigues.h:11,158,223` | dormant (calib flags off) | M6 |
| 36 | `Feature::Merge` fused covariance collapses to `I`; mean weights swapped | `feature.cpp:194` | dormant | M6 |
| 37 | `RefineDepth` two-view comparator compares `o1` with itself | `feature.cpp:309` | dormant | M6 |
| 38 | `RefineDepth` gates summed residual against a per-observation threshold | `feature.cpp:416` | dormant | M6 |
| 39 | Equidistant `UnProject`: mirrored ray for θ ≥ π/2; `rth` is 0/0 on the line x==cx; Jacobian uses `x1` from the previous iterate (uninit if `max_iter_==0`) | `camera_equidist.h:115,139,152` | **live — the only live M6 fix** | M6 |
| 40 | ATAN singular branches leave Jacobian off-diagonals uninitialised; wrong limit | `camera_atan.h` | dormant | M6 |
| 41 | `PrinceDormandStep` always returns 0 → adaptive step control is dead | `princedormand.cpp:216` | dormant | M6 |
| 42 | accessor row counts use `std::max` where the fill loop caps at `n_output` | `estimator_accessors.cpp` ×9 | live (output) | M6 |
| 43 | `InstateGroupCovs` resets `cnt` per row → 15 of 21 columns never written | `estimator_accessors.cpp:630` | live (output) | M6 |
| 44 | `gauge_group_ptr_` left dangling by `DiscardGroup` | `estimator.cpp:1321` | dormant | M6 |
| 45 | `Qmodel_` added per call, not scaled by `dt` | `estimator.cpp:590` | dormant (zero) | M6 |
| 46 | `State::td` never initialised in the mono build, and read | `core.h:131` | UB, but not result-affecting (all readers are ifdef'd) | M6 |
| 47 | `GoodTimestamp` compares ms-truncated ns timestamps; `dt == 0` only | `estimator.cpp:706` | dormant | M6 |
| 48 | `Track::Reset` does not clear `descriptors_` → pool reuse inherits them | `feature.h:38` | dormant | M6 |
| 49 | `~Estimator` joins a `for(;;)` worker with no stop flag | `estimator.cpp:94` | dormant | M6 |
| 50 | `Givens` precondition `n ^ 1` where `n % 2 == 0` was meant | `helpers.cpp:44` | dormant | M6 |
| 51 | group added to state even when its branch added zero features | `manager.cpp:561` | dormant | M6 |
| 52 | `InitializeJustCreatedTracks` badtri branch is unconditionally taken, shadowing the sim branch | `manager.cpp:592` | dormant (`sim_initialize_depths_` off) | M6 |
| 53 | `-ate_max_difference` parsed and never used | `run_and_eval_pyxivo.py` | harness | M6 |
| 54 | `SlowAndFastGivensMatch` asserts element equality of two different bases | `unittest_givens.cpp:87` | test | M6 |
| 55 | `Triangulation.Angular_Reprojection_Error` — the *test* is right, the code is wrong | — | test | M4 |
| 56 | `Qmodel` loader reads 3 of its 8 keys; `Tsb`/`Vsb`/`wb`/`ab`/`Tbc` silently ignored (`cfg/pcw.json` loses `Vsb: 0.01`) | `estimator.cpp:590` | dormant (zero on TUM-VI) | M6 |
| 57 | `Qmodel_` block squared by matrix self-multiplication `B *= B` instead of squaring the std devs | `estimator.cpp:590` | dormant (zero) | M6 |
| 58 | `PrinceDormand.attempts` read from config and never used | `princedormand.cpp:20` | dormant | M6 |
| 59 | adaptive-step loop never rejects/retries a step — `total_step += h` is unconditional | `princedormand.cpp:41` | dormant; documented, not fixed | M6 |
| 60 | `SlowGivens` resized its working copy of the fixed 30-row `oos_.Hx` → out-of-bounds heap write (silent under `NDEBUG`) | `helpers.cpp:13` | dormant (`use_OOS` off) | M6 |
| 61 | `InstateGroupCovs` declares `int cnt;` uninitialised → OOB heap write at an arbitrary index on the first iteration | `estimator_accessors.cpp:630` | live (output path only) | M6 |
| 62 | unconditional `std::cout` per integration step in the hot loop | `princedormand.cpp:49` | dormant | M6 |
| 63 | worker thread busy-spins at 100% CPU when the queue is empty | `estimator.cpp` `Run()` | dormant (`async_run` off) | M6 |
| 64 | filter output depends discontinuously on the last bits of its input: a 1e-11 relative perturbation of one undistortion expression moves mean ATE by 0.013 | hard gating in `OutlierRejection` | **live — deepest defect found**; measured, not fixed | M6 |
| 65 | `evaluate_rpe.py` snaps each interval endpoint to the *nearest* 120 Hz GT sample instead of interpolating, on a metric with 0.11 deg/ms sensitivity. Reports **0.2847 deg / 0.0038 m for a zero-error trajectory**; 17% of every RPE_rot number ever quoted for this project was the evaluator measuring itself | `tum_rgbd_benchmark_tools/evaluate_rpe.py` | **live (metric)** — fixed alongside, not in place | M7 |
| 66 | `P.td: 1e-5` is annotated `// 1ms`, but `P` entries are variances, so this is sigma = 3.16 ms | `cfg/*.json` | comment only | M7 |
| 67 | ATE scored with a hardcoded `--max_difference 0.001` (the tool's own default is 0.02), which associates only ~26% of frames — and not as a random subsample: they fall in contiguous blocks and **exclude the entire initialization phase**, where the largest errors live. Understates mean ATE by ~25% (0.1377 → 0.1071) and flatters poor initialization | `run_and_eval_pyxivo.py:30` | **live (metric)** — flag now honoured (#53); default kept for comparability, and `run_eval_bugfix.sh` now prints both windows by default | M8 |

## The two shipped failing tests

Both fail on a clean build of the base branch, and both are genuine bug
reports the authors already wrote:

- `Triangulation.Angular_Reprojection_Error` — **the code is wrong.** #19.
  The in-tree comment "fails in RELEASE but passes in DEBUG" is the
  fingerprint: whether `cos` lands on 1.0 or 1.0000000000000002 depends on
  FMA/x87 contraction. Fixed in M4.
- `NumericalLinearAlgebra.SlowAndFastGivensMatch` — **three independent
  bugs stacked**: #33, #34 and #54. Even with both implementations fixed,
  element-wise equality cannot hold, because any two orthonormal bases of
  the same left-nullspace differ by an arbitrary orthogonal factor. The
  test has to assert the invariants (`Hfᵗ`-projection zero, equal Gram
  matrices) instead. M6.

## Cross-cutting observations

**The shipped test suite could not have caught the worst bug.** Every one
of the 13 pre-existing Jacobian tests inspects `J_`, which was always
correct; the defect was in the *copy* of `J_` into the stacked `H`. Nothing
called `FillJacobianBlock`. Test coverage was aimed one layer below the
bug.

**Two defects had already been "documented" in comments.** `manager.cpp`
carried a comment describing the median fix that was never applied, and
`unittest_triangulation.cpp` carried a comment describing the release/debug
divergence without drawing the NaN conclusion.

**Fixing #3 silently changed the experimental setup.** `use_prediction` was
declared `true` in every config and read by nothing, so M3's plumbing fix
switched the tracker↔filter feedback loop *on* at the same commit as its code
changes. Every evaluation from M3 (`1c9e5a8`) onward therefore conflated code
with config, and that loop turns out to be the entire source of the filter's
chaotic sensitivity (#64): with `use_prediction: false` the ensemble spread
collapses from sd 0.0047 to 0.0004. See `m7-measurement-and-calibration.md` §3.
A dead config key is not only a missing feature — it is a silent change of
regime the moment someone fixes it.

**Dead config keys are a systematic problem, not a one-off.** Three keys
(`use_prediction`, `comparison_score_type`, and the misspelled
`filter_owner_change_cov_factor` lookup) were declared, plumbed into the
sweep infrastructure, and silently ignored. Anyone who swept them measured
noise. Cross-checking every key in the live config against its reader is
now part of the audit, not an afterthought.

**Several "fixes" are inert until a second fix lands.** The `fl_` correction
(#11) changes `init_std_x_/y_`, which are dead code because the badtri
branch is always taken (#52) — so #11 only becomes observable together with
#21. Similarly `feature_owner_change_cov_factor` (#22) is meaningless until
`inflate_cov` acts on the filter block (#27).

## Sub-agent claims that did NOT survive verification

- *"`num_zeros(cv::Mat)` at `tracker.cpp:743` is a type-confusion bug."*
  The implicit `cv::Mat` → `std::vector<uint8_t>` conversion is legal for
  an N×1 `CV_8U` mask. The real defect there is the empty mask (#7).
- *"`RK4Step`'s `0.5*K1` halves the position increment twice."* No term of
  the dynamics depends on `Tsb`, so the perturbed position never feeds back
  into a stage derivative. The sub-agent self-corrected.
- *"`Gt.transpose() = givens(...)` and `Hf = Gt*Hf.block(...)` alias."*
  Eigen products carry `EvalBeforeAssigningBit`; a temporary is
  materialised. Also self-corrected.
- *"`anynan()` indexes with compile-time dimensions."* Already fixed on
  `auto` before this work started (see `m0-baseline.md`).
- *"The `Rbc` non-orthogonality warning indicates a config bug."* Measured:
  max|RᵗR − I| = 9.1e-9, and SO(3) projection changes the rotation by 0°.
  Benign.
