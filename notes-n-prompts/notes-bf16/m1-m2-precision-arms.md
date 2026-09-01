# M1 / M2 -- what happens when `number_t` itself is narrowed

The requirement asks for bf16 "as the main numerical type (`number_t`)". This
note is the measurement of that literal reading, and of the fp32 rung below it.
Both arms are real builds of this branch, selected at configure time:

    cmake .. -DXIVO_NUMBER_T=double|float|bf16 -DXIVO_OUTPUT_SUFFIX=...

`XIVO_NUMBER_T` drives `XIVO_NUMBER_T_{DOUBLE,FLOAT,BF16}`, which `common/alias.h`
turns into `number_t` -- and with it every Eigen alias, both Sophus groups, the
camera models, the covariance and the state.

## Result, up front

| arm | mono wall | stereo wall | accuracy |
|---|---|---|---|
| `double` (baseline) | 130.2 s | 215.2 s | ATE@0.001 0.0686 / 0.0453 |
| `float` | 88.3 s (**1.47x**) | 152.8 s (**1.41x**) | **1 divergence in 36 runs per mode**; RPE_tra +86% / +41% |
| `bf16` | -- | -- | **diverges within 5 frames** |

Neither narrowing of `number_t` is deliverable. The efficiency work therefore
has to keep `number_t = double` -- fp64 storage and fp64 for anything integrated
over time -- and put the reduced precision *inside the covariance kernels*,
where the operand is a weighting matrix rather than a state. That is M3.

## The bf16 arm: 5 frames

`-DXIVO_NUMBER_T=bf16` builds and links (see "what it took" below) and then, on
TUM-VI room1 mono, produces exactly five poses:

    1520530308.302287    0.000022  -0.000049  -0.000019   ...
    1520530308.352447    0.000420  -0.000029  -0.000089   ...
    1520530308.402607    0.000671  -0.000172  -0.000368   ...
    1520530308.452765   -0.116211   0.033447  -0.006653   ...
    1520530308.502926 -756.000000 -166.000000  -8.062500  ...

Frame 4 is already 0.12 m off with the platform stationary; frame 5 is 756 m
away, and from frame 6 the state is NaN. With Sophus's assertions left on the
run aborts before writing anything at all, in
`SO3Base::normalize` -> `SOPHUS_ENSURE(length >= Constants<Scalar>::epsilon())`
reached from `SO3_from_rotvec` inside `Estimator::AbsorbError` -- i.e. the error
state handed back by the update is no longer a rotation.

This is the expected outcome and the reason the design is mixed. bf16 has a
7-bit stored significand: eps = 2^-8 = 3.9e-3, so *every element* of every
quantity carries 0.4% relative error. A covariance tolerates that -- it only has
to weight a correction. A state does not: `Estimator::AbsorbError` composes a
correction into `Rsb`, and position is integrated at 200 Hz. The gravity
alignment alone (`InitializeGravity`, three iterations of
`makeRotationMatrix`) starts from a rotation whose orthogonality residual is
already 2e-2.

### What it took to build at all

Worth recording, because "the type does not exist in this toolchain" is part of
the answer:

* Eigen 3.3 is vendored here and predates `Eigen::bfloat16`, so the scalar type
  is written out in `common/bf16.h`: RNE conversion, the operator set, ADL math,
  `std::numeric_limits`, `Eigen::NumTraits`, `Eigen::internal::cast_impl`.
* The mixed operators have to be spelled out for every integral type as well.
  With an implicit conversion in both directions, `x == 0` is ambiguous between
  `bf16 == bf16` and `float == float`; `common/camera_atan.h` alone compares
  against integer literals 75 times.
* The type has to be a *literal* type -- `src/feature.h`'s `kMaxLogDepth` and
  `src/imu.cpp`'s `kMaxTh`/`kMaxRw` are `constexpr number_t` -- so the bit
  punning goes through `__builtin_bit_cast`, not `memcpy`.
* `std::chrono::treat_as_floating_point` and `std::common_type` need
  specializations, because `Timer` instantiates `duration<number_t>`.
* **`std::is_floating_point<bf16>` has to be specialized**, which the standard
  does not permit for a program-defined type. Sophus gates
  `makeRotationMatrix`/`fitToSO3` on it, and those are what
  `Estimator::InitializeGravity` and `StereoRig` call. Eigen's own `half` and
  `bfloat16` hit the same wall.
* `Sophus::Constants` hard-codes its tolerance -- 1e-10 in the primary template,
  1e-5 in the one specialization (`float`) -- instead of deriving it from
  `numeric_limits`. `epsilonSqrt()` is the bound `SO3(Matrix3)` checks
  `|R R^T - I|_inf` against, so with the primary template a bf16 rotation matrix
  aborts on construction before the first frame is read.
  `common/bf16.h` specializes it at 1e-2 (~2.6x bf16's eps, so `epsilonSqrt()`
  is 0.1 -- above the ~2e-2 residual of a bf16-rounded rotation, and also the
  small-angle Taylor threshold in `SO3::exp`/`log`, at 0.1 rad, which is the
  right call at an 8-bit significand).
* Pangolin has no vertex format for a non-builtin scalar, so `src/viewer.*`
  keeps its trace and its draw calls in fp32. The display path is not part of
  the numerics.
* The five `pybind11/pyxivo.cpp` accessors (`gsb`, `gsc`, `gbc`, `Pstate`, `P`)
  now `.cast<double>()` explicitly: the Python side is numpy float64 whatever
  `number_t` is.
* `common/utils.h`'s `RandomMatrix`/`RandomVector` draw from
  `std::normal_distribution<double>` and round on assignment;
  `std::normal_distribution` is only defined for the builtin floating types.
  (These two functions also contained `using number_t = number_t;`, which had
  been a no-op self-alias.)

The measurement runs used `-DSOPHUS_DISABLE_ENSURES` so the divergence could be
*observed* rather than aborted on; the tree does not set it.

### And it is 52x *slower*

The literal arm cannot be timed end to end -- it has no trajectory to time. But
the kernel it would run can be, and that is the number that matters
(`bench/eigen_scalar.cpp`: `A * P * A^T` at n = 563 (the filter's n is 564), one pinned core, Eigen
matrices of each scalar type, i.e. exactly what `-DXIVO_NUMBER_T=` hands the
filter):

| scalar type | ms | vs fp64 |
|---|---|---|
| `double` | 14.45 | 1.00x |
| `float` | 6.94 | 2.08x |
| `bf16` | **752.27** | **0.02x** |

Eigen 3.3 has no bf16 path on x86. `Matrix<bf16,...>` falls off the vectorized
path entirely: every coefficient is a widen, an fp32 multiply-add and a round, so
the product runs **52x slower than fp64** -- and note that fp32 lands at 2.08x,
essentially the 2.00x arithmetic ceiling, so it is the bf16 type specifically
that is pathological, not the benchmark.

So the literal reading of the requirement loses on *both* criteria, by large
margins in both directions: 52x slower and divergent in 5 frames. The 2.67x in
`m0-baseline.md` is only reachable from a kernel that rounds the *inputs* to
bf16 and accumulates in fp32 with `vdpbf16ps` -- a property of a kernel, not
something a scalar type can express, because the accumulator has to be wider
than the operands. That is the whole argument for M3.

## The fp32 arm: 1.45x, and one divergence in 36

`-DXIVO_NUMBER_T=float`, both modes, 6 members x 6 rooms
(`merge/logs/bf16_f32_{mono,stereo}.log`). All 18 `ctest` binaries build; 11 of
them fail, all on hard-coded fp64 tolerances (`EXPECT_EQ` on floats, absolute
`1e-9` bounds, finite-difference steps of 1e-8) -- the suite's tolerance model is
tied to fp64, not a statement about the filter. This was not repaired, because
the arm does not survive the accuracy gate anyway:

| mode | member | room | ATE@0.001 |
|---|---|---|---|
| mono | m3 | room3 | **94938 m** |
| stereo | m3 | room2 | **45920 m** |

One run in 36 diverges completely, in each mode. Both are member 3, i.e. a
1e-6 m/s change in the initial velocity relative to member 0 -- the divergence
is not a property of a sequence, it is a coin flip the filter loses at fp32.

Dropping those two runs and comparing the remaining 35 pairs run-for-run:

| metric | mono base | mono fp32 | delta | stereo base | stereo fp32 | delta |
|---|---|---|---|---|---|---|
| ATE@0.001 | 0.0682 | 0.0764 | +12% | 0.0455 | 0.0474 | +4% |
| ATE@0.02 | 0.0843 | 0.1023 | +21% | 0.0590 | 0.0605 | +2% |
| RPE_tra | 0.0212 | **0.0394** | **+86%** | 0.0132 | **0.0185** | **+41%** |
| RPE_rot | 0.6172 | 0.6178 | +0% | 0.6185 | 0.6256 | +1% |

RPE_tra is the diagnostic one: it measures the drift over a 1 s window, so it
sees local integration error that ATE's global alignment partly hides. +86% is
far outside the +-0.0010 member spread. mono ATE@0.001 +12% is 2.4 member sds.

So even setting the divergences aside, fp32 as `number_t` fails
"do NOT degrade the accuracy metrics". It is kept as a build option -- it is the
right instrument for asking *which* quantity needs the width -- but it is not
the delivered configuration.

## What this fixes about the plan

The plan expected fp32 to be roughly accuracy-neutral and to carry most of the
speedup, with bf16 kernels on top. Half of that is right: fp32 does carry
1.45x, which is the bulk of what the arithmetic ceiling allows. But it buys that
by narrowing the *state*, and the state is what diverges. The remaining work is
therefore narrower and better targeted than the plan assumed:

* `number_t` stays `double`. Storage, timestamps, IMU pre-integration, SO3
  renormalization, the state and its correction are all untouched.
* The reduced precision goes only into the arithmetic of products *involving P*,
  which is 32.1 of mono's 48.0 ms and 52.8 of stereo's 80.1 ms.
* Those kernels round their inputs to bf16 and accumulate in fp32, so the error
  they introduce is a perturbation of a weighting matrix on a single frame, and
  is not integrated.
