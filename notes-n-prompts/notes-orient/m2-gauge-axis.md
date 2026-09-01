# M2 -- fix the gauge about gravity, not about the group's body z-axis

`experiments/results/orient_s_gauge` (mono screen) and `experiments/results/orient_final`
(both modes) vs `experiments/results/orient_m1`.

## What was wrong

`Estimator::SwitchRefGroup` pins the newly-elected gauge group by zeroing rows and
columns of `P_`. Under the default `group_degrees_fixed: 4` it did:

```cpp
int offset = kGroupBegin + 6 * g->sind();
P_.block(offset+2, 0, 4, err_.size()).setZero();
P_.block(0, offset+2, err_.size(), 4).setZero();
```

which is `dW(2)` plus all three translation coordinates: "the global position, and
the yaw". The three translation rows are right. `dW(2)` is not.

A group's 6-vector error is a *right* perturbation -- `SO3xR3::operator+=` in
`src/group.h` is `Rsb *= SO3::exp(dX.head<3>())` -- so `dW` lives in the group's own
body frame and `dW(2)` is a rotation about the **body z-axis**. The unobservable
direction is a rotation about **gravity**. Those coincide only if the rig is level.

Measured over the six groundtruth trajectories, angle between the body z-axis and
the world vertical:

| seq | median | mean | p90 | max |
|---|---|---|---|---|
| room1 | 11.9 | 13.6 | 25.4 | 49.3 |
| room2 | 6.8 | 9.2 | 20.7 | 55.1 |
| room3 | 6.8 | 9.7 | 22.1 | 63.3 |
| room4 | 17.3 | 19.9 | 40.9 | 69.1 |
| room5 | 10.8 | 16.5 | 40.9 | 74.5 |
| room6 | 8.4 | 11.3 | 25.2 | 38.5 |

(degrees; the rig is hand-held and pitched down at the room, so it is never level.)

Zeroing the wrong axis does two separate bad things:

1. It leaves a `sin(angle)` fraction of the true yaw gauge unfixed, so the thing the
   gauge was supposed to pin keeps moving.
2. Worse, it declares a component of the group's *observable* tilt to be known
   exactly. An identically-zero row of `P_` is permanent: the Kalman gain row is
   `P_.row(i) H' S^-1`, so a zero row gives a zero gain row, which leaves the row
   zero after the update. Whatever tilt error that group had at election time was
   frozen into it forever -- and into the anchor pose of every feature it owns.

## The fix

Project out the correct direction instead of a coordinate. A global rotation by
`dtheta` about the vertical `n_s` takes `Rsb -> exp(dtheta n_s^) Rsb =
Rsb exp(dtheta (Rsb' n_s)^)`, so in the group's body frame the unobservable
direction is `u = Rsb' n_s`:

```cpp
const Vec3 n_s = X_.Rsg * Vec3{0, 0, 1};
const Vec3 u = (g->Rsb().inverse() * n_s).normalized();
const Mat3 Pi = Mat3::Identity() - u * u.transpose();
P_.block(offset, 0, 3, N) = (Pi * P_.block(offset, 0, 3, N)).eval();
P_.block(0, offset, N, 3) = (P_.block(0, offset, N, 3) * Pi).eval();
P_.block(offset + 3, 0, 3, N).setZero();
P_.block(0, offset + 3, N, 3).setZero();
```

This is the same congruence `P <- M P M'` the old code performed -- for `u = e3`,
`I - u u'` *is* `diag(1,1,0)`, i.e. zeroing row and column 2 -- so it is a strict
generalization, PSD-preserving, and it fixes exactly one rotational degree of
freedom, not more and not fewer. The constraint `u' dW = 0` is self-maintaining for
the same reason the old one was: `u' P_{rows,:} = 0` after the projection, so
`u' K_{rows} = 0` and both the state correction and the covariance rows stay in the
constrained subspace.

`n_s` comes from the filter's own `Rsg`, which by M1's measurement is within
0.05-0.40 deg of true gravity, so `u` is right to well under a degree instead of
being 7-17 deg off.

Note this is *not* the gravity-alignment of M1 reappearing: M1 changed only what is
published, M2 changes the filter. They are independent, and M2 would have been
worth doing without M1.

## Numbers

Mono screen (room1-6, jitter 6), M1 -> M2:

| metric | m1 | s_gauge (M2) |
|---|---|---|
| ov_ate_ori_deg | 1.013 | **0.949** |
| ov_rpe8_ori_deg | 0.5185 | **0.5149** |
| ate_002 [m] | 0.0928 | 0.0957 |
| ov_ate_pos_m | 0.0936 | 0.0963 |
| ov_rpe8_pos_m | 0.0480 | 0.0489 |

Per-sequence orientation ATE, mono: 1.060 / 0.675 / 1.402 / 0.895 / 1.042 / 0.618
against M1's 1.225 / 0.920 / 1.423 / 0.841 / 0.955 / 0.713 -- better on four,
slightly worse on room4 and room5.

Both-modes confirmation, `experiments/results/orient_final` vs `orient_m1`:

| metric | mono m1 | mono final | stereo m1 | stereo final |
|---|---|---|---|---|
| ov_ate_ori_deg | 1.013 | **0.949** | 0.959 | **0.983** |
| ov_rpe8_ori_deg | 0.5185 | 0.5149 | 0.5088 | 0.5101 |
| ate_002 [m] | 0.0928 | 0.0957 | 0.0636 | 0.0636 |
| ov_ate_pos_m | 0.0936 | 0.0963 | 0.0640 | 0.0640 |

**Verdict: a correctness fix, and a statistical wash on the benchmark.** Mono
-0.064 deg is 1.6 sigma; stereo +0.024 deg is 0.6 sigma the other way; averaged over
the two modes, -0.020 +- 0.028. Every non-regression constraint holds in both modes
(`rpe_ori` 0.5149 / 0.5101 against a 0.53 limit, `ate_002` +0.0029 / +0.0000 against
a 0.005 budget, no divergences in 144 runs), and stereo is *untouched* on all three
position metrics to four decimals.

I kept it because the axis the old code fixed is demonstrably the wrong one and the
new code is a strict generalization of it, not because the ensemble can resolve the
difference. Note that the harness is deterministic -- re-running the same binary and
config reproduces `summary.csv` byte for byte -- so the 6-member jitter spread is the
only uncertainty estimate available, and it is not going to resolve a 0.02 deg
effect. **The commit is self-contained: if the merge is tight on mono position ATE,
revert it.**

## Where the gain came from (and it is not where I expected)

`notes-n-prompts/notes-orient/oridecomp.py` splits the posyaw-aligned orientation
error into the part about the vertical (yaw) and the part perpendicular to it
(tilt); the two sum in quadrature to the `ov_eval` number to three decimals, which
validates the script.

| | tilt | yaw | total |
|---|---|---|---|
| m1 mono | 0.419 | 0.908 | 1.013 |
| final mono | 0.420 | 0.832 | 0.949 |
| m1 stereo | 0.418 | 0.838 | 0.959 |
| final stereo | 0.419 | 0.867 | 0.983 |

**All of the movement is in yaw; the tilt part does not budge** -- per sequence it is
identical to three decimals (0.323, 0.469, 0.509, 0.448, 0.352, 0.416). Which is the
right signature: mono's yaw improves by 0.076, stereo's degrades by 0.029, and tilt
is untouched, exactly as a change to the *yaw* gauge should behave. The mechanism is
operating; the benchmark simply cannot resolve its sign in stereo.

So of the two mechanisms above, only (1) was actually operating. My prediction that
the frozen observable tilt would be costing roll/pitch accuracy was **wrong**: the
gauge group is elected as the lowest-covariance in-state group and re-elected often,
so its tilt error at election time is small and short-lived. What did matter is that
the yaw gauge was only ever `cos(angle)` fixed, and the leaked remainder shows up as
yaw drift -- which is the dominant term in this error budget (0.83-0.91 of 0.95-1.01).

The tilt floor of 0.42 deg has a different cause entirely, and it is not the
estimator's: OpenVINS reproduces it sequence by sequence to within 0.05 deg. See
`tilt-floor-is-the-benchmark.md`. Nothing here can move it.
