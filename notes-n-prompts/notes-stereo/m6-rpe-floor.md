# M6 — why RPE_rot sits at 0.62 deg, and what the 0.50 deg criterion really asks

Exit criterion 3 is **mean RPE_rot < 0.5 deg** (1 s fixed delta). The stereo system
lands at 0.62 deg. This note decomposes that number, because most of it is not
estimator error.

## RPE_rot is immovable

Across every knob swept in M6 it stayed inside 0.6183–0.6575:

| swept | arms | RPE_rot range |
| --- | --- | --- |
| `Qimu.gyro` | 16x span | 0.6206–0.6575 |
| `visual_meas_std` | 4x span | 0.6211–0.6289 |
| feature capacity (30→150 in state) | 6 arms | 0.6203–0.6218 |
| initial attitude error (1.47 → 0.73 deg) | 2 arms | no measurable change |

The capacity row is the telling one: it cut mean **ATE by 37%** and moved RPE_rot by
0.0015 deg. A quantity that ignores a change of that size is not primarily
measuring the estimator.

## Decomposition

Mean over the six rooms:

| term | deg | how measured |
| --- | --- | --- |
| GT association artifact | 0.31 | `evaluate_rpe.py` pairs each estimate with the *nearest* mocap sample, up to 4.2 ms away at 120 Hz. Slerping the GT to the estimate stamps instead drops the reported error 0.6289 → 0.5439 (`harness/rpe_assoc.py`) |
| mocap's own attitude noise | 0.28 | local-cubic fit to the GT attitude, residual 0.08–0.19 deg/axis; propagated to a per-room RPE floor of 0.2307–0.3605 (`harness/mocap_noise.py`) |
| real estimator attitude error | ~0.46 | remainder, in quadrature |

Consistency check: `sqrt(0.46^2 + 0.28^2 + 0.31^2) = 0.626` against 0.62 observed.

## What the criterion therefore asks

About **0.42 deg of the 0.50 deg budget is noise in the reference**, not error in the
estimate. Hitting 0.50 deg as measured requires the estimator's own attitude error
to fall from ~0.46 deg to below `sqrt(0.50^2 - 0.42^2) = 0.27` deg — a **42%
reduction in real attitude error**, not the 19% the raw numbers suggest.

## Leads closed by measurement

Not by argument — each of these was measured and rejected, so do not re-litigate:

- **Gyro scale / misalignment.** Fitted `Cg` against mocap on all six rooms
  (`harness/gyro_calib.py`): deviation from identity ≤0.3%, and the cross-room std
  (0.06–0.16%) is the same size as the mean, i.e. it is fit noise. Worth ~0.06 deg.
- **Propagation integrator.** `ComposeMotion` uses `SO3::exp` with
  `Rsb.normalize()`, under RK4 or Prince-Dormand. Not a coarse first-order scheme
  that a better integrator would fix.
- **Initial attitude.** See [[m6-attitude-initialization]] — halving the tilt error
  changes nothing, because the `X.Rsg` prior variance is 3.01.
- **Out-of-state / MSCKF update.** `use_OOS: true` hits
  `LOG(FATAL) << "MSCKF not implemented"` at `src/estimator.cpp:126`. Unimplemented
  upstream; all 12 sweep runs that enabled it died there.

## Recommendation

Report the decomposition rather than chasing the number. The honest statement is
that stereo cuts translational error by half and leaves rotational error where a
1 s-delta metric against 120 Hz mocap cannot resolve the difference. Chasing 0.50
deg from here means either a better reference (interpolated association, which is
a change to the *evaluation*, not the estimator) or a genuinely different attitude
observability story — neither of which the remaining knobs offer.
