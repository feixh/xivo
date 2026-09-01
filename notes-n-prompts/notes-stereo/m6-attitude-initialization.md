# M6 — attitude initialization, and why it did not matter

Referenced from `src/estimator.h` (`gravity_init_derotate_`). This is a **negative
result**: the code is in the tree, off by default, because a negative result is
only credible with the measurement it came from.

## The claim being tested

`Estimator::InitializeGravity` averages the first `gravity_init_counter` (shipped:
20) accelerometer samples, calls the mean gravity, and sets the initial attitude
from it. Its log line called them "stationary accel samples".

They are not stationary. Measured over the 20 samples each room sequence actually
uses (`m6-artifacts/harness/grav_init.py`):

| room | mean \|w\| over the init window (rad/s) | initial tilt error vs mocap (deg) |
| --- | --- | --- |
| room1 | 0.16 | 1.19 |
| room2 | 0.11 | 0.94 |
| room3 | 0.32 | 2.43 |
| room4 | 0.21 | 1.62 |
| room5 | 0.19 | 1.35 |
| room6 | 0.14 | 1.31 |
| **mean** | **0.19** | **1.47** |

So the rig is turning at 0.1–0.3 rad/s when the filter decides which way is down.

## Why a longer window is not the fix

The averaging window is caught between two errors that pull in opposite
directions:

- **too short** — the carrier's own linear acceleration has not averaged out, so
  the mean is gravity plus whatever sway happened during those 100 ms;
- **too long** — the samples are body-frame vectors taken across a turn, so
  averaging them smears the direction by roughly `|w| * window`.

At 0.19 rad/s, a 1 s window smears by ~11 deg. That is why upstream's window is
20 samples: lengthening it makes things worse.

De-rotating each sample into the body frame of the *last* sample — integrating
the gyro to get the relative attitude — removes the second error, which is what
makes a long window usable. Over 200 samples that halves the tilt error:

| | window | mean tilt error (deg) |
| --- | --- | --- |
| as shipped | 20, plain average | 1.47 |
| de-rotated | 200 | **0.73** |

Both halves of that claim are pinned by `src/test/unittest_gravity_init.cpp`,
which drives the real `InertialMeasInternal` entry point: the de-rotated average
is exact under pure rotation (<1e-6 deg, where the plain average is off by >3
deg and the error *grows* with the window), the two paths agree to 1e-12 on a
genuinely static start, and a 0.6 m/s^2 sway that a 20-sample window cannot
average away is gone over 200.

## Why it changes nothing end to end

Halving the initial tilt error moved neither ATE nor RPE by a measurable amount
on any of the six rooms. The reason is in the config: `initial_std_Wsg` gives
`X.Rsg` a prior variance of 3.01, i.e. the filter is told from the start that it
does not know which way is down to better than ~100 deg. A 1.5 deg error inside a
100 deg prior is absorbed within the first few visual updates.

**Conclusion.** Attitude initialization is not the reason RPE_rot sits at 0.62
deg. Kept as an opt-in flag (`gravity_init_derotate`, default `false`) so the
monocular baseline configs stay bit-for-bit as they were; anyone who tightens
`initial_std_Wsg` will want it on. See [[m6-rpe-floor]] for where the 0.62 deg
actually comes from.
