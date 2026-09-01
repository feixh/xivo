# The 0.43 deg tilt floor belongs to TUM-VI, not to XIVO

This is the most useful thing I found after M1, and it is a negative result about
the benchmark rather than a change to the code. It says how much of the remaining
orientation ATE is even addressable, and the answer is "the yaw part, and nothing
else".

## The split

`notes-n-prompts/notes-orient/oridecomp.py` redoes the yaw+position alignment that
`ov_eval error_singlerun posyaw` does, then splits the residual rotation, in the
world frame, into the component about the vertical (**yaw**) and the component
perpendicular to it (**tilt** = roll/pitch). The two add in quadrature to the
`ov_eval` number; on both XIVO and OpenVINS the reconstructed total matches
`summary.csv`'s `ov_ate_ori_deg` to four significant figures (XIVO m1 mono
1.013 vs 1.013; OpenVINS mono 1.574 vs 1.5742; OpenVINS stereo 1.444 vs 1.4440),
which is the script's validation.

## The floor

Tilt RMS per sequence, degrees. `XIVO m1` and `XIVO M2` are 6-member ensembles,
`OpenVINS` is from `experiments/results/ov_accuracy`:

| seq | XIVO m1 mono | XIVO M2 mono | XIVO m1 stereo | OpenVINS mono | OpenVINS stereo |
|---|---|---|---|---|---|
| room1 | 0.323 | 0.323 | 0.323 | 0.348 | 0.351 |
| room2 | 0.470 | 0.469 | 0.469 | 0.480 | 0.486 |
| room3 | 0.509 | 0.509 | 0.503 | 0.541 | 0.535 |
| room4 | 0.447 | 0.448 | 0.450 | 0.452 | 0.438 |
| room5 | 0.353 | 0.352 | 0.342 | 0.400 | 0.407 |
| room6 | 0.415 | 0.416 | 0.421 | 0.411 | 0.375 |
| **mean** | **0.419** | **0.420** | **0.418** | **0.439** | **0.432** |

Two entirely independent estimators -- different filter, different feature
representation, different initialization, different code base -- agree on the tilt
error of each sequence to within 0.01-0.05 deg, and each agrees with itself between
mono and stereo. A quantity that two unrelated systems reproduce sequence by
sequence is not an estimator error. It is a property of the data.

On the XIVO side it is also immune to every single thing I changed. Tilt stayed at
0.42 under: the gauge-axis fix (M2), `gravity` = 9.80766 and 9.75, `P.Wsg` = 0.1,
`P.ba` = 0.05, `P.bg` = 0.002, the kalibr bias random walks, `Qimu.gyro` = 1.6e-4,
a 200-sample derotated gravity init, and -- the decisive one -- injecting the
*measured true* accelerometer bias as `X.ba` (0.420 -> 0.416, i.e. nothing).

## A theory of mine that this killed

I had a good quantitative story for the tilt floor and it is wrong, so it is worth
writing down. A constant *body-frame* accel bias is indistinguishable from a tilt of
`|ba_horiz|/g`, and because it is fixed in the body the world-frame tilt error it
produces rotates with the rig -- which explains why the observed tilt error is
time-varying with a near-zero mean (RMS 0.419, of which only 0.088 is a constant
offset). I measured the true accel bias against the mocap (see
`negative-results.md`) at 0.03-0.06 m/s^2, whose horizontal part predicts
0.14-0.29 deg of tilt: the right size, and roughly the right per-sequence ordering.
The filter cannot represent it (its total `ba` uncertainty over a whole run is
0.004 m/s^2), so the story hung together.

Then I put the measured bias into `X.ba` directly, which sidesteps the filter's
prior entirely, and the tilt did not move: 0.420 -> 0.416 with `gravity` = 9.75,
0.420 -> 0.418 with `gravity` = 9.8. Both made yaw and position clearly worse.
Whatever produces the floor, it is not the accelerometer bias, and it is not
anything on the estimator's side of the interface.

What is left is the evaluation side: the TUM-VI mocap attitude groundtruth, the
mocap-marker-to-IMU rotation used to bring it into the IMU frame, or the mocap/IMU
time synchronisation. A per-sequence constant error in the marker-to-IMU rotation
would show up as a constant offset, and the constant part is only 0.09 deg, so the
likeliest candidate is a small time-varying inconsistency between the mocap
attitude and the IMU frame. I did not chase it further: it is not XIVO's to fix and
it does not change any decision.

## What it means for the goal

Of XIVO's post-M2 mono orientation ATE of 0.949 deg, 0.42 is this floor and 0.83 is
yaw. Only the yaw part was ever addressable, and on that part XIVO is now well
ahead of OpenVINS:

| yaw RMS [deg] | mono | stereo |
|---|---|---|
| XIVO baseline (auto @ 9e3ec06) | 0.908 | 0.838 |
| XIVO after M1+M2 | 0.832 | see `summary.md` |
| OpenVINS | 1.507 | 1.358 |

(M1 does not change yaw -- it removes a roll/pitch frame offset -- so the baseline
yaw and the m1 yaw are the same number; M1's entire 0.81 deg gain was in the tilt
column, on top of this floor.)

An orientation ATE of about 0.42 deg is therefore the best score anybody can post on
this benchmark with this evaluation protocol, and XIVO is now at 2.3x that rather
than OpenVINS' 3.6x.
