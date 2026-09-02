# M3 -- the XIVO baseline on EuRoC MAV

Milestone M3 of `notes-n-prompts/plan-euroc.md`. Branch `auto-euroc`, worktree
`xivo-euroc`, `xivo_git = pre-merge-auto-27-g721b417`.

This is the "before" measurement: XIVO with the M2 configuration (correct
calibration, the two bias priors, the 60 m depth window) and **nothing tuned for
accuracy**. 132 runs -- 11 sequences x 2 modes x 6 jitter members. Results in
`../results/euroc_xivo_base/`.

The number that matters is not the mean. It is that **15 of 66 stereo runs
diverge**, and that the divergences are not spread evenly -- they cluster on
four sequences, and on two of those they are intermittent. Intermittent
divergence is the useful signal, because a sequence that fails 3 times out of 6
under a 1e-6 m/s velocity perturbation is not failing because it is hard. It is
sitting on a knife edge, and something is amplifying a perturbation that should
be irrelevant.


## 1. Stereo

ATE RMSE [m], `evaluate_ate.py`, 0.02 s window, mean +- sd over the surviving
members. OpenVINS' M1 column is the target.

| sequence | XIVO stereo | diverged | OpenVINS |
| --- | --- | --- | --- |
| MH_01_easy | 0.1113+-0.0058 | 1/6 | 0.073 |
| MH_02_easy | **0.0574+-0.0058** | 0/6 | 0.090 |
| MH_03_medium | **0.1046+-0.0131** | 0/6 | 0.116 |
| MH_04_difficult | -- | **6/6** | 0.207 |
| MH_05_difficult | **0.0901+-0.0249** | 0/6 | 0.190 |
| V1_01_easy | 0.0623+-0.0044 | 0/6 | 0.055 |
| V1_02_medium | 9.2447+-14.1430 | 0/6 | 0.047 |
| V1_03_difficult | 71.2295 | **5/6** | 0.059 |
| V2_01_easy | 17.4586+-14.9833 | **3/6** | 0.054 |
| V2_02_medium | 1.5210+-2.1067 | 0/6 | 0.049 |
| V2_03_difficult | 1.1150+-0.3780 | 0/6 | 0.096 |

XIVO already **wins three sequences outright** -- MH_02 (0.057 vs 0.090), MH_03
(0.105 vs 0.116) and MH_05 (0.090 vs 0.190) -- and is close on V1_01 and MH_01.
So the estimator is not the problem, and this is not a "XIVO is worse" baseline.
Every sequence XIVO wins is one where it tracks stably from the first frame; every
sequence it loses badly is one where it *sometimes* does not.

Note also `V1_02_medium`: 0 divergences but a mean of 9.24 with an sd of 14.14.
An sd 1.5x the mean is not a measurement of accuracy, it is a bimodal
distribution -- some members track and some partially lose the trajectory
without crossing the 100 m divergence threshold. The same pattern shows in
V2_02 (1.52+-2.11). Reporting the mean alone here would be actively misleading,
which is why the sd travels with every number in this project.

## 2. Mono

| sequence | XIVO mono | diverged | OpenVINS mono |
| --- | --- | --- | --- |
| MH_01_easy | -- | **6/6** | 0.132 |
| MH_02_easy | 58.9271+-41.7120 | 4/6 | 0.124 |
| MH_03_medium | 0.1500+-0.0146 | 0/6 | 0.141 |
| MH_04_difficult | 0.5959+-0.1222 | 1/6 | 0.197 |
| MH_05_difficult | 0.3092+-0.0381 | 0/6 | 0.426 |
| V1_01_easy | 0.4010+-0.3652 | 0/6 | 0.058 |
| V1_02_medium | 0.2619+-0.0420 | 0/6 | 0.067 |
| V1_03_difficult | 1.9149+-1.1327 | 0/6 | 0.068 |
| V2_01_easy | 4.2669+-4.8287 | 0/6 | 0.124 |
| V2_02_medium | 14.7421+-32.8328 | 0/6 | 0.062 |
| V2_03_difficult | 15.2309+-1.8461 | 4/6 | 0.195 |

Mono is much worse, and interestingly it fails on a *different* set of sequences
than stereo does -- mono dies on MH_01 and MH_02 where stereo is fine, and
survives V1_03 and V2_01 where stereo dies. The two modes are not "the same
filter with less information"; the stereo update changes which failure mode
dominates. Mono is not the deliverable (the brief specifies stereo + IMU for the
final evaluation), so it is measured, reported, and not tuned.

## 3. What the divergence pattern says

Four stereo sequences fail, and they split into two mechanisms.

**MH_04_difficult -- fails 6/6, i.e. deterministically.** Dumping the estimated
poses shows the reported attitude jumping about **40 degrees between poses 1 and
3** (quaternion qy -0.837 -> -0.596, qw 0.547 -> 0.802), while the same dump on
V1_01_easy and MH_05_difficult stays flat over the same interval. So this is an
*initialisation* failure, not a tracking failure -- it is broken before vision
has a chance to contribute.

The obvious hypothesis, given that MH_04 is also the sequence that broke
OpenVINS' initialiser (M1), was that XIVO's gravity initialiser has the same
blind spot: it averages `gravity_init_counter` accelerometer samples with **no
stationarity gate whatsoever**. **Measurement says no.** Over MH_04's actual
init window (20 samples, 0.1 s) the mean specific force is 9.760 m/s^2 against
a gravity of 9.810 -- an error of 0.050 -- with sd 0.135 and max |w| of
0.11 rad/s. MH_04 is genuinely still when XIVO initialises; the take-off that
defeated OpenVINS begins 1.5 s later, and XIVO's window has closed by then.
So gravity init is *not* MH_04's problem, and the 40-degree jump has to come
from the covariance rather than from the initial value.

Checking the same window on all eleven sequences did turn up a real defect,
just on a different sequence -- **MH_01_easy**, whose init window is off by
**-1.463 m/s^2**, 30x the next worst. That is pursued in M4.

**V1_03 (5/6), V2_01 (3/6), MH_01 (1/6) -- intermittent.** A 1e-6 m/s velocity
perturbation should be unobservable. That it decides the outcome means the early
filter state has a divergent direction, and the jitter merely picks which side of
it a run lands on. Combined with the MH_04 attitude jump, both point at the same
suspect: the 2-DOF gravity state `Wsg`, whose prior variance `P.Wsg = 3.01`
corresponds to sigma = **1.73 rad**. With a prior that loose, early vision
residuals get absorbed by rotating the estimate of *which way is down* rather
than by correcting the pose -- and a wrong gravity direction then feeds straight
back into the acceleration estimate. That is a positive feedback loop, and it is
exactly the shape needed to turn 1e-6 m/s into 71 m.

Confirming and fixing that is M4. The relevant point for M3 is that the baseline
diagnosis is specific enough to test, rather than "needs tuning".

## 4. Throughput and memory

Measured during the accuracy pass, so these are **not the quotable efficiency
numbers** -- that pass ran many sequences concurrently. Recorded here only to
show nothing pathological:

| mode | FPS (wall) | peak RSS (MB) |
| --- | --- | --- |
| stereo | 21.3 | 127.8 |
| mono | 28.4 | 118.7 |

The real comparison -- one core, `taskset -c 0`, all thread pools pinned to 1,
serial, idle box, n=3 -- is M5/M6.

## 5. Reproduce

```bash
cd experiments/openvins
CPU_BASE=0 CPU_SPAN=60 ./run_xivo_reference.sh --profile euroc_mav \
  --mode both --jitter 6 --worktree xivo-euroc --cfg-prefix euroc \
  --out ../results/euroc_xivo_base
python3 agg_ensemble.py --mode stereo --arm xivo ../results/euroc_xivo_base
```

`--mode stereo` is not optional in the aggregation: the XIVO runner writes mono
and stereo rows into a single `summary.csv`, and averaging across sensor modes
produces a number that means nothing. The aggregator requires the flag when it
sees both.
