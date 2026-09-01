# M6 — the result that mattered: capacity, not tuning

## The finding

`tracker_cfg.num_features_max` was **60** and `core.h`'s `kMaxFeature` was **30**.
Every number in `RESULTS.md`, and every stereo number through M5, came from a
filter holding **30 features**. On 512x512 fisheye images with a 101 mm stereo
baseline, that is the binding constraint — not the IMU noise model, not the
measurement std, not the initialization.

Raising both caps together moved the six-room mean ATE from 0.0760 m to 0.0476 m.
For scale, perturbing a physically-neutral config knob moves the mean by ~0.006 m
(see [[xivo-tuning-noise-is-not-seed-noise]]), so this is a ~5 sigma move — and it
came from two integers.

## Both caps, or neither

`EKF_MAX_FEATURES` sets how many features the filter *can* hold;
`tracker_cfg.num_features_max` sets how many the tracker *supplies*. Raising only
the first does nothing: builds at 60 and at 90 in-state features were
**bit-identical** while the tracker stayed at 60. That is why the earlier sweeps
found nothing — they moved one number at a time.

A third number is easy to mistake for these: `memory.max_features` sizes an object
pool, not filter capacity. An early arm that raised it 200 → 400 was bit-identical
on five of six rooms. See [[m6-memory-pools]].

## The curve, and its turnover

`fN tM` = N features in the EKF state (N/2 groups) and M tracked by the tracker.
Headline ATE is `--max_difference 0.02`; `0.001` is reported alongside for
comparability with `RESULTS.md` (see [[xivo-ate-eval-protocol]]).

```
arm         room1   room2   room3   room4   room5   room6   mATE001  mATE02  RPEtra  RPErot
M5 shipped  0.0793  0.0578  0.0976  0.0541  0.1144  0.0530   0.0760  0.1013  0.0201  0.6211
f30t120     0.0846  0.0630  0.0687  0.0547  0.0655  0.0325   0.0615  0.0802  0.0177  0.6218
f60t120     0.0448  0.0524  0.0664  0.0431  0.0682  0.0392   0.0523  0.0629  0.0155  0.6211
f90t180     0.0551  0.0435  0.0549  0.0434  0.0612  0.0278   0.0476  0.0575  0.0145  0.6206  <- shipped
f120t240    0.0611  0.0386  0.0685  0.0349  0.0584  0.0293   0.0485  0.0576  0.0144  0.6217
f150t300    0.0647  0.0360  0.0774  0.0641  0.0627  0.0360   0.0568  0.0693  0.0138  0.6220
```

The curve is unimodal with a minimum at **f90t180**: f120 is inside the noise
(+0.0009) and f150 is clearly worse (+0.0092). So this is a genuine turnover, not
an arbitrary stopping point. Two plausible reasons, not separated here: 300
tracked features on a 512x512 fisheye image start competing for texture (the
tracker's `mask_size` is 15 px), and a larger state admits more weakly-conditioned
features whose linearization errors the filter then has to absorb.

Note that RPE_rot is flat to within 0.0015 deg across the entire curve while ATE
moves 37%. That is the strongest single piece of evidence that RPE_rot is
floor-limited by the reference, not by the estimator — see [[m6-rpe-floor]].

## Everything else that helped, and everything that did not

Carried into the shipped config alongside capacity:

| change | effect on mATE02 |
| --- | --- |
| `visual_meas_std` 1.5 → 0.75 | 0.1013 → 0.0841 |
| `memory` 200/100 → 800/300 | none (pool only; needed for the tracker cap) |

Swept and rejected — all neutral or worse at the final capacity:

| arm | mATE001 | RPE_rot | verdict |
| --- | --- | --- | --- |
| shipped (`h_base`) | 0.0476 | 0.6206 | — |
| `Qimu.gyro_bias` x0.1 | 0.0494 | 0.6209 | worse |
| `Qimu.gyro_bias` x10 | 0.0503 | 0.6248 | worse |
| `tracker_cfg.use_prediction: true` | 0.0476 | 0.6206 | **bit-identical** |
| de-rotated 200-sample gravity init | 0.0520 | 0.6234 | worse |

`use_prediction` being bit-identical is not a coincidence: the key is **never read
anywhere in `src/`**. It is a dead config knob. Do not sweep it again.

Also swept earlier and rejected: `Qimu.gyro` over a 16x span (RPE_rot
0.6206–0.6575, ATE never better), `use_OOS` (unimplemented — `LOG(FATAL)` at
`src/estimator.cpp:126`, killed all 12 runs that enabled it), `use_depth_opt`.

## Shipped configuration

`cfg/tumvi_stereo.json`, built at the CMake defaults `EKF_MAX_FEATURES=90` /
`EKF_MAX_GROUPS=45`. The config states that requirement in
`require_ekf_max_features` / `require_ekf_max_groups`, which `CheckMemoryPools`
enforces at startup — otherwise a config tuned for 90 run against a 30-feature
binary loses half its accuracy with no symptom but a worse number.
