# Config keys changed by the orientation work

**None.** `cfg/eff_mono.json` and `cfg/eff_stereo.json` are byte-identical to
`auto` @ 9e3ec06 on branch `auto-orient`. Nothing here needs to be hand-merged
against the position or speed agents' config edits.

That is deliberate. Every code change on this branch is gated on a config key
that **defaults to the new behaviour**, so the shipped configs need no edit and
the old behaviour is still reachable:

| key | default | off restores |
|---|---|---|
| `gravity_align_output` | `true` (M1) | the pre-M1 output convention, bit-for-bit |

`group_degrees_fixed` (M2) is *not* a new key and its value is unchanged (4); M2
changes what "4 degrees fixed" means, not how many.

## Config keys screened and rejected

Each was screened with the standard harness (mono, room1-6, `--jitter 6`) against
the M1 reference of ori 1.013 / rpe_ori 0.5185 / ate002 0.0928. All were reverted;
see `negative-results.md` for the reasoning.

| key(s) | value tried | why rejected |
|---|---|---|
| `Qimu.gyro_bias`, `Qimu.accel_bias` | kalibr values (3.33x larger) | ori 1.095 (worse); rpe_ori 0.5360 breaks the 0.53 limit |
| `gravity` | `[0, 0, -9.80766]` | ori -0.055 (1.3 sigma, not significant); ate002 0.1047 breaks the budget |
| `gravity` | `[0, 0, -9.75]` (the value the accelerometer is actually consistent with) | ori 1.088 (worse); ate002 0.1048 breaks the budget |
| `P.Wsg` | `0.1` (from 3.01) | ori -0.058 (1.3 sigma, not significant); ate002 +0.004 adverse |
| `P.ba` | `0.05` (from 0.001) | ori 1.041; rpe_ori 0.5342 breaks the limit; ate002 0.1002 breaks the budget |
| `P.bg` | `0.002` (from 0.0001) | ori 1.010 (no better than its base 0.949); ate002 0.0965 |
| `P.ba` + `P.bg` | both of the above | ori 1.091; rpe_ori 0.5412; ate002 0.1037 |
| `Qimu.gyro` | `1.6e-4` (kalibr) | ori 1.058 (worse); ate002 0.1014 breaks the budget |
| `X.ba` (+/- `gravity`) | the measured bias `[-0.025, 0.025, 0.031]` | ori 1.065-1.079 (worse); ate002 0.0997-0.1046. Best `rpe_ori` of anything screened (0.5047) but it does not touch the tilt error at all, which falsified my theory of it. |
| `gravity_init_counter` (+ derotation) | `200` | ori -0.004, a wash; ate002 +0.006 adverse |

The budget is baseline + 0.005 = 0.0978 m on `ate_002` and 0.53 deg on `rpe_ori`.
Bases are m1 (1.013) for the first four rows and M2 (0.949) for the rest.
