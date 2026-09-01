"""How much of RPE_rot is ground-truth association error rather than estimator error?

`evaluate_rpe.py` pairs each estimate stamp with the NEAREST ground-truth stamp
within --max_difference (default 0.02 s). Estimate stamps are image stamps at
20 Hz, ground truth is at 120 Hz, so the association is off by up to ~4 ms. While
the rig turns at |w|, that mis-association alone contributes |w| * dt of apparent
rotation error -- and it is indistinguishable from real attitude error.

Arm A reproduces the tool. Arm B replaces nearest-neighbour association with a
slerp of the ground truth to the exact estimate stamp, which removes the
artifact and leaves only the estimator's own error. Everything else -- the
fixed_delta pair construction, the dropout filter, the RMSE definition -- is
identical between the arms, so the difference is attributable.
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

DELTA = 1.0

def load(p):
    d = np.loadtxt(p)
    return d[:, 0], d[:, 1:4], Rotation.from_quat(d[:, 4:8])

def rmse_deg(Re, Rg_a, Rg_b, ie, je):
    """RPE rotational RMSE for pairs (ie, je), given per-index GT rotations."""
    rel_e = Re[ie].inv() * Re[je]
    rel_g = Rg_a.inv() * Rg_b
    ang = (rel_g.inv() * rel_e).magnitude()
    return np.rad2deg(np.sqrt(np.mean(ang ** 2))), len(ang)

for seq in ('room1', 'room2', 'room3', 'room4', 'room5', 'room6'):
    d = '%s/run_g_ctl_%s' % (sys.argv[1], seq)
    te, _, Re = load('%s/tumvi_%s_cam0' % (d, seq))
    tg, _, Rg = load('%s/tumvi_%s_gt' % (d, seq))
    gt_step = np.median(np.diff(tg))

    # fixed_delta pairs: for each estimate stamp, the estimate stamp closest to
    # t + DELTA, as evaluate_rpe.py does.
    j = np.searchsorted(te, te + DELTA)
    ok = j < len(te)
    ie, je = np.flatnonzero(ok), j[ok]
    ie, je = ie[np.abs(te[je] - te[ie] - DELTA) < 0.01], je[np.abs(te[je] - te[ie] - DELTA) < 0.01]

    # --- arm A: nearest GT, dropped if worse than two GT intervals (the tool's rule)
    ka = np.clip(np.searchsorted(tg, te[ie]), 1, len(tg) - 1)
    ka = np.where(np.abs(tg[ka] - te[ie]) < np.abs(tg[ka - 1] - te[ie]), ka, ka - 1)
    kb = np.clip(np.searchsorted(tg, te[je]), 1, len(tg) - 1)
    kb = np.where(np.abs(tg[kb] - te[je]) < np.abs(tg[kb - 1] - te[je]), kb, kb - 1)
    good = ((np.abs(tg[ka] - te[ie]) < 2 * gt_step) &
            (np.abs(tg[kb] - te[je]) < 2 * gt_step))
    a_rmse, n_a = rmse_deg(Re, Rg[ka[good]], Rg[kb[good]], ie[good], je[good])

    # --- arm B: slerp the GT to the exact estimate stamps, same pair set
    slerp = Slerp(tg, Rg)
    inside = good & (te[ie] > tg[0]) & (te[je] < tg[-1])
    b_rmse, n_b = rmse_deg(Re, slerp(te[ie[inside]]), slerp(te[je[inside]]),
                           ie[inside], je[inside])

    # the association error actually incurred, and the rotation rate it acts on
    assoc = np.abs(tg[ka[good]] - te[ie[good]])
    w = (Rg[ka[good]].inv() * Rg[kb[good]]).magnitude() / DELTA
    print('%-7s  nearest=%.4f  slerped=%.4f deg   removed=%.4f  '
          'median|assoc|=%.1f ms  median|w|=%.3f rad/s  n=%d/%d'
          % (seq, a_rmse, b_rmse, np.sqrt(max(0.0, a_rmse ** 2 - b_rmse ** 2)),
             1e3 * np.median(assoc), np.median(w), n_a, n_b))
