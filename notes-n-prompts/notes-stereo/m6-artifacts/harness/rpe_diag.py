"""Is the residual RPE_rot proportional to how much the rig actually rotated?

A gyro *scale/misalignment* error (the Cg matrix, which this build does not
estimate) produces a rotation error proportional to the rotation itself. Random
walk / bias error and the metric's own timestamp quantization do not. Binning
the per-pair rotation error by the ground-truth relative rotation angle
separates the two.
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation

def load(p):
    d = np.loadtxt(p)
    return d[:, 0], d[:, 1:4], Rotation.from_quat(d[:, 4:8])

tg, pg, Rg = load(sys.argv[1])
te, pe, Re = load(sys.argv[2])
delta = 1.0

# Pair up exactly as evaluate_rpe.py does: pairs from est stamps, GT by nearest.
def nearest(t, ts):
    j = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
    return np.where(np.abs(ts[j] - t) < np.abs(ts[j - 1] - t), j, j - 1)

j1 = nearest(te + delta, te)
i0 = np.arange(len(te))
ok = j1 != len(te) - 1
i0, j1 = i0[ok], j1[ok]
g0, g1 = nearest(te[i0], tg), nearest(te[j1], tg)

# evaluate_rpe.py drops any pair whose ground-truth association is worse than
# two GT intervals. That is not a formality here: TUM-VI's mocap has dropouts of
# up to several seconds, and without this filter a handful of pairs straddling a
# gap dominate the RMSE.
gt_tol = 2 * np.median(np.diff(tg))
good = ((np.abs(tg[g0] - te[i0]) <= gt_tol) & (np.abs(tg[g1] - te[j1]) <= gt_tol))
i0, j1, g0, g1 = i0[good], j1[good], g0[good], g1[good]

est_rel = Re[i0].inv() * Re[j1]
gt_rel = Rg[g0].inv() * Rg[g1]
err = np.rad2deg(np.linalg.norm((est_rel.inv() * gt_rel).as_rotvec(), axis=1))
mag = np.rad2deg(np.linalg.norm(gt_rel.as_rotvec(), axis=1))

print('  pairs %d   RPE_rot rmse %.4f deg' % (len(err), np.sqrt((err**2).mean())))
edges = [0, 10, 20, 40, 60, 90, 130, 400]
print('  |gt rel rot|      n    err_rmse   err/rot')
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (mag >= lo) & (mag < hi)
    if m.sum() < 20:
        continue
    r = np.sqrt((err[m]**2).mean())
    print('  %3d-%3d deg   %5d   %7.4f    %6.4f'
          % (lo, hi, m.sum(), r, r / max(mag[m].mean(), 1e-9)))
# Best-fit pure-proportional model: err ~ k * rot
k = (err * mag).sum() / (mag**2).sum()
resid = err - k * mag
print('  proportional fit: k = %.5f (%.3f%% of rotation), '
      'residual rmse %.4f deg' % (k, 100 * k, np.sqrt((resid**2).mean())))
