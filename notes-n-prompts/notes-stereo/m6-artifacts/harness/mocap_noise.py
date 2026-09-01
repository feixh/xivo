"""How noisy is TUM-VI's mocap *attitude*, and what floor does that put under RPE_rot?

RPE_rot compares a relative rotation from the estimate against one from the
mocap. Whatever attitude noise the mocap has enters the metric directly: for
white per-sample noise of std sigma, a pair contributes sqrt(2)*sigma of apparent
error no matter how good the estimator is. Unlike the association artifact this
cannot be removed by interpolating, because it is in the reference itself.

Estimate sigma by local polynomial smoothing: over a short window the true
attitude is smooth (the rig is hand-held, bandwidth a few Hz), so the residual
about a fitted cubic is noise. Reported per axis in the body frame.
"""
import numpy as np
from scipy.spatial.transform import Rotation

WIN, DEG = 9, 3          # 9 samples = 75 ms at 120 Hz, cubic

print('%-7s %22s %10s %10s' % ('seq', 'mocap attitude noise', 'RPE floor',
                               'blocks'))
print('%-7s %22s %10s' % ('', 'per axis, deg', 'deg'))
floors = []
for seq in ('room1', 'room2', 'room3', 'room4', 'room5', 'room6'):
    m = np.loadtxt('dataset-%s_512_16/mav0/mocap0/data.csv' % seq,
                   delimiter=',', skiprows=1)
    t, q = m[:, 0] * 1e-9, m[:, [5, 6, 7, 4]]
    R = Rotation.from_quat(q)
    step = np.median(np.diff(t))

    # split at dropouts so a gap is never fitted across
    brk = np.flatnonzero(np.diff(t) > 1.5 * step) + 1
    res, nblk = [], 0
    for blk in np.split(np.arange(len(t)), brk):
        if len(blk) < 4 * WIN:
            continue
        nblk += 1
        h = WIN // 2
        for c in range(h, len(blk) - h, WIN):   # non-overlapping centres
            w = blk[c - h:c + h + 1]
            # rotation vectors relative to the window centre: small, so a
            # polynomial fit in this chart is well conditioned
            v = (R[blk[c]].inv() * R[w]).as_rotvec()
            x = t[w] - t[blk[c]]
            V = np.vander(x, DEG + 1)
            fit = V @ np.linalg.lstsq(V, v, rcond=None)[0]
            res.append(v - fit)
    res = np.rad2deg(np.vstack(res))
    # A degree-DEG fit over WIN points absorbs part of the noise; scale back by
    # the residual degrees of freedom so sigma is not underestimated.
    sigma = res.std(0) * np.sqrt(WIN / (WIN - DEG - 1))
    floor = np.sqrt(2) * np.linalg.norm(sigma)   # 3-axis magnitude, two poses
    floors.append(floor)
    print('%-7s %22s %10.4f %10d'
          % (seq, np.array2string(sigma, 3,
                                  suppress_small=True), floor, nblk))
print('\nmean RPE_rot floor from mocap attitude noise alone: %.4f deg'
      % np.mean(floors))
