"""Is there a real gyro scale/misalignment error on TUM-VI?

The M5 RPE decomposition left a term proportional to rotation magnitude, worth
~0.49 deg of the 0.62 deg RPE_rot -- the largest of the three. A gyro scale
factor or axis misalignment is exactly that signature, and XIVO's configs set
imu_calib.Cg to identity and never estimate it.

Fit  v_gt = Cg * v_gyro - bg*dt  over every consecutive mocap pair, where v is
the rotation vector of the interval: v_gt from the mocap attitudes, v_gyro from
the gyro integrated over the same interval.

The question is not just "is Cg != I" -- least squares always says yes -- but
whether the six sequences AGREE. A sensor property is the same in all six; fit
noise is not.
"""
import numpy as np
from scipy.spatial.transform import Rotation

def fit(seq):
    d = np.loadtxt('dataset-%s_512_16/mav0/imu0/data.csv' % seq,
                   delimiter=',', skiprows=1)
    ti, w = d[:, 0] * 1e-9, d[:, 1:4]
    m = np.loadtxt('dataset-%s_512_16/mav0/mocap0/data.csv' % seq,
                   delimiter=',', skiprows=1)
    tm, q = m[:, 0] * 1e-9, m[:, [5, 6, 7, 4]]
    R = Rotation.from_quat(q)

    dt = np.diff(tm)
    # Drop pairs spanning a mocap dropout, and pairs outside the IMU's coverage.
    ok = (dt < 1.5 * np.median(dt)) & (tm[:-1] > ti[0]) & (tm[1:] < ti[-1])
    idx = np.flatnonzero(ok)

    v_gt = (R[idx].inv() * R[idx + 1]).as_rotvec()
    # The interval is ~1.7 IMU samples, so the midpoint gyro times dt is the
    # integral to the same second order as v_gt itself.
    tmid = 0.5 * (tm[idx] + tm[idx + 1])
    wmid = np.vstack([np.interp(tmid, ti, w[:, i]) for i in range(3)]).T
    v_w = wmid * dt[idx, None]

    A = np.hstack([v_w, -dt[idx, None] * np.ones((len(idx), 1))])
    # Solve the three rows of Cg independently; column 4 is that row's bias.
    Cg, bg, res = np.zeros((3, 3)), np.zeros(3), np.zeros(3)
    for r in range(3):
        X = np.hstack([np.zeros((len(idx), 0)), A])
        sol, *_ = np.linalg.lstsq(X, v_gt[:, r], rcond=None)
        Cg[r], bg[r] = sol[:3], sol[3]
        res[r] = np.std(v_gt[:, r] - X @ sol)
    raw = np.std(v_gt - v_w, axis=0)
    return Cg, bg, raw, res, len(idx)

print('Cg fitted per sequence (deviation from identity, in %):')
print('%-7s %s' % ('seq', '  '.join('%6s' % s for s in
      ('11-1', '12', '13', '21', '22-1', '23', '31', '32', '33-1'))))
Cgs = []
for seq in ('room1', 'room2', 'room3', 'room4', 'room5', 'room6'):
    Cg, bg, raw, res, n = fit(seq)
    Cgs.append(Cg)
    dev = 100 * (Cg - np.eye(3)).ravel()
    print('%-7s %s   n=%d' % (seq, '  '.join('%6.2f' % v for v in dev), n))
C = np.array(Cgs)
print('%-7s %s' % ('mean', '  '.join('%6.2f' % v for v in
      100 * (C.mean(0) - np.eye(3)).ravel())))
print('%-7s %s' % ('std', '  '.join('%6.2f' % v for v in
      100 * C.std(0).ravel())))
print('\nbg fitted (deg/s):')
for seq, Cg in zip(('room1','room2','room3','room4','room5','room6'), Cgs):
    _, bg, raw, res, _ = fit(seq)
    print('  %-7s %s   residual std per axis (deg/s): %s   raw: %s' %
          (seq, np.array2string(np.rad2deg(bg), precision=3),
           np.array2string(np.rad2deg(res / np.median(np.diff(
               np.loadtxt('dataset-%s_512_16/mav0/mocap0/data.csv' % seq,
                          delimiter=',', skiprows=1)[:, 0] * 1e-9))), precision=3),
           ''))
