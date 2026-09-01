"""How wrong is XIVO's initial attitude, and would a better average fix it?

`Estimator::InitializeGravity` averages the first `gravity_init_counter` (20)
accelerometer samples and calls the result gravity. The log line says
"stationary accel samples", but TUM-VI's room sequences are already in motion at
t=0 (|w| = 0.11-0.32 rad/s over those 20 samples). Two error sources follow:
linear acceleration contaminating the mean, and -- for any *longer* window --
the body frame rotating underneath the average, which smears the direction by
roughly |w| * window.

Compare, against ground truth:
  (a) the plain mean of the first N samples, as shipped
  (b) the same mean with each sample de-rotated into the frame at t0 using the
      gyro, which removes the smearing and lets N grow
"""
import numpy as np
from scipy.spatial.transform import Rotation

def gravity_in_mocap(Rg, ab_at_gt):
    """Mean of R_gt * a_b over the whole run: linear acceleration averages out
    over a closed trajectory, leaving -g."""
    v = Rg.apply(ab_at_gt).mean(0)
    return v / np.linalg.norm(v)

def integrate_gyro(t, w):
    """R(t_k) relative to t_0, midpoint rule."""
    R = np.eye(3)
    out = [R]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        rv = 0.5 * (w[k] + w[k - 1]) * dt
        R = R @ Rotation.from_rotvec(rv).as_matrix()
        out.append(R)
    return np.array(out)

print('%-7s %8s %8s %8s %8s %8s' % ('seq', 'N=20', 'N=20drot', 'N=100drot',
                                    'N=200drot', 'N=400drot'))
print('        ' + '-' * 44 + '  (initial tilt error, degrees)')
rows = []
for seq in ('room1', 'room2', 'room3', 'room4', 'room5', 'room6'):
    d = np.loadtxt('dataset-%s_512_16/mav0/imu0/data.csv' % seq,
                   delimiter=',', skiprows=1)
    ti, w, a = d[:, 0] * 1e-9, d[:, 1:4], d[:, 4:7]
    m = np.loadtxt('dataset-%s_512_16/mav0/mocap0/data.csv' % seq,
                   delimiter=',', skiprows=1)
    tm, q = m[:, 0] * 1e-9, m[:, [5, 6, 7, 4]]      # csv is qw qx qy qz
    Rg = Rotation.from_quat(q)

    # accel resampled onto the GT stamps, for the gravity-direction estimate
    ab = np.vstack([np.interp(tm, ti, a[:, i]) for i in range(3)]).T
    ghat = gravity_in_mocap(Rg, ab)

    # ground-truth body-frame gravity direction at the first IMU sample
    k0 = np.argmin(np.abs(tm - ti[0]))
    true_dir = Rg[k0].inv().apply(ghat)
    true_dir /= np.linalg.norm(true_dir)

    Rint = integrate_gyro(ti[:500], w[:500])
    out = []
    for N, drot in ((20, False), (20, True), (100, True), (200, True), (400, True)):
        v = (np.einsum('kij,kj->ki', Rint[:N], a[:N]) if drot else a[:N]).mean(0)
        v /= np.linalg.norm(v)
        out.append(np.rad2deg(np.arccos(np.clip(v @ true_dir, -1, 1))))
    rows.append(out)
    print('%-7s %8.3f %8.3f %8.3f %8.3f %8.3f' % (seq, *out))
r = np.array(rows)
print('%-7s %8.3f %8.3f %8.3f %8.3f %8.3f' % ('mean', *r.mean(0)))
print('\nA tilt error of x deg mis-projects gravity as a horizontal specific'
      '\nforce of 9.81*sin(x) m/s^2: 1 deg -> 0.17 m/s^2 -> 0.086 m of position'
      '\nerror per second squared, until vision corrects it.')
