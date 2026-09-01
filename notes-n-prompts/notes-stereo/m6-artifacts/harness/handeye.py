"""Estimate the constant marker<-body transform TUM-VI's evaluation ignores.

XIVO's eval-mode saver writes `gsb`, the *IMU body* pose. TUM-VI's
`mav0/mocap0/data.csv` is the pose of the *mocap marker frame*, which is a
different rigid frame on the same rig; the dataset ships no marker extrinsic.
So even a flawless estimator is scored against a trajectory related to its
output by an unknown constant `g_bm`, which shows up as

  R_gt(t) = R_w R_sb(t) R_bm          (a conjugation, so it inflates RPE_rot)
  p_gt(t) = R_w (p_sb(t) + R_sb(t) t_bm) + p_w
                                      (a *rotation-dependent* position offset,
                                       which Horn alignment cannot absorb)

Solve for (R_bm, t_bm, R_w, p_w) in closed form and report how much of the
measured error they account for.
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation

def load(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1:4], Rotation.from_quat(d[:, 4:8])

def associate(te, tg, tol):
    j = np.searchsorted(tg, te)
    j = np.clip(j, 1, len(tg) - 1)
    pick = np.where(np.abs(tg[j] - te) < np.abs(tg[j - 1] - te), j, j - 1)
    ok = np.abs(tg[pick] - te) < tol
    return np.where(ok)[0], pick[ok]

def kabsch(A, B):
    """R minimizing |A - R B| over columns of the 3xN clouds (no centering:
    these are rotation *axes* through the origin, not point clouds)."""
    U, _, Vt = np.linalg.svd(A @ B.T)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt

def solve(pe, Re, pg, Rg, stride):
    # --- R_bm from relative rotations: axis_est = R_bm axis_gt, angles equal.
    i0 = np.arange(0, len(pe) - stride)
    i1 = i0 + stride
    a = (Re[i0].inv() * Re[i1]).as_rotvec()
    b = (Rg[i0].inv() * Rg[i1]).as_rotvec()
    n = np.linalg.norm(a, axis=1)
    keep = n > np.deg2rad(5.0)          # tiny relative rotations carry no axis
    R_bm = kabsch(a[keep].T, b[keep].T)
    R_bm = Rotation.from_matrix(R_bm)

    # --- R_w by averaging R_gt (R_sb R_bm)^T
    Rw = (Rg * (Re * R_bm).inv()).mean()

    # --- t_bm, p_w by linear least squares on positions
    N = len(pe)
    A = np.zeros((3 * N, 6))
    y = np.zeros(3 * N)
    Rwm = Rw.as_matrix()
    Rem = Re.as_matrix()
    for k in range(N):
        A[3*k:3*k+3, :3] = Rwm @ Rem[k]
        A[3*k:3*k+3, 3:] = np.eye(3)
        y[3*k:3*k+3] = pg[k] - Rwm @ pe[k]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return R_bm, sol[:3], Rw, sol[3:]

def main():
    gt_f, est_f, out_f = sys.argv[1], sys.argv[2], sys.argv[3]
    tg, pg, Rg = load(gt_f)
    te, pe, Re = load(est_f)
    ie, ig = associate(te, tg, 0.005)
    R_bm, t_bm, Rw, p_w = solve(pe[ie], Re[ie], pg[ig], Rg[ig], stride=20)
    ang = np.rad2deg(np.linalg.norm(R_bm.as_rotvec()))
    print('  g_bm: rotation %.3f deg, translation %.4f m  %s'
          % (ang, np.linalg.norm(t_bm), np.round(t_bm, 4)))
    # Re-express the *whole* estimate as a marker pose, in its own world frame
    # (leave the world alignment to the evaluation tool, as before).
    Rm = Re * R_bm
    pm = pe + Re.apply(np.tile(t_bm, (len(pe), 1)))
    np.savetxt(out_f, np.column_stack([te, pm, Rm.as_quat()]), fmt='%.9f')

main()
