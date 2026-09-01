"""Is the residual ATE a global scale error?

The per-decile error profile is flat from the very first seconds, which is not
what accumulating drift looks like -- it is what a distorted trajectory *shape*
looks like. The cheapest shape error to test for is a global scale factor, so
align with and without scale and compare.
"""
import sys
import numpy as np

def load(p):
    d = np.loadtxt(p)
    return d[:, 0], d[:, 1:4]

def align(model, data, with_scale):
    mz = model - model.mean(0)
    dz = data - data.mean(0)
    W = mz.T @ dz
    U, D, Vh = np.linalg.svd(W.T)
    S = np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(Vh)])
    R = U @ S @ Vh
    s = 1.0
    if with_scale:
        s = (D * np.diag(S)).sum() / (mz ** 2).sum()
    t = data.mean(0) - s * R @ model.mean(0)
    return s * (R @ model.T).T + t, s

def main():
    tg, pg = load(sys.argv[1])
    te, pe = load(sys.argv[2])
    tol = float(sys.argv[3])
    j = np.clip(np.searchsorted(tg, te), 1, len(tg) - 1)
    j = np.where(np.abs(tg[j] - te) < np.abs(tg[j - 1] - te), j, j - 1)
    ok = np.abs(tg[j] - te) < tol
    pe, pgm = pe[ok], pg[j[ok]]
    a0, _ = align(pe, pgm, False)
    a1, s = align(pe, pgm, True)
    r0 = np.sqrt((np.linalg.norm(a0 - pgm, axis=1) ** 2).mean())
    r1 = np.sqrt((np.linalg.norm(a1 - pgm, axis=1) ** 2).mean())
    extent = np.linalg.norm(pgm.max(0) - pgm.min(0))
    print('%-10s se3 %.4f   sim3 %.4f   scale %.5f (%+.2f%%)  gt extent %.2f m'
          % (sys.argv[4], r0, r1, s, 100 * (s - 1), extent))

main()
