"""Where in time does the ATE come from?

The R_scale sweep moved per-room ATE by up to 26% between arms that differ only
in how much the filter trusts a right-camera pixel -- far more than such a
change should do smoothly. Either the error is spread out and the arms really
are that different, or it is concentrated in a few excursions whose presence is
effectively a coin flip. Align Horn-style (as evaluate_ate does) and report the
error's concentration over time.
"""
import sys
import numpy as np

def load(p):
    d = np.loadtxt(p)
    return d[:, 0], d[:, 1:4]

def align(model, data):
    """Horn's method, exactly as evaluate_ate.py's align(): no scale."""
    mz = model - model.mean(0)
    dz = data - data.mean(0)
    W = mz.T @ dz
    U, _, Vh = np.linalg.svd(W.T)
    S = np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(Vh)])
    R = U @ S @ Vh
    t = data.mean(0) - R @ model.mean(0)
    return (R @ model.T).T + t

def main():
    tg, pg = load(sys.argv[1])
    te, pe = load(sys.argv[2])
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    j = np.clip(np.searchsorted(tg, te), 1, len(tg) - 1)
    j = np.where(np.abs(tg[j] - te) < np.abs(tg[j - 1] - te), j, j - 1)
    ok = np.abs(tg[j] - te) < tol
    te, pe, pgm = te[ok], pe[ok], pg[j[ok]]
    al = align(pe, pgm)
    err = np.linalg.norm(al - pgm, axis=1)
    rmse = np.sqrt((err ** 2).mean())
    e2 = err ** 2
    order = np.argsort(-e2)
    n = len(err)
    top10 = e2[order[:max(1, n // 10)]].sum() / e2.sum()
    # RMSE with the worst 10% of samples excised, to see the "quiet" level
    keep = np.setdiff1d(np.arange(n), order[:max(1, n // 10)])
    rmse90 = np.sqrt(e2[keep].mean())
    # coarse time profile
    t0 = te[0]
    nb = 10
    bins = np.linspace(0, te[-1] - t0 + 1e-9, nb + 1)
    prof = []
    for k in range(nb):
        m = ((te - t0) >= bins[k]) & ((te - t0) < bins[k + 1])
        prof.append(np.sqrt(e2[m].mean()) if m.sum() else float('nan'))
    print('%-10s n=%4d rmse %.4f  worst10%%->%4.0f%% of SSE  rmse_excl_worst10%% %.4f'
          % (sys.argv[4] if len(sys.argv) > 4 else '', n, rmse, 100 * top10, rmse90))
    print('           decile rmse: ' + ' '.join('%.3f' % v for v in prof))

main()
