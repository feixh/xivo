#!/usr/bin/env python3
"""Split the orientation ATE into its tilt and yaw parts.

`ov_eval error_singlerun posyaw` aligns yaw and position only, so the reported
orientation error is (a) the roll/pitch offset plus drift, which no alignment can
remove, and (b) the residual yaw drift about the vertical that a single global
yaw cannot absorb. Those two have entirely different causes -- (a) is a levelling
/ gravity / accel-bias problem, (b) is a yaw-gauge / linearization one -- and the
combined number cannot tell you which one a change moved.

    oridecomp.py <tag> [<tag> ...]

reports, per sequence and averaged over the jitter ensemble, the RMS of each part
in degrees, after doing the same yaw+position alignment ov_eval does.
"""
import glob
import os
import sys

import numpy as np

WS = "/home/ubuntu/workspace/auto-slam-engineer"


def load(path):
    d = np.loadtxt(path, comments="#")
    return d[:, 0], d[:, 1:4], d[:, 4:8]  # t, xyz, (qx qy qz qw)


def quats_to_mats(q):
    """(N,4) with (qx,qy,qz,qw) -> (N,3,3)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def logs(R):
    """(N,3,3) -> (N,3) rotation vectors, via the numerically safe branch."""
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(tr)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2],
                  R[:, 0, 2] - R[:, 2, 0],
                  R[:, 1, 0] - R[:, 0, 1]], -1)
    small = th < 1e-8
    s = np.where(small, 0.5, th / (2 * np.sin(np.where(small, 1.0, th))))
    return v * s[:, None]


def associate(t_est, t_gt, max_diff=0.02):
    """Nearest-neighbour in time, like evaluate_ate.py's associate but vectorized."""
    idx = np.searchsorted(t_gt, t_est)
    idx = np.clip(idx, 1, len(t_gt) - 1)
    left = np.abs(t_est - t_gt[idx - 1]) < np.abs(t_est - t_gt[idx])
    j = np.where(left, idx - 1, idx)
    ok = np.abs(t_est - t_gt[j]) < max_diff
    return np.nonzero(ok)[0], j[ok]


def yaw_align(p_est, p_gt):
    """Best (yaw, translation) taking est into gt: minimize |R_z(a) p_e + t - p_g|."""
    ce, cg = p_est.mean(0), p_gt.mean(0)
    a, b = p_est - ce, p_gt - cg
    # d/da sum |Rz(a) a_i - b_i|^2 = 0  ->  tan(a) = S / C
    C = (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum()
    S = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()
    ang = np.arctan2(S, C)
    c, s = np.cos(ang), np.sin(ang)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz, cg - Rz @ ce


def one_run(traj, gt):
    te, pe, qe = load(traj)
    tg, pg, qg = load(gt)
    i, j = associate(te, tg)
    if len(i) < 50:
        return None
    pe, Re = pe[i], quats_to_mats(qe[i])
    pg, Rg = pg[j], quats_to_mats(qg[j])
    Rz, _ = yaw_align(pe, pg)
    # error rotation in the world frame: R_err = (Rz Re) Rg'
    Rerr = np.einsum("ij,njk,nlk->nil", Rz, Re, Rg)
    e = np.degrees(logs(Rerr))
    yaw, tilt = e[:, 2], e[:, :2]
    # How much of the tilt RMS is a *constant* offset of the published frame (the
    # norm of the mean error vector) and how much actually varies along the run.
    const = np.linalg.norm(tilt.mean(0))
    rms = np.sqrt((tilt ** 2).sum(1).mean())
    return (rms, np.sqrt((yaw ** 2).mean()), np.sqrt((e ** 2).sum(1).mean()),
            const, np.sqrt(max(rms ** 2 - const ** 2, 0.0)))


def main():
    print("%-10s %-7s %-7s %6s %6s %6s %8s %8s" %
          ("tag", "mode", "seq", "tilt", "yaw", "total", "tilt_c", "tilt_v"))
    for tag in sys.argv[1:]:
        root = os.path.join(WS, "experiments/results/orient_%s" % tag)
        for mode in ("mono", "stereo"):
            if not os.path.isdir(os.path.join(root, mode)):
                continue
            seqs = sorted(set(os.path.basename(d).rsplit("_r", 1)[0]
                              for d in glob.glob(os.path.join(root, mode, "*_r*"))))
            acc = []
            for seq in seqs:
                vals = [one_run(os.path.join(d, "traj.txt"),
                                os.path.join(root, "gt", "%s.txt" % seq))
                        for d in sorted(glob.glob(
                            os.path.join(root, mode, "%s_r*" % seq)))]
                vals = [v for v in vals if v]
                m = np.mean(vals, 0)
                acc.append(m)
                print("%-10s %-7s %-7s %6.3f %6.3f %6.3f %8.3f %8.3f" %
                      (tag, mode, seq, m[0], m[1], m[2], m[3], m[4]))
            m = np.mean(acc, 0)
            print("%-10s %-7s %-7s %6.3f %6.3f %6.3f %8.3f %8.3f" %
                  (tag, mode, "MEAN", m[0], m[1], m[2], m[3], m[4]))


if __name__ == "__main__":
    main()
