#!/usr/bin/env python3
"""Is Stage B's covariance worth seeding the filter with?

M6 wants to replace three config priors -- `P.Vsb`, `P.bg`, `P.ba` -- with the
bundle adjustment's own marginal over the same quantities, so that a filter
handed an easy window starts confident and one handed a hard window starts
humble. Before anything consumes that matrix it has to be checked against the
truth, because there are two documented reasons to expect it to be wrong:

  - `ba` is priored at 0.01 m/s^2 rather than estimated (init_ba.h), so its block
    reports the prior back, not an estimate's uncertainty.
  - the IMU edges are whitened by the diagonal preintegration approximation
    (`BAOptions::sigma_g`), which drops the alpha/beta correlation and the gyro's
    contribution to alpha.

So this script scores the covariance the way a covariance has to be scored: at
each window, against the actual error of that window's estimate.

Two questions, and they have different answers and different consequences.

  scale  Is `|e| / sigma` about 1? If it is 10, seeding verbatim makes the filter
         100x overconfident in variance, which is the failure that looks like a
         diverging run rather than a slightly worse one.
  rank   Across windows, does a larger reported sigma go with a larger true
         error? *This* is what a seeded covariance buys that a config prior
         cannot: even at the wrong scale, a matrix that ranks the windows lets
         one inflation constant serve every sequence. If the rank correlation is
         zero, the covariance carries no per-window information, a single
         inflated sigma is just a config prior with extra steps, and M6 should be
         reverted rather than tuned.

Truth comes from `seed_error.Seq`, i.e. EuRoC's `state_groundtruth_estimate0`,
including both bias columns -- the same source and the same handoff instant M5
scored the seeds at, so the errors here are comparable to
results/dyninit/m5-seed/ line for line.

Usage (from the repository root):
  python3 notes-n-prompts/notes-dyninit/harness/m6_cov.py --start 55
  python3 notes-n-prompts/notes-dyninit/harness/m6_cov.py --start 0 --frames 41
  # any unrecognized flag is passed through to bin/linear_probe
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seed_error import EUROC_SEQS, G, Seq, angle_deg  # noqa: E402

BLOCKS = (("v", 0, "m/s"), ("bg", 3, "rad/s"), ("ba", 6, "m/s^2"))


def run_probe(xivo, root, cfg, seq, start, frames, extra):
    """`bin/linear_probe -ba -cov -at_frame -1` -> (est dict, C_9x9) or (None, why)."""
    cmd = [os.path.join(xivo, "bin", "linear_probe"), "-cfg", cfg, "-root", root,
           "-dataset", "euroc", "-seq", seq, "-start", str(start),
           "-frames", str(frames), "-ba", "-cov", "-at_frame", "-1"] + list(extra)
    r = subprocess.run(cmd, cwd=xivo, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        return None, f"rc={r.returncode} {r.stderr.strip()[-160:]}"
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    body = [l for l in lines if not l.startswith("#")]
    if len(body) < 2:
        return None, "no BA row: " + " | ".join(lines)[:160]
    f = body[-1].split()[1:]
    est = dict(v=np.array([float(x) for x in f[0:3]]),
               g=np.array([float(x) for x in f[3:6]]),
               bg=np.array([float(x) for x in f[6:9]]),
               ba=np.array([float(x) for x in f[9:12]]),
               pmed=float(f[13]), imu=float(f[14]), rj=int(f[16]),
               ok=int(f[17]))
    at = next((l for l in lines if l.startswith("#at_frame")), "")
    m = re.search(r"t=([-+0-9.eE]+) s", at)
    est["t"] = float(m.group(1)) if m else float("nan")
    c9 = next((l for l in lines if l.startswith("#cov9")), None)
    if c9 is None:
        return None, "no covariance (" + next(
            (l for l in lines if l.startswith("#cov")), "no #cov line") + ")"
    C = np.array([float(x) for x in c9.split()[1:]]).reshape(9, 9)
    return (est, C), None


def mahalanobis(e, C):
    """sqrt(e' C^-1 e / 3): 1 when the block is calibrated, whatever its scale."""
    try:
        return float(np.sqrt(max(e @ np.linalg.solve(C, e), 0.0) / len(e)))
    except np.linalg.LinAlgError:
        return float("nan")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")

    def rank(x):
        r = np.empty(len(x))
        r[np.argsort(x)] = np.arange(len(x))
        return r

    ra, rb = rank(a[m]), rank(b[m])
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/euroc")
    ap.add_argument("--cfg", default="cfg/euroc_stereo.json")
    ap.add_argument("--seqs", nargs="+", default=EUROC_SEQS)
    ap.add_argument("--start", type=float, default=None,
                    help="window start, s after the first IMU sample; default is "
                         "the dispatcher's own, i.e. the first image")
    ap.add_argument("--frames", type=int, default=41,
                    help="window frames (the shipped `window_frames`)")
    ap.add_argument("--prior", type=float, nargs=3, default=(0.5, 0.01, 0.25),
                    metavar=("V", "BG", "BA"),
                    help="the config priors the covariance would replace, as "
                         "isotropic sigmas for (v, bg, ba). Defaults are the "
                         "shipped euroc `P.Vsb/P.bg/P.ba` -- which are standard "
                         "deviations, not variances: estimator.cpp squares the "
                         "whole of `P_` after reading them. Scored by the same "
                         "statistic as the covariance, so the two rows of the "
                         "summary are comparable.")
    args, extra = ap.parse_known_args()
    xivo = os.getcwd()

    print(f"Stage B covariance vs truth: {args.frames}-frame window, state and "
          f"covariance at its LAST frame"
          + (f", from t={args.start:g} s" if args.start is not None else
             ", from the first image"))
    print("z = sqrt(e' C^-1 e / 3), per block: 1 is calibrated, >1 overconfident\n")
    hdr = (f'{"sequence":<18}{"t_ho":>7}'
           f'{"|e_v|":>9}{"s_v":>10}{"z_v":>8}'
           f'{"|e_bg|":>9}{"s_bg":>10}{"z_bg":>8}'
           f'{"|e_ba|":>9}{"s_ba":>10}{"z_ba":>8}'
           f'{"tilt":>7}{"pmed":>7}')
    print(hdr)

    rows = []
    for seq in args.seqs:
        s = Seq(args.root, seq)
        start = args.start
        if start is None:
            cam = np.loadtxt(os.path.join(args.root, seq, "mav0", "cam0",
                                          "data.csv"), delimiter=",",
                             comments="#", usecols=0)
            start = s.rel(cam[0])
        got, err = run_probe(xivo, args.root, args.cfg, seq, start, args.frames,
                             extra)
        if err:
            print(f"{seq:<18}  probe: {err}")
            continue
        est, C = got
        gt = s.gt_at(est["t"])
        if gt is None:
            print(f"{seq:<18}{est['t']:>7.2f}  (no groundtruth at the handoff)")
            continue
        R, v_w, bg_t, ba_t = gt
        e = dict(v=est["v"] - R.T @ v_w, bg=est["bg"] - bg_t, ba=est["ba"] - ba_t)
        tilt = angle_deg(est["g"], R.T @ np.array([0.0, 0.0, -G]))
        row = dict(seq=seq, t=est["t"], tilt=tilt, pmed=est["pmed"],
                   imu=est["imu"])
        line = f"{seq:<18}{est['t']:>7.2f}"
        for name, i, _ in BLOCKS:
            Cb = C[i:i + 3, i:i + 3]
            en = float(np.linalg.norm(e[name]))
            # One sigma per block, for the rank question: the RMS of the block's
            # own axis sigmas, which is `sqrt(trace/3)` and is what an isotropic
            # replacement for the block would have to be.
            sn = float(np.sqrt(max(np.trace(Cb), 0.0) / 3.0))
            z = mahalanobis(e[name], Cb)
            row[f"e_{name}"], row[f"s_{name}"], row[f"z_{name}"] = en, sn, z
            row[f"_e_{name}"] = e[name] # the vector, for the prior's own z below
            fmt = ">9.5f" if name == "bg" else ">9.4f"
            line += f"{en:{fmt}}{sn:>10.2e}{z:>8.1f}"
        line += f"{tilt:>7.3f}{est['pmed']:>7.3f}"
        print(line)
        rows.append(row)

    if not rows:
        return 1
    print()
    print(f'{"block":<8}{"median z":>10}{"mean z":>9}{"max z":>9}'
          f'{"rms e/s":>9}{"spearman(s,e)":>15}{"sigma range":>22}')
    for bi, (name, _, unit) in enumerate(BLOCKS):
        e = np.array([r[f"e_{name}"] for r in rows])
        s = np.array([r[f"s_{name}"] for r in rows])
        z = np.array([r[f"z_{name}"] for r in rows])
        print(f"{name:<8}{np.median(z):>10.1f}{np.mean(z):>9.1f}{np.max(z):>9.1f}"
              f"{np.sqrt(np.mean((e / s) ** 2)):>9.1f}"
              f"{spearman(s, e):>15.2f}"
              f"{s.min():>11.2e}{s.max():>11.2e}")
        # The incumbent, by the same statistic: one isotropic sigma per block, the
        # same for every window. It cannot rank anything by construction -- that is
        # the whole of what a seeded covariance was supposed to add -- so what
        # matters in this row is only whether its z is nearer 1 than the row above.
        p = float(args.prior[bi])
        zp = np.array([mahalanobis(x, p * p * np.eye(3))
                       for x in ([r["_e_" + name] for r in rows])])
        print(f'{"  prior":<8}{np.median(zp):>10.2f}{np.mean(zp):>9.2f}'
              f'{np.max(zp):>9.2f}{np.sqrt(np.mean((e / p) ** 2)):>9.2f}'
              f'{"n/a":>15}{p:>22.2e}')
    print(f"\n{len(rows)} windows. rms e/s is the inflation a single constant would"
          "\nhave to supply; spearman(s, e) is whether the matrix ranks the windows"
          "\nat all, which is the only thing it can offer over a config prior.")

    # If the covariance does not rank the windows, the next question is whether
    # anything the solve already reports does -- because a scalar that ranks
    # difficulty can inflate a config prior per window, which is what M6 wanted
    # from the covariance in the first place. The two candidates are the residuals
    # themselves, and the covariance rescaled by them (the usual chi-square
    # rescale for a mis-whitened cost).
    pmed = np.array([r["pmed"] for r in rows])
    imu = np.array([r["imu"] for r in rows])
    print(f'\nrank correlation with the true error, per candidate predictor'
          f'\n{"predictor":<22}' + "".join(f'{"e_" + n:>10}' for n, _, _ in BLOCKS))
    for label, pred in (("sigma (block)", None),
                        ("pixel median", pmed),
                        ("imu rms", imu),
                        ("sigma * imu rms", "imu"),
                        ("sigma * pixel median", "pmed")):
        line = f"{label:<22}"
        for name, _, _ in BLOCKS:
            e = np.array([r[f"e_{name}"] for r in rows])
            s = np.array([r[f"s_{name}"] for r in rows])
            x = s if pred is None else (
                pred if isinstance(pred, np.ndarray) else
                s * (imu if pred == "imu" else pmed))
            line += f"{spearman(x, e):>10.2f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
