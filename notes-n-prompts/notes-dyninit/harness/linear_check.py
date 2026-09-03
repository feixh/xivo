#!/usr/bin/env python3
"""Score Stage A's real-data output against EuRoC ground truth.

`bin/linear_probe` runs the closed-form linear initializer on one window of a
real sequence and prints the velocity and gravity it recovered, both in `I0` --
the IMU frame at the window's first frame. This script supplies the answer.

Ground truth gives position, attitude and velocity in the gravity-aligned world
frame, so the two quantities Stage A claims are directly checkable:

    v_I0 = R(t0)' v_world(t0)          g_I0 = R(t0)' [0, 0, -9.81]

Reporting the gravity error as an angle rather than a vector norm is deliberate:
`|g|` is enforced by construction, so the only part of gravity that can be wrong
is its direction, and that error lands in the filter's roll and pitch where no
trajectory alignment can remove it.

Note what this can and cannot settle. The velocity error it reports is Stage A's
*total* error, which is dominated by the gyro-bias term Stage A holds at a prior
and cannot see -- that is Stage B's job, so a nonzero number here is the expected
result, not a failure. What it does settle is that the implementation works on
real tracks at all, and it measures how large a seed error Stage B must absorb.

Usage (from the repository root):
  python3 notes-n-prompts/notes-dyninit/harness/linear_check.py \
      --root ../data/euroc --cfg cfg/euroc_stereo.json
  python3 notes-n-prompts/notes-dyninit/harness/linear_check.py \
      --seqs MH_01_easy MH_02_easy --start 1.1 --frames 31 --span-sweep
"""

import argparse
import os
import subprocess
import sys

import numpy as np

EUROC_SEQS = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
    "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]

G = 9.81


def quat_to_R(q):
    """q = [w, x, y, z] -> R_world<-body."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def first_imu_ns(seq_dir):
    p = os.path.join(seq_dir, "mav0/imu0/data.csv")
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return int(line.split(",")[0])
    raise RuntimeError("no IMU samples in %s" % p)


def load_gt(seq_dir, t0_ns):
    p = os.path.join(seq_dir, "mav0/state_groundtruth_estimate0/data.csv")
    if not os.path.exists(p):
        return None
    d = np.loadtxt(p, delimiter=",", comments="#")
    return (d[:, 0] - t0_ns) * 1e-9, d[:, 4:8], d[:, 8:11], d[:, 11:14], d[:, 14:17]


def gt_at(gt, t):
    """Nearest-sample attitude (quaternions do not interpolate linearly) and
    linearly interpolated velocity and biases. GT is 200 Hz, so nearest is
    within 2.5 ms -- far inside the error being measured."""
    tg, quat, vel, bg, ba = gt
    if t < tg[0] or t > tg[-1]:
        return None
    i = int(np.argmin(np.abs(tg - t)))
    v = np.array([np.interp(t, tg, vel[:, k]) for k in range(3)])
    bgi = np.array([np.interp(t, tg, bg[:, k]) for k in range(3)])
    bai = np.array([np.interp(t, tg, ba[:, k]) for k in range(3)])
    return quat[i], v, bgi, bai


def run_probe(binary, cfg, root, seq, start, frames, extra):
    cmd = [binary, "-cfg", cfg, "-root", root, "-seq", seq,
           "-start", "%.6f" % start, "-frames", "%d" % frames] + extra
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout.strip()
    if not text or "FAILED" in text:
        return None, text.split("\n")[-1] if text else out.stderr.strip()[-200:]
    # Stage A's row has 20 fields and Stage B's 18; pick by count rather than by
    # position so that adding a column to either one cannot silently shift the
    # other's parse.
    a = b = None
    for line in text.split("\n"):
        f = line.split()
        if len(f) == 20:
            a = f
        elif len(f) == 19 and not f[0].startswith("#"):
            b = f
    if a is None:
        return None, text.split("\n")[-1]
    r = dict(t0=float(a[1]), span=float(a[2]), frames=int(a[3]),
             trks=int(a[4]), rows=int(a[5]), obs=int(a[6]),
             v=np.array([float(a[7]), float(a[8]), float(a[9])]),
             g=np.array([float(a[10]), float(a[11]), float(a[12])]),
             gp=np.array([float(a[13]), float(a[14]), float(a[15])]),
             resid=float(a[16]), pr_ang=float(a[17]), flip=int(a[18]),
             gcond=float(a[19]))
    if b is not None:
        r.update(bv=np.array([float(x) for x in b[1:4]]),
                 bg_=np.array([float(x) for x in b[4:7]]),
                 bbg=np.array([float(x) for x in b[7:10]]),
                 bba=np.array([float(x) for x in b[10:13]]),
                 bpix=float(b[13]), bpmed=float(b[14]), bimu=float(b[15]),
                 bit=int(b[16]), brej=int(b[17]), bok=int(b[18]))
    return r, None


def angle(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.arccos(np.clip(c, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/euroc")
    ap.add_argument("--cfg", default="cfg/euroc_stereo.json")
    ap.add_argument("--binary", default="bin/linear_probe")
    ap.add_argument("--seqs", nargs="*", default=EUROC_SEQS)
    ap.add_argument("--start", type=float, default=1.1)
    ap.add_argument("--frames", type=int, default=31)
    ap.add_argument("--auto-start", action="store_true",
                    help="Start each window just after that sequence's ground "
                         "truth begins, which varies from 0.9 s to 2.4 s.")
    ap.add_argument("--span-sweep", action="store_true",
                    help="Sweep window length instead of running one setting.")
    ap.add_argument("--gt-bias", action="store_true",
                    help="Pass GT's solved biases as the prior, to separate the "
                         "bias error from everything else.")
    ap.add_argument("--ba", action="store_true",
                    help="Also run Stage B and score it. Reports Stage A and "
                         "Stage B side by side, plus the recovered biases "
                         "against EuRoC's own solved ones -- an external check.")
    # One string, split here, rather than nargs="*": argparse eats a bare
    # "-ba_iters" as one of its own options and reports it as unrecognized.
    ap.add_argument("--extra", default="",
                    help="Extra probe flags as one quoted string, e.g. "
                         "--extra '-ba_iters 200 -sigma_ba_prior 0.2'.")
    args = ap.parse_args()

    frame_sets = [11, 21, 31, 41, 61] if args.span_sweep else [args.frames]

    if args.ba:
        print("%-20s %4s %7s %8s %8s %8s %8s %8s %8s %8s %8s %6s %5s" %
              ("sequence", "fr", "|v|gt", "vA", "vB", "gA_d", "gB_d",
               "bg_err", "ba_err", "|ba|gt", "pmed", "imu", "it/rj"))
    else:
        print("%-20s %4s %7s %6s %5s %8s %8s %8s %8s %7s %7s %4s" %
              ("sequence", "fr", "span", "trks", "flip", "|v|gt", "v_err",
               "g_err_d", "gp_err_d", "pr_ang", "resid", "used"))
    rows = []
    for seq in args.seqs:
        seq_dir = os.path.join(args.root, seq)
        try:
            t0_ns = first_imu_ns(seq_dir)
        except Exception as e:
            print("%-20s  no imu: %s" % (seq, e))
            continue
        gt = load_gt(seq_dir, t0_ns)
        if gt is None:
            print("%-20s  no ground truth" % seq)
            continue

        start = args.start
        if args.auto_start:
            start = max(args.start, float(gt[0][0]) + 0.05)
        for nfr in frame_sets:
            extra = args.extra.split()
            if args.ba and "-ba" not in extra:
                extra += ["-ba"]
            if args.gt_bias:
                # Peek at GT's solved bias at the window start and hand it to the
                # probe, which isolates "everything except the bias".
                got = gt_at(gt, start)
                if got is not None:
                    _, _, bgi, bai = got
                    extra += ["-bgx", "%.9f" % bgi[0], "-bgy", "%.9f" % bgi[1],
                              "-bgz", "%.9f" % bgi[2], "-bax", "%.9f" % bai[0],
                              "-bay", "%.9f" % bai[1], "-baz", "%.9f" % bai[2]]
            r, err = run_probe(args.binary, args.cfg, args.root, seq, start,
                               nfr, extra)
            if r is None:
                print("%-20s %4d  probe failed: %s" % (seq, nfr, err))
                continue
            got = gt_at(gt, r["t0"])
            if got is None:
                print("%-20s %4d  no GT at t=%.3f (GT spans the mocap room "
                      "only)" % (seq, nfr, r["t0"]))
                continue
            q, v_w, bg_gt, ba_gt = got
            R = quat_to_R(q)
            v_i0 = R.T @ v_w
            g_i0 = R.T @ np.array([0.0, 0.0, -G])
            v_err = float(np.linalg.norm(r["v"] - v_i0))
            g_err = np.degrees(angle(r["g"], g_i0))
            gp_err = np.degrees(angle(r["gp"], g_i0))
            if args.ba:
                if "bv" not in r:
                    print("%-20s %4d  Stage B produced no row" % (seq, nfr))
                    continue
                vB = float(np.linalg.norm(r["bv"] - v_i0))
                gB = np.degrees(angle(r["bg_"], g_i0))
                # The external check: EuRoC solves for the biases itself, in its
                # own batch estimator, so agreement here is not self-consistency.
                bg_err = float(np.linalg.norm(r["bbg"] - bg_gt))
                ba_err = float(np.linalg.norm(r["bba"] - ba_gt))
                print("%-20s %4d %7.4f %8.4f %8.4f %8.3f %8.3f %8.5f %8.4f "
                      "%8.4f %8.3f %8.3f %6s" %
                      (seq, nfr, float(np.linalg.norm(v_i0)), v_err, vB,
                       g_err, gB, bg_err, ba_err,
                       float(np.linalg.norm(ba_gt)), r["bpmed"], r["bimu"],
                       "%d/%d" % (r["bit"], r["brej"])))
                rows.append((seq, nfr, v_err, g_err, gp_err, r["flip"],
                             vB, gB, bg_err, ba_err))
                continue
            print("%-20s %4d %7.3f %6d %5d %8.4f %8.4f %8.3f %8.3f %7.4f "
                  "%7.1e %4d" %
                  (seq, nfr, r["span"], r["trks"], r["flip"],
                   float(np.linalg.norm(v_i0)), v_err, g_err, gp_err,
                   r["pr_ang"], r["resid"], r["obs"]))
            rows.append((seq, nfr, v_err, g_err, gp_err, r["flip"]))

    if rows:
        ve = np.array([r[2] for r in rows])
        ge = np.array([r[3] for r in rows])
        gp = np.array([r[4] for r in rows])
        nf = sum(r[5] for r in rows)
        print("\n%d windows: v_err mean %.4f median %.4f max %.4f | "
              "g_err mean %.3f deg max %.3f | prior g_err mean %.3f deg | "
              "%d flip(s)" % (len(rows), ve.mean(), np.median(ve), ve.max(),
                              ge.mean(), ge.max(), gp.mean(), nf))
        if args.ba and len(rows[0]) > 6:
            vb = np.array([r[6] for r in rows])
            gb = np.array([r[7] for r in rows])
            be = np.array([r[8] for r in rows])
            ae = np.array([r[9] for r in rows])
            print("Stage B: v_err mean %.4f median %.4f max %.4f | "
                  "g_err mean %.3f deg max %.3f | bg_err mean %.5f max %.5f | "
                  "ba_err mean %.4f max %.4f" %
                  (vb.mean(), np.median(vb), vb.max(), gb.mean(), gb.max(),
                   be.mean(), be.max(), ae.mean(), ae.max()))
            print("Stage A -> B: velocity %+.1f%%, gravity tilt %+.1f%%" %
                  (100 * (vb.mean() / ve.mean() - 1),
                   100 * (gb.mean() / ge.mean() - 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
