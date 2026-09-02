#!/usr/bin/env python3
"""What does XIVO actually believe at the instant it finishes initializing?

XIVO's initializer (`Estimator::InitializeGravity`, src/estimator.cpp) averages
the first `gravity_init_counter` accelerometer samples that pass the
`gravity_init_max_accel_dev` gate, uses the average as gravity, and leaves the
initial velocity and both bias states at their configured values -- zero. That is
correct only if the rig is at rest.

This script replays that gate sample by sample, exactly as
`InertialMeasInternal` does, finds the instant it fires, and then asks the ground
truth what was actually true at that instant. Two numbers matter:

  * `|v| @ init`  -- the velocity XIVO asserts is zero.
  * `tilt @ init` -- the angle between the gravity direction XIVO derives from
                     its accelerometer average and the true gravity direction
                     from the ground-truth attitude. This lands in the
                     orientation error undiminished; roll and pitch are not
                     quantities an evaluator can align away.

It also reports the statistics a *detector* would have to work with, so that the
design of one can be argued from data:

  * `a_sd(min)`   -- the smallest accelerometer sample standard deviation over
                     any candidate window in the search horizon. Instantaneous
                     variance does not separate moving from static on EuRoC
                     (V1_01 and V2_03 are static and noisier than MH_02 is while
                     moving); the minimum over windows does, barely.
  * `w_mag`       -- mean gyro magnitude, i.e. how much rotation the "stationary"
                     assumption is already absorbing.

Usage (from the repository root):
  python3 notes-n-prompts/notes-dyninit/harness/init_diag.py \
      --dataset euroc --root ../data/euroc --cfg cfg/euroc_stereo.json
  python3 notes-n-prompts/notes-dyninit/harness/init_diag.py \
      --dataset tumvi --root ../data/tumvi --cfg cfg/tumvi_stereo.json
"""

import argparse
import json
import os
import re
import sys

import numpy as np

EUROC_SEQS = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
    "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]

# The six mocap-room TUM-VI sequences, the ones whose ground truth covers the
# whole trajectory and which every previous round was scored on.
TUMVI_SEQS = ["dataset-room%d_512_16" % k for k in range(1, 7)]


def load_cfg(path):
    """XIVO's configs carry // comments, which json refuses."""
    with open(path) as f:
        text = re.sub(r"//[^\n]*", "", f.read())
    return json.loads(text)


def load_imu(seq_dir):
    """-> (t seconds since first sample, gyro Nx3, accel Nx3)."""
    d = np.loadtxt(os.path.join(seq_dir, "mav0/imu0/data.csv"),
                   delimiter=",", comments="#")
    t = (d[:, 0] - d[0, 0]) * 1e-9
    return t, d[:, 1:4], d[:, 4:7]


def load_gt(seq_dir, dataset, t0_ns):
    """Ground truth resampled onto seconds-since-IMU-t0.

    EuRoC's `state_groundtruth_estimate0` has position, quaternion (w first),
    velocity and both solved-for biases. TUM-VI's `mocap0` has position and
    quaternion only -- no velocity -- so velocity is differenced from position.

    Returns (t, quat_wxyz Nx4, vel Nx3) or None if there is no ground truth.
    """
    if dataset == "euroc":
        p = os.path.join(seq_dir, "mav0/state_groundtruth_estimate0/data.csv")
        if not os.path.exists(p):
            return None
        d = np.loadtxt(p, delimiter=",", comments="#")
        t = (d[:, 0] - t0_ns) * 1e-9
        return t, d[:, 4:8], d[:, 8:11]

    p = os.path.join(seq_dir, "mav0/mocap0/data.csv")
    if not os.path.exists(p):
        return None
    d = np.loadtxt(p, delimiter=",", comments="#")
    t = (d[:, 0] - t0_ns) * 1e-9
    # Central differences on position. Mocap at 120 Hz is quiet enough that this
    # is a usable velocity for the purpose here -- deciding whether |v| is 0.03
    # or 0.7 m/s -- even though it would not be for scoring.
    pos = d[:, 1:4]
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (t[2:] - t[:-2])[:, None]
    vel[0], vel[-1] = vel[1], vel[-2]
    return t, d[:, 4:8], vel


def quat_to_R(q):
    """q = (w, x, y, z) -> R_world_from_body."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def replay_gate(t, gyro, accel, counter, max_dev, max_skip, derotate, g_norm):
    """Replay `InertialMeasInternal`'s gravity-init path sample by sample.

    Mirrors src/estimator.cpp: the gyro is integrated across *every* sample,
    accepted or not (`gravity_init_R_run_`); a sample is rejected when
    `| |a| - |g| | > max_dev`, up to `max_skip` rejections, after which the gate
    gives up and accepts everything; init fires when the buffer reaches
    `counter` samples.

    Returns dict(idx, t, skipped, seen, mean_accel) or None if it never fires.
    """
    buf, Rbuf = [], []
    R_run = np.eye(3)
    skipped, seen = 0, 0
    last_gyro, last_t = None, None

    for i in range(len(t)):
        if seen > 0:
            dt = max(0.0, t[i] - last_t)
            R_run = R_run @ so3_exp(0.5 * (gyro[i] + last_gyro) * dt)
        last_gyro, last_t = gyro[i], t[i]
        seen += 1

        if (max_dev > 0 and skipped < max_skip
                and abs(np.linalg.norm(accel[i]) - g_norm) > max_dev):
            skipped += 1
            continue

        buf.append(accel[i])
        Rbuf.append(R_run.copy())

        if len(buf) >= counter:
            if derotate:
                R_N0 = Rbuf[-1].T
                mean_accel = np.mean([R_N0 @ Rbuf[k] @ buf[k]
                                      for k in range(len(buf))], axis=0)
            else:
                mean_accel = np.mean(buf, axis=0)
            return dict(idx=i, t=t[i], skipped=skipped, seen=seen,
                        mean_accel=mean_accel)
    return None


def min_window_sd(t, accel, win_s, horizon_s):
    """Smallest accelerometer sample sd over any window of `win_s` starting
    within `horizon_s`. This is the statistic a detector can actually use; the sd
    at one fixed instant is not (see the module docstring)."""
    best = np.inf
    i = 0
    while i < len(t) and t[i] <= horizon_s:
        j = np.searchsorted(t, t[i] + win_s)
        if j - i >= 10:
            seg = accel[i:j]
            # Sample sd of the vector, pooled over axes -- the same statistic
            # OpenVINS' StaticInitializer compares against init_imu_thresh.
            sd = np.sqrt(np.mean(np.sum((seg - seg.mean(axis=0)) ** 2, axis=1)))
            best = min(best, sd)
        i += max(1, (j - i) // 4)  # slide by a quarter window
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["euroc", "tumvi"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--win", type=float, default=0.5,
                    help="detector window length, seconds")
    ap.add_argument("--horizon", type=float, default=5.0,
                    help="how far into the sequence a detector may look")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    counter = int(cfg.get("gravity_init_counter", 20))
    max_dev = float(cfg.get("gravity_init_max_accel_dev", 0.0))
    max_skip = int(cfg.get("gravity_init_max_skip", 2000))
    derotate = bool(cfg.get("gravity_init_derotate", False))
    g = np.array(cfg["gravity"], dtype=float)
    g_norm = np.linalg.norm(g)

    seqs = EUROC_SEQS if args.dataset == "euroc" else TUMVI_SEQS

    print("# %s, cfg %s" % (args.dataset, args.cfg))
    print("# gravity_init: counter=%d max_accel_dev=%g max_skip=%d derotate=%s "
          "|g|=%.4f" % (counter, max_dev, max_skip, derotate, g_norm))
    print("# detector window %.2fs, horizon %.1fs" % (args.win, args.horizon))
    print()
    hdr = ("%-26s %8s %7s %7s %9s %9s %8s %8s %s"
           % ("sequence", "t_init", "skipped", "seen", "|v|@init",
              "tilt_deg", "a_sd_min", "w_mag", "verdict"))
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for s in seqs:
        seq_dir = os.path.join(args.root, s)
        if not os.path.isdir(seq_dir):
            print("%-26s  MISSING" % s)
            continue
        t, gyro, accel = load_imu(seq_dir)
        t0_ns = np.loadtxt(os.path.join(seq_dir, "mav0/imu0/data.csv"),
                           delimiter=",", comments="#", max_rows=1)[0]

        res = replay_gate(t, gyro, accel, counter, max_dev, max_skip,
                          derotate, g_norm)
        if res is None:
            print("%-26s  NEVER INITIALIZES" % s)
            continue

        sd_min = min_window_sd(t, accel, args.win, args.horizon)
        w_mag = float(np.mean(np.linalg.norm(gyro[:res["idx"] + 1], axis=1)))

        gt = load_gt(seq_dir, args.dataset, t0_ns)
        vmag, tilt = float("nan"), float("nan")
        if gt is not None:
            tg, quat, vel = gt
            k = int(np.argmin(np.abs(tg - res["t"])))
            # Guard against extrapolating: EuRoC ground truth starts 0.9-2.4 s
            # after IMU t0, so for the early-firing sequences the nearest GT
            # sample is the *first* one, which is the honest thing to report.
            vmag = float(np.linalg.norm(vel[k]))
            R_wb = quat_to_R(quat[k])
            up_true = R_wb.T @ np.array([0.0, 0.0, 1.0])
            up_xivo = res["mean_accel"] / np.linalg.norm(res["mean_accel"])
            c = float(np.clip(up_true @ up_xivo, -1.0, 1.0))
            tilt = np.degrees(np.arccos(c))

        verdict = "MOVING" if vmag > 0.15 else "static"
        rows.append((s, res, vmag, tilt, sd_min, w_mag, verdict))
        print("%-26s %8.3f %7d %7d %9.3f %9.3f %8.3f %8.4f %s"
              % (s, res["t"], res["skipped"], res["seen"], vmag, tilt,
                 sd_min, w_mag, verdict))

    moving = [r for r in rows if r[6] == "MOVING"]
    print()
    print("# %d of %d sequences are moving at the instant XIVO initializes"
          % (len(moving), len(rows)))
    if moving:
        print("#   " + ", ".join("%s (|v|=%.2f m/s, tilt=%.2f deg)"
                                 % (r[0], r[2], r[3]) for r in moving))
        print("# separation on a_sd_min: moving min %.3f, static max %.3f"
              % (min(r[4] for r in moving),
                 max([r[4] for r in rows if r[6] == "static"] or [0.0])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
