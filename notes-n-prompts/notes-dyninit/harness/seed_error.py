#!/usr/bin/env python3
"""What each initializer gets wrong, in the two quantities that matter.

The static path and the dynamic path do not fail in the same place, and the ATE of
a finished run cannot tell you which failure it is paying for. This scores both
against groundtruth, each at its own handoff instant:

  static path   `Estimator::InitializeGravity` averages the first
                `gravity_init_counter` accelerometer samples, calls the mean
                gravity, and asserts v = 0. Its errors are the angle between that
                mean and true up, and the true speed it is calling zero.
  dynamic path  `bin/linear_probe --ba -at_frame -1` reports the state at the
                window's **last** frame, which is where `SolveDynamic` seeds the
                filter from. Scoring frame 0 instead -- what `linear_check.py`
                does, correctly, for M2/M3 -- measures a quantity the dispatcher
                never consumes.

Why the two errors are not interchangeable. A velocity error is a state error the
filter sees and corrects from vision within a second or two. A gravity tilt error
rotates the frame the whole trajectory lives in, lands in roll and pitch where no
alignment removes it, and leaks gravity into horizontal acceleration for as long
as it survives. Trading a correctable velocity error for a larger persistent tilt
loses on ATE however good the velocity is, so both columns have to be read
together.

One wrinkle worth knowing about, and the reason for the gyro integration below:
EuRoC's `state_groundtruth_estimate0` starts ~1 s *after* the IMU, so at the
instant the static path fixes gravity there is no groundtruth at all. Its
attitude is therefore carried back from the first groundtruth sample by
integrating the gyro with the groundtruth bias removed -- 1 s of integration,
worth well under 0.1 deg, which is small against the errors being measured. The
same gap is why M3 validated the window at `-start 1.1` rather than at the
dispatcher's actual start.

Usage (from the repository root):
  python3 notes-n-prompts/notes-dyninit/harness/seed_error.py --ba
  python3 notes-n-prompts/notes-dyninit/harness/seed_error.py --ba \
      --seqs MH_01_easy MH_02_easy --start 0.005 --frames 31
  # any unrecognized flag is passed through to bin/linear_probe, e.g. -sigma_pix 2
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

G = 9.81
EUROC_SEQS = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult",
    "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]
# The two the M1 detector routes to the bundle adjustment.
DYNAMIC = {"MH_01_easy", "MH_02_easy"}


def load_csv(path):
    return np.loadtxt(path, delimiter=",", comments="#")


def quat_to_R(q):
    """EuRoC groundtruth stores (w, x, y, z), body -> world."""
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def so3_exp(w):
    t = np.linalg.norm(w)
    if t < 1e-12:
        return np.eye(3)
    k = w / t
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def angle_deg(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))


class Seq:
    def __init__(self, root, seq):
        d = os.path.join(root, seq, "mav0")
        self.imu = load_csv(os.path.join(d, "imu0", "data.csv"))
        self.gt = load_csv(os.path.join(d, "state_groundtruth_estimate0",
                                       "data.csv"))
        self.t0 = self.imu[0, 0]

    def rel(self, ts_ns):
        return (ts_ns - self.t0) / 1e9

    def gt_at(self, t_rel, tol_ms=20.0):
        """(R_wb, v_world, b_gyro, b_accel) at `t_rel` s after the first IMU
        sample. EuRoC's groundtruth carries both biases (columns 12-17), which is
        what makes the third seeded quantity checkable."""
        ts = self.t0 + t_rel * 1e9
        i = int(np.argmin(np.abs(self.gt[:, 0] - ts)))
        if abs(self.gt[i, 0] - ts) / 1e6 > tol_ms:
            return None
        return (quat_to_R(self.gt[i, 4:8]), self.gt[i, 8:11],
                self.gt[i, 11:14], self.gt[i, 14:17])

    def R_before_gt(self, t_rel):
        """Attitude at `t_rel`, which precedes the groundtruth, by integrating the
        gyro backwards from the first groundtruth sample with its bias removed."""
        R = quat_to_R(self.gt[0, 4:8])
        bw = self.gt[0, 11:14]
        ts = self.t0 + t_rel * 1e9
        m = (self.imu[:, 0] >= ts) & (self.imu[:, 0] <= self.gt[0, 0])
        rows = self.imu[m]
        # R(t_k) = R(t_{k+1}) Exp(-w_k dt): walk the samples in reverse.
        for k in range(len(rows) - 1, 0, -1):
            dt = (rows[k, 0] - rows[k - 1, 0]) / 1e9
            R = R @ so3_exp(-(rows[k - 1, 1:4] - bw) * dt)
        return R

    def static_seed(self, counter, t_start=0.0):
        """(tilt_deg, |mean_accel|, t_rel) for the accelerometer-average path.

        `t_start` mirrors `pyxivo.py -start_sec`: the estimator averages the first
        `counter` samples it is *given*, so a mid-flight start averages a moving
        rig and this has to average the same samples to describe it.
        """
        first = int(np.searchsorted(self.imu[:, 0], self.t0 + t_start * 1e9))
        head = self.imu[first:first + counter]
        mean_accel = head[:, 4:7].mean(axis=0)
        t = self.rel(head[-1, 0])
        g = self.gt_at(t)
        R = g[0] if g is not None else self.R_before_gt(t)
        # At rest the accelerometer reads specific force, i.e. R' [0,0,+G] -- up in
        # the body frame. Linear acceleration in the window contaminates both its
        # direction and its magnitude, and both are reported.
        return (angle_deg(mean_accel, R.T @ np.array([0.0, 0.0, G])),
                np.linalg.norm(mean_accel), t)


BA_RE = re.compile(r"^(\S+)\s+((?:[-+0-9.eE]+\s+){12,})")


def run_probe(xivo, root, cfg, seq, start, frames, extra):
    """Parse `bin/linear_probe -ba -at_frame -1` into (v_body, g_body, diag)."""
    cmd = [os.path.join(xivo, "bin", "linear_probe"), "-cfg", cfg, "-root", root,
           "-dataset", "euroc", "-seq", seq, "-start", str(start),
           "-frames", str(frames), "-ba", "-at_frame", "-1"] + list(extra)
    r = subprocess.run(cmd, cwd=xivo, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        return None, f"rc={r.returncode} {r.stderr.strip()[-160:]}"
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    at = next((l for l in lines if l.startswith("#at_frame")), "")
    body = [l for l in lines if not l.startswith("#")]
    if len(body) < 2:
        return None, "no BA row: " + " | ".join(lines)[:160]
    f = body[-1].split()[1:]
    v = np.array([float(x) for x in f[0:3]])
    g = np.array([float(x) for x in f[3:6]])
    diag = dict(bg=np.array([float(x) for x in f[6:9]]),
                ba=np.array([float(x) for x in f[9:12]]),
                pix=float(f[12]), pmed=float(f[13]), imu=float(f[14]),
                it=int(f[15]), rj=int(f[16]), ok=int(f[17]), at=at)
    m = re.search(r"t=([-+0-9.eE]+) s", at)
    diag["t"] = float(m.group(1)) if m else float("nan")
    return (v, g, diag), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/euroc")
    ap.add_argument("--cfg", default="cfg/euroc_stereo.json")
    ap.add_argument("--seqs", nargs="+", default=EUROC_SEQS)
    ap.add_argument("--counter", type=int, default=20,
                    help="gravity_init_counter (default: the shipped 20)")
    ap.add_argument("--ba", action="store_true",
                    help="also score the dynamic path via bin/linear_probe")
    ap.add_argument("--start", type=float, default=None,
                    help="window start, s after the first IMU sample; default is "
                         "the dispatcher's own, i.e. the first image. Both paths "
                         "follow it, so `--start 55` describes what each "
                         "initializer would do in a run launched with "
                         "`pyxivo.py -start_sec 55`.")
    ap.add_argument("--frames", type=int, default=31)
    args, extra = ap.parse_known_args()
    xivo = os.getcwd()

    print(f"static path: mean of the first {args.counter} accel samples "
          f"({args.counter / 200.0:.2f} s at 200 Hz), v := 0"
          + (f", from t={args.start:g} s" if args.start else ""))
    if args.ba:
        print(f"dynamic path: {args.frames}-frame window, state at its LAST "
              f"frame (the handoff)")
    print()
    hdr = f'{"sequence":<18}{"branch":<9}{"stat tilt":>10}{"stat |a|":>9}'
    if args.ba:
        hdr += f'{"dyn tilt":>10}{"dyn |dv|":>9}{"true |v|":>9}' \
               f'{"dyn |dba|":>10}{"stat|dba|":>10}{"dyn |dbg|":>10}' \
               f'{"stat|dbg|":>10}{"pmed":>7}{"imu":>7}{"rj":>4}{"t_ho":>7}'
    print(hdr)

    for seq in args.seqs:
        s = Seq(args.root, seq)
        tilt, amag, _ = s.static_seed(args.counter, args.start or 0.0)
        branch = "DYNAMIC" if seq in DYNAMIC else "static"
        line = f"{seq:<18}{branch:<9}{tilt:>10.3f}{amag:>9.4f}"
        if args.ba:
            start = args.start
            if start is None:
                # cam0/data.csv is (timestamp, filename), so read column 0 only.
                cam = np.loadtxt(os.path.join(args.root, seq, "mav0", "cam0",
                                              "data.csv"), delimiter=",",
                                 comments="#", usecols=0)
                start = s.rel(cam[0])
            got, err = run_probe(xivo, args.root, args.cfg, seq, start,
                                 args.frames, extra)
            if err:
                line += f"  probe: {err}"
            else:
                v, g, d = got
                gt = s.gt_at(d["t"])
                if gt is None:
                    line += f'{"":>10}{"":>9}{"":>9}{"":>10}{"":>10}{"":>10}' \
                            f'{"":>10}' \
                            f'{d["pmed"]:>7.3f}{d["imu"]:>7.3f}{d["rj"]:>4d}' \
                            f'{d["t"]:>7.2f}  (no gt at handoff)'
                else:
                    R, v_w, bg_t, ba_t = gt
                    dtilt = angle_deg(g, R.T @ np.array([0.0, 0.0, -G]))
                    dv = np.linalg.norm(v - R.T @ v_w)
                    # The static path seeds both biases at zero, so its error is
                    # the true bias itself -- which is the number the dynamic
                    # path's estimate has to beat to be worth seeding at all.
                    line += (f'{dtilt:>10.3f}{dv:>9.4f}'
                             f'{np.linalg.norm(v_w):>9.4f}'
                             f'{np.linalg.norm(d["ba"] - ba_t):>10.4f}'
                             f'{np.linalg.norm(ba_t):>10.4f}'
                             f'{np.linalg.norm(d["bg"] - bg_t):>10.5f}'
                             f'{np.linalg.norm(bg_t):>10.5f}'
                             f'{d["pmed"]:>7.3f}{d["imu"]:>7.3f}'
                             f'{d["rj"]:>4d}{d["t"]:>7.2f}')
        print(line)
    print("\nstat tilt / dyn tilt: gravity direction error, deg. dyn |dv|: velocity"
          "\nerror at the handoff, m/s -- against which the static path's error is"
          "\nthe true speed it calls zero (true |v|, reported at the handoff too)."
          "\n|dba| / |dbg|: bias error against groundtruth, m/s^2 and rad/s. The"
          "\nstatic path seeds both at zero, so `stat|dba|` is just the true bias:"
          "\nwherever `dyn` exceeds `stat`, seeding the solved bias is worse than"
          "\nseeding nothing.")


if __name__ == "__main__":
    sys.exit(main())
