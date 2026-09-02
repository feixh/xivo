#!/usr/bin/env python3
"""Score a run directory produced by run_openvins.sh.

Writes <dir>/summary.csv (one row per run) and <dir>/summary.md (tables), and
prints the tables. Safe to re-run: it only reads traj/timing/stats files.

Two ATE scorers are used, on purpose:

* evaluate_ate.py  (TUM RGB-D benchmark tool, as used by XIVO in this workspace)
  Horn SE(3) alignment over associated pose pairs, RMSE of translation.
  Reported at two association windows: 0.02 s (covers ~all frames) and 0.001 s
  (what xivo/scripts/run_and_eval_pyxivo.py uses -- kept only so the numbers are
  comparable with RESULTS.md, since it silently scores ~26% of frames).
* ov_eval error_singlerun posyaw  (OpenVINS' own scorer)
  yaw+position alignment, linear interpolation onto groundtruth times, and it
  also gives orientation ATE and relative pose error.

Env overrides: EVAL_ATE_PY, EVAL_PYTHON, OV_EVAL_BIN.
"""

import csv
import os
import re
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))

EVAL_ATE_PY = os.environ.get(
    "EVAL_ATE_PY", os.path.join(WORKSPACE, "xivo", "scripts", "tum_rgbd_benchmark_tools", "evaluate_ate.py")
)
EVAL_PYTHON = os.environ.get("EVAL_PYTHON", os.path.join(WORKSPACE, "dependencies", "venv", "bin", "python"))
OV_EVAL_BIN = os.environ.get("OV_EVAL_BIN", os.path.join(WORKSPACE, "experiments", "ov_build_eval", "error_singlerun"))

ATE_WINDOWS = [("ate_002", "0.02"), ("ate_0001", "0.001")]


def read_kv(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                out[k] = v
    return out


def run_evaluate_ate(gt, est, max_difference):
    """-> (rmse, n_pairs) or (None, None)"""
    if not (os.path.exists(EVAL_ATE_PY) and os.path.exists(EVAL_PYTHON)):
        return None, None
    try:
        res = subprocess.run(
            [EVAL_PYTHON, EVAL_ATE_PY, gt, est, "--verbose", "--max_difference", max_difference],
            capture_output=True, text=True, timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    rmse = n = None
    for line in res.stdout.splitlines():
        m = re.match(r"absolute_translational_error\.rmse ([0-9.eE+-]+)", line)
        if m:
            rmse = float(m.group(1))
        m = re.match(r"compared_pose_pairs (\d+)", line)
        if m:
            n = int(m.group(1))
    return rmse, n


def run_ov_eval(gt, est, align="posyaw"):
    """-> dict with ov_ate_pos / ov_ate_ori / ov_rpe8_pos / ov_rpe8_ori"""
    out = {}
    if not os.path.exists(OV_EVAL_BIN):
        return out
    try:
        res = subprocess.run([OV_EVAL_BIN, align, gt, est], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return out
    text = res.stdout
    m = re.search(r"Absolute Trajectory Error\s*=+\s*rmse_ori = ([0-9.eE+-]+) \| rmse_pos = ([0-9.eE+-]+)", text)
    if m:
        out["ov_ate_ori_deg"] = float(m.group(1))
        out["ov_ate_pos_m"] = float(m.group(2))
    m = re.search(r"seg 8 - median_ori = ([0-9.eE+-]+) \| median_pos = ([0-9.eE+-]+) \((\d+) samples\)", text)
    if m and int(m.group(3)) > 0:
        out["ov_rpe8_ori_deg"] = float(m.group(1))
        out["ov_rpe8_pos_m"] = float(m.group(2))
    return out


def traj_length(path):
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                n += 1
    return n


FIELDS = [
    "mode", "seq", "repeat", "ate_002", "ate_002_pairs", "ate_0001", "ate_0001_pairs",
    "ov_ate_pos_m", "ov_ate_ori_deg", "ov_rpe8_pos_m", "ov_rpe8_ori_deg",
    "update_mean_ms", "update_median_ms", "update_p95_ms", "update_max_ms",
    "fps_mean", "fps_median", "fps_wall", "realtime_factor", "peak_rss_mb", "init_time_s",
    "frames_processed", "traj_poses", "wall_total_s", "wall_imread_s", "num_opencv_threads",
]


def collect(outdir):
    rows = []
    for mode in sorted(os.listdir(outdir)):
        mode_dir = os.path.join(outdir, mode)
        if mode not in ("mono", "stereo") or not os.path.isdir(mode_dir):
            continue
        for run in sorted(os.listdir(mode_dir)):
            rundir = os.path.join(mode_dir, run)
            traj = os.path.join(rundir, "traj.txt")
            stats_path = os.path.join(rundir, "stats.txt")
            if not os.path.isdir(rundir) or not os.path.exists(stats_path):
                print("  (skipping %s/%s -- no stats.txt)" % (mode, run))
                continue
            # A throughput pass (run_xivo_reference.sh --timing uses XIVO's
            # `-mode runOnly`) writes no trajectory. Still emit the row: the FPS
            # and RSS columns are the whole point of that pass.
            have_traj = os.path.exists(traj)
            m = re.match(r"(.+)_r(\d+)$", run)
            seq, rep = (m.group(1), int(m.group(2))) if m else (run, 0)
            gt = os.path.join(outdir, "gt", seq + ".txt")
            row = {"mode": mode, "seq": seq, "repeat": rep}
            if have_traj:
                for key, window in ATE_WINDOWS:
                    rmse, pairs = run_evaluate_ate(gt, traj, window)
                    row[key] = rmse
                    row[key + "_pairs"] = pairs
                row.update(run_ov_eval(gt, traj))
            stats = read_kv(stats_path)
            for k in ("update_mean_ms", "update_median_ms", "update_p95_ms", "update_max_ms", "fps_mean",
                      "fps_median", "realtime_factor", "peak_rss_mb", "init_time_s", "frames_processed",
                      "wall_total_s", "wall_imread_s", "num_opencv_threads"):
                if k in stats:
                    row[k] = float(stats[k]) if "." in stats[k] else int(stats[k])
            # End-to-end throughput: whole replay loop (image decode + IMU feed +
            # tracking + update) over frames. This is what XIVO's fps_one.sh
            # measures, so it is the number to compare across systems; fps_mean
            # is estimator-only and always higher.
            if row.get("wall_total_s") and row.get("frames_processed"):
                row["fps_wall"] = row["frames_processed"] / row["wall_total_s"]
            if have_traj:
                row["traj_poses"] = traj_length(traj)
            rows.append(row)
            if have_traj:
                print("  scored %s/%s: ATE(0.02)=%s ATE(ov)=%s" % (mode, run, row.get("ate_002"), row.get("ov_ate_pos_m")))
            else:
                print("  timing only %s/%s (no traj.txt): fps_wall=%s" % (mode, run, fmt(row.get("fps_wall"), 1)))
    return rows


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%." + str(nd) + "f") % v
    return str(v)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def table(rows, modes, seqs, metric, nd=3, per_repeat_mean=True):
    """Rows = seq columns, one line per mode; mean over repeats within a cell."""
    lines = []
    head = "| mode | " + " | ".join(seqs) + " | mean |"
    lines.append(head)
    lines.append("|" + "---|" * (len(seqs) + 2))
    for mode in modes:
        cells, cellmeans = [], []
        for seq in seqs:
            vals = [r.get(metric) for r in rows if r["mode"] == mode and r["seq"] == seq]
            mv = mean(vals)
            cellmeans.append(mv)
            cells.append(fmt(mv, nd))
        lines.append("| %s | %s | %s |" % (mode, " | ".join(cells), fmt(mean(cellmeans), nd)))
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("usage: score_openvins.py <run-dir>")
        return 1
    outdir = os.path.abspath(sys.argv[1])
    print("scoring %s" % outdir)
    rows = collect(outdir)
    if not rows:
        print("no runs found")
        return 1

    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["mode"], r["seq"], r["repeat"])):
            w.writerow(r)

    modes = [m for m in ("stereo", "mono") if any(r["mode"] == m for r in rows)]
    seqs = sorted({r["seq"] for r in rows})
    nrep = max(r["repeat"] for r in rows) + 1

    md = ["# OpenVINS run summary", "", "`%s`" % outdir, ""]
    info = read_kv(os.path.join(outdir, "run_info.txt"))
    if info:
        md += ["```"] + ["%s = %s" % (k, v) for k, v in info.items()] + ["```", ""]
    if nrep > 1:
        md += ["Cells are the mean over %d repeats." % nrep, ""]
    for title, metric, nd in [
        ("ATE RMSE [m] -- evaluate_ate.py, 0.02 s association window", "ate_002", 4),
        ("ATE RMSE [m] -- evaluate_ate.py, 0.001 s window (comparability with RESULTS.md only)", "ate_0001", 4),
        ("ATE RMSE [m] -- ov_eval posyaw", "ov_ate_pos_m", 4),
        ("ATE RMSE [deg] -- ov_eval posyaw, orientation", "ov_ate_ori_deg", 3),
        ("RPE 8 m -- ov_eval, median translation [m]", "ov_rpe8_pos_m", 4),
        ("RPE 8 m -- ov_eval, median rotation [deg]", "ov_rpe8_ori_deg", 3),
        ("Per-frame track+update time, mean [ms]", "update_mean_ms", 2),
        ("Per-frame track+update time, median [ms]", "update_median_ms", 2),
        ("Per-frame track+update time, p95 [ms]", "update_p95_ms", 2),
        ("Throughput [FPS], estimator only = 1 / mean per-frame track+update time", "fps_mean", 1),
        ("Throughput [FPS], end-to-end = frames / wall clock (incl. image decode)", "fps_wall", 1),
        ("Realtime factor = data seconds per wall second", "realtime_factor", 2),
        ("Peak RSS [MB]", "peak_rss_mb", 1),
        ("Initialization delay [s]", "init_time_s", 2),
    ]:
        if not any(r.get(metric) is not None for r in rows):
            continue
        md += ["## " + title, "", table(rows, modes, seqs, metric, nd), ""]

    text = "\n".join(md)
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write(text + "\n")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
