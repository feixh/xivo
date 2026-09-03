#!/usr/bin/env python3
"""Is the static null's offset a real regression, or a different alignment set?

M5's nine static sequences are the null control: both arms take the static path,
so their delta is this comparison's noise floor. At n=3 stereo that delta came out
+0.0043 m with a standard error of 0.0012 -- small, but 3.6 sigma and the same
sign on 8 of 9 sequences. A systematic offset in a control that is supposed to be
centred on zero has to be explained before the two dynamic sequences are read
against it.

The mechanism to rule out first is not physics but bookkeeping. With the feature on
the filter is held back while the detector decides, so `on` never reports the first
~18 poses (m4-dispatch.md). `evaluate_ate.py` associates what it is given and then
Horn-aligns it: a shorter trajectory is aligned over a *different set of poses*, and
the alignment that minimizes RMSE over 2891 poses is not the one that minimizes it
over 2909. The two arms are therefore scored in slightly different frames, which
biases the comparison in whichever direction the dropped poses happened to pull.

The control: re-score `off` truncated to `on`'s first timestamp, so both arms are
aligned over the same poses, and see whether the offset survives. This needs no new
runs -- it is a re-scoring of trajectories already on disk.

  python3 notes-n-prompts/notes-dyninit/harness/trunc_control.py \
      ../results/dyninit/m5-n3 --on on [--mode stereo]
"""

import argparse
import os
import re
import statistics as st
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EVAL_PY = os.path.join(WORKSPACE, "xivo", "scripts", "tum_rgbd_benchmark_tools",
                       "evaluate_ate.py")
EVAL_PYTHON = os.path.join(WORKSPACE, "dependencies", "venv", "bin", "python")
DYN = {"MH_01_easy", "MH_02_easy"}


def ate(gt, est, window="0.02"):
    r = subprocess.run([EVAL_PYTHON, EVAL_PY, gt, est, "--verbose",
                        "--max_difference", window],
                       capture_output=True, text=True, timeout=600)
    rmse = n = None
    for line in r.stdout.splitlines():
        m = re.match(r"absolute_translational_error\.rmse ([0-9.eE+-]+)", line)
        if m:
            rmse = float(m.group(1))
        m = re.match(r"compared_pose_pairs (\d+)", line)
        if m:
            n = int(m.group(1))
    return rmse, n


def first_ts(path):
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                return float(line.split()[0])
    return None


def truncate(path, t_min, dst):
    """Copy `path` keeping only poses at or after t_min."""
    kept = 0
    with open(path) as f, open(dst, "w") as o:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            if float(line.split()[0]) >= t_min:
                o.write(line)
                kept += 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="an m5 output dir, holding gt/ and the arms")
    ap.add_argument("--off", default="off")
    ap.add_argument("--on", default="on")
    ap.add_argument("--modes", nargs="+", default=["stereo", "mono"])
    a = ap.parse_args()

    for mode in a.modes:
        off_d = os.path.join(a.root, a.off, mode)
        on_d = os.path.join(a.root, a.on, mode)
        if not (os.path.isdir(off_d) and os.path.isdir(on_d)):
            continue
        runs = sorted(set(os.listdir(off_d)) & set(os.listdir(on_d)))
        seqs = sorted({r.rsplit("_r", 1)[0] for r in runs})
        print(f"\n=== {mode}: off re-scored over on's pose set "
              f"({len(seqs)} sequences)")
        print(f'{"sequence":<18}{"off":>9}{"off_trunc":>11}{"on":>9}'
              f'{"raw":>9}{"trunc":>9}{"dropped":>9}  branch')
        raw_s, tr_s, raw_d, tr_d = [], [], [], []
        for seq in seqs:
            gt = os.path.join(a.root, "gt", f"{seq}.txt")
            if not os.path.exists(gt):
                gt = os.path.join(a.root, a.off, "gt", f"{seq}.txt")
            offs, trs, ons, drops = [], [], [], []
            for r in [x for x in runs if x.rsplit("_r", 1)[0] == seq]:
                fo = os.path.join(off_d, r, "traj.txt")
                fn = os.path.join(on_d, r, "traj.txt")
                if not (os.path.exists(fo) and os.path.exists(fn)):
                    continue
                t1 = first_ts(fn)
                with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                                 delete=False) as tf:
                    tmp = tf.name
                n_all = sum(1 for l in open(fo) if l.strip()
                            and not l.startswith("#"))
                n_keep = truncate(fo, t1, tmp)
                v_off, _ = ate(gt, fo)
                v_tr, _ = ate(gt, tmp)
                v_on, _ = ate(gt, fn)
                os.unlink(tmp)
                if None in (v_off, v_tr, v_on):
                    continue
                offs.append(v_off); trs.append(v_tr); ons.append(v_on)
                drops.append(n_all - n_keep)
            if not offs:
                continue
            o, t, n = st.mean(offs), st.mean(trs), st.mean(ons)
            tag = "DYNAMIC" if seq in DYN else "static"
            (raw_d if seq in DYN else raw_s).append(n - o)
            (tr_d if seq in DYN else tr_s).append(n - t)
            print(f'{seq:<18}{o:>9.4f}{t:>11.4f}{n:>9.4f}'
                  f'{n - o:>+9.4f}{n - t:>+9.4f}{st.mean(drops):>9.1f}  {tag}')
        for name, raw, tr in (("9 static (null)", raw_s, tr_s),
                              ("2 dynamic", raw_d, tr_d)):
            if not raw:
                continue
            sem = (st.stdev(raw) / len(raw) ** 0.5) if len(raw) > 1 else float("nan")
            semt = (st.stdev(tr) / len(tr) ** 0.5) if len(tr) > 1 else float("nan")
            print(f'{name:<18}{"":>9}{"":>11}{"":>9}'
                  f'{st.mean(raw):>+9.4f}{st.mean(tr):>+9.4f}'
                  f'   +-{sem:.4f} raw, +-{semt:.4f} truncated')
    print("\nraw = on - off as M5 scores it. trunc = on - off with `off` cut to"
          "\n`on`'s first timestamp, so both are Horn-aligned over the same poses."
          "\nA null offset that survives truncation is not an alignment artifact.")


if __name__ == "__main__":
    sys.exit(main())
