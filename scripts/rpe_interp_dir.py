#!/usr/bin/env python3
"""Interpolated RPE for every sequence in a run directory, plus the metric's
own artifact floor.

For each `<dataset>_<seq>_cam<id>` / `<dataset>_<seq>_gt` pair found in the
directory, reports:

  rot_i, tra_i    RPE against a ground truth SLERP-interpolated to the
                  estimate's timestamps (see evaluate_rpe_interp.py)
  rot_0, tra_0    the same metric's *stock* (nearest-neighbour) value on a
                  synthetic zero-error estimate, i.e. the ground truth
                  resampled at the estimate's timestamps. This is the floor
                  that `evaluate_rpe.py` would report for a perfect
                  trajectory, and it is not small: ~0.30 deg on TUM-VI rooms.

Usage:  rpe_interp_dir.py <run-dir> [--dataset tumvi] [--cam 0] [--no-floor]
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, 'tum_rgbd_benchmark_tools')


def slerp(a, b, u):
    a = a / numpy.linalg.norm(a)
    b = b / numpy.linalg.norm(b)
    d = float(numpy.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 1 - 1e-12:
        r = a + u * (b - a)
        return r / numpy.linalg.norm(r)
    th = numpy.arccos(min(1.0, d))
    s = numpy.sin(th)
    return numpy.sin((1 - u) * th) / s * a + numpy.sin(u * th) / s * b


def load(path):
    rows = []
    for l in open(path):
        v = [float(x) for x in l.split()] if l.strip() and not l.startswith('#') else []
        if len(v) >= 8:
            rows.append(v[:8])
    a = numpy.array(sorted(rows, key=lambda r: r[0]))
    keep = numpy.ones(len(a), dtype=bool)
    keep[1:] = numpy.diff(a[:, 0]) > 0
    return a[keep]


def resample_gt_at(gt, stamps):
    """Ground truth interpolated onto `stamps` -- a zero-error 'estimate'."""
    T = gt[:, 0]
    out = []
    for t in stamps:
        if t < T[0] or t > T[-1]:
            continue
        j = int(numpy.searchsorted(T, t))
        i = max(0, j - 1)
        j = min(len(T) - 1, max(i + 1, j))
        u = 0.0 if T[j] == T[i] else (t - T[i]) / (T[j] - T[i])
        tr = gt[i, 1:4] + u * (gt[j, 1:4] - gt[i, 1:4])
        q = slerp(gt[i, 4:8], gt[j, 4:8], u)
        out.append([t] + list(tr) + list(q))
    return out


def rpe(tool, gt, est, extra=()):
    # sys.executable, not 'python3': the evaluators need numpy, which is in the
    # venv this script is run from and not necessarily in the one on PATH.
    cmd = [sys.executable, os.path.join(TOOLS, tool), gt, est,
           '--delta', '1', '--delta_unit', 's'] + list(extra)
    if tool == 'evaluate_rpe.py':
        cmd.append('--fixed_delta')
    o = subprocess.run(cmd, capture_output=True, text=True)
    r = re.search(r'rotational_error\.rmse (\S+)', o.stdout)
    t = re.search(r'(?m)^translational_error\.rmse (\S+)', o.stdout)
    if not r or not t:
        return None
    return float(r.group(1)), float(t.group(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--dataset', default='tumvi')
    p.add_argument('--cam', default=0, type=int)
    p.add_argument('--no-floor', action='store_true',
                   help='skip the artifact-floor column (halves the runtime)')
    a = p.parse_args()

    pat = os.path.join(a.run_dir, '%s_*_cam%d' % (a.dataset, a.cam))
    seqs = []
    for f in sorted(glob.glob(pat)):
        m = re.search(r'%s_(.+)_cam%d$' % (a.dataset, a.cam), os.path.basename(f))
        if m and os.path.exists(os.path.join(
                a.run_dir, '%s_%s_gt' % (a.dataset, m.group(1)))):
            seqs.append(m.group(1))
    if not seqs:
        sys.exit('no <%s>_*_cam%d / _gt pairs in %s' % (a.dataset, a.cam, a.run_dir))

    hdr = '%-8s %-9s %-9s' % ('seq', 'rot_i', 'tra_i')
    if not a.no_floor:
        hdr += ' %-9s %-9s' % ('rot_0', 'tra_0')
    print(hdr)

    acc = {k: [] for k in ('ri', 'ti', 'r0', 't0')}
    for s in seqs:
        gt = os.path.join(a.run_dir, '%s_%s_gt' % (a.dataset, s))
        est = os.path.join(a.run_dir, '%s_%s_cam%d' % (a.dataset, s, a.cam))
        vi = rpe('evaluate_rpe_interp.py', gt, est)
        if vi is None:
            print('%-8s FAIL' % s)
            continue
        acc['ri'].append(vi[0]); acc['ti'].append(vi[1])
        line = '%-8s %-9.4f %-9.4f' % (s, vi[0], vi[1])
        if not a.no_floor:
            g = load(gt)
            rows = resample_gt_at(g, load(est)[:, 0])
            with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
                for r in rows:
                    f.write(' '.join('%.9f' % x for x in r) + '\n')
                tmp = f.name
            v0 = rpe('evaluate_rpe.py', gt, tmp)
            os.unlink(tmp)
            if v0:
                acc['r0'].append(v0[0]); acc['t0'].append(v0[1])
                line += ' %-9.4f %-9.4f' % (v0[0], v0[1])
        print(line)

    mean = lambda v: sum(v) / len(v) if v else float('nan')
    out = '\n%-8s %-9.4f %-9.4f' % ('MEAN', mean(acc['ri']), mean(acc['ti']))
    if not a.no_floor:
        out += ' %-9.4f %-9.4f' % (mean(acc['r0']), mean(acc['t0']))
    print(out)
    print('\nRPE_rot_i=%.4f RPE_tra_i=%.4f' % (mean(acc['ri']), mean(acc['ti'])))


if __name__ == '__main__':
    main()
