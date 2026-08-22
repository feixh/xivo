#!/usr/bin/env python3
"""Relative pose error against a ground truth interpolated to the estimate's
timestamps.

WHY
---
`evaluate_rpe.py` pairs each estimate timestamp with the *nearest* ground-truth
sample (`find_closest_index`), with no interpolation. TUM-VI ground truth is
logged at ~120 Hz, so every endpoint of every evaluated interval carries up to
+/-4.17 ms of timestamp quantization, and the two endpoints of an interval are
quantized independently.

RPE over a 1 s window is extremely sensitive to that. Measured on XIVO's
room1-room6 output: applying a *constant* shift to the estimate timestamps
changes mean RPE_rot by about 0.11 deg per ms. And decimating the ground truth,
which multiplies the quantization while leaving the estimate untouched, inflates
the reported error monotonically and steeply:

    GT spacing   8.33ms  16.67  25.00  33.33  50.00   (ms)
    RPE_rot      0.6205  0.8614 0.9951 1.2735 1.9733  (deg)

An estimator error cannot depend on how finely the reference was sampled, so
the trend is measurement artifact, and extrapolating it to zero spacing says a
large fraction of the reported 0.62 deg is quantization rather than estimator
error.

This script removes that term by interpolating the ground truth to each
estimate timestamp -- SLERP for rotation, linear for translation -- instead of
snapping to the nearest sample. Everything else (the pairing rule, `ominus`,
the reported statistics) is identical to `evaluate_rpe.py`, whose helpers are
imported rather than reimplemented.

IMPORTANT: this changes the *metric*, not the estimator. A lower number here is
a more accurate measurement of the same trajectory, not a better trajectory.
Always report it alongside the stock `evaluate_rpe.py` number.
"""

import argparse
import os
import random
import sys

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_rpe import (compute_angle, compute_distance, ominus,
                          percentile, transform44)


def read_trajectory_raw(filename):
    """(N,8) array of [t tx ty tz qx qy qz qw], sorted by t, duplicates dropped."""
    rows = []
    for line in open(filename):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        v = [float(x) for x in line.split()]
        if len(v) < 8:
            continue
        rows.append(v[:8])
    if not rows:
        raise Exception('no poses in %s' % filename)
    a = numpy.array(sorted(rows, key=lambda r: r[0]))
    keep = numpy.ones(len(a), dtype=bool)
    keep[1:] = numpy.diff(a[:, 0]) > 0
    return a[keep]


def slerp(q0, q1, alpha):
    """Shortest-arc SLERP between two unit quaternions in [x, y, z, w] order."""
    q0 = q0 / numpy.linalg.norm(q0)
    q1 = q1 / numpy.linalg.norm(q1)
    dot = float(numpy.dot(q0, q1))
    if dot < 0.0:  # antipodal: same rotation, take the short way round
        q1 = -q1
        dot = -dot
    if dot > 1.0 - 1e-12:  # ~parallel: lerp is exact to machine precision here
        q = q0 + alpha * (q1 - q0)
        return q / numpy.linalg.norm(q)
    theta = numpy.arccos(min(1.0, dot))
    s = numpy.sin(theta)
    return (numpy.sin((1.0 - alpha) * theta) / s) * q0 + \
           (numpy.sin(alpha * theta) / s) * q1


def interpolate_gt(gt, t, max_gap):
    """Pose of `gt` at time `t` as a 4x4, or None if `t` is not bracketed by
    two samples less than `max_gap` apart."""
    ts = gt[:, 0]
    if t < ts[0] or t > ts[-1]:
        return None
    j = int(numpy.searchsorted(ts, t))
    if j == 0:
        i, j = 0, 1
    else:
        i = j - 1
        if j >= len(ts):
            i, j = len(ts) - 2, len(ts) - 1
    if ts[j] - ts[i] > max_gap:
        return None
    alpha = 0.0 if ts[j] == ts[i] else (t - ts[i]) / (ts[j] - ts[i])
    trans = gt[i, 1:4] + alpha * (gt[j, 1:4] - gt[i, 1:4])
    quat = slerp(gt[i, 4:8], gt[j, 4:8], alpha)
    return transform44([t] + list(trans) + list(quat))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('groundtruth_file')
    p.add_argument('estimated_file')
    p.add_argument('--delta', type=float, default=1.0)
    p.add_argument('--delta_unit', default='s', choices=['s'],
                   help="only 's' is supported; the other units index by "
                        "estimate order and are unaffected by GT sampling")
    p.add_argument('--max_pairs', type=int, default=10000)
    p.add_argument('--offset', type=float, default=0.0,
                   help='constant time offset added to estimate timestamps')
    p.add_argument('--max_gap_factor', type=float, default=4.0,
                   help='reject an interpolation spanning a GT gap wider than '
                        'this many median sample intervals')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    gt = read_trajectory_raw(args.groundtruth_file)
    est = read_trajectory_raw(args.estimated_file)
    max_gap = args.max_gap_factor * float(numpy.median(numpy.diff(gt[:, 0])))

    t_est = est[:, 0] + args.offset
    # Pairs delta apart in estimate time -- same rule as evaluate_rpe.py's
    # fixed_delta branch.
    pairs = []
    for i in range(len(t_est)):
        j = int(numpy.searchsorted(t_est, t_est[i] + args.delta))
        if j >= len(t_est):
            continue
        if j > 0 and abs(t_est[j - 1] - (t_est[i] + args.delta)) < \
                     abs(t_est[j] - (t_est[i] + args.delta)):
            j -= 1
        if j != i and j != len(t_est) - 1:
            pairs.append((i, j))
    if args.max_pairs and len(pairs) > args.max_pairs:
        pairs = random.sample(pairs, args.max_pairs)

    result = []
    for i, j in pairs:
        g0 = interpolate_gt(gt, t_est[i], max_gap)
        g1 = interpolate_gt(gt, t_est[j], max_gap)
        if g0 is None or g1 is None:
            continue
        e0 = transform44(list(est[i]))
        e1 = transform44(list(est[j]))
        err = ominus(ominus(e1, e0), ominus(g1, g0))
        result.append([t_est[i], t_est[j], compute_distance(err),
                       compute_angle(err)])

    if len(result) < 2:
        raise Exception('too few evaluable pairs; check the input files')

    trans = numpy.array([r[2] for r in result])
    rot = numpy.array([r[3] for r in result])

    if args.verbose:
        print('compared_pose_pairs %d pairs' % len(result))
    print('translational_error.rmse %f m' % numpy.sqrt(numpy.dot(trans, trans) / len(trans)))
    print('translational_error.mean %f m' % numpy.mean(trans))
    print('translational_error.median %f m' % numpy.median(trans))
    print('translational_error.std %f m' % numpy.std(trans))
    print('rotational_error.rmse %f deg' % (numpy.sqrt(numpy.dot(rot, rot) / len(rot)) * 180.0 / numpy.pi))
    print('rotational_error.mean %f deg' % (numpy.mean(rot) * 180.0 / numpy.pi))
    print('rotational_error.median %f deg' % (numpy.median(rot) * 180.0 / numpy.pi))
    print('rotational_error.std %f deg' % (numpy.std(rot) * 180.0 / numpy.pi))


if __name__ == '__main__':
    main()
