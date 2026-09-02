#!/usr/bin/env python3
"""Aggregate an ensemble of runs into per-sequence mean +/- sd, and compare arms.

An ensemble is a set of directories that each hold one `summary.csv` written by
`run_openvins.sh` or `run_xivo_euroc.sh` -- one directory per ensemble member.
Members differ only in a neutral perturbation (XIVO `--jitter`, OpenVINS
`--gravity_mag` in its 9th significant digit), so the spread across members is
the run-to-run noise of the system, not a property of any one member.

Why this exists: a single VIO run's ATE is not a measurement. XIVO's gating is
chaotic enough that repeats of the same sequence spread by ~0.007 m, which is
larger than most of the differences worth arguing about, and OpenVINS repeats are
byte-identical so its noise has to be provoked deliberately. Every number in the
EuRoC report is therefore a mean over an ensemble with its sd attached.

Usage:
  agg_ensemble.py --arm openvins ../results/euroc_ov_acc/stereo_m*
  agg_ensemble.py --arm openvins ../results/euroc_ov_acc/stereo_m* \
                  --arm xivo ../results/euroc_xivo_acc/stereo_m*
"""
import argparse
import csv
import math
import os
import sys

# The metrics reported, in the order they appear in the tables. `ate_002` is
# evaluate_ate.py with a 20 ms association window; the four `ov_*` columns come
# from `ov_eval error_singlerun posyaw`, which unlike ate_002 charges roll and
# pitch error in full.
METRICS = [
    ('ate_002', 'ATE pos (m)', 3),
    ('ov_ate_pos_m', 'ov ATE pos (m)', 3),
    ('ov_ate_ori_deg', 'ov ATE ori (deg)', 2),
    ('ov_rpe8_pos_m', 'RPE8 pos (m)', 3),
    ('ov_rpe8_ori_deg', 'RPE8 ori (deg)', 2),
    ('fps_wall', 'FPS (wall)', 1),
    ('peak_rss_mb', 'peak RSS (MB)', 1),
]

# Above this the run did not track the sequence at all; averaging it in would
# turn one divergence into a meaningless mean. Counted and reported separately.
DIVERGED_M = 100.0


def load_arm(dirs, mode=None):
    """{seq: {metric: [values...]}} plus a per-sequence divergence count."""
    per_seq = {}
    diverged = {}
    order = []
    for d in dirs:
        path = os.path.join(d, 'summary.csv')
        if not os.path.exists(path):
            print(f'  (skipping {d}: no summary.csv)', file=sys.stderr)
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                # One summary.csv can hold several sensor modes (the XIVO runner
                # writes mono and stereo side by side), and averaging across
                # modes is meaningless -- so an arm is always one mode.
                if mode is not None and row.get('mode') != mode:
                    continue
                seq = row['seq']
                if seq not in per_seq:
                    per_seq[seq] = {}
                    diverged[seq] = 0
                    order.append(seq)
                try:
                    ate = float(row['ate_002'])
                except (KeyError, ValueError):
                    ate = float('nan')
                if not (ate < DIVERGED_M):
                    diverged[seq] += 1
                    continue
                for key, _, _ in METRICS:
                    v = row.get(key, '')
                    if v not in ('', 'nan', None):
                        per_seq[seq].setdefault(key, []).append(float(v))
    return per_seq, diverged, order


def mean_sd(vals):
    if not vals:
        return float('nan'), float('nan'), 0
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0, n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var), n


def fmt(m, sd, n, prec):
    if n == 0:
        return '--'
    if n == 1:
        return f'{m:.{prec}f}'
    return f'{m:.{prec}f}+-{sd:.{prec}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', nargs='+', metavar=('NAME', 'DIR'),
                    required=True,
                    help='an arm: a name followed by its member directories')
    ap.add_argument('--mode', help='keep only rows with this `mode` column '
                                   'value (mono|stereo); required when a '
                                   'summary.csv holds more than one')
    ap.add_argument('--csv', help='also write the aggregate as csv')
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        if len(spec) < 2:
            raise SystemExit(f'--arm needs a name and at least one directory: {spec}')
        name, dirs = spec[0], spec[1:]
        per_seq, diverged, order = load_arm(dirs, args.mode)
        arms.append((name, per_seq, diverged, order, len(dirs)))

    seqs = []
    for _, _, _, order, _ in arms:
        for s in order:
            if s not in seqs:
                seqs.append(s)

    rows = []
    for key, label, prec in METRICS:
        print(f'\n== {label} ' + '=' * max(0, 60 - len(label)))
        head = f'{"sequence":<18}' + ''.join(
            f'{name:>22}' for name, *_ in arms)
        print(head)
        for seq in seqs:
            line = f'{seq:<18}'
            for name, per_seq, diverged, _, _ in arms:
                m, sd, n = mean_sd(per_seq.get(seq, {}).get(key, []))
                cell = fmt(m, sd, n, prec)
                if diverged.get(seq):
                    cell += f'!{diverged[seq]}'
                line += f'{cell:>22}'
                rows.append({'metric': key, 'seq': seq, 'arm': name,
                             'mean': m, 'sd': sd, 'n': n,
                             'diverged': diverged.get(seq, 0)})
            print(line)
        # The mean over sequences is taken over per-sequence means, so a long
        # sequence does not get extra weight from having more frames, and it is
        # only printed when every arm has every sequence -- otherwise it would
        # silently compare different subsets.
        complete = all(per_seq.get(seq, {}).get(key)
                       for _, per_seq, _, _, _ in arms for seq in seqs)
        if complete:
            line = f'{"MEAN":<18}'
            for name, per_seq, _, _, _ in arms:
                vals = [mean_sd(per_seq[seq][key])[0] for seq in seqs]
                line += f'{sum(vals) / len(vals):>22.{prec}f}'
            print(line)
        else:
            print(f'{"MEAN":<18}' + '(not all arms have all sequences)'.rjust(22))

    print()
    for name, _, diverged, _, nmem in arms:
        tot = sum(diverged.values())
        print(f'{name}: {nmem} members; {tot} diverged run(s)'
              + (f' -- {", ".join(f"{s}x{c}" for s, c in diverged.items() if c)}'
                 if tot else ''))
    print(f'(cells are mean+-sd over members; "!k" marks k diverged runs '
          f'excluded from that cell, ATE > {DIVERGED_M:g} m)')

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    sys.exit(main())
