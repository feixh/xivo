#!/usr/bin/env python3
"""Verify extracted TUM-VI sequences: image counts, PNG framing, csv/imu sanity.

Usage: check_tumvi.py [seq ...]        (default: every dataset-* under data/tumvi)

Why more than a file count: a `tar -x` that dies part-way leaves a directory that
looks populated and one file truncated. Reading every image would be honest but
slow; the cheap sufficient check is the PNG container -- a valid file starts with
the 8-byte signature and ends with the 12-byte IEND chunk, and a truncated write
loses the latter. Also checks that every csv timestamp is strictly increasing,
which is what the dataset loader assumes.
"""
import os, sys, struct

ROOT = '/home/ubuntu/workspace/auto-slam-engineer/data/tumvi'
SIG = b'\x89PNG\r\n\x1a\n'
IEND = b'\x00\x00\x00\x00IEND\xaeB`\x82'


def rows(p):
    out = []
    for ln in open(p):
        ln = ln.strip()
        if ln and not ln.startswith('#'):
            out.append(ln)
    return out


def check_png_dir(d):
    bad, n = [], 0
    with os.scandir(d) as it:
        for e in it:
            if not e.name.endswith('.png'):
                continue
            n += 1
            sz = e.stat().st_size
            if sz < 100:
                bad.append((e.name, f'size {sz}')); continue
            with open(e.path, 'rb') as f:
                if f.read(8) != SIG:
                    bad.append((e.name, 'no PNG signature')); continue
                f.seek(-12, os.SEEK_END)
                if f.read(12) != IEND:
                    bad.append((e.name, 'no IEND (truncated)'))
    return n, bad


def check(seq):
    d = f'{ROOT}/dataset-{seq}_512_16'
    problems = []
    for cam in ('cam0', 'cam1'):
        csv = f'{d}/mav0/{cam}/data.csv'
        if not os.path.exists(csv):
            problems.append(f'{cam}: no data.csv'); continue
        listed = rows(csv)
        t = [int(r.split(',')[0]) for r in listed]
        if any(b <= a for a, b in zip(t, t[1:])):
            problems.append(f'{cam}: timestamps not increasing')
        n, bad = check_png_dir(f'{d}/mav0/{cam}/data')
        if n != len(listed):
            problems.append(f'{cam}: {n} png vs {len(listed)} csv rows')
        # Every csv row must name a file that is actually there.
        missing = sum(0 if os.path.exists(f'{d}/mav0/{cam}/data/{r.split(",")[1]}') else 1
                      for r in listed)
        if missing:
            problems.append(f'{cam}: {missing} csv rows without a file')
        if bad:
            problems.append(f'{cam}: {len(bad)} malformed png, e.g. {bad[:3]}')
    for extra in ('imu0', 'mocap0'):
        csv = f'{d}/mav0/{extra}/data.csv'
        if not os.path.exists(csv):
            problems.append(f'{extra}: no data.csv'); continue
        if len(rows(csv)) == 0:
            problems.append(f'{extra}: empty')
    nimg = len(rows(f'{d}/mav0/cam0/data.csv')) if os.path.exists(f'{d}/mav0/cam0/data.csv') else 0
    nmoc = len(rows(f'{d}/mav0/mocap0/data.csv')) if os.path.exists(f'{d}/mav0/mocap0/data.csv') else 0
    print(f'{seq:<12} {nimg:>6} pairs  {nmoc:>7} mocap  '
          + ('OK' if not problems else 'PROBLEMS: ' + '; '.join(problems)), flush=True)
    return not problems


if __name__ == '__main__':
    seqs = sys.argv[1:]
    if not seqs:
        seqs = sorted(x[len('dataset-'):-len('_512_16')] for x in os.listdir(ROOT)
                      if x.startswith('dataset-'))
    # Not `all(check(s) for s in seqs)`: that short-circuits and hides every
    # sequence after the first bad one.
    bad = [s for s in seqs if not check(s)]
    print('\nall good' if not bad else '\nBROKEN: ' + ' '.join(bad))
    sys.exit(0 if not bad else 1)
