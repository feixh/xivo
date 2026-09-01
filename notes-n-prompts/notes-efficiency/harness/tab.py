#!/usr/bin/env python3
"""Aggregate fps_batch.sh RESULT lines into a per-(arm, seq) table.

Usage: tab.py sweeps/foo.log [more.log ...]

FPS is frames/wall-clock; the frame counts are the fixed TUM-VI room lengths, so
the same sequence is always the same amount of work. Reports the median over
repeats (repeats are few, and the failure mode on a shared host is a slow
outlier, not a fast one).
"""
import re
import statistics
import sys

FRAMES = {  # image count per sequence, cam0
    'room1': 2821, 'room2': 2571, 'room3': 2593,
    'room4': 2521, 'room5': 2439, 'room6': 2636,
}
COLS = ['visual_meas', 'track', 'process_tracks', 'update', 'jacobian', 'mh',
        'actual_update', 'stereo_gating', 'oos_jac', 'propagation']


def main(paths):
    rows = {}
    for p in paths:
        for line in open(p):
            if not line.startswith('RESULT '):
                continue
            f = line.split()
            if 'FAILED' in line:
                print('FAILED:', line.strip(), file=sys.stderr)
                continue
            arm, seq = f[1], f[2]
            kv = dict(x.split('=', 1) for x in f[3:] if '=' in x)
            rows.setdefault((arm, seq), []).append(kv)

    arms, seqs = [], []
    for arm, seq in rows:
        if arm not in arms:
            arms.append(arm)
        if seq not in seqs:
            seqs.append(seq)
    seqs.sort()

    hdr = '%-22s %-7s %6s %6s %7s %7s' % ('arm', 'seq', 'wall', 'FPS', 'rss_MB', 'n')
    hdr += ''.join('%9s' % c[:9] for c in COLS)
    print(hdr)
    print('-' * len(hdr))
    base = {}
    for seq in seqs:
        for arm in arms:
            v = rows.get((arm, seq))
            if not v:
                continue
            wall = statistics.median(float(x['wall']) for x in v)
            rss = statistics.median(float(x['rss_kb']) for x in v) / 1024.0
            fps = FRAMES.get(seq, float('nan')) / wall
            line = '%-22s %-7s %6.1f %6.2f %7.0f %7d' % (arm, seq, wall, fps, rss, len(v))
            for c in COLS:
                try:
                    m = statistics.median(float(x[c]) for x in v)
                except (KeyError, ValueError):
                    m = float('nan')
                line += '%9.2f' % m
            print(line)
            base.setdefault(seq, fps)
        print()

    # Speedup, averaged over sequences, against the baseline arm *of the same
    # setting*: mono and stereo do different amounts of work, so a single global
    # reference would compare a stereo arm against a mono one. Arms are named
    # `<label>_<setting>` and the reference for a setting is the first arm with
    # that suffix (conventionally `base_<setting>`).
    def setting(arm):
        return arm.rsplit('_', 1)[-1]

    def fps_of(arm, seq):
        v = rows.get((arm, seq))
        if not v:
            return None
        return FRAMES[seq] / statistics.median(float(x['wall']) for x in v)

    if len(arms) > 1:
        refs = {}
        for arm in arms:
            refs.setdefault(setting(arm), arm)
        print('%-22s %-8s %8s' % ('arm', 'vs', 'FPSx'))
        for arm in arms:
            ref = refs[setting(arm)]
            r = [fps_of(arm, s) / fps_of(ref, s) for s in seqs
                 if fps_of(arm, s) and fps_of(ref, s)]
            if r:
                print('%-22s %-8s %8.3f' % (arm, ref, sum(r) / len(r)))


if __name__ == '__main__':
    main(sys.argv[1:])
