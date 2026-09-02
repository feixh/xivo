#!/usr/bin/env python3
"""Report ms/frame and peak RSS for sweep_fps.sh variants, frame-weighted.

Usage:
  report_fps.py DIR [DIR ...]           # each DIR is one variant (or a run root)

Frame-weighted rather than a mean of per-sequence means: the sequences differ in
length by 2x, and what a config costs is a per-frame quantity, so the long
sequence should count for more. The unweighted per-sequence table is printed too,
because a change that helps one half of EuRoC and hurts the other would average
away.

`sd` is over repeats of the aggregate, not over sequences -- it is the
reproducibility of the protocol (measured at 0.1% on this machine), which is what
tells you whether a delta between two variants is real.
"""
import os
import re
import statistics
import sys


def read_stats(path):
    d = {}
    with open(path) as f:
        for line in f:
            k, _, v = line.strip().partition('=')
            if _:
                d[k] = v
    return d


def collect(root):
    """-> {repeat: {seq: (frames, wall_s, rss_mb)}}"""
    out = {}
    for dirpath, _, files in os.walk(root):
        if 'stats.txt' not in files:
            continue
        s = read_stats(os.path.join(dirpath, 'stats.txt'))
        if s.get('exit_code') != '0':
            print(f'  !! nonzero exit in {dirpath}', file=sys.stderr)
            continue
        # .../<variant>/r<N>/<mode>/<seq>_r0/stats.txt, or .../<mode>/<seq>_r0
        parts = dirpath.split(os.sep)
        seq = re.sub(r'_r\d+$', '', parts[-1])
        rep = next((p for p in reversed(parts) if re.fullmatch(r'r\d+', p)
                    and p != parts[-1]), 'r0')
        out.setdefault(rep, {})[seq] = (int(s['frames_processed']),
                                        float(s['wall_total_s']),
                                        float(s['peak_rss_mb']))
    return out


def main():
    print(f'{"variant":22s} {"ms/frame":>9s} {"sd":>6s} {"FPS":>7s} '
          f'{"RSS MB":>7s} {"n":>3s}')
    per_seq_rows = {}
    for root in sys.argv[1:]:
        reps = collect(root)
        if not reps:
            print(f'{os.path.basename(root):22s}   no runs')
            continue
        aggs, rsss = [], []
        for rep, seqs in sorted(reps.items()):
            fr = sum(v[0] for v in seqs.values())
            wa = sum(v[1] for v in seqs.values())
            if fr:
                aggs.append(1000.0 * wa / fr)
            rsss.append(max(v[2] for v in seqs.values()))
        name = os.path.basename(os.path.normpath(root))
        sd = statistics.stdev(aggs) if len(aggs) > 1 else 0.0
        m = statistics.mean(aggs)
        print(f'{name:22s} {m:9.3f} {sd:6.3f} {1000.0 / m:7.2f} '
              f'{statistics.mean(rsss):7.1f} {len(aggs):3d}')
        # per-sequence, averaged over repeats
        for seq in sorted({s for v in reps.values() for s in v}):
            vals = [1000.0 * v[seq][1] / v[seq][0]
                    for v in reps.values() if seq in v and v[seq][0]]
            per_seq_rows.setdefault(seq, {})[name] = statistics.mean(vals)

    names = []
    for row in per_seq_rows.values():
        for n in row:
            if n not in names:
                names.append(n)
    if len(names) > 1:
        print('\nms/frame by sequence')
        print(f'{"sequence":18s}' + ''.join(f'{n[:11]:>12s}' for n in names))
        for seq in sorted(per_seq_rows):
            row = per_seq_rows[seq]
            print(f'{seq:18s}' + ''.join(
                f'{row[n]:12.3f}' if n in row else f'{"-":>12s}'
                for n in names))


if __name__ == '__main__':
    main()
