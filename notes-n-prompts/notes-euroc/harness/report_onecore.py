#!/usr/bin/env python3
"""Frame-weighted one-core ms/frame from --timing run trees, for any layout.

Usage:  report_onecore.py NAME=GLOB [NAME=GLOB ...]

Why this exists next to report_fps.py: report_fps.py reads the `time.txt` that
run_xivo_reference.sh writes for XIVO runs, and reports a nonzero-exit warning for
every OpenVINS run because the OpenVINS path does not write one. Both paths do
write `stats.txt`, so that is what this reads -- which makes XIVO and OpenVINS
timings comparable without special-casing either.

ms/frame is total wall over total frames, summed across sequences before dividing,
so the 2x spread in sequence length weights correctly. `sd` is over repeats of that
aggregate -- the reproducibility of the protocol (0.002 ms here), not the spread
across sequences.

A repeat is one directory under the glob; both layouts are accepted:
  <arm>/r0/stereo/<SEQ>_r0/stats.txt     (sweep_fps.sh: repeats are sibling dirs)
  <arm>/stereo/<SEQ>_r{0,1,2}/stats.txt  (run_xivo_reference.sh --jitter: in-tree)
"""
import collections
import glob
import os
import re
import statistics as st
import sys


def read(path):
    d = {}
    with open(path) as f:
        for line in f:
            k, sep, v = line.strip().partition('=')
            if sep:
                d[k] = v
    return d


def repeats_of(root, mode):
    """Group the runs under `root` into repeats, whichever layout it uses."""
    sib = sorted(d for d in glob.glob(f'{root}/*/{mode}') if os.path.isdir(d))
    if sib:
        return [sorted(glob.glob(f'{d}/*')) for d in sib]
    runs = collections.defaultdict(list)
    for run in sorted(glob.glob(f'{root}/{mode}/*')):
        m = re.search(r'_r(\d+)$', os.path.basename(run))
        runs[int(m.group(1)) if m else 0].append(run)
    return [runs[k] for k in sorted(runs)]


def arm(pattern, mode='stereo'):
    per, tot, rss = collections.defaultdict(list), [], []
    for root in sorted(glob.glob(pattern)):
        for group in repeats_of(root, mode):
            frames, wall, peaks = 0, 0.0, []
            for run in group:
                stats = os.path.join(run, 'stats.txt')
                if not os.path.exists(stats):
                    continue
                s = read(stats)
                n = int(float(s.get('frames_processed', 0)))
                w = float(s.get('wall_total_s', 0) or 0)
                if not n or not w:
                    continue
                seq = re.sub(r'_r\d+$', '', os.path.basename(run))
                per[seq].append(w / n * 1000)
                peaks.append(float(s.get('peak_rss_mb', 0)))
                frames, wall = frames + n, wall + w
            if frames:
                tot.append(wall / frames * 1000)
                rss.append(st.mean(peaks))
    return tot, per, rss


def main(args):
    for spec in args:
        name, _, pattern = spec.partition('=')
        tot, per, rss = arm(pattern)
        if not tot:
            print(f'{name:<16} no runs under {pattern}')
            continue
        sd = st.stdev(tot) if len(tot) > 1 else 0.0
        print(f'{name:<16}{st.mean(tot):7.3f} +- {sd:.3f} ms/frame   '
              f'{1000 / st.mean(tot):5.1f} FPS   {st.mean(rss):6.1f} MB   '
              f'reps={len(tot)}  seqs={len(per)}')
        for seq in sorted(per):
            print(f'    {seq:<18}{st.mean(per[seq]):7.3f} ms')


if __name__ == '__main__':
    main(sys.argv[1:])
