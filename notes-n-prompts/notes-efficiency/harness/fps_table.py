#!/usr/bin/env python3
"""Turn fps_full_tumvi.sh's RESULT lines into a per-sequence FPS table.

Usage: fps_table.py <fps-log> [<fps-log> ...]

FPS is frames/wall where `frames` is the number of rows in the sequence's
cam0/data.csv, i.e. every image the process had to decode -- not the number of
poses in the dump, which is a few short because the estimator needs a couple of
images before it emits anything. `-mode runOnly` means wall clock is PNG decode
plus the Python feed loop plus the estimator and nothing else.

Also reports peak RSS. Note that GNU time's ru_maxrss reads ~25% low against a
/proc high-water sampler, so treat these as a lower bound and as comparable to
each other rather than absolute.
"""
import os, re, sys, collections

ROOT = '/home/ubuntu/workspace/auto-slam-engineer/data/tumvi'
GROUPS = ['room', 'corridor', 'magistrale', 'outdoors', 'slides']
RESULT = re.compile(r'^RESULT (\S+) (\S+) wall=(\S+) user=(\S+) rss_kb=(\S+)(.*)$')


def frames(seq):
    p = f'{ROOT}/dataset-{seq}_512_16/mav0/cam0/data.csv'
    return sum(1 for l in open(p) if l.strip() and not l.startswith('#'))


def group_of(s):
    g = re.match(r'[a-z]+', s).group(0)
    return g if g in GROUPS else 'other'


def main():
    rows = {}
    for path in sys.argv[1:]:
        for ln in open(path):
            m = RESULT.match(ln.strip())
            if m:
                arm, seq, wall, user, rss, rest = m.groups()
                d = dict(re.findall(r'(\w+)=([\d.]+)', rest))
                rows[seq] = dict(arm=arm, wall=float(wall), user=float(user),
                                 rss=float(rss), **{k: float(v) for k, v in d.items()})
    if not rows:
        sys.exit('no RESULT lines found')
    order = sorted(rows, key=lambda s: (GROUPS.index(group_of(s))
                                        if group_of(s) in GROUPS else 9, s))
    print(f'{"seq":<12}{"frames":>8}{"wall_s":>9}{"FPS":>8}{"x real":>8}'
          f'{"RSS_MB":>9}{"visual":>9}{"update":>9}')
    tot_f = tot_w = 0.0
    acc = collections.defaultdict(list)
    for s in order:
        r = rows[s]
        n = frames(s)
        fps = n / r['wall']
        tot_f += n; tot_w += r['wall']
        acc[group_of(s)].append(fps)
        print(f'{s:<12}{n:>8}{r["wall"]:>9.1f}{fps:>8.1f}{fps / 20:>8.2f}'
              f'{r["rss"] / 1024:>9.0f}{r.get("visual_meas", float("nan")):>9.2f}'
              f'{r.get("update", float("nan")):>9.2f}')
    print('-' * 72)
    for g in GROUPS + ['other']:
        if g in acc:
            print(f'{g + " mean FPS":<20}{sum(acc[g]) / len(acc[g]):>8.1f}')
    print(f'{"WHOLE DATASET":<20}{tot_f / tot_w:>8.1f}   '
          f'({tot_f:.0f} frames in {tot_w / 60:.1f} min of compute, '
          f'{tot_f / 20 / 3600:.2f} h of data, {tot_f / tot_w / 20:.2f}x real time)')


if __name__ == '__main__':
    main()
