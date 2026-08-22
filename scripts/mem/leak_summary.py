#!/usr/bin/env python3
"""Summarise a LeakSanitizer report, grouped by the XIVO source line that
allocated the memory.

Two uses:

  leak_summary.py report.txt
      One line per allocation site: bytes, object count, and the first frame
      that lies in XIVO's own sources (falling back to the first frame at all,
      for third-party allocations).

  leak_summary.py short_report.txt long_report.txt
      Diff two runs of different lengths. A site whose byte count *grows with
      the length of the run* is an unbounded-growth leak; a site that is
      identical in both is a one-shot allocation. This is the way to find the
      leaks LeakSanitizer cannot flag on its own: memory that stays reachable
      from a global (a singleton, an object pool) for the whole run, which LSan
      treats as live. Produce the inputs with LSAN_OPTIONS=use_globals=0:...
      so that globals are not treated as roots.

Usage note: leak records are "Direct leak of N byte(s) in M object(s)" followed
by the stack, so parsing is a small state machine over those blocks.
"""
import re
import sys
from collections import defaultdict

HEAD = re.compile(r'^(Direct|Indirect) leak of ([\d,]+) byte\(s\) in ([\d,]+) object\(s\)')
FRAME = re.compile(r'^\s+#(\d+) 0x[0-9a-f]+ in (.+?) ([^\s]+:\d+)$')
FRAME_NOLOC = re.compile(r'^\s+#(\d+) 0x[0-9a-f]+ in (.+)$')
FRAME_LIB = re.compile(r'^\s+#(\d+) 0x[0-9a-f]+\s+\((.+?)[+)]')

# Frames in these are the allocator itself or the C++ runtime, never the culprit.
BORING = ('libsanitizer/asan', 'operator new', 'operator new[]', 'malloc', 'calloc',
          'realloc', 'strdup')


def site_of(frames):
    """Pick the frame to attribute a leak to: the shallowest one in xivo's own
    code, else the shallowest non-allocator frame."""
    for func, loc in frames:
        if '/xivo' in loc and 'thirdparty' not in loc:
            return f'{loc}  {func.split("(")[0].strip()}'
    for func, loc in frames:
        if not any(b in func for b in BORING) and not any(b in loc for b in BORING):
            return f'{loc}  {func.split("(")[0].strip()}'
    return 'unknown'


def parse(path):
    sites = defaultdict(lambda: [0, 0])   # site -> [bytes, objects]
    with open(path, errors='replace') as f:
        nbytes = nobj = 0
        frames = []
        for line in f:
            m = HEAD.match(line)
            if m:
                if frames:
                    s = sites[site_of(frames)]
                    s[0] += nbytes
                    s[1] += nobj
                nbytes = int(m.group(2).replace(',', ''))
                nobj = int(m.group(3).replace(',', ''))
                frames = []
                continue
            m = FRAME.match(line)
            if m:
                frames.append((m.group(2), m.group(3)))
                continue
            m = FRAME_LIB.match(line)
            if m:
                frames.append(('', m.group(2)))
                continue
            m = FRAME_NOLOC.match(line)
            if m:
                frames.append((m.group(2), ''))
        if frames:
            s = sites[site_of(frames)]
            s[0] += nbytes
            s[1] += nobj
    return sites


def show(sites, title):
    total = sum(v[0] for v in sites.values())
    print(f'== {title}: {total:,} bytes in {sum(v[1] for v in sites.values()):,} objects ==')
    for site, (b, n) in sorted(sites.items(), key=lambda kv: -kv[1][0]):
        print(f'{b:>12,} B  {n:>7,} obj  {site}')


def main():
    if len(sys.argv) == 2:
        show(parse(sys.argv[1]), sys.argv[1])
    elif len(sys.argv) == 3:
        a, b = parse(sys.argv[1]), parse(sys.argv[2])
        print(f'== growth: {sys.argv[2]} minus {sys.argv[1]} ==')
        print('(a site that grows with run length is an unbounded-growth leak)')
        for site in sorted(set(a) | set(b),
                           key=lambda s: -(b.get(s, [0, 0])[0] - a.get(s, [0, 0])[0])):
            ab, an = a.get(site, [0, 0])
            bb, bn = b.get(site, [0, 0])
            flag = 'GROWS' if bb > ab else ('same ' if bb == ab else 'shrnk')
            print(f'{flag} {bb - ab:>+12,} B  {bn - an:>+7,} obj   '
                  f'({ab:,}->{bb:,} B)  {site}')
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
