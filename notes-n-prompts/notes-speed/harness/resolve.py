#!/usr/bin/env python3
"""Turn sampler.so's raw address dump into a flat and an inclusive profile.

Usage: resolve.py RAWFILE [--depth N] [--top N]

Addresses are mapped back to (module, file offset) with the /proc/self/maps the
sampler appended, then resolved in one addr2line batch per module. Anything
outside a file-backed executable mapping (JIT, vdso) is reported as ??.
"""
import bisect
import collections
import os
import subprocess
import sys


def load(path):
    samples, maps = [], []
    in_maps = False
    for line in open(path):
        if line.startswith('=== MAPS'):
            in_maps = True
            continue
        if in_maps:
            maps.append(line.rstrip('\n'))
        else:
            parts = line.split()
            if parts:
                samples.append([int(p, 16) for p in parts])
    return samples, maps


def parse_maps(maps):
    """-> sorted list of (start, end, offset, path) for executable mappings."""
    out = []
    for line in maps:
        f = line.split(None, 5)
        if len(f) < 6:
            continue
        rng, perms, off, _dev, _ino, path = f[0], f[1], f[2], f[3], f[4], f[5].strip()
        if 'x' not in perms or not path.startswith('/'):
            continue
        a, b = rng.split('-')
        out.append((int(a, 16), int(b, 16), int(off, 16), path))
    out.sort()
    return out


def resolve_all(samples, mm):
    starts = [m[0] for m in mm]
    per_mod = collections.defaultdict(set)
    loc = {}
    for s in samples:
        for a in s:
            if a in loc:
                continue
            i = bisect.bisect_right(starts, a) - 1
            if i < 0 or a >= mm[i][1]:
                loc[a] = None
                continue
            st, _en, off, path = mm[i]
            # addr2line wants a file offset for a PIE/so; subtract the mapping's
            # base and add its file offset, then step back one byte so a return
            # address lands inside the calling instruction.
            fo = a - st + off - 1
            per_mod[path].add(fo)
            loc[a] = (path, fo)
    names = {}
    for path, offs in per_mod.items():
        if not os.path.exists(path):
            continue
        offs = sorted(offs)
        # A .so is mapped at its vaddr; for ET_DYN the file offset equals the
        # vaddr for the text segment in every layout we see here.
        try:
            p = subprocess.run(['addr2line', '-f', '-C', '-i', '-e', path],
                               input='\n'.join('0x%x' % o for o in offs),
                               capture_output=True, text=True, timeout=600)
        except Exception:
            continue
        lines = p.stdout.splitlines()
        # -i can emit several (func, file:line) pairs per address; keep the first
        # (innermost) and skip to the next address by counting pairs. Without a
        # separator we cannot tell them apart, so re-run without -i.
        p = subprocess.run(['addr2line', '-f', '-C', '-e', path],
                           input='\n'.join('0x%x' % o for o in offs),
                           capture_output=True, text=True, timeout=600)
        lines = p.stdout.splitlines()
        for k, o in enumerate(offs):
            fn = lines[2 * k] if 2 * k < len(lines) else '??'
            fl = lines[2 * k + 1] if 2 * k + 1 < len(lines) else '??'
            if fn == '??':
                fn = os.path.basename(path) + '+0x%x' % o
            names[(path, o)] = (fn, fl)
    return loc, names


def main():
    raw = sys.argv[1]
    top = 40
    if '--top' in sys.argv:
        top = int(sys.argv[sys.argv.index('--top') + 1])
    samples, maps = load(raw)
    mm = parse_maps(maps)
    loc, names = resolve_all(samples, mm)

    def name(a):
        l = loc.get(a)
        if l is None:
            return '??'
        return names.get(l, ('??', '??'))[0]

    n = len(samples)
    print('samples: %d' % n)

    # Drop the signal-delivery frames: backtrace() is called from inside the
    # SIGPROF handler, so every stack starts with the handler and the libc
    # trampoline. The first frame after them is the code that was interrupted.
    SKIP = ('handler', '__GI___sigaction', '__restore_rt', '??')

    def stack(s):
        f = [name(a) for a in s]
        i = 0
        while i < len(f) and f[i] in SKIP:
            i += 1
        return f[i:]

    stacks = [stack(s) for s in samples]
    stacks = [s for s in stacks if s]

    if '--callers' in sys.argv:
        pat = sys.argv[sys.argv.index('--callers') + 1]
        c = collections.Counter()
        hits = 0
        for s in stacks:
            for i, f in enumerate(s):
                if pat in f:
                    hits += 1
                    c[' <- '.join(s[i + 1:i + 4])[:180]] += 1
                    break
        print('\n=== CALLERS of %r (%d stacks, %.2f%%) ===' %
              (pat, hits, 100.0 * hits / len(stacks)))
        for k, v in c.most_common(top):
            print('%6.2f%% %5d  %s' % (100.0 * v / len(stacks), v, k))
        return

    if '--grep' in sys.argv:
        pat = sys.argv[sys.argv.index('--grep') + 1]
        flat = collections.Counter()
        incl = collections.Counter()
        for s in stacks:
            if pat in s[0]:
                flat[s[0]] += 1
            for f in set(s):
                if pat in f:
                    incl[f] += 1
        print('\n=== INCLUSIVE matching %r ===' % pat)
        for k, v in incl.most_common(top):
            print('%6.2f%% %5d  %s' % (100.0 * v / len(stacks), v, k[:130]))
        print('\n=== SELF matching %r ===' % pat)
        for k, v in flat.most_common(top):
            print('%6.2f%% %5d  %s' % (100.0 * v / len(stacks), v, k[:130]))
        return

    n = len(stacks)
    flat = collections.Counter()
    incl = collections.Counter()
    for s in stacks:
        flat[s[0]] += 1
        for f in set(s):
            incl[f] += 1
    print('\n=== FLAT (self, top of stack) ===')
    for k, v in flat.most_common(top):
        print('%6.2f%% %5d  %s' % (100.0 * v / n, v, k[:110]))
    print('\n=== INCLUSIVE (appears anywhere on the stack) ===')
    for k, v in incl.most_common(top):
        print('%6.2f%% %5d  %s' % (100.0 * v / n, v, k[:110]))


if __name__ == '__main__':
    main()
