#!/usr/bin/env python3
"""Diff two massif snapshots and rank allocation sites by how much they grew.

    scripts/mem/massif_diff.py --list out.massif
    scripts/mem/massif_diff.py out.massif 20 34 [frames]

`frames` (default 6) keeps only the innermost N frames of each call path, which
is where the interesting line usually is; massif's own paths run 15 frames deep
through the estimator's dispatch.

Why this exists: `ms_print` shows one tree per snapshot, and a leak on this
codebase is not a big node -- it is a node that is *bigger later*. Comparing two
detailed snapshots from the steady-state part of the run separates growth from
the (large, constant) pre-allocated feature pool.

Sites are keyed by their whole call path with the hex addresses stripped, so the
same source line matches across snapshots even when it was inlined at a
different address.
"""

import re
import sys
from collections import OrderedDict

ADDR = re.compile(r"^0x[0-9A-Fa-f]+: ")
NODE = re.compile(r"^(\s*)n(-?\d+): (\d+) (.*)$")

# Frames that only say "a container reallocated" and never say which container.
# Dropped when shortening a path, so the innermost frames that are shown are the
# innermost frames that mean something.
BOILERPLATE = re.compile(
    r"\((?:stl_vector\.h|vector\.tcc|stl_tree\.h|alloc_traits\.h|new_allocator\.h"
    r"|hashtable\.h|hashtable_policy\.h|unordered_map\.h|unordered_set\.h"
    r"|stl_map\.h|stl_set\.h|basic_string\.h|Memory\.h|allocator\.h):\d+\)$"
)


LOCATION = re.compile(r"\s\(([^()]*)\)$")


def frame(desc):
    """`ns::f(long, tmpl<...>&) (file.cpp:12)` -> `ns::f (file.cpp:12)`.

    Demangled signatures of the estimator's templated call path are hundreds of
    characters wide and never carry information the file:line does not.
    """
    m = LOCATION.search(desc)
    if not m:
        return desc
    where = m.group(1)
    if where.startswith("in "):                    # `(in /long/path/libfoo.so)`
        where = "in " + where.rsplit("/", 1)[-1]
    name = desc[: m.start()].split("(", 1)[0].strip()
    return f"{name} ({where})" if name else f"({where})"


def shorten(key, frames):
    """Keep the innermost `frames` meaningful frames of a massif call path.

    massif writes paths outermost-first (`main <- ... <- malloc`), so the frame
    that names the leaking line is near the end.
    """
    parts = [frame(p) for p in key.split(" <- ") if not BOILERPLATE.search(p)]
    if frames <= 0 or len(parts) <= frames:
        return " <- ".join(parts)
    return "... <- " + " <- ".join(parts[-frames:])


def parse(path):
    """-> OrderedDict {snapshot_index: {"heap": bytes, "sites": {path: bytes}}}"""
    snapshots = OrderedDict()
    cur = None          # current snapshot dict
    stack = []          # description of each open tree level
    pending = {}        # path -> bytes, for the snapshot being read

    def flush():
        if cur is not None:
            cur["sites"] = pending.copy()

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("snapshot="):
                flush()
                pending.clear()
                stack.clear()
                cur = {"heap": 0, "sites": {}}
                snapshots[int(line.split("=", 1)[1])] = cur
                continue
            if cur is None:
                continue
            if line.startswith("mem_heap_B="):
                cur["heap"] = int(line.split("=", 1)[1])
                continue
            m = NODE.match(line)
            if not m:
                continue
            indent, nchildren, nbytes, desc = m.groups()
            depth = len(indent)
            desc = ADDR.sub("", desc).strip()
            stack[:] = stack[:depth] + [desc]
            # Only leaves carry attributable bytes; interior nodes are the sum of
            # their children and would be double counted.
            if int(nchildren) == 0:
                key = " <- ".join(reversed(stack[1:])) or stack[0]
                pending[key] = pending.get(key, 0) + int(nbytes)
    flush()
    return snapshots


def main(argv):
    if len(argv) == 3 and argv[1] == "--list":
        snaps = parse(argv[2])
        detailed = [(i, s) for i, s in snaps.items() if s["sites"]]
        print(f"{len(snaps)} snapshots, {len(detailed)} detailed")
        for i, s in snaps.items():
            mark = "detailed" if s["sites"] else ""
            print(f"  {i:>3}  heap={s['heap'] / 1e6:8.2f} MB  {mark}")
        return 0

    if len(argv) not in (4, 5):
        print(__doc__.strip(), file=sys.stderr)
        return 2

    path, a, b = argv[1], int(argv[2]), int(argv[3])
    frames = int(argv[4]) if len(argv) == 5 else 6
    snaps = parse(path)
    for i in (a, b):
        if i not in snaps:
            print(f"no snapshot {i} in {path}", file=sys.stderr)
            return 1
        if not snaps[i]["sites"]:
            print(f"snapshot {i} is not detailed (heap_tree=empty)", file=sys.stderr)
            return 1

    sa, sb = snaps[a]["sites"], snaps[b]["sites"]
    print(f"snapshot {a}: heap {snaps[a]['heap'] / 1e6:.2f} MB -> "
          f"snapshot {b}: heap {snaps[b]['heap'] / 1e6:.2f} MB "
          f"({(snaps[b]['heap'] - snaps[a]['heap']) / 1e6:+.2f} MB)")
    print()

    deltas = []
    for key in set(sa) | set(sb):
        before, after = sa.get(key, 0), sb.get(key, 0)
        if after != before:
            deltas.append((after - before, before, after, key))
    deltas.sort(reverse=True)

    print(f"{'delta':>12}  {'before':>12}  {'after':>12}  site")
    for delta, before, after, key in deltas:
        if abs(delta) < 1024:
            continue
        print(f"{delta:>+12,}  {before:>12,}  {after:>12,}  "
              f"{shorten(key, frames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
