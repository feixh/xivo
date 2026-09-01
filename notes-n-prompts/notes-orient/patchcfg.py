#!/usr/bin/env python3
"""Textual edits to XIVO's jsonc configs, for screening.

XIVO's configs carry `//` comments, so Python's json cannot round-trip them and
the harness patches them textually too (see run_xivo_reference.sh). This does the
same for arbitrary keys.

    patchcfg.py cfg/eff_mono.json Qimu.gyro_bias=2.2e-5 gravity=-9.80766 P.bg=1e-3

A dotted name means "this key inside that parent object". A value replaces a
scalar in place; if the existing value is a 3-array it is replaced by three
copies of the value, except for `gravity`, where it becomes [0, 0, value].
Exits nonzero unless every key matched exactly once, so a typo is loud.
"""
import re
import sys


def span_of(text, key, parent=None):
    start = 0
    if parent is not None:
        m = re.search(r'"%s"\s*:\s*\{' % re.escape(parent), text)
        if not m:
            raise SystemExit("no parent object %r" % parent)
        start = m.end()
        depth = 1
        i = start
        while depth:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        end = i
    else:
        end = len(text)
    hits = list(re.finditer(r'"%s"\s*:\s*(\[[^\]]*\]|[^,\n}]+)' % re.escape(key),
                            text[start:end]))
    if len(hits) != 1:
        raise SystemExit("%r matched %d times%s" %
                         (key, len(hits), " in %s" % parent if parent else ""))
    h = hits[0]
    return start + h.start(1), start + h.end(1), h.group(1)


def main():
    path = sys.argv[1]
    text = open(path).read()
    for arg in sys.argv[2:]:
        name, _, val = arg.partition("=")
        parent, _, key = name.rpartition(".")
        a, b, old = span_of(text, key, parent or None)
        if old.strip().startswith("["):
            if "," in val:      # explicit per-component array
                new = "[%s]" % val
            else:
                new = ("[0, 0, %s]" % val) if key == "gravity" else \
                      ("[%s, %s, %s]" % (val, val, val))
        else:
            new = val
        text = text[:a] + new + text[b:]
        print("%-28s %s -> %s" % (name, " ".join(old.split()), new))
    open(path, "w").write(text)


if __name__ == "__main__":
    main()
