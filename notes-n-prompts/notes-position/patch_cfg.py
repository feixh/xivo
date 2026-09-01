#!/usr/bin/env python3
"""Set keys in one of XIVO's //-commented JSON configs, in place.

Usage: patch_cfg.py FILE dotted.key=value [dotted.key=value ...]

Values are parsed as JSON when possible (true/false/numbers/arrays), else kept
as strings. Comments are stripped -- this is meant for throwaway sweep copies,
so always back the file up first and restore it afterwards (sweep.sh does).
"""
import json
import sys
from collections import OrderedDict


def strip_comments(s):
    out = []
    i, n = 0, len(s)
    in_str = False
    while i < n:
        c = s[i]
        if in_str:
            out.append(c)
            if c == "\\":
                if i + 1 < n:
                    out.append(s[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            while i < n and s[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "*":
            i += 2
            while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def main():
    path = sys.argv[1]
    raw = open(path).read()
    cfg = json.loads(strip_comments(raw), object_pairs_hook=OrderedDict)
    for assign in sys.argv[2:]:
        dotted, _, val = assign.partition("=")
        try:
            v = json.loads(val)
        except ValueError:
            v = val
        parts = dotted.split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node:
                node[p] = OrderedDict()
            node = node[p]
        node[parts[-1]] = v
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
