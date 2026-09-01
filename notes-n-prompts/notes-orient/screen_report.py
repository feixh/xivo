#!/usr/bin/env python3
"""Six-room means for one or more orient_* result directories, mono and stereo,
with the per-member sd of the room mean so a difference can be read against the
noise. `orient_m1` is always printed first as the reference."""
import csv
import os
import statistics
import sys

WS = "/home/ubuntu/workspace/auto-slam-engineer"
COLS = [("ov_ate_ori_deg", "ori"), ("ov_rpe8_ori_deg", "rpe_ori"),
        ("ate_002", "ate002"), ("ov_ate_pos_m", "ov_pos"),
        ("ov_rpe8_pos_m", "rpe_pos")]


def summarize(tag):
    path = os.path.join(WS, "experiments/results/orient_%s/summary.csv" % tag)
    rows = list(csv.DictReader(open(path)))
    out = {}
    for mode in ("mono", "stereo"):
        sel = [r for r in rows if r["mode"] == mode]
        if not sel:
            continue
        nrep = len(set(r["repeat"] for r in sel))
        cell = {}
        for key, _ in COLS:
            # per-member six-room mean, then mean and sd over members: the sd is
            # the spread of the quantity actually being compared.
            per = []
            for k in sorted(set(r["repeat"] for r in sel)):
                v = [float(r[key]) for r in sel if r["repeat"] == k]
                per.append(sum(v) / len(v))
            cell[key] = (sum(per) / len(per),
                         statistics.stdev(per) if len(per) > 1 else 0.0)
        seqs = sorted(set(r["seq"] for r in sel))
        cell["_per_seq"] = [(s, sum(float(r["ov_ate_ori_deg"])
                                    for r in sel if r["seq"] == s) /
                            len([r for r in sel if r["seq"] == s]))
                            for s in seqs]
        cell["_n"] = nrep
        out[mode] = cell
    return out


def main():
    tags = ["m1"] + [t for t in sys.argv[1:] if t != "m1"]
    hdr = "%-10s %-7s %-4s" % ("tag", "mode", "n")
    for _, lbl in COLS:
        hdr += " %9s" % lbl
    print(hdr)
    print("-" * len(hdr))
    for tag in tags:
        try:
            s = summarize(tag)
        except FileNotFoundError:
            print("%-10s (no results)" % tag)
            continue
        for mode, cell in s.items():
            line = "%-10s %-7s %-4d" % (tag, mode, cell["_n"])
            for key, _ in COLS:
                m, sd = cell[key]
                line += " %6.4f" % m if key != "ov_ate_ori_deg" else " %6.3f" % m
                line += ""
            print(line + "   sd(ori)=%.3f" % s[mode]["ov_ate_ori_deg"][1])
    print()
    print("orientation ATE per sequence")
    for tag in tags:
        try:
            s = summarize(tag)
        except FileNotFoundError:
            continue
        for mode, cell in s.items():
            print("%-10s %-7s %s" % (tag, mode, "  ".join(
                "%s=%.3f" % (a.replace("room", "r"), b)
                for a, b in cell["_per_seq"])))


if __name__ == "__main__":
    main()
