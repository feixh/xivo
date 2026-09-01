#!/bin/bash
# Config sweep for the front-end/image-path work, on the xivo-frontfast worktree.
#
#   sweep.sh TAG [--mono|--stereo|--both] [--timing] [--cpu BASE SPAN] KEY=VAL ...
#
# Patches cfg/eff_mono.json and cfg/eff_stereo.json in xivo-frontfast, runs the
# reference harness, and restores both configs no matter how it exits. Results
# land in experiments/results/frontfast_TAG.
#
# --both --jitter 6 is the accuracy contract pass (~5 min). --timing is the
# one-core throughput pass; do not run two --timing passes at once, and do not
# run one while an accuracy pass is using the same cpus.
#
# NEVER run two of these at once, even on disjoint cpus: run_xivo_reference.sh
# reads cfg/eff_{mono,stereo}.json out of the worktree on every run, so a second
# invocation would silently swap the first one's config mid-sweep. --cpu exists
# for sharing the box with *another worktree*, not with another sweep here.
set -euo pipefail

WS=/home/ubuntu/workspace/auto-slam-engineer
WT=$WS/xivo-frontfast
CFG_M=$WT/cfg/eff_mono.json
CFG_S=$WT/cfg/eff_stereo.json
PATCHER=$WS/notes-n-prompts/notes-position/patch_cfg.py

TAG="$1"; shift
MODE="both"
TIMING=0
BASE=64
SPAN=60
while [ $# -gt 0 ]; do
  case "$1" in
    --mono)   MODE=mono; shift ;;
    --stereo) MODE=stereo; shift ;;
    --both)   MODE=both; shift ;;
    --timing) TIMING=1; shift ;;
    --cpu)    BASE="$2"; SPAN="$3"; shift 3 ;;
    *) break ;;
  esac
done

BK=$(mktemp -d)
cp "$CFG_M" "$BK/m.json"
cp "$CFG_S" "$BK/s.json"
restore() { cp "$BK/m.json" "$CFG_M"; cp "$BK/s.json" "$CFG_S"; rm -rf "$BK"; }
trap restore EXIT

if [ $# -gt 0 ]; then
  python3 "$PATCHER" "$CFG_M" "$@"
  python3 "$PATCHER" "$CFG_S" "$@"
fi

OUT=$WS/experiments/results/frontfast_$TAG
rm -rf "$OUT"
ARGS=(--worktree xivo-frontfast --out "$OUT" --mode "$MODE")
if [ "$TIMING" = 1 ]; then
  ARGS+=(--timing)
else
  ARGS+=(--jitter 6)
fi
echo "=== sweep $TAG: mode=$MODE timing=$TIMING keys: $* ==="
cd "$WS"
CPU_BASE=$BASE CPU_SPAN=$SPAN experiments/openvins/run_xivo_reference.sh "${ARGS[@]}" \
  > "$OUT.log" 2>&1 || { echo "RUN FAILED, see $OUT.log"; tail -30 "$OUT.log"; exit 1; }
printf 'keys: %s\nmode: %s\ntiming: %s\n' "$*" "$MODE" "$TIMING" > "$OUT/sweep_keys.txt"
# grep -A, not `sed -n '/hdr/,/^$/p'`: every section header in summary.md is
# followed by a blank line before its table, so a range ending at /^$/ prints the
# header and nothing else. That is what made the first three arms here look like
# they had produced no numbers.
show() { grep -A5 -F "$1" "$OUT/summary.md" || true; }
if [ "$TIMING" = 1 ]; then
  show '## Throughput'
  show '## Peak RSS'
else
  show '## ATE RMSE [m] -- evaluate_ate.py, 0.02'
  show '## ATE RMSE [deg] -- ov_eval posyaw, orientation'
  show '## RPE 8 m -- ov_eval, median translation'
  show '## RPE 8 m -- ov_eval, median rotation'
fi
