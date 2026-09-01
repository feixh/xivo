#!/bin/bash
# Fast config sweep for the position work.
#
#   sweep.sh TAG [--full] [--mono-only-keys] KEY=VAL ...
#
# Patches cfg/eff_mono.json (and, unless --mono-only-keys, cfg/eff_stereo.json)
# in the xivo-position worktree, runs the reference harness, then restores both
# configs from backup no matter what.
#
# Default is the fast pass: mono only, all six rooms, --jitter 6 -- i.e. exactly
# the mono half of a full run, ~2 min, so its 6-room mean is directly comparable
# with the baseline. `--full` adds stereo (both modes, --jitter 6).
#
# Results land in experiments/results/position_TAG.
set -euo pipefail

WS=/home/ubuntu/workspace/auto-slam-engineer
WT=$WS/xivo-position
CFG_M=$WT/cfg/eff_mono.json
CFG_S=$WT/cfg/eff_stereo.json
PATCHER=$WS/notes-n-prompts/notes-position/patch_cfg.py

TAG="$1"; shift
FULL=0
MONO_KEYS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --full) FULL=1; shift ;;
    --mono-only-keys) MONO_KEYS=1; shift ;;
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
  [ "$MONO_KEYS" = 0 ] && python3 "$PATCHER" "$CFG_S" "$@"
fi

OUT=$WS/experiments/results/position_$TAG
rm -rf "$OUT"
ARGS=(--worktree xivo-position --out "$OUT" --jitter 6)
[ "$FULL" = 0 ] && ARGS+=(--mode mono)
echo "=== sweep $TAG: $* (full=$FULL) ==="
cd "$WS"
CPU_BASE=64 CPU_SPAN=60 experiments/openvins/run_xivo_reference.sh "${ARGS[@]}" \
  > "$OUT.log" 2>&1 || { echo "RUN FAILED, see $OUT.log"; tail -30 "$OUT.log"; exit 1; }
printf 'keys: %s\n' "$*" > "$OUT/sweep_keys.txt"
grep -A4 '^## ATE RMSE \[m\] -- evaluate_ate.py, 0.02' "$OUT/summary.md"
echo
grep -A4 '^## RPE 8 m -- ov_eval, median translation' "$OUT/summary.md"
