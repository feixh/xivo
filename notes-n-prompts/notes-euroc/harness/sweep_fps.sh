#!/bin/bash
# Screen XIVO config variants for THROUGHPUT, the timing counterpart of
# sweep_xivo.sh.
#
# Usage:
#   ./sweep_fps.sh --name NAME --patch 'dotted.key=json' [--patch ...] [options]
#   ./sweep_fps.sh --name base                        # unpatched control
#
# Options:
#   --base P       config to patch, cfg/P_<mode>.json  (default: euroc)
#   --seqs "a b"   sequences                           (default: the screen set)
#   --repeats N    timing repeats                      (default: 2)
#   --mode M       mono | stereo                       (default: stereo)
#   --worktree W                                       (default: xivo-eurocfps)
#   --out DIR      root for results       (default: ../results/euroc_fps)
#
# Every run goes through run_xivo_reference.sh --timing, which is the one-core
# protocol: `taskset -c 0 setarch -R`, every thread pool pinned to 1, sequences
# run serially, and no scoring. That is the only way a ms/frame delta is
# readable -- under the contended parallel protocol the same config measured
# 27.9 FPS and 65.7 FPS depending only on what else was running.
#
# Why a three-sequence screen is defensible here when it was not for accuracy:
# one-core FPS across the full 11 spans 65.7-72.9, a 10% range, because the cost
# is set by the image size (constant) and the feature count (87.8 of 90 slots
# occupied on every sequence). Accuracy, by contrast, spans 0.039-0.169 and the
# two halves of EuRoC disagree about which direction is better -- which is
# exactly how the M4 screen went wrong. Any arm that survives this screen still
# gets confirmed on the full 11, and gets its accuracy measured with
# sweep_xivo.sh on the full 11, before it ships.
#
# The report is ms/frame rather than FPS: ms/frame is what the price list in
# notes-euroc/m5-xivo-efficiency-tuning.md is denominated in, and it is additive
# across stages, which FPS is not.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"

NAME=""
BASE="euroc"
SEQS="MH_01_easy V1_02_medium V2_03_difficult"
REPEATS=2
MODE="stereo"
WT="xivo-eurocfps"
OUT="../results/euroc_fps"
PATCHES=()

while [ $# -gt 0 ]; do
  case "$1" in
    --name)     NAME="$2"; shift 2 ;;
    --base)     BASE="$2"; shift 2 ;;
    --patch)    PATCHES+=("$2"); shift 2 ;;
    --seqs)     SEQS="$2"; shift 2 ;;
    --repeats)  REPEATS="$2"; shift 2 ;;
    --mode)     MODE="$2"; shift 2 ;;
    --worktree) WT="$2"; shift 2 ;;
    --out)      OUT="$2"; shift 2 ;;
    -h|--help)  sed -n '2,35p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[ -n "$NAME" ] || { echo "--name is required" >&2; exit 1; }

XIVO="$WORKSPACE/$WT"
PREFIX="fps_$NAME"

# Same patcher as sweep_xivo.sh, including its refusal to invent a key: a knob
# the build does not read would otherwise report as "no cost and no benefit",
# which is indistinguishable from a correct null result.
python3 - "$XIVO/cfg/${BASE}_${MODE}.json" "$XIVO/cfg/${PREFIX}_${MODE}.json" \
  "${PATCHES[@]+"${PATCHES[@]}"}" <<'PY'
import json, re, sys
src, dst = sys.argv[1], sys.argv[2]
cfg = json.loads(re.sub(r'(?m)//.*$', '', open(src).read()))
for spec in sys.argv[3:]:
    path, _, raw = spec.partition('=')
    if not _:
        sys.exit(f'patch must be key=value, got {spec!r}')
    node, keys = cfg, path.split('.')
    for k in keys[:-1]:
        if k not in node:
            sys.exit(f'no such config key: {path} (missing {k!r})')
        node = node[k]
    if keys[-1] not in node:
        sys.exit(f'no such config key: {path} (missing {keys[-1]!r})')
    node[keys[-1]] = json.loads(raw)
json.dump(cfg, open(dst, 'w'), indent=2)
PY

echo "=== fps variant $NAME: ${PATCHES[*]-<none>}"
mkdir -p "$OUT"
for r in $(seq 0 $((REPEATS - 1))); do
  CPU_BASE=0 "$HERE/run_xivo_reference.sh" --profile euroc_mav --mode "$MODE" \
    --timing --worktree "$WT" --cfg-prefix "$PREFIX" --no-score \
    --seqs "$SEQS" --out "$OUT/$NAME/r$r" >> "$OUT/$NAME.log" 2>&1 \
    || { echo "FAILED, see $OUT/$NAME.log"; tail -20 "$OUT/$NAME.log"; exit 1; }
done
echo "=== $NAME done -> $OUT/$NAME"
