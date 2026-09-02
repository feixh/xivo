#!/bin/bash
# Screen XIVO config variants on a subset of sequences, with an ensemble per
# variant so the comparison is against noise rather than against one draw.
#
# Usage:
#   ./sweep_xivo.sh --name NAME --patch 'dotted.key=json' [--patch ...] [options]
#   ./sweep_xivo.sh --name base                      # unpatched control
#
# Options:
#   --base P       config to patch, cfg/P_<mode>.json  (default: euroc)
#   --seqs "a b"   sequences                           (default: the screen set)
#   --members N    jitter ensemble members             (default: 3)
#   --mode M       mono | stereo | both                (default: stereo)
#   --worktree W                                       (default: xivo-euroc)
#   --out DIR      root for results       (default: ../results/euroc_tune)
#
# The screen set is four sequences chosen to cover the two failure modes found
# in M3, plus one healthy control, so a variant that fixes one thing by breaking
# another cannot hide:
#   MH_04_difficult  diverged 6/6 -- ~40 deg attitude jump in the first 3 frames
#   V1_03_difficult  diverged 5/6 -- fast flight, heavy feature churn
#   V2_01_easy       diverged 3/6 -- intermittent
#   V1_01_easy       healthy 0.062 -- the control; must not regress
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"

NAME=""
BASE="euroc"
SEQS="MH_04_difficult V1_03_difficult V2_01_easy V1_01_easy"
MEMBERS=3
MODE="stereo"
WT="xivo-euroc"
OUT="../results/euroc_tune"
PATCHES=()

while [ $# -gt 0 ]; do
  case "$1" in
    --name)     NAME="$2"; shift 2 ;;
    --base)     BASE="$2"; shift 2 ;;
    --patch)    PATCHES+=("$2"); shift 2 ;;
    --seqs)     SEQS="$2"; shift 2 ;;
    --members)  MEMBERS="$2"; shift 2 ;;
    --mode)     MODE="$2"; shift 2 ;;
    --worktree) WT="$2"; shift 2 ;;
    --out)      OUT="$2"; shift 2 ;;
    -h|--help)  sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[ -n "$NAME" ] || { echo "--name is required" >&2; exit 1; }

XIVO="$WORKSPACE/$WT"
PREFIX="tune_$NAME"

# Generate the variant config for each mode from the base config. Patches are
# `dotted.path=<json value>`, e.g. `P.Wsg=0.01` or `tracker_cfg.KLT.win_size=21`.
for mode in $( [ "$MODE" = both ] && echo "mono stereo" || echo "$MODE" ); do
  python3 - "$XIVO/cfg/${BASE}_${mode}.json" "$XIVO/cfg/${PREFIX}_${mode}.json" \
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
        # Refuse to invent a key: a typo would otherwise look like a knob that
        # simply had no effect, which is the most expensive kind of silent
        # failure in a tuning sweep.
        sys.exit(f'no such config key: {path} (missing {keys[-1]!r})')
    node[keys[-1]] = json.loads(raw)
json.dump(cfg, open(dst, 'w'), indent=2)
PY
done

echo "=== variant $NAME: ${PATCHES[*]-<none>}"
# The log redirect below is evaluated before run_xivo_reference.sh gets a chance
# to create anything, so a fresh --out root fails at the redirect -- with the
# variant reported as FAILED and no log to explain why.
mkdir -p "$OUT"
CPU_BASE="${CPU_BASE:-0}" CPU_SPAN="${CPU_SPAN:-48}" \
  "$HERE/run_xivo_reference.sh" --profile euroc_mav --mode "$MODE" \
  --jitter "$MEMBERS" --worktree "$WT" --cfg-prefix "$PREFIX" \
  --seqs "$SEQS" --out "$OUT/$NAME" > "$OUT/$NAME.log" 2>&1 \
  || { echo "FAILED, see $OUT/$NAME.log"; tail -20 "$OUT/$NAME.log"; exit 1; }
echo "=== $NAME done -> $OUT/$NAME"
