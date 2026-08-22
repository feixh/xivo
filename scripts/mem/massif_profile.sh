#!/bin/bash
# Profile the heap of `bin/vio` with valgrind massif and attribute growth to
# source lines.
#
#   scripts/mem/massif_profile.sh room1 /tmp/massif_room1.out [cfg] [entries]
#
# cfg      defaults to vio_tumvi        (cfg/<cfg>.json, as bin/vio expects)
# entries  defaults to 8000             (-max_entries; massif is ~40x slower)
#
# Two flags matter and are easy to get wrong:
#
#   --threshold=0.05  massif's threshold is applied when the tree is *recorded*,
#                     not just when ms_print formats it. At the default 1.0 every
#                     growing site here collapses into a single
#                     "in N places, all below massif's threshold" line and no
#                     attribution is possible at all.
#   --time-unit=B     wall-clock snapshots are useless under a 40x slowdown;
#                     bytes-allocated puts the detailed snapshots at comparable
#                     points of the run.
#
# Needs the `valgrind` build variant: -march=native emits AVX-512 that valgrind
# 3.26 cannot execute (SIGILL inside Eigen's static initialisers).
#   scripts/mem/build.sh valgrind
set -euo pipefail

SEQ="${1:?usage: massif_profile.sh <sequence> <out.massif> [cfg] [entries]}"
OUT="${2:?usage: massif_profile.sh <sequence> <out.massif> [cfg] [entries]}"
CFG="${3:-vio_tumvi}"
ENTRIES="${4:-8000}"

XIVO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="$(cd "$XIVO/.." && pwd)"
BIN="$XIVO/out-valgrind/bin/vio"
DATA="$WS/data/tumvi/dataset-${SEQ}_512_16"

[ -x "$BIN" ] || { echo "missing $BIN -- run scripts/mem/build.sh valgrind" >&2; exit 1; }
[ -d "$DATA" ] || { echo "missing dataset $DATA" >&2; exit 1; }

export LD_LIBRARY_PATH="$WS/dependencies/opencv_install/lib:${LD_LIBRARY_PATH:-}"
export XIVO_RANDOM_SEED=0

cd "$XIVO"
valgrind --tool=massif \
  --massif-out-file="$OUT" \
  --threshold=0.05 \
  --detailed-freq=1 \
  --max-snapshots=100 \
  --time-unit=B \
  "$BIN" \
  -cfg "cfg/${CFG}.json" \
  -root "$DATA" \
  -dataset tumvi \
  -max_entries "$ENTRIES" \
  >/dev/null

echo "wrote $OUT"
echo "snapshots:   scripts/mem/massif_diff.py --list $OUT"
echo "attribution: scripts/mem/massif_diff.py $OUT <early> <late>"
