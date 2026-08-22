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
# --detailed-freq/--max-snapshots are the values the M0/M1 baseline profiles were
# taken with; keep them if you want to diff against those files.
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
# bin/vio builds the sequence directory itself (`<root>/dataset-<seq>_512_16`,
# see GetDirs in src/loader.cpp), so -root is the parent of the sequences.
ROOT="$WS/data/tumvi"

[ -x "$BIN" ] || { echo "missing $BIN -- run scripts/mem/build.sh valgrind" >&2; exit 1; }
[ -d "$ROOT/dataset-${SEQ}_512_16" ] || {
  echo "missing dataset $ROOT/dataset-${SEQ}_512_16" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"

export LD_LIBRARY_PATH="$WS/dependencies/opencv_install/lib:${LD_LIBRARY_PATH:-}"
export XIVO_RANDOM_SEED=0

cd "$XIVO"
valgrind --tool=massif \
  --massif-out-file="$OUT" \
  --threshold=0.05 \
  --detailed-freq=4 \
  --max-snapshots=40 \
  --time-unit=B \
  "$BIN" \
  -cfg "cfg/${CFG}.json" \
  -root "$ROOT" \
  -dataset tumvi \
  -seq "$SEQ" \
  -out "$OUT.traj" \
  -max_entries "$ENTRIES" \
  >/dev/null

echo "wrote $OUT"
echo "snapshots:   scripts/mem/massif_diff.py --list $OUT"
echo "attribution: scripts/mem/massif_diff.py $OUT <early> <late>"
