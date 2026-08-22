#!/bin/bash
# Count what the pooled `Feature` objects are still holding at the end of a run.
#
#   scripts/mem/pool_census.sh <sequence> [cfg] [entries]
#
# cfg      defaults to vio_tumvi   (cfg/<cfg>.json, as bin/vio expects)
# entries  defaults to 0           (-max_entries; 0 = the whole sequence)
#
# Stops the process in MemoryManager::~MemoryManager -- after the sequence has
# been played, before the pool is torn down -- and runs
# scripts/mem/pool_census.py there. Needs no instrumentation and no special
# build: the release build carries enough debug info to walk the pool.
#
# This is the direct measurement of the leak class LeakSanitizer is blind to;
# see notes-memory/m1-leak-register.md and m3-unbounded-growth.md.
set -euo pipefail

SEQ="${1:?usage: pool_census.sh <sequence> [cfg] [entries]}"
CFG="${2:-vio_tumvi}"
ENTRIES="${3:-0}"

XIVO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="$(cd "$XIVO/.." && pwd)"
BIN="$XIVO/bin/vio"
ROOT="$WS/data/tumvi"

[ -x "$BIN" ] || { echo "missing $BIN -- run scripts/mem/build.sh release" >&2; exit 1; }
[ -d "$ROOT/dataset-${SEQ}_512_16" ] || {
  echo "missing dataset $ROOT/dataset-${SEQ}_512_16" >&2; exit 1; }

export LD_LIBRARY_PATH="$WS/dependencies/opencv_install/lib:${LD_LIBRARY_PATH:-}"
export XIVO_RANDOM_SEED=0

cd "$XIVO"
gdb -batch -nx \
  -ex "set pagination off" \
  -ex "set confirm off" \
  -ex "break xivo::MemoryManager::~MemoryManager" \
  -ex run \
  -ex "source scripts/mem/pool_census.py" \
  -ex kill \
  --args "$BIN" \
    -cfg "cfg/${CFG}.json" \
    -root "$ROOT" \
    -dataset tumvi \
    -seq "$SEQ" \
    -out /dev/null \
    -max_entries "$ENTRIES" \
  2>&1 | sed -n '/POOL CENSUS/,$p'
