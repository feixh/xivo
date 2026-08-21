#!/bin/bash
# Run the ASan/LSan-instrumented `vio` binary on a TUM-VI sequence and report
# leaks. `vio` is preferred over the python binding for leak hunting: it is a
# plain main() around the mono+IMU path, so anything LSan reports is XIVO's.
#
# Usage:
#   scripts/mem/leakcheck.sh <seq> [out-dir] [cfg]
#
#   seq      room1 .. room6                          (default room1)
#   out-dir  where the report and trajectory go      (default /tmp/xivo-leak/<seq>)
#   cfg      vio-app config in cfg/                  (default vio_tumvi)
#
# Env:
#   MAX_ENTRIES=N   stop after N dataset entries (default 0 = whole sequence).
#                   A few thousand is enough for one-shot leaks; use the whole
#                   sequence to exercise MemoryManager slot recycling.
#   DATA=<dir>      dataset root (default ../data/tumvi)
#   EXTRA_ASAN=...  appended to ASAN_OPTIONS
#   REACHABLE=1     census mode: stop treating globals/stacks/TLS as GC roots,
#                   so memory that a singleton or the object pool still points
#                   at is reported too. Plain LSan calls that live, which is why
#                   an accumulating pool slot is invisible by default. Compare
#                   two census runs of different MAX_ENTRIES with
#                   scripts/mem/leak_summary.py: the sites that grow with run
#                   length are the real unbounded-growth leaks.
#
# Exit status is the binary's: ASan returns 23 when it reports a leak.
set -uo pipefail

SEQ="${1:-room1}"
OUT="${2:-/tmp/xivo-leak/$SEQ}"
CFG="${3:-vio_tumvi}"

XIVO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="$(cd "$XIVO/.." && pwd)"
DATA="${DATA:-$WS/data/tumvi}"
BIN="$XIVO/out-asan/bin/vio"

if [ ! -x "$BIN" ]; then
  echo "no instrumented binary at $BIN -- run scripts/mem/build.sh asan" >&2
  exit 1
fi

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

export ASAN_OPTIONS="detect_leaks=1:detect_stack_use_after_return=1:strict_string_checks=1:check_initialization_order=1:halt_on_error=0:log_path=$OUT/asan${EXTRA_ASAN:+:$EXTRA_ASAN}"
export LSAN_OPTIONS="suppressions=$XIVO/scripts/mem/lsan.supp:print_suppressions=1:report_objects=1:max_leaks=0"
if [ -n "${REACHABLE:-}" ]; then
  # No suppressions in census mode: the point is to see everything, including
  # what the third-party libraries hold, and then diff two run lengths.
  export LSAN_OPTIONS="use_globals=0:use_stacks=0:use_tls=0:use_registers=0:max_leaks=0"
fi
export XIVO_RANDOM_SEED="${XIVO_RANDOM_SEED:-0}"
export GLOG_minloglevel=2

cd "$XIVO"
"$BIN" \
  -cfg "cfg/$CFG.json" \
  -root "$DATA/" \
  -dataset tumvi -seq "$SEQ" -cam_id 0 \
  -max_entries "${MAX_ENTRIES:-0}" \
  -out "$OUT/traj_$SEQ" > "$OUT/run.log" 2>&1
status=$?

# ASan writes to $log_path.<pid>; fold those into one file for convenience.
cat "$OUT"/asan.* > "$OUT/report.txt" 2>/dev/null
rm -f "$OUT"/asan.*

echo "=== $SEQ (cfg=$CFG, max_entries=${MAX_ENTRIES:-0}) exit=$status ==="
if [ -s "$OUT/report.txt" ]; then
  grep -E 'ERROR: (AddressSanitizer|LeakSanitizer)|Direct leak|Indirect leak|SUMMARY:' "$OUT/report.txt" \
    | sed 's/^/  /'
else
  echo "  no sanitizer output"
fi
echo "  full report: $OUT/report.txt   stdout: $OUT/run.log"
exit $status
