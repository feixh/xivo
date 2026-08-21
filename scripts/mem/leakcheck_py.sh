#!/bin/bash
# Leak-check the path the evaluation harness actually uses: the pyxivo binding
# driven by scripts/pyxivo.py. The ASan runtime has to be preloaded, because
# the instrumented code is in a shared object that a non-instrumented python
# dlopen()s.
#
# Usage:
#   scripts/mem/leakcheck_py.sh <seq> [out-dir] [cfg]
#
#   seq      room1 .. room6                       (default room1)
#   out-dir  report destination                   (default /tmp/xivo-leak-py/<seq>)
#   cfg      estimator config in cfg/             (default tumvi_cam0)
#
# CPython's own still-reachable allocations are filtered by
# scripts/mem/lsan.supp; anything left is XIVO's.
set -uo pipefail

SEQ="${1:-room1}"
OUT="${2:-/tmp/xivo-leak-py/$SEQ}"
CFG="${3:-tumvi_cam0}"

XIVO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="$(cd "$XIVO/.." && pwd)"
DATA="${DATA:-$WS/data/tumvi}"
VENV="$WS/dependencies/venv"
LIB="$XIVO/out-asan/lib"

if ! ls "$LIB"/pyxivo*.so >/dev/null 2>&1; then
  echo "no instrumented binding in $LIB -- run scripts/mem/build.sh asan" >&2
  exit 1
fi

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$LIB${PYTHONPATH:+:$PYTHONPATH}"
export XIVO_RANDOM_SEED="${XIVO_RANDOM_SEED:-0}"
export LD_PRELOAD="$(gcc -print-file-name=libasan.so)"
export ASAN_OPTIONS="detect_leaks=1:halt_on_error=0:log_path=$OUT/asan${EXTRA_ASAN:+:$EXTRA_ASAN}"
export LSAN_OPTIONS="suppressions=$XIVO/scripts/mem/lsan.supp:print_suppressions=0:max_leaks=0"
export GLOG_minloglevel=2

cd "$XIVO"
python3 scripts/pyxivo.py \
  -root "$DATA" -dataset tumvi -seq "$SEQ" -cam_id 0 \
  -cfg "cfg/$CFG.json" -dump "$OUT" -mode eval > "$OUT/run.log" 2>&1
status=$?

cat "$OUT"/asan.* > "$OUT/report.txt" 2>/dev/null
rm -f "$OUT"/asan.*

echo "=== pyxivo $SEQ (cfg=$CFG) exit=$status ==="
if [ -s "$OUT/report.txt" ]; then
  grep -E 'ERROR: (AddressSanitizer|LeakSanitizer)|Direct leak|Indirect leak|SUMMARY:' \
    "$OUT/report.txt" | sed 's/^/  /'
else
  echo "  no sanitizer output"
fi
echo "  full report: $OUT/report.txt"
exit $status
