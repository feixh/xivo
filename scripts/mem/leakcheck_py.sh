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
#
# Two things about loading the instrumented binding are easy to get wrong:
#
#   * `scripts/pyxivo.py` does `sys.path.insert(0, 'lib')`, which beats
#     PYTHONPATH -- setting PYTHONPATH alone silently runs the *release*
#     binding under the ASan runtime. The run therefore happens in a scratch
#     directory whose `lib` is a symlink to out-asan/lib and whose other
#     entries are symlinks back into the source tree.
#   * libstdc++ has to be preloaded next to libasan, or ASan aborts in
#     asan_interceptors.cpp:470 ("real___cxa_throw != 0") the first time
#     jsoncpp throws -- which the Estimator constructor does while parsing the
#     config.
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

# Scratch run root: everything symlinked from the source tree, except `lib`,
# which points at the instrumented binding.
RUN="$OUT/run-root"
rm -rf "$RUN"
mkdir -p "$RUN"
for entry in "$XIVO"/*; do
  [ "$(basename "$entry")" = lib ] && continue
  ln -s "$entry" "$RUN/$(basename "$entry")"
done
ln -s "$LIB" "$RUN/lib"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$LIB${PYTHONPATH:+:$PYTHONPATH}"
export XIVO_RANDOM_SEED="${XIVO_RANDOM_SEED:-0}"
export LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libstdc++.so)"
export ASAN_OPTIONS="detect_leaks=1:halt_on_error=0:log_path=$OUT/asan${EXTRA_ASAN:+:$EXTRA_ASAN}"
export LSAN_OPTIONS="suppressions=$XIVO/scripts/mem/lsan.supp:print_suppressions=0:max_leaks=0"
export GLOG_minloglevel=2

cd "$RUN"
python3 scripts/pyxivo.py \
  -root "$DATA" -dataset tumvi -seq "$SEQ" -cam_id 0 \
  -cfg "cfg/$CFG.json" -dump "$OUT" -mode eval > "$OUT/run.log" 2>&1
status=$?

cat "$OUT"/asan.* > "$OUT/report.txt" 2>/dev/null
rm -f "$OUT"/asan.*

echo "=== pyxivo $SEQ (cfg=$CFG) exit=$status ==="
if [ -s "$OUT/report.txt" ]; then
  grep -E 'ERROR: (AddressSanitizer|LeakSanitizer)|SUMMARY:' "$OUT/report.txt" |
    sed 's/^/  /'
  # Attribution, not the raw total: a python run always ends with a few hundred
  # kB of interpreter and numpy module state that LSan calls a leak and that no
  # narrow suppression can match (their stacks go through
  # _PyEval_EvalFrameDefault, which every call into pyxivo also goes through, so
  # suppressing it would mask XIVO's own leaks too). What matters is whether any
  # leaked block was allocated in XIVO's sources.
  echo "  --- attribution (scripts/mem/leak_summary.py) ---"
  python3 "$XIVO/scripts/mem/leak_summary.py" "$OUT/report.txt" | sed 's/^/  /'
  xivo_blocks="$(grep -cE "$XIVO/(src|common|pybind11)/" "$OUT/report.txt" 2>/dev/null | tail -1)"
  xivo_blocks="${xivo_blocks:-0}"
  echo "  frames in XIVO sources: $xivo_blocks"
  [ "$xivo_blocks" = 0 ] || status=1
else
  echo "  no sanitizer output"
fi
echo "  full report: $OUT/report.txt"
exit $status
