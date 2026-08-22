#!/bin/bash
# Run scripts/mem/pybind_buffer_check.py against the ASan build of pyxivo.
#
#   scripts/mem/pybind_buffer_check.sh [frames]
#
# Needs `scripts/mem/build.sh asan` first (out-asan/lib/pyxivo*.so). Both the
# grayscale and the colour buffer overload are exercised; anything other than two
# "OK ..." lines is a failure.
#
# Two preloads are needed because the instrumented code lives in a module that
# python dlopen()s rather than in the executable: libasan.so because the main
# binary was not built with -fsanitize=address, and libstdc++.so because ASan
# cannot intercept __cxa_throw if libstdc++ is loaded after the interceptors are
# initialised (it aborts in asan_interceptors.cpp with real___cxa_throw == 0).
set -euo pipefail

FRAMES="${1:-60}"

XIVO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="$(cd "$XIVO/.." && pwd)"
PY="$WS/dependencies/venv/bin/python"

ls "$XIVO"/out-asan/lib/pyxivo*.so >/dev/null 2>&1 || {
  echo "missing out-asan/lib/pyxivo*.so -- run scripts/mem/build.sh asan" >&2; exit 1; }

export LD_LIBRARY_PATH="$WS/dependencies/opencv_install/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$XIVO/out-asan/lib:${PYTHONPATH:-}"
export ASAN_OPTIONS="detect_leaks=0"
export LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libstdc++.so)"

cd "$XIVO"
status=0
for layout in gray color; do
  echo "### $layout"
  "$PY" scripts/mem/pybind_buffer_check.py "$layout" "$FRAMES" 2>&1 \
    | grep -vE '^[EIWF][0-9]' || status=1
done
exit "$status"
