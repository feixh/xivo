#!/bin/bash
# Build a XIVO variant out-of-tree, so an instrumented build can live beside
# the plain one.
#
#   scripts/mem/build.sh release        -> build/           bin/  lib/
#   scripts/mem/build.sh asan           -> build-asan/      out-asan/{bin,lib}
#   scripts/mem/build.sh asan-ub        -> build-asan-ub/   out-asan-ub/{bin,lib}
#   scripts/mem/build.sh valgrind       -> build-valgrind/  out-valgrind/{bin,lib}
#
# `release` writes to the default bin/ and lib/ so the evaluation harness
# (which puts xivo/lib on PYTHONPATH) keeps working unchanged.
#
# Expects the workspace layout from ../../build_all.sh: OpenCV under
# dependencies/opencv_install and the python venv under dependencies/venv.
set -euo pipefail

VARIANT="${1:-release}"
XIVO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="$(cd "$XIVO/.." && pwd)"
OPENCV="$WS/dependencies/opencv_install"
VENV="$WS/dependencies/venv"
JOBS="${JOBS:-$(( $(nproc) > 32 ? 32 : $(nproc) ))}"

ARCH="-mtune=native -march=native"
case "$VARIANT" in
  release) SAN="" ;;
  asan)    SAN="address" ;;
  asan-ub) SAN="address,undefined" ;;
  # valgrind cannot execute the AVX-512 that -march=native emits here (SIGILL
  # in Eigen's static initialisers), so the profiling build targets AVX2.
  valgrind) SAN=""; ARCH="-mtune=generic -march=x86-64-v3" ;;
  *) echo "unknown variant: $VARIANT (release | asan | asan-ub | valgrind)" >&2; exit 1 ;;
esac

if [ "$VARIANT" = release ]; then
  OUT="$XIVO"                 # bin/ and lib/, where build_all.sh puts them
  BUILD="$XIVO/build"
else
  OUT="$XIVO/out-$VARIANT"    # out-asan/bin, out-asan/lib
  BUILD="$XIVO/build-$VARIANT"
fi

mkdir -p "$BUILD"
cd "$BUILD"
cmake "$XIVO" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DXIVO_SANITIZE="$SAN" \
  -DXIVO_ARCH_FLAGS="$ARCH" \
  -DXIVO_OUTPUT_DIR="$OUT" \
  -DOpenCV_DIR="$OPENCV/lib/cmake/opencv4" \
  -DPython3_EXECUTABLE="$VENV/bin/python" \
  -DPYTHON_EXECUTABLE="$VENV/bin/python" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j "$JOBS"

echo
echo "variant=$VARIANT  binaries=$OUT/bin  libs=$OUT/lib"
