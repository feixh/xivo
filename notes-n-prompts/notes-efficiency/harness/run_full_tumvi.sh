#!/bin/bash
# Run one config over a list of TUM-VI sequences and evaluate it.
#
# Usage: run_full_tumvi.sh <config-name> <out-dir> <seq-list-file>
#
# Two reasons it exists:
#
#   - it keeps the sequence list out of a shell variable. The interactive shell
#     here is zsh, which does *not* word-split unquoted parameters, so
#     `./run_eval_bugfix.sh cfg out $SEQS` silently passes all 23 names as a
#     single argument and XIVO is asked for a sequence called
#     "corridor1 corridor2 ...". The list is read from a file and expanded by bash.
#   - it pins the thread pools. run_eval_bugfix.sh launches every sequence
#     concurrently, and each pyxivo process otherwise spawns an OpenCV pool sized
#     to the whole machine: 46 runs x 192 threads gave load average 6400 and 51%
#     system time, i.e. the batch measured context switching. OMP_NUM_THREADS
#     alone is not enough -- OpenCV's parallel_for backend needs
#     OPENCV_FOR_THREADS_NUM. Pinned output is bit-identical (measured), so this
#     costs nothing but the ~6% single-run wall clock.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CFG=$1; OUT=$2; LIST=$3
mapfile -t SEQS < <(grep -v '^\s*\(#\|$\)' "$LIST")
echo "$CFG -> $OUT over ${#SEQS[@]} sequences"
exec env OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 \
         MKL_NUM_THREADS=1 XIVO_WT="${XIVO_WT:-xivo}" \
  "$ROOT/run_eval_bugfix.sh" "$CFG" "$OUT" "${SEQS[@]}"
