#!/bin/bash
# Whole-dataset throughput: run fps_one.sh over every TUM-VI sequence and print
# one RESULT line each.
#
# Usage: fps_full_tumvi.sh <arm> <worktree> <cfg-path> <seq-list-file> [n-parallel]
#
# n-parallel defaults to 1 (strictly sequential, the cleanest measurement).
# The dataset is 289994 stereo pairs, so a sequential stereo pass is ~1.9 hours;
# with n-parallel > 1 the runs are still one-core each (fps_one.sh pins every
# thread pool), so on a 192-core host the interference is memory bandwidth only.
# Both were measured: see notes-efficiency/full-tumvi.md for the agreement
# between the parallel and sequential numbers.
set -u
ARM=$1; WT=$2; CFG=$3; LIST=$4; NPAR=${5:-1}
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t SEQS < <(grep -v '^\s*\(#\|$\)' "$LIST")
echo "# fps arm=$ARM wt=$WT cfg=$CFG seqs=${#SEQS[@]} npar=$NPAR"

n=0
for s in "${SEQS[@]}"; do
  "$H/fps_one.sh" "$ARM" "$WT" "$CFG" "$s" &
  n=$((n + 1))
  if [ "$n" -ge "$NPAR" ]; then wait -n; n=$((n - 1)); fi
done
wait
echo "# fps done"
