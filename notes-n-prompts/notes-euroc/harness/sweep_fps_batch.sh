#!/bin/bash
# Run a batch of sweep_fps.sh variants back to back. Same contract as
# sweep_batch.sh -- each argument is 'name key=val [key=val ...]' -- and a real
# bash file for the same reason: zsh does not word-split, so writing the loop
# inline in the interactive shell silently produces variants with no patches.
#
# Strictly serial, and it must stay that way: the whole point of --timing is a
# single pinned core with nothing else on it, so two variants in flight would
# measure each other.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SEQS:=MH_01_easy V1_02_medium V2_03_difficult}"
: "${COMMON:=}"
: "${WORKTREE:=xivo-eurocfps}"
: "${REPEATS:=2}"
: "${OUT:=../results/euroc_fps}"
echo "fps batch: worktree=$WORKTREE repeats=$REPEATS common='$COMMON'"
echo "           seqs=$SEQS"

rc=0
for spec in "$@"; do
  set -- $spec                    # deliberate word split: name then patches
  name="$1"; shift
  args=()
  for p in "$@"; do args+=(--patch "$p"); done
  for p in $COMMON; do args+=(--patch "$p"); done
  if "$HERE/sweep_fps.sh" --name "$name" --seqs "$SEQS" --out "$OUT" \
       --worktree "$WORKTREE" --repeats "$REPEATS" "${args[@]}"; then
    :
  else
    echo "!!! fps variant $name FAILED"; rc=1
  fi
done
echo "FPS_BATCH_DONE rc=$rc"
exit $rc
