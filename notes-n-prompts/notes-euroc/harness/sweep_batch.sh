#!/bin/bash
# Run a batch of sweep_xivo.sh variants back to back.
#
# Usage:
#   ./sweep_batch.sh 'name key=val [key=val ...]' 'name2 key=val' ...
#
# Each argument is one variant: a name, then whitespace-separated patches, each
# forwarded as `--patch`. The common patches every variant in the batch shares go
# in $COMMON (also whitespace-separated), the screen set in $SEQS, the worktree
# to run against in $WORKTREE, the members per variant in $MEMBERS, and the
# results root in $OUT.
#
# $WORKTREE matters more than it looks: a batch that patches a config key the
# target worktree's binary does not read will run happily and report the control
# twice. sweep_xivo.sh catches the typo case (it refuses a key absent from the
# config) but not the wrong-build case, so set it deliberately.
#
# This exists because the interactive shell here is zsh, which does not
# word-split unquoted parameters and has no `read -ra`. Writing the loop inline
# silently produced six variants with an empty patch list -- sweep_xivo.sh caught
# it, but only because it refuses an unparseable patch. Keeping the batch driver
# in a real bash file removes the whole class of mistake.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SEQS:=MH_04_difficult V1_03_difficult V2_01_easy V1_01_easy MH_01_easy}"
: "${COMMON:=}"
: "${WORKTREE:=xivo-euroc}"
: "${MEMBERS:=3}"
# $OUT keeps one batch's variants out of another's namespace. It matters because
# the variant name is the only thing distinguishing two runs in a shared root, so
# a batch that reuses a name from an earlier milestone silently overwrites it --
# and the overwrite is invisible once the log has scrolled.
: "${OUT:=../results/euroc_tune}"
export CPU_BASE="${CPU_BASE:-0}" CPU_SPAN="${CPU_SPAN:-60}"
echo "batch: worktree=$WORKTREE members=$MEMBERS out=$OUT common='$COMMON'"

rc=0
for spec in "$@"; do
  set -- $spec                    # deliberate word split: name then patches
  name="$1"; shift
  args=()
  for p in "$@"; do args+=(--patch "$p"); done
  for p in $COMMON; do args+=(--patch "$p"); done
  if "$HERE/sweep_xivo.sh" --name "$name" --seqs "$SEQS" --out "$OUT" \
       --worktree "$WORKTREE" --members "$MEMBERS" "${args[@]}"; then
    :
  else
    echo "!!! variant $name FAILED"; rc=1
  fi
done
echo "BATCH_DONE rc=$rc"
exit $rc
