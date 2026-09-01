#!/bin/bash
# Run a set of (arm, worktree, cfg) triples over a set of sequences, R repeats,
# strictly sequentially, with the arms interleaved inside each repeat so that
# load drift on this shared host hits every arm equally.
#
# Usage:
#   ARMS="name:worktree:cfg[:lib-suffix] ..." SEQS="room1 room6" R=2 \
#     fps_batch.sh > sweeps/foo.log
#
# The optional fourth field is the XIVO_OUTPUT_SUFFIX of the build to run, for
# timing several flag-variants of one source tree against each other.
set -u
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQS=${SEQS:-"room1 room6"}
R=${R:-2}
: "${ARMS:?set ARMS=\"name:worktree:cfg ...\"}"

echo "# date=$(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "# arms=$ARMS seqs=$SEQS repeats=$R"
for ((r = 1; r <= R; r++)); do
  for seq in $SEQS; do
    for a in $ARMS; do
      IFS=: read -r name wt cfg libsuf <<<"$a"
      out=$("$H/fps_one.sh" "$name" "$wt" "$cfg" "$seq" "${libsuf:-}")
      # Per-run load, because this host is shared: interleaving spreads a drift
      # across the arms but does not remove it, and a run that overlapped a
      # 1000-load spike has to be identifiable after the fact.
      echo "$out rep=$r load=$(cut -d' ' -f1 /proc/loadavg)"
    done
  done
done
echo "# done loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
