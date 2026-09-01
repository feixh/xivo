#!/bin/bash
# Screening pass for the orientation work: mono only, all six rooms, 6-member
# jitter ensemble (36 runs, ~5 min on cpus 0-59). Same harness as the headline
# runs, just half of it -- the mono/stereo pair moves together on everything
# tried so far, so mono alone is enough to decide whether a candidate is worth a
# full both-modes confirmation.
#
#   screen.sh <tag>
#
# Reads cfg/eff_mono.json out of the worktree as it stands, so patch the config
# (or rebuild) first. Prints the 6-room means next to the M1 reference.
set -euo pipefail
WS=/home/ubuntu/workspace/auto-slam-engineer
tag="$1"
CPU_BASE=0 CPU_SPAN=60 "$WS/experiments/openvins/run_xivo_reference.sh" \
  --worktree xivo-orient --mode mono --out "$WS/experiments/results/orient_$tag" \
  --jitter 6 > "/tmp/orient_$tag.log" 2>&1
"$WS/notes-n-prompts/notes-orient/screen_report.py" "$tag"
