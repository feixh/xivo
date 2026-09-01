#!/bin/bash
# Fetch TUM-VI 512x16 sequences into data/tumvi/. Per sequence: download the tar,
# extract it, delete the tar. Tars are not kept -- the full 28-sequence set is
# ~190 GB and the extraction is the same bytes again.
#
# Usage: fetch_tumvi.sh [n-parallel] [seq ...]
#   With no sequence list, every 512_16 sequence is considered. A sequence is
#   fetched only if it is *incomplete*: `complete` requires both cameras' data.csv
#   and an image count per camera that matches its own csv. That is the check to
#   use rather than "the directory exists" -- an interrupted extraction leaves a
#   directory with a few hundred images in it.
#
# Two things this script deliberately does not do, both learned the hard way:
#
#   - it does not pass `curl -C -` blindly. Against an already-complete file curl
#     asks for a range starting at EOF, this server answers 200 with the whole
#     body, and curl *appends* it: a 1.6 GB room tar became 3.2 GB. `tar` still
#     extracted most of it (it stops at the end-of-archive marker) but the
#     extraction was cut short, leaving directories that were full by file count
#     with one truncated PNG inside -- four sequences had to be re-fetched, and
#     only harness/check_tumvi.py saw it. So resume is decided by comparing the
#     local size against Content-Length first.
#   - it does not use `xargs -P` with an exported shell function. The pool below
#     is plain `wait -n`, so the worker inherits the environment by the ordinary
#     rules and there is nothing to be subtle about.
set -u
ROOT=/home/ubuntu/workspace/auto-slam-engineer/data/tumvi
BASE=https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16
TMP=$ROOT/archives

ALL="corridor1 corridor2 corridor3 corridor4 corridor5
     magistrale1 magistrale2 magistrale3 magistrale4 magistrale5 magistrale6
     outdoors1 outdoors2 outdoors3 outdoors4 outdoors5 outdoors6 outdoors7 outdoors8
     room1 room2 room3 room4 room5 room6
     slides1 slides2 slides3"

NPAR=${1:-8}
shift || true
SEQS="${*:-$ALL}"

complete () {
  local d=$ROOT/dataset-${1}_512_16 c n m
  for c in cam0 cam1; do
    [ -f "$d/mav0/$c/data.csv" ] || return 1
    n=$(grep -vc '^#' "$d/mav0/$c/data.csv")
    m=$(ls "$d/mav0/$c/data" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] && [ "$n" -eq "$m" ] || return 1
  done
  [ -f "$d/mav0/imu0/data.csv" ] && [ -f "$d/mav0/mocap0/data.csv" ]
}

one () {
  local s=$1 d=$ROOT/dataset-${s}_512_16 t=$TMP/dataset-${s}_512_16.tar
  local url=$BASE/dataset-${s}_512_16.tar remote have resume=""
  remote=$(curl -sIL "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  have=$( [ -f "$t" ] && stat -c %s "$t" || echo 0 )
  if [ -n "$remote" ] && [ "$have" -gt 0 ]; then
    if   [ "$have" -lt "$remote" ]; then resume="-C -"      # genuine partial
    elif [ "$have" -gt "$remote" ]; then rm -f "$t"         # the append bug, above
    else resume="skip"; fi                                  # already whole
  elif [ "$have" -gt 0 ]; then
    rm -f "$t"                     # no Content-Length to check against: start over
  fi
  if [ "$resume" = skip ]; then
    echo "HAVE  $s (tar already complete)"
  else
    echo "GET   $s${resume:+ (resuming at $have/$remote)}"
    if ! curl -sSL --retry 10 --retry-delay 15 $resume -o "$t" "$url"; then
      echo "FAIL  $s (download)"; return 1
    fi
  fi
  [ -d "$d" ] && rm -rf "$d"     # an interrupted extraction, replaced wholesale
  echo "UNTAR $s ($(du -h "$t" | cut -f1))"
  if ! tar -xf "$t" -C "$ROOT"; then echo "FAIL  $s (untar)"; rm -f "$t"; return 1; fi
  rm -f "$t"
  if complete "$s"; then
    echo "OK    $s $(ls "$d/mav0/cam0/data" | wc -l) stereo pairs"
  else
    echo "FAIL  $s (incomplete after untar)"; return 1
  fi
}

mkdir -p "$TMP"
# FORCE=1 re-fetches even a sequence that passes `complete`. Needed because
# `complete` counts files and does not read them: harness/check_tumvi.py verifies
# PNG framing and is what catches a truncated image inside a full-looking
# directory. Pass the sequences it named.
todo=""
for s in $SEQS; do
  if [ -z "${FORCE:-}" ] && complete "$s"; then echo "SKIP  $s (complete)"
  else todo="$todo $s"; fi
done
echo "TODO ($(echo $todo | wc -w)):$todo"

n=0
for s in $todo; do
  one "$s" &
  n=$((n + 1))
  # Keep exactly NPAR workers busy: as soon as one finishes, start the next.
  if [ "$n" -ge "$NPAR" ]; then wait -n; n=$((n - 1)); fi
done
wait
echo "ALL DONE"
