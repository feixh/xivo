#!/bin/bash
# Sample one pyxivo run's resident set twice a second.
#
# Usage: rss_probe.sh <seq> <cfg-basename> <tag> [worktree]
# Writes /tmp/rss_<tag>.rss  (columns: epoch_s VmRSS_kB VmHWM_kB)
#    and /tmp/rss_<tag>.log  (the run's own output)
#
# Both fields matter and they answer different questions:
#
#   VmRSS is instantaneous, so it gives the steady-state footprint -- but at any
#     sampling interval it can alias away a short burst. The finding this script
#     exists for is a ~350 MB allocation inside the OOS update that lives for
#     1.5 s and hands its pages straight back to the OS; a 2-second sampler
#     caught it at 226 MB and a 0.5-second one at 376 MB.
#   VmHWM is the kernel's own high-water mark, so it never misses a burst. It is
#     what GNU time reports as ru_maxrss, and it is the number to size a machine
#     by. VmHWM > VmRSS at every poll is itself the signal that a transient
#     happened and was released.
#
# Threads are pinned for the same reason every other harness here pins them (see
# notes-efficiency/harness/run_full_tumvi.sh); it also keeps two concurrent
# probes from perturbing each other.
set -uo pipefail

SEQ=$1
CFG=$2
TAG=$3
WT=${4:-xivo}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT=/tmp/rss_${TAG}

cd "$W/$WT"
export OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 XIVO_EIGEN_INIT=${XIVO_EIGEN_INIT:-none}

"$W/dependencies/venv/bin/python3" scripts/pyxivo.py -cfg "cfg/$CFG.json" \
    -root "$W/data/tumvi" -dataset tumvi -seq "$SEQ" -cam_id 0 \
    -mode runOnly > "$OUT.log" 2>&1 &
PID=$!

: > "$OUT.rss"
while kill -0 $PID 2>/dev/null; do
  # One read of /proc/<pid>/status per sample; awk rather than grep so that the
  # two fields come out in a fixed order even though /proc lists HWM first.
  awk -v t="$(date +%s)" '
    /^VmRSS:/ {rss=$2} /^VmHWM:/ {hwm=$2}
    END {if (rss != "") print t, rss, hwm}' "/proc/$PID/status" >> "$OUT.rss" 2>/dev/null
  sleep 0.5
done
wait $PID
rc=$?

awk '{if ($2>mr) mr=$2; if ($3>mh) mh=$3; last=$2; n++}
     END {printf "%s: %d samples  steady(last) %.0f MB  max VmRSS %.0f MB  VmHWM %.0f MB\n",
                 "'"$TAG"'", n, last/1024, mr/1024, mh/1024}' "$OUT.rss"
echo "$TAG rc=$rc"
exit $rc
