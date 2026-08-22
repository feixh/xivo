#!/bin/bash
# Sample the resident set size of a command while it runs and report the growth
# trend. Catches the leak class LeakSanitizer is blind to: memory that keeps
# accumulating for the whole run and is only released at teardown (a container
# that is never cleared, a pooled object that is never fully reset).
#
# Usage:
#   scripts/mem/rss_profile.sh <csv-out> <command> [args ...]
#
# Example:
#   scripts/mem/rss_profile.sh /tmp/rss_room1.csv \
#     bin/vio -cfg cfg/vio_tumvi.json -root ../data/tumvi/ -seq room1 -out /tmp/t
#
# Env: INTERVAL=<seconds between samples, default 0.25>
#
# Reports RSS at the start, the peak, the end, and a least-squares slope in
# kB/s fitted over the second half of the run (the first half includes
# start-up and the pool warming up, which is legitimate growth). A leak-free
# run has a slope that is a rounding error next to its own peak.
set -uo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,18p' "${BASH_SOURCE[0]}"
  exit 1
fi

CSV="$1"; shift
INTERVAL="${INTERVAL:-0.25}"

: > "$CSV"
echo "t_s,rss_kb" >> "$CSV"

"$@" &
pid=$!

start=$(date +%s.%N)
while kill -0 "$pid" 2>/dev/null; do
  rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null)
  [ -n "${rss:-}" ] && \
    echo "$(echo "$(date +%s.%N) - $start" | bc),$rss" >> "$CSV"
  sleep "$INTERVAL"
done
wait "$pid"
status=$?

awk -F, 'NR>1 {n++; t[n]=$1; r[n]=$2; if ($2>peak) peak=$2}
  END {
    if (n < 4) { print "too few samples (" n ")"; exit }
    half = int(n/2)
    for (i = half; i <= n; i++) { sx+=t[i]; sy+=r[i]; sxy+=t[i]*r[i]; sxx+=t[i]*t[i]; m++ }
    slope = (m*sxy - sx*sy) / (m*sxx - sx*sx)
    printf "samples=%d  duration=%.1fs\n", n, t[n]
    printf "rss  start=%.1f MB  peak=%.1f MB  end=%.1f MB\n", r[1]/1024, peak/1024, r[n]/1024
    printf "slope over 2nd half = %.1f kB/s  (%.2f MB over the sampled window)\n", \
           slope, slope*(t[n]-t[half])/1024
  }' "$CSV"

echo "samples in $CSV"
exit $status
