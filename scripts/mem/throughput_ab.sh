#!/bin/bash
# Alternate two `bin/vio` builds over the same sequence and report wall clock,
# CPU time, peak RSS and page faults for each. Used for the M5 no-regression
# check: the fixes must not cost throughput on the mono+IMU path.
#
#   scripts/mem/throughput_ab.sh <tree-a> <tree-b> [reps]
#
# <tree-a>/<tree-b> are XIVO source trees that each contain bin/vio and cfg/.
# Runs are strictly sequential and interleaved (a, b, a, b, ...) so that a slow
# drift in machine load hits both builds equally.
#
# The peak_MB column is /usr/bin/time's ru_maxrss, which reads ~25% low here
# (61 MB against the kernel's own VmHWM of 80 MB for the same run). It is
# comparable between A and B but is not the process's peak RSS -- use
# scripts/mem/rss_profile.sh, which samples /proc, for absolute numbers.
#
# `vio` runs at >1000% CPU (OpenCV's internal threading), so user/sys time swing
# by ~15% run to run; wall clock over several reps is the metric to read.
#
# Env:
#   SEQ=room1                 sequence to run
#   CFGS="vio_tumvi vio_tumvi_nodesc"
#   ROOT=<dataset root>       default ../data/tumvi relative to <tree-a>
#   OUT=/tmp/xivo-throughput  where the raw /usr/bin/time output goes
set -uo pipefail

if [ $# -lt 2 ]; then sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 1; fi

A="$(cd "$1" && pwd)"; B="$(cd "$2" && pwd)"; REPS="${3:-3}"
SEQ="${SEQ:-room1}"
CFGS="${CFGS:-vio_tumvi vio_tumvi_nodesc}"
ROOT="${ROOT:-$(cd "$A/../data/tumvi" && pwd)}"
OUT="${OUT:-/tmp/xivo-throughput}"

mkdir -p "$OUT"

one_run() {  # tree label cfg rep
  local tree="$1" label="$2" cfg="$3" rep="$4"
  local stem="$OUT/${label}_${cfg}_${rep}"
  ( cd "$tree" && XIVO_RANDOM_SEED=0 /usr/bin/time -v -o "$stem.time" \
      ./bin/vio -cfg "cfg/$cfg.json" -root "$ROOT" -seq "$SEQ" \
      -out "$stem.traj" > "$stem.log" 2>&1 )
  awk -v label="$label" -v cfg="$cfg" -v rep="$rep" '
    /Elapsed \(wall clock\)/ { split($NF, t, ":"); wall = (length(t)==3 ? t[1]*3600+t[2]*60+t[3] : t[1]*60+t[2]) }
    /User time/     { user = $NF }
    /System time/   { sys  = $NF }
    /Maximum resident/ { rss = $NF }
    /Minor \(reclaiming a frame\) page faults/ { mf = $NF }
    END { printf "%-6s %-18s %-4s %8.2f %8.2f %8.2f %9.1f %12d\n",
                 label, cfg, rep, wall, user, sys, rss/1024, mf }' "$stem.time"
}

printf '%-6s %-18s %-4s %8s %8s %8s %9s %12s\n' \
       build cfg rep wall_s user_s sys_s peak_MB minflt
for cfg in $CFGS; do
  for rep in $(seq 1 "$REPS"); do
    one_run "$A" A "$cfg" "$rep"
    one_run "$B" B "$cfg" "$rep"
  done
done

echo "A = $A"
echo "B = $B"
echo "raw /usr/bin/time output in $OUT"
