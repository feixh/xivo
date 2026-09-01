#!/bin/bash
# Sweep `async_queue_limit` on one sequence: how deep does the producer/consumer
# queue have to be before the overlap saturates, and what does the depth cost in
# RSS?
#
# Usage: async_depth_sweep.sh <seq> <cfg-basename-with-async_run-true> [depths...]
#   e.g. async_depth_sweep.sh room4 tumvi_stereo_oos_async 11 24 48 128 512
#
# Why a sweep and not one number: the depth is bounded below by the message
# buffer's reordering window (MAX_SIZE = 10, and the limit must exceed it or
# producer and worker deadlock), and at TUM-VI's ~10 IMU messages per image the
# producer cannot get a whole frame ahead until the queue can hold more than ~11
# messages. So the interesting range starts right at the floor. Above it, each
# extra queued image costs two 512x512 cv::Mats, and -- because the eval savers
# poll estimator state from Python right after pushing -- also adds that many
# messages of lag to any dumped trajectory. Depth is a throughput/memory/fidelity
# knob, not a free parameter.
set -uo pipefail

SEQ=$1
CFG=$2
shift 2
DEPTHS=("$@")
[ ${#DEPTHS[@]} -eq 0 ] && DEPTHS=(11 24 48 128 512)

W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WT=${WT:-xivo-async}
OUT=${OUT_DIR:-/tmp/async_depth}
mkdir -p "$OUT"
cd "$W/$WT"
export OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 XIVO_EIGEN_INIT=${XIVO_EIGEN_INIT:-none}

for d in "${DEPTHS[@]}"; do
  tmp=$OUT/cfg_${d}.json
  "$W/dependencies/venv/bin/python3" - "cfg/$CFG.json" "$tmp" "$d" <<'EOF'
import sys
src, dst, depth = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = open(src).read()
j = s.index('"async_queue_limit"')
k = s.index('\n', j)
open(dst, 'w').write(s[:j] + '"async_queue_limit": %d,' % depth + s[k:])
EOF
  log=$OUT/${SEQ}_d${d}.log
  /usr/bin/time -v "$W/dependencies/venv/bin/python3" scripts/pyxivo.py \
      -cfg "$tmp" -root "$W/data/tumvi" -dataset tumvi -seq "$SEQ" \
      -cam_id 0 -mode runOnly > "$log" 2>&1
  rc=$?
  wall=$(awk -F': ' '/Elapsed \(wall clock\)/{print $NF}' "$log" |
         awk -F: '{if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2}')
  user=$(awk -F': ' '/User time/{print $NF}' "$log")
  sys=$(awk -F': ' '/System time/{print $NF}' "$log")
  rss=$(awk -F': ' '/Maximum resident set size/{print $NF}' "$log")
  frames=$(awk '/^stereo: [0-9]+ frames,/{print $2; exit}' "$log")
  matched=$(awk '/^stereo: [0-9]+ frames,/{for(i=1;i<=NF;i++) if($i=="matched") print $(i-1); exit}' "$log")
  printf 'DEPTH %-4s %-12s rc=%d frames=%s wall=%.1f fps=%.1f cpu=%.2f rss_mb=%.0f matched=%s\n' \
    "$d" "$SEQ" "$rc" "${frames:--}" "$wall" \
    "$(awk -v f="${frames:-0}" -v w="$wall" 'BEGIN{print (w>0)?f/w:0}')" \
    "$(awk -v u="$user" -v s="$sys" -v w="$wall" 'BEGIN{print (u+s)/w}')" \
    "$(awk -v r="$rss" 'BEGIN{print r/1024}')" "${matched:--}"
done
