#!/bin/bash
# A/B one sequence between async_run false and true, same binary, same config
# except the flag.
#
# Usage: async_ab.sh <seq> <cfg-basename> [worktree]
#   e.g. async_ab.sh room4 tumvi_stereo_oos xivo-async
# Expects <cfg>_async.json to sit next to <cfg>.json.
#
# Runs `-mode runOnly` in BOTH arms. Not a convenience: the eval/dump savers poll
# estimator state from Python immediately after pushing each image
# (scripts/pyxivo.py, `saver.onVisionUpdate`), and under async_run that state
# belongs to whatever frame the worker has reached, not the one just pushed. The
# dumped trajectory is therefore lagged by construction and cannot be compared
# between arms. Equivalence is checked instead on the estimator's own counters,
# which are printed after the queue is drained.
#
# What it records and why each column is needed:
#
#   wall / user   `async_run: true` starts a worker thread inside the Estimator
#                 (estimator.cpp, `Run`) that drains the message buffer while the
#                 Python feed loop keeps pushing. user/wall > 1 is the *only*
#                 direct proof that two threads did work concurrently; the whole
#                 point of the A/B is that the sync arm sits at 0.99.
#   maxrss        `async_queue_limit` bounds the queue, but each queued Visual
#                 message still owns two cv::Mats, so async trades memory for
#                 overlap. This is the size of that trade.
#   frames        stereo pairs the ESTIMATOR processed, from `print_stereo_stats`
#                 -- not pairs fed. Before backpressure and `Wait()` existed, the
#                 async arm looked 3x faster while silently processing 831 of
#                 2228 frames. If this differs between arms, the wall-clock
#                 comparison is meaningless, so it is checked first.
#   matched, oos  deterministic per-run totals that depend on every update the
#                 filter did. They stand in for the trajectory md5: identical
#                 counters across arms means the worker executed the same
#                 messages in the same order.
#
# Threads are pinned as everywhere else here, which matters more than usual: it
# keeps OpenCV's pool from confounding user/wall, the one number being measured.
set -uo pipefail

SEQ=$1
CFG=$2
WT=${3:-xivo-async}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT=${OUT_DIR:-/tmp/async_ab}
mkdir -p "$OUT"

cd "$W/$WT"
export OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 XIVO_EIGEN_INIT=${XIVO_EIGEN_INIT:-none}

fed=$(grep -vc '^#' "$W/data/tumvi/dataset-${SEQ}_512_16/mav0/cam0/data.csv")

for arm in sync async; do
  cfg=$CFG; [ "$arm" = async ] && cfg=${CFG}_async
  log=$OUT/${SEQ}_${arm}.log
  /usr/bin/time -v "$W/dependencies/venv/bin/python3" scripts/pyxivo.py \
      -cfg "cfg/$cfg.json" -root "$W/data/tumvi" -dataset tumvi -seq "$SEQ" \
      -cam_id 0 -mode runOnly > "$log" 2>&1
  rc=$?

  wall=$(awk -F': ' '/Elapsed \(wall clock\)/{print $NF}' "$log" |
         awk -F: '{if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2}')
  user=$(awk -F': ' '/User time/{print $NF}' "$log")
  sys=$(awk -F': ' '/System time/{print $NF}' "$log")
  rss=$(awk -F': ' '/Maximum resident set size/{print $NF}' "$log")
  # "stereo: N frames, M match attempts, K matched (x%)". The `frames,` anchor
  # matters: the stereo *loader* also prints a line starting "stereo: ", with the
  # count of pairs FED, which is exactly the number this column exists to not be.
  frames=$(awk '/^stereo: [0-9]+ frames,/{print $2; exit}' "$log")
  matched=$(awk '/^stereo: [0-9]+ frames,/{for(i=1;i<=NF;i++) if($i=="matched") print $(i-1); exit}' "$log")
  oos=$(awk '/^candidates=/{for(i=1;i<=NF;i++) if($i ~ /^used=/){split($i,a,"="); print a[2]}; exit}' "$log")

  printf 'ARM %-5s %-12s rc=%d fed=%d frames=%s wall=%.1f fps=%.1f user=%s sys=%s cpu=%.2f rss_mb=%.0f matched=%s oos_used=%s\n' \
    "$arm" "$SEQ" "$rc" "$fed" "${frames:--}" "$wall" \
    "$(awk -v f="${frames:-0}" -v w="$wall" 'BEGIN{print (w>0)?f/w:0}')" \
    "$user" "$sys" \
    "$(awk -v u="$user" -v s="$sys" -v w="$wall" 'BEGIN{print (u+s)/w}')" \
    "$(awk -v r="$rss" 'BEGIN{print r/1024}')" "${matched:--}" "${oos:--}"
done
