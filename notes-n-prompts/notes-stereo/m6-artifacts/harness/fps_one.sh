#!/bin/bash
# Time one XIVO run. Usage: fps_one.sh <arm> <cfg> <lib> <seq>
# Single-threaded and seed-pinned, so repeats are comparable; see the notes on
# why pinning is required (each unpinned process spawns ~255 OpenCV/OpenMP threads).
set -u
ARM=$1; CFG=$2; LIB=$3; SEQ=$4
W=/home/ubuntu/workspace/auto-slam-engineer
T=/home/ubuntu/.claude/jobs/041e1899/tmp
LOG=$T/fps_${ARM}_${SEQ}_$$.log

cd $W/xivo
export OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 XIVO_RANDOM_SEED=0 XIVO_LIB=$LIB

/usr/bin/time -f "%e %U %S %M" -o $T/tm_$$.txt \
  setarch -R $W/dependencies/venv/bin/python3 scripts/pyxivo.py \
    -root $W/data/tumvi -dataset tumvi -seq $SEQ -cfg $CFG -mode runOnly \
    > $LOG 2>&1
rc=$?
read WALL USR SYS MAXRSS < $T/tm_$$.txt; rm -f $T/tm_$$.txt
if [ $rc -ne 0 ]; then echo "RESULT $ARM $SEQ FAILED rc=$rc log=$LOG"; exit 1; fi

# Last timing block printed by the estimator = cumulative mean per occurrence.
python3 - "$LOG" "$ARM" "$SEQ" "$WALL" "$USR" "$MAXRSS" <<'EOF'
import re, sys
log, arm, seq, wall, usr, rss = sys.argv[1:]
txt = open(log, errors='replace').read()
blocks = txt.split('.....\n')
d = {}
for line in blocks[-1].splitlines():
    m = re.match(r'\[estimator\](\S+):([\d.]+) ms', line)
    if m: d[m.group(1)] = float(m.group(2))
get = lambda k: d.get(k, float('nan'))
print('RESULT %s %s wall=%s user=%s rss_kb=%s visual_meas=%.3f track=%.3f '
      'process_tracks=%.3f update=%.3f jacobian=%.3f mh=%.3f actual_update=%.3f '
      'stereo_gating=%.3f propagation=%.4f'
      % (arm, seq, wall, usr, rss, get('visual-meas'), get('track'),
         get('process-tracks'), get('update'), get('jacobian'), get('MH-gating'),
         get('actual-update'), get('stereo-gating'), get('propagation')))
EOF
