#!/bin/bash
# Time one XIVO run and report the estimator's own per-component means.
#
# Usage: fps_one.sh <arm> <worktree> <cfg-name> <seq> [lib-suffix]
#   arm        label for the output line
#   worktree   directory under the workspace, e.g. xivo-efficiency or xivo
#   cfg-name   config basename under <worktree>/cfg (no .json), or a path
#              containing a '/' which is passed through verbatim -- so the same
#              config file can drive a baseline worktree without adding untracked
#              files to it
#   seq        room1 .. room6
#   lib-suffix optional XIVO_OUTPUT_SUFFIX of the build to run, e.g. `_none`.
#              Empty means plain lib/. This is how a *flag* experiment is timed:
#              several suffixed builds of one unchanged source tree, so the arms
#              differ in nothing but the compiler flags (harness/build_variant.sh).
#
# Single-threaded and seed-pinned so repeats are comparable: an unpinned process
# spawns ~255 OpenCV/OpenMP threads for a 6% wall-clock gain, and a batch of them
# measures memory bandwidth rather than the estimator. Output is bit-identical
# pinned (see notes-n-prompts/notes-stereo/cost-and-throughput.md).
#
# `-mode runOnly` keeps the Python saver out of the loop, so the wall clock is
# PNG decode + the Python feed loop + the estimator, and nothing else.
set -u
ARM=$1; WT=$2; CFG=$3; SEQ=$4; LIBSUF=${5:-}
W=/home/ubuntu/workspace/auto-slam-engineer
T=${FPS_TMP:-/tmp/xivo_fps}
mkdir -p "$T"
LOG=$T/fps_${ARM}_${SEQ}_$$.log

case "$CFG" in
  */*) CFG_PATH="$CFG" ;;
  *)   CFG_PATH="cfg/$CFG.json" ;;
esac

cd "$W/$WT" || exit 1
export OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 XIVO_RANDOM_SEED=0 XIVO_LIB="$W/$WT/lib$LIBSUF"

/usr/bin/time -f "%e %U %S %M" -o "$T/tm_$$.txt" \
  setarch -R "$W/dependencies/venv/bin/python3" scripts/pyxivo.py \
    -root "$W/data/tumvi" -dataset tumvi -seq "$SEQ" -cfg "$CFG_PATH" \
    -mode runOnly \
    > "$LOG" 2>&1
rc=$?
read -r WALL USR SYS MAXRSS < "$T/tm_$$.txt"; rm -f "$T/tm_$$.txt"
if [ $rc -ne 0 ]; then
  echo "RESULT $ARM $SEQ FAILED rc=$rc log=$LOG"; exit 1
fi

# The estimator prints a cumulative total-ms/occurrences block every 50 frames
# (common/timer.h), so the *last* block is the whole-run mean per occurrence,
# not the last 50 frames.
python3 - "$LOG" "$ARM" "$SEQ" "$WALL" "$USR" "$MAXRSS" <<'EOF'
import re, sys
log, arm, seq, wall, usr, rss = sys.argv[1:]
txt = open(log, errors='replace').read()
blocks = txt.split('.....\n')
d = {}
for line in blocks[-1].splitlines():
    m = re.match(r'\[estimator\](\S+):([\d.]+) ms', line)
    if m:
        d[m.group(1)] = float(m.group(2))
nframes = len(re.findall(r'\.\.\.\.\.\n', txt)) * 50 or float('nan')
get = lambda k: d.get(k, float('nan'))
print('RESULT %s %s wall=%s user=%s rss_kb=%s visual_meas=%.3f track=%.3f '
      'process_tracks=%.3f update=%.3f jacobian=%.3f mh=%.3f actual_update=%.3f '
      'stereo_gating=%.3f oos_jac=%.3f propagation=%.4f'
      % (arm, seq, wall, usr, rss, get('visual-meas'), get('track'),
         get('process-tracks'), get('update'), get('jacobian'), get('MH-gating'),
         get('actual-update'), get('stereo-gating'), get('oos-jacobian'),
         get('propagation')))
EOF
rm -f "$LOG"
