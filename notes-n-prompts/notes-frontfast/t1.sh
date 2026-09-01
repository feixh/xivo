#!/bin/bash
# One-sequence timing probe. Usage: t1.sh WORKTREE MODE SEQ CPU [CFG]
# Prints the last print_timing block and the wall FPS.
set -euo pipefail
W=$1; MODE=$2; SEQ=$3; CPU=$4; CFG=${5:-cfg/eff_$MODE.json}
WS=/home/ubuntu/workspace/auto-slam-engineer
X=$WS/$W
export PATH="$WS/dependencies/venv/bin:$PATH"
export PYTHONPATH="$X/lib${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$X"
n=$(grep -c '^[0-9]' "$WS/data/tumvi/dataset-${SEQ}_512_16/mav0/cam0/data.csv")
out=$(mktemp -d)
XIVO_RANDOM_SEED=0 /usr/bin/time -f "%e %U %S %M" -o $out/time.txt \
  taskset -c "$CPU" setarch -R python3 scripts/pyxivo.py -root "$WS/data/tumvi" \
  -dataset tumvi -seq "$SEQ" -cam_id 0 -cfg "$CFG" -dump "$out/dump" -mode runOnly \
  > "$out/run.log" 2>&1
read -r wall usr sys maxrss < $out/time.txt
echo "### $W $MODE $SEQ cpu$CPU cfg=$CFG"
awk '/visual-meas|track|process-tracks|actual-update|jacobian|MH-gating|stereo-gating|^update|imu/' "$out/run.log" | tail -30
python3 -c "print('FPS=%.2f  wall=%.2fs  frames=%d  ms/frame=%.3f  peakRSS=%.1fMB' % ($n/$wall, $wall, $n, 1000*$wall/$n, $maxrss/1024.0))"
echo "logdir=$out"
