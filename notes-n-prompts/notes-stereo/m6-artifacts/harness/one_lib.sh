#!/bin/bash
# one.sh, but against a variant build's lib directory (different EKF state size).
#
# Threads are pinned to 1. Each pyxivo process otherwise spawns an OpenCV/OpenMP
# pool sized to the whole machine (~255 threads on 192 cores) to gain 6% of wall
# clock at 708% CPU -- which makes a batch of 30 thrash at load 5000. Output is
# bit-identical either way (verified on room4), so pinning costs nothing.
cd /home/ubuntu/workspace/auto-slam-engineer/xivo
CFG=$1; SEQ=$2; TAG=$3; LIB=$4
D=/home/ubuntu/.claude/jobs/041e1899/tmp/run_${TAG}_${SEQ}
PY=../dependencies/venv/bin/python3
export OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1
S=$(XIVO_LIB=$LIB XIVO_RANDOM_SEED=0 setarch -R $PY scripts/pyxivo.py -root ../data/tumvi -dataset tumvi -seq $SEQ -cfg $CFG -mode eval -dump $D 2>&1 | grep "^stereo_init:" | head -1)
A1=$($PY scripts/tum_rgbd_benchmark_tools/evaluate_ate.py --max_difference 0.001 --verbose $D/tumvi_${SEQ}_gt $D/tumvi_${SEQ}_cam0 2>/dev/null | awk '/absolute_translational_error.rmse/{print $2}')
A2=$($PY scripts/tum_rgbd_benchmark_tools/evaluate_ate.py --max_difference 0.02 --verbose $D/tumvi_${SEQ}_gt $D/tumvi_${SEQ}_cam0 2>/dev/null | awk '/absolute_translational_error.rmse/{print $2}')
R=$($PY scripts/tum_rgbd_benchmark_tools/evaluate_rpe.py --fixed_delta --delta_unit s --delta 1 --verbose $D/tumvi_${SEQ}_gt $D/tumvi_${SEQ}_cam0 2>/dev/null | awk '/rotational_error.rmse/{printf "%s ",$2} /^translational_error.rmse/{printf "%s ",$2}')
echo "RESULT $TAG $SEQ $A1 $A2 $R| $S"
