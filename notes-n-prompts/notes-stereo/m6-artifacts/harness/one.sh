#!/bin/bash
# run one (cfg, seq); print "TAG SEQ ate001 ate02 rpe_rot rpe_tra | seed stats"
cd /home/ubuntu/workspace/auto-slam-engineer/xivo
CFG=$1; SEQ=$2; TAG=$3
D=/home/ubuntu/.claude/jobs/041e1899/tmp/run_${TAG}_${SEQ}
PY=../dependencies/venv/bin/python3
S=$(XIVO_RANDOM_SEED=0 setarch -R $PY scripts/pyxivo.py -root ../data/tumvi -dataset tumvi -seq $SEQ -cfg $CFG -mode eval -dump $D 2>&1 | grep "^stereo_init:" | head -1)
A1=$($PY scripts/tum_rgbd_benchmark_tools/evaluate_ate.py --max_difference 0.001 --verbose $D/tumvi_${SEQ}_gt $D/tumvi_${SEQ}_cam0 2>/dev/null | awk '/absolute_translational_error.rmse/{print $2}')
A2=$($PY scripts/tum_rgbd_benchmark_tools/evaluate_ate.py --max_difference 0.02 --verbose $D/tumvi_${SEQ}_gt $D/tumvi_${SEQ}_cam0 2>/dev/null | awk '/absolute_translational_error.rmse/{print $2}')
R=$($PY scripts/tum_rgbd_benchmark_tools/evaluate_rpe.py --fixed_delta --delta_unit s --delta 1 --verbose $D/tumvi_${SEQ}_gt $D/tumvi_${SEQ}_cam0 2>/dev/null | awk '/rotational_error.rmse/{printf "%s ",$2} /^translational_error.rmse/{printf "%s ",$2}')
echo "RESULT $TAG $SEQ $A1 $A2 $R| $S"
