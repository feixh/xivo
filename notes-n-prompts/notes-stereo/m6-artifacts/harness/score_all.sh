#!/bin/bash
# Re-score every existing dump (no estimator re-runs) into RESULT lines.
cd /home/ubuntu/workspace/auto-slam-engineer/xivo
PY=../dependencies/venv/bin/python3
T=/home/ubuntu/.claude/jobs/041e1899/tmp
score() {
  D=$1; TAG=$2; SEQ=$3
  A1=$($PY scripts/tum_rgbd_benchmark_tools/evaluate_ate.py --max_difference 0.001 --verbose $D/tumvi_${SEQ}_gt $D/tumvi_${SEQ}_cam0 2>/dev/null | awk '/absolute_translational_error.rmse/{print $2}')
  A2=$($PY scripts/tum_rgbd_benchmark_tools/evaluate_ate.py --max_difference 0.02 --verbose $D/tumvi_${SEQ}_gt $D/tumvi_${SEQ}_cam0 2>/dev/null | awk '/absolute_translational_error.rmse/{print $2}')
  R=$($PY scripts/tum_rgbd_benchmark_tools/evaluate_rpe.py --fixed_delta --delta_unit s --delta 1 --verbose $D/tumvi_${SEQ}_gt $D/tumvi_${SEQ}_cam0 2>/dev/null | awk '/rotational_error.rmse/{printf "%s ",$2} /^translational_error.rmse/{printf "%s ",$2}')
  echo "RESULT $TAG $SEQ $A1 $A2 $R"
}
for D in $T/run_*_room*; do
  b=$(basename $D); rest=${b#run_}; TAG=${rest%_room*}; SEQ=room${rest##*_room}
  score $D $TAG $SEQ &
done
wait
