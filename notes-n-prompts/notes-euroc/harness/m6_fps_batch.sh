#!/bin/bash
# M6: one-core full-eleven throughput for both operating points, measured in a
# single session so the two numbers share machine state. `fast` duplicates
# results/euroc_fps_ship11 on purpose -- it is the control that says this
# session's timings are comparable to M5's.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQS="MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult"
OUT="${OUT:-../results/euroc_m6_fps}"
CPU_BASE=0 ./sweep_fps.sh --name fast --seqs "$SEQS" --repeats 3 --out "$OUT"
CPU_BASE=0 ./sweep_fps.sh --name acc  --seqs "$SEQS" --repeats 3 --out "$OUT" \
  --patch 'tracker_cfg.histogram_method="CLAHE"' --patch 'tracker_cfg.FAST.threshold=20'
echo "M6_FPS_DONE rc=0"
