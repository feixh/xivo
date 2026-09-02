#!/bin/bash
# M6 final evaluation: the two XIVO operating points, at n=10 on all eleven
# sequences, both modes.
#
# Why n=10 and not the n=3 the tuning screens used. The M5 screen scored the
# shipped config at 0.098 m; extending the same ensemble to six members moved it
# to 0.102 with r0-r2 bit-identical, i.e. the 0.098 was three lucky draws, not a
# different configuration. V2_03_difficult alone carries +-0.052 m of member
# spread, which puts ~0.002 m of standard error on the eleven-sequence mean at
# n=6 -- the same order as the difference this pass has to resolve. At n=10 the
# standard error of the acc-vs-fast difference is ~0.0031, so the 0.007 m gap
# lands at 2.3 sigma instead of 1.8.
#
# Why two arms rather than one. The front end tuned in M5 is not a free win: it
# buys 3.3 ms/frame for 0.007 m of ATE, and neither end of that trade dominates
# OpenVINS (which is both faster and, on ate_002, slightly more accurate). The
# shipped configuration has to be one of them, so both get measured at the same
# n and on the same eleven sequences, and the choice is made from that table
# rather than from the n=3 screens.
#
#   fast  the shipped cfg/euroc_{mono,stereo}.json as committed in M5:
#         histogram_method NONE, FAST.threshold 7, KLT.max_level 2,
#         fast_png_decode false.  11.588 ms/frame one-core over all eleven.
#   acc   the same, with CLAHE restored and FAST.threshold back to 20 -- i.e.
#         M4's front end plus M5's two free wins (klt_max_level 2, which also
#         saves 4.7 MB, and dropping the slower built-in PNG decoder, which is
#         bit-identical).  ~14.0 ms/frame on the three-sequence screen.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT="${OUT:-../results/euroc_m6_final}"
MEMBERS="${MEMBERS:-10}"
WT="${WORKTREE:-xivo-eurocfps}"
SEQS="${SEQS:-MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult}"

echo "m6 final: worktree=$WT members=$MEMBERS out=$OUT"

run_arm() {
  local name="$1"; shift
  local args=()
  for p in "$@"; do args+=(--patch "$p"); done
  CPU_BASE="${CPU_BASE:-0}" CPU_SPAN="${CPU_SPAN:-60}" \
    ./sweep_xivo.sh --name "$name" --members "$MEMBERS" --mode both \
    --worktree "$WT" --seqs "$SEQS" --out "$OUT" "${args[@]+"${args[@]}"}"
}

run_arm fast
run_arm acc 'tracker_cfg.histogram_method="CLAHE"' 'tracker_cfg.FAST.threshold=20'

echo "M6_FINAL_DONE rc=0"
