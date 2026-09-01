#!/bin/bash
# Sequential timing matrix: 2x2 (mono/stereo) x (upstream 30/60, shipped 90/180).
# Runs are strictly sequential -- concurrency would make the numbers meaningless --
# and arms are interleaved across repeats so machine-load drift hits every arm.
T=/home/ubuntu/.claude/jobs/041e1899/tmp
for rep in 1 2; do
  for seq in room1 room6; do
    for arm in m30 s30 m90 s90; do
      case $arm in
        m30) cfg=$T/t_m30.json; lib=lib_f30 ;;
        s30) cfg=$T/t_s30.json; lib=lib_f30 ;;
        m90) cfg=$T/t_m90.json; lib=lib     ;;
        s90) cfg=$T/t_s90.json; lib=lib     ;;
      esac
      echo "# rep$rep $arm $seq load=$(cut -d' ' -f1 /proc/loadavg)"
      $T/fps_one.sh $arm $cfg $lib $seq
    done
  done
done
echo "# BATCH DONE"
