T=/home/ubuntu/.claude/jobs/041e1899/tmp
for s in room1 room2 room3 room4 room5 room6; do
  bash $T/one_lib.sh cfg/tumvi_stereo.json  $s FINAL_stereo lib >> $T/final.log 2>&1 &
  bash $T/one_lib.sh cfg/m6_h_mono.json     $s FINAL_mono180 lib >> $T/final.log 2>&1 &
  bash $T/one_lib.sh cfg/m6_h_mono60.json   $s FINAL_mono60  lib >> $T/final.log 2>&1 &
done
wait
echo DONE >> $T/final.log
