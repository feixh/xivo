#!/bin/bash
T=/home/ubuntu/.claude/jobs/041e1899/tmp
for s in room1 room2 room3 room4 room5 room6; do
  bash $T/one_lib.sh cfg/m6_d_t240.json      $s f120t240    lib_f120 >> $T/m6f.log 2>&1 &
  bash $T/one_lib.sh cfg/m6_d_t300.json      $s f150t300    lib_f150 >> $T/m6f.log 2>&1 &
  bash $T/one_lib.sh cfg/m6_d_t180_mono.json $s monof90t180 lib_f90  >> $T/m6f.log 2>&1 &
  bash $T/one_lib.sh cfg/m6_d_t60_mono.json  $s monof90t60  lib_f90  >> $T/m6f.log 2>&1 &
done
wait
echo DONE >> $T/m6f.log
