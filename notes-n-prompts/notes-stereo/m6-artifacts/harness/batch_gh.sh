T=/home/ubuntu/.claude/jobs/041e1899/tmp
for s in room1 room2 room3 room4 room5 room6; do
  for a in base gb01 gb10 pred grav; do
    bash $T/one_lib.sh cfg/m6_h_$a.json $s h_$a lib_f90 >> $T/m6h.log 2>&1 &
  done
  bash $T/one_lib.sh cfg/m6_e2_t240.json $s f120t240b lib_f120 >> $T/m6g.log 2>&1 &
  bash $T/one_lib.sh cfg/m6_e2_t300.json $s f150t300b lib_f150 >> $T/m6g.log 2>&1 &
done
wait
echo DONE >> $T/m6h.log; echo DONE >> $T/m6g.log
