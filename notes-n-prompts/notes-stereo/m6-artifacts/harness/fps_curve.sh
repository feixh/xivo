#!/bin/bash
# FPS for the rest of the published capacity curve, room1, one rep each.
T=/home/ubuntu/.claude/jobs/041e1899/tmp
while pgrep -f fps_batch.sh > /dev/null; do sleep 20; done
$T/fps_one.sh s30t120  $T/t_s30t120.json  lib_f30  room1
$T/fps_one.sh s60t120  $T/t_s60t120.json  lib_f60  room1
$T/fps_one.sh s120t240 $T/t_s120t240.json lib_f120 room1
echo "# CURVE DONE"
