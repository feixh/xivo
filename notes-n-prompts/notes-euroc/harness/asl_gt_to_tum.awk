# Convert an ASL/EuRoC groundtruth csv to TUM txt format.
#
#   in:  #timestamp [ns], p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z [, ...]
#   out: timestamp(s) tx ty tz qx qy qz qw
#
# Works for TUM-VI mav0/mocap0/data.csv and EuRoC mav0/state_groundtruth_estimate0/data.csv
# (the latter has extra velocity/bias columns which we ignore).
#
# Timestamps keep full nanosecond resolution, which matters when a scorer
# associates estimate-to-groundtruth with a 1 ms window.
BEGIN { FS = ","; print "# timestamp(s) tx ty tz qx qy qz qw" }
/^[ \t]*#/ { next }
NF >= 8 {
  sec = substr($1, 1, length($1) - 9)
  nsec = substr($1, length($1) - 8)
  printf "%s.%s %s %s %s %s %s %s %s\n", sec, nsec, $2, $3, $4, $6, $7, $8, $5
}
