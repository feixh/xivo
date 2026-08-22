#!/bin/bash
# Run scripts/mem/leakcheck.sh over every TUM-VI room sequence and both mono+IMU
# `vio` configs, in parallel, and print one line per run.
#
#   scripts/mem/leakcheck_matrix.sh [out-root]
#
# out-root defaults to /tmp/xivo-leak/matrix. Each run gets
# <out-root>/<cfg>/<seq>/{report.txt,run.log,traj_<seq>}.
#
# Env:
#   MAX_ENTRIES=N   passed through to leakcheck.sh (default 0 = whole sequence)
#   SEQS=".."       override the sequence list
#   CFGS=".."       override the config list
#   JOBS=N          how many runs at once (default 12, i.e. all of them)
#
# Exit status is non-zero if any run reported anything: ASan exits 23 when LSan
# finds a leak, so this is the M5 leak gate. Everything printed comes from the
# per-run report.txt files, which are left in place for inspection.
set -uo pipefail

OUT_ROOT="${1:-/tmp/xivo-leak/matrix}"
SEQS="${SEQS:-room1 room2 room3 room4 room5 room6}"
CFGS="${CFGS:-vio_tumvi vio_tumvi_nodesc}"
JOBS="${JOBS:-12}"

XIVO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$XIVO/scripts/mem"

mkdir -p "$OUT_ROOT"

running=0
for cfg in $CFGS; do
  for seq in $SEQS; do
    out="$OUT_ROOT/$cfg/$seq"
    mkdir -p "$out"
    ( "$HERE/leakcheck.sh" "$seq" "$out" "$cfg" > "$out/leakcheck.log" 2>&1
      echo "$?" > "$out/exit" ) &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
  done
done
wait

fail=0
printf '%-18s %-7s %-5s %-9s %s\n' cfg seq exit report findings
for cfg in $CFGS; do
  for seq in $SEQS; do
    out="$OUT_ROOT/$cfg/$seq"
    status="$(cat "$out/exit" 2>/dev/null || echo '?')"
    bytes="$(wc -c < "$out/report.txt" 2>/dev/null || echo '?')"
    # `grep -c` prints 0 *and* exits 1 when there is no match, so an `|| echo 0`
    # fallback here would append a second line and make every clean run look
    # like a failure. Take the last line and default an unreadable file to 0.
    findings="$(grep -cE 'ERROR: (AddressSanitizer|LeakSanitizer)|Direct leak|Indirect leak' \
                "$out/report.txt" 2>/dev/null | tail -1)"
    findings="${findings:-0}"
    printf '%-18s %-7s %-5s %-9s %s\n' "$cfg" "$seq" "$status" "$bytes" "$findings"
    [ "$status" = 0 ] && [ "$findings" = 0 ] || fail=1
  done
done

if [ "$fail" = 0 ]; then
  echo "all runs clean: exit 0, no sanitizer findings"
else
  echo "SOME RUNS REPORTED -- see $OUT_ROOT/<cfg>/<seq>/report.txt" >&2
fi
exit "$fail"
