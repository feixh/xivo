#!/bin/bash
# Run OpenVINS over a dataset profile (all sequences x requested sensor modes)
# and score every run. See HOWTO.md for the full picture.
#
# Usage:
#   ./run_openvins.sh --out DIR [options]
#
# Options:
#   --profile NAME       dataset profile in profiles/ (default: tumvi_room)
#   --mode  M            mono | stereo | both            (default: both)
#   --seqs "a b c"       subset of the profile's sequences (default: all)
#   --repeats N          run each (mode,seq) N times, as r0..r(N-1)  (default: 1)
#   --cpus-per-run N     taskset width for each concurrent run       (default: 4)
#   --serial             run one at a time (use this for timing/FPS numbers)
#   --onecore            implies --serial, plus: pin to a single cpu, force every
#                        thread pool to 1, disable ASLR. This reproduces the
#                        protocol XIVO's FPS numbers use
#                        (notes-n-prompts/notes-efficiency/harness/fps_one.sh),
#                        so it is the setting for cross-system throughput.
#   --threads N          override num_opencv_threads (default: from config, 4)
#   --no-score           skip the scoring pass at the end
#   --extra "..."        extra args passed through to run_euroc_folder
#
# Layout of the output directory:
#   DIR/<mode>/<seq>_r<k>/{traj.txt,timing.csv,stats.txt,run.log}
#   DIR/gt/<seq>.txt          groundtruth in TUM format (shared by all runs)
#   DIR/summary.csv           one row per run (written by score_openvins.py)
#   DIR/summary.md            markdown tables
#
# Examples:
#   ./run_openvins.sh --out /tmp/ov_all                          # 6 seq x 2 modes
#   ./run_openvins.sh --out /tmp/ov_fps --serial --repeats 1     # timing-quality run
#   ./run_openvins.sh --out /tmp/ov_var --mode mono --seqs room1 --repeats 5
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"
OV_REPO="$WORKSPACE/experiments/open_vins"
OV_BIN="${OV_BIN:-$WORKSPACE/experiments/ov_build/run_euroc_folder}"

PROFILE="tumvi_room"
MODE="both"
SEQS=""
REPEATS=1
CPUS_PER_RUN=4
SERIAL=""
ONECORE=""
THREADS=""
EXTRA=""
OUT=""
SCORE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --profile)      PROFILE="$2"; shift 2 ;;
    --mode)         MODE="$2"; shift 2 ;;
    --seqs)         SEQS="$2"; shift 2 ;;
    --repeats)      REPEATS="$2"; shift 2 ;;
    --cpus-per-run) CPUS_PER_RUN="$2"; shift 2 ;;
    --serial)       SERIAL=1; shift ;;
    --onecore)      SERIAL=1; ONECORE=1; CPUS_PER_RUN=1; shift ;;
    --threads)      THREADS="$2"; shift 2 ;;
    --extra)        EXTRA="$2"; shift 2 ;;
    --out)          OUT="$2"; shift 2 ;;
    --no-score)     SCORE=0; shift ;;
    -h|--help)      sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

[ -z "$OUT" ] && { echo "--out is required" >&2; exit 1; }
[ -x "$OV_BIN" ] || { echo "no run_euroc_folder binary at $OV_BIN (see HOWTO.md to build)" >&2; exit 1; }
PROFILE_FILE="$HERE/profiles/$PROFILE.sh"
[ -f "$PROFILE_FILE" ] || { echo "no such profile: $PROFILE_FILE" >&2; ls "$HERE/profiles" >&2; exit 1; }
# shellcheck disable=SC1090
source "$PROFILE_FILE"

case "$MODE" in
  mono)   MODES="mono" ;;
  stereo) MODES="stereo" ;;
  both)   MODES="mono stereo" ;;
  *) echo "--mode must be mono, stereo or both" >&2; exit 1 ;;
esac
[ -z "$SEQS" ] && SEQS="$PROFILE_SEQS"

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
mkdir -p "$OUT/gt"

# Groundtruth: convert once per sequence, shared by all runs
for seq in $SEQS; do
  if [ ! -s "$OUT/gt/$seq.txt" ]; then
    awk -f "$HERE/asl_gt_to_tum.awk" "$(seq_gt_csv "$seq")" > "$OUT/gt/$seq.txt"
  fi
done

# Record what produced these numbers, so a directory is self-describing
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "profile=$PROFILE"
  echo "modes=$MODES"
  echo "seqs=$SEQS"
  echo "repeats=$REPEATS"
  echo "serial=${SERIAL:-0}"
  echo "onecore=${ONECORE:-0}"
  echo "cpus_per_run=$CPUS_PER_RUN"
  echo "threads_override=${THREADS:-none}"
  echo "extra=$EXTRA"
  echo "binary=$OV_BIN"
  echo "openvins_git=$(git -C "$OV_REPO" describe --tags --always --dirty 2>/dev/null || echo unknown)"
  echo "host=$(hostname), $(nproc) cpus"
} > "$OUT/run_info.txt"

# Assign each concurrent run its own block of cpus (see notes: one shared OpenCV
# pool across many processes burns cores for almost no wall-clock gain)
# Before the ONECORE export below: coreutils `nproc` honours OMP_NUM_THREADS and
# would otherwise answer 1.
NCPU="$(nproc)"
slot=0
if [ -n "$ONECORE" ]; then
  # Same knobs as XIVO's fps_one.sh, so the two systems' FPS are comparable:
  # every thread pool at 1, and ASLR off so repeats are timing-stable.
  export OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  [ -z "$THREADS" ] && THREADS=1
fi
pids=()
labels=()

for mode in $MODES; do
  for seq in $SEQS; do
    for ((r = 0; r < REPEATS; r++)); do
      rundir="$OUT/$mode/${seq}_r$r"
      mkdir -p "$rundir"
      args=(
        "$(seq_config "$seq" "$mode")"
        --dataset "$(seq_folder "$seq")"
        --traj "$rundir/traj.txt"
        --timing "$rundir/timing.csv"
        --stats "$rundir/stats.txt"
        --verbosity WARNING
      )
      # shellcheck disable=SC2206
      args+=($(seq_extra "$seq" "$mode"))
      [ -n "$THREADS" ] && args+=(--num_opencv_threads "$THREADS")
      # shellcheck disable=SC2206
      [ -n "$EXTRA" ] && args+=($EXTRA)

      cmd=("$OV_BIN" "${args[@]}")
      if [ -n "$ONECORE" ]; then
        cmd=(taskset -c 0 setarch -R "${cmd[@]}")
      elif [ -z "$SERIAL" ]; then
        lo=$(((slot * CPUS_PER_RUN) % NCPU))
        hi=$((lo + CPUS_PER_RUN - 1))
        cmd=(taskset -c "$lo-$hi" "${cmd[@]}")
        slot=$((slot + 1))
      fi

      echo "=== $mode/$seq r$r ==="
      printf '%s\n' "${cmd[*]}" > "$rundir/cmd.txt"
      "${cmd[@]}" > "$rundir/run.log" 2>&1 &
      pids+=($!)
      labels+=("$mode/${seq}_r$r")
      [ -n "$SERIAL" ] && wait "${pids[-1]}"
    done
  done
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED: ${labels[$i]} (see $OUT/${labels[$i]}/run.log)" >&2
    fail=1
  fi
done

if [ "$SCORE" = 1 ]; then
  echo
  "$HERE/score_openvins.py" "$OUT"
fi

echo
echo "output in $OUT"
exit $fail
