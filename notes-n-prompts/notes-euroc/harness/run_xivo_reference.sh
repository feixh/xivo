#!/bin/bash
# Run XIVO (the workspace's own VIO, worktree ../../xivo) over the same
# sequences, in the same two sensor modes, and lay the output out exactly like
# run_openvins.sh does -- so score_openvins.py can score both systems with one
# code path, one groundtruth and one association window.
#
# Usage:
#   ./run_xivo_reference.sh --out DIR [options]
#
# Options:
#   --profile NAME   dataset profile in profiles/            (default: tumvi_room)
#   --mode M         mono | stereo | both                    (default: both)
#   --seqs "a b"     subset of the profile's sequences       (default: all)
#   --seeds N        XIVO_RANDOM_SEED 0..N-1, one run each   (default: 1)
#   --jitter N       N-member ensemble that perturbs the initial velocity
#                    X.Vsb by k * 1e-6 m/s instead of the seed. Use THIS for
#                    error bars: at HEAD the seed changes nothing (mono is
#                    bit-identical across 6 seeds), while a 1e-6 m/s IC
#                    perturbation -- six orders of magnitude inside the
#                    filter's own prior of 0.7 m/s -- moves per-sequence ATE by
#                    ~0.01. Same device as run_ensemble_bugfix.sh.
#   --worktree W     XIVO worktree under the workspace       (default: xivo)
#   --timing         throughput pass instead of accuracy: `-mode runOnly`,
#                    one core, ASLR off, every thread pool at 1, serial.
#                    This is the protocol of
#                    notes-n-prompts/notes-efficiency/harness/fps_one.sh, which
#                    run_openvins.sh --onecore also reproduces.
#   --cfg-prefix P   use cfg/P_<mode>.json instead of the profile's config,
#                    for sweeping tuning variants without editing the committed
#                    config
#   --no-score       skip the scoring pass
#
# Env: CPU_BASE / CPU_SPAN restrict which cpus this invocation uses (accuracy runs
# land on CPU_BASE..CPU_BASE+CPU_SPAN-1, a --timing pass pins to CPU_BASE). Set
# them when more than one worktree is benchmarking at once.
#
# Configs are cfg/<prefix>_mono.json and cfg/<prefix>_stereo.json inside the
# worktree, where <prefix> comes from the profile (PROFILE_XIVO_CFG_PREFIX,
# default `eff`). For TUM-VI those are the shipped
# tumvi_mono_ctl / tumvi_stereo configs plus print_timing, which differ from each
# other in exactly three keys, so mono-vs-stereo is a controlled pair (see
# notes-n-prompts/plan-efficiency.md).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$HERE/../.." && pwd)"

PROFILE="tumvi_room"
MODE="both"
SEQS=""
SEEDS=1
JITTER=""
WT="xivo"
TIMING=""
OUT=""
SCORE=1
CFG_PREFIX=""

while [ $# -gt 0 ]; do
  case "$1" in
    --profile)  PROFILE="$2"; shift 2 ;;
    --mode)     MODE="$2"; shift 2 ;;
    --seqs)     SEQS="$2"; shift 2 ;;
    --seeds)    SEEDS="$2"; shift 2 ;;
    --jitter)   JITTER=1; SEEDS="$2"; shift 2 ;;
    --worktree) WT="$2"; shift 2 ;;
    --timing)   TIMING=1; shift ;;
    --out)      OUT="$2"; shift 2 ;;
    --cfg-prefix) CFG_PREFIX="$2"; shift 2 ;;
    --no-score) SCORE=0; shift ;;
    -h|--help)  sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

[ -z "$OUT" ] && { echo "--out is required" >&2; exit 1; }
XIVO="$WORKSPACE/$WT"
[ -f "$XIVO/scripts/pyxivo.py" ] || { echo "no XIVO worktree at $XIVO" >&2; exit 1; }
PROFILE_FILE="$HERE/profiles/$PROFILE.sh"
[ -f "$PROFILE_FILE" ] || { echo "no such profile: $PROFILE_FILE" >&2; exit 1; }
OV_REPO="$WORKSPACE/experiments/open_vins" # profiles reference it in seq_config, unused here
# shellcheck disable=SC1090
source "$PROFILE_FILE"

case "$MODE" in
  mono)   MODES="mono" ;;
  stereo) MODES="stereo" ;;
  both)   MODES="mono stereo" ;;
  *) echo "--mode must be mono, stereo or both" >&2; exit 1 ;;
esac
[ -z "$SEQS" ] && SEQS="$PROFILE_SEQS"

mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"; mkdir -p "$OUT/gt"
for seq in $SEQS; do
  [ -s "$OUT/gt/$seq.txt" ] || awk -f "$HERE/asl_gt_to_tum.awk" "$(seq_gt_csv "$seq")" > "$OUT/gt/$seq.txt"
done

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "system=xivo"
  echo "profile=$PROFILE"
  echo "modes=$MODES"
  echo "seqs=$SEQS"
  echo "repeats=$SEEDS"
  echo "repeat_is=$([ -n "$JITTER" ] && echo "X.Vsb + k*1e-6 m/s" || echo XIVO_RANDOM_SEED)"
  echo "timing_pass=${TIMING:-0}"
  echo "worktree=$WT"
  echo "xivo_git=$(git -C "$XIVO" describe --tags --always --dirty 2>/dev/null || echo unknown)"
  echo "xivo_branch=$(git -C "$XIVO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "host=$(hostname), $(nproc) cpus"
} > "$OUT/run_info.txt"

export PATH="$WORKSPACE/dependencies/venv/bin:$PATH"
export PYTHONPATH="$XIVO/lib${PYTHONPATH:+:$PYTHONPATH}"
export XIVO_LIB="$XIVO/lib"
# Always single-threaded, always one cpu per run: an unpinned XIVO process spawns
# ~255 OpenCV/OpenMP threads for a 6% wall-clock gain, and a batch of them
# measures memory bandwidth. Output is bit-identical pinned
# (see the workspace note "XIVO: pin threads for batch sweeps").
#
# Read the cpu count BEFORE exporting the caps: coreutils `nproc` honours
# OMP_NUM_THREADS, so afterwards it answers 1 and every run lands on cpu 0.
NCPU="$(nproc)"
# Which cpus this invocation may use: runs land on CPU_BASE .. CPU_BASE+CPU_SPAN-1,
# and a --timing pass pins to CPU_BASE alone. Defaults cover the whole box. Set
# these when several agents/worktrees benchmark concurrently -- otherwise every
# invocation starts at cpu 0 and they fight over the same low cpus while the rest
# of the box idles.
CPU_BASE="${CPU_BASE:-0}"
CPU_SPAN="${CPU_SPAN:-$NCPU}"
slot=0
export OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$XIVO"
pids=(); labels=()
# Which XIVO dataset back end and which config the profile wants. Defaults keep
# the TUM-VI behaviour this script started with; profiles/euroc_mav.sh sets both,
# because EuRoC needs its own loader (`-dataset euroc`) and its own generated
# config (see scripts/make_euroc_cfg.py).
XIVO_DATASET="${PROFILE_XIVO_DATASET:-tumvi}"
# --cfg-prefix wins over the profile, so a tuning sweep can drop
# cfg/<prefix>_{mono,stereo}.json into the worktree and run without editing the
# committed config.
XIVO_CFG_PREFIX="${CFG_PREFIX:-${PROFILE_XIVO_CFG_PREFIX:-eff}}"

for mode in $MODES; do
  cfg="cfg/${XIVO_CFG_PREFIX}_$mode.json"
  [ -f "$cfg" ] || { echo "no config $XIVO/$cfg" >&2; exit 1; }
  for seq in $SEQS; do
    # frames fed = images in cam0, the same denominator run_euroc_folder reports
    nframes=$(grep -c '^[0-9]' "$(seq_folder "$seq")/mav0/cam0/data.csv")
    for ((k = 0; k < SEEDS; k++)); do
      rundir="$OUT/$mode/${seq}_r$k"
      mkdir -p "$rundir"
      runcfg="$cfg"
      if [ -n "$JITTER" ]; then
        # Member config, generated once per (mode, k). XIVO's configs carry //
        # comments, which jsoncpp accepts and Python's json does not, so patch
        # the Vsb line textually -- same approach as run_ensemble_bugfix.sh.
        runcfg="$OUT/cfg/${XIVO_CFG_PREFIX}_${mode}_m$k.json"
        if [ ! -s "$runcfg" ]; then
          mkdir -p "$OUT/cfg"
          python3 - "$XIVO/$cfg" "$runcfg" "$k" <<'PY'
import re, sys
src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = open(src).read()
new = '    "Vsb"   : [%.12g, 0, 0], // ENSEMBLE MEMBER %d\n' % (k * 1e-6, k)
s2, n = re.subn(r'^ *"Vsb"\s*:\s*\[[^\]]*\],.*\n', new, s, count=1, flags=re.M)
if n != 1:
    sys.exit('could not find X.Vsb in %s (matched %d)' % (src, n))
open(dst, 'w').write(s2)
PY
        fi
      fi
      run=(python3 scripts/pyxivo.py -root "$PROFILE_ROOT" -dataset "$XIVO_DATASET" -seq "$seq"
           -cam_id 0 -cfg "$runcfg" -dump "$rundir/dump")
      if [ -n "$TIMING" ]; then
        run+=(-mode runOnly)
        run=(taskset -c "$CPU_BASE" setarch -R "${run[@]}")
      else
        run+=(-mode eval)
        run=(taskset -c "$((CPU_BASE + slot % CPU_SPAN))" "${run[@]}")
        slot=$((slot + 1))
      fi
      seedval=$([ -n "$JITTER" ] && echo 0 || echo "$k")
      printf '%s\n' "XIVO_RANDOM_SEED=$seedval ${run[*]}" > "$rundir/cmd.txt"
      (
        set +e
        mkdir -p "$rundir/dump"
        XIVO_RANDOM_SEED=$seedval /usr/bin/time -f "%e %U %S %M" -o "$rundir/time.txt" \
          "${run[@]}" > "$rundir/run.log" 2>&1
        rc=$?
        read -r wall usr sys maxrss < "$rundir/time.txt"
        # The eval-mode saver writes <dump>/<dataset>_<seq>_cam0 in TUM format
        # (get_xivo_gt_filename in scripts/savers.py names it after the dataset,
        # so this has to follow -dataset, not assume tumvi).
        est="$rundir/dump/${XIVO_DATASET}_${seq}_cam0"
        [ -f "$est" ] && cp "$est" "$rundir/traj.txt"
        {
          echo "frames_processed=$nframes"
          echo "wall_total_s=$wall"
          echo "user_time_s=$usr"
          echo "peak_rss_mb=$(python3 -c "print('%.1f' % ($maxrss / 1024.0))")"
          echo "exit_code=$rc"
        } > "$rundir/stats.txt"
        exit $rc
      ) &
      pids+=($!); labels+=("$mode/${seq}_r$k")
      echo "=== $mode/$seq seed$k ==="
      [ -n "$TIMING" ] && wait "${pids[-1]}"
    done
  done
done

fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "FAILED: ${labels[$i]} (see $OUT/${labels[$i]}/run.log)" >&2; fail=1; }
done

[ "$SCORE" = 1 ] && { echo; "$HERE/score_openvins.py" "$OUT"; }
echo; echo "output in $OUT"
exit $fail
