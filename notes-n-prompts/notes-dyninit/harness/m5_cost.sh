#!/bin/bash
# What enabling dynamic initialization costs in compute, in the two places it can
# be paid.
#
# The claim this measures is the one init_dispatch.h makes at the top: the cost is
# a one-off startup latency and never a per-frame regression. That claim cannot be
# checked with a stopwatch on a whole run -- initialization is a few tens of
# milliseconds inside a 40-second sequence, four orders of magnitude below the
# run-to-run spread of a wall clock -- so it is measured from both ends:
#
#   probe   `bin/init_probe -dispatch` drives the real dispatcher and reports its
#           own timers: `buf_ms`, the detector's KLT and the window's over the
#           buffered frames, and `slv_ms`, one window build plus Stage A plus
#           Stage B. This is the one-off, isolated: no Estimator is constructed.
#   timing  the workspace's throughput protocol (`--timing`: `-mode runOnly`, one
#           core, ASLR off, every thread pool at 1, serial) with the feature off
#           and on. This is where a *per-frame* regression would show up, and
#           where peak RSS is comparable -- an accuracy pass measures neither
#           (see the workspace note "XIVO memory/timing measurement caveats").
#
# Both parts want a quiet machine. Run nothing else; an accuracy ensemble on the
# other cpus invalidates the timing pass through memory bandwidth alone.
#
# Usage, from the worktree root:
#   ./notes-n-prompts/notes-dyninit/harness/m5_cost.sh [options]
#     --out DIR        default ../results/dyninit/m5-cost
#     --cfg-on P       config prefix for the `on` arm   (default: euroc)
#     --cfg-off P      config prefix for the `off` arm  (default: same as --cfg-on)
#     --starts "0 55"  start offsets for the probe, s
#     --repeat N       probe repetitions per (sequence, start)  (default: 5)
#     --only probe|timing
#
# With the default `--cfg-on euroc` this measures the *shipped* configuration, on
# the shipped side of `dynamic_init.enabled`; the off arm is the same config with
# the block disabled, so the pair differs in exactly one key.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XIVO="$(cd "$HERE/../../.." && pwd)"
WORKSPACE="$(cd "$XIVO/.." && pwd)"
OV="$WORKSPACE/experiments/openvins"
WT="$(basename "$XIVO")"

OUT="$WORKSPACE/results/dyninit/m5-cost"
CFG_ON="euroc"
CFG_OFF=""
STARTS="0 55"
REPEAT=5
ONLY=""
SEQS="MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult"

while [ $# -gt 0 ]; do
  case "$1" in
    --out)     OUT="$2"; shift 2 ;;
    --cfg-on)  CFG_ON="$2"; shift 2 ;;
    --cfg-off) CFG_OFF="$2"; shift 2 ;;
    --starts)  STARTS="$2"; shift 2 ;;
    --repeat)  REPEAT="$2"; shift 2 ;;
    --only)    ONLY="$2"; shift 2 ;;
    -h|--help) awk 'NR>1 && /^#/; NR>1 && !/^#/ {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
CFG_OFF="${CFG_OFF:-$CFG_ON}"
ROOT="$WORKSPACE/data/euroc"
CPU="${CPU_BASE:-0}"
mkdir -p "$OUT"

# ---------------------------------------------------------------- probe
if [ "$ONLY" != timing ]; then
  # One cpu, ASLR off, thread pools at 1 -- the same conditions as the throughput
  # protocol, because the numbers are reported alongside per-frame costs measured
  # under it. `-dispatch` reads the config's own `dynamic_init` block through
  # `InitDispatcher::OptionsFromJson`, the function the estimator uses.
  export OMP_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 OPENBLAS_NUM_THREADS=1 \
         MKL_NUM_THREADS=1
  for mode in stereo mono; do
    cfg="cfg/${CFG_ON}_${mode}.json"
    [ -f "$XIVO/$cfg" ] || { echo "no $XIVO/$cfg" >&2; exit 1; }
    for start in $STARTS; do
      f="$OUT/probe_${mode}_t${start}.txt"
      echo "=== probe $mode, start ${start}s -> $f"
      : > "$f"
      hdr="-header"
      for seq in $SEQS; do
        # The dispatcher's window is monocular whatever the config's `stereo` says
        # (see `VisualStereo::image`), so mono and stereo differ here only through
        # the tracker and BA knobs -- worth measuring, not worth assuming equal.
        taskset -c "$CPU" setarch -R "$XIVO/bin/init_probe" -dispatch $hdr \
          -repeat "$REPEAT" -cfg "$XIVO/$cfg" -dataset euroc -root "$ROOT" \
          -seq "$seq" -start "$start" 2>/dev/null >> "$f" || true
        hdr=""
      done
      cat "$f"
    done
  done
fi

# ---------------------------------------------------------------- timing
if [ "$ONLY" != probe ]; then
  for arm in off on; do
    prefix="$([ "$arm" = on ] && echo "$CFG_ON" || echo "$CFG_OFF")"
    # The off arm is the same config with the one key flipped, generated the same
    # way the ensemble's control is, so the pair is a controlled comparison rather
    # than two configs that happen to differ.
    "$OV/sweep_xivo.sh" --name "cost_$arm" --base "$prefix" --mode both \
      --gen-only --worktree "$WT" \
      --patch "dynamic_init.enabled=$([ "$arm" = on ] && echo true || echo false)"
    echo "=== timing $arm (one core, serial, runOnly)"
    CPU_BASE="$CPU" "$OV/run_xivo_reference.sh" --profile euroc_mav --mode both \
      --timing --worktree "$WT" --cfg-prefix "tune_cost_$arm" --seqs "$SEQS" \
      --out "$OUT/timing_$arm" > "$OUT/timing_$arm.log" 2>&1 \
      || { echo "FAILED, see $OUT/timing_$arm.log"; tail -20 "$OUT/timing_$arm.log"; }
  done

  python3 - "$OUT" <<'PY'
import csv, os, statistics as st, sys
out = sys.argv[1]
def load(arm):
    d = {}
    p = os.path.join(out, f'timing_{arm}', 'summary.csv')
    if not os.path.exists(p):
        return d
    for r in csv.DictReader(open(p)):
        d[(r['mode'], r['seq'])] = r
    return d
a, b = load('off'), load('on')
keys = sorted(set(a) & set(b))
if not keys:
    sys.exit('no timing summaries to compare')
def col(r, name):
    try:
        return float(r[name])
    except (KeyError, ValueError, TypeError):
        return float('nan')
for metric, unit, better in (('fps_wall', 'fps', 'higher'),
                             ('fps_mean', 'fps', 'higher'),
                             ('peak_rss_mb', 'MB', 'lower')):
    if metric not in a[keys[0]]:
        continue
    print(f'\n--- {metric} ({unit}, {better} is better), one core, serial')
    print(f'{"mode/sequence":<30}{"off":>10}{"on":>10}{"delta":>10}{"%":>8}')
    for mode in ('stereo', 'mono'):
        rows = [(k, col(a[k], metric), col(b[k], metric))
                for k in keys if k[0] == mode]
        for (m, s), x, y in rows:
            print(f'{m + "/" + s:<30}{x:>10.1f}{y:>10.1f}{y - x:>+10.1f}'
                  f'{100 * (y - x) / x:>+8.1f}')
        if rows:
            xs = [x for _, x, _ in rows]; ys = [y for _, _, y in rows]
            print(f'{mode + " mean":<30}{st.mean(xs):>10.1f}{st.mean(ys):>10.1f}'
                  f'{st.mean(ys) - st.mean(xs):>+10.1f}'
                  f'{100 * (st.mean(ys) - st.mean(xs)) / st.mean(xs):>+8.1f}')
print('\nA per-frame regression and the one-off both land in fps_wall, and they are'
      '\nnot separable here: the one-off is ~0.9 s of compute on a dynamic start,'
      '\nwhich is a couple of percent of a whole run. What distinguishes them is'
      '\nthat the one-off is a constant, so it shrinks with run length and vanishes'
      '\nfrom fps_median. Read the size of it off the probe above, not off this'
      '\ntable; read this table for whether anything *per frame* moved.')
PY
fi

echo
echo "cost measurements under $OUT"
