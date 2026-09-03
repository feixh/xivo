#!/bin/bash
# M5's evaluation: `dynamic_init` off against on, as two jitter ensembles.
#
# Why an ensemble and not a pair of runs. M4 proved that with the temporal
# calibration frozen the nine static EuRoC sequences are bit-identical with the
# feature on -- the same arithmetic, to the last digit -- and yet with the shipped
# config their ATE still moves by 0.009 m on average and 0.071 m at worst, because
# holding the messages back changes an image's enqueue timestamp by a few hundred
# nanoseconds (see m4-dispatch.md) and that flips gating decisions. A single
# off/on pair therefore cannot tell a 0.01 m improvement from that reshuffling.
#
# The design that can: run both arms as n-member ensembles, and read the nine
# static sequences as a **built-in null control**. They take the static path in
# both arms, so whatever the ensemble says about them is the noise floor of this
# very comparison; only MH_01 and MH_02 can carry a real effect. If the static
# nine come out flat and the dynamic two move, the move is the feature.
#
# The jitter knob is `ptsb`, not the usual `vsb`, and that is not cosmetic: the
# dynamic initializer *solves for* the initial velocity and overwrites X.Vsb, so
# with the default knob every member of the `on` arm is bit-identical on MH_01 and
# MH_02 and the ensemble reports +-0.0000 on precisely the two sequences it exists
# to measure. Measured, not assumed -- see m5-eval.md. `P.Tsb` is the prior
# standard deviation of a quantity that is zero by definition, no initializer
# touches it, and at 1 ppm it moves per-sequence ATE by the same ~0.01 m the
# velocity knob does.
#
# Usage, from the worktree root:
#   ./notes-n-prompts/notes-dyninit/harness/m5_ensemble.sh \
#       [--members 3] [--mode both] [--seqs "..."] [--out DIR] \
#       [--tag NAME] [--patch 'dotted.key=json']... [--start-sec N]
#
# --start-sec turns both arms on N seconds into every sequence, which converts all
# eleven into dynamic starts. At N=0 only MH_01 and MH_02 leave the table already
# moving, so nine of eleven sequences can only ever measure the cost of the
# feature; a mid-flight start is where it is supposed to pay. Note that the
# built-in null control below assumes N=0 -- with a start offset there is no
# static sequence left to be the control, and the comparison is off-vs-on on
# eleven dynamic starts.
#
# --patch applies to the `on` arm only, which is what a tuning variant is: the
# same base config, dynamic_init enabled, one knob moved. The `off` arm is patched
# with nothing but `enabled=false`, so every variant is compared against the
# identical control. That one patch is not redundant: once M5 ships the feature on,
# a control that merely left the base config alone would be an `on` arm.
#
# n=3 over all eleven sequences in both modes takes ~15 min; n=10 ~45 min.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XIVO="$(cd "$HERE/../../.." && pwd)"
WORKSPACE="$(cd "$XIVO/.." && pwd)"
OV="$WORKSPACE/experiments/openvins"
WT="$(basename "$XIVO")"

MEMBERS=3
MODE="both"
SEQS="MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult"
TAG=""
OUT=""
PATCHES=()
ONLY=""
START_SEC=""

while [ $# -gt 0 ]; do
  case "$1" in
    --members) MEMBERS="$2"; shift 2 ;;
    --mode)    MODE="$2"; shift 2 ;;
    --seqs)    SEQS="$2"; shift 2 ;;
    --tag)     TAG="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    --patch)   PATCHES+=("$2"); shift 2 ;;
    --only)    ONLY="$2"; shift 2 ;;   # off | on | report
    --start-sec) START_SEC="$2"; shift 2 ;;
    -h|--help) awk 'NR>1 && /^#/; NR>1 && !/^#/ {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
OUT="${OUT:-$WORKSPACE/results/dyninit/m5-n$MEMBERS}"
OFF_ARM="off"
ON_ARM="on${TAG:+_$TAG}"

mkdir -p "$OUT"
run_arm() { # run_arm <name> <patch...>
  local name="$1"; shift
  local args=()
  for p in "$@"; do args+=(--patch "$p"); done
  echo "=== arm $name (n=$MEMBERS): ${*:-<control>}"
  CPU_BASE="${CPU_BASE:-0}" CPU_SPAN="${CPU_SPAN:-48}" \
    "$OV/sweep_xivo.sh" --name "$name" --members "$MEMBERS" --mode "$MODE" \
    --jitter-knob ptsb --worktree "$WT" --seqs "$SEQS" --out "$OUT" \
    ${START_SEC:+--start-sec "$START_SEC"} \
    "${args[@]+"${args[@]}"}"
}

case "$ONLY" in
  off)    run_arm "$OFF_ARM" 'dynamic_init.enabled=false' ;;
  on)     run_arm "$ON_ARM" 'dynamic_init.enabled=true' "${PATCHES[@]+"${PATCHES[@]}"}" ;;
  report) ;;
  "")     [ -s "$OUT/$OFF_ARM/summary.csv" ] || run_arm "$OFF_ARM" 'dynamic_init.enabled=false'
          run_arm "$ON_ARM" 'dynamic_init.enabled=true' "${PATCHES[@]+"${PATCHES[@]}"}" ;;
  *) echo "--only must be off, on or report" >&2; exit 1 ;;
esac

# One aggregate per mode: agg_ensemble.py averages the rows of a summary.csv, and
# averaging mono and stereo together would be meaningless.
for mode in $( [ "$MODE" = both ] && echo "stereo mono" || echo "$MODE" ); do
  echo
  echo "################ $mode, n=$MEMBERS ################"
  python3 "$OV/agg_ensemble.py" --mode "$mode" \
    --arm off "$OUT/$OFF_ARM" --arm "$ON_ARM" "$OUT/$ON_ARM" \
    --csv "$OUT/agg_${mode}_${ON_ARM}.csv"
done

# The null control, stated as a number rather than left to the reader: the nine
# static sequences take the same path in both arms, so their mean delta is what
# this comparison's noise floor looks like at this n, and the two dynamic ones
# have to beat it to mean anything.
python3 - "$OUT" "$OFF_ARM" "$ON_ARM" "$MODE" "${START_SEC:-0}" <<'PY'
import csv, math, os, statistics as st, sys
out, off_arm, on_arm, mode = sys.argv[1:5]
start = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
# Which sequences reach the bundle adjustment. At a start-at-zero these
# two are the only ones that leave the table moving, and the other nine
# are the null control; with --start-sec the rig is airborne on nearly
# all of them, so there is no null left and every row is a real test.
DYN = {'MH_01_easy', 'MH_02_easy'} if start == 0.0 else None
# A run whose ATE is metres-to-kilometres has not been degraded, it has failed,
# and averaging it with runs that worked produces a number that describes neither.
# The same threshold agg_ensemble.py uses.
DIVERGED_M = 100.0
def load(arm, mode):
    d = {}
    with open(os.path.join(out, arm, 'summary.csv')) as f:
        for r in csv.DictReader(f):
            if r['mode'] != mode:
                continue
            try:
                v = float(r['ate_002'])
            except ValueError:
                continue
            d.setdefault(r['seq'], []).append(v)
    return d
def split(v):
    """(finite members, count diverged)"""
    ok = [x for x in v if x < DIVERGED_M]
    return ok, len(v) - len(ok)
def sem(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float('nan')
for m in (['stereo', 'mono'] if mode == 'both' else [mode]):
    a, b = load(off_arm, m), load(on_arm, m)
    seqs = [s for s in a if s in b]
    if not seqs:
        continue
    print(f'\n--- {m}: delta ATE (on - off)'
          + (', and the static null control' if DYN is not None
             else f', all starts {start:g} s in'))
    print(f'{"sequence":<18}{"off":>16}{"on":>16}{"delta":>10}{"+-sem":>9}  branch')
    stat, dyn = [], []
    div_off = div_on = 0
    for s in sorted(seqs):
        ao, na = split(a[s])
        bo, nb = split(b[s])
        div_off += na
        div_on += nb
        dynamic = (s in DYN) if DYN is not None else True
        tag = 'DYNAMIC' if dynamic else 'static'
        if na or nb:
            # Not comparable as a delta: excluding the failures would credit the
            # arm that failed, and including them would swamp the metre-scale
            # numbers the rest of the table is made of. Report it as what it is.
            tag += f'  DIVERGED off {na}/{len(a[s])}, on {nb}/{len(b[s])}'
            fo = f'{st.mean(ao):.4f}' if ao else 'all failed'
            fn = f'{st.mean(bo):.4f}' if bo else 'all failed'
            print(f'{s:<18}{fo:>16}{fn:>16}{"":>10}{"":>9}  {tag}')
            continue
        d = st.mean(bo) - st.mean(ao)
        e = math.sqrt(sem(ao) ** 2 + sem(bo) ** 2)
        (dyn if dynamic else stat).append(d)
        print(f'{s:<18}{st.mean(ao):>10.4f}+-{sem(ao):.4f}'
              f'{st.mean(bo):>10.4f}+-{sem(bo):.4f}{d:>+10.4f}{e:>9.4f}  {tag}')
    if div_off or div_on:
        print(f'{"divergence census":<18}{"":>16}{"":>16}'
              f'  off {div_off} run(s), on {div_on} run(s) above {DIVERGED_M:g} m'
              f'   <- the headline, not the deltas below')
    if stat:
        print(f'{"9 static (null)":<18}{"":>16}{"":>16}{st.mean(stat):>+10.4f}'
              f'{sem(stat):>9.4f}  mean |delta| {st.mean(map(abs, stat)):.4f}, '
              f'max {max(map(abs, stat)):.4f}')
    if dyn:
        label = '2 dynamic' if DYN is not None else f'{len(dyn)} mid-flight'
        extra = '' if DYN is not None else (
            f'{sem(dyn):>9.4f}  mean |delta| {st.mean(map(abs, dyn)):.4f}, '
            f'best {min(dyn):+.4f}, worst {max(dyn):+.4f}')
        print(f'{label:<18}{"":>16}{"":>16}{st.mean(dyn):>+10.4f}{extra}')
PY

echo
echo "results under $OUT"
