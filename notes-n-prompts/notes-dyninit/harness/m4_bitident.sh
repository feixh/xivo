#!/bin/bash
# M4's prediction, checked directly: turning `dynamic_init` on must change the
# trajectory of *exactly* the sequences the M1 detector routes to the dynamic
# branch, and leave every other one alone.
#
# "Leave alone" needs saying precisely, because the obvious version of it --
# `cmp` on the dumped file -- is not achievable and the reason is instructive.
# The static path is not tuned differently or re-derived: the messages are held
# back, the detector says static, and they are replayed into an estimator that
# never saw them. `MaintainBuffer` still pops one message per message pushed, so
# at the instant of the decision the replay has executed exactly as many
# messages, in the same order, as the unbuffered filter would have. Two things
# still differ, and neither is the buffering:
#
#   1. The poses inside the init window are never reported. The filter has not
#      started yet, so there is nothing to report. `on` is therefore missing a
#      prefix of `off`, and comparison has to align on timestamp, not line.
#   2. With USE_ONLINE_TEMPORAL_CALIB the *enqueue* timestamp of an image is
#      stamped with the current `X_.td` (estimator.cpp, `VisualMeas`), because
#      the message heap sorts images against IMU samples on the corrected clock.
#      Inside the init window no EKF update has run, so every buffered frame
#      carries `td_0`, while the unbuffered filter had already moved `td` a few
#      hundred nanoseconds by frame 5. Sub-microsecond -- and enough to flip an
#      image/IMU tie in the heap, after which the two runs diverge chaotically.
#
# So this script checks two things:
#
#   GATE (exact)   with the temporal calibration frozen (`P.td = 0`, so every
#                  frame is enqueued with the same offset in both runs) every
#                  static sequence must match to the last digit on every shared
#                  pose. This is the real test of the divert-and-replay, and it
#                  is exact: any mismatch is a bug in the buffering.
#   REPORT (ship)  with the shipped config, how far the sub-microsecond
#                  perturbation of (2) actually moves things. Not a pass/fail --
#                  M5 quantifies it against run-to-run noise -- but a number
#                  worth having before the milestone is committed.
#
# Usage, from the worktree root:
#   ./notes-n-prompts/notes-dyninit/harness/m4_bitident.sh \
#       [--profile euroc_mav] [--mode both] [--seqs "MH_01_easy V1_01_easy"] \
#       [--dynamic "MH_01_easy MH_02_easy"] [--no-ship] [--compare-only] [--out DIR]
#
# --dynamic lists the sequences allowed to differ under the exact gate, i.e. the
# ones the detector routes to the BA. On EuRoC that is MH_01/MH_02, the split
# `bin/init_probe` measured in M1 before any of this was wired up; on TUM-VI it
# is empty, since all six rooms start static.
#
# Takes ~30 min for all 11 EuRoC sequences in both modes with both variants.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XIVO="$(cd "$HERE/../../.." && pwd)"
WORKSPACE="$(cd "$XIVO/.." && pwd)"
RUNNER="$WORKSPACE/experiments/openvins/run_xivo_reference.sh"
WT="$(basename "$XIVO")"

PROFILE="euroc_mav"
MODE="both"
SEQS=""
DYNAMIC="MH_01_easy MH_02_easy"
SHIP=1
COMPARE_ONLY=0
OUT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --seqs) SEQS="$2"; shift 2 ;;
    --dynamic) DYNAMIC="$2"; shift 2 ;;
    --no-ship) SHIP=0; shift ;;
    --compare-only) COMPARE_ONLY=1; shift ;;
    --out)  OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,49p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
OUT="${OUT:-$WORKSPACE/results/dyninit/m4-bitident-$PROFILE}"

[ -f "$RUNNER" ] || { echo "no runner at $RUNNER" >&2; exit 1; }

# The base config is whatever the profile hands XIVO, so this script works on a
# dataset whose committed config has no `dynamic_init` block at all (absent means
# disabled -- see estimator.cpp). Same defaulting as the runner.
PROFILE_FILE="$WORKSPACE/experiments/openvins/profiles/$PROFILE.sh"
[ -f "$PROFILE_FILE" ] || { echo "no profile at $PROFILE_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$PROFILE_FILE"
BASE="${PROFILE_XIVO_CFG_PREFIX:-eff}"

# Every variant is the base config with one or two keys changed, generated rather
# than committed: a second checked-in config would drift from the first, and the
# whole point is that nothing else can differ.
gen_cfg() { # gen_cfg <mode> <prefix> <enabled> <freeze_td>
  python3 - "$XIVO/cfg/${BASE}_$1.json" "$XIVO/cfg/$2_$1.json" "$3" "$4" <<'PY'
import json, re, sys
src, dst, enabled, freeze = sys.argv[1:5]
cfg = json.loads(re.sub(r'(?m)//.*$', '', open(src).read()))
dyn = cfg.setdefault("dynamic_init", {})
assert not dyn.get("enabled", False), f"{src} already has dynamic_init on"
dyn["enabled"] = enabled == "1"
# Defaults for a config that carries no block of its own (TUM-VI). Same values as
# the documented block in cfg/euroc_*.json.
dyn.setdefault("max_wait_sec", 3.0)
dyn.setdefault("window_frames", 31)
dyn.setdefault("min_frames", 12)
if freeze == "1":
    # Zero prior variance on td: the EKF can never move it, so an image's enqueue
    # timestamp is the same in both runs whether or not the divert delayed it.
    cfg["P"]["td"] = 0.0
json.dump(cfg, open(dst, "w"), indent=2)
PY
}

MODES=$([ "$MODE" = both ] && echo "mono stereo" || echo "$MODE")
if [ "$COMPARE_ONLY" = 0 ]; then
  for m in $MODES; do
    gen_cfg "$m" m4frzoff 0 1
    gen_cfg "$m" m4frzon  1 1
    [ "$SHIP" = 1 ] && gen_cfg "$m" m4shpon 1 0
  done
fi

mkdir -p "$OUT"
run_variant() { # run_variant <name> <cfg-prefix> <score?>
  local name="$1" prefix="$2" score="$3"
  echo "=== running $name (cfg/${prefix}_*.json)"
  local extra=()
  [ "$score" = 1 ] || extra+=(--no-score)
  CPU_BASE="${CPU_BASE:-0}" CPU_SPAN="${CPU_SPAN:-48}" \
    "$RUNNER" --profile "$PROFILE" --mode "$MODE" --seeds 1 --worktree "$WT" \
    --cfg-prefix "$prefix" ${SEQS:+--seqs "$SEQS"} "${extra[@]}" \
    --out "$OUT/$name" > "$OUT/$name.log" 2>&1 \
    || { echo "FAILED, see $OUT/$name.log"; tail -30 "$OUT/$name.log"; exit 1; }
}

if [ "$COMPARE_ONLY" = 0 ]; then
  run_variant frz_off m4frzoff 0
  run_variant frz_on  m4frzon  0
  if [ "$SHIP" = 1 ]; then
    run_variant ship_off "$BASE"  1
    run_variant ship_on  m4shpon  1
  fi
fi

# Timestamp-aligned comparison. Prints one of:
#   exact        every shared pose byte-identical
#   CHANGED      a shared pose differs -- with the first one and how far apart
#   NO-OVERLAP   the two files share no timestamp at all
#
# Alignment goes in whichever direction overlaps. Usually `on` starts *later*
# (the init window produces no poses), but on a sequence that takes the dynamic
# branch it can start *earlier*: the static path has to wait for the filter to
# converge out of a wrong initial velocity before `VisionInitialized()` turns
# true, and seeding from the BA skips that wait. Reporting the offset signed
# makes that visible instead of looking like a failure.
compare() {
  python3 - "$1" "$2" <<'PY'
import sys
def load(p):
    return [l.rstrip("\n") for l in open(p) if l[:1].isdigit()]
A, B = load(sys.argv[1]), load(sys.argv[2])
if not A or not B:
    print("EMPTY|one side produced no poses")
    raise SystemExit
def align(X, Y):
    """Index in X of Y's first timestamp, or None."""
    t = Y[0].split()[0]
    return next((i for i, l in enumerate(X) if l.split()[0] == t), None)
s = align(A, B)
if s is not None:
    a, b, lead = A[s:], B, f"on starts {s} poses later"
else:
    s = align(B, A)
    if s is None:
        print(f"NO-OVERLAP|off starts {A[0].split()[0]}, on {B[0].split()[0]}")
        raise SystemExit
    a, b = A, B[s:]
    dt = float(B[0].split()[0]) - float(A[0].split()[0])
    lead = f"on starts {s} poses ({-dt:.2f} s) EARLIER"
n = min(len(a), len(b))
bad = [i for i in range(n) if a[i] != b[i]]
if not bad:
    tag = "exact" if len(a) == len(b) else "CHANGED"
    extra = "" if len(a) == len(b) else f", but lengths differ {len(a)} vs {len(b)}"
    print(f"{tag}|{n} shared poses, {lead}{extra}")
    raise SystemExit
def xyz(r): return [float(v) for v in r.split()[1:4]]
d0 = sum((p - q) ** 2 for p, q in zip(xyz(a[bad[0]]), xyz(b[bad[0]]))) ** .5
dN = sum((p - q) ** 2 for p, q in zip(xyz(a[n - 1]), xyz(b[n - 1]))) ** .5
print(f"CHANGED|{lead}, {len(bad)}/{n} shared poses differ, "
      f"first at +{bad[0]} by {d0:.6f} m, last by {dN:.4f} m")
PY
}

report() { # report <label> <off-dir> <on-dir> <strict?>
  local label="$1" offd="$2" ond="$3" strict="$4"
  echo
  echo "--- $label"
  printf '%-18s %-7s %-9s %-8s %s\n' sequence mode expect verdict detail
  for mode in $MODES; do
    for d in "$offd/$mode"/*_r0; do
      [ -d "$d" ] || continue
      local seq; seq="$(basename "$d")"; seq="${seq%_r0}"
      local expect=static
      case " $DYNAMIC " in *" $seq "*) expect=dynamic ;; esac
      local a="$d/traj.txt" b="$ond/$mode/${seq}_r0/traj.txt"
      if [ ! -f "$a" ] || [ ! -f "$b" ]; then
        printf '%-18s %-7s %-9s %-8s %s\n' "$seq" "$mode" "$expect" MISSING \
               "one side produced no trajectory"
        RC=1; continue
      fi
      local out verdict detail
      out="$(compare "$a" "$b")"
      verdict="${out%%|*}"; detail="${out#*|}"
      printf '%-18s %-7s %-9s %-8s %s\n' "$seq" "$mode" "$expect" "$verdict" "$detail"
      # Under the exact gate a static sequence that is not `exact` is a bug, and a
      # dynamic one that *is* exact means the dispatcher never took the branch.
      if [ "$strict" = 1 ]; then
        if [ "$expect" = static ] && [ "$verdict" != exact ]; then RC=1; fi
        if [ "$expect" = dynamic ] && [ "$verdict" = exact ]; then RC=1; fi
      fi
    done
  done
}

RC=0
report "GATE: temporal calibration frozen (P.td = 0) -- must be exact" \
       "$OUT/frz_off" "$OUT/frz_on" 1
if [ "$SHIP" = 1 ]; then
  report "REPORT: shipped config -- td is live, so a shared pose may move" \
         "$OUT/ship_off" "$OUT/ship_on" 0
  echo
  echo "--- ATE, shipped config (from the scoring pass)"
  for v in ship_off ship_on; do
    echo "  [$v]"
    sed -n '/^seq/,$p' "$OUT/$v.log" | sed 's/^/  /'
  done
fi

echo
echo "results under $OUT"
[ "$RC" = 0 ] && echo "GATE PASSED" || echo "GATE FAILED"
exit $RC
