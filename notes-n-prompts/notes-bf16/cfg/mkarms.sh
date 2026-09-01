#!/bin/bash
# Generate the arm configs: each is a shipped config with one
# `kernel_precision` block inserted, so the arms differ in nothing else.
#
# Written rather than hand-edited because there are N arms x 2 modes x 2 variants
# (plain for the accuracy ensembles, `_timing` for the FPS harness), and the
# shipped configs carry `//` comments that a JSON round-trip would drop.
#
# Usage: bash mkarms.sh
set -eu
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W=/home/ubuntu/workspace/auto-slam-engineer
mkdir -p "$H/arms"

# arm: joseph innovation gating batch_gating covariance_form
declare -A ARMS=(
  # All-fp64 with the gating sweep left per-feature. Not a no-op: the update
  # itself no longer computes H*P twice. This arm is what separates the
  # refactor from the precision.
  [f64]="f64 f64 f64 false"
  # Blocking only: the gating sweep batched into one product, still fp64.
  [batch]="f64 f64 f64 true"
  # Precision in the gating sweep only, batched.
  [gate_bf16]="f64 f64 bf16 true"
  # Precision in the covariance rotation only: K*H, (KH-I) P (KH-I)^T, K R K^T.
  # Nothing here reaches the state estimate on this frame -- only the covariance,
  # and hence the *next* frame's gain. The theoretically safe arm.
  [jos_bf16]="bf16 f64 bf16 true"
  # Precision in the gain path only: H P and (H P) H^T, which feed S.ldlt() and
  # therefore err_ = K inn_ directly. The theoretically unsafe arm.
  [inn_bf16]="f64 bf16 bf16 true"
  # Precision everywhere.
  [bf16]="bf16 bf16 bf16 true"
  # The fp32 rung of the same kernels, for the 2.00x-vs-2.67x comparison.
  [f32]="f32 f32 f32 true"

  # ---- M4: the short covariance form, P - K (H P) ----------------------------
  # The flop win on its own, no precision change. 4.1x fewer multiply-adds in
  # the update than the Joseph form, arithmetic still fp64.
  [short]="f64 f64 f64 true short"
  # The arm this project is aiming at: the *correction* computed at bf16 and
  # subtracted in fp64, with the gain path left alone. The error injected into
  # the covariance is 3e-3 of |dP| rather than 3e-3 of |P|.
  [short_bf16c]="bf16 f64 bf16 true short"
  # ... and with the gain path narrowed too, which M3 showed is the dangerous one.
  [short_bf16]="bf16 bf16 bf16 true short"
  # The fp32 rung of the short form.
  [short_f32]="f32 f32 f32 true short"
  # Isolations: exactly one product narrowed, everything else fp64.
  [short_jbf16]="bf16 f64 f64 true short"
  [short_jf32]="f32 f64 f64 true short"
  [short_gbf16]="f64 f64 bf16 true short"
  # The two candidate deliverables. Both take the short form, batch the gating
  # sweeps and run them at bf16; they differ in whether the *gain* path -- H P,
  # H P H^T, the products that reach err_ through S.ldlt() -- is narrowed too.
  [short_f32c_gbf16]="f32 f64 bf16 true short"
  [short_f32_gbf16]="f32 f32 bf16 true short"
)

for arm in "${!ARMS[@]}"; do
  # shellcheck disable=SC2086  # deliberate word splitting of the arm spec
  set -- ${ARMS[$arm]}
  j=$1 i=$2 g=$3 b=$4 form=${5:-joseph}
  blk="  \"kernel_precision\": {\"joseph\": \"$j\", \"innovation\": \"$i\", \"gating\": \"$g\", \"batch_gating\": $b, \"covariance_form\": \"$form\"},"
  for src in "$W/xivo-bf16/cfg/tumvi_mono_ctl_oos.json" \
             "$W/xivo-bf16/cfg/tumvi_stereo_oos.json" \
             "$H/tumvi_mono_ctl_oos_timing.json" \
             "$H/tumvi_stereo_oos_timing.json"; do
    base=$(basename "$src" .json)
    out="$H/arms/${base}_${arm}.json"
    # After the first `{` and before everything else. The shipped configs now
    # carry a `kernel_precision` block of their own, and a duplicate key would be
    # the *last* one to win in jsoncpp -- i.e. the shipped one, silently making
    # every arm identical -- so drop the existing block first. It is a single
    # object at one level of indentation, so "from the key to the closing brace
    # at the same indent" is exact, and the leading comment goes with it.
    awk -v blk="$blk" '
      /^  \/\/ Dense linear algebra in the EKF update\./ {incomment=1}
      incomment && !/^  \/\// {incomment=0}
      incomment {next}
      /^  "kernel_precision": \{/ {drop=1}
      drop {if (/^  \},$/) drop=0; next}
      {print}
      NR==1 && /^\{/ {print blk}
    ' "$src" > "$out"
  done
done
ls "$H/arms" | wc -l
