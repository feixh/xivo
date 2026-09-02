#!/bin/bash
# OpenVINS EuRoC accuracy ensemble.
#
# OpenVINS is deterministic: repeating a run reproduces the trajectory bit for
# bit, so `--repeats` measures nothing. To get an honest error bar we perturb a
# knob that is physically neutral at the size of the perturbation -- gravity
# magnitude in its 9th significant digit, 9.81 -> 9.810000001 -- which changes no
# physics but does reshuffle the order in which gating decisions land. Member 0
# is the unperturbed shipped value, so the ensemble contains the canonical run.
#
# `--init_dyn_use 1` is applied to every member and every sequence. With the
# shipped config (static initializer only) OpenVINS diverges on MH_04_difficult
# in 6 of 6 members: MH_04 begins with a take-off/hover/land, and because the
# Machine Hall scene is far away, a 0.4 m/s hover produces less than the
# `init_max_disparity: 10.0` pixels of disparity, so the platform reads as
# stationary and the static initializer asserts zero velocity while it is
# moving. Enabling OpenVINS' own dynamic initializer fixes MH_04 and improves
# MH_01/02/03/05, while leaving all six V sequences bit-identical (they start
# genuinely still, so static init is still selected and the dynamic path never
# runs). See notes-euroc/m1-openvins-baseline.md.
set -e
cd "$(dirname "$0")"

MEMBERS=(9.81 9.810000001 9.810000002 9.810000003 9.810000004 9.810000005)
OUT=../results/euroc_ov_acc_dyn

for mode in stereo mono; do
  for i in "${!MEMBERS[@]}"; do
    g="${MEMBERS[$i]}"
    echo "=== $mode member $i (gravity_mag=$g)"
    ./run_openvins.sh --profile euroc_mav --mode "$mode" --cpus-per-run 8 \
      --extra "--init_dyn_use 1 --gravity_mag $g" \
      --out "$OUT/${mode}_m${i}"
  done
done
echo "ALL DONE"
