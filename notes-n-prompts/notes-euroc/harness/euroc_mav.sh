# Dataset profile: EuRoC MAV, all 11 sequences (machine hall + vicon room 1/2).
# Sourced by run_openvins.sh.
#
# Unlike TUM-VI these are 752x480 pinhole+radtan, and groundtruth is the
# whole-trajectory Leica/Vicon+IMU fusion in state_groundtruth_estimate0 rather
# than a mocap-room-only track, so every pose is scorable.
#
# The authors' shipped euroc_mav config is used unmodified and is the same for
# all 11 sequences (no per-sequence init_imu_thresh as TUM-VI room6 needs), which
# is what makes this a fair one-config-for-all comparison on both sides.

PROFILE_NAME="euroc_mav"
PROFILE_ROOT="$WORKSPACE/data/euroc"
PROFILE_SEQS="MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult"

# Where the mav0/ folder for a sequence lives
seq_folder() { echo "$PROFILE_ROOT/${1}"; }

# ASL-format groundtruth csv (converted to TUM by the runner). EuRoC's
# state_groundtruth_estimate0 has the same leading columns as mocap0
# (ts, p_xyz, q_wxyz) so asl_gt_to_tum.awk handles it as-is.
seq_gt_csv() { echo "$(seq_folder "$1")/mav0/state_groundtruth_estimate0/data.csv"; }

# Estimator config. Stereo is the authors' shipped file; mono is that file with
# max_cameras/use_stereo flipped, kept next to a copy of the kalibr chains
# because OpenVINS resolves relative_config_* against the config's own directory.
seq_config() {
  local mode="$2"
  if [ "$mode" = "mono" ]; then
    echo "$HERE/configs/euroc_mav_mono/estimator_config.yaml"
  else
    echo "$OV_REPO/config/euroc_mav/estimator_config.yaml"
  fi
}

# No per-sequence overrides: one configuration for all 11 sequences.
seq_extra() { echo ""; }

# --- XIVO side ------------------------------------------------------------
# EuRoC needs its own loader in scripts/pyxivo.py (the ASL folder sits directly
# under the sequence name) and its own config, generated from the dataset's
# sensor.yaml files by scripts/make_euroc_cfg.py. One config for all 11
# sequences, matching what OpenVINS does here.
PROFILE_XIVO_DATASET="euroc"
PROFILE_XIVO_CFG_PREFIX="euroc"
