#!/usr/bin/env python3
"""Generate a XIVO stereo config from a TUM-VI `dso/camchain.yaml`.

The stereo intrinsics and the left->right extrinsic are derived directly from
the dataset's own calibration file rather than transcribed by hand, so the
config is traceable to its source and cannot drift from it.

Usage:
  make_stereo_cfg.py --base cfg/sweep_dlt_nodesc.json \
                     --camchain ../data/tumvi/dataset-room1_512_16/dso/camchain.yaml \
                     --out cfg/tumvi_stereo.json

The base config supplies everything non-stereo (IMU noise, EKF sizes, tracker
settings, ...) and is copied through unchanged apart from the added stereo keys.
Comments in the base JSON are stripped, since Python's json module rejects them.
"""
import argparse
import json
import re
import sys


def load_jsonc(path):
    """Load JSON that may contain // comments (as XIVO's configs do)."""
    with open(path) as f:
        text = f.read()
    # Strip // comments, but not inside strings. XIVO's configs have no URLs or
    # other embedded '//', so a conservative line-wise strip is safe; assert
    # that the result parses to catch any surprise.
    text = re.sub(r'(?m)//.*$', '', text)
    return json.loads(text)


def parse_camchain(path):
    """Read a kalibr camchain.yaml (as TUM-VI ships under dso/)."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def camera_block(cam, comment):
    """Build a XIVO `camera_cfg` block from one camchain camera entry."""
    model = cam['distortion_model']
    if model != 'equidistant':
        raise SystemExit(
            f"unsupported distortion_model {model!r}; TUM-VI is equidistant")
    fx, fy, cx, cy = cam['intrinsics']
    rows, cols = cam['resolution'][1], cam['resolution'][0]
    return {
        "comment": comment,
        "model": "equidistant",
        "max_iter": 15,
        "rows": rows,
        "cols": cols,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k0123": list(cam['distortion_coeffs']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True,
                    help='monocular config to extend')
    ap.add_argument('--camchain', required=True,
                    help='TUM-VI dso/camchain.yaml')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = load_jsonc(args.base)
    cams = parse_camchain(args.camchain)

    if 'cam0' not in cams or 'cam1' not in cams:
        raise SystemExit(f"camchain is missing cam0/cam1: {list(cams)}")

    T_cn_cnm1 = cams['cam1'].get('T_cn_cnm1')
    if not T_cn_cnm1 or len(T_cn_cnm1) != 4:
        raise SystemExit("cam1.T_cn_cnm1 missing or malformed; cannot build rig")

    cfg['camera_cfg'] = camera_block(cams['cam0'], '512-cam0 (from camchain)')
    cfg['camera1_cfg'] = camera_block(cams['cam1'], '512-cam1 (from camchain)')
    cfg['stereo'] = True

    # Seed a new feature's depth from its stereo pair instead of `initial_z`.
    # See notes-stereo/m4-stereo-depth-init.md for how these were chosen.
    cfg['stereo_init'] = {
        "comment": "metric depth at first observation; see notes-stereo/m4",
        "enable": True,
        # Assumed left->right match error, in pixels, used to propagate a
        # log-depth std. Swept over [0.10, 1.0] on all six rooms: a broad
        # optimum near 0.15, clearly worse at >=0.5. Note this is *tighter*
        # than a defensible estimate of true sub-pixel KLT error -- it is
        # acting partly as a tuning knob on how much the filter trusts the
        # seed. See notes-stereo/m4-stereo-depth-init.md.
        "sigma_px": 0.15,
        # Metres; reject a triangulation whose rays miss by more than this.
        # Measured to be non-binding at 0.10 (raising it to 0.30 gives a
        # byte-identical trajectory), and tightening it to 0.02 hurts. Kept as
        # insurance against a miscalibrated rig rather than as a live gate.
        "max_gap": 0.10,
        # clamp on the seeded log-depth std
        "min_std_z": 0.01,
        "max_std_z": 1.0,
    }

    cfg['stereo_update'] = {
        "comment": "right-camera rows in the EKF update; see notes-stereo/m5",
        "enable": True,
        # Variance of a right pixel as a multiple of `visual_meas_std`^2. A
        # ratio rather than an absolute so that re-tuning the left camera's
        # noise keeps the two cameras' relative weighting intact.
        "R_scale": 1.0,
        # Threshold of the (separate, 2-dof) right-camera Mahalanobis gate as a
        # multiple of MH_thresh. Rejecting here costs a feature its right
        # measurement only, not its place in the state.
        "mh_scale": 1.0,
    }

    # Left->right matching gates. Defaults live here rather than only in
    # tracker.cpp so a sweep can override them without a rebuild; see
    # notes-stereo/m3-stereo-tracking.md for how each was chosen.
    cfg.setdefault('tracker_cfg', {})
    if isinstance(cfg['tracker_cfg'], dict):
        cfg['tracker_cfg']['stereo_matching'] = {
            "comment": "gates on the left->right KLT match; see notes-stereo/m3",
            # radians of angular epipolar miss
            "epipolar_thresh": 0.005,
            # pixels of left->right->left round-trip error
            "circular_thresh": 1.0,
            # pixels; below this there is no usable parallax
            "min_disparity": 1.0,
            # pixels; a 10 cm baseline cannot produce more than this
            "max_disparity": 150.0,
        }
    # kalibr's T_cn_cnm1 maps cam(n-1) -> cam(n), i.e. cam0 -> cam1 here, which
    # is exactly what StereoRig's "T_c1c0" key expects.
    cfg['stereo_cfg'] = {
        "comment": "T_cn_cnm1 from camchain.yaml: maps a point cam0 -> cam1",
        "T_c1c0": T_cn_cnm1,
    }

    baseline = sum(T_cn_cnm1[i][3] ** 2 for i in range(3)) ** 0.5
    with open(args.out, 'w') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')

    print(f"wrote {args.out}")
    print(f"  cam0 fx={cams['cam0']['intrinsics'][0]:.4f} "
          f"cam1 fx={cams['cam1']['intrinsics'][0]:.4f}")
    print(f"  baseline = {baseline*1000:.2f} mm")


if __name__ == '__main__':
    sys.exit(main())
