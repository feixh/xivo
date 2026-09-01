#!/usr/bin/env python3
"""Generate a XIVO config for EuRoC MAV from the dataset's own sensor.yaml files.

EuRoC ships its calibration per sensor, under
`<seq>/mav0/{cam0,cam1,imu0}/sensor.yaml`, and those files are byte-identical
across all 11 sequences -- so one generated config serves the whole dataset and
is traceable to the source rather than transcribed by hand.

What comes from the dataset:
  * cam0/cam1 pinhole intrinsics and radial-tangential distortion,
  * X.Wbc / X.Tbc, the body<-camera transform, from cam0's `T_BS`,
  * stereo_cfg.T_c1c0 = T_BS(cam1)^-1 T_BS(cam0),
  * Qimu, from imu0's four noise densities.

What comes from flags (because it is scene geometry or tuning, not calibration):
  gravity magnitude, the depth window, and optional scale factors on the IMU
  noise densities.

Conventions, both verified numerically against TUM-VI's camchain and XIVO's
shipped TUM-VI config:
  * `X.Wbc` given as a 3x3 is R_body_from_camera (estimator.cpp reads a 3-vector
    as so(3) and falls back to a row-major matrix), and `X.Tbc` is the matching
    translation -- i.e. together they are T_imu_cam, which is exactly EuRoC's
    `T_BS`. No inversion is needed here, and applying one is the natural mistake.
  * `stereo_cfg.T_c1c0` maps a point from cam0 into cam1.
  * EuRoC's body frame *is* imu0 (imu0's own `T_BS` is identity), which is also
    the frame `state_groundtruth_estimate0` reports and the frame XIVO's `gsb`
    estimates, so nothing needs re-framing for scoring either.

Usage:
  make_euroc_cfg.py --base cfg/eff_stereo.json \
                    --seqdir ../data/euroc/MH_01_easy \
                    --out cfg/euroc_stereo.json
  make_euroc_cfg.py --base cfg/eff_mono.json --mono \
                    --seqdir ../data/euroc/MH_01_easy \
                    --out cfg/euroc_mono.json
"""
import argparse
import json
import re
import sys

import numpy as np
import yaml


def load_jsonc(path):
    """Load JSON that may contain // comments (as XIVO's configs do)."""
    with open(path) as f:
        text = f.read()
    return json.loads(re.sub(r'(?m)//.*$', '', text))


def load_sensor(seqdir, sensor):
    with open(f'{seqdir}/mav0/{sensor}/sensor.yaml') as f:
        return yaml.safe_load(f)


def T_BS(sensor_yaml):
    """The 4x4 body<-sensor transform from a sensor.yaml."""
    d = sensor_yaml['T_BS']
    if (d['rows'], d['cols']) != (4, 4):
        raise SystemExit(f"T_BS is {d['rows']}x{d['cols']}, expected 4x4")
    return np.array(d['data'], dtype=float).reshape(4, 4)


def camera_block(cam, comment):
    """A XIVO `camera_cfg` block from one EuRoC camera sensor.yaml."""
    if cam['camera_model'] != 'pinhole':
        raise SystemExit(f"unexpected camera_model {cam['camera_model']!r}")
    if cam['distortion_model'] != 'radial-tangential':
        raise SystemExit(
            f"unexpected distortion_model {cam['distortion_model']!r}")
    fx, fy, cx, cy = cam['intrinsics']
    cols, rows = cam['resolution']
    d = list(cam['distortion_coefficients'])
    if len(d) != 4:
        raise SystemExit(f"expected 4 radtan coefficients, got {len(d)}")
    k1, k2, p1, p2 = d
    # XIVO's RadTan takes three radial coefficients in `k012` and the two
    # tangential ones separately. Its expansion (common/camera_radtan.h) is
    # x(1+k1 r^2+k2 r^4+k3 r^6) + 2 p1 x y + p2(r^2+2x^2), which is OpenCV's and
    # Kalibr's ordering, so [k1, k2, p1, p2] maps across directly with k3 = 0.
    return {
        "comment": comment,
        "model": "radtan",
        "max_iter": 15,
        "rows": rows,
        "cols": cols,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k012": [k1, k2, 0.0],
        "p1": p1,
        "p2": p2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='config to extend')
    ap.add_argument('--seqdir', required=True,
                    help='an EuRoC sequence directory (any of them: the '
                         'calibration is identical across all 11)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mono', action='store_true',
                    help='mono config: drop the stereo keys entirely')
    ap.add_argument('--gravity', type=float, default=9.81,
                    help='gravity magnitude; 9.81 is what OpenVINS uses on '
                         'EuRoC (default: %(default)s)')
    ap.add_argument('--min_depth', type=float, default=0.05,
                    help='min_depth / triangulation.zmin (default: %(default)s)')
    ap.add_argument('--max_depth', type=float, default=60.0,
                    help='max_depth / triangulation.zmax. A feature whose depth '
                         'falls outside this window is not merely down-weighted, '
                         'it is refused as an instate candidate '
                         '(Criteria::Candidate) and as a stereo seed '
                         '(manager.cpp), so on EuRoC the TUM-VI value of 5 m is '
                         'not survivable: Machine Hall diverges outright at 5 m '
                         '(ATE 23937 m), reaches 3.6 m at 15-30 m and 0.099 m at '
                         '60 m. Vicon room is flat from 60 m up. '
                         '(default: %(default)s)')
    ap.add_argument('--initial_z', type=float, default=5.0,
                    help='initial_z, the depth prior for a new feature before '
                         'stereo or the subfilter says otherwise. EuRoC scenes '
                         'are deeper than a TUM-VI room; adaptive_initial_depth '
                         'takes over within a few frames. (default: %(default)s)')
    ap.add_argument('--P_bg', type=float, default=0.01,
                    help='initial variance of the gyroscope bias. THIS IS THE '
                         'LOAD-BEARING ONE ON EuRoC. The shipped TUM-VI value is '
                         '1e-4, i.e. a prior std of 0.01 rad/s, but EuRoC\'s own '
                         'groundtruth reports gyro biases up to 0.076 rad/s -- '
                         'seven times the prior. The filter therefore cannot '
                         'correct the bias it actually has, attitude drifts at '
                         'about 4 deg/s, gravity leaks into acceleration and the '
                         'position runs away quadratically from the first frame; '
                         'no feature survives long enough to be promoted, so '
                         'vision never gets the chance to fix it. On V1_01 this '
                         'one number is ATE 22594 m -> 0.071 m. The default here '
                         'is a prior std of 0.1 rad/s, which covers the observed '
                         'range. (default: %(default)s)')
    ap.add_argument('--P_ba', type=float, default=0.25,
                    help='initial variance of the accelerometer bias. Same story, '
                         'smaller effect: EuRoC groundtruth reports accel biases '
                         'up to 0.55 m/s^2 against a TUM-VI prior std of 0.032. '
                         'The default here is a prior std of 0.5 m/s^2. Worth '
                         '0.071 -> 0.069 m on V1_01 on top of --P_bg. '
                         '(default: %(default)s)')
    ap.add_argument('--noise_scale', type=float, default=1.0,
                    help='multiply the gyro/accel noise densities by this. The '
                         'shipped TUM-VI config uses 1.5x its datasheet values; '
                         'default here is the datasheet itself (%(default)s).')
    ap.add_argument('--bias_scale', type=float, default=1.0,
                    help='multiply the two random-walk densities by this. The '
                         'shipped TUM-VI config uses ~0.3x (%(default)s).')
    args = ap.parse_args()

    cfg = load_jsonc(args.base)
    cam0 = load_sensor(args.seqdir, 'cam0')
    cam1 = load_sensor(args.seqdir, 'cam1')
    imu0 = load_sensor(args.seqdir, 'imu0')

    # EuRoC defines the body frame as imu0. Assert it rather than assume it: if a
    # future release moved the body frame, every extrinsic below would be silently
    # wrong by that transform.
    if not np.allclose(T_BS(imu0), np.eye(4), atol=1e-12):
        raise SystemExit('imu0 T_BS is not identity; body frame is not imu0 and '
                         'every extrinsic in this script would need rebasing')

    T_b_c0 = T_BS(cam0)
    T_b_c1 = T_BS(cam1)

    cfg['camera_cfg'] = camera_block(cam0, 'EuRoC cam0 (from mav0/cam0/sensor.yaml)')

    cfg['X']['Wbc'] = [list(map(float, r)) for r in T_b_c0[:3, :3]]
    cfg['X']['Tbc'] = [float(x) for x in T_b_c0[:3, 3]]

    cfg['gravity'] = [0.0, 0.0, -args.gravity]

    # Qimu holds standard deviations (estimator.cpp squares the whole block), so
    # these are the continuous-time noise densities straight from the dataset.
    g_n = float(imu0['gyroscope_noise_density']) * args.noise_scale
    a_n = float(imu0['accelerometer_noise_density']) * args.noise_scale
    g_w = float(imu0['gyroscope_random_walk']) * args.bias_scale
    a_w = float(imu0['accelerometer_random_walk']) * args.bias_scale
    cfg['Qimu'] = {
        "comment": (f"from mav0/imu0/sensor.yaml, noise x{args.noise_scale:g}, "
                    f"random walk x{args.bias_scale:g}"),
        "gyro": [g_n] * 3,
        "gyro_bias": [g_w] * 3,
        "accel": [a_n] * 3,
        "accel_bias": [a_w] * 3,
    }

    cfg['min_depth'] = args.min_depth
    cfg.setdefault('triangulation', {})['zmin'] = args.min_depth
    cfg['max_depth'] = args.max_depth
    cfg.setdefault('triangulation', {})['zmax'] = args.max_depth
    cfg['initial_z'] = args.initial_z
    cfg['P']['bg'] = args.P_bg
    cfg['P']['ba'] = args.P_ba

    if args.mono:
        cfg['stereo'] = False
        for k in ('camera1_cfg', 'stereo_cfg', 'stereo_init', 'stereo_update'):
            cfg.pop(k, None)
    else:
        cfg['camera1_cfg'] = camera_block(
            cam1, 'EuRoC cam1 (from mav0/cam1/sensor.yaml)')
        cfg['stereo'] = True
        T_c1_c0 = np.linalg.inv(T_b_c1) @ T_b_c0
        cfg['stereo_cfg'] = {
            "comment": "T_BS(cam1)^-1 T_BS(cam0) from sensor.yaml: cam0 -> cam1",
            "T_c1c0": [[float(x) for x in row] for row in T_c1_c0],
        }

    with open(args.out, 'w') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')

    baseline = float(np.linalg.norm(T_b_c1[:3, 3] - T_b_c0[:3, 3]))
    print(f"wrote {args.out}  ({'mono' if args.mono else 'stereo'})")
    print(f"  cam0 fx={cam0['intrinsics'][0]:.3f} cx={cam0['intrinsics'][2]:.3f} "
          f"{cam0['resolution'][0]}x{cam0['resolution'][1]} radtan")
    print(f"  baseline = {baseline * 1000:.2f} mm")
    print(f"  gravity = {args.gravity}, "
          f"depth window = [{cfg['min_depth']}, {cfg['max_depth']}] m, "
          f"initial_z = {cfg['initial_z']}")
    print(f"  Qimu gyro={g_n:.6g} accel={a_n:.6g} "
          f"gyro_bias={g_w:.6g} accel_bias={a_w:.6g}")
    print(f"  P.bg = {args.P_bg} (std {args.P_bg ** 0.5:.4g} rad/s), "
          f"P.ba = {args.P_ba} (std {args.P_ba ** 0.5:.4g} m/s^2)")


if __name__ == '__main__':
    sys.exit(main())
