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
    ap.add_argument('--P_Wsg', type=float, default=0.002,
                    help='initial variance of the two-degree-of-freedom gravity '
                         'direction state. The shipped TUM-VI value is 3.01, a '
                         'prior std of 1.73 rad, which lets the filter absorb '
                         'early vision residuals by rotating its estimate of '
                         'which way is down rather than by correcting the pose '
                         '-- and a wrong gravity direction then feeds straight '
                         'back into the acceleration estimate, so it is a '
                         'positive feedback loop. That is what made the M3 '
                         'baseline diverge intermittently under a 1e-6 m/s '
                         'velocity jitter: 15 of 66 stereo runs, and 5 of 6 on '
                         'V1_03. Tightening it removes every divergence. '
                         '(default: %(default)s, a prior std of 0.045 rad)')
    ap.add_argument('--visual_meas_std', type=float, default=1.2,
                    help='pixel measurement std for the in-state feature '
                         'update -- with --adapt_visual_meas (the default), '
                         'only the value the online estimate STARTS from. '
                         'Fixed, no value works for all 11 sequences: the five '
                         'Machine Hall ones want 0.75 px and the six Vicon room '
                         'ones want 2.4, and each choice costs the other about '
                         '40%% of its ATE (full-11 means 0.322 at 0.75, 0.138 '
                         'at 2.4, against 0.098 for a per-sequence oracle). '
                         'Machine Hall is slow and sharp, the Vicon room is '
                         'fast and motion-blurred, so the real tracking noise '
                         'differs by about 3x and one number cannot describe '
                         'both. 1.2 is a neutral place to start the estimate '
                         'from, roughly the geometric mean of the two. '
                         '(default: %(default)s)')
    ap.add_argument('--adapt_visual_meas', action='store_true', default=True,
                    help='estimate visual_meas_std online from the Mahalanobis '
                         'distances, which are chi-square(2) exactly when the '
                         'assumed noise is right. On by default here; see '
                         '--no_adapt_visual_meas.')
    ap.add_argument('--no_adapt_visual_meas', dest='adapt_visual_meas',
                    action='store_false',
                    help='pin visual_meas_std to the value given, the way every '
                         'shipped config does.')
    ap.add_argument('--adapt_alpha', type=float, default=0.15,
                    help='step size of the geometric EMA on the estimate, per '
                         'update: a time constant of ~1/alpha updates, so 0.15 '
                         'is 0.33 s at EuRoC\'s 20 Hz. THIS IS THE LOAD-BEARING '
                         'ONE, and only because it has to be fast enough. The '
                         'Vicon room\'s motion blur comes in sub-second bursts, '
                         'so at the 1 s time constant this first used the '
                         'estimate was still climbing when a burst ended and '
                         'still falling when the next began -- it never reached '
                         'the value it was heading for (V2_01 excursed to 2.11 '
                         'px and ended back on the 0.6 floor). Full-11 means: '
                         '0.163 at 0.02, 0.144 at 0.05, 0.095 at 0.15, 0.097 at '
                         '0.30, 0.099 at 0.50 -- a knee between 0.05 and 0.15 '
                         'and flat above it, so pick the flat region rather than '
                         'the exact minimum. (default: %(default)s)')
    ap.add_argument('--adapt_min_std', type=float, default=0.6,
                    help='floor on the estimate, in px. (default: %(default)s)')
    ap.add_argument('--adapt_max_std', type=float, default=4.0,
                    help='ceiling on the estimate, in px. The ceiling is the '
                         'load-bearing bound: the Mahalanobis gate radius grows '
                         'with the estimate, so an unbounded upward walk would '
                         'admit progressively worse measurements. '
                         '(default: %(default)s)')
    ap.add_argument('--MH_thresh', type=float, default=5.991,
                    help='Mahalanobis gate threshold, a chi-square(2) quantile; '
                         '5.991 is the 95%% one. Worth stating explicitly '
                         'because this and --visual_meas_std are not '
                         'independent: the gate is on r\' (H P H\' + R)^-1 r, so '
                         'the measurement std sets the gate radius as well as '
                         'the weight, and at a fixed std the two halves of '
                         'EuRoC want different thresholds for the same reason '
                         'they want different stds (full-11 means at '
                         'std = 0.75: 0.322 at 5.991, 0.125 at 30, 0.154 at '
                         '80). Adapting the std moves the gate with it, so this '
                         'can stay at the textbook value -- and measurably '
                         'should: opening it to 12 *on top of* the adaptation '
                         'costs 0.095 -> 0.099, because both knobs widen the '
                         'same effective gate and doing both over-widens it on '
                         'the sharp Machine Hall sequences (MH_02 0.038 -> '
                         '0.069, MH_04 0.095 -> 0.128). (default: %(default)s)')
    ap.add_argument('--MH_max_strikes', type=int, default=1,
                    help='how many CONSECUTIVE gate failures destroy an in-state '
                         'feature. 1 is the original policy. (default: '
                         '%(default)s)')
    ap.add_argument('--gravity_init_max_accel_dev', type=float, default=0.1,
                    help='reject an accel sample from gravity initialization '
                         'when | |a| - |g| | exceeds this, in m/s^2; 0 '
                         'disables. MH_01_easy is already being carried when '
                         'its first IMU sample lands, so its 20-sample window '
                         'averages to 8.347 m/s^2 against a gravity of 9.810; '
                         'the other ten sequences are within 0.30. At 0.1 the '
                         'gate moves MH_01 (0.145 -> 0.118) and leaves the '
                         'others identical to four decimals. '
                         '(default: %(default)s)')
    ap.add_argument('--noise_scale', type=float, default=1.0,
                    help='multiply the gyro/accel noise densities by this. The '
                         'shipped TUM-VI config uses 1.5x its datasheet values; '
                         'default here is the datasheet itself (%(default)s).')
    ap.add_argument('--bias_scale', type=float, default=1.0,
                    help='multiply the two random-walk densities by this. The '
                         'shipped TUM-VI config uses ~0.3x (%(default)s).')

    # --- front end: the knobs the efficiency pass moves ------------------------
    # These are written out explicitly even when left at their default. Two
    # reasons. Reading a shipped config should not require reading tracker.cpp to
    # find out what `clahe_grid_size` actually is; and sweep_xivo.sh deliberately
    # refuses to patch a key that is absent from the config, because inventing one
    # is how a sweep silently measures the control eleven times -- so a knob that
    # is not spelled out here cannot be screened at all.
    ap.add_argument('--histogram_method', default='NONE',
                    choices=['NONE', 'HISTOGRAM', 'CLAHE', 'GAINMAP'],
                    help='pre-equalization. CLAHE is the single most expensive '
                         'front-end stage on EuRoC -- 2.06 of the 6.26 ms/frame '
                         'front end, all of it the bilinear interpolation pass, '
                         'so no grid size makes it cheap -- and what it buys is '
                         'not supply or match quality but the *distribution* of '
                         'corners, by lifting dark regions over --fast_threshold. '
                         'NONE plus --fast_threshold 7 buys that back for 0.36 '
                         'ms/frame instead of 2.06. See '
                         'notes-euroc/m5-xivo-efficiency-tuning.md sec 5 and 8.2. '
                         '(default: %(default)s)')
    ap.add_argument('--fast_threshold', type=int, default=7,
                    help='FAST detector threshold. Tuned jointly with '
                         '--histogram_method, not independently: 20 on a CLAHE\'d '
                         'image and 7 on the raw one yield near-identical '
                         'candidate supply (6913 vs 6357 per detecting frame). '
                         'Leaving this at 20 with NONE gives 1615, which is '
                         'still ample for --num_features_max but costs 0.007 m '
                         'of ATE and, together with --klt_max_level 2, makes '
                         'V2_02_medium intermittently diverge. '
                         '(default: %(default)s)')
    ap.add_argument('--equalize_for', default='ALL', choices=['ALL', 'DETECT'],
                    help='whether equalization feeds everything or only the '
                         'detector. DETECT with CLAHE is actively unsafe on '
                         'EuRoC -- V2_03_difficult reaches ATE 1.53 -- because '
                         'CLAHE is local and non-monotonic, so the corners it '
                         'finds do not sit on gradients of the raw image the KLT '
                         'then tracks. (default: %(default)s)')
    ap.add_argument('--clahe_clip_limit', type=float, default=10.0,
                    help='CLAHE contrast clip; the tracker.cpp default '
                         '(%(default)s)')
    ap.add_argument('--clahe_grid_size', type=int, default=8,
                    help='CLAHE tile grid, per side; the tracker.cpp default '
                         '(%(default)s)')
    ap.add_argument('--klt_win_size', type=int, default=15,
                    help='KLT window, per side (default: %(default)s)')
    ap.add_argument('--klt_max_level', type=int, default=2,
                    help='KLT pyramid levels minus one. 2 is accuracy-neutral '
                         'on all 11 sequences and saves 0.56 ms/frame plus 4.7 '
                         'MB of pyramid. (default: %(default)s)')
    ap.add_argument('--klt_max_iter', type=int, default=30,
                    help='KLT iterations per level. Do not lower this without '
                         'remeasuring: 15 costs +0.001 m of ATE with CLAHE on '
                         'and +0.006 with it off, because a lower-contrast image '
                         'needs more iterations to converge. Knob prices on this '
                         'front end are not independent of each other. '
                         '(default: %(default)s)')
    ap.add_argument('--num_features_max', type=int, default=180,
                    help='feature supply target (default: %(default)s)')
    ap.add_argument('--num_features_min', type=int, default=135,
                    help='detect when the live count falls below this '
                         '(default: %(default)s)')
    ap.add_argument('--stereo_max_level', type=int, default=2,
                    help='pyramid levels minus one for the left->right match. '
                         'Kept <= --klt_max_level so MatchStereo can reuse the '
                         'temporal pyramid instead of building its own. '
                         '(default: %(default)s)')
    ap.add_argument('--seed_prev_disparity', action='store_true',
                    help='seed the left->right KLT from the previous frame\'s '
                         'disparity. Raises the stereo match rate 79.2 -> 82.0%% '
                         'and saves 0.21 ms/frame, at +0.002 m ATE.')
    ap.add_argument('--fast_png_decode', action='store_true', default=False,
                    help='use XIVO\'s own PNG decoder instead of cv::imdecode. '
                         'Off by default here, unlike TUM-VI: the fast path wins '
                         'when there is a 16->8 bit strip to fuse into the '
                         'unfilter, and EuRoC is 8-bit and near-incompressible, '
                         'so it is a 0.30 ms/frame *loss*. Output is '
                         'bit-identical either way (results/euroc_pngcheck).')
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
    cfg['P']['Wsg'] = args.P_Wsg
    cfg['visual_meas_std'] = args.visual_meas_std
    cfg['visual_meas_adapt'] = {
        "comment": ("estimate visual_meas_std online from the median "
                    "Mahalanobis distance; see notes-euroc/"
                    "m4-xivo-accuracy-tuning.md"),
        "enable": args.adapt_visual_meas,
        "alpha": args.adapt_alpha,
        "min_std": args.adapt_min_std,
        "max_std": args.adapt_max_std,
        "warmup_updates": 20,
        "min_samples": 10,
    }
    cfg['MH_thresh'] = args.MH_thresh
    cfg['MH_max_strikes'] = args.MH_max_strikes
    cfg['gravity_init_max_accel_dev'] = args.gravity_init_max_accel_dev

    tc = cfg['tracker_cfg']
    tc['histogram_method'] = args.histogram_method
    tc['equalize_for'] = args.equalize_for
    tc['clahe_clip_limit'] = args.clahe_clip_limit
    tc['clahe_grid_size'] = args.clahe_grid_size
    tc['num_features_max'] = args.num_features_max
    tc['num_features_min'] = args.num_features_min
    tc['FAST']['threshold'] = args.fast_threshold
    tc['KLT']['win_size'] = args.klt_win_size
    tc['KLT']['max_level'] = args.klt_max_level
    tc['KLT']['max_iter'] = args.klt_max_iter
    cfg['fast_png_decode'] = args.fast_png_decode
    if args.stereo_max_level > args.klt_max_level:
        # MatchStereo reuses the temporal pyramid when the stereo match needs no
        # more levels and no larger window than the KLT already built; violating
        # that silently doubles the pyramid work instead of erroring.
        raise SystemExit(
            f'--stereo_max_level {args.stereo_max_level} exceeds --klt_max_level '
            f'{args.klt_max_level}, which forces MatchStereo to build a second '
            'pyramid')

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
        tc['stereo_matching']['max_level'] = args.stereo_max_level
        tc['stereo_matching']['seed_prev_disparity'] = args.seed_prev_disparity

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
          f"P.ba = {args.P_ba} (std {args.P_ba ** 0.5:.4g} m/s^2), "
          f"P.Wsg = {args.P_Wsg} (std {args.P_Wsg ** 0.5:.4g} rad)")
    if args.adapt_visual_meas:
        print(f"  visual_meas_std = {args.visual_meas_std} px, adapting online "
              f"in [{args.adapt_min_std}, {args.adapt_max_std}] "
              f"at alpha = {args.adapt_alpha}")
    else:
        print(f"  visual_meas_std = {args.visual_meas_std} px, fixed")
    print(f"  MH_thresh = {args.MH_thresh}, "
          f"MH_max_strikes = {args.MH_max_strikes}, "
          f"gravity_init_max_accel_dev = "
          f"{args.gravity_init_max_accel_dev} m/s^2")


if __name__ == '__main__':
    sys.exit(main())
