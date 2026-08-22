import numpy as np
import argparse
import os, sys
from shutil import copyfile
from utils import get_xivo_gt_filename, get_xivo_output_filename



parser = argparse.ArgumentParser()
parser.add_argument("-dump", default=".", type=str)
parser.add_argument("-npts", default=1000, type=int)
parser.add_argument("-xlim", default=[-10, 10], nargs=2, type=float)
parser.add_argument("-ylim", default=[-10, 10], nargs=2, type=float)
parser.add_argument("-zlim", default=[-5, 5], nargs=2, type=float)
parser.add_argument("-pcw_seed", default=0, type=int)
parser.add_argument("-imu_seed", default=1, type=int)
parser.add_argument("-noise_accel", default=1e-4, type=float)
parser.add_argument("-noise_gyro", default=1e-5, type=float)
parser.add_argument("-motion_type", default="calib_traj", type=str)
parser.add_argument("-noise_vision_std", default="1.0", type=float)
parser.add_argument("-total_time", default=100.0, type=float)
parser.add_argument("-imu_dt", default=0.0025, type=float)
parser.add_argument("-vision_dt", default=0.04, type=float)
parser.add_argument('-cfg', type=str, default="cfg/pcw.json")
parser.add_argument('-viewer_cfg', type=str, default="cfg/pcw_viewer.json")
parser.add_argument('-use_viewer', default=False, action='store_true', help='visualize if set')
parser.add_argument('-plot', default=False, action="store_true", help="plot if set")
parser.add_argument('-ate_max_difference', default=0.001, type=float, help="maximum timestamp difference to use when aligning ground-truth and output points in calculating ATE")
parser.add_argument('-stdout', default=False, action='store_true', help='write to stdout instead of a benchmark file if set')
args = parser.parse_args()

if __name__ == '__main__':

    ########################################
    # BACKUP CONFIGURATION
    ########################################
    if not os.path.exists(args.dump):
        os.makedirs(args.dump)

    cfg = args.cfg

    copyfile(cfg, os.path.join(args.dump, os.path.basename(cfg)))

    ########################################
    # RUN THE FILTER
    ########################################
    cmd = 'python scripts/pyxivo_pcw.py \
-dump {dump:} \
-npts {npts:} \
-xlim {xlim:} \
-ylim {ylim:} \
-zlim {zlim:} \
-pcw_seed {pcw_seed} \
-imu_seed {imu_seed} \
-noise_accel {noise_accel:} \
-noise_gyro {noise_gyro:} \
-motion_type {motion_type:} \
-noise_vision_std {noise_vision_std:} \
-total_time {total_time:} \
-imu_dt {imu_dt:} \
-vision_dt {vision_dt:} \
-cfg {cfg:} \
-viewer_cfg {viewer_cfg:} \
{use_viewer:} \
-mode eval'.format(
        dump=args.dump, npts=args.npts,
        xlim="{} {}".format(args.xlim[0], args.xlim[1]),
        ylim="{} {}".format(args.ylim[0], args.ylim[1]),
        zlim="{} {}".format(args.zlim[0], args.zlim[1]),
        pcw_seed=args.pcw_seed, imu_seed=args.imu_seed,
        noise_accel=args.noise_accel, noise_gyro=args.noise_gyro,
        motion_type=args.motion_type, noise_vision_std=args.noise_vision_std,
        total_time=args.total_time, imu_dt=args.imu_dt,
        vision_dt=args.vision_dt, cfg=args.cfg, viewer_cfg=args.viewer_cfg,
        use_viewer=('-use_viewer' if args.use_viewer else ''))
    print('*** COMMAND TO BE EXECUTED ***')
    print(cmd)
    os.system(cmd)

    result_file = "pcw_{}".format(args.motion_type)
    groundtruth_file = "{}_gt".format(result_file)
    benchmark_file = "{}_bench".format(result_file)

    os.system('echo pcw sequence {} >> {}'.format(args.motion_type, benchmark_file))


    ########################################
    # COMPUTE ATE
    ########################################
    cmd = 'python3 scripts/tum_rgbd_benchmark_tools/evaluate_ate.py \
--max_difference {ate_max_difference:} \
{groundtruth_file:} \
{result_file:} {write_to:}'.format(
        groundtruth_file=groundtruth_file,
        result_file=result_file,
        ate_max_difference=args.ate_max_difference,
        write_to='>> {}'.format(benchmark_file) if not args.stdout else '')
    if args.plot:
        cmd += " --plot {}_ate_plot.png".format(result_file)
    print('*** COMMAND TO BE EXECUTED ***')
    print(cmd)
    os.system(cmd)

    ########################################
    # COMPUTE RPE
    ########################################
    cmd = 'python3 scripts/tum_rgbd_benchmark_tools/evaluate_rpe.py \
--fixed_delta \
--delta_unit s \
--delta 0.25 \
{groundtruth_file:} \
{result_file:} {write_to:}'.format(
        groundtruth_file=groundtruth_file,
        result_file=result_file,
        write_to='>> {}'.format(benchmark_file) if not args.stdout else '')
    print('*** COMMAND TO BE EXECUTED ***')
    if args.plot:
        cmd += " --plot {}_rpe_plot.png".format(result_file)
    print(cmd)
    os.system(cmd)





