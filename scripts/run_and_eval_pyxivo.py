import numpy as np
import argparse
import os, sys
from shutil import copyfile
from utils import get_xivo_gt_filename, get_xivo_output_filename


TP_ROOT = '/home/feixh/Data/tumvi/exported/euroc/512_16'
KIF_ROOT = '/local2/Data/tumvi/exported/euroc/512_16'
double_fusion = False

parser = argparse.ArgumentParser()
parser.add_argument(
    '-root', default=KIF_ROOT, help='root directory of the dataset')
parser.add_argument('-dataset', default='tumvi')
parser.add_argument(
    '-cam_id', default=0, type=int,
    help='camera from stereo camera pair, if dataset is stereo')
parser.add_argument('-cfg', default='cfg/tumvi_cam0.json',
    help='path to the estimator configuration')
parser.add_argument(
    '-seq', default='room6', help='short tag for the seuqence name')
parser.add_argument(
    '-out_dir', default='.', help='output directory to save results')
parser.add_argument(
    '-use_viewer', default=False, action='store_true', help='visualize if set')
parser.add_argument(
    '-plot', default=False, action="store_true", help="plot if set")
parser.add_argument(
    '-ate_max_difference', default=0.001, type=float,
    help="maximum timestamp difference to use when aligning ground-truth and output points in calculating ATE"
)

# parser.add_argument(
#     '-double-fusion', default=False, action='store_true', help='if ture, take average of two trajectories in the same coordinate system')
#
parser.add_argument(
    '-stdout', default=False, action='store_true', help='write to stdout instead of a benchmark file if set')
args = parser.parse_args()

if __name__ == '__main__':

    ########################################
    # BACKUP CONFIGURATION
    ########################################
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)


    cam_ids = [0, 1] if double_fusion else [0]

    for cam_id in cam_ids:
        cfg = args.cfg

        copyfile(cfg, os.path.join(args.out_dir, os.path.basename(cfg)))

        ########################################
        # RUN THE FILTER
        ########################################
        cmd = 'python3 scripts/pyxivo.py \
-root {root:} \
-dataset {dataset:} \
-cfg {cfg:} \
-seq {seq:} \
-cam_id {cam_id:} \
-dump {out_dir:} \
-mode eval \
{use_viewer:}'.format(
            root=args.root, cfg=cfg, seq=args.seq, cam_id=cam_id,
            out_dir=args.out_dir, dataset=args.dataset,
            use_viewer='-use_viewer' if args.use_viewer else '')
        print('*** COMMAND TO BE EXECUTED ***')
        print(cmd)
        os.system(cmd)

        result_file = get_xivo_output_filename(args.out_dir, args.dataset, args.seq, cam_id=cam_id)
        #result_file = os.path.join(args.out_dir, '{}_{}_cam{}'.format(args.dataset, args.seq, cam_id))
        #groundtruth_file = os.path.join(args.out_dir, '{}_{}_gt'.format(args.dataset, args.seq))
        groundtruth_file = get_xivo_gt_filename(args.out_dir, args.dataset, args.seq)
        benchmark_file = os.path.join(args.out_dir, '{}_{}_bench'.format(args.dataset, args.seq))

        if cam_id == 0:
            os.system('echo {} sequence {} >> {}'.format(args.dataset, args.seq, benchmark_file))

        os.system('echo camera {} >> {}'.format(cam_id, benchmark_file))

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
--delta 1 \
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

    if double_fusion:
        # fuse trajectories
        cmd = 'python3 scripts/double_fusion.py \
-root {root:} \
-working-dir {workdir:}  \
-seq {seq:}'.format( root=args.root, workdir=args.out_dir, seq=args.seq)
        print('*** COMMAND TO BE EXECUTED ***')
        print(cmd)
        os.system(cmd)

        os.system('echo double-fusion >> {}'.format(benchmark_file))

        # evaluate the fused trajectory
        result_file = os.path.join(args.out_dir, '{}_{}_fused'.format(args.dataset, args.seq))
        cmd = 'python3 scripts/tum_rgbd_benchmark_tools/evaluate_ate.py \
--max_difference {ate_max_difference:} \
{groundtruth_file:} \
{result_file:} {write_to:}'.format(
            groundtruth_file=groundtruth_file,
            result_file=result_file,
            ate_max_difference=args.ate_max_difference,
            write_to='>> {}'.format(benchmark_file) if not args.stdout else '')
        print('*** COMMAND TO BE EXECUTED ***')
        print(cmd)
        os.system(cmd)






