import argparse
import os, glob

import numpy as np
import glog
from transforms3d.quaternions import mat2quat

import sys
sys.path.insert(0, 'lib')
import pyxivo
import savers

TP_ROOT = '/home/feixh/Data/tumvi/exported/euroc/512_16'
KIF_ROOT = '/local2/Data/tumvi/exported/euroc/512_16'

parser = argparse.ArgumentParser()
parser.add_argument(
    '-root', default=KIF_ROOT, help='root directory of the tumvi dataset')
parser.add_argument(
    '-cfg', default='cfg/phab.json', help='path to the estimator configuration')
parser.add_argument(
    '-seq', default='room6', help='short tag for the seuqence name')
parser.add_argument(
    '-cam_id', default=0, type=int, help='specify which camera to use (for TUM VI dataset)')
parser.add_argument(
    '-sen', default="tango_top", type=str, help="specify which camera to use (COSYVIO dataset)"
)
parser.add_argument(
    '-out_dir', default='.', help='output directory to save results')
parser.add_argument(
        '-dataset', default='tumvi', help='dataset type')
parser.add_argument(
    '-use_viewer', default=False, action='store_true',
    help='visualize trajectory and feature tracks if set')
parser.add_argument(
     '-mode', default='eval', help='[eval|dump] mode to handle the state estimates. eval: save states for evaluation; dump: save to json file for further processing')


def main(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    ########################################
    # CHOOSE SAVERS
    ########################################
    if args.mode == 'eval':
        if args.dataset == 'tumvi':
            saver = savers.TUMVIEvalModeSaver(args)
        elif args.dataset == 'cosyvio':
            saver = savers.COSYVIOEvalModeSaver(args)
        elif args.dataset == 'xivo':
            saver = savers.XIVOEvalModeSaver(args)
    elif args.mode == 'dump':
        if args.dataset == 'tumvi':
            saver = savers.TUMVIDumpModeSaver(args)
        elif args.dataset == 'cosyvio':
            saver = savers.COSYVIODumpModeSaver(args)
        elif args.dataset == 'xivo':
            saver = savers.XIVODumpModeSaver(args)
    elif args.mode == 'runOnly':
        pass
    else:
        raise ValueError('mode=[eval|dump|dumpCov|runOnly]')

    ########################################
    # LOAD DATA
    ########################################
    if args.dataset == 'tumvi':
        img_dir = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                               'mav0', 'cam{}'.format(args.cam_id), 'data')

        imu_path = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                                'mav0', 'imu0', 'data.csv')
    elif args.dataset == 'xivo':
        img_dir = os.path.join(args.root, args.seq, 'cam0', 'data')

        imu_path = os.path.join(args.root, args.seq, 'imu0', 'data.csv')
    elif args.dataset == "cosyvio":
        img_dir = os.path.join(args.root, 'data', args.sen, args.seq, 'frames')
        imu_path = os.path.join(args.root, 'data', args.sen, args.seq, 'data.csv')
    else:
        raise ValueError('unknown dataset argument; choose from tumvi or xivo')

    data = []

    if args.dataset in ['tumvi', 'xivo', 'carla', 'alphred', 'sabr']:
        for p in glob.glob(os.path.join(img_dir, '*.png')):
            ts = int(os.path.basename(p)[:-4])
            data.append((ts, p))
    elif args.dataset == 'cosyvio':
        img_filelist = os.path.join(img_dir, 'data.csv')
        with open(img_filelist, 'r') as fid:
            for l in fid.readlines():
                if l[0].isdigit():
                    larr = l.strip().split(',')
                    ts = int(larr[0])
                    png_file = os.path.join(img_dir, larr[1])
                    data.append((ts,png_file))

    with open(imu_path, 'r') as fid:
        for l in fid.readlines():
            if l[0].isdigit():
                v = l.strip().split(',')
                ts = int(v[0])
                w = [float(x) for x in v[1:4]]
                t = [float(x) for x in v[4:]]
                data.append((ts, (w, t)))

    data.sort(key=lambda tup: tup[0])


    ########################################
    # INITIALIZE ESTIMATOR
    ########################################
    viewer_cfg = ''
    if args.use_viewer:
        viewer_cfg = os.path.join('cfg', 'viewer.json' if args.dataset == 'tumvi' else 'phab_viewer.json')

    estimator = pyxivo.Estimator(args.cfg, viewer_cfg, args.seq)
    for i, (ts, content) in enumerate(data):
        # if i > 0 and i % 500 == 0:
        #     print('{:6}/{:6}'.format(i, len(data)))
        if isinstance(content, tuple):
            gyro, accel = content
            estimator.InertialMeas(ts, gyro[0], gyro[1], gyro[2], accel[0],
                                   accel[1], accel[2])
        else:
            estimator.VisualMeas(ts, content)
            estimator.Visualize()
            saver.onVisionUpdate(estimator, datum=(ts, content))

    saver.onResultsReady()

if __name__ == '__main__':
    main(args=parser.parse_args())
