import argparse
import array
import json
import os, glob
import re

import sys
# The EKF state size is a compile-time constant, so comparing state sizes means
# comparing two builds. XIVO_LIB points at the matching output directory (see
# XIVO_OUTPUT_SUFFIX in CMakeLists.txt); it must be read before `import pyxivo`,
# which is why it is an environment variable rather than an argparse flag.
sys.path.insert(0, os.environ.get('XIVO_LIB', 'lib'))
import pyxivo
# `savers` is imported lazily, inside the mode branches that use one. It pulls in
# numpy and transforms3d, which together cost ~25 MB of RSS -- 20% of the peak of
# a whole run -- and `-mode runOnly`, which is what a throughput measurement and a
# deployment both use, never constructs a saver at all. Nothing else in this file
# touches numpy. See notes-speed/m4-memory.md.


def is_stereo_cfg(cfg_path):
    """True if the estimator config asks for stereo.

    The config is the single source of truth for stereo-vs-mono; there is no
    command-line flag, so a config and a run mode cannot disagree. XIVO's configs
    permit // comments, which json.loads does not, so strip them first.
    """
    with open(cfg_path) as f:
        text = re.sub(r'(?m)//.*$', '', f.read())
    return bool(json.loads(text).get('stereo', False))



parser = argparse.ArgumentParser()
parser.add_argument("-root", default="/media/data1/Data/tumvi/exported/euroc/512_16",
    help="location of VIO dataset")
parser.add_argument("-dump", default=".",
    help="location of xivo's output data from a dataset")
parser.add_argument("-dataset", default="tumvi",
    help="name of a (supported) VIO dataset [tumvi|cosyvio|alphred|xivo|void]")
parser.add_argument("-seq", default="room1",
    help="short tag for sequence name")
parser.add_argument("-cam_id", default=0, type=int,
    help="camera from stereo camera pair (only used for tumvi dataset)")
parser.add_argument('-cfg', default='cfg/tumvi_cam0.json',
    help='path to the estimator configuration')
parser.add_argument('-use_viewer', default=False, action='store_true',
    help='visualize trajectory and feature tracks if set')
parser.add_argument('-mode', default='eval',
    help='[eval|dump|dumpCov|runOnly] mode to handle the state estimates. eval: save states for evaluation; dump: save to json file for further processing')
parser.add_argument(
    '-save_full_cov', default=False, action='store_true',
    help='save the entire covariance matrix, not just that of the motion state, if set')


def main(args):
    if not os.path.exists(args.dump):
        os.makedirs(args.dump)

    ########################################
    # CHOOSE SAVERS
    ########################################
    if args.mode != 'runOnly':
        import savers

    if args.mode == 'eval':
        if args.dataset == 'tumvi':
            saver = savers.TUMVIEvalModeSaver(args)
        elif args.dataset == 'cosyvio':
            saver = savers.COSYVIOEvalModeSaver(args)
        elif args.dataset == "xivo":
            saver = savers.XIVOEvalModeSaver(args)
        elif args.dataset == "void":
            saver = savers.VOIDEvalModeSaver(args)
        elif args.dataset == 'carla':
            saver = savers.CarlaEvalModeSaver(args)
    elif args.mode == 'dump':
        if args.dataset == 'tumvi':
            saver = savers.TUMVIDumpModeSaver(args)
        elif args.dataset == 'cosyvio':
            saver = savers.COSYVIODumpModeSaver(args)
        elif args.dataset == "xivo":
            saver = savers.XIVODumpModeSaver(args)
        elif args.dataset == "void":
            saver = savers.VOIDDumpModeSaver(args)
        elif args.dataset == 'carla':
            saver = savers.CarlaDumpModeSaver(args)
    elif args.mode == 'dumpCov':
        if args.dataset == 'tumvi':
            saver = savers.TUMVICovDumpModeSaver(args)
        elif args.dataset == 'cosyvio':
            saver = savers.COSYVIOCovDumpModeSaver(args)
        elif args.dataset == "xivo":
            saver = savers.XIVOCovDumpModeSaver(args)
        elif args.dataset == "void":
            saver = savers.VOIDCovDumpModeSaver(args)
        elif args.dataset == 'carla':
            saver = savers.CarlaCovDumpModeSaver(args)
    elif args.mode == 'runOnly':
        pass
    else:
        raise ValueError('mode=[eval|dump|dumpCov|runOnly]')

    ########################################
    # LOAD DATA
    ########################################
    stereo = is_stereo_cfg(args.cfg)
    img_dir_r = None
    if args.dataset == 'tumvi':
        img_dir = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                               'mav0', 'cam{}'.format(args.cam_id), 'data')

        imu_path = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                                'mav0', 'imu0', 'data.csv')
        if stereo:
            img_dir_r = os.path.join(
                args.root, 'dataset-{}_512_16'.format(args.seq), 'mav0',
                'cam{}'.format(1 - args.cam_id), 'data')
    elif args.dataset == 'cosyvio':
        img_dir = os.path.join(args.root, 'data', args.sen, args.seq, 'frames')
        imu_path = os.path.join(args.root, 'data', args.sen, args.seq, 'data.csv')
    elif args.dataset in ['xivo', 'void']:
        img_dir = os.path.join(args.root, args.seq, 'cam0', 'data')
        imu_path = os.path.join(args.root, args.seq, 'imu0', 'data.csv')
    elif args.dataset == 'carla':
        img_dir = os.path.join(args.root, args.seq, 'rgb')
        imu_path = os.path.join(args.root, args.seq, 'imu', 'data.csv')
    else:
        raise ValueError('unknown dataset argument; choose from tumvi, xivo, cosyvio, carla')

    if stereo and img_dir_r is None:
        raise ValueError(
            'config {} requests stereo, but dataset {} has no stereo pair '
            'configured'.format(args.cfg, args.dataset))

    # Images and IMU samples are kept in two separate containers and merged in
    # the run loop, rather than concatenated into one list of Python objects.
    # The reason is memory: a TUM-VI room sequence has ~2.8k images but ~28k IMU
    # samples at 200 Hz, and one boxed sample -- an outer tuple, an inner tuple,
    # two 3-element lists and six float objects -- costs ~460 B, so the combined
    # list was ~13 MB resident (measured: +17.6 MB across the whole load step),
    # about 14% of the peak RSS of an entire run and larger than every one of
    # XIVO's own buffers. The flat `array.array` form below is 56 B per sample.
    #
    # This does not change what the estimator sees. `float()` produces the same
    # IEEE double as before and `array('d')` stores it without conversion, and
    # the merge in the run loop reproduces the old visit order exactly -- see the
    # comment there. Verified bit-identical on room1+room3, mono and stereo.
    frames = []
    imu_ts = array.array('q')
    imu_v = array.array('d')

    # Stereo: map each left timestamp to its right-image path. TUM-VI names both
    # files after the (shared, hardware-triggered) timestamp, so pairing is exact
    # -- but the pairing is still built by *checking the right file exists*, not
    # by assuming it does, so a partial download shows up as dropped frames
    # rather than as a crash mid-run.
    right_of = {}
    if stereo:
        right_ts = {int(os.path.basename(p)[:-4]): p
                    for p in glob.glob(os.path.join(img_dir_r, '*.png'))}

    if args.dataset in ['tumvi', 'xivo', 'carla', 'void']:
        for p in glob.glob(os.path.join(img_dir, '*.png')):
            ts = int(os.path.basename(p)[:-4])
            if stereo:
                if ts not in right_ts:
                    continue
                right_of[ts] = right_ts[ts]
            frames.append((ts, p))
    elif args.dataset == 'cosyvio':
        img_filelist = os.path.join(img_dir, 'data.csv')
        with open(img_filelist, 'r') as fid:
            for l in fid:
                if l[0].isdigit():
                    larr = l.strip().split(',')
                    ts = int(larr[0])
                    png_file = os.path.join(img_dir, larr[1])
                    frames.append((ts,png_file))

    # Iterated rather than readlines()'d: the file is ~28k lines and the list of
    # strings was a 3 MB transient for no benefit.
    with open(imu_path, 'r') as fid:
        for l in fid:
            if l[0].isdigit():
                v = l.strip().split(',')
                imu_ts.append(int(v[0]))
                imu_v.append(float(v[1]))
                imu_v.append(float(v[2]))
                imu_v.append(float(v[3]))
                imu_v.append(float(v[4]))
                imu_v.append(float(v[5]))
                imu_v.append(float(v[6]))

    frames.sort(key=lambda tup: tup[0])

    # The IMU side used to be ordered by the same stable sort, which for equal
    # timestamps means file order. It already is in file order, so normally there
    # is nothing to do; the permutation below is the fallback for a data file that
    # is not monotonic, and it is applied to the arrays so the run loop stays
    # branch-free.
    if any(imu_ts[k] < imu_ts[k - 1] for k in range(1, len(imu_ts))):
        order = sorted(range(len(imu_ts)), key=imu_ts.__getitem__)
        imu_ts = array.array('q', (imu_ts[k] for k in order))
        permuted = array.array('d')
        for k in order:
            permuted.extend(imu_v[6 * k:6 * k + 6])
        imu_v = permuted
        del order, permuted

    if stereo:
        n_left = len(glob.glob(os.path.join(img_dir, '*.png')))
        print('stereo: {} pairs from {} left / {} right frames'.format(
            len(right_of), n_left, len(right_ts)))
        if len(right_of) < n_left:
            print('stereo: WARNING dropped {} left frames with no right '
                  'partner'.format(n_left - len(right_of)))
        if not right_of:
            raise ValueError(
                'stereo requested but no timestamp matched between {} and '
                '{}'.format(img_dir, img_dir_r))

    ########################################
    # INITIALIZE ESTIMATOR
    ########################################
    viewer_cfg = ''
    if args.use_viewer:
        if args.dataset == 'tumvi':
            viewer_cfg = os.path.join('cfg', 'viewer.json')
        elif args.dataset == "xivo":
            viewer_cfg = os.path.join('cfg', 'phab_viewer.json')
        elif args.dataset == 'void':
            viewer_cfg = os.path.join('cfg', 'void_viewer.json')
        elif args.dataset == 'cosyvio':
            if args.sen == 'tango_top':
                viewer_cfg = os.path.join('cfg', 'phab_viewer.json')
            elif args.sen == 'tango_bottom':
                viewer_cfg = os.path.join('cfg', 'cosyvio_tango_bottom_viewer.json')
        elif args.dataset == 'carla':
            viewer_cfg = os.path.join('cfg', 'carla_viewer.json')

    #########################################
    # RUN ESTIMATOR AND SAVE DATA
    #########################################
    # this is wrapped in a try/finally block so that data will save even when
    # we hit an exception (namely, KeyboardInterrupt)
    # Bound before the try so a constructor failure surfaces as itself rather
    # than as a NameError from the finally block.
    estimator = None
    try:
        estimator = pyxivo.Estimator(args.cfg, viewer_cfg, args.seq, False)
        # Merge of the two timestamp-ordered streams. This is exactly the order a
        # single stable sort of [images..., imu...] produced before: `<=` on the
        # image side visits the image first when a frame and a sample share a
        # timestamp, which is what the old code did because every image was
        # appended to the list before any IMU sample.
        n_frames = len(frames)
        n_imu = len(imu_ts)
        n_total = n_frames + n_imu
        i = 0
        i_frame = 0
        i_imu = 0
        while i_frame < n_frames or i_imu < n_imu:
            if i > 0 and i % 1000 == 0:
                print('{:6}/{:6}'.format(i, n_total))
            i += 1
            if i_imu >= n_imu or (i_frame < n_frames
                                  and frames[i_frame][0] <= imu_ts[i_imu]):
                ts, content = frames[i_frame]
                i_frame += 1
                if stereo:
                    estimator.VisualMeasStereo(ts, content, right_of[ts])
                else:
                    estimator.VisualMeas(ts, content)
                if estimator.UsingLoopClosure():
                    estimator.CloseLoop()
                estimator.Visualize()
                if (args.mode != 'runOnly') and (estimator.VisionInitialized()):
                    saver.onVisionUpdate(estimator, datum=(ts, content))
            else:
                b = 6 * i_imu
                estimator.InertialMeas(imu_ts[i_imu], imu_v[b], imu_v[b + 1],
                                       imu_v[b + 2], imu_v[b + 3],
                                       imu_v[b + 4], imu_v[b + 5])
                i_imu += 1

    finally:
        if stereo and estimator is not None:
            print_stereo_stats(estimator)
        if args.mode != 'runOnly':
            saver.onResultsReady()


def print_stereo_stats(estimator):
    """Summarize left->right matching and stereo depth seeding.

    Printed at the end of every stereo run so the numbers land in the eval logs
    alongside the ATE, rather than needing a separate harness to recover them.
    """
    frames = estimator.num_stereo_frames()
    attempted = estimator.num_stereo_attempted()
    matched = estimator.num_stereo_matched()
    print('stereo: {} frames, {} match attempts, {} matched ({:.1f}%)'.format(
        frames, attempted, matched,
        100.0 * matched / attempted if attempted else 0.0))
    print('stereo: rejected klt={} epipolar={} circular={} disparity={}'.format(
        estimator.num_stereo_rejected_klt(),
        estimator.num_stereo_rejected_epipolar(),
        estimator.num_stereo_rejected_circular(),
        estimator.num_stereo_rejected_disparity()))
    ok = estimator.num_stereo_init_ok()
    no_match = estimator.num_stereo_init_no_match()
    rejected = estimator.num_stereo_init_rejected()
    total = ok + no_match + rejected
    print('stereo_init: {} seeded, {} no-match, {} rejected '
          '({:.1f}% of {} new features seeded)'.format(
              ok, no_match, rejected,
              100.0 * ok / total if total else 0.0, total))
    print('stereo_init: rejected degenerate={} gap={} range={} std={}'.format(
        estimator.num_stereo_init_rej_degenerate(),
        estimator.num_stereo_init_rej_gap(),
        estimator.num_stereo_init_rej_range(),
        estimator.num_stereo_init_rej_std()))
    used = estimator.num_stereo_upd_used()
    rej_geom = estimator.num_stereo_upd_rej_geom()
    rej_mh = estimator.num_stereo_upd_rej_mh()
    offered = used + rej_geom + rej_mh
    print('stereo_update: {} right measurements used, rejected geom={} mh={} '
          '({:.1f}% of {} offered)'.format(
              used, rej_geom, rej_mh,
              100.0 * used / offered if offered else 0.0, offered))


if __name__ == '__main__':
    main(args=parser.parse_args())
