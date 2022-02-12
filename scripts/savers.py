import numpy as np
import os, sys
import json
from transforms3d.quaternions import mat2quat

from cosyvio_prep import cosyvio_extrinsics_data as cd



def get_xivo_output_filename(dumpdir, dataset, seq, cam_id=0, sen='tango_top'):
  if dataset=="tumvi":
    estimator_datafile = os.path.join(dumpdir,
      "tumvi_{}_cam{}".format(seq, cam_id))
  elif dataset=="cosyvio":
    estimator_datafile = os.path.join(dumpdir,
      "cosyvio_{}_{}".format(sen, seq))
  else:
    estimator_datafile = os.path.join(dumpdir, "{}_{}".format(dataset, seq))
  return estimator_datafile


def get_xivo_gt_filename(dumpdir, dataset, seq, sen='tango_top'):
  if dataset=="cosyvio":
    gt_data = os.path.join(dumpdir, "cosyvio_{}_{}_gt".format(sen, seq))
  else:
    gt_data = os.path.join(dumpdir, "{}_{}_gt".format(dataset, seq))
  return gt_data



class BaseSaver:
    """Abstract class that outlines the functions that the other savers need
    to have."""
    def __init__(self, args):
        self.results = []
        self.resultsPath = get_xivo_output_filename(args.out_dir, args.dataset,
            args.seq, cam_id=args.cam_id, sen=args.sen)
    def onVisionUpdate(self, estimator, datum):
        pass
    def onResultsReady(self):
        pass


class TUMVISaver:

    def __init__(self, args):
        # parse mocap and save gt in desired format
        mocapPath = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                                 'mav0', 'mocap0', 'data.csv')
        groundtruthPath = get_xivo_gt_filename(args.out_dir, "tumvi", args.seq)
        self.saveMocapAs(mocapPath, groundtruthPath)

    def saveMocapAs(self, mocapPath, groundtruthPath):
        gt = []
        with open(mocapPath, 'r') as fid:
            for l in fid.readlines():
                if l[0] != '#':
                    v = l.strip().split(',')
                    if (len(v) >= 8):
                        ts = int(v[0])
                        t = [float(x) for x in v[1:4]]
                        q = [float(x) for x in v[4:]]  # [w, x, y, z]
                        gt.append(
                            [ts * 1e-9, t[0], t[1], t[2], q[1], q[2], q[3], q[0]])

        np.savetxt(
            groundtruthPath,
            gt,
            fmt='%f %f %f %f %f %f %f %f')


class COSYVIOSaver:
    def __init__(self, args):
        # parse mocap and save gt in desired format
        self.sensor = args.sen
        raw_gt_path = os.path.join(args.root, 'data', 'ground_truth',
            args.seq, 'pose.txt')
        groundtruthPath = get_xivo_gt_filename(args.out_dir, "cosyvio", args.seq,
            sen=args.sen)
        self.save_cosyvio_gt(raw_gt_path, groundtruthPath, int(1427875200 * 1e9))

    def save_cosyvio_gt(self, raw_gt_path, groundtruthPath, collection_time):
        gt = []
        with open(raw_gt_path, 'r') as fid:
            for l in fid.readlines():
                larr = l.split()
                ts = float(int(float(larr[0])*1e9) + collection_time) * 1e-9
                r00 = float(larr[1])
                r01 = float(larr[2])
                r02 = float(larr[3])
                t0 = float(larr[4])
                r10 = float(larr[5])
                r11 = float(larr[6])
                r12 = float(larr[7])
                t1 = float(larr[8])
                r20 = float(larr[9])
                r21 = float(larr[10])
                r22 = float(larr[11])
                t2 = float(larr[12])

                Rsb0 = np.array([[r00, r01, r02],
                                 [r10, r11, r12],
                                 [r20, r21, r22]])
                Tsb0 = np.array([t0, t1, t2])

                if self.sensor == 'tango_bottom':
                    Rsb = Rsb0.dot(cd.R_b0_tbi)
                    Tsb = Rsb0.dot(cd.T_b0_tbi) + Tsb0
                elif self.sensor == 'tango_top':
                    Rsb = Rsb0.dot(cd.R_b0_tti)
                    Tsb = Rsb0.dot(cd.T_b0_tti) + Tsb0

                # Convert Rsb to a quaternion
                qsb = mat2quat(Rsb) # [w, x, y, z]

                gt.append([
                    ts, Tsb[0], Tsb[1], Tsb[2], qsb[1], qsb[2], qsb[3], qsb[0]
                ])

        np.savetxt(
            groundtruthPath,
            gt,
            fmt='%f %f %f %f %f %f %f %f')


class EvalModeSaver(BaseSaver):
    """ Callback functions used in eval mode of pyxivo.
    """
    def __init__(self, args):
        BaseSaver.__init__(self, args)

    def onVisionUpdate(self, estimator, datum):
        now = estimator.now()
        gsb = np.array(estimator.gsb())
        Tsb = gsb[:, 3]

        # print gsb[:3, :3]
        try:
            q = mat2quat(gsb[:3, :3])  # [w, x, y, z]
            # format compatible with tumvi rgbd benchmark scripts
            self.results.append(
                [now * 1e-9, Tsb[0], Tsb[1], Tsb[2], q[1], q[2], q[3], q[0]])
        except np.linalg.linalg.LinAlgError:
            pass

    def onResultsReady(self):
        np.savetxt(
            self.resultsPath,
            self.results,
            fmt='%f %f %f %f %f %f %f %f')



class DumpModeSaver(BaseSaver):
    """ Callback functions used by dump mode of pyxivo.
    """
    def __init__(self, args):
        BaseSaver.__init__(self, args)

    def onVisionUpdate(self, estimator, datum):
        ts, content = datum
        #now = estimator.now()
        g = np.array(estimator.gsc())
        T = g[:, 3]

        if np.linalg.norm(T) > 0:
            try:
                q = mat2quat(g[:3, :3])  # [w, x, y, z]
                # format compatible with tumvi rgbd benchmark scripts
                entry = dict()
                entry['ImagePath'] = str(content)
                entry['Timestamp'] = ts
                entry['TranslationXYZ'] = [T[0], T[1], T[2]]
                entry['QuaternionWXYZ'] = [q[0], q[1], q[2], q[3]]
                self.results.append(entry)

                with open(self.resultsPath, 'w') as fid:
                    json.dump(self.results, fid, indent=2)
            except np.linalg.linalg.LinAlgError:
                pass

    def onResultsReady(self):
        with open(self.resultsPath, 'w') as fid:
            json.dump(self.results, fid, indent=2)


class TUMVIEvalModeSaver(EvalModeSaver, TUMVISaver):
    def __init__(self, args):
        EvalModeSaver.__init__(self, args)
        TUMVISaver.__init__(self, args)


class TUMVIDumpModeSaver(DumpModeSaver, TUMVISaver):
    def __init__(self, args):
        DumpModeSaver.__init__(self, args)
        TUMVISaver.__init__(self, args)


class COSYVIOEvalModeSaver(EvalModeSaver, COSYVIOSaver):
    def __init__(self, args):
        EvalModeSaver.__init__(self, args)
        COSYVIOSaver.__init__(self, args)


class COSYVIODumpModeSaver(DumpModeSaver, COSYVIOSaver):
    def __init__(self, args):
        DumpModeSaver.__init__(self, args)
        COSYVIOSaver.__init__(self, args)


class XIVOEvalModeSaver(EvalModeSaver, BaseSaver):
    def __init__(self, args):
        EvalModeSaver.__init__(self, args)
        BaseSaver.__init__(self, args)


class XIVODumpModeSaver(DumpModeSaver, BaseSaver):
    def __init__(self, args):
        DumpModeSaver.__init__(self, args)
        BaseSaver.__init__(self, args)
