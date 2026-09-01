import numpy as np
import os, sys
import json
from transforms3d.quaternions import mat2quat


sys.path.append(os.path.join(os.getcwd(), "scripts"))
from to_json import to_json
from utils import upper_triangular_list, get_xivo_gt_filename, \
    get_xivo_output_filename


class BaseSaver:
    """Abstract class that outlines the functions that the other savers need
    to have."""
    def __init__(self, args):
        self.results = []
        self.resultsPath = get_xivo_output_filename(args.dump, args.dataset,
            args.seq, cam_id=args.cam_id)
    def onVisionUpdate(self, estimator, datum):
        pass
    def onResultsReady(self):
        pass


class TUMVISaver:

    def __init__(self, args):
        # parse mocap and save gt in desired format
        mocapPath = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                                 'mav0', 'mocap0', 'data.csv')
        groundtruthPath = get_xivo_gt_filename(args.dump, "tumvi", args.seq)
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


class CarlaSaver(TUMVISaver):
    def __init__(self, args):
        mocapPath = os.path.join(args.root, args.seq, 'ground_truth',
            'data.csv')
        groundtruthPath = get_xivo_gt_filename(args.dump, args.dataset, args.seq)
        self.saveMocapAs(mocapPath, groundtruthPath)


class VOIDSaver:
    def __init__(self, args):
        # Move posenet file to current directory
        posenetPath = os.path.join(args.root, args.seq, "traj_posenet")
        output_path = get_xivo_gt_filename(args.dump, args.dataset, args.seq)
        os.system("cp {} {}".format(posenetPath, output_path))


class PCWSaver:
    def __init__(self, imu_traj, args):
        gt_path = os.path.join(args.dump, "pcw_{}_gt".format(args.motion_type))
        self.save_gt(imu_traj, gt_path)

    def save_gt(self, imu_traj, gt_path):
        gt = []
        nt = len(imu_traj.t)
        for i in range(nt):
            ts = imu_traj.t[i]
            qsb_xyzw = imu_traj.qsb[:,i]
            Tsb = imu_traj.Tsb[:,i]
            Vsb = imu_traj.Vsb[:,i]

            # quaternions are wxyz here
            gt.append([
                ts, Tsb[0], Tsb[1], Tsb[2],
                qsb_xyzw[0], qsb_xyzw[1], qsb_xyzw[2], qsb_xyzw[3],
                Vsb[0], Vsb[1], Vsb[2]
            ])

        np.savetxt(gt_path, gt,
                   fmt='%f %f %f %f %f %f %f %f %f %f %f')



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
        except np.linalg.LinAlgError:
            pass

    def onResultsReady(self):
        # `%f` is six decimals, i.e. 1 um on a trajectory of order 1 m -- fine for
        # the TUM benchmark scripts, but it hides differences of the size that a
        # change of arithmetic makes, so an md5 match on a `%f` dump proves
        # agreement only to 1e-6 m and is NOT a bit-identity check.
        # XIVO_DUMP_PRECISE=1 writes round-trippable values instead, so two builds
        # can be compared exactly.
        fmt = '%.17g' if os.environ.get('XIVO_DUMP_PRECISE') else '%f'
        np.savetxt(
            self.resultsPath,
            self.results,
            fmt=' '.join([fmt] * 8))



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
            except np.linalg.LinAlgError:
                pass

    def onResultsReady(self):
        with open(self.resultsPath, 'w') as fid:
            json.dump(self.results, fid, indent=2)


class CovDumpModeSaver(BaseSaver):
    def __init__(self, args):
        BaseSaver.__init__(self, args)
        self.save_full_cov = args.save_full_cov

    def onVisionUpdate(self, estimator, datum):
        ts, content = datum

        # Get camera pose
        gsb = np.array(estimator.gsb())
        gbc = np.array(estimator.gbc())
        gsc = np.array(estimator.gsc())
        qsb = mat2quat(gsb[:3, :3])
        Tsb = gsb[:, 3]
        qbc = mat2quat(gbc[:3, :3])
        Tbc = gbc[:, 3]
        qsc = mat2quat(gsc[:3, :3])
        Tsc = gsc[:, 3]
        Vsb = np.array(estimator.Vsb())

        # Get calibration states
        bg = estimator.bg()
        ba = estimator.ba()
        qg = mat2quat(estimator.Rg())
        td = estimator.td()
        Ca = estimator.Ca()
        Cg = estimator.Cg()
        Cam = estimator.CameraIntrinsics()
        CamType = estimator.CameraDistortionType()

        # For some reason, this interface is broken, and I can't figure out
        # why (ST, May 2022)
        # Get filter covariance
        #if self.save_full_cov:
        #    Pstate = np.array(estimator.P())
        #else:
        #    Pstate = np.array(estimator.Pstate())

        # Group ID
        GaugeGroup = estimator.gauge_group()

        # Get features
        num_instate_features = estimator.num_instate_features()
        if num_instate_features > 0:
            feature_positions = estimator.InstateFeaturePositions()
            feature_covs = estimator.InstateFeatureCovs()
            feature_ids = estimator.InstateFeatureIDs()
            feature_sinds = estimator.InstateFeatureSinds()
        else:
            feature_positions = []
            feature_covs = []
            feature_ids = []
            feature_sinds = []

        # Get groups
        num_instate_groups = estimator.num_instate_groups()
        if num_instate_groups > 0:
            group_covs = estimator.InstateGroupCovs()
            group_sinds = estimator.InstateGroupSinds()
            group_ids = estimator.InstateGroupIDs()
            group_poses = estimator.InstateGroupPoses()
        else:
            group_poses = []
            group_covs = []
            group_ids = []
            group_sinds = []

        # Outlier rejection stats
        num_mh_rejected = estimator.num_mh_rejected()
        num_oneptransac_rejected = estimator.num_oneptransac_rejected()
        num_tracker_outlier_rejected = estimator.num_tracker_outlier_rejected()

        # Save results
        entry = dict()

        entry['group'] = GaugeGroup

        entry['ImagePath'] = str(content)
        entry['Timestamp'] = ts

        entry['Tsb_XYZ'] = Tsb.tolist()
        entry['qsb_WXYZ'] = qsb.tolist()
        entry['Tbc_XYZ'] = Tbc.tolist()
        entry['qbc_WXYZ'] = qbc.tolist()
        entry['Tsc_XYZ'] = Tsc.tolist()
        entry['qsc_WXYZ'] = qsc.tolist()
        entry['Vsb_XYZ'] = Vsb.tolist()

        # Related to broken interface above
        #entry['Pstate'] = upper_triangular_list(Pstate)

        entry['bg'] = bg.tolist()
        entry['ba'] = ba.tolist()
        entry['qg_WXYZ'] = qg.tolist()
        entry['td'] = td
        entry['Ca'] = Ca
        entry['Cg'] = Cg
        entry['camera_intrinsics'] = Cam
        entry['camera_type'] = CamType

        entry['num_instate_features'] = num_instate_features
        entry['feature_positions'] = feature_positions
        entry['feature_covs'] = feature_covs
        entry['feature_ids'] = feature_ids
        entry['feature_sinds'] = feature_sinds

        entry['num_instate_groups'] = num_instate_groups
        entry['group_poses'] = group_poses
        entry['group_covs'] = group_covs
        entry['group_ids'] = group_ids
        entry['group_sinds'] = group_sinds

        entry['num_mh_rejected'] = num_mh_rejected
        entry['num_oneptransac_rejected'] = num_oneptransac_rejected
        entry['num_tracker_outlier_rejected'] = num_tracker_outlier_rejected

        self.results.append(entry)


    def onResultsReady(self):
        with open(self.resultsPath, 'w') as fid:
            output = { "data": self.results }
            fid.write(to_json(output))


class TrackerDumpModeSaver(BaseSaver):
    """ Callback functions used by tracker dump mode of pyxivo.
    """
    def __init__(self, args):
        BaseSaver.__init__(self, args)

    def onVisionUpdate(self, estimator, datum, descriptor_format):
        ts, content = datum

        tracked_features = estimator.tracked_features()

        for f in tracked_features:
            id, kp, des = f

            # Convert timestamp to seconds from nanoseconds
            ts_and_id = np.array([ts*1e-9, id]).reshape(1,2)
            kp = np.reshape(kp, (1, 2))

            # Number of dimensions of the descriptor
            N = des.shape[1]

            feature_info = np.concatenate((ts_and_id, kp, des), axis=1)

            with open(self.resultsPath, "a") as f:
                np.savetxt(
                    f,
                    feature_info,
                    delimiter=',',
                    fmt=','.join(['%f'] + ['%d'] + ['%f'] * 2 + [descriptor_format] * N))

    def onResultsReady(self):
        pass


class TUMVIEvalModeSaver(EvalModeSaver, TUMVISaver):
    def __init__(self, args):
        EvalModeSaver.__init__(self, args)
        TUMVISaver.__init__(self, args)


class TUMVIDumpModeSaver(DumpModeSaver, TUMVISaver):
    def __init__(self, args):
        DumpModeSaver.__init__(self, args)
        TUMVISaver.__init__(self, args)


class TUMVICovDumpModeSaver(CovDumpModeSaver, TUMVISaver):
    def __init__(self, args):
        CovDumpModeSaver.__init__(self, args)
        TUMVISaver.__init__(self, args)


class TUMVITrackerDumpModeSaver(TrackerDumpModeSaver, TUMVISaver):
    def __init__(self, args):
        TrackerDumpModeSaver.__init__(self, args)
        TUMVISaver.__init__(self, args)


class CarlaEvalModeSaver(EvalModeSaver, CarlaSaver):
    def __init__(self, args):
        EvalModeSaver.__init__(self, args)
        CarlaSaver.__init__(self, args)


class CarlaDumpModeSaver(DumpModeSaver, CarlaSaver):
    def __init__(self, args):
        DumpModeSaver.__init__(self, args)
        CarlaSaver.__init__(self, args)


class CarlaCovDumpModeSaver(CovDumpModeSaver, CarlaSaver):
    def __init__(self, args):
        CovDumpModeSaver.__init__(self, args)
        CarlaSaver.__init__(self, args)


class CarlaTrackerDumpModeSaver(TrackerDumpModeSaver, CarlaSaver):
    def __init__(self, args):
        TrackerDumpModeSaver.__init__(self, args)
        CarlaSaver.__init__(self, args)


class XIVOEvalModeSaver(EvalModeSaver, BaseSaver):
    def __init__(self, args):
        EvalModeSaver.__init__(self, args)
        BaseSaver.__init__(self, args)


class XIVODumpModeSaver(DumpModeSaver, BaseSaver):
    def __init__(self, args):
        DumpModeSaver.__init__(self, args)
        BaseSaver.__init__(self, args)


class XIVOCovDumpModeSaver(CovDumpModeSaver, BaseSaver):
    def __init__(self, args):
        CovDumpModeSaver.__init__(self, args)
        BaseSaver.__init__(self, args)

class XIVOTrackerDumpModeSaver(TrackerDumpModeSaver, BaseSaver):
    def __init__(self, args):
        TrackerDumpModeSaver.__init__(self, args)
        BaseSaver.__init__(self, args)


class VOIDEvalModeSaver(EvalModeSaver, VOIDSaver):
    def __init__(self, args):
        EvalModeSaver.__init__(self, args)
        VOIDSaver.__init__(self, args)


class VOIDDumpModeSaver(DumpModeSaver, VOIDSaver):
    def __init__(self, args):
        DumpModeSaver.__init__(self, args)
        VOIDSaver.__init__(self, args)


class VOIDCovDumpModeSaver(CovDumpModeSaver, VOIDSaver):
    def __init__(self, args):
        CovDumpModeSaver.__init__(self, args)
        VOIDSaver.__init__(self, args)


class VOIDTrackerDumpModeSaver(TrackerDumpModeSaver, VOIDSaver):
    def __init__(self, args):
        TrackerDumpModeSaver.__init__(self, args)
        VOIDSaver.__init__(self, args)


class PCWEvalModeSaver(EvalModeSaver, PCWSaver):
    def __init__(self, imu_traj, args):
        args.dataset = "pcw"
        args.cam_id = 0
        args.seq = args.motion_type
        EvalModeSaver.__init__(self, args)
        PCWSaver.__init__(self, imu_traj, args)


class PCWDumpModeSaver(DumpModeSaver, PCWSaver):
    def __init__(self, imu_traj, args):
        args.dataset = "pcw"
        args.cam_id = 0
        args.seq = args.motion_type
        DumpModeSaver.__init__(self, args)
        PCWSaver.__init__(self, imu_traj, args)


class PCWCovDumpModeSaver(CovDumpModeSaver, PCWSaver):
    def __init__(self, imu_traj, args):
        args.dataset = "pcw"
        args.cam_id = 0
        args.seq = args.motion_type
        CovDumpModeSaver.__init__(self, args)
        PCWSaver.__init__(self, imu_traj, args)


class PCWTrackerDumpModeSaver(PCWSaver, BaseSaver):
    def __init__(self, imu_traj, args):
        args.dataset = "pcw"
        args.cam_id = 0
        args.seq = args.motion_type
        PCWSaver.__init__(self, imu_traj, args)
        BaseSaver.__init__(self, args)

    def onVisionUpdate(self, estimator, datum):
        ts, content = datum

        tracked_features = estimator.tracked_features_no_descriptor()

        for f in tracked_features:
            id, kp = f

            # Convert timestamp to seconds from nanoseconds
            ts_and_id = np.array([ts*1e-9, id]).reshape(1,2)
            kp = np.reshape(kp, (1, 2))

            feature_info = np.concatenate((ts_and_id, kp), axis=1)

            with open(self.resultsPath, "a") as f:
                np.savetxt(
                    f,
                    feature_info,
                    delimiter=',',
                    fmt=','.join(['%f'] + ['%d'] + ['%f'] * 2))

    def onResultsReady(self):
        pass

