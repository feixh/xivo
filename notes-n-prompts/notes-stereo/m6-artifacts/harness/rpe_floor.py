"""Metric noise floor of evaluate_rpe.py at delta=1s.

A *perfect* estimator outputs the true pose at each image timestamp. The tool
compares its motion between two image stamps against ground-truth motion
between the two *nearest ground-truth stamps* -- so it charges the estimator
for omega * (delta1 - delta0), where each delta is up to half a GT interval.
Build exactly that perfect estimate (GT slerp'd/lerp'd to the image stamps) and
measure what the tool reports for it.
"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

gt_f, est_f, out_f = sys.argv[1], sys.argv[2], sys.argv[3]
gt = np.loadtxt(gt_f)
est = np.loadtxt(est_f)
tg, tq = gt[:, 0], gt[:, 4:8]           # tx ty tz qx qy qz qw
rot = Rotation.from_quat(tq)
slerp = Slerp(tg, rot)
te = est[:, 0]
m = (te >= tg[0]) & (te <= tg[-1])
te = te[m]
pos = np.vstack([np.interp(te, tg, gt[:, i]) for i in (1, 2, 3)]).T
q = slerp(te).as_quat()
np.savetxt(out_f, np.column_stack([te, pos, q]), fmt='%.9f')
print('%d of %d est stamps inside GT span' % (m.sum(), len(est)))
