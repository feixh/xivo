#!/usr/bin/env python3
"""Does gyro-de-rotated optical flow separate a moving rig from a static one?

M0 established that no accelerometer statistic can see constant-velocity motion,
so the detector needs a cue that responds to *translation*. Vision is the only
candidate, but the obvious visual statistic -- raw pixel disparity between two
frames, which is what OpenVINS thresholds at 10 px -- conflates two things:

  * translation, scaled by 1/depth, which is the signal, and
  * rotation, which produces disparity at *any* depth and which every one of
    these 17 sequences has (0.08-0.32 rad/s at init).

Removing the rotational component with the gyro leaves a statistic that is zero
for a stationary camera regardless of depth and regardless of how fast it is
turning. This script measures both, so the choice between them is a measurement
rather than an argument:

  raw_med   median |x2 - x1| in pixels over tracked features
  derot_med median |x2 - predict(R_gyro, x1)| in pixels, same features

  rotfit_med median |x2 - predict(R_fitted, x1)| in pixels, where R_fitted is
            the rotation that *best* explains the flow, found from the images

and reports the minimum over candidate windows for each, mirroring the
min-over-windows structure M0 showed the IMU cue needs.

The third statistic exists because the second one does not work. De-rotating with
the gyro assumes the gyro is unbiased, and at initialization it is not: EuRoC's
ADIS16448 has a turn-on gyro bias of 0.079-0.085 rad/s, which is essentially all
of the 0.08 rad/s these sequences read while sitting still. Over one 50 ms frame
gap that is 0.08 * 0.05 * 458 = 1.8 px of predicted motion that never happened,
and it lands on top of a translation signal of comparable size. The bias is one
of the quantities dynamic initialization is trying to estimate, so using it to
build the detector that decides whether to run the estimator is circular.

Fitting the rotation from the images instead removes the circularity. A camera
that only rotates produces flow that *some* rotation explains exactly, at any
depth and any rate; a camera that translates produces parallax that no rotation
can explain, because parallax depends on depth and rotation does not. So the
residual of the best-fit rotation is a translation signal that is independent of
the gyro bias and of scene scale. Its known degeneracy is a scene at constant
depth, where a homography absorbs translation -- reported, not hidden.

Usage (from the repository root, venv on PATH):
  python3 notes-n-prompts/notes-dyninit/harness/flow_diag.py \
      --dataset euroc --root ../data/euroc --cfg cfg/euroc_stereo.json
"""

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from init_diag import (EUROC_SEQS, TUMVI_SEQS, load_cfg, load_imu, so3_exp)


def camera_from_cfg(cfg):
    """-> (K, D, model) in the form cv2 wants."""
    c = cfg["camera_cfg"]
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])
    if c["model"] == "radtan":
        k = c["k012"]
        D = np.array([k[0], k[1], c["p1"], c["p2"], k[2]], dtype=float)
        return K, D, "radtan"
    if c["model"] == "equidistant":
        return K, np.array(c["k0123"], dtype=float), "equi"
    raise SystemExit("unhandled camera model %s" % c["model"])


def undistort(pts, K, D, model):
    """pixels (N,2) -> normalized bearings (N,3)."""
    p = pts.reshape(-1, 1, 2).astype(np.float64)
    if model == "radtan":
        n = cv2.undistortPoints(p, K, D)
    else:
        n = cv2.fisheye.undistortPoints(p, K, D)
    n = n.reshape(-1, 2)
    return np.hstack([n, np.ones((len(n), 1))])


def project(bearings, K, D, model):
    """normalized bearings (N,3) -> pixels (N,2)."""
    xyz = bearings.reshape(-1, 1, 3).astype(np.float64)
    zero = np.zeros(3)
    if model == "radtan":
        px, _ = cv2.projectPoints(xyz, zero, zero, K, D)
    else:
        px, _ = cv2.fisheye.projectPoints(xyz, zero, zero, K, D)
    return px.reshape(-1, 2)


def gyro_rotation(t, gyro, t1, t2):
    """R_{b1<-b2} from trapezoidal integration of body-frame rate over [t1,t2].

    Rdot_{w<-b} = R_{w<-b} [w]x, so composing exp(w dt) on the right accumulates
    R_{b1<-b2} directly.
    """
    i0 = int(np.searchsorted(t, t1))
    i1 = int(np.searchsorted(t, t2))
    R = np.eye(3)
    for k in range(max(1, i0), min(i1 + 1, len(t))):
        dt = t[k] - t[k - 1]
        R = R @ so3_exp(0.5 * (gyro[k] + gyro[k - 1]) * dt)
    return R


def fit_rotation(u1, u2, iters=4, huber_rad=0.002):
    """The rotation that best explains **unit** bearings `u1` landing at `u2`.

    Fitting on the unit sphere rather than in normalized image coordinates: this
    is Wahba's problem, `min_R sum w_i |u2_i - R u1_i|^2`, whose solution is
    closed form. With `M = sum w_i u2_i u1_i'` and `M = U S V'`,

        R = U diag(1, 1, det(U V')) V'

    which needs no seed, no iteration for the rotation itself, and -- the reason
    it is used here rather than a Gauss-Newton fit on normalized coordinates --
    is uniformly conditioned across the field of view. Normalized coordinates are
    `tan` of the field angle, so on a 180-degree fisheye like TUM-VI's they
    diverge toward the image edge and both the fit weights and the residual
    become meaningless. Unit bearings are well behaved for any camera model.

    Three IRLS passes with a Huber weight keep a few bad tracks from dragging the
    fit, which matters because this fit's residual *is* the statistic.

    Returns (R, per-point angular residual in radians).
    """
    w = np.ones(len(u1))
    R = np.eye(3)
    for _ in range(iters):
        M = (u2 * w[:, None]).T @ u1
        U, _, Vt = np.linalg.svd(M)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        ang = np.arccos(np.clip(np.sum((R @ u1.T).T * u2, axis=1), -1.0, 1.0))
        w = np.where(ang < huber_rad, 1.0, huber_rad / np.maximum(ang, 1e-12))
    return R, ang


def measure(seq_dir, cfg, horizon_s, win_s, max_feats=200):
    t_imu, gyro, _ = load_imu(seq_dir)
    t0_ns = np.loadtxt(os.path.join(seq_dir, "mav0/imu0/data.csv"),
                       delimiter=",", comments="#", max_rows=1)[0]

    K, D, model = camera_from_cfg(cfg)
    R_cb = np.array(cfg["X"]["Wbc"], dtype=float).T  # Wbc is R_body_from_camera

    cam = np.genfromtxt(os.path.join(seq_dir, "mav0/cam0/data.csv"),
                        delimiter=",", comments="#", dtype=None, encoding=None)
    ts = np.array([int(r[0]) for r in cam])
    names = [str(r[1]).strip() for r in cam]
    t_cam = (ts - t0_ns) * 1e-9
    keep = (t_cam >= 0) & (t_cam <= horizon_s)
    t_cam, names = t_cam[keep], [n for n, k in zip(names, keep) if k]

    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev_img, prev_pts, prev_t = None, None, None
    raw, derot, rotfit, bias = [], [], [], []

    for name, tc in zip(names, t_cam):
        img = cv2.imread(os.path.join(seq_dir, "mav0/cam0/data", name),
                         cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if prev_img is not None and prev_pts is not None and len(prev_pts) >= 20:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_img, img, prev_pts,
                                                  None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(img, prev_img, nxt,
                                                    None, **lk)
            # Forward-backward consistency, so that a bad track cannot masquerade
            # as translation. This is the only outlier rejection here.
            fb = np.linalg.norm(back.reshape(-1, 2) - prev_pts.reshape(-1, 2),
                                axis=1)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
            if ok.sum() >= 15:
                p1 = prev_pts.reshape(-1, 2)[ok]
                p2 = nxt.reshape(-1, 2)[ok]
                raw.append((tc, float(np.median(
                    np.linalg.norm(p2 - p1, axis=1)))))

                R_b1b2 = gyro_rotation(t_imu, gyro, prev_t, tc)
                R_c1c2 = R_cb @ R_b1b2 @ R_cb.T
                b1 = undistort(p1, K, D, model)
                # A world-fixed point seen by a camera that only rotated:
                # x_{c2} ~ R_{c2<-c1} x_{c1}.
                b2p = (R_c1c2.T @ b1.T).T
                keepz = b2p[:, 2] > 1e-6
                if keepz.sum() >= 15:
                    pred = project(b2p[keepz] / b2p[keepz, 2:3], K, D, model)
                    derot.append((tc, float(np.median(
                        np.linalg.norm(p2[keepz] - pred, axis=1)))))

                # The bias-free cue: the best-fit rotation, from the images alone.
                b2 = undistort(p2, K, D, model)
                u1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
                u2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
                R_fit, _ = fit_rotation(u1, u2)
                # Report the residual in **pixels**, by projecting the rotated
                # bearing through the real camera model rather than scaling an
                # angle by the focal length -- the two differ by a factor of
                # several toward the edge of a wide-angle image.
                rot1 = (R_fit @ u1.T).T
                fwd = rot1[:, 2] > 0.05  # keep the projection well posed
                if fwd.sum() >= 15:
                    pf = project(rot1[fwd], K, D, model)
                    rotfit.append((tc, float(np.median(
                        np.linalg.norm(p2[fwd] - pf, axis=1)))))
                # How far the gyro was from what the images say, as a rate. If
                # this comes out near the datasheet turn-on bias while the rig is
                # still, the gyro-de-rotation cue is measuring bias.
                dR = R_c1c2 @ R_fit
                ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
                bias.append((tc, ang / max(tc - prev_t, 1e-6)))
        # Re-detect every frame: over a 5 s horizon we care about frame-to-frame
        # flow, not long tracks, and re-detecting keeps the sample count high.
        prev_img, prev_t = img, tc
        prev_pts = cv2.goodFeaturesToTrack(img, maxCorners=max_feats,
                                          qualityLevel=0.01, minDistance=12)
    return raw, derot, rotfit, bias


def min_over_windows(series, win_s):
    """Smallest window-mean of a (t, value) series over any `win_s` window."""
    if not series:
        return float("nan")
    t = np.array([s[0] for s in series])
    v = np.array([s[1] for s in series])
    best = np.inf
    for i in range(len(t)):
        j = int(np.searchsorted(t, t[i] + win_s))
        if j - i >= 5:
            best = min(best, float(np.mean(v[i:j])))
    return best if np.isfinite(best) else float(np.mean(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["euroc", "tumvi"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--horizon", type=float, default=5.0)
    ap.add_argument("--win", type=float, default=0.5)
    ap.add_argument("--seqs", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    seqs = args.seqs or (EUROC_SEQS if args.dataset == "euroc" else TUMVI_SEQS)

    print("# %s, cfg %s, horizon %.1fs, window %.2fs"
          % (args.dataset, args.cfg, args.horizon, args.win))
    hdr = ("%-26s %8s %8s %8s %8s %9s %9s %8s %5s"
           % ("sequence", "raw_med", "raw_min", "derot_md", "derot_mn",
              "rotfit_md", "rotfit_mn", "w_err", "pairs"))
    print(hdr)
    print("-" * len(hdr))
    out = {}

    def med(series):
        return float(np.median([v for _, v in series])) if series else float("nan")

    for s in seqs:
        seq_dir = os.path.join(args.root, s)
        if not os.path.isdir(seq_dir):
            print("%-26s  MISSING" % s)
            continue
        raw, derot, rotfit, bias = measure(seq_dir, cfg, args.horizon, args.win)
        out[s] = dict(raw_med=med(raw), raw_min=min_over_windows(raw, args.win),
                      derot_med=med(derot),
                      derot_min=min_over_windows(derot, args.win),
                      rotfit_med=med(rotfit),
                      rotfit_min=min_over_windows(rotfit, args.win),
                      w_err=med(bias), pairs=len(rotfit))
        o = out[s]
        print("%-26s %8.3f %8.3f %8.3f %8.3f %9.3f %9.3f %8.4f %5d"
              % (s, o["raw_med"], o["raw_min"], o["derot_med"], o["derot_min"],
                 o["rotfit_med"], o["rotfit_min"], o["w_err"], o["pairs"]))
    print()
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
