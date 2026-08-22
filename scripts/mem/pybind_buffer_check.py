"""Exercises the numpy-buffer overload of the pyxivo VisualMeas bindings.

Run it through scripts/mem/pybind_buffer_check.sh, which points PYTHONPATH at an
ASan build and preloads the runtime.  Two properties are under test (M4 / L3-4):

  * the geometry of the `cv::Mat` must come from the array *shape*, not from its
    strides plus a hard-coded CV_8UC3.  With the old code a (H, W) grayscale
    array became an H x W CV_8UC3 image over an H*W-byte buffer -- reproducibly a
    SEGV in OpenCV's copy for a 512x512 frame.

  * the pixels must be copied.  `Estimator::VisualMeas` buffers the message and
    processes the *oldest* one it holds (`MaintainBuffer`), so a `cv::Mat` that
    merely wraps `info.ptr` is read after this call returns -- after python may
    have freed the array.  This one is not reliably sanitizer-visible (the block
    is usually recycled by the next numpy allocation, so the read lands in valid
    memory and merely returns garbage), which is why the fix is a clone rather
    than a test assertion.

Usage: pybind_buffer_check.py {gray|color} [frames]
"""

import sys

import numpy as np
import pyxivo

CFG = "cfg/tumvi_cam0.json"
WIDTH = HEIGHT = 512  # the tumvi_cam0 camera model
IMU_PER_FRAME = 10  # ~200 Hz IMU against ~20 Hz frames
IMU_PERIOD_NS = 5_000_000


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "gray"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if which not in ("gray", "color"):
        print(f"expecting 'gray' or 'color', got {which!r}", file=sys.stderr)
        return 2
    shape = (HEIGHT, WIDTH) if which == "gray" else (HEIGHT, WIDTH, 3)

    rng = np.random.default_rng(0)
    est = pyxivo.Estimator(CFG, "", "buffer_check", False)

    ts = 0
    for _ in range(frames):
        for _ in range(IMU_PER_FRAME):
            est.InertialMeas(ts, 0.0, 0.0, 0.0, 0.0, 0.0, 9.8)
            ts += IMU_PERIOD_NS
        img = rng.integers(0, 255, size=shape, dtype=np.uint8)
        est.VisualMeas(ts, img)
        del img
        # Churn the heap so that a buffer the estimator still points at is both
        # freed and reused before the frame is processed.
        for _ in range(3):
            junk = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
            del junk

    print(f"OK {which}: detections={est.num_tracker_new_detections()} "
          f"instate={est.num_instate_features()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
