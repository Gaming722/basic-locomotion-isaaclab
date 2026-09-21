"""Preview a live D435 depth frame like the DAgger dual-pane student-depth view."""

import argparse

import cv2
import numpy as np

from _common import add_stream_arguments, depth_stream, run


STUDENT_SIZE = (106, 60)  # (width, height)
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 2.0
WINDOW = 'D435 student-depth preview'


def prepare_depth(raw_z16: np.ndarray, depth_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one 106x60 metre frame and its dual-pane-equivalent grayscale image.

    Sample source pixels without blending: averaging raw zero pixels into valid
    depths would turn missing measurements into false near obstacles.
    """
    if raw_z16.ndim != 2 or raw_z16.dtype != np.uint16:
        raise ValueError('Expected a 2-D Z16 (uint16) depth frame')
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError('Depth scale must be finite and positive')

    sampled = cv2.resize(raw_z16, STUDENT_SIZE, interpolation=cv2.INTER_NEAREST_EXACT)
    depth_m = sampled.astype(np.float32) * depth_scale
    depth_m[sampled == 0] = MAX_DEPTH_M
    np.clip(depth_m, MIN_DEPTH_M, MAX_DEPTH_M, out=depth_m)

    # Same near=white, far/zero=black mapping as --dual_pane_student_depth.
    normalized = (depth_m - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M)
    gray = (255.0 * (1.0 - normalized)).astype(np.uint8)
    return depth_m, gray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stream_arguments(parser)
    args = parser.parse_args()

    print('[INFO] 106x60 single-frame preview: zero -> 2.0 m (black); '
          'valid depth clipped to [0.1, 2.0] m. Q/Esc exits.')
    print('[INFO] No added blur, noise, dropout, delay, or history.')
    try:
        with depth_stream(args) as (_, pipeline, _, scale):
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 848, 480)
            while True:
                frame = pipeline.wait_for_frames(args.timeout_ms).get_depth_frame()
                if not frame:
                    continue
                raw = np.asanyarray(frame.get_data())
                _, gray = prepare_depth(raw, scale)
                cv2.imshow(WINDOW, gray)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run(main)
