"""Display raw depth, invalid pixels and an observed host frame rate."""
import argparse
import time

from _common import add_stream_arguments, depth_stream, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_stream_arguments(parser)
    parser.add_argument('--max_distance', type=float, default=3., help='Display range in metres')
    args = parser.parse_args()
    if not 0 < args.max_distance < float('inf'):
        parser.error('--max_distance must be finite and positive')
    import cv2
    import numpy as np

    try:
        with depth_stream(args) as (_, pipeline, _, scale):
            start, count, fps = time.monotonic(), 0, 0.
            cv2.namedWindow('RealSense raw depth', cv2.WINDOW_NORMAL)
            while True:
                frame = pipeline.wait_for_frames(args.timeout_ms).get_depth_frame()
                if not frame:
                    continue
                raw = np.asanyarray(frame.get_data())
                metres = raw.astype(np.float32) * scale
                valid = raw > 0
                image = (np.clip(metres / args.max_distance, 0, 1) * 255).astype(np.uint8)
                image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
                image[~valid] = 0
                count += 1
                elapsed = time.monotonic() - start
                if elapsed >= 1:
                    fps, start, count = count / elapsed, time.monotonic(), 0
                centre = float(metres[raw.shape[0] // 2, raw.shape[1] // 2])
                label = f'{centre:.3f}m' if centre > 0 else 'invalid'
                cv2.putText(image, f'host FPS={fps:.1f} valid={valid.mean():.1%} centre Z={label}',
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                cv2.imshow('RealSense raw depth', image)
                if cv2.waitKey(1) & 0xff in (27, ord('q')):
                    break
                if cv2.getWindowProperty('RealSense raw depth', cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run(main)
