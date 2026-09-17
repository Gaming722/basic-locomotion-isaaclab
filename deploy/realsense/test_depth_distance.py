"""Read the centre pixel's raw optical Z-depth in metres."""
import argparse
import time

from _common import add_stream_arguments, depth_stream, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_stream_arguments(parser)
    args = parser.parse_args()
    last_print = -float('inf')
    with depth_stream(args) as (_, pipeline, _, _):
        while True:
            depth = pipeline.wait_for_frames(args.timeout_ms).get_depth_frame()
            if not depth:
                continue
            now = time.monotonic()
            if now - last_print < .2:
                continue
            u, v = depth.get_width() // 2, depth.get_height() // 2
            distance = depth.get_distance(u, v)
            text = f'{distance:.3f} m' if distance > 0 else 'invalid / no depth'
            print(f'[DEPTH] pixel=({u},{v}) Z={text:<24}', end='\r', flush=True)
            last_print = now


if __name__ == '__main__':
    run(main)
