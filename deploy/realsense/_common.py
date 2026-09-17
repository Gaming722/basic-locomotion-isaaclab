"""Shared CLI configuration for raw, unaligned RealSense depth tests."""
from contextlib import contextmanager


def positive_int(value):
    import argparse
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return number


def add_stream_arguments(parser):
    parser.add_argument('--width', type=positive_int, default=848)
    parser.add_argument('--height', type=positive_int, default=480)
    parser.add_argument('--fps', type=positive_int, default=30)
    parser.add_argument('--serial', help='Select a device when multiple cameras are connected')
    parser.add_argument('--timeout_ms', type=positive_int, default=5000)


@contextmanager
def depth_stream(args):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError('Install deploy/realsense/requirements.txt in your active environment') from exc
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        scale = device.first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        print(f'[INFO] {device.get_info(rs.camera_info.name)} '
              f'serial={device.get_info(rs.camera_info.serial_number)} '
              f'depth={intr.width}x{intr.height}@{args.fps} '
              f'scale={scale:.9g} m/unit')
        yield rs, pipeline, profile, scale
    finally:
        pipeline.stop()


def run(main):
    try:
        main()
    except KeyboardInterrupt:
        print('\n[INFO] Stopped.')
    except RuntimeError as exc:
        raise SystemExit(f'[ERROR] {exc}\nCheck USB connection, device permissions, '
                         'stream support and whether another process owns the camera.') from exc
