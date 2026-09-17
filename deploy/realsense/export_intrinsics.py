"""Export actual raw depth calibration; does not measure robot mounting extrinsics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from _common import add_stream_arguments, depth_stream, positive_int, run


def calibration(intr, width, height):
    sx, sy = width / intr.width, height / intr.height
    return dict(width=width, height=height,
                intrinsics=[[intr.fx * sx, 0., intr.ppx * sx],
                            [0., intr.fy * sy, intr.ppy * sy], [0., 0., 1.]],
                distortion_model=str(intr.model), distortion_coefficients=list(intr.coeffs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_stream_arguments(parser)
    parser.add_argument('--target_width', type=positive_int, default=106)
    parser.add_argument('--target_height', type=positive_int, default=60)
    parser.add_argument('--output', type=Path, help='Output JSON (default: timestamped logs/realsense file)')
    args = parser.parse_args()
    with depth_stream(args) as (rs, pipeline, profile, scale):
        # Confirm that this configuration actually produces frames.
        pipeline.wait_for_frames(args.timeout_ms)
        stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = stream.get_intrinsics()
        device = profile.get_device()
        data = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    device_name=device.get_info(rs.camera_info.name),
                    serial=device.get_info(rs.camera_info.serial_number),
                    firmware=device.get_info(rs.camera_info.firmware_version),
                    fps=stream.fps(), format=str(stream.format()), depth_scale_m_per_unit=scale,
                    convention='ros', depth_type='distance_to_image_plane', aligned_to_color=False,
                    native=calibration(intr, intr.width, intr.height),
                    resized=calibration(intr, args.target_width, args.target_height),
                    resize_convention='continuous image coordinates; pixel centres at index + 0.5',
                    position_base=None, quaternion_wxyz=None,
                    note='Mounting extrinsics require separate measurement/calibration. '
                         'Resized K assumes full-image resize, no crop and no rectification.')
    output = args.output or (Path(__file__).resolve().parents[2] / 'logs' / 'realsense' /
                            f'intrinsics_{data["serial"]}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never silently replace previous calibration measurements.
    with output.open('x', encoding='utf-8') as file:
        json.dump(data, file, indent=2)
        file.write('\n')
    print(json.dumps(data, indent=2))
    print(f'[INFO] Saved: {output}')


if __name__ == '__main__':
    run(main)
