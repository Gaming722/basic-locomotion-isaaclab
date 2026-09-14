"""Nominal D435 bottom-screw to depth-origin geometry (metres).

Source: RealSense official _d435.urdf.xacro (ros2-master).
https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d435.urdf.xacro
Bottom-screw axes: X forward, Y left, Z up. The nominal depth origin
coincides with depth/infra1 frame; it is not the housing or RGB centre.
"""

import math

D435_BOTTOM_SCREW_POSITION_BASE = (0.26, 0.0, 0.12)
D435_MOUNT_PITCH_DEG = 30.0
# Forward offset includes front-glass and zero-depth-reference corrections.
D435_DEPTH_ORIGIN_IN_SCREW = (0.0149 - 0.0001 - 0.0042, 0.0175, 0.025 / 2)


def d435_depth_position_base(screw_position=D435_BOTTOM_SCREW_POSITION_BASE,
                             pitch_deg=D435_MOUNT_PITCH_DEG):
    """Compose screw translation with Ry(pitch) * local depth-origin offset."""
    pitch = math.radians(pitch_deg)
    c, s = math.cos(pitch), math.sin(pitch)
    x, y, z = D435_DEPTH_ORIGIN_IN_SCREW
    return (screw_position[0] + c * x + s * z,
            screw_position[1] + y,
            screw_position[2] - s * x + c * z)


D435_DEPTH_POSITION_BASE = d435_depth_position_base()
# ROS optical axes: X right, Y down, Z forward. Ry(30 deg) already
# composed with the fixed body-to-optical rotation; do not apply it twice.
D435_DEPTH_QUATERNION_ROS_WXYZ = (-0.353553, 0.612372, -0.612372, 0.353553)
