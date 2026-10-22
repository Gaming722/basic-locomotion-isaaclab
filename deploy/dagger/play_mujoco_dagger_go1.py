"""Sim2sim: run the GO1 DAgger depth student in MuJoCo.

Loads a dagger_policy.pt (Tiled or RayCaster student; both share the same D435
depth geometry) + its metadata, builds a scene XML from the committed copy of
mujoco_menagerie's unitree_go1/go1.xml (deploy/mujoco_models/go1/), and rolls the
policy out in MuJoCo. Depth is rendered by the MuJoCo offscreen renderer at the
training D435 pose and preprocessed exactly like training `_sanitize_depth_data`
(clip [0.1,2.0] + far-saturation + blur + delay, all from the checkpoint metadata).

Reference: basic-locomotion-isaaclab / m1-perceptive-baseline deploy/dagger.
"""
import os
import sys

# --- GL backend selection --------------------------------------------------
# --viewer needs a windowed backend (glfw); headless uses egl. MUJOCO_GL is read
# once at `import mujoco`, so pick the backend BEFORE that import.
if "--viewer" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import argparse
import json
import math
import re
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import mujoco
import mujoco.viewer  # imports fine headless; only window creation needs a display
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts", "dagger"))
from mujoco_depth import MujocoDepthCamera
from dagger_network import DaggerNet
from depth_pipeline import preprocess_depth
from proprio_noise import ProprioNoiseCfg, ProprioceptiveNoise
from go1_deploy_core import (COMMAND_LIMIT, DEFAULT_JOINT, DAMPING, PASSIVE,
                             POLICY_JOINT_NAMES, RL, SDK_JOINT_NAMES,
                             SDK_VELOCITY_LIMIT, SITDOWN, STAND,
                             DeploymentCore, StateMachine, core_sha256,
                             sdk_position_limit, sdk_to_policy, verify_bundle)
sys.path.insert(0, os.path.join(REPO, "source", "basic_locomotion_isaaclab",
                              "basic_locomotion_isaaclab", "assets"))
from d435_geometry import D435_DEPTH_POSITION_BASE, D435_MOUNT_PITCH_DEG, D435_DEPTH_QUATERNION_ROS_WXYZ

GO1_XML = os.path.join(REPO, "deploy", "mujoco_models", "go1", "go1.xml")

# Policy action/joint order (matches Go1FlatEnvCfg.desired_joints_order):
# hip block, thigh block, calf block, each FL/FR/RL/RR.
DESIRED_ORDER = list(POLICY_JOINT_NAMES)
SDK_ORDER = list(SDK_JOINT_NAMES)
# Official Go1 URDF velocity limits, in Unitree SDK joint order.  MuJoCo does
# not impose these as hard constraints, so report violations explicitly before
# transferring a controller to hardware.
GO1_VELOCITY_LIMIT_SDK = SDK_VELOCITY_LIMIT

# GO1 training defaults (fallback if params/env.yaml is missing).
GO1_DEFAULTS = dict(action_scale=0.5, clip_actions=3.0, use_filter=True,
                    step_freq=1.4, sim_dt=0.004, decimation=5, history_length=5,
                    kp=35.0, kd=0.5, armature=0.01)

# MuJoCo camera geometry baked into the scene XML (training D435-equivalent).
CAM_POS = D435_DEPTH_POSITION_BASE   # depth origin in trunk/base frame
CAM_PITCH_DEG = D435_MOUNT_PITCH_DEG
# Training renders at 106x60 (D435 848x480 / 8, aspect 1.7667) with HFOV 87 deg, so the
# vertical FOV is 2*atan(tan(43.5 deg) * 60/106) ~ 56.5 deg. MuJoCo cameras specify fovy
# (vertical), which reproduces the same projection at 106x60.
CAM_FOVY = 56.5              # vertical FOV deg
DEPTH_W, DEPTH_H = 106, 60

# Training dynamics parameters to align (go1_asset.py DelayedPDActuatorCfg).
GO1_MIN_DELAY = 0             # actuator command delay (physics steps), randomized per env
GO1_MAX_DELAY = 2
# NOTE: soft_joint_pos_limit_factor=0.95 is NOT a command clamp in training -- it only feeds
# the joint_pos_limits reward / joint_pos_out_of_limits termination (JointPositionActionCfg
# sets the target directly via set_joint_position_target with no clamp). So we do NOT replicate
# it here; position actuator target clamping is disabled below.


# --- XML building (m1-style: extract inner, absolute meshdir, inject terrain+camera) ----

def _go1_xml_inner(kp: float = 35.0, kd: float = 0.5, armature: float = 0.01,
                   torque_control: bool = False):
    """Return GO1 XML with training gains and unclamped position targets."""
    with open(GO1_XML) as source:
        txt = source.read()
    i = txt.find("<mujoco")
    j = txt.find(">", i) + 1
    k = txt.rfind("</mujoco>")
    inner = txt[j:k]
    mdir = os.path.join(os.path.dirname(GO1_XML), "assets")
    inner = re.sub(r'meshdir="[^"]*"', f'meshdir="{mdir}"', inner)
    # Override the Menagerie defaults with nominal training dynamics.
    root = ET.fromstring('<mujoco>' + inner + '</mujoco>')
    for joint in root.findall('.//default/joint'):
        joint.set('damping', str(kd))
        joint.set('armature', str(armature))
        joint.set('frictionloss', '0')
    for actuator in root.findall('.//default/position'):
        if torque_control:
            actuator.tag = 'motor'
            actuator.attrib.pop('kp', None)
            actuator.attrib.pop('ctrlrange', None)
        else:
            if 'kp' in actuator.attrib:
                actuator.set('kp', str(kp))
            # IsaacLab clips raw actions, not the resulting position targets.
            actuator.set('ctrllimited', 'false')
    if torque_control:
        for actuator in root.findall('.//actuator/position'):
            actuator.tag = 'motor'
    inner = ''.join(ET.tostring(child, encoding='unicode') for child in root)
    return inner


def _fbm_values(shape, scales=(1, 2, 4, 8, 16), amplitudes=(0.25, 0.25, 0.2, 0.18, 0.12), seed=0):
    """Multi-scale fractal noise in [0, 1] (perlin terrain heightfield)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape)
    total = sum(amplitudes)
    for scale, amp in zip(scales, amplitudes):
        gx, gy = max(2, shape[0] // scale + 1), max(2, shape[1] // scale + 1)
        grid = rng.uniform(-1.0, 1.0, (gx, gy))
        xs = np.linspace(0, gx - 1, shape[0])
        ys = np.linspace(0, gy - 1, shape[1])
        x0, x1 = np.floor(xs).astype(int), np.minimum(np.floor(xs).astype(int) + 1, gx - 1)
        y0, y1 = np.floor(ys).astype(int), np.minimum(np.floor(ys).astype(int) + 1, gy - 1)
        fx, fy = (xs - x0)[:, None], (ys - y0)[None, :]
        out += amp * (
            grid[x0][:, y0] * (1 - fx) * (1 - fy) + grid[x1][:, y0] * fx * (1 - fy)
            + grid[x0][:, y1] * (1 - fx) * fy + grid[x1][:, y1] * fx * fy
        )
    return (out / total + 1.0) / 2.0


def _sync_realtime(wall_start, sim_elapsed):
    """Wait for the cumulative simulation deadline; never change physics dt."""
    remaining = wall_start + sim_elapsed - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def _parse_fsm_script(value):
    allowed = {name: name for name in (PASSIVE, DAMPING, STAND, SITDOWN, RL)}
    events = []
    for item in value.split(','):
        try:
            at, mode = item.split(':', 1)
            at = float(at)
            mode = mode.strip().upper()
        except ValueError as error:
            raise ValueError("FSM script must be 'seconds:MODE,...'") from error
        if not np.isfinite(at) or at < 0 or mode not in allowed:
            raise ValueError(f"invalid FSM event: {item}")
        events.append([at, allowed[mode]])
    if any(events[i][0] >= events[i + 1][0] for i in range(len(events) - 1)):
        raise ValueError("FSM event times must be strictly increasing")
    return events


def _roll_pitch(quat_wxyz):
    w, x, y, z = quat_wxyz
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(np.clip(2 * (w*y - z*x), -1.0, 1.0))
    return roll, pitch


def _camera_quat():
    """Orthogonal camera frame quat (image-up = world-up projected onto the plane
    perpendicular to the 30-deg-down look axis); avoids a skewed depth image."""
    cam_look = np.array([math.cos(math.radians(CAM_PITCH_DEG)), 0.0, -math.sin(math.radians(CAM_PITCH_DEG))])
    cam_z = -cam_look
    cam_x = np.cross(np.array([0.0, 0.0, 1.0]), cam_z)
    cam_x = cam_x / np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)
    cam_R = np.column_stack([cam_x, cam_y, cam_z]).flatten()
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, cam_R)
    return q


def build_go1_scene_xml(scene="flat", step_rise=0.12, step_tread=0.30, n_steps=12,
                        width=6.0, perlin_amp=0.18, kp=35.0, kd=0.5, armature=0.01,
                        camera_position=CAM_POS, camera_quaternion=None,
                        camera_intrinsics=None, camera_size=(DEPTH_H, DEPTH_W),
                        torque_control=False):
    """Compose a scene XML: go1 robot + terrain + baked depth camera."""
    inner = _go1_xml_inner(kp=kp, kd=0.0 if torque_control else kd,
                           armature=armature, torque_control=torque_control)
    cam_q = " ".join(f"{v:.6f}" for v in (_camera_quat() if camera_quaternion is None else camera_quaternion))

    assets = (
        '<texture type="2d" name="groundplane" builtin="checker" mark="edge" '
        'rgb1="1 1 1" rgb2="0.72 0.72 0.72" markrgb="0.25 0.25 0.25" '
        'width="200" height="200"/>\n'
        '<material name="groundplane" texture="groundplane" texuniform="true" '
        'texrepeat="5 5" reflectance="0.0"/>\n'
    )
    geoms = ""
    if scene == "flat":
        geoms = '<geom name="ground" type="plane" size="20 20 0.1" material="groundplane"/>\n'
    elif scene == "stairs":
        # ascending staircase, flat top, descending (up-over-down). A full ground
        # plane under everything so the robot starts on level ground and lands on
        # level ground after the descent; the stair boxes sit on top of the plane.
        geoms = '<geom name="ground" type="plane" size="20 20 0.1" material="groundplane"/>\n'
        boxes = []
        n_up = n_steps
        x = step_tread / 2.0
        for s in range(1, n_up + 1):
            z = s * step_rise
            boxes.append(f'<geom type="box" pos="{x} 0 {z / 2.0}" size="{step_tread / 2.0} {width / 2.0} {z / 2.0}" '
                         f'material="groundplane"/>\n')
            x += step_tread
        # flat top
        top_z = n_up * step_rise
        boxes.append(f'<geom type="box" pos="{x + step_tread / 2.0} 0 {top_z / 2.0}" '
                     f'size="{step_tread} {width / 2.0} {top_z / 2.0}" material="groundplane"/>\n')
        x += step_tread
        for s in range(n_up, 0, -1):
            z = s * step_rise
            boxes.append(f'<geom type="box" pos="{x + step_tread / 2.0} 0 {z / 2.0}" '
                         f'size="{step_tread / 2.0} {width / 2.0} {z / 2.0}" material="groundplane"/>\n')
            x += step_tread
        geoms += "".join(boxes)   # keep the ground plane, append the stair boxes
    elif scene == "perlin":
        nrow, ncol = 100, 100
        png = os.path.join(HERE, "assets", "go1_perlin.png")
        os.makedirs(os.path.dirname(png), exist_ok=True)
        Hf = _fbm_values((nrow, ncol), amplitudes=(0.25, 0.25, 0.2, 0.18, 0.12))
        # flat approach zone in front of the robot (x<4m) so it starts on level ground
        xf = int(round(4.0 / 10.0 * ncol))
        Hf[:, :xf] = 0.5
        cv2.imwrite(png, (Hf * 255.0).astype(np.uint8))
        assets += f'<hfield name="perlin" size="5 10 {perlin_amp} 0.001" nrow="{nrow}" ncol="{ncol}" file="{png}"/>\n'
        geoms = '<geom type="hfield" hfield="perlin" pos="9 0 0" size="1 1 1" material="groundplane"/>\n'
        # flat ground under the robot + safety floor
        geoms += '<geom name="flat_approach" type="plane" pos="0 0 0" size="8 20 0.05" material="groundplane"/>\n'
        geoms += '<geom name="safety_floor" type="plane" size="40 40 0.05" pos="0 0 -2.0"/>\n'
    elif scene == "course":
        # flat start -> stairs up/over/down -> perlin rough terrain, all along +x.
        geoms = '<geom name="ground" type="plane" size="20 20 0.1" material="groundplane"/>\n'
        boxes = []
        x = step_tread / 2.0
        for s in range(1, n_steps + 1):
            z = s * step_rise
            boxes.append(f'<geom type="box" pos="{x} 0 {z / 2.0}" size="{step_tread / 2.0} {width / 2.0} {z / 2.0}" '
                         f'material="groundplane"/>\n')
            x += step_tread
        top_z = n_steps * step_rise
        boxes.append(f'<geom type="box" pos="{x + step_tread / 2.0} 0 {top_z / 2.0}" '
                     f'size="{step_tread} {width / 2.0} {top_z / 2.0}" material="groundplane"/>\n')
        x += step_tread
        for s in range(n_steps, 0, -1):
            z = s * step_rise
            boxes.append(f'<geom type="box" pos="{x + step_tread / 2.0} 0 {z / 2.0}" '
                         f'size="{step_tread / 2.0} {width / 2.0} {z / 2.0}" material="groundplane"/>\n')
            x += step_tread
        geoms += "".join(boxes)
        # perlin rough terrain after the stairs, entry edge flush with the ground
        nrow, ncol = 100, 100
        png = os.path.join(HERE, "assets", "go1_course_perlin.png")
        os.makedirs(os.path.dirname(png), exist_ok=True)
        Hf = _fbm_values((nrow, ncol), amplitudes=(0.25, 0.25, 0.2, 0.18, 0.12))
        Hf[:, 0] = 0.0   # hfield starts at height 0 where the robot steps off the plane
        cv2.imwrite(png, (Hf * 255.0).astype(np.uint8))
        perlin_x_half = 3.5
        perlin_x_center = x + 1.0 + perlin_x_half   # small flat gap after the stairs
        assets += (f'<hfield name="perlin" size="{perlin_x_half} 8 {perlin_amp} 0.001" '
                   f'nrow="{nrow}" ncol="{ncol}" file="{png}"/>\n')
        geoms += f'<geom type="hfield" hfield="perlin" pos="{perlin_x_center} 0 0" size="1 1 1" material="groundplane"/>\n'
        geoms += '<geom name="safety_floor" type="plane" size="40 40 0.05" pos="0 0 -2.0"/>\n'
    else:
        raise ValueError(f"Unknown scene '{scene}' (flat|stairs|perlin|course)")

    # bake the depth camera into the XML as the FIRST child of the trunk body so its
    # pose (0.26, 0, 0.12) is in the trunk frame and it tracks the robot (a camera at
    # worldbody level would not follow the trunk).
    h, w = camera_size
    if camera_intrinsics is None:
        camera_projection = f'fovy="{CAM_FOVY}"'
    else:
        K = np.asarray(camera_intrinsics, dtype=float)
        if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError('Invalid camera intrinsics')
        if not np.allclose(K[[0, 1], [1, 0]], 0) or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError('Only zero-skew pinhole camera intrinsics are supported')
        camera_projection = (f'sensorsize="{w} {h}" resolution="{w} {h}" '
                             f'focal="{K[0, 0]} {K[1, 1]}" '
                             f'principal="{w / 2 - K[0, 2]} {h / 2 - K[1, 2]}"')
    camera = f'<camera name="depth_cam" pos="{camera_position[0]} {camera_position[1]} {camera_position[2]}" quat="{cam_q}" {camera_projection}/>\n'
    inner = inner.replace("<worldbody>", "<worldbody>\n" + geoms)
    inner = inner.replace("<asset>", "<asset>\n" + assets)
    trunk_open = '<body name="trunk"'
    idx = inner.find(trunk_open)
    assert idx >= 0, "trunk body not found in go1.xml"
    gt = inner.find(">", idx) + 1
    inner = inner[:gt] + "\n" + camera + inner[gt:]
    return f'<mujoco model="go1_{scene}">{inner}</mujoco>'


# --- obs / control helpers -------------------------------------------------

def _projected_gravity(quat_wxyz):
    """Body-frame projected gravity unit vector (matches IsaacLab quat_rotate_inverse)."""
    q = torch.tensor(quat_wxyz, dtype=torch.float32).view(1, 4)
    v = torch.tensor((0.0, 0.0, -1.0), dtype=torch.float32).view(1, 3)
    q_w, q_vec = q[..., 0], q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(1, 1, 3), v.view(1, 3, 1)).squeeze(-1) * 2.0
    return (a - b + c).numpy().flatten()


def _make_key_callback(ctl, command_a, fsm=False):
    """Keyboard commands bounded by the student's training amplitudes.

    This intentionally follows the original sim2sim mapping.  W and R are not
    used because MuJoCo assigns them to wireframe and reload.  Left/Right also
    reach the viewer's history-step binding, but that is inert in passive mode.
    """
    vx_max, vy_max, wz_max = command_a
    def key_callback(keycode):
        if fsm and keycode in (256, 32):         # ESC/Space -> DAMPING
            ctl["request"] = DAMPING
            ctl["rl_enabled"] = False
        elif keycode == 256:                     # ESC -> quit outside FSM
            ctl["quit"] = True
        elif fsm and keycode == 88:              # X -> quit
            ctl["quit"] = True
        elif fsm and keycode == 48:              # 0 -> PASSIVE
            ctl["request"] = PASSIVE
            ctl["rl_enabled"] = False
        elif fsm and keycode == 49:              # 1 -> STAND
            ctl["request"] = STAND
            ctl["rl_enabled"] = False
        elif fsm and keycode == 50:              # 2 -> SITDOWN
            ctl["request"] = SITDOWN
            ctl["rl_enabled"] = False
        elif fsm and keycode == 51:              # 3 -> RL
            ctl["request"] = RL
            ctl["rl_enabled"] = True
        elif keycode == 265:                     # Up -> vx +0.1
            ctl["vx"] = min(ctl["vx"] + 0.1, vx_max)
        elif keycode == 264:                     # Down -> vx -0.1
            ctl["vx"] = max(ctl["vx"] - 0.1, -vx_max)
        elif keycode == 263:                     # Left -> vy +0.1
            ctl["vy"] = min(ctl["vy"] + 0.1, vy_max)
        elif keycode == 262:                     # Right -> vy -0.1
            ctl["vy"] = max(ctl["vy"] - 0.1, -vy_max)
        elif keycode == 81:                      # Q -> wz +0.1
            ctl["wz"] = min(ctl["wz"] + 0.1, wz_max)
        elif keycode == 69:                      # E -> wz -0.1
            ctl["wz"] = max(ctl["wz"] - 0.1, -wz_max)
        elif keycode == 86:                      # V -> zero command
            ctl["vx"] = ctl["vy"] = ctl["wz"] = 0.0
        elif keycode == 67:                      # C -> cruise
            ctl["vx"], ctl["vy"], ctl["wz"] = min(0.4, vx_max), 0.0, 0.0
        elif keycode == 66:                      # B -> camera failure / recovery
            ctl["sensor_blind"] = not ctl.get("sensor_blind", False)
            print(f"[SENSOR] depth input {'ZERO' if ctl['sensor_blind'] else 'RESTORED'}", flush=True)
        elif keycode == 70:                      # F -> freeze/unfreeze processed depth
            ctl["depth_frozen"] = not ctl.get("depth_frozen", False)
            ctl["frozen_depth"] = None
            print(f"[SENSOR] depth input {'FROZEN' if ctl['depth_frozen'] else 'LIVE'}", flush=True)
        elif keycode == 78:                      # N -> proprio noise on/off
            ctl["proprio_noise"] = not ctl.get("proprio_noise", False)
            print(f"[SENSOR] proprio noise {'ON' if ctl['proprio_noise'] else 'OFF'}", flush=True)
    return key_callback


# --- depth preprocessing (mirrors training _sanitize_depth_data) ------------

def _preprocess_depth(depth, blur_sigma, device, *, min_z=0.0,
                      additive_noise_std=0.0, dropout_prob=0.0):
    d = np.array(depth, dtype=np.float32, copy=True)
    d[d < 0.01] = 0.0
    t = torch.from_numpy(d)[None, None].to(device)
    return preprocess_depth(t, blur_sigma=blur_sigma, min_z=min_z,
                            additive_noise_std=additive_noise_std, dropout_prob=dropout_prob)


def _load_fixed_depth(path, expected_shape):
    """Load an actor-ready metric depth frame without applying preprocessing again."""
    depth = np.load(os.path.abspath(os.path.expanduser(path)), allow_pickle=False)
    if depth.shape != tuple(expected_shape):
        raise ValueError(f"Fixed depth must have shape {tuple(expected_shape)}, got {depth.shape}")
    if not np.issubdtype(depth.dtype, np.number) or not np.isfinite(depth).all():
        raise ValueError("Fixed depth must contain only finite numeric values")
    depth = np.asarray(depth, dtype=np.float32)
    if depth.min() < 0.1 or depth.max() > 2.0:
        raise ValueError("Fixed actor-ready depth must be within [0.1, 2.0] metres")
    return depth


def _close_cv_windows():
    """Close HighGUI windows and pump one event so GTK removes them promptly."""
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except cv2.error:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run the GO1 DAgger depth student in MuJoCo (sim2sim).")
    parser.add_argument("--ckpt", required=True,
                        help="path to dagger_policy.pt (Tiled or RayCaster student checkpoint)")
    parser.add_argument("--policy_steps", type=int, default=1000,
                        help="number of policy steps to run (default: 1000, -1 = until viewer closed)")
    parser.add_argument("--scene", default="flat",
                        help="terrain: flat|stairs|perlin|course (default: flat; course = flat start -> "
                             "stairs up/over/down -> perlin rough)")
    parser.add_argument("--step_rise", type=float, default=None,
                        help="arbitrary stair step height in metres (default 0.12)")
    parser.add_argument("--stair_height_cm", type=int, choices=(6, 8, 10, 12, 15, 18),
                        default=None, help="convenient stair-height preset in cm")
    parser.add_argument("--step_tread", type=float, default=0.30, help="stair step depth (m, default 0.30 = training step_width)")
    parser.add_argument("--n_steps", type=int, default=12, help="stair steps up (then down)")
    parser.add_argument("--stair_width", type=float, default=6.0, help="stair width (m, default 6.0)")
    parser.add_argument("--kp", type=float, default=None, help="Override training stiffness.")
    parser.add_argument("--kd", type=float, default=None, help="Override training damping.")
    parser.add_argument("--no_act_delay", action="store_true",
                        help="disable the actuator command delay (training uses DelayedPDActuator "
                             "0-2 physics steps; use for an ablation)")
    parser.add_argument("--act_delay", type=int, choices=range(GO1_MIN_DELAY, GO1_MAX_DELAY + 1),
                        default=None, help="fix actuator delay in physics steps for reproducible tests")
    parser.add_argument("--perlin_amp", type=float, default=0.18, help="perlin amplitude (m)")
    parser.add_argument("--viewer", action="store_true",
                        help="open an interactive mujoco viewer + live depth window (needs a display). "
                             "Keys: Up/Down=vx Left/Right=vy Q/E=wz (+-0.1/press), "
                             "V=zero C=cruise B=toggle zero depth F=freeze depth ESC=quit")
    parser.add_argument("--cmd", default="0.0 0.0 0.0",
                        help="fixed velocity command 'vx vy wz' for headless runs (default 0 0 0); "
                             "with --viewer the keyboard overrides it")
    parser.add_argument("--proprio_noise", action="store_true",
                        help="enable the same zero-mean per-step proprioceptive uniform noise "
                             "used by noisy DAgger training; press N in the viewer to toggle")
    parser.add_argument("--proprio_noise_seed", type=int, default=0,
                        help="PyTorch RNG seed for reproducible proprioceptive noise (default: 0)")
    parser.add_argument("--proprio_angular_velocity_amplitude", type=float, default=0.05)
    parser.add_argument("--proprio_projected_gravity_amplitude", type=float, default=0.02)
    parser.add_argument("--proprio_joint_position_amplitude", type=float, default=0.01)
    parser.add_argument("--proprio_joint_velocity_amplitude", type=float, default=0.05)
    parser.add_argument("--no_depth_noise", action="store_true",
                        help="disable stochastic depth additive/dropout noise while preserving "
                             "checkpoint blur, clipping, sampling and delay")
    parser.add_argument("--fixed_depth_file",
                        help="actor-ready HxW float .npy depth in metres; replaces the MuJoCo "
                             "camera for the entire run without applying blur/noise again")
    parser.add_argument("--no_video", action="store_true", help="skip dual-pane mp4 recording")
    parser.add_argument("--video_stride", type=int, default=1,
                        help="record every Nth policy frame without changing simulation timing")
    parser.add_argument("--trace_output",
                        help="optional NPZ policy-step trace for sim/real diagnostics")
    parser.add_argument("--fsm", action="store_true",
                        help="run PASSIVE/DAMPING/STAND/SITDOWN/RL through the deployment FSM")
    parser.add_argument("--fsm_config", default=os.path.join(HERE, "fsm_sim_config.json"),
                        help="FSM pose/gain configuration (simulation only)")
    parser.add_argument("--rl_entry_seconds", type=float, default=None,
                        help="override FSM STAND->RL target blend duration for ablation")
    parser.add_argument("--fsm_scripted", action="store_true",
                        help="headless regression: STAND->RL->SITDOWN->DAMPING->PASSIVE")
    parser.add_argument("--fsm_script", default=None,
                        help="custom headless events, e.g. '0:STAND,4:RL,25:SITDOWN'")
    args = parser.parse_args()
    if args.step_rise is not None and args.stair_height_cm is not None:
        parser.error("choose --step_rise or --stair_height_cm, not both")
    if args.step_rise is None:
        args.step_rise = (args.stair_height_cm / 100.0
                          if args.stair_height_cm is not None else 0.12)
    if not np.isfinite(args.step_rise) or args.step_rise <= 0:
        parser.error("--step_rise must be a positive finite value")
    if args.video_stride < 1:
        parser.error("--video_stride must be >= 1")
    proprio_amplitudes = (
        args.proprio_angular_velocity_amplitude,
        args.proprio_projected_gravity_amplitude,
        args.proprio_joint_position_amplitude,
        args.proprio_joint_velocity_amplitude,
    )
    if any(not np.isfinite(value) or value < 0 for value in proprio_amplitudes):
        parser.error("proprioceptive noise amplitudes must be finite and nonnegative")
    if args.no_act_delay and args.act_delay is not None:
        parser.error("choose --no_act_delay or --act_delay")
    if args.rl_entry_seconds is not None and (
            not np.isfinite(args.rl_entry_seconds) or args.rl_entry_seconds < 0):
        parser.error("--rl_entry_seconds must be nonnegative and finite")
    if (args.fsm_scripted or args.fsm_script) and (not args.fsm or args.viewer):
        parser.error("FSM scripts require --fsm and no --viewer")
    if args.fsm_scripted and args.fsm_script:
        parser.error("choose --fsm_scripted or --fsm_script")
    try:
        _vx, _vy, _wz = (float(x) for x in args.cmd.split())
    except ValueError:
        raise SystemExit("--cmd must be 'vx vy wz' (three floats)")
    if args.scene in ("stairs", "course"):
        print(f"[INFO] staircase: rise={args.step_rise:.3f} m, "
              f"tread={args.step_tread:.3f} m, steps={args.n_steps} up/down.")

    ckpt_path = os.path.abspath(args.ckpt)
    assert os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"
    run_dir = os.path.dirname(ckpt_path)
    fsm_cfg = None
    if args.fsm:
        verify_bundle(os.path.join(HERE, "deployment_manifest.json"), args.fsm_config)
        with open(args.fsm_config) as source:
            fsm_cfg = json.load(source)
        required = ("stand_pose_sdk", "initial_pose_sdk", "pose_kp", "pose_kd",
                    "rl_kp", "rl_kd", "damping_kd", "stand_seconds",
                    "sit_seconds", "pose_tolerance", "rl_entry_seconds")
        missing = [name for name in required if name not in fsm_cfg]
        if missing:
            raise ValueError(f"FSM config missing: {missing}")

    # --- training config: read params/env.yaml (fallback to GO1 defaults) ---
    tcfg = GO1_DEFAULTS
    env_yaml = os.path.join(run_dir, "params", "env.yaml")
    if os.path.exists(env_yaml):
        import yaml
        with open(env_yaml) as source:
            y = yaml.unsafe_load(source)
        tcfg = dict(
            action_scale=y.get("action_scale", GO1_DEFAULTS["action_scale"]),
            clip_actions=y.get("desired_clip_actions", GO1_DEFAULTS["clip_actions"]),
            use_filter=y.get("use_filter_actions", GO1_DEFAULTS["use_filter"]),
            step_freq=y.get("desired_step_freq", GO1_DEFAULTS["step_freq"]),
            sim_dt=y["sim"]["dt"] if "sim" in y else GO1_DEFAULTS["sim_dt"],
            decimation=y.get("decimation", GO1_DEFAULTS["decimation"]),
            history_length=y.get("history_length", GO1_DEFAULTS["history_length"]),
            kp=GO1_DEFAULTS['kp'], kd=GO1_DEFAULTS['kd'], armature=GO1_DEFAULTS['armature'],
        )
        actuators = y.get('robot', {}).get('actuators', {})
        gains = [tuple(a.get(key, GO1_DEFAULTS[default]) for key, default in
                       [('stiffness', 'kp'), ('damping', 'kd'), ('armature', 'armature')])
                 for a in actuators.values()]
        if gains:
            if any(g != gains[0] for g in gains):
                raise ValueError('Per-joint PD gains require explicit sim2sim mapping; uniform gains expected')
            tcfg['kp'], tcfg['kd'], tcfg['armature'] = gains[0]
    else:
        print('[WARN] params/env.yaml missing; using nominal GO1 dynamics, not verified run parameters.')
    rl_freq = 1.0 / (tcfg["sim_dt"] * tcfg["decimation"])   # 50 Hz

    # --- load student policy ---
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    md = ckpt["metadata"]
    command_a = np.asarray(md.get('command_a', [0.8, 0.4, 0.8]), dtype=float)
    if command_a.shape != (3,) or not np.isfinite(command_a).all() or (command_a < 0).any():
        raise ValueError('command_a must contain three finite nonnegative amplitudes')
    if 'command_a' not in md:
        print('[WARN] command_a missing; keyboard uses nominal student limits [0.8, 0.4, 0.8].')
    print(f'[INFO] keyboard command limits: +/-{command_a.tolist()} (vx, vy, wz).')
    control_period = float(md.get('depth_camera', {}).get('control_period', 1.0 / rl_freq))
    if not np.isclose(control_period, 1.0 / rl_freq):
        raise ValueError('Checkpoint control period differs from params/env.yaml')
    if md['common_obs_size'] % tcfg['history_length']:
        raise ValueError('Observation history length does not divide common_obs_size')
    obs_frame_size = md['common_obs_size'] // tcfg['history_length']
    if obs_frame_size not in (45, 49) or md['action_size'] != 12 or md['depth_channels'] != 1:
        raise ValueError('Expected GO1 student: 45/49-dim frames, 12 actions, one depth channel')
    use_clock = obs_frame_size == 49
    if use_clock and args.proprio_noise:
        parser.error("--proprio_noise supports the current 45-dim observation frame, not legacy clock models")
    depth_h, depth_w = map(int, md.get('depth_image_size', (DEPTH_H, DEPTH_W)))
    fixed_depth_t = None
    if args.fixed_depth_file:
        fixed_depth = _load_fixed_depth(args.fixed_depth_file, (depth_h, depth_w))
        fixed_depth_t = torch.from_numpy(fixed_depth)[None, None]
        print(f"[SENSOR] fixed actor-ready depth={os.path.abspath(os.path.expanduser(args.fixed_depth_file))} "
              f"range=[{fixed_depth.min():.3f}, {fixed_depth.max():.3f}] m")
    net = DaggerNet(vec_size=md["common_obs_size"], output_size=md["action_size"],
                    depth_channels=md["depth_channels"]).to("cpu")
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    depth_len = int(md["depth_history_length"])
    depth_fps = float(md.get("depth_fps", rl_freq))
    if md.get("depth_delay_units") == "camera_frames":
        print("[WARN] This student used camera-frame history; current sim2sim uses control-step history.")
    blur_sigma = float(md.get("depth_sensor_noise", {}).get("blur_sigma", 0.0))
    delay = int(md.get("depth_delay_frames", 0))
    print(f"[INFO] rl_freq={rl_freq:.1f} obs={md['common_obs_size']} depth_len={depth_len} "
          f"depth_fps={depth_fps} blur_sigma={blur_sigma} delay={delay} history step(s)")
    if args.fsm and (use_clock or md["common_obs_size"] != 225 or
                     depth_len != 5 or delay != 1 or (depth_h, depth_w) != (60, 106)):
        raise ValueError("Shared deployment core requires the current student format: "
                         "common=225, depth=5x60x106, delay=1, no clock")
    if args.fsm and not np.allclose(command_a, COMMAND_LIMIT):
        raise ValueError("Checkpoint command limits differ from the shared deployment core")
    if args.fsm:
        print(f"[CORE] go1_deploy_core.py sha256={core_sha256()}")

    # Use saved optical pose, not the mounting screw origin.
    camera_meta = md.get("depth_camera", {})
    camera_position = camera_meta.get("position_base", D435_DEPTH_POSITION_BASE)
    camera_quaternion = None
    if "quaternion_wxyz" in camera_meta:
        if camera_meta.get("convention") != "ros":
            raise ValueError("Saved camera pose must use ROS optical convention.")
        w, x, y, z = camera_meta["quaternion_wxyz"]
        # ROS -> OpenGL camera axes: postmultiply by Rx(pi).
        camera_quaternion = (-x, w, z, -y)
    else:
        w, x, y, z = D435_DEPTH_QUATERNION_ROS_WXYZ
        camera_quaternion = (-x, w, z, -y)
        print('[WARN] Camera extrinsics missing; using current nominal D435 optical pose.')
    camera_intrinsics = camera_meta.get('intrinsics')
    if camera_intrinsics is None:
        f = depth_w * 24.0 / 45.55
        camera_intrinsics = [[f, 0, depth_w / 2], [0, f, depth_h / 2], [0, 0, 1]]
        print('[WARN] Camera intrinsics missing; using nominal D435 pinhole intrinsics.')
    if camera_meta.get('depth_type', 'distance_to_image_plane') != 'distance_to_image_plane':
        raise ValueError('Only optical Z-depth checkpoints are supported')
    # --- build + load the scene (go1 + terrain + depth cam) ---
    xml = build_go1_scene_xml(scene=args.scene, step_rise=args.step_rise, step_tread=args.step_tread,
                              n_steps=args.n_steps, width=args.stair_width, perlin_amp=args.perlin_amp,
                              kp=tcfg['kp'] if args.kp is None else args.kp,
                              kd=tcfg['kd'] if args.kd is None else args.kd,
                              armature=tcfg['armature'], camera_position=camera_position,
                              camera_quaternion=camera_quaternion,
                              camera_intrinsics=camera_intrinsics, camera_size=(depth_h, depth_w))
    if args.fsm:
        xml = build_go1_scene_xml(
            scene=args.scene, step_rise=args.step_rise, step_tread=args.step_tread,
            n_steps=args.n_steps, width=args.stair_width, perlin_amp=args.perlin_amp,
            kp=0.0, kd=0.0, armature=tcfg['armature'],
            camera_position=camera_position, camera_quaternion=camera_quaternion,
            camera_intrinsics=camera_intrinsics, camera_size=(depth_h, depth_w),
            torque_control=True)
    model = mujoco.MjModel.from_xml_string(xml)
    # Use the training run's physics timestep, not only its control frequency.
    model.opt.timestep = tcfg['sim_dt']
    data = mujoco.MjData(model)
    depth_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")
    assert depth_cam_id >= 0, "depth_cam camera missing from scene XML"
    cam = MujocoDepthCamera(model, data, depth_cam_id, width=depth_w, height=depth_h,
                            update_period=1.0 / depth_fps, intrinsics=camera_intrinsics,
                            ray_max_distance=3.0 if 'RayCaster' in md.get('task', '') else None)
    print(f'[INFO] obs_frame={obs_frame_size}, clock={use_clock}, depth={depth_w}x{depth_h}, K={camera_intrinsics}')

    # --- actuator -> policy-joint mapping (per-leg actuators vs hip/thigh/calf blocks) ---
    # go1.xml actuator names are "{joint}" (e.g. "FR_hip" controls "FR_hip_joint").
    joint_of_actuator = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) + "_joint"
                         for i in range(model.nu)]
    policy_idx = {nm: i for i, nm in enumerate(DESIRED_ORDER)}
    # ctrl[actuator_i] gets target_pos[policy_idx[joint_of_actuator[i]]]

    # --- init + settle: spawn feet clear of the terrain and let the robot drop and
    # rest in the home pose before the policy takes over. Starting embedded in the
    # terrain (feet are below ground at trunk 0.3) made the robot tip over the stair
    # edge and fall through on the stairs scene. For stairs/course, spawn 1 m back on
    # the flat approach so the robot walks onto the first step from level ground.
    spawn_x = -1.0 if args.scene in ("stairs", "course") else 0.0
    data.qpos[:3] = [spawn_x, 0.0, 0.18 if args.fsm else 0.4]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    initial_policy_pose = DEFAULT_JOINT
    if args.fsm:
        initial_sdk = sdk_position_limit(fsm_cfg["initial_pose_sdk"])
        initial_by_name = dict(zip(SDK_ORDER, initial_sdk))
        initial_policy_pose = np.asarray([initial_by_name[name] for name in DESIRED_ORDER])
    for nm, val in zip(DESIRED_ORDER, initial_policy_pose):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = val
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if not args.fsm:
        for _ in range(int(3.0 / model.opt.timestep)):     # 3 s hold home pose
            for i, jn in enumerate(joint_of_actuator):
                if jn in policy_idx:
                    data.ctrl[i] = DEFAULT_JOINT[policy_idx[jn]]
            mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    print(f"[INFO] settled trunk z={data.xpos[model.body('trunk').id, 2]:.3f} (dt={model.opt.timestep})")

    # --- state buffers (newest at END, matching training) ---
    S = obs_frame_size
    obs_hist = np.zeros((tcfg["history_length"], S), dtype=np.float32)
    hist = np.zeros((depth_len + delay, 1, depth_h, depth_w), dtype=np.float16)
    phase = np.array([0.0, 0.5, 0.5, 0.0])                   # FL, FR, RL, RR
    past_actions = np.zeros(model.nu, dtype=np.float32)
    ctl = {"vx": _vx, "vy": _vy, "wz": _wz,
           "proprio_noise": bool(args.proprio_noise),
           "sensor_blind": False, "depth_frozen": False, "frozen_depth": None,
           "request": None, "rl_enabled": False}
    proprio_noise = None
    if not use_clock:
        torch.manual_seed(args.proprio_noise_seed)
        proprio_noise_cfg = ProprioNoiseCfg(
            angular_velocity_half_range=args.proprio_angular_velocity_amplitude,
            projected_gravity_half_range=args.proprio_projected_gravity_amplitude,
            joint_position_half_range=args.proprio_joint_position_amplitude,
            joint_velocity_half_range=args.proprio_joint_velocity_amplitude,
        )
        proprio_noise = ProprioceptiveNoise(
            num_envs=1, history_length=tcfg["history_length"], device="cpu",
            cfg=proprio_noise_cfg)
        print(f"[INFO] proprio noise={'ON' if args.proprio_noise else 'OFF'} "
              f"seed={args.proprio_noise_seed} uniform_half_range={proprio_noise_cfg.to_dict()}")
    if args.viewer:
        initial_cmd = np.array([_vx, _vy, _wz])
        if not np.isfinite(initial_cmd).all():
            raise ValueError('Viewer initial command must be finite')
        bounded_cmd = np.clip(initial_cmd, -command_a, command_a)
        if not np.array_equal(initial_cmd, bounded_cmd):
            print(f'[WARN] Viewer initial --cmd clipped to {bounded_cmd.tolist()}.')
        ctl.update(zip(('vx', 'vy', 'wz'), bounded_cmd))
    last_cmd = None

    sim_dt = model.opt.timestep
    policy_every = max(1, round(1.0 / (rl_freq * sim_dt)))
    base_id = model.body("trunk").id
    N_policy = args.policy_steps if args.policy_steps >= 0 else int(1e9)
    base_contact = 0
    device = "cpu"

    # --- actuator command delay (DelayedPDActuator min_delay..max_delay physics steps) ---
    # sampled once like the per-env randomization in training; --no_act_delay disables it
    act_delay = (args.act_delay if args.act_delay is not None else
                 0 if args.no_act_delay else
                 int(np.random.randint(GO1_MIN_DELAY, GO1_MAX_DELAY + 1)))
    delay_line = np.tile(DEFAULT_JOINT, (GO1_MAX_DELAY + 1, 1)).astype(np.float32)
    print(f"[INFO] actuator delay={act_delay} phys-steps, policy_every={policy_every} @ dt={sim_dt}")

    machine = None
    deploy_core = None
    if args.fsm:
        ones = np.ones(12, dtype=np.float32)
        machine = StateMachine(
            fsm_cfg["stand_pose_sdk"], ones * float(fsm_cfg["pose_kp"]),
            ones * float(fsm_cfg["pose_kd"]), ones * float(fsm_cfg["damping_kd"]),
            rl_kp=ones * float(fsm_cfg["rl_kp"]),
            rl_kd=ones * float(fsm_cfg["rl_kd"]),
            stand_seconds=float(fsm_cfg["stand_seconds"]),
            sit_seconds=float(fsm_cfg["sit_seconds"]),
            pose_tolerance=float(fsm_cfg["pose_tolerance"]),
            position_limit=sdk_position_limit,
            rl_entry_seconds=(float(args.rl_entry_seconds)
                              if args.rl_entry_seconds is not None
                              else float(fsm_cfg["rl_entry_seconds"])))
        deploy_core = DeploymentCore(machine)
        print("[INFO] FSM gains: pose Kp/Kd={}/{}; RL Kp/Kd={}/{}; damping Kd={}.".format(
            fsm_cfg["pose_kp"], fsm_cfg["pose_kd"], fsm_cfg["rl_kp"],
            fsm_cfg["rl_kd"], fsm_cfg["damping_kd"]))
        print("[INFO] Starts PASSIVE at the configured crouch pose; press 1 promptly to test STAND.")

    # --- optional viewer ---
    viewer = None
    if args.viewer:
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("[ERROR] --viewer needs a display; run headless without it.")
            sys.exit(1)
        ctl["quit"] = False
        viewer = mujoco.viewer.launch_passive(
            model, data, key_callback=_make_key_callback(ctl, command_a, args.fsm))
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 270.0
        viewer.cam.elevation = -18.0
        cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("depth", 480, 280)
        if args.fsm:
            print("[INFO] FSM viewer: 0=PASSIVE Space/Esc=DAMPING 1=STAND "
                  "2=SITDOWN 3=RL X=quit; Up/Down=vx Left/Right=vy Q/E=wz "
                  "V=zero C=cruise "
                  "B=toggle zero depth F=freeze depth N=toggle proprio noise.")
        else:
            print("[INFO] viewer on: Up/Down=vx Left/Right=vy Q/E=wz V=zero "
                  "C=cruise B=toggle zero depth F=freeze depth "
                  "N=toggle proprio noise ESC=quit.")

    # --- optional dual-pane video ---
    recording = not args.no_video
    writer = None
    rgb_renderer = None
    vcam = None
    video_path = None
    if recording:
        rgb_renderer = mujoco.Renderer(model, height=240, width=480)
        vcam = mujoco.MjvCamera()
        vcam.distance = 3.2
        vcam.azimuth = 180.0
        vcam.elevation = -18.0
        ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]   # e.g. dagger_policy, dagger_policy_raycaster
        if args.fsm:
            ckpt_tag += "_fsm"
        video_path = os.path.join(run_dir, "videos", "dagger_play", f"mj_sim2sim_go1_{ckpt_tag}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

    completed_steps = 0
    trace_records = []
    # Start after settling and renderer/viewer setup, not at application launch.
    wall_start = time.perf_counter()
    sim_start = data.time
    scripted_events = ([[0.0, STAND], [4.0, RL], [5.5, SITDOWN],
                        [9.0, DAMPING], [10.0, PASSIVE]] if args.fsm_scripted
                       else _parse_fsm_script(args.fsm_script) if args.fsm_script else [])
    fsm_metrics = {}
    if args.viewer:
        print('[INFO] viewer real-time synchronization enabled (target speed=1x).')
    for p in range(N_policy):
        if args.viewer and (not viewer.is_running() or ctl.get("quit")):
            print("[INFO] viewer closed -> stopping early")
            break

        if scripted_events and data.time - sim_start >= scripted_events[0][0]:
            _, ctl["request"] = scripted_events.pop(0)
            ctl["rl_enabled"] = ctl["request"] == RL

        quat = data.qpos[3:7]
        # Free-joint angular qvel is already in the local body frame, matching
        # IsaacLab root_ang_vel_b. Do not rotate it from world to body again.
        ang_vel_b = data.qvel[3:6].copy()
        sdk_q = np.asarray([
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name)]] for name in SDK_ORDER],
            dtype=np.float32)
        sdk_dq = np.asarray([
            data.qvel[model.jnt_dofadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name)]] for name in SDK_ORDER],
            dtype=np.float32)
        jp = np.array([data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)]]
                       for nm in DESIRED_ORDER])
        jv = np.array([data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)]]
                       for nm in DESIRED_ORDER])
        cmd = np.array([ctl["vx"], ctl["vy"], ctl["wz"]], dtype=np.float32)
        if args.viewer and (last_cmd is None or np.any(np.abs(cmd - last_cmd) > 1e-3)):
            print(f"[CMD] vx={ctl['vx']:+.2f} vy={ctl['vy']:+.2f} wz={ctl['wz']:+.2f}", flush=True)
            last_cmd = cmd
        if deploy_core is None:
            # Legacy non-FSM playback remains available for older 49-D students.
            grav_b = _projected_gravity(quat)
            obs1 = np.concatenate([ang_vel_b, grav_b, cmd, jp - DEFAULT_JOINT,
                                   jv, past_actions])
            if use_clock:
                phase = (phase + tcfg["step_freq"] / rl_freq) % 1.0
                obs1 = (np.concatenate([obs1, phase]) if np.linalg.norm(cmd) > 0.01
                        else np.concatenate([obs1, -np.ones(4)]))
            obs_hist = np.roll(obs_hist, -1, axis=0)
            obs_hist[-1] = obs1

        if ctl.get("sensor_blind", False):
            depth_t = torch.zeros((1, 1, depth_h, depth_w), dtype=torch.float32, device=device)
        elif fixed_depth_t is not None:
            depth_t = fixed_depth_t
        elif ctl.get("depth_frozen", False) and ctl.get("frozen_depth") is not None:
            depth_t = ctl["frozen_depth"]
        else:
            depth_t = _preprocess_depth(
                cam.render(), blur_sigma, device, min_z=float(md.get("depth_min_z", 0.0)),
                additive_noise_std=(0.0 if args.no_depth_noise else
                                    float(md.get("depth_sensor_noise", {}).get("additive_noise_std", 0.0))),
                dropout_prob=(0.0 if args.no_depth_noise else
                              float(md.get("depth_sensor_noise", {}).get("dropout_prob", 0.0))),
            )
            if ctl.get("depth_frozen", False):
                # Freeze the complete post-processed actor input, including any
                # configured blur/noise, rather than repeatedly reprocessing a raw frame.
                ctl["frozen_depth"] = depth_t.clone()
        frame = depth_t.squeeze(0).squeeze(0).numpy().astype(np.float32)
        if deploy_core is not None:
            seq, common_array = deploy_core.prepare_policy_inputs(
                ang_vel_b, quat, cmd, sdk_q, sdk_dq, frame)
        else:
            frame16 = frame.astype(np.float16)
            if p == 0:
                hist[:] = frame16
            else:
                hist = np.roll(hist, -1, axis=0)
                hist[-1] = frame16
            if delay > 0:
                seq = hist[-depth_len - delay:-delay]
            else:
                seq = hist[-depth_len:]
            common_array = obs_hist.reshape(-1)

        if proprio_noise is not None:
            noisy_common = proprio_noise.update(
                torch.from_numpy(common_array[None]).float()).numpy()[0]
            if ctl.get("proprio_noise", False):
                common_array = noisy_common

        if args.viewer:
            d_show = np.clip(depth_t.squeeze(0).squeeze(0).numpy(), 0.0, 2.0) / 2.0
            d_gray = (255.0 * (1.0 - d_show)).astype(np.uint8)
            cv2.imshow("depth", cv2.resize(d_gray, (480, 280), interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(1)
            try:
                if cv2.getWindowProperty("depth", cv2.WND_PROP_VISIBLE) < 1:
                    ctl["quit"] = True
            except cv2.error:
                ctl["quit"] = True

        if recording and p % args.video_stride == 0:
            wq, xq, yq, zq = data.qpos[3:7]
            yaw = np.arctan2(2.0 * (wq * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq))
            vcam.azimuth = np.degrees(np.arctan2(-np.cos(yaw), -np.sin(yaw)))
            vcam.elevation = -18.0
            vcam.lookat = data.xpos[base_id]
            rgb_renderer.update_scene(data, camera=vcam)
            rgb = rgb_renderer.render()
            d_show = np.clip(depth_t.squeeze(0).squeeze(0).numpy(), 0.0, 2.0) / 2.0
            d_gray = (255.0 * (1.0 - d_show)).astype(np.uint8)
            d_bgr = cv2.cvtColor(d_gray, cv2.COLOR_GRAY2BGR)
            # preserve aspect and match the rgb pane height (240) so hstack works
            h = rgb.shape[0]
            w = int(round(d_bgr.shape[1] * h / d_bgr.shape[0]))
            d_resized = cv2.resize(d_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
            frame = np.hstack([rgb, d_resized])
            if writer is None:
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         rl_freq / args.video_stride,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)

        seq_array = seq[:, None] if seq.ndim == 3 else seq
        seq = torch.from_numpy(seq_array)[None].float().contiguous()
        if ctl.get("sensor_blind", False):
            seq.zero_()  # immediate actor failure; zero frames also persist in the history
        common = torch.from_numpy(common_array[None]).float()
        should_infer = (deploy_core is None or machine.mode == RL or
                        ctl.get("request") == RL)
        actions = None
        if should_infer:
            with torch.no_grad():
                actions_out, _ = net(seq, common, hidden=None)
            actions = actions_out[0].numpy()

        if deploy_core is None:
            actions = np.clip(actions, -tcfg["clip_actions"], tcfg["clip_actions"])
            temp = 0.8 * actions + 0.2 * past_actions if tcfg["use_filter"] else actions
            target_pos = tcfg["action_scale"] * temp + DEFAULT_JOINT
        else:
            target_pos = None

        fsm_result = None
        if machine is not None:
            request = ctl.get("request")
            ctl["request"] = None
            previous_mode = machine.mode
            fsm_result = deploy_core.step(
                data.time, sdk_q, request=request, network_action=actions,
                state_fresh=True, depth_fresh=True,
                rl_enabled=bool(ctl.get("rl_enabled", False)))
            if machine.mode != previous_mode:
                print(f"[FSM] {previous_mode} -> {machine.mode}", flush=True)
                if machine.mode == STAND:
                    print(f"[POSE] saved SITDOWN={machine.sit_pose.tolist()}", flush=True)
            if request == RL and machine.mode != RL:
                print("[FSM] RL rejected: STAND must be complete first.", flush=True)
        # The observation's previous_action is the action that was actually selected
        # by the RL controller. Predictions made while STAND/SITDOWN are active were
        # never executed and must not leak into the RL history.
        if machine is None:
            past_actions = actions.copy()
        elif fsm_result.mode == RL:
            target_pos = sdk_to_policy(fsm_result.q)

        if args.trace_output and fsm_result is not None:
            target_sdk = np.asarray(fsm_result.q, dtype=np.float32)
            estimated_pd_tau = (np.asarray(fsm_result.kp, dtype=np.float32) *
                                (target_sdk - sdk_q) -
                                np.asarray(fsm_result.kd, dtype=np.float32) * sdk_dq)
            trace_records.append(dict(
                policy_step=np.int32(p), sim_time=np.float32(data.time - sim_start),
                mode=fsm_result.mode, q_sdk=sdk_q.copy(), dq_sdk=sdk_dq.copy(),
                quaternion_wxyz=np.asarray(quat, dtype=np.float32).copy(),
                gyro_base=np.asarray(ang_vel_b, dtype=np.float32).copy(),
                command=cmd.copy(), common=np.asarray(common_array, dtype=np.float32).copy(),
                depth_sequence=seq[0, :, 0].numpy().astype(np.float32, copy=True),
                raw_action=(np.asarray(actions, dtype=np.float32).copy()
                            if actions is not None else np.full(12, np.nan, dtype=np.float32)),
                target_sdk=target_sdk.copy(), target_delta=target_sdk - sdk_q,
                estimated_pd_tau=estimated_pd_tau))

        for _ in range(policy_every):
            # actuator command delay: apply the target from `act_delay` physics steps ago
            # (matches DelayedPDActuator; no soft-limit clamp -- training doesn't clamp targets)
            if fsm_result is None or fsm_result.mode == RL:
                delay_line = np.roll(delay_line, 1, axis=0)
                delay_line[0] = target_pos
            ctrl_target = delay_line[act_delay]
            if fsm_result is None:
                for i, jn in enumerate(joint_of_actuator):
                    if jn in policy_idx:
                        data.ctrl[i] = ctrl_target[policy_idx[jn]]
            else:
                # LowCmd-equivalent explicit joint PD. Motor actuators enforce the
                # Menagerie/URDF effort limits after this computation.
                q_command = fsm_result.q
                if fsm_result.mode == RL:
                    delayed_by_name = dict(zip(DESIRED_ORDER, ctrl_target))
                    q_command = np.asarray([delayed_by_name[name] for name in SDK_ORDER])
                q_by_name = dict(zip(SDK_ORDER, q_command))
                kp_by_name = dict(zip(SDK_ORDER, fsm_result.kp))
                kd_by_name = dict(zip(SDK_ORDER, fsm_result.kd))
                for i, jn in enumerate(joint_of_actuator):
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                    q_now = data.qpos[model.jnt_qposadr[jid]]
                    dq_now = data.qvel[model.jnt_dofadr[jid]]
                    data.ctrl[i] = (kp_by_name[jn] * (q_by_name[jn] - q_now)
                                    - kd_by_name[jn] * dq_now)
            mujoco.mj_step(model, data)
        # mj_step integrates qpos after computing kinematics. Refresh poses so
        # camera/gravity observations correspond to the same state as qpos/qvel.
        mujoco.mj_forward(model, data)
        completed_steps += 1

        if args.viewer:
            viewer.cam.lookat[:] = data.xpos[base_id]
            viewer.sync()

        trunk_contact = any(base_id in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
                            for c in data.contact)
        if trunk_contact:
            base_contact += 1
        if machine:
            roll, pitch = _roll_pitch(data.qpos[3:7])
            metric = fsm_metrics.setdefault(machine.mode, dict(
                steps=0, trunk_contact_steps=0, fall_steps=0, min_z=float('inf'),
                max_abs_roll=0.0, max_abs_pitch=0.0, max_dq=0.0, max_q_error=0.0,
                velocity_limit_exceed_steps=0, velocity_limit_exceed_samples=0))
            metric['steps'] += 1
            metric['trunk_contact_steps'] += int(trunk_contact)
            stable_mode = machine.mode == RL or (machine.mode == STAND and machine.pose_complete)
            metric['fall_steps'] += int(stable_mode and
                                        (data.qpos[2] < 0.12 or abs(roll) > 0.8 or abs(pitch) > 0.8))
            metric['min_z'] = min(metric['min_z'], float(data.qpos[2]))
            metric['max_abs_roll'] = max(metric['max_abs_roll'], abs(float(roll)))
            metric['max_abs_pitch'] = max(metric['max_abs_pitch'], abs(float(pitch)))
            dq_abs = np.abs(np.asarray([
                data.qvel[model.jnt_dofadr[mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, name)]] for name in SDK_ORDER]))
            dq_exceeded = dq_abs > GO1_VELOCITY_LIMIT_SDK
            metric['velocity_limit_exceed_steps'] += int(np.any(dq_exceeded))
            metric['velocity_limit_exceed_samples'] += int(np.count_nonzero(dq_exceeded))
            dq_index = int(np.argmax(dq_abs))
            if float(dq_abs[dq_index]) > metric['max_dq']:
                metric['max_dq'] = float(dq_abs[dq_index])
                metric['max_dq_time'] = float(data.time - sim_start)
                metric['max_dq_joint'] = SDK_ORDER[dq_index]
            if fsm_result.mode in (STAND, SITDOWN, RL):
                measured_sdk = np.asarray([
                    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, name)]] for name in SDK_ORDER])
                q_error = np.abs(fsm_result.q - measured_sdk)
                error_index = int(np.argmax(q_error))
                if float(q_error[error_index]) > metric['max_q_error']:
                    metric['max_q_error'] = float(q_error[error_index])
                    metric['max_q_error_time'] = float(data.time - sim_start)
                    metric['max_q_error_joint'] = SDK_ORDER[error_index]
        report_every = 50 if machine else 200
        if p % report_every == 0:
            mode_text = f" mode={machine.mode} reached={machine.pose_complete}" if machine else ""
            if fsm_result is not None and fsm_result.mode in (STAND, SITDOWN, RL):
                measured_sdk = np.asarray([
                    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, name)]] for name in SDK_ORDER])
                mode_text += (f" max_q_error={np.max(np.abs(fsm_result.q-measured_sdk)):.3f}"
                              f" max_dq={np.max(np.abs(data.qvel[6:])):.3f}")
            print(f"[INFO] p={p}{mode_text} base=({data.qpos[0]:.2f},{data.qpos[1]:.2f},{data.qpos[2]:.2f}) "
                  f"base_contact={base_contact}")
        if args.viewer:
            _sync_realtime(wall_start, data.time - sim_start)

    print(f"[RESULT] policy_steps={completed_steps} base_contact={base_contact} "
          f"base=({data.qpos[0]:.2f},{data.qpos[1]:.2f},{data.qpos[2]:.2f})")
    if machine:
        print("[FSM_METRICS] " + json.dumps(fsm_metrics, sort_keys=True))
    if args.trace_output:
        trace_path = os.path.abspath(os.path.expanduser(args.trace_output))
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        arrays = {name: np.asarray([record[name] for record in trace_records])
                  for name in trace_records[0]} if trace_records else {}
        np.savez_compressed(trace_path, **arrays)
        print(f"[INFO] sim trace saved: {trace_path} samples={len(trace_records)}")
    if args.viewer:
        viewer.close()
        _close_cv_windows()
    if writer is not None:
        writer.release()
        print(f"[INFO] dual-pane video saved: {video_path}")
    cam.close()
    if rgb_renderer is not None:
        rgb_renderer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] interrupted -> closing viewer windows")
    finally:
        _close_cv_windows()
