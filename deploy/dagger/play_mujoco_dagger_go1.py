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
sys.path.insert(0, os.path.join(REPO, "source", "basic_locomotion_isaaclab",
                              "basic_locomotion_isaaclab", "assets"))
from d435_geometry import D435_DEPTH_POSITION_BASE, D435_MOUNT_PITCH_DEG, D435_DEPTH_QUATERNION_ROS_WXYZ

GO1_XML = os.path.join(REPO, "deploy", "mujoco_models", "go1", "go1.xml")

# Policy action/joint order (matches Go1FlatEnvCfg.desired_joints_order):
# hip block, thigh block, calf block, each FL/FR/RL/RR.
DESIRED_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
DEFAULT_JOINT = np.array([0.0] * 4 + [0.9] * 4 + [-1.8] * 4)  # GO1 home pose

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

def _go1_xml_inner(kp: float = 35.0, kd: float = 0.5, armature: float = 0.01):
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
        if 'kp' in actuator.attrib:
            actuator.set('kp', str(kp))
        # IsaacLab clips raw actions, not the resulting position targets.
        actuator.set('ctrllimited', 'false')
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
                        camera_intrinsics=None, camera_size=(DEPTH_H, DEPTH_W)):
    """Compose a scene XML: go1 robot + terrain + baked depth camera."""
    inner = _go1_xml_inner(kp=kp, kd=kd, armature=armature)
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


def _make_key_callback(ctl, command_a):
    """Keyboard commands bounded by the student's training amplitudes."""
    vx_max, vy_max, wz_max = command_a
    def key_callback(keycode):
        if keycode == 256:                       # ESC
            ctl["quit"] = True
        elif keycode == 265:                     # Up -> vx +0.1
            ctl["vx"] = min(ctl["vx"] + 0.1, vx_max)
        elif keycode == 264:                     # Down -> vx -0.1
            ctl["vx"] = max(ctl["vx"] - 0.1, -vx_max)
        elif keycode in (263, 65):               # Left / A -> vy +0.1
            ctl["vy"] = min(ctl["vy"] + 0.1, vy_max)
        elif keycode in (262, 68):               # Right / D -> vy -0.1
            ctl["vy"] = max(ctl["vy"] - 0.1, -vy_max)
        elif keycode == 81:                      # Q -> wz +0.1
            ctl["wz"] = min(ctl["wz"] + 0.1, wz_max)
        elif keycode == 69:                      # E -> wz -0.1
            ctl["wz"] = max(ctl["wz"] - 0.1, -wz_max)
        elif keycode == 86:                      # V -> stop
            ctl["vx"] = ctl["vy"] = ctl["wz"] = 0.0
        elif keycode == 67:                      # C -> cruise
            ctl["vx"], ctl["vy"], ctl["wz"] = min(0.4, vx_max), 0.0, 0.0
    return key_callback


# --- depth preprocessing (mirrors training _sanitize_depth_data) ------------

def _preprocess_depth(depth, blur_sigma, device, *, min_z=0.0,
                      additive_noise_std=0.0, dropout_prob=0.0):
    d = np.array(depth, dtype=np.float32, copy=True)
    d[d < 0.01] = 0.0
    t = torch.from_numpy(d)[None, None].to(device)
    return preprocess_depth(t, blur_sigma=blur_sigma, min_z=min_z,
                            additive_noise_std=additive_noise_std, dropout_prob=dropout_prob)


def main():
    parser = argparse.ArgumentParser(description="Run the GO1 DAgger depth student in MuJoCo (sim2sim).")
    parser.add_argument("--ckpt", required=True,
                        help="path to dagger_policy.pt (Tiled or RayCaster student checkpoint)")
    parser.add_argument("--policy_steps", type=int, default=1000,
                        help="number of policy steps to run (default: 1000, -1 = until viewer closed)")
    parser.add_argument("--scene", default="flat",
                        help="terrain: flat|stairs|perlin|course (default: flat; course = flat start -> "
                             "stairs up/over/down -> perlin rough)")
    parser.add_argument("--step_rise", type=float, default=0.12, help="stair step height (m, default 0.12)")
    parser.add_argument("--step_tread", type=float, default=0.30, help="stair step depth (m, default 0.30 = training step_width)")
    parser.add_argument("--n_steps", type=int, default=12, help="stair steps up (then down)")
    parser.add_argument("--stair_width", type=float, default=6.0, help="stair width (m, default 6.0)")
    parser.add_argument("--kp", type=float, default=None, help="Override training stiffness.")
    parser.add_argument("--kd", type=float, default=None, help="Override training damping.")
    parser.add_argument("--no_act_delay", action="store_true",
                        help="disable the actuator command delay (training uses DelayedPDActuator "
                             "0-2 physics steps; use for an ablation)")
    parser.add_argument("--perlin_amp", type=float, default=0.18, help="perlin amplitude (m)")
    parser.add_argument("--viewer", action="store_true",
                        help="open an interactive mujoco viewer + live depth window (needs a display). "
                             "Keys: Up/Down=vx Left/Right(+A/D)=vy Q/E=wz (+-0.1/press), V=stop C=cruise ESC=quit")
    parser.add_argument("--cmd", default="0.0 0.0 0.0",
                        help="fixed velocity command 'vx vy wz' for headless runs (default 0 0 0); "
                             "with --viewer the keyboard overrides it")
    parser.add_argument("--no_video", action="store_true", help="skip dual-pane mp4 recording")
    args = parser.parse_args()
    try:
        _vx, _vy, _wz = (float(x) for x in args.cmd.split())
    except ValueError:
        raise SystemExit("--cmd must be 'vx vy wz' (three floats)")

    ckpt_path = os.path.abspath(args.ckpt)
    assert os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"
    run_dir = os.path.dirname(ckpt_path)

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
    depth_h, depth_w = map(int, md.get('depth_image_size', (DEPTH_H, DEPTH_W)))
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
    data.qpos[:3] = [spawn_x, 0.0, 0.4]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for nm, val in zip(DESIRED_ORDER, DEFAULT_JOINT):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = val
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(int(3.0 / model.opt.timestep)):         # 3 s hold home pose
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
    ctl = {"vx": _vx, "vy": _vy, "wz": _wz}   # headless: fixed --cmd; --viewer keyboard overrides
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
    act_delay = 0 if args.no_act_delay else int(np.random.randint(GO1_MIN_DELAY, GO1_MAX_DELAY + 1))
    delay_line = np.tile(DEFAULT_JOINT, (GO1_MAX_DELAY + 1, 1)).astype(np.float32)
    print(f"[INFO] actuator delay={act_delay} phys-steps, policy_every={policy_every} @ dt={sim_dt}")

    # --- optional viewer ---
    viewer = None
    if args.viewer:
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("[ERROR] --viewer needs a display; run headless without it.")
            sys.exit(1)
        ctl["quit"] = False
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=_make_key_callback(ctl, command_a))
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 270.0
        viewer.cam.elevation = -18.0
        cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("depth", 480, 280)
        print("[INFO] viewer on: Up/Down=vx Left/Right(+A/D)=vy Q/E=wz V=stop C=cruise ESC=quit.")

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
        video_path = os.path.join(run_dir, "videos", "dagger_play", f"mj_sim2sim_go1_{ckpt_tag}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

    completed_steps = 0
    # Start after settling and renderer/viewer setup, not at application launch.
    wall_start = time.perf_counter()
    sim_start = data.time
    if args.viewer:
        print('[INFO] viewer real-time synchronization enabled (target speed=1x).')
    for p in range(N_policy):
        if args.viewer and (not viewer.is_running() or ctl.get("quit")):
            print("[INFO] viewer closed -> stopping early")
            break

        quat = data.qpos[3:7]
        # Free-joint angular qvel is already in the local body frame, matching
        # IsaacLab root_ang_vel_b. Do not rotate it from world to body again.
        ang_vel_b = data.qvel[3:6].copy()
        grav_b = _projected_gravity(quat)
        jp = np.array([data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)]]
                       for nm in DESIRED_ORDER])
        jv = np.array([data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)]]
                       for nm in DESIRED_ORDER])
        cmd = np.array([ctl["vx"], ctl["vy"], ctl["wz"]], dtype=np.float32)
        if args.viewer and (last_cmd is None or np.any(np.abs(cmd - last_cmd) > 1e-3)):
            print(f"[CMD] vx={ctl['vx']:+.2f} vy={ctl['vy']:+.2f} wz={ctl['wz']:+.2f}", flush=True)
            last_cmd = cmd
        # 45-dim frame without clock; older clock-enabled students use 49 dims.
        obs1 = np.concatenate([ang_vel_b, grav_b, cmd, jp - DEFAULT_JOINT, jv, past_actions])
        if use_clock:
            phase = (phase + tcfg["step_freq"] / rl_freq) % 1.0
            obs1 = np.concatenate([obs1, phase]) if np.linalg.norm(cmd) > 0.01 else np.concatenate([obs1, -np.ones(4)])
        obs_hist = np.roll(obs_hist, -1, axis=0)
        obs_hist[-1] = obs1

        depth_t = _preprocess_depth(
            cam.render(), blur_sigma, device, min_z=float(md.get("depth_min_z", 0.0)),
            additive_noise_std=float(md.get("depth_sensor_noise", {}).get("additive_noise_std", 0.0)),
            dropout_prob=float(md.get("depth_sensor_noise", {}).get("dropout_prob", 0.0)),
        )
        frame = depth_t.squeeze(0).squeeze(0).numpy().astype(np.float16)
        if p == 0:
            hist[:] = frame
        else:
            hist = np.roll(hist, -1, axis=0)
            hist[-1] = frame
        # Match control-step history sampling and delay in the original DAgger.
        if delay > 0:
            seq = hist[-depth_len - delay:-delay]
        else:
            seq = hist[-depth_len:]

        if args.viewer:
            d_show = np.clip(depth_t.squeeze(0).squeeze(0).numpy(), 0.0, 2.0) / 2.0
            d_gray = (255.0 * (1.0 - d_show)).astype(np.uint8)
            cv2.imshow("depth", cv2.resize(d_gray, (480, 280), interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(1)

        if recording:
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
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), rl_freq,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)

        seq = torch.from_numpy(seq)[None].float().contiguous()   # (1, depth_len, 1, H, W)
        common = torch.from_numpy(obs_hist.reshape(1, -1)).float()
        with torch.no_grad():
            actions, _ = net(seq, common, hidden=None)
        actions = actions[0].numpy()

        # action processing (training order: clip -> filter -> scale -> +default)
        actions = np.clip(actions, -tcfg["clip_actions"], tcfg["clip_actions"])
        temp = 0.8 * actions + 0.2 * past_actions if tcfg["use_filter"] else actions
        target_pos = tcfg["action_scale"] * temp + DEFAULT_JOINT
        past_actions = actions.copy()

        for _ in range(policy_every):
            # actuator command delay: apply the target from `act_delay` physics steps ago
            # (matches DelayedPDActuator; no soft-limit clamp -- training doesn't clamp targets)
            delay_line = np.roll(delay_line, 1, axis=0)
            delay_line[0] = target_pos
            ctrl_target = delay_line[act_delay]
            for i, jn in enumerate(joint_of_actuator):
                if jn in policy_idx:
                    data.ctrl[i] = ctrl_target[policy_idx[jn]]
            mujoco.mj_step(model, data)
        # mj_step integrates qpos after computing kinematics. Refresh poses so
        # camera/gravity observations correspond to the same state as qpos/qvel.
        mujoco.mj_forward(model, data)
        completed_steps += 1

        if args.viewer:
            viewer.cam.lookat[:] = data.xpos[base_id]
            viewer.sync()

        if any(base_id in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
               for c in data.contact):
            base_contact += 1
        if p % 200 == 0:
            print(f"[INFO] p={p} base=({data.qpos[0]:.2f},{data.qpos[1]:.2f},{data.qpos[2]:.2f}) "
                  f"base_contact={base_contact}")
        if args.viewer:
            _sync_realtime(wall_start, data.time - sim_start)

    print(f"[RESULT] policy_steps={completed_steps} base_contact={base_contact} "
          f"base=({data.qpos[0]:.2f},{data.qpos[1]:.2f},{data.qpos[2]:.2f})")
    if args.viewer:
        viewer.close()
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"[INFO] dual-pane video saved: {video_path}")
    cam.close()
    if rgb_renderer is not None:
        rgb_renderer.close()


if __name__ == "__main__":
    main()
