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

import cv2
import numpy as np
import mujoco
import mujoco.viewer  # imports fine headless; only window creation needs a display
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts", "dagger"))
from mujoco_depth import MujocoDepthCamera
from dagger_network import DaggerNet

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
                    step_freq=1.4, sim_dt=0.005, decimation=4, history_length=5)

# MuJoCo camera geometry baked into the scene XML (training D435-equivalent).
CAM_POS = (0.26, 0.0, 0.12)   # in trunk frame
CAM_PITCH_DEG = 30.0
CAM_FOVY = 58.0               # vertical FOV deg
DEPTH_W, DEPTH_H = 240, 140

# Training dynamics parameters to align (go1_asset.py DelayedPDActuatorCfg).
TRAIN_DT = 0.005              # IsaacLab sim.dt
GO1_MIN_DELAY = 0             # actuator command delay (physics steps), randomized per env
GO1_MAX_DELAY = 2
# NOTE: soft_joint_pos_limit_factor=0.95 is NOT a command clamp in training -- it only feeds
# the joint_pos_limits reward / joint_pos_out_of_limits termination (JointPositionActionCfg
# sets the target directly via set_joint_position_target with no clamp). So we do NOT replicate
# it here; the MuJoCo position actuator's ctrlrange is the real hard joint limit.


# --- XML building (m1-style: extract inner, absolute meshdir, inject terrain+camera) ----

def _go1_xml_inner(kp: float = 30.0):
    """Return the go1.xml body inside <mujoco>..</mujoco> with absolute meshdir
    and the position-actuator stiffness set to ``kp`` (default 30 to match IsaacLab
    training stiffness; 100 = the mujoco_menagerie model's own tuned default)."""
    txt = open(GO1_XML).read()
    i = txt.find("<mujoco")
    j = txt.find(">", i) + 1
    k = txt.rfind("</mujoco>")
    inner = txt[j:k]
    mdir = os.path.join(os.path.dirname(GO1_XML), "assets")
    inner = re.sub(r'meshdir="[^"]*"', f'meshdir="{mdir}"', inner)
    # override the position-actuator servo stiffness (damping/armature in the XML
    # defaults already match GO1_DAMPING=2 / GO1_ARMATURE=0.01).
    inner = re.sub(r'<position kp="100"', f'<position kp="{kp}"', inner)
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
                        width=6.0, perlin_amp=0.18, kp=30.0):
    """Compose a scene XML: go1 robot + terrain + baked depth camera."""
    inner = _go1_xml_inner(kp=kp)
    cam_q = " ".join(f"{v:.6f}" for v in _camera_quat())

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
        assets += f'<hfield name="perlin" size="5 10 0.001 {perlin_amp}" nrow="{nrow}" ncol="{ncol}" file="{png}"/>\n'
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
        assets += (f'<hfield name="perlin" size="{perlin_x_half} 8 0.001 {perlin_amp}" '
                   f'nrow="{nrow}" ncol="{ncol}" file="{png}"/>\n')
        geoms += f'<geom type="hfield" hfield="perlin" pos="{perlin_x_center} 0 0" size="1 1 1" material="groundplane"/>\n'
        geoms += '<geom name="safety_floor" type="plane" size="40 40 0.05" pos="0 0 -2.0"/>\n'
    else:
        raise ValueError(f"Unknown scene '{scene}' (flat|stairs|perlin|course)")

    # bake the depth camera into the XML as the FIRST child of the trunk body so its
    # pose (0.26, 0, 0.12) is in the trunk frame and it tracks the robot (a camera at
    # worldbody level would not follow the trunk).
    camera = f'<camera name="depth_cam" pos="{CAM_POS[0]} {CAM_POS[1]} {CAM_POS[2]}" quat="{cam_q}" fovy="{CAM_FOVY}"/>\n'
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


def _make_key_callback(ctl):
    """glfw keycode callback editing the shared command dict (matches reference)."""
    def key_callback(keycode):
        if keycode == 256:                       # ESC
            ctl["quit"] = True
        elif keycode == 265:                     # Up -> vx +0.1
            ctl["vx"] = min(ctl["vx"] + 0.1, 1.0)
        elif keycode == 264:                     # Down -> vx -0.1
            ctl["vx"] = max(ctl["vx"] - 0.1, -0.5)
        elif keycode in (263, 65):               # Left / A -> vy +0.1
            ctl["vy"] = min(ctl["vy"] + 0.1, 0.5)
        elif keycode in (262, 68):               # Right / D -> vy -0.1
            ctl["vy"] = max(ctl["vy"] - 0.1, -0.5)
        elif keycode == 81:                      # Q -> wz +0.1
            ctl["wz"] = min(ctl["wz"] + 0.1, 1.0)
        elif keycode == 69:                      # E -> wz -0.1
            ctl["wz"] = max(ctl["wz"] - 0.1, -1.0)
        elif keycode == 86:                      # V -> stop
            ctl["vx"] = ctl["vy"] = ctl["wz"] = 0.0
        elif keycode == 67:                      # C -> cruise
            ctl["vx"], ctl["vy"], ctl["wz"] = 0.4, 0.0, 0.0
    return key_callback


# --- depth preprocessing (mirrors training _sanitize_depth_data) ------------

_gauss_cache = {}


def _gaussian_kernel(kernel_size, sigma, device):
    key = (kernel_size, sigma)
    if key in _gauss_cache:
        return _gauss_cache[key].to(device)
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :]).view(1, 1, kernel_size, kernel_size)
    _gauss_cache[key] = k
    return k.to(device)


def _blur_depth(depth, sigma):
    """Gaussian blur with replicate padding (matches training blur stage)."""
    if sigma <= 0:
        return depth
    kernel_size = max(3, int(2 * math.ceil(2 * sigma) + 1))
    k = _gaussian_kernel(kernel_size, sigma, depth.device)
    pad = kernel_size // 2
    depth = F.pad(depth, (pad, pad, pad, pad), mode="replicate")
    return F.conv2d(depth, k)


def _preprocess_depth(depth, blur_sigma, device):
    """Raw MuJoCo depth (H,W) -> (1,1,H,W) training sanitized depth [0.1, 2.0]."""
    d = np.asarray(depth, dtype=np.float32)
    d[d > 3.0] = np.inf          # no-hit / beyond far -> far
    d[d < 0.01] = np.inf         # below training near clip -> far
    d = np.nan_to_num(d, nan=0.0, posinf=3.0, neginf=-1.0)
    d = np.clip(d, 0.1, 2.0)
    t = torch.from_numpy(d)[None, None].to(device)   # (1,1,H,W)
    return _blur_depth(t, blur_sigma)


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
    parser.add_argument("--kp", type=float, default=30.0,
                        help="position-actuator stiffness (default 30 = IsaacLab training stiffness; "
                             "100 = mujoco_menagerie model default). kp=30 may be too soft for the "
                             "menagerie mass -> legs flop -> tips over on stairs; try --kp 100.")
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
        y = yaml.unsafe_load(open(env_yaml))
        tcfg = dict(
            action_scale=y.get("action_scale", GO1_DEFAULTS["action_scale"]),
            clip_actions=y.get("desired_clip_actions", GO1_DEFAULTS["clip_actions"]),
            use_filter=y.get("use_filter_actions", GO1_DEFAULTS["use_filter"]),
            step_freq=y.get("desired_step_freq", GO1_DEFAULTS["step_freq"]),
            sim_dt=y["sim"]["dt"] if "sim" in y else GO1_DEFAULTS["sim_dt"],
            decimation=y.get("decimation", GO1_DEFAULTS["decimation"]),
            history_length=y.get("history_length", GO1_DEFAULTS["history_length"]),
        )
    rl_freq = 1.0 / (tcfg["sim_dt"] * tcfg["decimation"])   # 50 Hz

    # --- load student policy ---
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    md = ckpt["metadata"]
    net = DaggerNet(vec_size=md["common_obs_size"], output_size=md["action_size"],
                    depth_channels=md["depth_channels"]).to("cpu")
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    depth_len = int(md["depth_history_length"])
    blur_sigma = float(md.get("depth_sensor_noise", {}).get("blur_sigma", 0.0))
    delay = int(md.get("depth_delay_frames", 0))
    print(f"[INFO] rl_freq={rl_freq:.1f} obs={md['common_obs_size']} depth_len={depth_len} "
          f"blur_sigma={blur_sigma} delay={delay}")

    # --- build + load the scene (go1 + terrain + depth cam) ---
    xml = build_go1_scene_xml(scene=args.scene, step_rise=args.step_rise, step_tread=args.step_tread,
                              n_steps=args.n_steps, width=args.stair_width, perlin_amp=args.perlin_amp,
                              kp=args.kp)
    model = mujoco.MjModel.from_xml_string(xml)
    # align physics timestep with training (sim.dt = 0.005)
    model.opt.timestep = TRAIN_DT
    data = mujoco.MjData(model)
    depth_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_cam")
    assert depth_cam_id >= 0, "depth_cam camera missing from scene XML"
    cam = MujocoDepthCamera(model, data, depth_cam_id, width=DEPTH_W, height=DEPTH_H)

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
    S = md["common_obs_size"] // tcfg["history_length"]      # 49 (no base_lin_vel)
    obs_hist = np.zeros((tcfg["history_length"], S), dtype=np.float32)
    hist = np.zeros((depth_len + delay, 1, DEPTH_H, DEPTH_W), dtype=np.float16)
    phase = np.array([0.0, 0.5, 0.5, 0.0])                   # FL, FR, RL, RR
    past_actions = np.zeros(model.nu, dtype=np.float32)
    ctl = {"vx": _vx, "vy": _vy, "wz": _wz}   # headless: fixed --cmd; --viewer keyboard overrides
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
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=_make_key_callback(ctl))
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

    for p in range(N_policy):
        if args.viewer and (not viewer.is_running() or ctl.get("quit")):
            print("[INFO] viewer closed -> stopping early")
            break

        quat = data.qpos[3:7]
        R = data.xmat[base_id].reshape(3, 3)
        # MuJoCo free-joint qvel[3:6] is the WORLD-frame angular velocity; IsaacLab's
        # root_ang_vel_b is body-frame, so rotate into the trunk frame (verified by test:
        # roll 90 deg + world-z rotation -> qvel=[0,0,1], body-frame=[0,1,0]).
        ang_vel_b = R.T @ data.qvel[3:6]
        grav_b = _projected_gravity(quat)
        jp = np.array([data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)]]
                       for nm in DESIRED_ORDER])
        jv = np.array([data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)]]
                       for nm in DESIRED_ORDER])
        cmd = np.array([ctl["vx"], ctl["vy"], ctl["wz"]], dtype=np.float32)
        if args.viewer and (last_cmd is None or np.any(np.abs(cmd - last_cmd) > 1e-3)):
            print(f"[CMD] vx={ctl['vx']:+.2f} vy={ctl['vy']:+.2f} wz={ctl['wz']:+.2f}", flush=True)
            last_cmd = cmd
        # 49-dim student obs frame (NO base_lin_vel)
        obs1 = np.concatenate([ang_vel_b, grav_b, cmd, jp - DEFAULT_JOINT, jv, past_actions])
        phase = (phase + tcfg["step_freq"] / rl_freq) % 1.0
        obs1 = np.concatenate([obs1, phase]) if np.linalg.norm(cmd) > 0.01 else np.concatenate([obs1, -np.ones(4)])
        obs_hist = np.roll(obs_hist, -1, axis=0)
        obs_hist[-1] = obs1

        depth_t = _preprocess_depth(cam.render(), blur_sigma, device)   # (1,1,H,W)
        hist = np.roll(hist, -1, axis=0)
        hist[-1] = depth_t.squeeze(0).squeeze(0).numpy().astype(np.float16)
        # depth delay: feed the sequence ending `delay` steps earlier (matches training)
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
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 50,
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

        if args.viewer:
            viewer.cam.lookat[:] = data.xpos[base_id]
            viewer.sync()

        if np.linalg.norm(data.cfrc_ext[base_id]) > 1e-3:
            base_contact += 1
        if p % 200 == 0:
            print(f"[INFO] p={p} base=({data.qpos[0]:.2f},{data.qpos[1]:.2f},{data.qpos[2]:.2f}) "
                  f"base_contact={base_contact}")

    print(f"[RESULT] policy_steps={p + 1} base_contact={base_contact} "
          f"base=({data.qpos[0]:.2f},{data.qpos[1]:.2f},{data.qpos[2]:.2f})")
    if args.viewer:
        viewer.close()
        cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"[INFO] dual-pane video saved: {video_path}")


if __name__ == "__main__":
    main()
