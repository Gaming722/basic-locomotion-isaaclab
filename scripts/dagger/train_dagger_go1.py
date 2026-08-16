# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a depth-conditioned student policy with online DAgger supervision."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from contextlib import nullcontext

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a depth-conditioned DAgger policy from an RSL-RL teacher.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=5000, help="Steps between video recordings.")
parser.add_argument("--cam_distance", type=float, default=3.5, help="Camera distance behind the robot (m).")
parser.add_argument("--cam_height", type=float, default=2.2, help="Camera height above the robot (m).")
parser.add_argument("--cam_side", type=float, default=0.0,
                    help="Lateral offset of the follow camera (m, +x = robot's left). Use e.g. 2.0 for a 3/4 view.")
parser.add_argument("--dual_pane", action="store_true", default=False,
                    help="Record side-by-side video: left = Isaac Sim view, right = live grayscale depth.")
parser.add_argument("--follow_env", type=int, default=600,
                    help="Env index the camera/depth pane follow. GO1 envs 0-499 are command-zero (stand still); follow >=500.")
parser.add_argument("--terrain", type=str, default="rough",
                    help="Tiled env terrain: rough | stairs | slope | flat (e.g. --terrain stairs to record stair climbing).")
parser.add_argument("--difficulty", type=float, default=None,
                    help="Fixed terrain difficulty 0.0-1.0 applied to every sub-terrain (e.g. --difficulty 1.0 with "
                         "--terrain stairs gives all-max-height stairs). Default None = random per sub-terrain.")
parser.add_argument("--dual_pane_student_depth", action="store_true", default=False,
                    help="Show the student's sanitized depth (clip [0.1, 2.0], no-hit -> 1.0) in the dual-pane "
                         "right pane instead of the raw camera depth.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--depth_history_length",
    type=int,
    default=5,
    help="Number of depth frames consumed by the student depth GRU.",
)
parser.add_argument(
    "--depth_blur_sigma",
    type=float,
    default=1.0,
    help="Gaussian blur sigma (px) applied to the student depth to simulate optics / "
         "848->240 downscale smoothing. 0 disables.",
)
parser.add_argument(
    "--depth_additive_noise_std",
    type=float,
    default=0.0,
    help="Std (m) of additive Gaussian depth noise (D435 measurement noise). 0 disables.",
)
parser.add_argument(
    "--depth_dropout_prob",
    type=float,
    default=0.0,
    help="Probability of dropping a depth pixel to the far-saturation code 2.0 "
         "(simulate D435 holes/invalid pixels). 0 disables.",
)
parser.add_argument(
    "--depth_delay_frames",
    type=int,
    default=1,
    help="Delay (in env steps) of the depth frames fed to the student, modelling capture->inference "
         "latency. Must be < --depth_history_length.",
)
parser.add_argument(
    "--dagger_buffer_size",
    type=int,
    default=2048,
    help="Maximum number of aggregated in-memory DAgger samples.",
)
parser.add_argument(
    "--dagger_samples_per_step",
    type=int,
    default=64,
    help="Maximum number of environments added to the DAgger buffer per simulator step.",
)
parser.add_argument("--dagger_batch_size", type=int, default=64, help="Total batch size sampled from the CPU buffer.")
parser.add_argument(
    "--dagger_train_micro_batch_size",
    type=int,
    default=16,
    help="GPU micro-batch size for each behavior-cloning update.",
)
parser.add_argument(
    "--dagger_inference_batch_size",
    type=int,
    default=32,
    help="Maximum number of student-controlled environments evaluated on GPU at once.",
)
parser.add_argument("--dagger_learning_rate", type=float, default=3e-4, help="Student optimizer learning rate.")
parser.add_argument(
    "--disable_dagger_amp",
    action="store_true",
    default=False,
    help="Disable CUDA autocast for the DAgger student.",
)
parser.add_argument(
    "--dagger_train_every",
    type=int,
    default=4,
    help="Run student gradient updates every N simulator steps.",
)
parser.add_argument(
    "--dagger_updates_per_train",
    type=int,
    default=1,
    help="Number of student mini-batch updates each time training is triggered.",
)
parser.add_argument(
    "--dagger_warmup_steps",
    type=int,
    default=100,
    help="Number of initial simulator steps executed only with the teacher while filling the buffer.",
)
parser.add_argument(
    "--expert_beta_start",
    type=float,
    default=1.0,
    help="Initial probability of executing the teacher action after warmup.",
)
parser.add_argument(
    "--expert_beta_end",
    type=float,
    default=0.0,
    help="Final probability of executing the teacher action.",
)
parser.add_argument(
    "--expert_beta_decay_steps",
    type=int,
    default=10000,
    help="Number of simulator steps used to linearly decay teacher action mixing.",
)
parser.add_argument(
    "--max_training_steps",
    type=int,
    default=None,
    help="Optional maximum number of simulator steps. Defaults to running until the app closes.",
)
parser.add_argument(
    "--dagger_policy_path",
    type=str,
    default=None,
    help="Where to save the student policy checkpoint. Defaults to <teacher_run_dir>/dagger_policy.pt.",
)
parser.add_argument(
    "--dagger_save_interval",
    type=int,
    default=10000,
    help="Save a student policy checkpoint every N simulator steps. Use 0 to disable periodic saves.",
)
parser.add_argument(
    "--resume_from",
    type=str,
    default=None,
    help="Path to a dagger_policy.pt checkpoint to resume training from (restores network, optimizer, step).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# The dagger depth student always reads a depth camera. TiledCamera (rendered depth) requires
# the rendering kit unconditionally, so enable it regardless of --video.
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
from collections import deque

import cv2
import numpy as np

import gymnasium as gym
import torch
import torch.nn.functional as F

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
# Import extensions to set up environment tasks
import basic_locomotion_isaaclab.tasks  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

from dagger_network import DaggerNet, DaggerReplayBuffer

import isaaclab.utils.math as math_utils
from isaacsim.core.utils.viewports import set_camera_view


# Cache for the Gaussian blur kernels so they are built once per (size, sigma, device).
_gaussian_kernel_cache: dict[tuple, torch.Tensor] = {}


def _gaussian_kernel(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Return a (1, 1, k, k) normalized Gaussian blur kernel on ``device`` (cached)."""
    key = (kernel_size, sigma, str(device))
    kernel = _gaussian_kernel_cache.get(key)
    if kernel is not None:
        return kernel
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).view(1, 1, kernel_size, kernel_size)
    _gaussian_kernel_cache[key] = kernel
    return kernel


def _apply_depth_sensor_noise(depth: torch.Tensor) -> torch.Tensor:
    """Add D435-like artifacts to a (N, 1, H, W) depth image.

    Order matters: optical blur first (there are no holes yet, so the blur does not smear
    invalid pixels), then measurement noise, then dropout (holes) last so missing pixels stay
    clean zeros. Parameters mirror the on-robot depth preprocessing -- keep them in sync with
    the deploy side (--depth_*_... flags).
    """
    # 1) optical / downscale blur (Gaussian smoothing of the depth field).
    #    F.pad(mode="replicate") before the conv avoids the zero-padding artifact where border
    #    pixels get pulled toward 0 (which would read as a false "very near" border). Note:
    #    done this way because some torch builds do not accept padding_mode= in F.conv2d.
    if args_cli.depth_blur_sigma > 0:
        kernel_size = max(3, int(2 * math.ceil(2 * args_cli.depth_blur_sigma) + 1))
        kernel = _gaussian_kernel(kernel_size, args_cli.depth_blur_sigma, depth.device)
        pad = kernel_size // 2
        depth = F.pad(depth, (pad, pad, pad, pad), mode="replicate")
        depth = F.conv2d(depth, kernel)

    # 2) measurement noise (depth quantization / stereo error), then re-clamp so the student
    #    input stays in the [0.1, 2.0] range the deploy pipeline is defined on.
    if args_cli.depth_additive_noise_std > 0:
        depth = depth + torch.randn_like(depth) * args_cli.depth_additive_noise_std
    depth = depth.clip(0.1, 2.0)

    # 3) missing pixels (holes) -> far-saturation (2.0), matching the "unmeasurable = far"
    #    convention (no-hit -> 3.0 -> clip -> 2.0). The deploy side must map real D435 invalid
    #    pixels (0) to 2.0 as well, so holes stay in-distribution.
    if args_cli.depth_dropout_prob > 0:
        drop = torch.rand(depth.shape, device=depth.device) < args_cli.depth_dropout_prob
        depth = depth.masked_fill(drop, 2.0)

    return depth


def _sanitize_depth_data(env: RslRlVecEnvWrapper) -> torch.Tensor:
    if not hasattr(env.unwrapped, "_depth_camera"):
        raise RuntimeError(
            "The DAgger student needs env.unwrapped._depth_camera, matching collect_depth_to_heightmap.py. "
            "Run this with a depth-enabled vision task/config."
        )
    depth_data = env.unwrapped._depth_camera.data.output["distance_to_image_plane"]
    # Validated depth pipeline (Aliengo follow dagger): camera clipping_range=(0.01, 3.0)
    # with depth_clipping_behavior="max" (no-hit -> 3.0); clip to [0.1, 2.0]: real depth
    # in [0.01, 0.1] m saturates to 0.1 (near-saturation), beyond 2 m to 2.0 (far-saturation).
    depth_data = torch.nan_to_num(depth_data, nan=0.0, posinf=1.0, neginf=-1.0)
    depth_data = depth_data.clip(0.1, 2.0)
    depth_data = depth_data.permute(0, 3, 1, 2).contiguous()
    # D435-like sensor noise (blur -> measurement noise -> dropout holes).
    depth_data = _apply_depth_sensor_noise(depth_data)
    # D435 Min-Z alignment (env_cfg.depth_min_z): sub-Min-Z pixels (incl. blurred near-edges)
    # become far-saturated like every other unmeasurable pixel (deploy maps real D435 0 -> 2.0).
    # 0.0 disables (GO1 default).
    depth_min_z = getattr(env.unwrapped.cfg, "depth_min_z", 0.0)
    if depth_min_z > 0:
        depth_data = torch.where(depth_data < depth_min_z, 2.0, depth_data)
    return depth_data


def _delayed_newest_index(history_index: int, history_length: int) -> int:
    """Index into ``depth_history`` treated as the newest frame fed to the student.

    Mirrors the pipeline latency on the real robot: by the time an action is computed for the
    current state, the depth image available was captured ``--depth_delay_frames`` steps earlier.
    """
    return (history_index - args_cli.depth_delay_frames) % history_length


def _depth_sequence_from_history(
    depth_history: torch.Tensor,
    newest_index: int,
    env_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    history_length = depth_history.shape[0]
    ordered_history = torch.cat(
        (
            torch.arange(newest_index + 1, history_length, device=depth_history.device),
            torch.arange(0, newest_index + 1, device=depth_history.device),
        )
    )
    if env_indices is not None:
        depth_history = depth_history.index_select(1, env_indices.to(device=depth_history.device))
    return depth_history.index_select(0, ordered_history).permute(1, 0, 2, 3, 4).contiguous()


def _sample_env_indices(num_envs: int, max_samples: int | None) -> torch.Tensor | None:
    if max_samples is None or max_samples <= 0 or max_samples >= num_envs:
        return None
    return torch.randperm(num_envs)[:max_samples]


def _use_cuda_amp(device: torch.device | str) -> bool:
    return not args_cli.disable_dagger_amp and torch.device(device).type == "cuda"


def _autocast_context(device: torch.device | str):
    if _use_cuda_amp(device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _predict_student_actions_chunked(
    dagger_net: DaggerNet,
    depth_history: torch.Tensor,
    newest_index: int,
    common_obs: torch.Tensor,
    env_indices_cpu: torch.Tensor | None,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if env_indices_cpu is None:
        total_count = common_obs.shape[0]
    else:
        total_count = env_indices_cpu.numel()

    student_actions_batches: list[torch.Tensor] = []
    student_index_batches: list[torch.Tensor] = []
    chunk_size = max(1, args_cli.dagger_inference_batch_size)

    for start in range(0, total_count, chunk_size):
        stop = min(start + chunk_size, total_count)
        if env_indices_cpu is None:
            chunk_indices_cpu = torch.arange(start, stop)
        else:
            chunk_indices_cpu = env_indices_cpu[start:stop]

        chunk_depth_cpu = _depth_sequence_from_history(
            depth_history=depth_history,
            newest_index=newest_index,
            env_indices=chunk_indices_cpu,
        )
        chunk_depth = chunk_depth_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        chunk_indices_gpu = chunk_indices_cpu.to(device=device)
        chunk_common_obs = common_obs.index_select(0, chunk_indices_gpu)

        with torch.inference_mode(), _autocast_context(device):
            chunk_actions, _ = dagger_net(chunk_depth, chunk_common_obs, hidden=None)

        student_actions_batches.append(chunk_actions.detach().to(dtype=common_obs.dtype))
        if env_indices_cpu is not None:
            student_index_batches.append(chunk_indices_gpu)

        del chunk_depth_cpu, chunk_depth, chunk_common_obs

    student_actions = torch.cat(student_actions_batches, dim=0)
    student_indices_gpu = torch.cat(student_index_batches, dim=0) if student_index_batches else None
    return student_actions, student_indices_gpu


def _teacher_beta(step: int) -> float:
    if step < args_cli.dagger_warmup_steps:
        return 1.0

    if args_cli.expert_beta_decay_steps <= 0:
        return args_cli.expert_beta_end

    decay_step = step - args_cli.dagger_warmup_steps
    progress = min(1.0, max(0.0, decay_step / args_cli.expert_beta_decay_steps))
    return args_cli.expert_beta_start + progress * (args_cli.expert_beta_end - args_cli.expert_beta_start)


def _save_dagger_policy(
    path: str,
    dagger_net: DaggerNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    updates: int,
    metadata: dict,
) -> None:
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": dagger_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "updates": updates,
            "metadata": metadata,
        },
        path,
    )
    print(f"[INFO] Saved DAgger student policy checkpoint to: {path}")


def _dual_pane_frame(env, env_index=600, depth_range=(0.2, 5.0), student_depth=False) -> np.ndarray:
    """Composite a single frame: [left: Isaac Sim view | right: live grayscale depth] for one env.

    If ``student_depth`` is True, the right pane shows exactly what the student obs receives
    (the sanitized depth clipped to [0.1, 2.0], no-hit -> 1.0) instead of the raw camera depth.
    """
    sim_frame = env.unwrapped.render()
    sim_bgr = cv2.cvtColor(sim_frame, cv2.COLOR_RGB2BGR)
    if student_depth:
        # Student obs input: _sanitize_depth_data clips to [0.1, 2.0], no-hit -> 1.0.
        d = _sanitize_depth_data(env)[env_index, 0].float()
        d = (d - 0.1) / (2.0 - 0.1)
    else:
        raw = env.unwrapped._depth_camera.data.output["distance_to_image_plane"]
        d = raw[env_index, ..., 0].float()
        d = torch.nan_to_num(d, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        d = (d - depth_range[0]) / (depth_range[1] - depth_range[0])
    d = torch.clamp(d, 0.0, 1.0)  # no-hit -> 1.0 -> black
    gray = (255.0 * (1.0 - d)).to(torch.uint8).cpu().numpy()  # near = bright, far = dark
    depth_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = sim_bgr.shape[:2]
    hd, wd = depth_bgr.shape[:2]
    scale = h / hd
    depth_resized = cv2.resize(depth_bgr, (int(round(wd * scale)), h))
    return np.hstack([sim_bgr, depth_resized])


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train a depth-conditioned student with online DAgger supervision."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # Same expert handling as train_dagger.py: the M1 TrueVel teacher needs the student
    # env's privileged "teacher_obs" group (sim base_lin_vel spliced into the obs).
    agent_cfg.obs_groups = {"policy": ["teacher_obs"], "critic": ["teacher_obs"]}

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.terrain_type = args_cli.terrain
    if args_cli.difficulty is not None and not 0.0 <= args_cli.difficulty <= 1.0:
        raise ValueError(f"--difficulty must be in [0.0, 1.0], got {args_cli.difficulty}")
    env_cfg.difficulty = args_cli.difficulty
    env_cfg.rebuild_terrain()  # __post_init__ already ran at hydra parse; rebuild with the CLI terrain
    if args_cli.depth_history_length <= 0:
        raise ValueError("--depth_history_length must be positive.")
    if args_cli.depth_delay_frames < 0:
        raise ValueError("--depth_delay_frames must be non-negative.")
    if args_cli.depth_delay_frames >= args_cli.depth_history_length:
        raise ValueError("--depth_delay_frames must be smaller than --depth_history_length.")
    if args_cli.depth_blur_sigma < 0:
        raise ValueError("--depth_blur_sigma must be non-negative.")
    if args_cli.depth_additive_noise_std < 0:
        raise ValueError("--depth_additive_noise_std must be non-negative.")
    if not 0.0 <= args_cli.depth_dropout_prob < 1.0:
        raise ValueError("--depth_dropout_prob must be in [0.0, 1.0).")
    if args_cli.dagger_train_every <= 0:
        raise ValueError("--dagger_train_every must be positive.")
    if args_cli.dagger_updates_per_train <= 0:
        raise ValueError("--dagger_updates_per_train must be positive.")
    if args_cli.dagger_batch_size <= 0:
        raise ValueError("--dagger_batch_size must be positive.")
    if args_cli.dagger_train_micro_batch_size <= 0:
        raise ValueError("--dagger_train_micro_batch_size must be positive.")
    if args_cli.dagger_inference_batch_size <= 0:
        raise ValueError("--dagger_inference_batch_size must be positive.")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # the dagger student needs the depth camera sensor to be created
    env_cfg.use_depth_camera = True
    # keep the depth camera's red raycast markers out of the recorded viewport
    env_cfg.depth_camera.debug_vis = False

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video and args_cli.dual_pane:
        # dual-pane recording is handled manually in the training loop
        print("[INFO] Dual-pane recording: left = Isaac Sim view, right = live grayscale depth.")
    elif args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "dagger"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # GO1 teacher uses an asymmetric critic (372 obs) that does not match teacher_obs
    # (323). Only the actor + std are needed for DAgger labeling, so load those directly
    # and leave the critic untouched.
    loaded_dict = torch.load(resume_path, weights_only=False, map_location="cpu")
    model_sd = loaded_dict["model_state_dict"]
    actor_sd = {k.replace("actor.", ""): v for k, v in model_sd.items() if k.startswith("actor.")}
    runner.alg.policy.actor.load_state_dict(actor_sd)
    runner.alg.policy.std.data.copy_(model_sd["std"])

    teacher_policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    current_depth_cpu = _sanitize_depth_data(env).detach().to(device="cpu", dtype=torch.float16)

    num_envs = current_depth_cpu.shape[0]
    common_obs_size = obs["common"].shape[-1]
    single_action_space = getattr(env, "single_action_space", None)
    if single_action_space is None:
        single_action_space = getattr(env.unwrapped, "single_action_space", None)
    action_size = (
        gym.spaces.flatdim(single_action_space) if single_action_space is not None else env.action_space.shape[-1]
    )
    depth_channels = current_depth_cpu.shape[1]
    device = env.unwrapped.device

    depth_history = current_depth_cpu.unsqueeze(0).repeat(args_cli.depth_history_length, 1, 1, 1, 1).contiguous()
    history_index = args_cli.depth_history_length - 1

    dagger_net = DaggerNet(
        vec_size=common_obs_size,
        output_size=action_size,
        depth_channels=depth_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(dagger_net.parameters(), lr=args_cli.dagger_learning_rate)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=_use_cuda_amp(device))
    loss_fn = torch.nn.MSELoss()
    replay_buffer = DaggerReplayBuffer(capacity=args_cli.dagger_buffer_size)
    recent_losses: deque[float] = deque(maxlen=100)

    policy_path = (
        os.path.abspath(args_cli.dagger_policy_path)
        if args_cli.dagger_policy_path
        else os.path.join(log_dir, "dagger_policy.pt")
    )
    metadata = {
        "task": args_cli.task,
        "teacher_checkpoint_path": resume_path,
        "robot_obs_key": "common",
        "depth_history_length": args_cli.depth_history_length,
        "depth_history_storage": "cpu_float16",
        "depth_delay_frames": args_cli.depth_delay_frames,
        "depth_sensor_noise": {
            "blur_sigma": args_cli.depth_blur_sigma,
            "additive_noise_std": args_cli.depth_additive_noise_std,
            "dropout_prob": args_cli.depth_dropout_prob,
        },
        "depth_image_size": tuple(current_depth_cpu.shape[-2:]),
        "depth_channels": depth_channels,
        "common_obs_size": common_obs_size,
        "action_size": action_size,
        "num_envs": num_envs,
        "dagger_amp": _use_cuda_amp(device),
        "dagger_train_micro_batch_size": args_cli.dagger_train_micro_batch_size,
        "dagger_inference_batch_size": args_cli.dagger_inference_batch_size,
    }

    print(
        "[INFO] Starting online DAgger training "
        f"(num_envs={num_envs}, depth_history={args_cli.depth_history_length}, "
        f"depth_delay={args_cli.depth_delay_frames} step(s), depth_blur_sigma={args_cli.depth_blur_sigma}, "
        f"depth_additive_std={args_cli.depth_additive_noise_std}, depth_dropout={args_cli.depth_dropout_prob}, "
        f"buffer={args_cli.dagger_buffer_size}, batch={args_cli.dagger_batch_size}, "
        f"train_micro_batch={args_cli.dagger_train_micro_batch_size}, "
        f"inference_batch={args_cli.dagger_inference_batch_size}, amp={_use_cuda_amp(device)})."
    )

    # camera-follow setup for env 0 (video recording view)
    robot = env.unwrapped._robot
    eye_offset_b = torch.tensor([-args_cli.cam_distance, args_cli.cam_side, args_cli.cam_height], device=device)
    tgt_offset_b = torch.tensor([0.6, 0.0, 0.3], device=device)

    # dual-pane video recorder state
    dual_writer = None
    dual_frames_left = 0

    step = 0
    updates = 0

    # resume from a previous run if requested
    if args_cli.resume_from:
        resume_ckpt_path = os.path.abspath(args_cli.resume_from)
        if not os.path.exists(resume_ckpt_path):
            raise FileNotFoundError(f"--resume_from path not found: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        dagger_net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = int(ckpt["step"])
        updates = int(ckpt["updates"])
        print(
            f"[INFO] Resumed DAgger training from {resume_ckpt_path} "
            f"(step={step}, updates={updates}). Replay buffer starts empty and refills in ~32 steps."
        )

    while simulation_app.is_running() and (
        args_cli.max_training_steps is None or step < args_cli.max_training_steps
    ):
        common_obs = obs["common"]
        # feed the student depth delayed by --depth_delay_frames (capture->inference latency)
        newest_index = _delayed_newest_index(history_index, args_cli.depth_history_length)

        dagger_net.eval()
        with torch.inference_mode():
            expert_actions = teacher_policy(obs).detach()

        beta = _teacher_beta(step)
        if step < args_cli.dagger_warmup_steps or len(replay_buffer) < args_cli.dagger_batch_size:
            actions = expert_actions
        else:
            actions = expert_actions
            if beta <= 0.0:
                student_env_indices_cpu = None
            else:
                use_student_cpu = torch.rand(num_envs) >= beta
                student_env_indices_cpu = use_student_cpu.nonzero(as_tuple=False).flatten()

            if student_env_indices_cpu is None or student_env_indices_cpu.numel() > 0:
                student_actions, student_env_indices_gpu = _predict_student_actions_chunked(
                    dagger_net=dagger_net,
                    depth_history=depth_history,
                    newest_index=newest_index,
                    common_obs=common_obs,
                    env_indices_cpu=student_env_indices_cpu,
                    device=device,
                )
                if student_env_indices_cpu is None:
                    actions = student_actions
                else:
                    actions = expert_actions.clone()
                    actions.index_copy_(0, student_env_indices_gpu, student_actions)

                del student_actions

        replay_env_indices_cpu = _sample_env_indices(num_envs, args_cli.dagger_samples_per_step)
        replay_depth_cpu = _depth_sequence_from_history(
            depth_history=depth_history,
            newest_index=newest_index,
            env_indices=replay_env_indices_cpu,
        )
        if replay_env_indices_cpu is None:
            replay_common_cpu = common_obs.detach().to(device="cpu", dtype=torch.float32)
            replay_expert_cpu = expert_actions.detach().to(device="cpu", dtype=torch.float32)
        else:
            replay_env_indices_gpu = replay_env_indices_cpu.to(device=device)
            replay_common_cpu = common_obs.index_select(0, replay_env_indices_gpu).detach().to(
                device="cpu", dtype=torch.float32
            )
            replay_expert_cpu = expert_actions.index_select(0, replay_env_indices_gpu).detach().to(
                device="cpu", dtype=torch.float32
            )

        replay_buffer.add_batch(
            depth_sequences=replay_depth_cpu,
            common_obs=replay_common_cpu,
            expert_actions=replay_expert_cpu,
        )
        del replay_depth_cpu, replay_common_cpu, replay_expert_cpu

        obs, _, dones, _ = env.step(actions)
        dones_cpu = dones.bool().to(device="cpu")
        current_depth_cpu = _sanitize_depth_data(env).detach().to(device="cpu", dtype=torch.float16)

        history_index = (history_index + 1) % args_cli.depth_history_length
        depth_history[history_index].copy_(current_depth_cpu)
        if dones_cpu.any():
            depth_history[:, dones_cpu] = current_depth_cpu[dones_cpu].unsqueeze(0).expand(
                args_cli.depth_history_length, -1, -1, -1, -1
            )

        # follow a walking env (>=500) so recorded videos show the robot moving
        with torch.inference_mode():
            root_pos = robot.data.root_state_w[args_cli.follow_env, :3]
            yaw_q = math_utils.yaw_quat(robot.data.root_quat_w[args_cli.follow_env])
            eye = root_pos + math_utils.quat_apply(yaw_q, eye_offset_b)
            tgt = root_pos + math_utils.quat_apply(yaw_q, tgt_offset_b)
        set_camera_view(eye.cpu().numpy(), tgt.cpu().numpy(), "/OmniverseKit_Persp")

        # dual-pane recording: left = sim view, right = live depth
        if args_cli.video and args_cli.dual_pane:
            if dual_frames_left == 0 and step % args_cli.video_interval == 0:
                dual_frames_left = args_cli.video_length
                dual_fname = os.path.join(log_dir, "videos", "dagger", f"dual_step-{step}.mp4")
                os.makedirs(os.path.dirname(dual_fname), exist_ok=True)
            if dual_frames_left > 0:
                frame = _dual_pane_frame(env, env_index=args_cli.follow_env,
                                        student_depth=args_cli.dual_pane_student_depth)
                if dual_writer is None:
                    dual_writer = cv2.VideoWriter(
                        dual_fname, cv2.VideoWriter_fourcc(*"mp4v"), 50, (frame.shape[1], frame.shape[0])
                    )
                dual_writer.write(frame)
                dual_frames_left -= 1
                if dual_frames_left == 0:
                    dual_writer.release()
                    dual_writer = None
                    print(f"[INFO] dual-pane video saved: {dual_fname}")

        step += 1

        if len(replay_buffer) >= args_cli.dagger_batch_size and step % args_cli.dagger_train_every == 0:
            dagger_net.train()
            for _ in range(args_cli.dagger_updates_per_train):
                batch_depth, batch_common, batch_expert = replay_buffer.sample(
                    batch_size=args_cli.dagger_batch_size,
                    device="cpu",
                )

                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for start in range(0, args_cli.dagger_batch_size, args_cli.dagger_train_micro_batch_size):
                    stop = min(start + args_cli.dagger_train_micro_batch_size, args_cli.dagger_batch_size)
                    micro_batch_size = stop - start

                    micro_depth = batch_depth[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
                    micro_common = batch_common[start:stop].to(device=device, non_blocking=True)
                    micro_expert = batch_expert[start:stop].to(device=device, non_blocking=True)

                    with _autocast_context(device):
                        predicted_actions, _ = dagger_net(micro_depth, micro_common, hidden=None)
                        micro_loss = loss_fn(predicted_actions.float(), micro_expert)

                    weighted_loss = micro_loss * (micro_batch_size / args_cli.dagger_batch_size)
                    if grad_scaler.is_enabled():
                        grad_scaler.scale(weighted_loss).backward()
                    else:
                        weighted_loss.backward()

                    total_loss += micro_loss.item() * micro_batch_size
                    del micro_depth, micro_common, micro_expert, predicted_actions, micro_loss, weighted_loss

                if grad_scaler.is_enabled():
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(dagger_net.parameters(), max_norm=1.0)
                if grad_scaler.is_enabled():
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()

                updates += 1
                recent_losses.append(total_loss / args_cli.dagger_batch_size)
                del batch_depth, batch_common, batch_expert

        if step % 1000 == 0:
            mean_loss = sum(recent_losses) / len(recent_losses) if recent_losses else float("nan")
            print(
                f"[INFO] step={step} updates={updates} buffer={len(replay_buffer)} "
                f"teacher_beta={beta:.3f} recent_bc_loss={mean_loss:.5f}"
            )

        if args_cli.dagger_save_interval > 0 and step % args_cli.dagger_save_interval == 0:
            _save_dagger_policy(
                path=policy_path,
                dagger_net=dagger_net,
                optimizer=optimizer,
                step=step,
                updates=updates,
                metadata=metadata,
            )

    if step > 0:
        _save_dagger_policy(
            path=policy_path,
            dagger_net=dagger_net,
            optimizer=optimizer,
            step=step,
            updates=updates,
            metadata=metadata,
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
