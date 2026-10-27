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
parser.add_argument(
    "--video_output_dir",
    type=str,
    default=None,
    help="Directory to save dagger videos. Defaults to <teacher_run>/videos/dagger. "
         "Use a distinct dir per concurrent run so videos don't overwrite each other.",
)
parser.add_argument("--cam_distance", type=float, default=3.5, help="Camera distance behind the robot (m).")
parser.add_argument("--cam_height", type=float, default=2.2, help="Camera height above the robot (m).")
parser.add_argument("--cam_side", type=float, default=0.0,
                    help="Lateral offset of the follow camera (m, +x = robot's left). Use e.g. 2.0 for a 3/4 view.")
parser.add_argument("--dual_pane", action="store_true", default=False,
                    help="Record side-by-side video: left = Isaac Sim view, right = live grayscale depth.")
parser.add_argument("--rotate_video_env", action="store_true",
                    help="Rotate followed env across available terrain groups for each dual-pane video clip.")
parser.add_argument("--follow_env", type=int, default=600,
                    help="Env index the camera/depth pane follows. It is automatically clamped past the "
                         "leading command-zero environments.")
parser.add_argument("--terrain", type=str, default="rough",
                    help="Tiled env terrain: rough | stairs | slope | flat (e.g. --terrain stairs to record stair climbing).")
parser.add_argument("--stair_direction", choices=("up", "down"), default=None,
                    help="Eval with --terrain stairs only: keep inverted stairs for uphill play, "
                         "or regular pyramid stairs for downhill play. Default keeps both.")
parser.add_argument("--difficulty", type=float, default=None,
                    help="Fixed terrain difficulty 0.0-1.0 applied to every sub-terrain (e.g. --difficulty 1.0 with "
                         "--terrain stairs gives all-max-height stairs). Default None = random per sub-terrain.")
parser.add_argument("--dual_pane_student_depth", action="store_true", default=False,
                    help="Show the student's sanitized depth (clip [0.1, 2.0], invalid/no-hit -> 2.0) in the dual-pane "
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
    default=0.5,
    help="Gaussian blur sigma (px) applied to the student depth at native 106x60 to "
         "emulate the residual smoothing of the real D435 848->106 (8x area-average) "
         "downscale / optics. 0 disables.",
)
parser.add_argument(
    "--depth_additive_noise_std",
    type=float,
    default=0.01,
    help="Std (m) of additive Gaussian depth noise (D435 measurement noise). 0 disables.",
)
parser.add_argument(
    "--depth_dropout_prob",
    type=float,
    default=0.01,
    help="Probability of dropping a depth pixel to the far-saturation code 2.0 "
         "(simulate D435 holes/invalid pixels). 0 disables.",
)
parser.add_argument("--depth_fps", type=float, default=30.0,
                    help="Nominal sensor update rate; depth is read and written to history every control step.")
parser.add_argument("--depth_min_z", type=float, default=0.0,
                    help="Measured minimum valid depth in metres; 0 disables the near-invalid mask.")
parser.add_argument(
    "--depth_delay_frames",
    type=int,
    default=1,
    help="Additional delay in control-step history entries; default 1 is 20 ms at 50 Hz.",
)
parser.add_argument(
    "--disable_proprio_noise", action="store_true", default=False,
    help="Disable student-only proprioceptive noise during DAgger training. Evaluation is always noise-free.",
)
parser.add_argument("--proprio_gyro_noise_amplitude", type=float, default=0.5,
                    help="Student angular-velocity uniform-noise half-range in rad/s.")
parser.add_argument("--proprio_gravity_noise_amplitude", type=float, default=0.05,
                    help="Student projected-gravity component uniform-noise half-range; vector is renormalized.")
parser.add_argument("--proprio_joint_pos_noise_amplitude", type=float, default=0.03,
                    help="Student joint-position uniform-noise half-range in rad.")
parser.add_argument("--proprio_joint_vel_noise_amplitude", type=float, default=1.5,
                    help="Student joint-velocity uniform-noise half-range in rad/s.")
parser.add_argument(
    "--dagger_buffer_size",
    type=int,
    default=32768,
    help="Long-term reservoir capacity. At 106x60x5 float16 depth this uses about 2.1 GB of host RAM.",
)
parser.add_argument("--dagger_recent_buffer_size", type=int, default=8192,
                    help="FIFO capacity used to keep recent on-policy recovery states.")
parser.add_argument("--dagger_recent_fraction", type=float, default=0.5,
                    help="Fraction of each training batch sampled from the recent FIFO buffer.")
parser.add_argument(
    "--dagger_samples_per_step",
    type=int,
    default=32,
    help="Maximum number of environments added to the DAgger buffer per simulator step.",
)
parser.add_argument("--dagger_batch_size", type=int, default=128, help="Total batch size sampled from the CPU buffer.")
parser.add_argument(
    "--dagger_train_micro_batch_size",
    type=int,
    default=64,
    help="GPU micro-batch size for each behavior-cloning update.",
)
parser.add_argument(
    "--dagger_inference_batch_size",
    type=int,
    default=128,
    help="Maximum number of student-controlled environments evaluated on GPU at once.",
)
parser.add_argument("--dagger_learning_rate", type=float, default=1e-4, help="Initial student learning rate.")
parser.add_argument("--dagger_min_learning_rate", type=float, default=3e-5,
                    help="Learning-rate floor after linear decay.")
parser.add_argument("--dagger_lr_decay_steps", type=int, default=300000,
                    help="Simulator steps over which learning rate decays linearly to its floor.")
parser.add_argument("--dagger_consistency_weight", type=float, default=0.1,
                    help="Weight for action consistency between two independent noisy views of one clean state.")
parser.add_argument("--proprio_delay_prob", type=float, default=0.5,
                    help="Probability per environment of delaying measured proprioceptive terms by one 20-ms policy step.")
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
    default=20000,
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
parser.add_argument(
    "--eval",
    action="store_true",
    default=False,
    help="Evaluate a trained dagger_policy.pt student only: no teacher, no replay buffer, no gradient "
         "updates, no checkpoint saves. The step counter starts at 0.",
)
parser.add_argument(
    "--keyboard",
    action="store_true",
    default=False,
    help="During --eval, use viewport keyboard velocity commands instead of random commands. "
         "Requires a visible Isaac Sim window.",
)
parser.add_argument(
    "--show_depth", action="store_true", default=False,
    help="During visible --eval, open a live window showing the latest delayed depth frame fed to the student.",
)
parser.add_argument(
    "--eval_policy",
    type=str,
    default=None,
    help="Path to a trained dagger_policy.pt to evaluate. Required iff --eval.",
)
parser.add_argument(
    "--eval_with_training_noise",
    action="store_true",
    default=False,
    help="Eval only: reproduce the student-side proprioceptive and depth corruption used during training.",
)
parser.add_argument(
    "--eval_metrics_path",
    type=str,
    default=None,
    help="Eval only: write aggregate rollout metrics as JSON.",
)
parser.add_argument(
    "--eval_metrics_warmup_steps",
    type=int,
    default=100,
    help="Eval only: exclude this many initial control steps from tracking/action metrics.",
)
parser.add_argument(
    "--obs_dump",
    type=str,
    default=None,
    help="Eval only: save the exact student input sequence and raw actions for one environment to an NPZ file.",
)
parser.add_argument(
    "--obs_dump_env",
    type=int,
    default=0,
    help="Environment index recorded by --obs_dump (default: 0).",
)
parser.add_argument(
    "--cmd",
    type=str,
    default=None,
    help="Fixed velocity command 'vx vy wz' for all envs (eval only; overrides the env's random command "
         "generator, e.g. --cmd \"0.5 0 0\" for constant forward).",
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
import time
import json
from collections import deque

import cv2
import numpy as np

import gymnasium as gym
import torch

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

from basic_locomotion_isaaclab.assets.d435_geometry import (
    D435_BOTTOM_SCREW_POSITION_BASE, D435_DEPTH_ORIGIN_IN_SCREW, D435_MOUNT_PITCH_DEG,
)
from basic_locomotion_isaaclab.tasks.custom_events import get_num_fixed_command_envs
from dagger_network import DaggerNet, DaggerReplayBuffer
from keyboard_control import install_isaac_keyboard_commands
from depth_pipeline import PIPELINE_VERSION, preprocess_depth, depth_sequence_from_history, student_depth_grayscale
from depth_preview import IsaacDepthPreview
from proprio_noise import (
    ProprioNoiseCfg,
    ProprioceptiveNoise,
    corrupt_common_observation,
    delay_sensor_terms,
)
from video_follow import rotating_video_env

import isaaclab.utils.math as math_utils
from isaacsim.core.utils.viewports import set_camera_view


def _sanitize_depth_data(env: RslRlVecEnvWrapper, *, zero_depth: bool = False) -> torch.Tensor:
    if not hasattr(env.unwrapped, "_depth_camera"):
        raise RuntimeError("DAgger requires a depth-enabled student task (Tiled or RayCaster).")
    depth = env.unwrapped._depth_camera.data.output["distance_to_image_plane"]
    depth = depth.permute(0, 3, 1, 2).contiguous()
    # Synthetic corruption is a training augmentation. Evaluation/play and physical
    # deployment must consume the sensor measurement without adding artificial noise.
    randomize = not args_cli.eval or args_cli.eval_with_training_noise
    processed = preprocess_depth(
        depth, blur_sigma=args_cli.depth_blur_sigma,
        additive_noise_std=args_cli.depth_additive_noise_std if randomize else 0.0,
        dropout_prob=args_cli.depth_dropout_prob if randomize else 0.0,
        min_z=args_cli.depth_min_z,
    )
    if zero_depth:
        processed.zero_()
    env.unwrapped._dagger_processed_depth = processed
    env.unwrapped._dagger_force_zero_depth = zero_depth
    return processed


def _delayed_newest_index(history_index: int, history_length: int) -> int:
    """Index into ``depth_history`` treated as the newest frame fed to the student.

    Mirrors the pipeline latency on the real robot: by the time an action is computed for the
    current state, the depth image available was captured ``--depth_delay_frames`` control steps earlier.
    """
    return (history_index - args_cli.depth_delay_frames) % history_length


def _depth_sequence_from_history(
    depth_history: torch.Tensor,
    newest_index: int,
    env_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    return depth_sequence_from_history(
        depth_history, newest_index, args_cli.depth_history_length, env_indices
    )



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
    zero_depth: bool = False,
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
        if zero_depth:
            chunk_depth.zero_()
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
    (the sanitized depth clipped to [0.1, 2.0], invalid/no-hit -> 2.0) instead of the raw camera depth.
    """
    sim_frame = env.unwrapped.render()
    sim_bgr = cv2.cvtColor(sim_frame, cv2.COLOR_RGB2BGR)
    if student_depth or getattr(env.unwrapped, "_dagger_force_zero_depth", False):
        # Student obs input: _sanitize_depth_data clips to [0.1, 2.0], invalid/no-hit -> 2.0.
        # Reuse the captured frame; a preview must not sample fresh sensor noise.
        d = env.unwrapped._dagger_processed_depth[env_index, 0].float()
        d = (d - 0.1) / (2.0 - 0.1)
    else:
        raw = env.unwrapped._depth_camera.data.output["distance_to_image_plane"]
        d = raw[env_index, ..., 0].float()
        d = torch.nan_to_num(d, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        d = (d - depth_range[0]) / (depth_range[1] - depth_range[0])
    d = torch.clamp(d, 0.0, 1.0)  # invalid/no-hit -> 2.0 -> black
    gray = (255.0 * (1.0 - d)).to(torch.uint8).cpu().numpy()  # near = bright, far = dark
    depth_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = sim_bgr.shape[:2]
    hd, wd = depth_bgr.shape[:2]
    scale = h / hd
    depth_resized = cv2.resize(depth_bgr, (int(round(wd * scale)), h))
    return np.hstack([sim_bgr, depth_resized])


def _validate_eval_cli_args() -> str:
    """Validate the --eval flags and return the absolute eval-policy path."""
    if not args_cli.eval_policy:
        raise SystemExit("--eval requires --eval_policy <path to dagger_policy.pt>")
    eval_policy_path = os.path.abspath(args_cli.eval_policy)
    if not os.path.exists(eval_policy_path):
        raise FileNotFoundError(f"--eval_policy not found: {eval_policy_path}")
    if args_cli.keyboard and args_cli.headless:
        raise SystemExit("--keyboard requires a visible Isaac Sim window (omit --headless).")
    if args_cli.headless and args_cli.max_training_steps is None:
        raise SystemExit(
            "--eval with --headless needs --max_training_steps <N> "
            "(otherwise the evaluation would run forever on a server)."
        )
    if args_cli.keyboard and args_cli.cmd:
        raise SystemExit("Use --keyboard or --cmd, not both.")
    if args_cli.eval_metrics_path is not None and args_cli.max_training_steps is None:
        raise SystemExit("--eval_metrics_path requires --max_training_steps <N>.")
    if args_cli.eval_metrics_warmup_steps < 0:
        raise SystemExit("--eval_metrics_warmup_steps must be non-negative.")
    if (args_cli.max_training_steps is not None
            and args_cli.eval_metrics_warmup_steps >= args_cli.max_training_steps):
        raise SystemExit("--eval_metrics_warmup_steps must be smaller than --max_training_steps.")
    if args_cli.cmd:
        try:
            _ = [float(x) for x in args_cli.cmd.split()]
        except ValueError:
            raise SystemExit("--cmd must be 'vx vy wz' (three floats)")
    return eval_policy_path


def _check_student_ckpt_compat(ckpt: dict, net: torch.nn.Module, depth_image_hw: tuple) -> None:
    """Hard-fail on student architecture mismatch; warn on metadata distribution shift."""
    state_dict = ckpt.get("model_state_dict", {})
    net_sd = net.state_dict()
    if set(state_dict.keys()) != set(net_sd.keys()):
        missing = sorted(set(net_sd.keys()) - set(state_dict.keys()))
        extra = sorted(set(state_dict.keys()) - set(net_sd.keys()))
        raise RuntimeError(
            "student checkpoint architecture mismatch with the live env: "
            f"missing={missing} extra={extra}. The checkpoint was trained for a different "
            "obs/action/depth shape or network."
        )
    bad = [k for k in state_dict if tuple(state_dict[k].shape) != tuple(net_sd[k].shape)]
    if bad:
        raise RuntimeError(
            "student checkpoint shape mismatch for: " + ", ".join(bad)
            + ". Cannot load a policy trained at a different resolution/size."
        )
    meta = ckpt.get("metadata", {})
    if meta.get("depth_pipeline_version", 1) != PIPELINE_VERSION:
        print("[WARN] Student checkpoint uses an older depth pipeline; capture timing or delay units changed.")
    if "depth_fps" in meta and meta["depth_fps"] != args_cli.depth_fps:
        print(f"[WARN] Checkpoint depth_fps={meta['depth_fps']} differs from current {args_cli.depth_fps}.")
    if "depth_image_size" in meta and tuple(meta["depth_image_size"]) != tuple(depth_image_hw):
        print(
            f"[WARN] --eval_policy metadata depth_image_size={meta['depth_image_size']} != live env "
            f"{tuple(depth_image_hw)}; the depth input distribution may differ from training."
        )
    if "depth_history_length" in meta and meta["depth_history_length"] != args_cli.depth_history_length:
        print(
            f"[WARN] --eval_policy metadata depth_history_length={meta['depth_history_length']} != current "
            f"--depth_history_length={args_cli.depth_history_length}."
        )
    if "depth_delay_frames" in meta and meta["depth_delay_frames"] != args_cli.depth_delay_frames:
        print(
            f"[WARN] --eval_policy metadata depth_delay_frames={meta['depth_delay_frames']} != current "
            f"--depth_delay_frames={args_cli.depth_delay_frames}."
        )


def _install_fixed_command(env) -> None:
    """Replace the env's random-command generator with a fixed velocity command (mirrors play.py)."""
    try:
        _vx, _vy, _wz = (float(x) for x in args_cli.cmd.split())
    except ValueError:
        raise SystemExit("--cmd must be 'vx vy wz' (three floats)")
    if hasattr(env.unwrapped, "_commands"):
        import basic_locomotion_isaaclab.tasks.custom_events as _ce

        def _fixed_random_commands(env_obj, env_ids=None):
            env_obj._commands[:, 0] = _vx
            env_obj._commands[:, 1] = _vy
            env_obj._commands[:, 2] = _wz

        _ce._get_new_random_commands = _fixed_random_commands
        print(f"[INFO] Fixed velocity command: {_vx:.2f} {_vy:.2f} {_wz:.2f} (all envs)")
    else:
        print("[WARN] --cmd ignored: env has no _commands buffer.")


def _effective_follow_env(follow_env: int, env, has_fixed_cmd: bool) -> int:
    """Clamp --follow_env onto an env that is actually moving.

    DAgger student envs keep the same standing-env ratio as the teacher regardless
    of num_envs. With a fixed --cmd every env moves.
    """
    num_envs = env.unwrapped.num_envs
    first_moving_env = 0 if has_fixed_cmd else get_num_fixed_command_envs(env.unwrapped.cfg, num_envs)
    return max(first_moving_env, min(follow_env, num_envs - 1))


def _video_follow_for_clip(env, clip_index):
    terrain = env.unwrapped._terrain
    gen = terrain.cfg.terrain_generator
    num_envs = env.unwrapped.num_envs
    first_env = 0 if args_cli.cmd is not None else get_num_fixed_command_envs(env.unwrapped.cfg, num_envs)
    initial = _effective_follow_env(args_cli.follow_env, env, args_cli.cmd is not None)
    if gen is None:
        columns = [0] * num_envs
        proportions = None
        num_columns = 1
    else:
        # Tiled spacing assigns row-major cells rather than importer terrain_types.
        columns = ([i % gen.num_cols for i in range(num_envs)]
                   if getattr(env.unwrapped.cfg, "enforce_env_spacing", False)
                   else terrain.terrain_types.cpu().tolist())
        proportions = ([(name, cfg.proportion) for name, cfg in gen.sub_terrains.items()]
                       if gen.curriculum else None)
        num_columns = gen.num_cols
    selected, label = rotating_video_env(
        columns, first_env=first_env, initial_env=initial, clip_index=clip_index,
        num_columns=num_columns, terrain_proportions=proportions,
    )
    level = int(terrain.terrain_levels[selected]) if gen is not None else None
    print(f"[INFO] Video subject: env={selected}, terrain={label}, level={level}, clip={clip_index}")
    return selected


def _run_student_eval(env, eval_policy_path: str, video_out_dir: str) -> None:
    """Roll an already-trained DAgger student (no teacher / replay / gradients / checkpoint saves).

    The step counter starts at 0 (never restored from the checkpoint). Reuses the exact training
    depth pipeline, camera follow and dual-pane recording so the video shows the true student
    input/output.
    """
    device = env.unwrapped.device
    ckpt = torch.load(eval_policy_path, map_location=device, weights_only=False)

    # Install before the first explicit observation, including after episode resets.
    keyboard_subscription = None
    if args_cli.keyboard:
        keyboard_subscription = install_isaac_keyboard_commands(
            env, ckpt.get("metadata", {}).get("command_a", env.unwrapped.cfg.command_a))
    elif args_cli.cmd:
        _install_fixed_command(env)
    keyboard_controller = keyboard_subscription[3] if keyboard_subscription is not None else None

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
    follow_env = _effective_follow_env(
        args_cli.follow_env, env, has_fixed_cmd=args_cli.cmd is not None or args_cli.keyboard)

    history_capacity = args_cli.depth_history_length + args_cli.depth_delay_frames
    depth_history = current_depth_cpu.unsqueeze(0).repeat(history_capacity, 1, 1, 1, 1).contiguous()
    history_index = history_capacity - 1

    dagger_net = DaggerNet(
        vec_size=common_obs_size,
        output_size=action_size,
        depth_channels=depth_channels,
    ).to(device)
    _check_student_ckpt_compat(ckpt, dagger_net, tuple(current_depth_cpu.shape[-2:]))
    dagger_net.load_state_dict(ckpt["model_state_dict"])
    dagger_net.eval()

    proprio_noise = None
    if args_cli.eval_with_training_noise:
        noise_metadata = ckpt.get("metadata", {}).get("proprioceptive_noise", {})
        legacy_gaussian = noise_metadata.get("model") == "per_step_zero_mean_gaussian"
        proprio_noise_cfg = ProprioNoiseCfg(
            distribution="gaussian" if legacy_gaussian else "uniform",
            angular_velocity_half_range=float(noise_metadata.get(
                "angular_velocity_half_range",
                noise_metadata.get("angular_velocity_std", args_cli.proprio_gyro_noise_amplitude))),
            projected_gravity_half_range=float(noise_metadata.get(
                "projected_gravity_half_range",
                noise_metadata.get("projected_gravity_std", args_cli.proprio_gravity_noise_amplitude))),
            joint_position_half_range=float(noise_metadata.get(
                "joint_position_half_range",
                noise_metadata.get("joint_position_std", args_cli.proprio_joint_pos_noise_amplitude))),
            joint_velocity_half_range=float(noise_metadata.get(
                "joint_velocity_half_range",
                noise_metadata.get("joint_velocity_std", args_cli.proprio_joint_vel_noise_amplitude))),
        )
        proprio_noise = ProprioceptiveNoise(
            num_envs, env.unwrapped.cfg.history_length, device, proprio_noise_cfg
        )

    robot = env.unwrapped._robot
    eye_offset_b = torch.tensor([-args_cli.cam_distance, args_cli.cam_side, args_cli.cam_height], device=device)
    tgt_offset_b = torch.tensor([0.6, 0.0, 0.3], device=device)

    dual_writer = None
    dual_frames_left = 0
    video_clip_index = 0
    max_steps = args_cli.max_training_steps
    env_cfg = env.unwrapped.cfg
    print(
        "[INFO] Evaluating DAgger student "
        f"(policy={eval_policy_path}, terrain={getattr(env_cfg, 'terrain_type', '?')}, "
        f"difficulty={getattr(env_cfg, 'difficulty', None)}, num_envs={num_envs}, follow_env={follow_env}, "
        f"depth_history={args_cli.depth_history_length}, depth_delay={args_cli.depth_delay_frames}, "
        f"steps={'until the window closes' if max_steps is None else max_steps}, "
        f"cmd={'keyboard' if args_cli.keyboard else (args_cli.cmd if args_cli.cmd else 'random')}, "
        f"video={'dual-pane' if (args_cli.video and args_cli.dual_pane) else ('rgb' if args_cli.video else 'off')})."
    )

    depth_preview = None
    if args_cli.show_depth:
        depth_preview = IsaacDepthPreview(current_depth_cpu.shape[-1], current_depth_cpu.shape[-2])
        print(f"[INFO] Live depth preview: env={follow_env}, latest frame in actor's delayed depth history "
              "(near=white, far=black). Click the Isaac Sim window for keyboard commands.")

    dump_records = []
    dump_env = args_cli.obs_dump_env
    if args_cli.obs_dump is not None:
        if not 0 <= dump_env < num_envs:
            raise ValueError(f"--obs_dump_env must be in [0, {num_envs}), got {dump_env}")
        print(f"[INFO] Observation audit: recording env={dump_env} to {args_cli.obs_dump}")

    metrics_enabled = args_cli.eval_metrics_path is not None
    metric_steps = 0
    fall_events = 0
    fallen_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
    first_fall_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    action_sq_sum = 0.0
    action_count = 0
    action_abs_max = 0.0
    action_delta_sq_sum = 0.0
    action_delta_count = 0
    action_delta_abs_max = 0.0
    lin_vel_abs_error_sum = torch.zeros(2, dtype=torch.float64, device=device)
    yaw_rate_abs_error_sum = 0.0
    tracking_count = 0
    previous_actions = None
    dones_cpu = torch.zeros(num_envs, dtype=torch.bool, device="cpu")

    step = 0
    while simulation_app.is_running() and (max_steps is None or step < max_steps):
        start_time = time.time()
        clean_common_obs = obs["common"]
        common_obs = (
            proprio_noise.update(clean_common_obs, dones_cpu)
            if proprio_noise is not None else clean_common_obs
        )
        newest_index = _delayed_newest_index(history_index, history_capacity)
        if depth_preview is not None:
            preview_depth = depth_history[newest_index, follow_env, 0]
            if keyboard_controller is not None and keyboard_controller.sensor_blind:
                preview_depth = torch.zeros_like(preview_depth)
            gray = student_depth_grayscale(preview_depth)
            depth_preview.update(gray)

        # All envs are controlled by the student (env_indices_cpu=None) -- the same code path as
        # the training loop's beta<=0 branch (training L715-724).
        student_actions, _ = _predict_student_actions_chunked(
            dagger_net=dagger_net,
            depth_history=depth_history,
            newest_index=newest_index,
            common_obs=common_obs,
            env_indices_cpu=None,
            device=device,
            zero_depth=keyboard_controller.sensor_blind if keyboard_controller is not None else False,
        )

        if metrics_enabled and step >= args_cli.eval_metrics_warmup_steps:
            metric_steps += 1
            actions64 = student_actions.to(dtype=torch.float64)
            action_sq_sum += actions64.square().sum().item()
            action_count += actions64.numel()
            action_abs_max = max(action_abs_max, actions64.abs().max().item())
            if previous_actions is not None:
                valid_delta = ~dones_cpu.to(device=device)
                if valid_delta.any():
                    delta = actions64[valid_delta] - previous_actions[valid_delta]
                    action_delta_sq_sum += delta.square().sum().item()
                    action_delta_count += delta.numel()
                    action_delta_abs_max = max(action_delta_abs_max, delta.abs().max().item())

            commands = env.unwrapped._commands
            lin_vel_abs_error_sum += (
                commands[:, :2] - robot.data.root_lin_vel_b[:, :2]
            ).abs().to(dtype=torch.float64).sum(dim=0)
            yaw_rate_abs_error_sum += (
                commands[:, 2] - robot.data.root_ang_vel_b[:, 2]
            ).abs().to(dtype=torch.float64).sum().item()
            tracking_count += num_envs

        previous_actions = student_actions.detach().to(dtype=torch.float64)

        if args_cli.obs_dump is not None:
            depth_sequence = _depth_sequence_from_history(depth_history, newest_index)
            dump_records.append(
                (
                    common_obs[dump_env].detach().to(device="cpu", dtype=torch.float32).numpy().copy(),
                    depth_sequence[dump_env].detach().to(device="cpu", dtype=torch.float32).numpy().copy(),
                    student_actions[dump_env].detach().to(device="cpu", dtype=torch.float32).numpy().copy(),
                )
            )

        obs, _, dones, _ = env.step(student_actions)
        dones_cpu = dones.bool().to(device="cpu")
        if metrics_enabled:
            dones_device = dones.bool().to(device=device)
            fall_events += int(dones_device.sum().item())
            first_now = dones_device & ~fallen_envs
            first_fall_step[first_now] = step + 1
            fallen_envs |= dones_device
        # Match the original DAgger: read sensor data and append a snapshot each control step.
        current_depth_cpu = _sanitize_depth_data(
            env, zero_depth=keyboard_controller.sensor_blind if keyboard_controller is not None else False
        ).detach().to(device="cpu", dtype=torch.float16)
        history_index = (history_index + 1) % history_capacity
        depth_history[history_index].copy_(current_depth_cpu)
        if dones_cpu.any():
            depth_history[:, dones_cpu] = current_depth_cpu[dones_cpu].unsqueeze(0).expand(
                history_capacity, -1, -1, -1, -1
            )

        if (args_cli.rotate_video_env and dual_frames_left == 0
                and step % args_cli.video_interval == 0):
            follow_env = _video_follow_for_clip(env, video_clip_index)
            video_clip_index += 1

        # Camera follow -- mirrors training loop (keep in sync).
        with torch.inference_mode():
            root_pos = robot.data.root_state_w[follow_env, :3]
            yaw_q = math_utils.yaw_quat(robot.data.root_quat_w[follow_env])
            eye = root_pos + math_utils.quat_apply(yaw_q, eye_offset_b)
            tgt = root_pos + math_utils.quat_apply(yaw_q, tgt_offset_b)
        set_camera_view(eye.cpu().numpy(), tgt.cpu().numpy(), "/OmniverseKit_Persp")

        # Dual-pane recording -- mirrors training loop (keep in sync). Non-dual-pane --video uses
        # the RecordVideo wrapper applied in main(), so there is no double recording here.
        if args_cli.video and args_cli.dual_pane:
            if dual_frames_left == 0 and step % args_cli.video_interval == 0:
                dual_frames_left = args_cli.video_length
                dual_suffix = f"_env-{follow_env}" if args_cli.rotate_video_env else ""
                dual_fname = os.path.join(video_out_dir, f"dual_step-{step}{dual_suffix}.mp4")
                os.makedirs(os.path.dirname(dual_fname), exist_ok=True)
            if dual_frames_left > 0:
                frame = _dual_pane_frame(env, env_index=follow_env,
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

        if step % 100 == 0:
            print(f"[INFO] eval step={step}/{max_steps if max_steps is not None else 'inf'}")

        # Real-time pacing for interactive viewing (mirrors play.py).
        sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    if depth_preview is not None:
        depth_preview.close()
    if args_cli.obs_dump is not None and dump_records:
        output = os.path.abspath(os.path.expanduser(args_cli.obs_dump))
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        np.savez_compressed(
            output,
            common=np.stack([record[0] for record in dump_records]),
            depth=np.stack([record[1] for record in dump_records]),
            raw_action=np.stack([record[2] for record in dump_records]),
            env=np.int64(dump_env),
            step_dt=np.float32(env.unwrapped.step_dt),
        )
        print(f"[INFO] Observation audit saved: {output} ({len(dump_records)} samples)")
    if metrics_enabled:
        output = os.path.abspath(os.path.expanduser(args_cli.eval_metrics_path))
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        horizon_seconds = max_steps * env.unwrapped.step_dt
        first_fall_seconds = first_fall_step.to(dtype=torch.float64) * env.unwrapped.step_dt
        observed_first_fall = first_fall_seconds[first_fall_step >= 0]
        metrics = {
            "policy": os.path.abspath(eval_policy_path),
            "checkpoint_step": int(ckpt.get("step", -1)),
            "seed": int(env_cfg.seed),
            "terrain": getattr(env_cfg, "terrain_type", None),
            "stair_direction": args_cli.stair_direction,
            "difficulty": getattr(env_cfg, "difficulty", None),
            "command": [float(x) for x in args_cli.cmd.split()] if args_cli.cmd else None,
            "with_training_noise": bool(args_cli.eval_with_training_noise),
            "depth_noise": {
                "blur_sigma": float(args_cli.depth_blur_sigma),
                "additive_std": float(args_cli.depth_additive_noise_std) if args_cli.eval_with_training_noise else 0.0,
                "dropout_prob": float(args_cli.depth_dropout_prob) if args_cli.eval_with_training_noise else 0.0,
            },
            "proprioceptive_noise": (
                proprio_noise.cfg.to_dict() if proprio_noise is not None else None
            ),
            "num_envs": int(num_envs),
            "steps": int(max_steps),
            "warmup_steps": int(args_cli.eval_metrics_warmup_steps),
            "step_dt": float(env.unwrapped.step_dt),
            "horizon_seconds": float(horizon_seconds),
            "fall_events": int(fall_events),
            "fallen_envs": int(fallen_envs.sum().item()),
            "survival_fraction": float((~fallen_envs).float().mean().item()),
            "mean_first_fall_seconds": (
                float(observed_first_fall.mean().item()) if observed_first_fall.numel() else None
            ),
            "action_rms": math.sqrt(action_sq_sum / max(action_count, 1)),
            "action_abs_max": float(action_abs_max),
            "action_delta_rms": math.sqrt(action_delta_sq_sum / max(action_delta_count, 1)),
            "action_delta_abs_max": float(action_delta_abs_max),
            "linear_velocity_abs_error": (
                lin_vel_abs_error_sum / max(tracking_count, 1)
            ).cpu().tolist(),
            "yaw_rate_abs_error": float(yaw_rate_abs_error_sum / max(tracking_count, 1)),
            "metric_steps": int(metric_steps),
        }
        with open(output, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, sort_keys=True)
            file.write("\n")
        print(f"[INFO] Evaluation metrics saved: {output}")
        print(f"[INFO] Evaluation metrics: {json.dumps(metrics, sort_keys=True)}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train a depth-conditioned student with online DAgger supervision."""
    if args_cli.keyboard and not args_cli.eval:
        raise ValueError("--keyboard is only available with --eval.")
    if args_cli.show_depth and (not args_cli.eval or args_cli.headless):
        raise ValueError("--show_depth requires visible --eval (omit --headless).")
    if args_cli.obs_dump is not None and not args_cli.eval:
        raise ValueError("--obs_dump is only available with --eval.")
    if args_cli.stair_direction and (not args_cli.eval or args_cli.terrain != "stairs"):
        raise ValueError("--stair_direction requires --eval --terrain stairs.")
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # Same expert handling as train_dagger.py: the M1 TrueVel teacher needs the student
    # env's privileged "teacher_obs" group (sim base_lin_vel spliced into the obs).
    agent_cfg.obs_groups = {"policy": ["teacher_obs"], "critic": ["teacher_obs"]}

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.eval:
        env_cfg.disable_episode_timeout = True
        print("[INFO] Student play: 20-second episode timeout disabled; contact termination remains active.")
    env_cfg.terrain_type = args_cli.terrain
    if args_cli.difficulty is not None and not 0.0 <= args_cli.difficulty <= 1.0:
        raise ValueError(f"--difficulty must be in [0.0, 1.0], got {args_cli.difficulty}")
    env_cfg.difficulty = args_cli.difficulty
    env_cfg.rebuild_terrain()  # __post_init__ already ran at hydra parse; rebuild with the CLI terrain
    if args_cli.stair_direction:
        # The terrain origin is the central platform. An inverted pyramid starts
        # in the low center and climbs outward; a regular pyramid descends.
        key = "pyramid_stairs_inv" if args_cli.stair_direction == "up" else "pyramid_stairs"
        stair_cfg = env_cfg.terrain.terrain_generator.sub_terrains[key]
        stair_cfg.proportion = 1.0
        env_cfg.terrain.terrain_generator.sub_terrains = {key: stair_cfg}
        print(f"[INFO] Stair direction: {args_cli.stair_direction} ({key}).")
    if args_cli.rotate_video_env and not (args_cli.video and args_cli.dual_pane):
        raise ValueError("--rotate_video_env requires --video --dual_pane.")
    if args_cli.video and (args_cli.video_interval <= 0 or args_cli.video_length <= 0):
        raise ValueError("Video interval and length must be positive.")
    if args_cli.depth_history_length <= 0:
        raise ValueError("--depth_history_length must be positive.")
    if args_cli.depth_delay_frames < 0:
        raise ValueError("--depth_delay_frames must be non-negative.")
    if not math.isfinite(args_cli.depth_min_z) or not 0.0 <= args_cli.depth_min_z <= 2.0:
        raise ValueError("--depth_min_z must be finite and in [0, 2] metres.")
    if not getattr(env_cfg, "emit_teacher_obs", False) or getattr(env_cfg, "use_lin_vel_obs", True):
        raise ValueError("Use Locomotion-Go1-Rough-Vision-Tiled or -RayCaster for DAgger.")
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
    if args_cli.dagger_buffer_size <= 0 or args_cli.dagger_recent_buffer_size <= 0:
        raise ValueError("Both DAgger replay capacities must be positive.")
    if not 0.0 <= args_cli.dagger_recent_fraction <= 1.0:
        raise ValueError("--dagger_recent_fraction must be in [0, 1].")
    if not 0.0 <= args_cli.proprio_delay_prob <= 1.0:
        raise ValueError("--proprio_delay_prob must be in [0, 1].")
    if args_cli.dagger_consistency_weight < 0:
        raise ValueError("--dagger_consistency_weight must be non-negative.")
    if not 0 < args_cli.dagger_min_learning_rate <= args_cli.dagger_learning_rate:
        raise ValueError("Require 0 < --dagger_min_learning_rate <= --dagger_learning_rate.")
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
    if not math.isfinite(args_cli.depth_fps) or args_cli.depth_fps <= 0:
        raise ValueError("--depth_fps must be finite and positive.")
    env_cfg.depth_camera.update_period = 1.0 / args_cli.depth_fps

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.eval:
        # eval: load the student policy directly -- no teacher run is involved.
        eval_policy_path = _validate_eval_cli_args()
        log_dir = os.path.dirname(eval_policy_path)
    else:
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

    # where dagger videos are written (defaults to <teacher_run>/videos/dagger -- dagger_eval in
    # eval mode so the two never clobber each other's dual_step-0.mp4; override with
    # --video_output_dir so concurrent runs don't clobber each other's recordings).
    video_out_dir = (
        os.path.abspath(args_cli.video_output_dir)
        if args_cli.video_output_dir
        else os.path.join(log_dir, "videos", "dagger_eval" if args_cli.eval else "dagger")
    )

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
            "video_folder": video_out_dir,
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.eval:
        # Pure student rollout: no teacher, no replay, no gradient updates, no checkpoint saves.
        # __main__ closes simulation_app after main() returns, so returning here is safe.
        _run_student_eval(env, eval_policy_path=eval_policy_path, video_out_dir=video_out_dir)
        env.close()
        return

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
    num_fixed_command_envs = get_num_fixed_command_envs(env.unwrapped.cfg, num_envs)

    proprio_noise_cfg = ProprioNoiseCfg(
        angular_velocity_half_range=args_cli.proprio_gyro_noise_amplitude,
        projected_gravity_half_range=args_cli.proprio_gravity_noise_amplitude,
        joint_position_half_range=args_cli.proprio_joint_pos_noise_amplitude,
        joint_velocity_half_range=args_cli.proprio_joint_vel_noise_amplitude,
    )
    proprio_noise_cfg.validate()
    use_proprio_noise = not args_cli.disable_proprio_noise and not args_cli.eval
    proprio_noise = (
        ProprioceptiveNoise(num_envs, env_cfg.history_length, device, proprio_noise_cfg)
        if use_proprio_noise else None
    )

    history_capacity = args_cli.depth_history_length + args_cli.depth_delay_frames
    depth_history = current_depth_cpu.unsqueeze(0).repeat(history_capacity, 1, 1, 1, 1).contiguous()
    history_index = history_capacity - 1

    dagger_net = DaggerNet(
        vec_size=common_obs_size,
        output_size=action_size,
        depth_channels=depth_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        dagger_net.parameters(), lr=args_cli.dagger_learning_rate, weight_decay=0.0
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=_use_cuda_amp(device))
    loss_fn = torch.nn.MSELoss()
    replay_buffer = DaggerReplayBuffer(capacity=args_cli.dagger_buffer_size, reservoir=True)
    recent_replay_buffer = DaggerReplayBuffer(capacity=args_cli.dagger_recent_buffer_size)
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
        "depth_pipeline_version": PIPELINE_VERSION,
        "depth_fps": args_cli.depth_fps,
        "depth_delay_units": "control_steps",
        "depth_history_update_rate": 1.0 / env.unwrapped.step_dt,
        "depth_source_resolution": (480, 848),
        "depth_calibration_status": "nominal_unverified",
        "depth_mount_reference": {
            "bottom_screw_position_base": D435_BOTTOM_SCREW_POSITION_BASE,
            "bottom_screw_pitch_deg": D435_MOUNT_PITCH_DEG,
            "depth_origin_in_screw": D435_DEPTH_ORIGIN_IN_SCREW,
            "source": "RealSense official _d435.urdf.xacro",
        },
        "depth_min_z": args_cli.depth_min_z,
        "depth_camera": {
            "intrinsics": env.unwrapped._depth_camera.data.intrinsic_matrices[0].cpu().tolist(),
            "position_base": tuple(env_cfg.depth_camera.offset.pos),
            "quaternion_wxyz": tuple(env_cfg.depth_camera.offset.rot),
            "convention": env_cfg.depth_camera.offset.convention,
            "update_period": env_cfg.depth_camera.update_period,
            "sensor_update_period": env_cfg.depth_camera.update_period,
            "history_sampling": "every_control_step",
            "control_period": env.unwrapped.step_dt,
            "depth_type": "distance_to_image_plane",
        },
        "depth_delay_frames": args_cli.depth_delay_frames,
        "depth_sensor_noise": {
            "blur_sigma": args_cli.depth_blur_sigma,
            "additive_noise_std": args_cli.depth_additive_noise_std,
            "dropout_prob": args_cli.depth_dropout_prob,
        },
        "proprioceptive_noise": {
            "enabled_during_training": use_proprio_noise,
            "model": "per_step_zero_mean_uniform",
            "noisy_terms": ("angular_velocity", "projected_gravity", "joint_position", "joint_velocity"),
            "clean_terms": ("command", "previous_raw_action"),
            **proprio_noise_cfg.to_dict(),
        },
        "depth_image_size": tuple(current_depth_cpu.shape[-2:]),
        "depth_channels": depth_channels,
        "common_obs_size": common_obs_size,
        "action_size": action_size,
        "num_envs": num_envs,
        "command_a": tuple(env_cfg.command_a),
        "fixed_command_zero_ratio": env_cfg.fixed_command_zero_ratio,
        "num_fixed_command_envs": num_fixed_command_envs,
        "command_curriculum": {
            "fixed_zero_assignment": "randomized_per_episode",
            "fixed_zero_ratio": float(env_cfg.fixed_command_zero_ratio),
            "transition_ratio": float(getattr(env_cfg, "command_transition_ratio", 0.0)),
            "transition_move_duration_s": tuple(
                getattr(env_cfg, "command_transition_move_duration_s", ())
            ),
            "transition_stop_duration_s": tuple(
                getattr(env_cfg, "command_transition_stop_duration_s", ())
            ),
        },
        "dagger_amp": _use_cuda_amp(device),
        "dagger_train_micro_batch_size": args_cli.dagger_train_micro_batch_size,
        "dagger_inference_batch_size": args_cli.dagger_inference_batch_size,
        "dagger_replay": {
            "long_term_capacity": args_cli.dagger_buffer_size,
            "long_term_retention": "reservoir",
            "recent_capacity": args_cli.dagger_recent_buffer_size,
            "recent_fraction": args_cli.dagger_recent_fraction,
        },
        "dagger_consistency_weight": args_cli.dagger_consistency_weight,
        "proprio_delay_steps": (0, 1),
        "proprio_delay_probability": args_cli.proprio_delay_prob,
        "policy_action_delay_steps": (0, int(env_cfg.policy_action_delay_max_steps)),
        "optimizer": {
            "name": "AdamW", "weight_decay": 0.0,
            "initial_lr": args_cli.dagger_learning_rate,
            "minimum_lr": args_cli.dagger_min_learning_rate,
            "linear_decay_steps": args_cli.dagger_lr_decay_steps,
        },
    }

    print(
        "[INFO] Starting online DAgger training "
        f"(num_envs={num_envs}, depth_history={args_cli.depth_history_length}, "
        f"depth_fps={args_cli.depth_fps}, depth_delay={args_cli.depth_delay_frames} control step(s), depth_blur_sigma={args_cli.depth_blur_sigma}, "
        f"depth_additive_std={args_cli.depth_additive_noise_std}, depth_dropout={args_cli.depth_dropout_prob}, "
        f"proprio_noise={'on' if use_proprio_noise else 'off'}, "
        f"proprio_uniform_half_range=[gyro={proprio_noise_cfg.angular_velocity_half_range}, "
        f"gravity={proprio_noise_cfg.projected_gravity_half_range}, q={proprio_noise_cfg.joint_position_half_range}, "
        f"dq={proprio_noise_cfg.joint_velocity_half_range}], "
        f"reservoir={args_cli.dagger_buffer_size}, recent={args_cli.dagger_recent_buffer_size}, "
        f"recent_fraction={args_cli.dagger_recent_fraction}, batch={args_cli.dagger_batch_size}, "
        f"train_micro_batch={args_cli.dagger_train_micro_batch_size}, "
        f"inference_batch={args_cli.dagger_inference_batch_size}, amp={_use_cuda_amp(device)}, "
        f"command_a={env_cfg.command_a}, fixed_command_envs={num_fixed_command_envs}, "
        f"transition_envs={round(num_envs * float(getattr(env_cfg, 'command_transition_ratio', 0.0)))}, "
        f"transition_move_s={getattr(env_cfg, 'command_transition_move_duration_s', None)}, "
        f"transition_stop_s={getattr(env_cfg, 'command_transition_stop_duration_s', None)})."
    )

    # camera-follow setup for env 0 (video recording view)
    robot = env.unwrapped._robot
    eye_offset_b = torch.tensor([-args_cli.cam_distance, args_cli.cam_side, args_cli.cam_height], device=device)
    tgt_offset_b = torch.tensor([0.6, 0.0, 0.3], device=device)

    # dual-pane video recorder state
    dual_writer = None
    dual_frames_left = 0
    video_clip_index = 0
    follow_env = _effective_follow_env(args_cli.follow_env, env, args_cli.cmd is not None)

    step = 0
    updates = 0
    dones_cpu = torch.zeros(num_envs, dtype=torch.bool, device="cpu")
    proprio_delay_mask = torch.rand(num_envs, device=device) < args_cli.proprio_delay_prob

    # resume from a previous run if requested
    if args_cli.resume_from:
        resume_ckpt_path = os.path.abspath(args_cli.resume_from)
        if not os.path.exists(resume_ckpt_path):
            raise FileNotFoundError(f"--resume_from path not found: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        _check_student_ckpt_compat(ckpt, dagger_net, tuple(current_depth_cpu.shape[-2:]))
        dagger_net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["weight_decay"] = 0.0
        step = int(ckpt["step"])
        updates = int(ckpt["updates"])
        print(
            f"[INFO] Resumed DAgger training from {resume_ckpt_path} "
            f"(step={step}, updates={updates}). Replay buffer starts empty and refills in ~32 steps."
        )

    while simulation_app.is_running() and (
        args_cli.max_training_steps is None or step < args_cli.max_training_steps
    ):
        clean_common_obs = delay_sensor_terms(obs["common"], proprio_delay_mask)
        common_obs = (
            proprio_noise.update(clean_common_obs, dones_cpu)
            if proprio_noise is not None else clean_common_obs
        )
        # feed the student depth delayed by --depth_delay_frames (capture->inference latency)
        newest_index = _delayed_newest_index(history_index, history_capacity)

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
            replay_common_cpu = clean_common_obs.detach().to(device="cpu", dtype=torch.float32)
            replay_expert_cpu = expert_actions.detach().to(device="cpu", dtype=torch.float32)
        else:
            replay_env_indices_gpu = replay_env_indices_cpu.to(device=device)
            replay_common_cpu = clean_common_obs.index_select(0, replay_env_indices_gpu).detach().to(
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
        recent_replay_buffer.add_batch(
            depth_sequences=replay_depth_cpu,
            common_obs=replay_common_cpu,
            expert_actions=replay_expert_cpu,
        )
        del replay_depth_cpu, replay_common_cpu, replay_expert_cpu

        obs, _, dones, _ = env.step(actions)
        dones_cpu = dones.bool().to(device="cpu")
        if dones_cpu.any():
            reset_ids = dones_cpu.nonzero(as_tuple=False).flatten().to(device=device)
            proprio_delay_mask[reset_ids] = (
                torch.rand(reset_ids.numel(), device=device) < args_cli.proprio_delay_prob
            )
        # Match the original DAgger: read sensor data and append a snapshot each control step.
        current_depth_cpu = _sanitize_depth_data(env).detach().to(device="cpu", dtype=torch.float16)
        history_index = (history_index + 1) % history_capacity
        depth_history[history_index].copy_(current_depth_cpu)
        if dones_cpu.any():
            depth_history[:, dones_cpu] = current_depth_cpu[dones_cpu].unsqueeze(0).expand(
                history_capacity, -1, -1, -1, -1
            )

        if (args_cli.rotate_video_env and dual_frames_left == 0
                and step % args_cli.video_interval == 0):
            follow_env = _video_follow_for_clip(env, video_clip_index)
            video_clip_index += 1

        # Follow one subject throughout each video clip.
        with torch.inference_mode():
            root_pos = robot.data.root_state_w[follow_env, :3]
            yaw_q = math_utils.yaw_quat(robot.data.root_quat_w[follow_env])
            eye = root_pos + math_utils.quat_apply(yaw_q, eye_offset_b)
            tgt = root_pos + math_utils.quat_apply(yaw_q, tgt_offset_b)
        set_camera_view(eye.cpu().numpy(), tgt.cpu().numpy(), "/OmniverseKit_Persp")

        # dual-pane recording: left = sim view, right = live depth
        if args_cli.video and args_cli.dual_pane:
            if dual_frames_left == 0 and step % args_cli.video_interval == 0:
                dual_frames_left = args_cli.video_length
                dual_suffix = f"_env-{follow_env}" if args_cli.rotate_video_env else ""
                dual_fname = os.path.join(video_out_dir, f"dual_step-{step}{dual_suffix}.mp4")
                os.makedirs(os.path.dirname(dual_fname), exist_ok=True)
            if dual_frames_left > 0:
                frame = _dual_pane_frame(env, env_index=follow_env,
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
                recent_count = int(round(args_cli.dagger_batch_size * args_cli.dagger_recent_fraction))
                recent_count = min(recent_count, args_cli.dagger_batch_size)
                long_count = args_cli.dagger_batch_size - recent_count
                pieces = []
                if long_count:
                    pieces.append(replay_buffer.sample(batch_size=long_count, device="cpu"))
                if recent_count:
                    pieces.append(recent_replay_buffer.sample(batch_size=recent_count, device="cpu"))
                batch_depth = torch.cat([piece[0] for piece in pieces], dim=0)
                batch_common = torch.cat([piece[1] for piece in pieces], dim=0)
                batch_expert = torch.cat([piece[2] for piece in pieces], dim=0)

                decay_progress = min(1.0, step / max(1, args_cli.dagger_lr_decay_steps))
                current_lr = (
                    args_cli.dagger_learning_rate
                    + decay_progress * (args_cli.dagger_min_learning_rate - args_cli.dagger_learning_rate)
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for start in range(0, args_cli.dagger_batch_size, args_cli.dagger_train_micro_batch_size):
                    stop = min(start + args_cli.dagger_train_micro_batch_size, args_cli.dagger_batch_size)
                    micro_batch_size = stop - start

                    micro_depth = batch_depth[start:stop].to(device=device, dtype=torch.float32, non_blocking=True)
                    micro_common_clean = batch_common[start:stop].to(device=device, non_blocking=True)
                    micro_expert = batch_expert[start:stop].to(device=device, non_blocking=True)

                    if proprio_noise is not None:
                        micro_common = corrupt_common_observation(micro_common_clean, proprio_noise_cfg)
                        micro_common_second = corrupt_common_observation(micro_common_clean, proprio_noise_cfg)
                    else:
                        micro_common = micro_common_clean
                        micro_common_second = micro_common_clean

                    with _autocast_context(device):
                        predicted_actions, _ = dagger_net(micro_depth, micro_common, hidden=None)
                        if args_cli.dagger_consistency_weight > 0:
                            second_actions, _ = dagger_net(micro_depth, micro_common_second, hidden=None)
                            bc_loss = 0.5 * (
                                loss_fn(predicted_actions.float(), micro_expert)
                                + loss_fn(second_actions.float(), micro_expert)
                            )
                            consistency_loss = loss_fn(predicted_actions.float(), second_actions.float())
                            micro_loss = bc_loss + args_cli.dagger_consistency_weight * consistency_loss
                        else:
                            second_actions = None
                            consistency_loss = None
                            bc_loss = loss_fn(predicted_actions.float(), micro_expert)
                            micro_loss = bc_loss

                    weighted_loss = micro_loss * (micro_batch_size / args_cli.dagger_batch_size)
                    if grad_scaler.is_enabled():
                        grad_scaler.scale(weighted_loss).backward()
                    else:
                        weighted_loss.backward()

                    total_loss += micro_loss.item() * micro_batch_size
                    del micro_depth, micro_common_clean, micro_common, micro_common_second
                    del micro_expert, predicted_actions, second_actions, consistency_loss, bc_loss, micro_loss, weighted_loss

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
                f"[INFO] step={step} updates={updates} reservoir={len(replay_buffer)} "
                f"recent={len(recent_replay_buffer)} teacher_beta={beta:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} recent_total_loss={mean_loss:.5f}"
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
