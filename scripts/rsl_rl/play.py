# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
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
    "--terrain",
    type=str,
    default=None,
    help="Single-type play terrain (rough|stairs|slope|flat). Only applied if the env cfg has a "
         "rebuild_terrain() method.",
)
parser.add_argument(
    "--difficulty",
    type=float,
    default=None,
    help="Fixed terrain difficulty 0-1 (e.g. --terrain stairs --difficulty 0.467 -> 12 cm steps; "
         "0.667 -> 15 cm).",
)
parser.add_argument(
    "--cmd",
    type=str,
    default=None,
    help="Fixed velocity command 'vx vy wz' for all envs (overrides the env's random command "
         "generator, e.g. --cmd \"0.5 0 0\" for constant forward).",
)
parser.add_argument(
    "--visualize_edge_map",
    action="store_true",
    default=False,
    help="Visualize the feet_edge reward edge map in the viewport (black = edge cells, "
         "white = feasible/flat cells). Only applies to envs that set use_vision=True "
         "(rough-vision tasks with an edge_height_scanner).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
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

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
import basic_locomotion_isaaclab.tasks  # noqa: F401


def _install_env_switch(env):
    """Register keyboard shortcuts to switch which env the viewport camera follows.

    E / Q = next / previous env, 1-9 = jump to env 0-8. Re-centres the camera on the
    selected env via viewport_camera_controller.set_view_env_index. Only active when a
    display is present (local play); headless runs just warn.
    """
    env_unwrapped = env.unwrapped
    controller = getattr(env_unwrapped, "viewport_camera_controller", None)
    if controller is None:
        print("[INFO] env-switch keys disabled (env has no viewport_camera_controller).")
        return
    try:
        from omni.appwindow import get_default_app_window
        import carb

        num_envs = env_unwrapped.num_envs
        _input_iface = carb.input.acquire_input_interface()
        _keyboard = get_default_app_window().get_keyboard()

        def _on_key(event, *_):
            if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                return
            name = event.input.name
            cur = controller.cfg.env_index
            if name == "E":
                controller.set_view_env_index((cur + 1) % num_envs)
            elif name == "Q":
                controller.set_view_env_index((cur - 1) % num_envs)
            elif name.isdigit() and 1 <= int(name) <= 9:
                controller.set_view_env_index(int(name) - 1)
            else:
                return
            print(f"[INFO] camera -> env {controller.cfg.env_index}")

        _input_iface.subscribe_to_keyboard_events(_keyboard, _on_key)
        print("[INFO] env-switch keys on: E/Q = next/prev env, 1-9 = jump to env N-1.")
    except Exception as e:  # headless / no window
        print(f"[WARN] env-switch keys unavailable ({e}).")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # clamp the viewer env_index to the actual number of envs (config may default to 500)
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.env_index = max(0, min(env_cfg.viewer.env_index, env_cfg.scene.num_envs - 1))

    # apply a single-type / fixed-difficulty play terrain if the env supports it
    if args_cli.terrain is not None and hasattr(env_cfg, "rebuild_terrain"):
        env_cfg.terrain_type = args_cli.terrain
        if args_cli.difficulty is not None:
            env_cfg.difficulty = args_cli.difficulty
        env_cfg.rebuild_terrain()
        print(f"[INFO] Play terrain: {env_cfg.terrain_type} difficulty={env_cfg.difficulty}")

    # visualize the feet_edge reward edge map (black = edge, white = feasible) if requested
    if args_cli.visualize_edge_map:
        if hasattr(env_cfg, "visualize_edge_map"):
            env_cfg.visualize_edge_map = True
            print("[INFO] Visualizing feet_edge reward edge map (black=edges, white=feasible).")
        else:
            print("[WARN] --visualize_edge_map ignored: env cfg has no visualize_edge_map field.")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

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
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # interactive viewport camera: switch which env is followed (E/Q, 1-9)
    _install_env_switch(env)

    # fixed velocity command for all envs if requested (replaces the random generator)
    if args_cli.cmd:
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
            print(f"[WARN] --cmd ignored: env has no _commands buffer.")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if args_cli.cmd and hasattr(env.unwrapped, "_commands"):
        print(f"[INFO] actual commands[:3] = {env.unwrapped._commands[:3].cpu().numpy().tolist()}")
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
