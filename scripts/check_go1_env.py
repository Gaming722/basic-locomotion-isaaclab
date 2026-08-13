"""Smoke-test the Go1 locomotion environment: instantiate and step it headless."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import basic_locomotion_isaaclab.tasks  # noqa: F401  (registers the Go1 envs)
from basic_locomotion_isaaclab.tasks.locomotion.go1_env_cfg import Go1FlatEnvCfg


def main():
    env_cfg = Go1FlatEnvCfg()
    env_cfg.scene.num_envs = 4

    env = gym.make("Locomotion-Go1-Flat", cfg=env_cfg)
    print(f"env created: {env.unwrapped.__class__.__name__}", flush=True)

    obs, _ = env.reset()
    print(f"obs type: {type(obs).__name__}", flush=True)
    if isinstance(obs, dict):
        print(f"obs keys: {list(obs.keys())}", flush=True)

    for i in range(5):
        action = torch.randn(env.action_space.shape, device=env.unwrapped.device)
        obs, reward, term, trunc, info = env.step(action)
    print("PASS: stepped 5 times", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
