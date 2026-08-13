"""Compute the default-pose lateral foot-vs-hip offset for the GO1.

Prints the heading-frame y positions of the feet and hips in the default standing
pose, so the ``desired_hip_offset`` value for the feet-to-hip reward can be set to the
natural stance offset.
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.utils.stage import use_stage

from basic_locomotion_isaaclab.assets.go1_asset import GO1_CFG


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0))
    with use_stage(sim.get_initial_stage()):
        robot = Articulation(GO1_CFG.replace(prim_path="/World/go1"))
    sim.reset()
    robot.reset()
    sim.step()

    feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    hips = ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]
    feet_ids, _ = robot.find_bodies(feet, preserve_order=True)
    hip_ids, _ = robot.find_bodies(hips, preserve_order=True)

    foot_pos = robot.data.body_pos_w[0, feet_ids, :]
    hip_pos = robot.data.body_pos_w[0, hip_ids, :]

    print("=" * 70)
    print(f"{'leg':<4} {'foot_y':>10} {'hip_y':>10} {'foot_y-hip_y':>12} {'hip_y-foot_y':>12}")
    for i, (f, h) in enumerate(zip(feet, hips)):
        fy = foot_pos[i, 1].item()
        hy = hip_pos[i, 1].item()
        print(f"{f[:-5]:<4} {fy:>10.4f} {hy:>10.4f} {fy - hy:>12.4f} {hy - fy:>12.4f}")

    # desired_hip_offset convention: FL/RL negative, FR/RR positive
    foot_minus_hip = foot_pos[:, 1] - hip_pos[:, 1]
    off = (foot_minus_hip[0] - foot_minus_hip[1]) / 2.0  # left minus right / 2
    print("=" * 70)
    print(f"suggested desired_hip_offset = {off.abs().item():.4f}  (FL/RL outboard - FR/RR outboard)/2")


if __name__ == "__main__":
    main()
    simulation_app.close()
