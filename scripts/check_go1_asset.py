"""Smoke-test the GO1 asset: spawn it headless and print body/joint structure.

Run with:
    python scripts/check_go1_asset.py
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

    # Spawn the articulation inside the initial stage, then play the simulator so the
    # asset's PhysX handles get initialized (mirrors DirectRLEnv's scene setup order).
    print("STEP: spawning articulation...", flush=True)
    with use_stage(sim.get_initial_stage()):
        robot = Articulation(GO1_CFG.replace(prim_path="/World/go1"))
    # ArticulationRootAPI lives on the root *link* prim (base), not the spawn prim.
    import omni.usd
    from pxr import PhysxSchema
    stage = omni.usd.get_context().get_stage()
    api = PhysxSchema.PhysxArticulationAPI(stage.GetPrimAtPath("/World/go1/base"))
    print(f"enabled_self_collisions: {api.GetEnabledSelfCollisionsAttr().Get()}", flush=True)
    print("STEP: sim.reset()...", flush=True)
    sim.reset()
    print("STEP: robot.reset()...", flush=True)
    robot.reset()
    print("STEP: sim.step()...", flush=True)
    sim.step()
    print("STEP: done stepping", flush=True)

    print("=" * 60)
    print(f"num_bodies : {robot.num_bodies}")
    print(f"num_joints : {robot.num_joints}")
    print("-" * 60)
    print("body names:")
    for i, name in enumerate(robot.data.body_names):
        print(f"  [{i:2d}] {name}  mass={robot.data.default_mass[0, i].item():.4f}")
    print("-" * 60)
    print("joint names:")
    for i, name in enumerate(robot.data.joint_names):
        print(f"  [{i:2d}] {name}")
    print("-" * 60)
    print(f"default_joint_pos (env 0): {robot.data.default_joint_pos[0].tolist()}")

    # Verify the names the locomotion framework depends on.
    print("=" * 60)
    required_bodies = ["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot",
                       "FL_hip", "FR_hip", "RL_hip", "RR_hip",
                       "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
                       "FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    missing_bodies = []
    for name in required_bodies:
        _, names = robot.find_bodies(name)
        if len(names) == 0:
            missing_bodies.append(name)
        else:
            print(f"  body  '{name}' -> {names}")

    required_joints = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
                       "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
                       "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]
    missing_joints = []
    for name in required_joints:
        _, names = robot.find_joints(name)
        if len(names) == 0:
            missing_joints.append(name)
        else:
            print(f"  joint '{name}' -> {names}")

    print("=" * 60)
    if not missing_bodies and not missing_joints and robot.num_joints == 12:
        print("PASS: all required bodies/joints found, 12 actuated joints.")
    else:
        print(f"FAIL: missing bodies={missing_bodies}, missing joints={missing_joints}, num_joints={robot.num_joints}")


if __name__ == "__main__":
    main()
    simulation_app.close()
