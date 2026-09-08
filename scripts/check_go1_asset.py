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
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
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

    # LIMIT VERIFICATION: actuator limits are intentionally not duplicated in
    # go1_asset.py. They must propagate from Unitree's URDF through the importer.
    expected_limits = {
        "hip": (-0.863, 0.863, 30.1, 23.7),
        "thigh": (-0.686, 4.501, 30.1, 23.7),
        "calf": (-2.818, -0.888, 20.06, 35.55),
    }
    limit_errors = []
    print("=" * 60)
    print("LIMIT VERIFICATION (position / velocity / effort from official URDF)")
    for joint_id, joint_name in enumerate(robot.data.joint_names):
        joint_type = next(name for name in expected_limits if f"_{name}_joint" in joint_name)
        expected_lower, expected_upper, expected_velocity, _ = expected_limits[joint_type]
        actual_position = robot.data.joint_pos_limits[0, joint_id]
        actual_velocity = robot.data.joint_vel_limits[0, joint_id]
        actual_soft_velocity = robot.data.soft_joint_vel_limits[0, joint_id]
        print(
            f"  {joint_name:16s} pos=[{actual_position[0]:+.3f}, {actual_position[1]:+.3f}]"
            f" vel={actual_velocity:.2f} soft_vel={actual_soft_velocity:.2f}"
        )
        expected_position = torch.tensor(
            [expected_lower, expected_upper], device=actual_position.device, dtype=actual_position.dtype
        )
        if not torch.allclose(actual_position, expected_position, atol=1.0e-4, rtol=0.0):
            limit_errors.append(f"{joint_name} position")
        if not torch.isclose(
            actual_velocity, actual_velocity.new_tensor(expected_velocity), atol=1.0e-4, rtol=0.0
        ):
            limit_errors.append(f"{joint_name} simulation velocity")
        if not torch.isclose(
            actual_soft_velocity, actual_soft_velocity.new_tensor(expected_velocity), atol=1.0e-4, rtol=0.0
        ):
            limit_errors.append(f"{joint_name} actuator velocity")

    for actuator_name, actuator in robot.actuators.items():
        expected_effort = expected_limits[actuator_name][3]
        print(f"  actuator {actuator_name:6s} effort={actuator.effort_limit[0].tolist()}")
        if not torch.allclose(
            actuator.effort_limit,
            torch.full_like(actuator.effort_limit, expected_effort),
            atol=1.0e-4,
            rtol=0.0,
        ):
            limit_errors.append(f"{actuator_name} actuator effort")

    # MERGE VERIFICATION: with merge_fixed_joints=True, the *_foot_fixed joints carry
    # dont_collapse=true, so the 0.06 kg foot links must survive as distinct bodies and
    # keep their frame at the leg tip. A buggy merge would (a) drop them (find_bodies
    # returns empty), (b) fold their mass into the calf, or (c) leave a stale/offset
    # frame so body_pos_w[foot] points at the calf.
    print("=" * 60)
    print("MERGE VERIFICATION (fixed joints / foot links / masses)")
    feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    calfs = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    foot_ix, foot_names = robot.find_bodies(feet, preserve_order=True)
    calf_ix, calf_names = robot.find_bodies(calfs, preserve_order=True)
    foot_pos = robot.data.body_pos_w[0, foot_ix]
    calf_pos = robot.data.body_pos_w[0, calf_ix]
    print("  foot world pos:")
    for n, p in zip(foot_names, foot_pos):
        print(f"    {n:8s} xyz=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
    print("  calf world pos:")
    for n, p in zip(calf_names, calf_pos):
        print(f"    {n:8s} xyz=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
    dz = foot_pos[:, 2] - calf_pos[:, 2]
    print(f"  foot_z - calf_z = {[f'{d:+.3f}' for d in dz]}   (all must be negative: feet below calf)")
    total_mass = robot.data.default_mass[0].sum().item()
    print(f"  total body mass = {total_mass:.4f} kg   (official URDF sum = 13.1005 -> merge must conserve mass)")
    if "base" in robot.data.body_names:
        bix = robot.data.body_names.index("base")
        print(
            f"  'base' (merged trunk) mass = {robot.data.default_mass[0, bix].item():.4f} kg"
            "   (expect ~5.5611: trunk + imu + four hip rotors + auxiliary sensors)"
        )

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
    if not missing_bodies and not missing_joints and not limit_errors and robot.num_joints == 12:
        print("PASS: all required bodies/joints and official limits found, 12 actuated joints.")
    else:
        print(
            f"FAIL: missing bodies={missing_bodies}, missing joints={missing_joints},"
            f" limit errors={limit_errors}, num_joints={robot.num_joints}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
