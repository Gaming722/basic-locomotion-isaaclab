"""Configuration for the Unitree GO1 quadruped, loaded directly from URDF.

The GO1 is a 12-DOF quadruped (4 legs x hip/thigh/calf revolute joints), driven by
delayed PD actuators (``DelayedPDActuatorCfg``).  No actuator identification is
available yet, so the motors use nominal PD gains, the effort/velocity limits parsed
from the URDF (``<limit>`` tags), and a small placeholder command delay (0-2 physics
steps, randomized per-env at reset) to model real control latency for sim-to-real.
Once real friction / encoder-bias / delay parameters are identified, the actuator group
can be upgraded to ``IdentifiedActuatorElectricCfg`` or ``PaceDCMotorCfg`` without
touching the rest of the config.

URDF import notes
-----------------
* ``merge_fixed_joints=True`` (default) consolidates the mass-less dummy links (``base``
  frame, ``imu_link``, cameras, ultraSound) into the ``trunk`` body, which becomes the
  ``base`` rigid body (~4.8 kg).  The four ``*_foot`` links carry mass (0.06 kg) and are
  preserved as distinct bodies, so ``find_bodies(["FL_foot", ...])`` still works.
  Using ``merge_fixed_joints=False`` leaves the mass-less ``base`` as a zero-mass floating
  root, which PhysX cannot initialize (it hangs) -- so it must stay True.
* The actuator groups are named ``hip``/``thigh``/``calf`` to match the asymmetric-critic
  privileged observation in ``custom_observations.py``.
* Default standing pose matches the Go1 Mujoco keyframe "home":
  hip L/R = 0, thigh 0.9, calf -1.8. Spawn base z = 0.4 (kept slightly high so the
  feet drop into the terrain instead of spawning inside it); the *standing* height
  target is desired_base_height = 0.29 in the env cfg.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from basic_locomotion_isaaclab.assets import ISAAC_ASSET_DIR

GO1_ROOT = os.path.join(ISAAC_ASSET_DIR, "go1_asset")
GO1_URDF = os.path.join(GO1_ROOT, "urdf", "go1.urdf")

# Nominal PD gains aligned to mujoco_playground GO1 (Kp=35.0, Kd=0.5).
# Damping was previously raised 0.6 -> 2.0 (Aliengo-style) to suppress a degenerate
# "wheelbarrow" gait; aligning to mj's soft 0.5 undoes that and requires a fresh run.
GO1_STIFFNESS = 35.0  # N m / rad
GO1_DAMPING = 0.5  # N m s / rad
# Motor rotor inertia added to joint-space inertia (matches Aliengo's identified DCMotor).
# Not yet identified for GO1; keeps joint response from being unrealistically light.
GO1_ARMATURE = 0.01  # kg m^2
# Placeholder command latency (physics steps), randomized per-env at reset. Mirrors
# Go2's PaceDCMotor max_delay=2 and the m1-perceptive go1 (max_delay=2).
GO1_MIN_DELAY = 0
GO1_MAX_DELAY = 2


def _assert_urdf_exists() -> str:
    if not os.path.exists(GO1_URDF):
        raise FileNotFoundError(f"GO1 URDF not found: {GO1_URDF}")
    return GO1_URDF


GO1_HIP_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_hip_joint"],
    effort_limit=23.7,
    velocity_limit=30.1,
    stiffness=GO1_STIFFNESS,
    damping=GO1_DAMPING,
    armature=GO1_ARMATURE,
    min_delay=GO1_MIN_DELAY,
    max_delay=GO1_MAX_DELAY,
)

GO1_THIGH_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_thigh_joint"],
    effort_limit=23.7,
    velocity_limit=30.1,
    stiffness=GO1_STIFFNESS,
    damping=GO1_DAMPING,
    armature=GO1_ARMATURE,
    min_delay=GO1_MIN_DELAY,
    max_delay=GO1_MAX_DELAY,
)

GO1_CALF_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_calf_joint"],
    effort_limit=35.55,
    velocity_limit=20.06,
    stiffness=GO1_STIFFNESS,
    damping=GO1_DAMPING,
    armature=GO1_ARMATURE,
    min_delay=GO1_MIN_DELAY,
    max_delay=GO1_MAX_DELAY,
)

GO1_CFG = ArticulationCfg(
    prim_path=None,
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_assert_urdf_exists(),
        fix_base=False,
        # Keep True: merges the mass-less `base` frame into `trunk` (the real body) while
        # preserving the mass-bearing `*_foot` links as separate bodies.
        merge_fixed_joints=True,
        make_instanceable=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="none",
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Disable self-collision: the URDF's thigh/calf collision boxes overlap during
            # normal leg motion and produce spurious contacts (legged_gym also disables it).
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            ".*_thigh_joint": 0.9,
            ".*_calf_joint": -1.8,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"hip": GO1_HIP_ACTUATOR_CFG, "thigh": GO1_THIGH_ACTUATOR_CFG, "calf": GO1_CALF_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
