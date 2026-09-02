import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    RayCasterCfg,
    MultiMeshRayCasterCfg,
    MultiMeshRayCasterCameraCfg,
    TiledCameraCfg,
    patterns,
)
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.envs import ViewerCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

from basic_locomotion_isaaclab.assets.go1_asset import GO1_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

import basic_locomotion_isaaclab.tasks.custom_events as custom_events
import basic_locomotion_isaaclab.tasks.custom_curriculums as custom_curriculums
from basic_locomotion_isaaclab.tasks.locomotion.unitree_l2_lidar import UnitreeL2PatternCfg

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    scale_all_link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )

    
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-5.0, 5.0),
        },
    )
    
    
    randomize_joint_parameters = EventTerm(
        func=custom_events.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                                   "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}},
    )




@configclass
class Go1FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    terrain_curriculum_move_up_error_percent = 25.0
    terrain_curriculum_move_down_error_percent = 50.0
    decimation = 5   # 5 physics substeps per control step (0.02 s / 0.004 s), matches mujoco_playground GO1
    action_scale = 0.5
    action_space = 12
    # DAgger student envs (Go1RoughVisionTiledEnvCfg) set this False: the real robot has
    # no base_lin_vel, so it is excluded from the obs and the history buffer dimension.
    use_lin_vel_obs = True
    observation_space = 3 if use_lin_vel_obs else 0  # base linear velocity
    observation_space += 3 # base angular velocity  
    observation_space += 3 # projected gravity in base frame
    observation_space += 3 # command (desired linear vel in x and y, desired yaw rate)
    observation_space += 12 # joint positions
    observation_space += 12 # joint velocities
    observation_space += 12 # last actions

    use_clock_signal = False   # no explicit clock signal in obs (mujoco_playground GO1 has none)
    if(use_clock_signal):
        observation_space += 4 # clock signal for periodic gait

    single_observation_space = observation_space # Usefull for concatenating history

    # observation history
    use_observation_history = True
    if(use_observation_history):
        history_length = 5
        observation_space *= history_length
    else:
        history_length = 1


    use_imu = False
    # an imu sensor in case we don't want any state estimator (for now we can't use sites from the xml)
    imu = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base", 
        offset=ImuCfg.OffsetCfg(
            pos=(-0.01592, -0.06659, -0.00617)
        ),
        debug_vis=False)

    
    use_concurrent_state_est = False
    if(use_concurrent_state_est):
        concurrent_state_est_network_type = "tcn" # "mlp" or "tcn"
        
        concurrent_state_est_output_space = 3 #lin_vel_b
        
        single_concurrent_state_est_observation_space = 3 # base linear acceleration
        single_concurrent_state_est_observation_space += 3 # base angular velocity  
        single_concurrent_state_est_observation_space += 3 # projected gravity in base frame
        single_concurrent_state_est_observation_space += 3 # command (desired linear vel in x and y, desired yaw rate)
        single_concurrent_state_est_observation_space += 12 # joint positions
        single_concurrent_state_est_observation_space += 12 # joint velocities
        single_concurrent_state_est_observation_space += 12 # last actions
        concurrent_state_est_history_length = 5 
        concurrent_state_est_observation_space = single_concurrent_state_est_observation_space*concurrent_state_est_history_length
        
        concurrent_state_est_batch_size = 512
        concurrent_state_est_train_epochs = 1000
        concurrent_state_est_lr = 1e-3
        concurrent_state_est_ep_saving_interval = 1000
        concurrent_state_est_ep_saving_start = 6000


    use_rma = False
    if(use_rma):
        rma_network_type = "mlp" # "mlp" or "tcn"
        rma_use_latent_space = True
        if(rma_use_latent_space):
            rma_latent_space = 8
            rma_latent_encoder_hidden_features = 128
            rma_latent_encoder_seed = 0
        
        rma_privileged_observation_space = 12 # P gain
        rma_privileged_observation_space += 12 # D gain
        rma_privileged_observation_space += 12 # static friction
        rma_privileged_observation_space += 12 # viscous friction

        rma_output_space = rma_latent_space if rma_use_latent_space else rma_privileged_observation_space
        observation_space += rma_output_space

        single_rma_observation_space = 3 # base linear acceleration
        single_rma_observation_space += 3 # base angular velocity  
        single_rma_observation_space += 3 # projected gravity in base frame
        single_rma_observation_space += 3 # command (desired linear vel in x and y, desired yaw rate)
        single_rma_observation_space += 12 # joint positions
        single_rma_observation_space += 12 # joint velocities
        single_rma_observation_space += 12 # last actions
        rma_history_length = 5
        rma_observation_space = single_rma_observation_space*rma_history_length
    
        rma_batch_size = 512
        rma_train_epochs = 1000
        rma_lr = 1e-3
        rma_ep_saving_interval = 1000
        rma_ep_saving_start = 6000


    # Base-centered height scanner for pose-related rewards and privileged observations.
    pose_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.6, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Template copied onto each foot link to measure the terrain immediately around that foot.
    foot_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.1, 0.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


    # asymmetric ppo
    use_asymmetric_ppo = True
    if(use_asymmetric_ppo):
        state_space = observation_space
        state_space += 12 #P gain
        state_space += 12 #D gain
        state_space += 2 #base pitch and height
        state_space += 3 #clean lin vel b
        state_space += 4 #contacts foot
        
        pattern_cfg = pose_height_scanner.pattern_cfg
        height_map_x_points = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
        height_map_y_points = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
        state_space += height_map_x_points * height_map_y_points
    else:
        state_space = 0


    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 250,   # 0.004 s physics dt (matches mujoco_playground GO1 sim_dt)
        render_interval=decimation,
        #disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=2**23,
            #gpu_max_rigid_patch_count= 5 * 2 ** 16,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # viewer: camera tracks a walking robot close-up (same as m1-perceptive-baseline).
    # origin_type="asset_root" makes the ViewportCameraController follow the robot root
    # every render step. envs 0-499 are the fixed command-zero envs (stand still), so
    # follow env 500 (the first that actually receives a velocity command).
    viewer: ViewerCfg = ViewerCfg(
        eye=(-3.0, 1.2, 1.8),
        lookat=(0.0, 0.0, 0.35),
        origin_type="asset_root",
        asset_name="robot",
        env_index=500,
        resolution=(1280, 720),
    )

    # events
    events: EventCfg = EventCfg()


    # Oracle teacher: no action/observation noise. Robustness (depth noise, action delay)
    # is introduced on the student side during distillation, not on the teacher.
    action_noise_model = None
    observation_noise_model = None

    # robot
    robot: ArticulationCfg = GO1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.004, track_air_time=True
    )

    desired_joints_order = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
                           'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',  
                           'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']

    
    # If you want to use AMP
    use_amp = False
    
    # Desired tracking variables
    desired_base_height = 0.29
    desired_feet_height = 0.1  # = mujoco_playground GO1 max_foot_height; reused as the mj feet-reward reference height


    # Desired clip actions
    desired_clip_actions = 3.0
    use_filter_actions = True
        

    # Tracking reward scale
    tracking_sigma = 0.25             # exp tracking bandwidth, = mujoco_playground GO1 reward_config.tracking_sigma
    command_a = [1.5, 0.8, 1.2]       # command amplitudes (lin_vel_x, lin_vel_y, yaw_rate), = mujoco_playground command_config.a
    lin_vel_reward_scale = 1.0 * 2.0       # = mj tracking_lin_vel
    yaw_rate_reward_scale = 0.5 * 2.0      # = mj tracking_ang_vel
    z_vel_reward_scale = -0.5         # = mj lin_vel_z
    ang_vel_reward_scale = -0.05      # = mj ang_vel_xy
    orientation_reward_scale = -3.0 * 1.5  # custom (mj orientation=-5.0, softened)
    height_reward_scale = 0.5         # custom (mj has no base-height term)
    

    # Joint reward scale
    joints_torque_reward_scale = -2.5e-6 
    joints_accel_reward_scale = -2.5e-7
    joints_energy_reward_scale = -1e-4
    joints_hip_position_reward_scale = 0.0  # disabled: superseded by the mj pose reward
    joints_thigh_position_reward_scale = 0.0  # disabled: superseded by the mj pose reward
    joints_calf_position_reward_scale = 0.0  # disabled: superseded by the mj pose reward
   
    
    # Undesired contacts reward scale
    undersired_contact_reward_scale = -1.0
    # GO1 penalizes calf contacts too, so the shin doesn't drag on the ground.
    # Other robots keep the default undesired-contact set (base + hips + thighs).
    undesired_contact_body_names = [
        "base",
        "FL_hip", "FR_hip", "RL_hip", "RR_hip",
        "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
        "FL_calf", "FR_calf", "RL_calf", "RR_calf",
    ]
    action_rate_reward_scale = -0.01
    action_smoothness_reward_scale = -0.001


    # Feet reward scale
    feet_air_time_reward_scale = 0.25 * 0.0  # aligned with Aliengo
    feet_air_time_variance_reward_scale = -1.0*0.0

    feet_height_clearance_aperiodic_reward_scale = 0.25*0.0  
    feet_height_clearance_periodic_reward_scale = 0.0  # disabled: superseded by the mj feet rewards (clearance/height)
    
    feet_height_clearance_mujoco_aperiodic_reward_scale = 0.25*0.0
    feet_height_clearance_mujoco_periodic_reward_scale = 0.25*0.0  # aligned with Aliengo
    
    feet_slide_reward_scale = -0.25*0.0
    
    feet_to_hip_distance_reward_scale = 1.5 * 0.5
    # This is used in loocmotion_env.py for the above reward
    desired_hip_offset = 0.08

    feet_edge_reward_scale = 0.0
    feet_edge_height_threshold = 0.05
    feet_edge_horizontal_radius = 0.10
    feet_edge_radius_px = 0
    visualize_edge_map = False

    feet_vertical_surface_contacts_reward_scale = -0.25  # aligned with Aliengo


    # Contact suggestion reward scale
    periodic_contact_suggestion_reward_scale = 0.0  # disabled: no explicit clock signal (mujoco_playground-style)
    # Desired step freq and duty factor (if periodic gait contact suggestion is used)
    desired_step_freq = 1.4
    desired_duty_factor = 0.65
    desired_phase_offset = [0.0, 0.5, 0.5, 0.0] #FL, FR, RL, RR

    stance_contact_suggestion_reward_scale = 0.25  # aligned with Aliengo


    # mujoco_playground GO1 joystick reference rewards (ported faithfully)
    pose_reward_scale = 0.5 * 2.0
    dof_pos_limits_reward_scale = -1.0
    termination_reward_scale = -1.0

    mj_feet_clearance_reward_scale = -1.0 * 0.05
    mj_feet_height_reward_scale = -0.2
    mj_feet_slip_reward_scale = -0.1
    mj_feet_air_time_reward_scale = 0.5 * 10.0 * 5.0



from .rough_terrains import COMMON_ROUGH_TERRAINS_CFG, GO1_ROUGH_TERRAINS_CFG, MJ_ROUGH_TERRAINS_CFG
@configclass
class Go1RoughBlindEnvCfg(Go1FlatEnvCfg):

    # Original GO1 curriculum terrain (curriculum=True, discrete obstacles + more
    # stairs). Switch to COMMON_ROUGH_TERRAINS_CFG to match Aliengo's terrain.
    ROUGH_TERRAINS_CFG = GO1_ROUGH_TERRAINS_CFG

    """Rough terrains configuration."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=10,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )


@configclass
class MjRoughEventCfg:
    """GO1 mj teacher events: physics-material setup only.

    mujoco_playground's go1 joystick trains with no perturbation (pert_config.enable
    = False) and no domain randomization (randomize.py is never wired into joystick),
    so the mj teacher drops the push / force-torque / mass / friction / gain events.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )


@configclass
class Go1RoughVisionEnvCfg(Go1RoughBlindEnvCfg):

    # Clip the per-step total reward to [0, 10000], mirroring mujoco_playground
    # joystick.py `reward = jp.clip(sum(...) * dt, 0.0, 10000.0)`. The 0 lower bound
    # caps the large negative spikes on fall steps -> lower return variance -> stable
    # PPO (this was the missing piece behind the mj-env late-stage collapse). Only envs
    # that set this field (vision + mj teacher) clip; blind/flat envs are unchanged.
    reward_clip: tuple = (0.0, 10000.0)

    def __post_init__(self) -> None:
        pattern_cfg = self.perceptive_height_scanner.pattern_cfg
        height_map_x_points = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
        height_map_y_points = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
        self.observation_space = self.observation_space + height_map_x_points * height_map_y_points

        self.feet_edge_reward_scale = -1.0 * 2.0

    # Play-time terrain override (mirrors Go1RoughVisionTiledEnvCfg): select a single
    # terrain type / difficulty, e.g. `--terrain stairs --difficulty 0.467` plays on 12 cm
    # stairs (step_height = 0.05 + difficulty*0.15). Only applied when the play script sets
    # these; teacher training keeps GO1_ROUGH_TERRAINS_CFG (curriculum=True) unchanged.
    terrain_type: str = "rough"
    difficulty: float | None = None

    def rebuild_terrain(self) -> None:
        if self.terrain_type == "rough":
            sub_terrains = GO1_ROUGH_TERRAINS_CFG.sub_terrains
        elif self.terrain_type == "flat":
            sub_terrains = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
        elif self.terrain_type == "stairs":
            sub_terrains = {
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.5, step_height_range=(0.05, 0.20), step_width=0.3,
                    platform_width=3.0, border_width=1.0, holes=False,
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.5, step_height_range=(0.05, 0.20), step_width=0.3,
                    platform_width=3.0, border_width=1.0, holes=False,
                ),
            }
        elif self.terrain_type == "slope":
            sub_terrains = {
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.5, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.5, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
                ),
            }
        else:
            raise ValueError(f"Unknown terrain_type: {self.terrain_type}")
        difficulty_range = (
            (self.difficulty, self.difficulty) if self.difficulty is not None else (0.0, 1.0)
        )
        self.terrain.terrain_generator = TerrainGeneratorCfg(
            curriculum=False,  # fixed terrain for play (training keeps curriculum=True)
            size=GO1_ROUGH_TERRAINS_CFG.size,
            border_width=GO1_ROUGH_TERRAINS_CFG.border_width,
            num_rows=GO1_ROUGH_TERRAINS_CFG.num_rows,
            num_cols=GO1_ROUGH_TERRAINS_CFG.num_cols,
            horizontal_scale=GO1_ROUGH_TERRAINS_CFG.horizontal_scale,
            vertical_scale=GO1_ROUGH_TERRAINS_CFG.vertical_scale,
            slope_threshold=GO1_ROUGH_TERRAINS_CFG.slope_threshold,
            use_cache=GO1_ROUGH_TERRAINS_CFG.use_cache,
            sub_terrains=sub_terrains,
            difficulty_range=difficulty_range,
        )

    use_vision = True

    # we add a height scanner for perceptive locomotion
    perceptive_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 0.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.6, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # we add a height scanner for feet edge reward
    edge_height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.8, 0.8]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    #camera_usd = CAMERA_USD_CFG
    use_depth_camera = False
    depth_camera = MultiMeshRayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.33, 0.0, 0.08), rot=(-0.405579, 0.579228, -0.579228, 0.405579)),
        mesh_prim_paths=[
            "/World/ground",
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/base/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/FL_.*/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/FR_.*/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/RL_.*/visuals"),
            #MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/RR_.*/visuals"),
        ],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=120,
            width=240,
        ),
        debug_vis=True,
    )

    use_unitree_l2_lidar = False
    unitree_l2_lidar = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=1 / 5.55,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.31, 0.0, 0.02)),
        ray_alignment="base",
        pattern_cfg=UnitreeL2PatternCfg(
            vertical_fov_orientation="down",
            enable_downsample=True,
            keep_ratio=0.1,
        ),
        debug_vis=True,
        mesh_prim_paths=[
            "/World/ground",
            #MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/base/visuals"),
            #MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/FL_.*/visuals"),
            #MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/FR_.*/visuals"),
            #MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/RL_.*/visuals"),
            #MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Robot/RR_.*/visuals"),
        ],
        max_distance=2.0,
    )


@configclass
class Go1RoughMjEnvCfg(Go1RoughVisionEnvCfg):
    """mujoco_playground GO1 rough-terrain teacher, with the perceptive obs.

    The policy obs keeps the base-centered heightmap scanner (``use_vision=True``
    inherited from Go1RoughVisionEnvCfg), like the existing Go1-Rough-Vision teacher.
    Only the *reward & training* setup mirrors mujoco_playground's go1 joystick:
    reward_config.scales, mj command sampling, fixed rough terrain (no curriculum),
    no perturbation / no DR, no action filter. Stays a noiseless Oracle teacher
    (robustness is introduced on the student side during distillation).

    Train with the existing RoughPPORunnerCfg (already aligned with mj rsl_rl_config).
    """

    def __post_init__(self) -> None:
        super().__post_init__()  # Go1RoughVisionEnvCfg: adds the heightmap dims to observation_space
        # Go1RoughVisionEnvCfg enables the feet-edge reward; mj has no such term.
        self.feet_edge_reward_scale = 0.0

    # ---- mj reward scales (joystick.py reward_config.scales) ----
    lin_vel_reward_scale = 1.0            # mj tracking_lin_vel
    yaw_rate_reward_scale = 0.5           # mj tracking_ang_vel
    mj_z_vel_reward_scale = -0.5          # mj lin_vel_z (world-frame, faithful form)
    mj_ang_vel_xy_reward_scale = -0.05    # mj ang_vel_xy (world-frame, faithful form)
    mj_orientation_reward_scale = -5.0    # mj orientation (keep the base level)
    pose_reward_scale = 0.5               # mj pose
    dof_pos_limits_reward_scale = -1.0    # mj dof_pos_limits
    termination_reward_scale = -1.0       # mj termination
    action_rate_reward_scale = -0.01      # mj action_rate
    mj_torques_reward_scale = -0.0002     # mj torques
    joints_energy_reward_scale = -0.001   # mj energy (same form as joints_energy_l1)
    stand_still_reward_scale = -1.0       # mj stand_still
    mj_feet_clearance_reward_scale = -2.0  # mj feet_clearance
    mj_feet_height_reward_scale = -0.2     # mj feet_height
    mj_feet_slip_reward_scale = -0.1       # mj feet_slip
    mj_feet_air_time_reward_scale = 0.1    # mj feet_air_time

    # ---- disable non-mj reward terms (replaced by the faithful mj forms above) ----
    z_vel_reward_scale = 0.0              # body-frame -> mj_z_vel_reward_scale
    ang_vel_reward_scale = 0.0            # body-frame -> mj_ang_vel_xy_reward_scale
    orientation_reward_scale = 0.0        # terrain-relative -> mj_orientation_reward_scale
    height_reward_scale = 0.0             # mj has no base-height term
    joints_torque_reward_scale = 0.0      # -> mj_torques_reward_scale
    joints_accel_reward_scale = 0.0       # mj has no joint-acceleration term
    action_smoothness_reward_scale = 0.0  # mj has no second-order action term
    feet_to_hip_distance_reward_scale = 0.0
    stance_contact_suggestion_reward_scale = 0.0
    feet_vertical_surface_contacts_reward_scale = 0.0
    # undersired_contact_reward_scale intentionally kept at -1.0: mujoco_playground is
    # feet-only collision (no calf drag); Isaac Lab uses full collision, so this is the
    # single non-mj term compensating for the physics difference.

    # ---- mj command sampling (~5 s exponential resample + b zeroing rule) ----
    mj_command_sampling = True
    command_b = [0.9, 0.25, 0.5]  # mj command_config.b

    # ---- mj action application: target = default + action*scale, no filter ----
    use_filter_actions = False

    # ---- mj termination: die only when the base flips past 90 deg (not on hip contact) ----
    mj_termination = True

    # ---- fixed rough terrain (mj: single 10x10 hfield, no curriculum) ----
    ROUGH_TERRAINS_CFG = MJ_ROUGH_TERRAINS_CFG
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # ---- no perturbation / no DR (mj joystick trains with neither) ----
    events = MjRoughEventCfg()


@configclass
class Go1RoughVisionTiledEnvCfg(Go1RoughVisionEnvCfg):
    """GO1 DAgger student env: TiledCamera depth, no base_lin_vel obs, teacher_obs emitted.

    Replaces the raycast depth with a GPU-rendered TiledCamera mounted at the URDF d435
    pose (d435_bottom_screw_frame), so the student sees depth like the real D435. The
    student obs ("common") excludes base_lin_vel (not available on the real robot); the
    privileged teacher obs ("teacher_obs") carries the sim base_lin_vel + heightmap so the
    expert can label student states during DAgger. Requires --enable_cameras at launch.
    """

    # Terrain composition for the Tiled env. "rough" = teacher's GO1_ROUGH mix;
    # "stairs" / "slope" / "flat" select a single terrain type (handy for recording,
    # e.g. --terrain stairs to capture stair climbing).
    terrain_type: str = "rough"

    # Optional fixed terrain difficulty (0.0-1.0). When set, every sub-terrain is
    # generated at exactly this difficulty (difficulty_range=(d, d)) instead of a
    # random sample, so e.g. --terrain stairs --difficulty 1.0 gives all-max-height
    # stairs. None (default) keeps the random per-sub-terrain sampling.
    difficulty: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rebuild_terrain()
        self.scene.num_envs = min(self.scene.num_envs, 46 * 46)
        if not self.use_lin_vel_obs:
            # base_lin_vel excluded -> single obs space and the history buffer shrink by 3.
            self.single_observation_space = self.single_observation_space - 3
            self.observation_space = self.single_observation_space * self.history_length

    def rebuild_terrain(self) -> None:
        """(Re)build the terrain generator from ``terrain_type``.

        Called from ``__post_init__`` and from the dagger script after ``terrain_type``
        is overridden at runtime (config __post_init__ runs at hydra parse time, before
        the dagger CLI can set terrain_type). Must rebuild the generator instead of
        mutating the shared GO1_ROUGH_TERRAINS_CFG (the teacher's Go1RoughBlindEnvCfg
        references it). curriculum stays False: with enforce_env_spacing (one robot fixed
        per sub-terrain) the curriculum's update_env_origins would move envs to random
        sub-terrains -> overlaps -> physics crashes (matches reference M1/Aliengo Tiled).
        """
        if self.terrain_type == "rough":
            sub_terrains = GO1_ROUGH_TERRAINS_CFG.sub_terrains
        elif self.terrain_type == "flat":
            sub_terrains = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
        elif self.terrain_type == "stairs":
            sub_terrains = {
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.5, step_height_range=(0.05, 0.20), step_width=0.3,
                    platform_width=3.0, border_width=1.0, holes=False,
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.5, step_height_range=(0.05, 0.20), step_width=0.3,
                    platform_width=3.0, border_width=1.0, holes=False,
                ),
            }
        elif self.terrain_type == "slope":
            sub_terrains = {
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.5, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.5, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
                ),
            }
        else:
            raise ValueError(f"Unknown terrain_type: {self.terrain_type}")
        # a fixed difficulty pins every sub-terrain to the same value
        # (uniform(d,d) == d), so e.g. difficulty=1.0 renders all-max stairs.
        difficulty_range = (
            (self.difficulty, self.difficulty) if self.difficulty is not None else (0.0, 1.0)
        )
        self.terrain.terrain_generator = TerrainGeneratorCfg(
            curriculum=False,
            size=GO1_ROUGH_TERRAINS_CFG.size,
            border_width=GO1_ROUGH_TERRAINS_CFG.border_width,
            num_rows=46,
            num_cols=46,
            horizontal_scale=GO1_ROUGH_TERRAINS_CFG.horizontal_scale,
            vertical_scale=GO1_ROUGH_TERRAINS_CFG.vertical_scale,
            slope_threshold=GO1_ROUGH_TERRAINS_CFG.slope_threshold,
            use_cache=GO1_ROUGH_TERRAINS_CFG.use_cache,
            sub_terrains=sub_terrains,
            difficulty_range=difficulty_range,
        )

    use_lin_vel_obs = False       # student obs: no base_lin_vel (real robot has no odometry)
    emit_teacher_obs = True       # emit teacher_obs (sim base_lin_vel + heightmap) for the expert
    enforce_env_spacing = True    # one robot per sub-terrain so depth can't see neighbours
    # viewer env_index=500 assumes >=501 envs; use 0 so the Tiled env works at low num_envs.
    viewer: ViewerCfg = ViewerCfg(
        eye=(-3.0, 1.2, 1.8),
        lookat=(0.0, 0.0, 0.35),
        origin_type="asset_root",
        asset_name="robot",
        env_index=0,
        resolution=(1280, 720),
    )
    use_depth_camera = True
    visualize_camera_mount = False
    depth_camera = TiledCameraCfg(
        # Mount on base with the URDF d435_joint pose (0.23, 0, 0.10, 30 deg down).
        # The d435 link cannot survive the fixed-joint merge (absorbed into trunk), so
        # the camera pose is replicated via this offset instead of a dedicated link.
        prim_path="/World/envs/env_.*/Robot/base/d435",
        update_period=1 / 60,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.26, 0.0, 0.12),  # URDF d435_joint mount
            rot=(-0.353553, 0.612372, -0.612372, 0.353553),  # 30 deg down + upright image (w,x,y,z)
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=45.55,  # D435 depth HFOV ~87 deg (real D435 alignment)
            clipping_range=(0.01, 3.0),
        ),
        depth_clipping_behavior="max",
        data_types=["distance_to_image_plane"],
        height=140,
        width=240,
        debug_vis=False,
    )


@configclass
class Go1RoughVisionRayCasterEnvCfg(Go1RoughVisionEnvCfg):
    """GO1 DAgger student env: MultiMeshRayCasterCamera depth, no base_lin_vel, teacher_obs emitted.

    Reuses the original author's raycast depth camera (``Go1RoughVisionEnvCfg.depth_camera``)
    but re-targets it to the real-D435 intrinsics (87 deg HFOV, 240x140) and the same d435
    mount pose as the Tiled student, so the two students see the same view. The depth is a
    Warp ray-cast over ``/World/ground`` + the robot's own links (self-occlusion), which is
    cheaper than TiledCamera (no renderer) and inherently never sees neighbouring envs.
    Terrain is the standard curriculum (``GO1_ROUGH_TERRAINS_CFG``, curriculum=True) so 4096
    envs train without the ``enforce_env_spacing`` the Tiled env needs.
    """

    # ---- DAgger student flags (mirror Go1RoughVisionTiledEnvCfg) ----
    use_lin_vel_obs = False       # student obs: no base_lin_vel (real robot has no odometry)
    emit_teacher_obs = True       # emit teacher_obs (sim base_lin_vel + heightmap) for the expert

    # ---- terrain_type / difficulty / rebuild_terrain (train_dagger_go1.py depends on this API) ----
    terrain_type: str = "rough"   # rough | stairs | slope | flat (for --terrain recording)
    difficulty: float | None = None

    use_depth_camera = True
    visualize_camera_mount = False
    depth_camera = MultiMeshRayCasterCameraCfg(
        # Mount on base with the same URDF d435_joint pose as the Tiled student so the two
        # students see an identical view (87 deg / 240x140).
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=1 / 60,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.26, 0.0, 0.12),
            rot=(-0.353553, 0.612372, -0.612372, 0.353553),  # 30 deg down + upright image (w,x,y,z)
            convention="ros",
        ),
        mesh_prim_paths=[
            "/World/ground",
            # self-occlusion: ray-cast the robot's own body too, like the real D435 sees it.
            # Use RaycastTargetCfg (not plain strings): is_shared=True shares the warp mesh
            # geometry across envs (fast init, low memory), and track_mesh_transforms=True is
            # REQUIRED so the moving legs are ray-cast at their current pose (plain strings
            # would freeze them at the initial pose and disable the shared-mesh dedup).
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Robot/base/visuals", is_shared=True, track_mesh_transforms=True
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Robot/FL_.*/visuals", is_shared=True, track_mesh_transforms=True
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Robot/FR_.*/visuals", is_shared=True, track_mesh_transforms=True
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Robot/RL_.*/visuals", is_shared=True, track_mesh_transforms=True
            ),
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Robot/RR_.*/visuals", is_shared=True, track_mesh_transforms=True
            ),
        ],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=45.55,  # D435 depth HFOV ~87 deg (matches Tiled student)
            height=140,
            width=240,
        ),
        data_types=["distance_to_image_plane"],
        # no-hit / >3 m -> 3.0 -> pipeline clip -> 2.0 (far-saturation), same as Tiled far clip.
        depth_clipping_behavior="max",
        max_distance=3.0,
        debug_vis=False,
    )

    # viewer env_index=0 works at any num_envs (same as Tiled).
    viewer: ViewerCfg = ViewerCfg(
        eye=(-3.0, 1.2, 1.8),
        lookat=(0.0, 0.0, 0.35),
        origin_type="asset_root",
        asset_name="robot",
        env_index=0,
        resolution=(1280, 720),
    )

    def __post_init__(self) -> None:
        super().__post_init__()  # Go1RoughVisionEnvCfg: adds the heightmap to observation_space
        self.rebuild_terrain()
        if not self.use_lin_vel_obs:
            # base_lin_vel excluded -> single obs space and the history buffer shrink by 3.
            self.single_observation_space = self.single_observation_space - 3
            self.observation_space = self.single_observation_space * self.history_length

    def rebuild_terrain(self) -> None:
        """Rebuild the terrain generator from ``terrain_type`` with the standard curriculum.

        Unlike the Tiled env (curriculum=False + enforce_env_spacing), the ray-cast camera
        only ever sees ``/World/ground`` + its own links, so envs can use the normal
        per-sub-terrain curriculum (curriculum=True) without overlapping. Same
        ``--terrain``/``--difficulty`` API as the Tiled env so train_dagger_go1.py works
        unchanged.
        """
        if self.terrain_type == "rough":
            sub_terrains = GO1_ROUGH_TERRAINS_CFG.sub_terrains
        elif self.terrain_type == "flat":
            sub_terrains = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
        elif self.terrain_type == "stairs":
            sub_terrains = {
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.5, step_height_range=(0.05, 0.20), step_width=0.3,
                    platform_width=3.0, border_width=1.0, holes=False,
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.5, step_height_range=(0.05, 0.20), step_width=0.3,
                    platform_width=3.0, border_width=1.0, holes=False,
                ),
            }
        elif self.terrain_type == "slope":
            sub_terrains = {
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.5, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.5, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25
                ),
            }
        else:
            raise ValueError(f"Unknown terrain_type: {self.terrain_type}")
        difficulty_range = (
            (self.difficulty, self.difficulty) if self.difficulty is not None else (0.0, 1.0)
        )
        self.terrain.terrain_generator = TerrainGeneratorCfg(
            curriculum=True,   # standard per-sub-terrain curriculum (ray-cast can't see neighbours)
            size=GO1_ROUGH_TERRAINS_CFG.size,
            border_width=GO1_ROUGH_TERRAINS_CFG.border_width,
            num_rows=GO1_ROUGH_TERRAINS_CFG.num_rows,
            num_cols=GO1_ROUGH_TERRAINS_CFG.num_cols,
            horizontal_scale=GO1_ROUGH_TERRAINS_CFG.horizontal_scale,
            vertical_scale=GO1_ROUGH_TERRAINS_CFG.vertical_scale,
            slope_threshold=GO1_ROUGH_TERRAINS_CFG.slope_threshold,
            use_cache=GO1_ROUGH_TERRAINS_CFG.use_cache,
            sub_terrains=sub_terrains,
            difficulty_range=difficulty_range,
        )
