# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    Imu,
    MultiMeshRayCaster,
    MultiMeshRayCasterCamera,
    MultiMeshRayCasterCameraCfg,
    RayCaster,
    RayCasterCamera,
    RayCasterCameraCfg,
    RayCasterCfg,
    TiledCamera,
    TiledCameraCfg,
    patterns,
)
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


from .aliengo_env_cfg import AliengoFlatEnvCfg, AliengoRoughBlindEnvCfg, AliengoRoughVisionEnvCfg
from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughVisionEnvCfg, Go2RoughBlindEnvCfg
from .go1_env_cfg import (
    Go1FlatEnvCfg,
    Go1RoughVisionEnvCfg,
    Go1RoughBlindEnvCfg,
    Go1RoughMjEnvCfg,
    Go1RoughVisionTiledEnvCfg,
    Go1RoughVisionRayCasterEnvCfg,
)
from .hyqreal_env_cfg import HyQRealFlatEnvCfg, HyQRealRoughVisionEnvCfg, HyQRealRoughBlindEnvCfg
from .b2_env_cfg import B2FlatEnvCfg, B2RoughVisionEnvCfg, B2RoughBlindEnvCfg
from .pegasus_env_cfg import PegasusFlatEnvCfg, PegasusRoughVisionEnvCfg, PegasusRoughBlindEnvCfg

from basic_locomotion_isaaclab.tasks import custom_observations, custom_rewards, custom_events
from basic_locomotion_isaaclab.tasks.supervised_learning_networks import FrozenRandomMlpEncoder, create_supervised_network

class LocomotionEnv(DirectRLEnv):

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self._edge_map_visualizer = None
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Swing peak
        self._swing_peak = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
        self._swing_peak_periodic = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)

        # mujoco_playground-style foot-contact state for the mj feet rewards
        self._mj_swing_peak = torch.zeros(self.num_envs, 4, device=self.device)
        self._mj_feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self._mj_last_contact = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        
        # Desired Hip Offset
        self._desired_hip_offset = torch.tensor([-self.cfg.desired_hip_offset, self.cfg.desired_hip_offset, -self.cfg.desired_hip_offset, self.cfg.desired_hip_offset], device=self.device)
        
        # Periodic gait
        self._step_freq = torch.tensor(self.cfg.desired_step_freq, device=self.device)
        self._duty_factor = torch.tensor(self.cfg.desired_duty_factor, device=self.device)
        self._phase_offset = torch.tensor(self.cfg.desired_phase_offset, device=self.device).repeat(self.num_envs,1)
        self._phase_signal = self._phase_offset.clone()# + self.step_dt * self._step_freq * torch.rand(self.num_envs, 1, device=self.device)*10.
        self._phase_signal = self._phase_signal % 1.0


        # Observation history
        self._observation_history = torch.zeros(self.num_envs, cfg.history_length, cfg.single_observation_space, device=self.device)
        # Sim base_lin_vel history (DAgger: spliced into teacher_obs, never the student obs)
        self._vel_history = torch.zeros(self.num_envs, cfg.history_length, 3, device=self.device)

        # RMA
        if(cfg.use_rma == True):
            self._rma_network = create_supervised_network(
                cfg.rma_observation_space,
                cfg.rma_output_space,
                network_type=getattr(cfg, "rma_network_type", "mlp"),
                sequence_length=cfg.rma_history_length,
            )
            self._rma_network.to(self.device)
            
            if self.cfg.rma_use_latent_space:
                self._rma_latent_encoder = FrozenRandomMlpEncoder(
                    cfg.rma_privileged_observation_space,
                    cfg.rma_output_space,
                    hidden_features=getattr(cfg, "rma_latent_encoder_hidden_features", 128),
                    seed=getattr(cfg, "rma_latent_encoder_seed", 0),
                )
                self._rma_latent_encoder.to(self.device)
            self._observation_history_rma = torch.zeros(self.num_envs, cfg.rma_history_length, cfg.single_rma_observation_space, device=self.device)
            if self.cfg.observation_noise_model:
                self._observation_noise_model_rma: NoiseModel = self.cfg.observation_noise_model.class_type(
                    self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
                )

        # Learned State Estimator
        if(cfg.use_concurrent_state_est == True):
            self._concurrent_state_est_network = create_supervised_network(
                cfg.concurrent_state_est_observation_space,
                cfg.concurrent_state_est_output_space,
                network_type=getattr(cfg, "concurrent_state_est_network_type", "mlp"),
                sequence_length=cfg.concurrent_state_est_history_length,
            )
            self._concurrent_state_est_network.to(self.device)
            self._observation_history_concurrent_state_est = torch.zeros(self.num_envs, cfg.concurrent_state_est_history_length, cfg.single_concurrent_state_est_observation_space, device=self.device)
            if self.cfg.observation_noise_model:
                self._observation_noise_model_concurrent_state_est: NoiseModel = self.cfg.observation_noise_model.class_type(
                    self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
                )

        # Logging: per-term episode reward sums, keyed lazily by active (non-zero-scale)
        # rewards in _get_rewards so disabled terms are never logged.
        self._episode_sums = {}
        # Per-environment velocity-tracking error used by the terrain curriculum.
        # Accumulating both the L1 error and command magnitude gives a stable
        # episode-level percentage even when individual commands are small.
        self._lin_vel_l1_error_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._lin_vel_command_l1_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Get specific body indices
        self._base_contact_sensor_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self._hip_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_hip", "FR_hip", "RL_hip", "RR_hip"], preserve_order=True)
        self._thigh_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"], preserve_order=True)
        # Undesired-contact body set is per-robot configurable (GO1 penalizes calf
        # ground drag). Default matches the previous hardcoded base + hips + thighs.
        undesired_names = getattr(self.cfg, "undesired_contact_body_names", None)
        if undesired_names is None:
            undesired_names = (
                ["base"]
                + ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]
                + ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
            )
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(
            undesired_names, preserve_order=True
        )

        
        self._feet_ids_robot, _ = self._robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self._hip_ids_robot, _ = self._robot.find_bodies(["FL_hip", "FR_hip", "RL_hip", "RR_hip"], preserve_order=True)

        # Ensure the order is consistent with the one expected in the cfg
        self._ids_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order, preserve_order=True)[0]

        if getattr(self.cfg, "visualize_edge_map", False):
            self.set_debug_vis(True)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Keep the base-centered scanner for base-height and terrain-orientation terms.
        self._pose_height_scanner = RayCaster(self.cfg.pose_height_scanner)
        self.scene.sensors["pose_height_scanner"] = self._pose_height_scanner

        # Use one small height map centered on each foot for the clearance rewards.
        self._foot_height_scanners = []
        for foot_name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
            scanner_cfg = self.cfg.foot_height_scanner.replace(
                prim_path=f"/World/envs/env_.*/Robot/{foot_name}",
                visualizer_cfg=self.cfg.foot_height_scanner.visualizer_cfg.replace(
                    prim_path=f"/Visuals/{foot_name}HeightScanner"
                ),
            )
            scanner = RayCaster(scanner_cfg)
            self.scene.sensors[f"{foot_name.lower()}_height_scanner"] = scanner
            self._foot_height_scanners.append(scanner)

        # Add the perceptive and edge scanners only for vision-based locomotion.
        if(getattr(self.cfg, "use_vision", False)):
            self._perceptive_height_scanner = RayCaster(self.cfg.perceptive_height_scanner)
            self.scene.sensors["perceptive_height_scanner"] = self._perceptive_height_scanner

            self._edge_height_scanner = RayCaster(self.cfg.edge_height_scanner)
            self.scene.sensors["edge_height_scanner"] = self._edge_height_scanner

        # we add a depth camera if needed for vision-based locomotion
        if(getattr(self.cfg, "use_depth_camera", False)):
            if isinstance(self.cfg.depth_camera, TiledCameraCfg):
                self._depth_camera = TiledCamera(self.cfg.depth_camera)
            else:
                self._depth_camera = MultiMeshRayCasterCamera(self.cfg.depth_camera)
            self.scene.sensors["depth_camera"] = self._depth_camera

        # we add the Unitree L2 LiDAR if needed for vision-based locomotion
        if(getattr(self.cfg, "use_unitree_l2_lidar", False)):
            self._unitree_l2_lidar = MultiMeshRayCaster(self.cfg.unitree_l2_lidar)
            self.scene.sensors["unitree_l2_lidar"] = self._unitree_l2_lidar

        # we add an imu
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # TiledCamera dagger envs (cfg.enforce_env_spacing): one robot per sub-terrain
        # (8 m spacing >> 2 m far clip) so the rendered depth never sees a neighbour.
        if getattr(self.cfg, "enforce_env_spacing", False) and self.cfg.terrain.terrain_generator is not None:
            gen = self.cfg.terrain.terrain_generator
            rows, cols = gen.num_rows, gen.num_cols
            if rows * cols < self.scene.cfg.num_envs:
                raise RuntimeError(
                    f"terrain {rows}x{cols} sub-terrains < num_envs {self.scene.cfg.num_envs}; "
                    "enlarge the terrain (Go1RoughVisionTiledEnvCfg.__post_init__) or reduce --num_envs"
                )
            idx = torch.arange(self.scene.cfg.num_envs, device=self.device)
            r = idx // cols
            c = idx % cols
            self._terrain.env_origins = self._terrain.terrain_origins[r, c]
            self._terrain.terrain_levels = r

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        default_joint_pos_ordered = self._robot.data.default_joint_pos[:, self._ids_joints_order]
        
        # Clip the action
        self._actions = torch.clamp(self._actions, -self.cfg.desired_clip_actions, self.cfg.desired_clip_actions)

        # Filter the action
        if(self.cfg.use_filter_actions):
            alpha = 0.8
            temp = alpha * self._actions + (1 - alpha) * self._previous_actions
            self._processed_actions = self.cfg.action_scale * temp + default_joint_pos_ordered
        else:
            self._processed_actions = self.cfg.action_scale * self._actions + default_joint_pos_ordered


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._ids_joints_order)


    def _get_observations(self) -> dict:
        
        # Sample new commands if needed
        custom_events._get_new_random_commands(self)


        # Observation --------------------------------------------------------------------------------------
        clock_data = None
        if(self.cfg.use_clock_signal):
            self._phase_signal += self.step_dt * self._step_freq
            self._phase_signal = self._phase_signal % 1.0
            clock_data = torch.vstack([self._phase_signal[:,0], self._phase_signal[:,1], self._phase_signal[:,2], self._phase_signal[:,3]]).T
            # all the envs that are not moving, we put -1
            should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
            clock_data[:, :] = clock_data[:, :]*should_move.unsqueeze(1).expand(-1, 4) + -1.0* ~should_move.unsqueeze(1).expand(-1, 4)
            

        # Choosing the main source of observation
        if(self.cfg.use_concurrent_state_est):
            # If concurrent SE/Learned State Estimator, we predict linear and angular vel from IMU
            base_linear = custom_observations._get_concurrent_state_estimation(self)
            base_ang_vel = self._imu.data.ang_vel_b
            projected_gravity_b = self._imu.data.projected_gravity_b
        elif(self.cfg.use_imu):
            # Using directly the IMU
            base_linear = self._imu.data.lin_acc_b
            base_ang_vel = self._imu.data.ang_vel_b
            projected_gravity_b = self._imu.data.projected_gravity_b
        else:
            #Using a model-based state estimation
            base_linear = self._robot.data.root_lin_vel_b
            base_ang_vel = self._robot.data.root_ang_vel_b
            projected_gravity_b = self._robot.data.projected_gravity_b

        # DAgger student: exclude base_lin_vel (not available on the real robot).
        # The teacher still receives it via teacher_obs (emit_teacher_obs) when enabled.
        if not getattr(self.cfg, "use_lin_vel_obs", True):
            base_linear = None


        # Standard Obs for the Actor/Critic
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    base_linear,
                    base_ang_vel,
                    projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos[:, self._ids_joints_order] - self._robot.data.default_joint_pos[:, self._ids_joints_order],
                    self._robot.data.joint_vel[:, self._ids_joints_order],
                    self._actions,
                    clock_data,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        if(self.cfg.use_observation_history):
            #the bottom element is the newest observation!!
            self._observation_history = torch.cat((self._observation_history[:,1:,:], obs.unsqueeze(1)), dim=1)
            obs = torch.flatten(self._observation_history, start_dim=1)


        # DAgger: keep a per-frame sim base_lin_vel history so teacher_obs can carry it
        # (spliced in front of each history frame, matching the teacher's training layout).
        emit_teacher_obs = getattr(self.cfg, "emit_teacher_obs", False)
        if emit_teacher_obs:
            self._vel_history = torch.cat(
                (self._vel_history[:, 1:], self._robot.data.root_lin_vel_b.unsqueeze(1)), dim=1
            )

        observations = {"common": obs}


        # Add heightmap data to obs if needed
        if(getattr(self.cfg, "use_vision", False)):
            height_data = (
                self._perceptive_height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self._perceptive_height_scanner.data.ray_hits_w[..., 2]
                - 0.5
            )
            height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
            height_data = height_data.clip(-1.0, 1.0)
            obs = torch.cat((obs, height_data), dim=-1)

        # DAgger: build the privileged teacher obs (sim base_lin_vel + heightmap) so the
        # expert policy can label the student states. Teacher reads via obs_groups.
        if emit_teacher_obs:
            teacher_hist = torch.cat((self._vel_history, self._observation_history), dim=-1)
            teacher_obs = torch.flatten(teacher_hist, start_dim=1)
            if getattr(self.cfg, "use_vision", False):
                teacher_obs = torch.cat((teacher_obs, height_data), dim=-1)
            observations["teacher_obs"] = teacher_obs   


        # Critic OBS could be different if needed
        if(self.cfg.use_asymmetric_ppo):
            obs_critic = custom_observations._get_privileged_observation_asymmetric(self)
            observations["critic"] = torch.cat((obs, obs_critic), dim=-1)
        else:
            observations["critic"] = obs


        # If RMA, we add some other predicted obs AFTER the critic asymmetric obs to avoid duplication
        if(self.cfg.use_rma):
            # Predict the RMA observation
            obs_rma = custom_observations._get_rma(self)
            obs = torch.cat((obs, obs_rma), dim=-1)


        # Actor OBS - here after the critic to avoid duplication with rma obs
        # if asymmetric ppo is used
        observations["policy"] = obs    
        # ------------------------------------------------------------------------------------------

        # AMP related observation if used
        if(self.cfg.use_amp):
            obs_amp = torch.cat(
                [
                    tensor
                    for tensor in (
                        #self._robot.data.root_quat_w,
                        self._robot.data.joint_pos[:, self._ids_joints_order],
                        self._robot.data.joint_vel[:, self._ids_joints_order],
                        self._robot.data.root_lin_vel_b,
                        self._robot.data.root_ang_vel_b,
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
            observations["amp"] = obs_amp

        # --------------------------------------------------------------------------------------------
        return observations


    def _get_rewards(self) -> torch.Tensor:

        track_height_exp = custom_rewards.track_height_exp(self)
        track_lin_vel_xy_exp = custom_rewards.track_lin_vel_xy_exp(self)
        track_lin_vel_z_l2 = custom_rewards.track_lin_vel_z_l2(self)
        track_orientation_l2 = custom_rewards.track_orientation_l2(self)
        track_ang_vel_xy_l2 = custom_rewards.track_ang_vel_xy_l2(self)
        track_ang_vel_z_exp = custom_rewards.track_ang_vel_z_exp(self)

        undesired_contacts = custom_rewards.undesired_contacts(self)
        action_rate_l2 = custom_rewards.action_rate_l2(self)
        action_smoothness_l2 = custom_rewards.action_smoothness_l2(self)

        joints_hip_pos_l2 = custom_rewards.joints_hip_pos_l2(self)
        joints_thigh_pos_l2 = custom_rewards.joints_thigh_pos_l2(self)
        joints_calf_pos_l2 = custom_rewards.joints_calf_pos_l2(self)
        joints_acc_l2 = custom_rewards.joints_acc_l2(self)
        joints_torques_l2 = custom_rewards.joints_torques_l2(self)
        joints_energy_l1 = custom_rewards.joints_energy_l1(self)

        feet_air_time = custom_rewards.feet_air_time(self)
        feet_air_time_variance = custom_rewards.feet_air_time_variance(self)

        feet_slide = custom_rewards.feet_slide(self)
        feet_edge = custom_rewards.feet_edge(self)
        periodic_contact_suggestion = custom_rewards.periodic_contact_suggestion(self)
        stance_contact_suggestion = custom_rewards.stance_contact_suggestion(self)
        feet_height_clearance_mujoco_aperiodic = custom_rewards.feet_height_clearance_mujoco_aperiodic(self)
        feet_height_clearance_mujoco_periodic = custom_rewards.feet_height_clearance_mujoco_periodic(self)
        feet_height_clearance_periodic = custom_rewards.feet_height_clearance_periodic(self)
        feet_height_clearance_aperiodic = custom_rewards.feet_height_clearance_aperiodic(self)
        feet_to_hip_distance_l2 = custom_rewards.feet_to_hip_distance_l2(self)
        feet_vertical_surface_contacts = custom_rewards.feet_vertical_surface_contacts(self)

        custom_rewards._mj_feet_update_state(self)
        mj_feet_clearance = custom_rewards.mj_feet_clearance(self)
        mj_feet_height = custom_rewards.mj_feet_height(self)
        mj_feet_slip = custom_rewards.mj_feet_slip(self)
        mj_feet_air_time = custom_rewards.mj_feet_air_time(self)
        dof_pos_limits = custom_rewards.dof_pos_limits(self)
        pose = custom_rewards.pose(self)
        termination = custom_rewards.termination(self)

        # mujoco_playground GO1 joystick base rewards (faithful forms for Go1RoughMjEnvCfg;
        # scale 0 on every other env -> dropped from reward_terms, never logged).
        mj_orientation = custom_rewards.mj_orientation(self)
        mj_lin_vel_z = custom_rewards.mj_lin_vel_z(self)
        mj_ang_vel_xy = custom_rewards.mj_ang_vel_xy(self)
        mj_torques = custom_rewards.mj_torques(self)
        stand_still = custom_rewards.stand_still(self)

        # Build the reward terms, dropping any with a zero scale so disabled rewards
        # neither contribute to the total nor get logged.
        reward_terms = [
            ("track_height_exp", track_height_exp, self.cfg.height_reward_scale),
            ("track_lin_vel_xy_exp", track_lin_vel_xy_exp, self.cfg.lin_vel_reward_scale),
            ("track_lin_vel_z_l2", track_lin_vel_z_l2, self.cfg.z_vel_reward_scale),
            ("track_orientation_l2", track_orientation_l2, self.cfg.orientation_reward_scale),
            ("track_ang_vel_xy_l2", track_ang_vel_xy_l2, self.cfg.ang_vel_reward_scale),
            ("track_ang_vel_z_exp", track_ang_vel_z_exp, self.cfg.yaw_rate_reward_scale),

            ("undesired_contacts", undesired_contacts, self.cfg.undersired_contact_reward_scale),
            ("action_rate_l2", action_rate_l2, self.cfg.action_rate_reward_scale),
            ("action_smoothness_l2", action_smoothness_l2, self.cfg.action_smoothness_reward_scale),

            ("joints_hip_pos_l2", joints_hip_pos_l2, self.cfg.joints_hip_position_reward_scale),
            ("joints_thigh_pos_l2", joints_thigh_pos_l2, self.cfg.joints_thigh_position_reward_scale),
            ("joints_calf_pos_l2", joints_calf_pos_l2, self.cfg.joints_calf_position_reward_scale),
            ("joints_acc_l2", joints_acc_l2, self.cfg.joints_accel_reward_scale),
            ("joints_torques_l2", joints_torques_l2, self.cfg.joints_torque_reward_scale),
            ("joints_energy_l1", joints_energy_l1, self.cfg.joints_energy_reward_scale),

            ("feet_air_time", feet_air_time, self.cfg.feet_air_time_reward_scale),
            ("feet_air_time_variance", feet_air_time_variance, self.cfg.feet_air_time_variance_reward_scale),

            ("feet_height_clearance_aperiodic", feet_height_clearance_aperiodic, self.cfg.feet_height_clearance_aperiodic_reward_scale),
            ("feet_height_clearance_periodic", feet_height_clearance_periodic, self.cfg.feet_height_clearance_periodic_reward_scale),
            ("feet_height_clearance_mujoco_aperiodic", feet_height_clearance_mujoco_aperiodic, self.cfg.feet_height_clearance_mujoco_aperiodic_reward_scale),
            ("feet_height_clearance_mujoco_periodic", feet_height_clearance_mujoco_periodic, self.cfg.feet_height_clearance_mujoco_periodic_reward_scale),

            ("feet_slide", feet_slide, self.cfg.feet_slide_reward_scale),
            ("feet_to_hip_distance_l2", feet_to_hip_distance_l2, self.cfg.feet_to_hip_distance_reward_scale),
            ("feet_edge", feet_edge, self.cfg.feet_edge_reward_scale),
            ("feet_vertical_surface_contacts", feet_vertical_surface_contacts, self.cfg.feet_vertical_surface_contacts_reward_scale),

            ("periodic_contact_suggestion", periodic_contact_suggestion, self.cfg.periodic_contact_suggestion_reward_scale),
            ("stance_contact_suggestion", stance_contact_suggestion, self.cfg.stance_contact_suggestion_reward_scale),

            ("dof_pos_limits", dof_pos_limits, getattr(self.cfg, "dof_pos_limits_reward_scale", 0.0)),
            ("pose", pose, getattr(self.cfg, "pose_reward_scale", 0.0)),
            ("termination", termination, getattr(self.cfg, "termination_reward_scale", 0.0)),

            ("mj_feet_clearance", mj_feet_clearance, getattr(self.cfg, "mj_feet_clearance_reward_scale", 0.0)),
            ("mj_feet_height", mj_feet_height, getattr(self.cfg, "mj_feet_height_reward_scale", 0.0)),
            ("mj_feet_slip", mj_feet_slip, getattr(self.cfg, "mj_feet_slip_reward_scale", 0.0)),
            ("mj_feet_air_time", mj_feet_air_time, getattr(self.cfg, "mj_feet_air_time_reward_scale", 0.0)),

            ("mj_orientation", mj_orientation, getattr(self.cfg, "mj_orientation_reward_scale", 0.0)),
            ("mj_lin_vel_z", mj_lin_vel_z, getattr(self.cfg, "mj_z_vel_reward_scale", 0.0)),
            ("mj_ang_vel_xy", mj_ang_vel_xy, getattr(self.cfg, "mj_ang_vel_xy_reward_scale", 0.0)),
            ("mj_torques", mj_torques, getattr(self.cfg, "mj_torques_reward_scale", 0.0)),
            ("stand_still", stand_still, getattr(self.cfg, "stand_still_reward_scale", 0.0)),
        ]
        rewards = {key: value * scale * self.step_dt for key, value, scale in reward_terms if scale != 0.0}
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Per-step total reward clip to [0, 10000] (mujoco_playground joystick.py:278).
        # Only envs with cfg.reward_clip set clip; all others are unchanged.
        if getattr(self.cfg, "reward_clip", None) is not None:
            reward = torch.clip(reward, min=self.cfg.reward_clip[0], max=self.cfg.reward_clip[1])

        # Check for NaNs and Infs
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            print("NaN or Inf detected in reward computation. Setting reward to zero for affected environments.")
            breakpoint()  # For debugging purposes
            reward = torch.where(torch.isnan(reward) | torch.isinf(reward), torch.zeros_like(reward), reward)
        
        # Logging (lazily initialize the per-term accumulators so disabled rewards are never logged)
        for key, value in rewards.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._episode_sums[key] += value
        lin_vel_command_l1 = torch.sum(torch.abs(self._commands[:, :2]), dim=1)
        tracks_linear_command = lin_vel_command_l1 > 0.01
        lin_vel_l1_error = torch.sum(
            torch.abs(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        self._lin_vel_l1_error_sum += lin_vel_l1_error * tracks_linear_command
        self._lin_vel_command_l1_sum += lin_vel_command_l1 * tracks_linear_command

        custom_rewards._mj_feet_finalize_state(self)
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died_check_base = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_contact_sensor_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        died_check_hips = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._hip_contact_sensor_ids], dim=-1), dim=1)[0] > 1.0, dim=1) 
        died = torch.logical_or(died_check_base, died_check_hips)
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        lin_vel_command_l1_sum = self._lin_vel_command_l1_sum[env_ids]
        has_linear_velocity_commands = lin_vel_command_l1_sum > 0.0
        lin_vel_l1_error_percent = 100.0 * self._lin_vel_l1_error_sum[env_ids] / torch.clamp(
            lin_vel_command_l1_sum, min=1.0e-6
        )

        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            # The command changes during an episode, so displacement from the
            # origin is not a reliable measure of tracking quality. Use the
            # episode-level linear-velocity L1 error relative to the commands.
            move_up = torch.logical_and(
                has_linear_velocity_commands,
                lin_vel_l1_error_percent
                < getattr(self.cfg, "terrain_curriculum_move_up_error_percent", 20.0),
            )
            move_down = torch.logical_and(
                has_linear_velocity_commands,
                lin_vel_l1_error_percent
                > getattr(self.cfg, "terrain_curriculum_move_down_error_percent", 50.0),
            )
            # update terrain levels
            self._terrain.update_env_origins(env_ids, move_up, move_down)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs: 
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        # Reset actions and action filtering
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0
        
        # Reset commands
        custom_events._get_new_random_commands(self, env_ids)

        # Reset swing peak
        self._swing_peak[env_ids] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        self._swing_peak_periodic[env_ids] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)

        # Reset mujoco-style foot-contact state
        self._mj_swing_peak[env_ids] = 0.0
        self._mj_feet_air_time[env_ids] = 0.0
        self._mj_last_contact[env_ids] = False
        
        # Reset contact periodic
        self._phase_signal[env_ids] = self._phase_offset[env_ids].clone()# + self.step_dt * self._step_freq * torch.rand(env_ids.shape[0], 1, device=self.device)*10.
        self._phase_signal[env_ids] = self._phase_signal[env_ids]  % 1.0

        # Reset observation history
        self._observation_history[env_ids] *= 0.0
        self._vel_history[env_ids] *= 0.0

        # Reset obs and noise concurrent
        if(self.cfg.use_concurrent_state_est):
            self._observation_history_concurrent_state_est[env_ids] *= 0.0
            if self.cfg.observation_noise_model:
                self._observation_noise_model_concurrent_state_est.reset(env_ids)
        
        # Reset obs and noise rma
        if(self.cfg.use_rma):
            self._observation_history_rma[env_ids] *= 0.0
            if self.cfg.observation_noise_model:
                self._observation_noise_model_rma.reset(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos += torch.zeros_like(joint_pos).uniform_(-0.2, 0.2)
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 3:7] = math_utils.random_yaw_orientation(env_ids.shape[0], device=self.device)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Metric/lin_vel_l1_error_percent"] = torch.sum(lin_vel_l1_error_percent) / torch.clamp(
            torch.count_nonzero(has_linear_velocity_commands), min=1
        )
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        
        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            extras["Episode_Curriculum/terrain_levels"] = torch.mean(self._terrain.terrain_levels.float())
        
        self.extras["log"].update(extras)

        self._lin_vel_l1_error_sum[env_ids] = 0.0
        self._lin_vel_command_l1_sum[env_ids] = 0.0


    def _set_debug_vis_impl(self, debug_vis: bool):
        custom_rewards._set_debug_vis_impl(self, debug_vis)


    def _debug_vis_callback(self, event):
        custom_rewards._debug_vis_callback(self, event)
