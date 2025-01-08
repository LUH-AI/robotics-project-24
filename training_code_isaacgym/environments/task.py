from typing import Any, Callable
import warnings

import torch
from isaacgym import gymapi

from legged_gym.utils.task_registry import task_registry

from ..configs.robots import GO2DefaultCfg
from ..configs.scenes import BaseSceneCfg
from ..configs.algorithms import PPODefaultCfg
from .compatible_legged_robot import CompatibleLeggedRobot
from . import task_utils


GO2DefaultCfg()
# do CONFIGURABLE adaptations in this file


# register all tasks derived from CustomLeggedRobot
def register_task(
    robot_cfg: GO2DefaultCfg,
    scene_cfg: BaseSceneCfg,
    algorithm_cfg: PPODefaultCfg,
    robot_class: Callable,
) -> str:
    """Register task based on robot, scene and algorithm configurations
    All configs need a name attribute.

    Args:
        robot_cfg (GO2DefaultCfg): Robot config
        scene_cfg (BaseSceneCfg): Scene config
        algorithm_cfg (PPODefaultCfg): Algorithm config

    Returns:
        str: Task name
    """
    name = f"{robot_cfg.name}_{scene_cfg.name}_{algorithm_cfg.name}"

    # add scene_cfg to robot_cfg for compatibility with task_registry
    robot_cfg.scene = scene_cfg
    robot_cfg.env.env_spacing = scene_cfg.size + scene_cfg.spacing
    algorithm_cfg.runner.experiment_name = name
    task_registry.register(name, robot_class, robot_cfg, algorithm_cfg)
    return name


class CustomLeggedRobot(CompatibleLeggedRobot):
    def __init__(
        self, cfg: GO2DefaultCfg, sim_params, physics_engine, sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # for language server purposes only
        self.cfg: GO2DefaultCfg = self.cfg

        self.absolute_plant_locations: torch.Tensor
        """
        Absolute locations of plants in each environment
        shape: (|environments| x |plants_per_env| x 3)
        (Attribute is instantiated in self._place_static_objects())
        """

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :6] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        # All entries in torch.cat have dimensions, n_envs * n where you can determine n, but have to add it to n_obs in the configuration of the robot
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    # add custom rewards... here (use your robot_cfg for control)


class HighLevelPlantPolicyLeggedRobot(CompatibleLeggedRobot):
    def __init__(
        self, cfg: GO2DefaultCfg, sim_params, physics_engine, sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # for language server purposes only
        self.cfg: GO2DefaultCfg = self.cfg

        self.absolute_plant_locations: torch.Tensor
        """
        Absolute locations of plants in each environment
        shape: (|environments| x |plants_per_env| x 3)
        (Attribute is instantiated in self._place_static_objects())
        """
        self.object_handles.append([])
        self.num_static_objects = len(self.cfg.scene.static_objects)
        for object_idx, static_obj in enumerate(self.cfg.scene.static_objects):
            if len(self.object_assets) - 1 > object_idx:
                obj_asset = self.object_assets[object_idx]
            else:
                obj_asset = self.gym.load_asset(
                    self.sim,
                    str(static_obj.asset_root),
                    str(static_obj.asset_file),
                    static_obj.asset_options,
                )
                self.object_assets.append(obj_asset)

            start_pose = gymapi.Transform()
            location_offset = self.env_origins[env_idx].clone()
            init_location = static_obj.init_location.to(self.device)
            random_loc_offset = (
                static_obj.max_random_loc_offset
                * (torch.rand(static_obj.max_random_loc_offset.shape) - 0.5)
                * 2
            ).to(self.device)
            start_pose.p = gymapi.Vec3(
                *(init_location + location_offset + random_loc_offset)
            )

            # env_idx sets collision group, -1 default for collision_filter
            object_handle = self.gym.create_actor(
                env_handle,
                obj_asset,
                start_pose,
                static_obj.name,
                env_idx,
                -1,
                static_obj.segmentation_id,
            )
            self.object_handles[env_idx].append(object_handle)

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :6] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        # All entries in torch.cat have dimensions, n_envs * n where you can determine n, but have to add it to n_obs in the configuration of the robot
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    # add custom rewards... here (use your robot_cfg for control)

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        raise NotImplementedError
        # Here we get high level actions and need to translate them to low level actions
        modified_actions = actions
        step_return = super().step(modified_actions)
        return step_return

    def get_observations(self):
        raise NotImplementedError
        # Here we get low level observations and need to transform them to high level observations
        # This will probably also need customization of the configuration
        observations = self.obs_buf
        modified_observations = observations
        return modified_observations


    def _reward_tracking_ang_vel2(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = (self.commands[:, 2] * self.base_ang_vel[:, 2])
        # ang_vel_error[ang_vel_error < 0] = 0
        # ang_vel_error[self.commands[:, 2]**2>0.1**2] = 0
        # stability_term = 0.1 * self.base_ang_vel[:, 2]/self.cfg.rewards.tracking_sigma
        return ang_vel_error / (
                    ang_vel_error * ang_vel_error + 0.01) / self.cfg.rewards.tracking_sigma - 0.05 * self.base_ang_vel[
                                                                                                     :, 2] ** 2
    def _reward_fast_rotation(self):
        """
        Reward for fast rotations based on the z-axis angular velocity of the base.
        """
        # Reward is proportional to the absolute value of the z-axis angular velocity
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_base_height_tracking(self):
        """Reward for tracking the base height command."""
        height_error = torch.abs(self.commands[:, 3] - self.base_pos[:, 2])  # Compare command to current height
        return torch.exp(-height_error / 0.1)  # Adjust scaling factor (e.g., 0.1) as needed

    def _reward_base_pitch_tracking(self):
        """Reward for tracking the base pitch angle command."""
        pitch_error = torch.abs(self.commands[:, 4] - self.rpy[:, 1])  # Compare command to current pitch
        return torch.exp(-pitch_error / 0.1)  # Adjust scaling factor (e.g., 0.1) as needed

    def _reward_base_roll_tracking(self):
        """
        Reward for fast rotations based on the z-axis angular velocity of the base.
        """
        # Reward is proportional to the absolute value of the z-axis angular velocity
        roll_error = torch.abs(self.commands[:, 5] - self.rpy[:, 0])
        return torch.exp(-roll_error / 0.1)

    def _reward_tracking_lin_vel_stability(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)

        # Standard tracking reward
        reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

        # Apply negative reward if target velocity is zero but actual velocity is non-zero
        zero_target_mask = torch.all(self.commands[:, :2] == 0, dim=1)  # Check if both target velocities are zero
        actual_velocity_non_zero = torch.any(self.base_lin_vel[:, :2] != 0,
                                             dim=1)  # Check if any actual velocity component is non-zero
        penalty_mask = zero_target_mask & actual_velocity_non_zero  # Apply penalty only when both conditions are met

        negative_reward = -1.0  # Define the penalty value
        reward += penalty_mask.float() * negative_reward  # Add negative reward for violating condition

        return reward

    def _reward_foot_in_the_air(self):
        """
        Penalize feet floating in the air unnecessarily based on the current motion command.
        This reward encourages stable and efficient foot-ground contact patterns.

        Returns:
            torch.Tensor: Reward values penalizing feet in the air.
        """
        # Calculate whether each foot is in contact
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0  # Z-axis force threshold for contact

        # Identify commands requiring consistent foot-ground contact
        is_moving = torch.norm(self.commands[:, :3], dim=1) > 0.1  # Commands with significant linear/angular velocity
        floating_feet = ~contact  # Feet not in contact

        # Penalize floating feet for moving commands
        penalization = floating_feet.sum(dim=1) * is_moving.float()

        # Scale the penalty to fit within the reward framework
        penalty = -1.0 * penalization  # Adjust scale factor as needed

        # Add this penalty to the reward buffer
        return penalty