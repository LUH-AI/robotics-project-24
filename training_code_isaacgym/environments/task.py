from typing import Any, Callable, List

import torch
import cv2
import os
import sys

from isaacgym import gymapi
from isaacgym.torch_utils import *
from legged_gym import LEGGED_GYM_ROOT_DIR

from legged_gym.utils.task_registry import task_registry

from ..configs.robots import GO2DefaultCfg
from ..configs.scenes import BaseSceneCfg
from ..configs.algorithms import PPODefaultCfg
from .compatible_legged_robot import CompatibleLeggedRobot

GO2DefaultCfg()
# do CONFIGURABLE adaptations in this file

import numpy as np



# register all tasks derived from CustomLeggedRobot
def register_task(
    robot_cfg: GO2DefaultCfg, scene_cfg: BaseSceneCfg, algorithm_cfg: PPODefaultCfg, robot_class: Callable
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
    robot_cfg.env.env_spacing = scene_cfg.size
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

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        Additionally, sets segmentation_id to 1 (index of ObjectType="ground")
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        plane_params.segmentation_id = 1  # added for compatibility
        self.gym.add_ground(self.sim, plane_params)

    def _place_static_objects(self, env_idx: int, env_handle: Any):
        """Places static objects like walls into the provided environment
        It is called in the environment creation loop in super()._create_envs()

        Args:
            env_idx (int): Index of environment
            env_handle (Any): Environment handle
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
                self.commands[:, :3] * self.commands_scale,
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
        self._prepare_camera(cfg.camera)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # for language server purposes only
        self.cfg: GO2DefaultCfg = self.cfg

    def _prepare_camera(self, camera):
        print("Preparing")
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.horizontal_fov = camera.horizontal_fov
        self.camera_props.width = camera.width
        self.camera_props.height = camera.height
        self.camera_props.enable_tensors = camera.enable_tensors
        self.camera_props.use_collision_geometry = True

    def _create_ground_plane(self):
        """Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        Additionally, sets segmentation_id to 1 (index of ObjectType="ground")
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        plane_params.segmentation_id = 1  # added for compatibility
        self.gym.add_ground(self.sim, plane_params)

    def _place_static_objects(self, env_idx: int, env_handle: Any):
        """Places static objects like walls into the provided environment
        It is called in the environment creation loop in super()._create_envs()

        Args:
            env_idx (int): Index of environment
            env_handle (Any): Environment handle
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
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # All entries in torch.cat have dimensions, n_envs * n where you can determine n, but have to add it to n_obs in the configuration of the robot
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # add custom rewards... here (use your robot_cfg for control)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        raise NotImplementedError
        # Here we get high level actions and need to translate them to low level actions
        modified_actions = actions
        step_return = super().step(modified_actions)
        return step_return

    def get_observations(self):
        depth_arrays = torch.tensor([self.gym.get_camera_image(
            self.sim, self.envs[i], self.cameras[i], gymapi.IMAGE_DEPTH
        ) for i in range(len(self.cameras))]) # Has shape (num_envs, 12) TODO: Check if extra dimension somehow sneaked in
        #TODO: Add Distance measure to observation / preprocess it to a good range. Currently: -inf = infinitely far away and -0.1 is 10cm away
        raise NotImplementedError
        # Here we get low level observations and need to transform them to high level observations
        # This will probably also need customization of the configuration
        observations = self.obs_buf
        modified_observations = observations
        return modified_observations

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        super()._create_envs()
        # ADD Camera Functionality
        self.cameras = []
        for env_handle, actor_handle in zip(self.envs, self.actor_handles):
            camera_handle = self.gym.create_camera_sensor(env_handle, self.camera_props)
            local_transform = gymapi.Transform()
            local_transform.p = self.cfg.camera.vec_from_body_center
            local_transform.r = self.cfg.camera.rot_of_camera
            self.gym.attach_camera_to_body(
                camera_handle,
                env_handle,
                actor_handle,
                local_transform,
                gymapi.FOLLOW_TRANSFORM,
            )
            self.cameras.append(camera_handle)
    
    def render(self, sync_frame_time=True):
        super().render(sync_frame_time)
        # This renders all cameras each simulation step
        self.gym.render_all_camera_sensors(self.sim)
