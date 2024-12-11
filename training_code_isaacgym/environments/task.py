from typing import Any

import torch
from isaacgym import gymapi

from legged_gym.utils.task_registry import task_registry

from ..configs.robots import GO2DefaultCfg
from ..configs.scenes import BaseSceneCfg
from ..configs.algorithms import PPODefaultCfg
from .compatible_legged_robot import CompatibleLeggedRobot

GO2DefaultCfg()
# do CONFIGURABLE adaptations in this file


# register all tasks derived from CustomLeggedRobot
def register_task(robot_cfg: GO2DefaultCfg, scene_cfg: BaseSceneCfg, algorithm_cfg: PPODefaultCfg) -> str:
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
    task_registry.register(name, CustomLeggedRobot, robot_cfg, algorithm_cfg)
    return name


class CustomLeggedRobot(CompatibleLeggedRobot):
    def __init__(self, cfg: GO2DefaultCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # for language server purposes only
        self.cfg: GO2DefaultCfg = self.cfg

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        Additionally, sets segmentation_id to 1 (index of ObjectType="ground")
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        plane_params.segmentation_id = 1 # added for compatibility
        self.gym.add_ground(self.sim, plane_params)

    def _place_static_objects(self, env_idx: int, env_handle: Any):
        """Places static objects like walls into the provided environment

        Args:
            env_idx (int): Index of environment
            env_handle (Any): Environment handle
        """
        self.object_handles.append([])
        self.num_static_objects = len(self.cfg.scene.static_objects)
        for static_obj in self.cfg.scene.static_objects:
            obj_asset = self.gym.load_asset(self.sim, str(static_obj.asset_root), str(static_obj.asset_file), static_obj.asset_options)

            start_pose = gymapi.Transform()
            location_offset = self.env_origins[env_idx].clone()
            init_location = static_obj.init_location.to(self.device)
            random_loc_offset = (static_obj.max_random_loc_offset * (torch.rand(static_obj.max_random_loc_offset.shape) - 0.5) * 2).to(self.device)
            start_pose.p = gymapi.Vec3(*(init_location + location_offset + random_loc_offset))

            # env_idx sets collision group, -1 default for collision_filter
            object_handle = self.gym.create_actor(env_handle, obj_asset, start_pose, static_obj.name, env_idx, -1, static_obj.segmentation_id)
            self.object_handles[env_idx].append(object_handle)

    # add custom rewards... here (use your robot_cfg for control)
