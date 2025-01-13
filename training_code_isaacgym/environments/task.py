from typing import Any, Callable

import torch
from isaacgym import gymapi

from legged_gym.utils.task_registry import task_registry

from ..configs.robots import GO2DefaultCfg
from ..configs.scenes import BaseSceneCfg
from ..configs.algorithms import PPODefaultCfg
from .compatible_legged_robot import CompatibleLeggedRobot
from rsl_rl.modules import ActorCritic
from .util import load_low_level_policy

GO2DefaultCfg()
# do CONFIGURABLE adaptations in this file


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


class HighLevelPlantPolicyLeggedRobot(CompatibleLeggedRobot):
    def __init__(
        self, cfg: GO2DefaultCfg, sim_params, physics_engine, sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
   
        # for language server purposes only
        self.cfg: GO2DefaultCfg = self.cfg

        # Load low-level policy
        self.low_level_policy = load_low_level_policy(cfg, sim_device)
        # use `self.low_level_policy.act(observations)` to get an action. See `ActorCritic` in rsl_rl/modules/actor_critic.py

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


    #computes high level observations
    def compute_high_level_observations(self):
        """ Computes observations
        """
        #change this
        self.high_level_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.projected_gravity,
                                    ),dim=-1)
        print(f"Computed high level observations: {self.high_level_obs_buf.shape}")
        
  
    #dont rename, onPolicyRunner calls this
    def get_observations(self):
        print(f"get_observations Self.obs.shape {self.obs_buf.shape}")
        return self.high_level_obs_buf
    
    # add custom rewards... here (use your robot_cfg for control)

    def step(self, high_level_actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Here we get high level actions and need to translate them to low level actions
        #actions are always low_level (dim=12) and high_level_actions are high level (dim=5)
        print(f"HighLevelPlant.step(actions: {high_level_actions.shape})")
        self.high_level_actions = high_level_actions
        actions = self.low_level_policy.act_inference(self.obs_buf)
        
        #obs buffer is low level, cannot rename because leggedrobot uses it
        obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras = super().step(actions)
        self.compute_high_level_observations()
        return self.high_level_obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras


    #this computes low level observations
    #cannot be renamed, because leggedrobot calls this
    def compute_observations(self):
        """ Computes observations
        """
       
        # Set the first command (index 0) for each robot to 0.5
        #set fixed command to let the robot rotate
        print(f"Self.high_level_actions: {self.high_level_actions.shape}")
        self.high_level_actions = torch.tensor([0, 0, 0.2], dtype=torch.float).repeat(32, 1).to(self.device)
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.high_level_actions,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        print(f"Computed low level observations: {self.obs_buf.shape}")
        #print(self.obs_buf[0])
        
