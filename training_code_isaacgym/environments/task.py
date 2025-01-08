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
        raise NotImplementedError
        # Here we get low level observations and need to transform them to high level observations
        # This will probably also need customization of the configuration
        observations = self.obs_buf
        modified_observations = observations
        return modified_observations
