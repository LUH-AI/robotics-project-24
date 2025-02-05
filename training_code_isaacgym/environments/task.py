from typing import Callable

import torch
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *

from legged_gym.utils.task_registry import task_registry

from ..configs.robots import GO2DefaultCfg, GO2HighLevelPlantPolicyCfg
from ..configs.scenes import BaseSceneCfg
from ..configs.algorithms import PPODefaultCfg
from .compatible_legged_robot import CompatibleLeggedRobot
from . import utils


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
        self.absolute_plant_locations: torch.Tensor = torch.tensor([])
        """
        Absolute locations of plants in each environment
        shape: (|environments| x |plants_per_env| x 3)
        (Attribute is instantiated in self._place_static_objects())
        """
        self.absolute_obstacle_locations: torch.Tensor = torch.tensor([])

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # for language server purposes only
        self.cfg: GO2DefaultCfg = self.cfg

    # add custom rewards... here (use your robot_cfg for control)


class HighLevelPlantPolicyLeggedRobot(CompatibleLeggedRobot):
    def __init__(
            self, cfg: GO2DefaultCfg, sim_params, physics_engine, sim_device, headless
    ):
        self._prepare_camera(cfg.camera)

        self.absolute_plant_locations: torch.Tensor = torch.tensor([])
        """
        Absolute locations of plants in each environment
        shape: (|environments| x |plants_per_env| x 3)
        (Attribute is instantiated in self._place_static_objects())
        """
        self.absolute_obstacle_locations: torch.Tensor = torch.tensor([])

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # for language server purposes only
        self.cfg: GO2HighLevelPlantPolicyCfg = self.cfg

        # Load low-level policy
        self.low_level_policy = utils.load_low_level_policy(cfg, sim_device)
        # use `self.low_level_policy.act_inference(observations)`
        # or self.low_level_policy.actor(observations) to get an action.
        # See `ActorCritic` in rsl_rl/modules/actor_critic.py
        self.detected_objects = self._detect_objects()

        # Make the past action visible to the high level agent
        self.past_actions = torch.zeros((self.num_envs, 3)).to(self.device)


    def _prepare_camera(self, camera):
        print("Preparing")
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.horizontal_fov = camera.horizontal_fov
        self.camera_props.width = camera.width
        self.camera_props.height = camera.height
        self.camera_props.enable_tensors = camera.enable_tensors
        self.camera_props.use_collision_geometry = True
        self.third_image_index = camera.height // 3


    def _init_buffers(self):
        super()._init_buffers()
        #  self.obs_buf is for low level
        self.obs_buf = torch.zeros(self.num_envs, self.cfg.low_level_policy.num_observations, device=self.device,
                                   dtype=torch.float)

        #  overwrites default initializations for compatibilty with low-level policy interactions
        self.torques = torch.zeros(
            self.num_envs,
            self.cfg.low_level_policy.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.cfg.low_level_policy.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.cfg.low_level_policy.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        #  added high_level_actions buffer
        self.high_level_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def _init_gain_buffers(self):
        """overwrites default initializations for compatibilty with low-level policy interactions (moved outside for super() call)
        """
        self.p_gains = torch.zeros(
            self.cfg.low_level_policy.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.cfg.low_level_policy.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )


    def compute_low_level_observations(self, high_level_actions):
        """
        Computes observations, including distances and angles to detected plants.
        Args:
            high_level_actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Base observation components combined with plant-related features
        self.past_actions = high_level_actions
        self.low_level_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                high_level_actions,  # * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

    # computes high level observations
    def compute_observations(self):
        """ Computes observations
        """
        # Call object detection method
        self.detected_objects = self._detect_objects()
        plants_across_envs = [obj["plants"] for obj in self.detected_objects]

        plant_probability = utils.convert_object_property(plants_across_envs, "probability", self.device).unsqueeze(1)
        plant_distances = utils.convert_object_property(plants_across_envs, "distance", self.device).unsqueeze(1)
        plant_angles = utils.convert_object_property(plants_across_envs, "angle", self.device).unsqueeze(1)

        # Distance sensors WITH ACCESS AND END ACCESS IT actually gets GPU tensors
        self.gym.start_access_image_tensors(self.sim)
        depth_information = -torch.stack([
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i], self.cameras[i], gymapi.IMAGE_DEPTH
                )
            ) for i in range(len(self.cameras))
        ])
        self.gym.end_access_image_tensors(self.sim)
        # USE DEPTH INFORMATION TO CALCULATE UPPER AND LOWER IMAGE MIN VALUE
        # upper_image_min = depth_information[:, :self.half_image_idx, :].min(dim=1).values
        # lower_image_min = depth_information[:, self.half_image_idx:, :].min(dim=1).values
        third_image = depth_information[:, self.third_image_index:2*self.third_image_index, :].min(dim=1).values
        observable_depth_information = torch.tanh(third_image)
        # to have a separate policy trained without depth sensing as a backup
        observable_depth_information = torch.zeros_like(observable_depth_information).to(self.device)
        self.obs_buf = torch.cat((plant_probability,
                                  torch.mul(plant_distances, plant_probability),
                                  torch.mul(plant_angles, plant_probability),
                                  observable_depth_information,
                                  self.past_actions,
                                  ), dim=-1)

    # add custom rewards... here (use your robot_cfg for control)

    # computes high level observations
    def compute_observations(self):
        """ Computes observations
        """
        # Call object detection method
        self.detected_objects = self._detect_objects()
        plants_across_envs = [obj["plants"] for obj in self.detected_objects]

        plant_probability = utils.convert_object_property(plants_across_envs, "probability", self.device).unsqueeze(1)
        plant_distances = utils.convert_object_property(plants_across_envs, "distance", self.device).unsqueeze(1)
        plant_angles = utils.convert_object_property(plants_across_envs, "angle", self.device).unsqueeze(1)

        # Distance sensors WITH ACCESS AND END ACCESS IT actually gets GPU tensors
        self.gym.start_access_image_tensors(self.sim)
        depth_information = -torch.stack([
            gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i], self.cameras[i], gymapi.IMAGE_DEPTH
                )
            ) for i in range(len(self.cameras))
        ])
        self.gym.end_access_image_tensors(self.sim)
        # USE DEPTH INFORMATION TO CALCULATE UPPER AND LOWER IMAGE MIN VALUE
        # upper_image_min = depth_information[:, :self.half_image_idx, :].min(dim=1).values
        # lower_image_min = depth_information[:, self.half_image_idx:, :].min(dim=1).values
        third_image = depth_information[:, self.third_image_index:2 * self.third_image_index, :].min(dim=1).values
        observable_depth_information = torch.tanh(third_image)
        self.obs_buf = torch.cat((  # self.base_lin_vel * self.obs_scales.lin_vel,
            # self.projected_gravity,
            plant_probability,
            torch.mul(plant_distances, plant_probability),
            torch.mul(plant_angles, plant_probability),
            observable_depth_information,
        ), dim=-1)

    # add custom rewards... here (use your robot_cfg for control)
    def _reward_sanity_check(self):
        # Tracking of angular velocity commands (yaw)
        # Provide slight reward for moving ahead
        ang_vel_error = torch.square(1.0 - self.base_ang_vel[:, 2])  # base_ang_vel
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_minimal_policy(self):
        # Tracking of angular velocity commands (yaw)
        # Provide slight reward for moving ahead
        ang_vel_error = torch.square(self.base_ang_vel[:, 2])
        lin_vel_error = torch.square(self.base_lin_vel[:, :2]).mean(dim=1)
        return ang_vel_error + lin_vel_error + 0.05

    def _reward_smooth_commands(self, threshold=0.25):
        # Difference between subsequent policies should below a threshold of 0.2
        difference_in_commands = torch.abs(
            self.high_level_actions_prev1.float() - self.high_level_actions_prev2.float())
        difference_in_commands_with_threshold = torch.where(difference_in_commands > threshold,
                                                            (difference_in_commands - threshold).float(), 0.0).float()
        return difference_in_commands_with_threshold.mean(dim=1)

    def _reward_plant_closeness(self):
        # Tracking of angular velocity commands (yaw)
        plants_across_envs = [obj["plants"] for obj in self.detected_objects]

        # TODO: improve reward
        plant_probability = utils.convert_object_property(plants_across_envs, "probability", self.device)
        plant_distances = utils.convert_object_property(plants_across_envs, "distance", self.device)

        # combined_reward = torch.exp(-plant_distances) * plant_probability
        # combined_reward += 10 * torch.exp(-plant_distances * 10.) * plant_probability

        # combined_reward = torch.mul(torch.exp(-plant_distances.float() ** 2).float(), plant_probability)
        combined_reward = (torch.exp(-plant_distances.float() * 2.5).float() +
                           torch.exp(-plant_distances.float() * 25.).float())
        return torch.mul(combined_reward, plant_probability)

    def _reward_plant_ahead(self):
        # Tracking of angular velocity commands (yaw)
        plants_across_envs = [obj["plants"] for obj in self.detected_objects]
        # TODO: improve reward
        plant_probability = utils.convert_object_property(plants_across_envs, "probability", self.device)
        plant_angles = utils.convert_object_property(plants_across_envs, "angle", self.device)
        reward = (torch.exp(-torch.abs(plant_angles).float() * 100.).float().to(self.device).float() +
                  torch.exp(-torch.abs(plant_angles).float() * 1000.).float().to(self.device).float())
        return torch.mul(reward, plant_probability)

    def _reward_obstacle_closeness(self):
        # Tracking of angular velocity commands (yaw)
        obstacles_across_envs = [obj["obstacles"] for obj in self.detected_objects]

        # TODO: improve reward
        obstacle_probability = utils.convert_object_property(obstacles_across_envs, "probability", self.device)
        obstacle_distances = utils.convert_object_property(obstacles_across_envs, "distance", self.device)
        obstacle_angles = utils.convert_object_property(obstacles_across_envs, "angle", self.device)
        return (obstacle_distances < 1.5).float() * torch.exp(-obstacle_distances) * obstacle_probability

    def _detect_objects(self):
        """Detects objects in the environment and classifies them into obstacles and plants/targets.
        Additionally, computes angle and distance from the robot to each detected object.
        Only objects within the robot's field of view (120 degrees in both axes) are detected.

        Returns:
            dict: A dictionary containing detected objects for each environment, their classification, distances, and angles. (only returns most likely obstacle and plant)
        """
        detected_objects = []
        fov_angle = torch.deg2rad(torch.tensor(120.0 / 2))  # Half of 120 degrees in radians

        for env_idx in range(self.num_envs):
            robot_position = self.base_pos[env_idx]  # Get robot position for the current environment
            robot_orientation = self.rpy[env_idx, 2]  # Get robot yaw (orientation) for the current environment

            obstacles = [utils.get_dummy_object_observation(self.device)]
            plants = [utils.get_dummy_object_observation(self.device)]

            if len(self.absolute_plant_locations):
                for plant_location in self.absolute_plant_locations[env_idx]:
                    distance, angle = utils.get_distance_and_angle(robot_position, robot_orientation, plant_location)
                    # TODO: use a better way of linking distance to a reduced prediction probability
                    probability = 1.0
                    plants.append(utils.get_object_observation(plant_location, distance, angle, probability, fov_angle))

            if len(self.absolute_obstacle_locations):
                for obstacle_location in self.absolute_obstacle_locations[env_idx]:
                    distance, angle = utils.get_distance_and_angle(robot_position, robot_orientation, obstacle_location)
                    # TODO: use a better way of linking distance to a reduced prediction probability
                    probability = 1.0
                    # if np.random.random()>0.99: print("angle", angle)
                    obstacles.append(
                        utils.get_object_observation(obstacle_location, distance, angle, probability, fov_angle))

            # Block plant visibility by obstacles, if in front
            threashold = 10  # degrees
            for plant_index, plant in enumerate(plants):
                for obstacle_index, obstacle in enumerate(obstacles):
                    if obstacle["angle"].abs() < threashold:
                        plants[plant_index]["probability"] = 0.0
                        # print("threashold", env_idx)

            detected_objects.append({
                "env_idx": env_idx,
                "obstacles": sorted(obstacles, key=lambda p: p["probability"], reverse=True)[:1],
                "plants": sorted(plants, key=lambda p: p["probability"], reverse=True)[:1]
            })

        return detected_objects

    def step(self, high_level_actions: torch.Tensor):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            high_level_actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Here we get high level actions and need to translate them to low level actions
        # actions are always low_level (dim=12) and high_level_actions are high level (dim=5) # TODO update dims
        """
        # fixed dummy actions for testing
        high_level_actions = torch.zeros_like(high_level_actions, device=self.device)
        high_level_actions[:, 0] = 2.0
        """

        bounded_high_level_actions = torch.tanh(high_level_actions)
        self.high_level_actions = bounded_high_level_actions

        for _ in range(self.cfg.low_level_policy.steps_per_high_level_action):
            self.compute_low_level_observations(bounded_high_level_actions)
            actions = self.low_level_policy.act_inference(self.low_level_obs_buf)
            info = super().step(actions)
        return info

    def _create_envs(self):
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
