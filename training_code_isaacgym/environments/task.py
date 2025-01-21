from typing import Any, Callable

import torch
from isaacgym import gymapi

from legged_gym.utils.task_registry import task_registry

from ..configs.robots import GO2DefaultCfg
from ..configs.scenes import BaseSceneCfg
from ..configs.algorithms import PPODefaultCfg
from .compatible_legged_robot import CompatibleLeggedRobot
from .utils import load_low_level_policy
from . import utils

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
        self.absolute_plant_locations: torch.Tensor = torch.tensor([])
        self.absolute_obstacle_locations: torch.Tensor = torch.tensor([])
        """
        Absolute locations of plants in each environment
        shape: (|environments| x |plants_per_env| x 3)
        (Attribute is instantiated in self._place_static_objects())
        """

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

    def _place_static_objects(self, env_idx: int, env_handle: Any, robot_position: torch.Tensor):
        """Places static objects like walls into the provided environment
        It is called in the environment creation loop in super()._create_envs()

        Args:
            env_idx (int): Index of environment
            env_handle (Any): Environment handle
            robot_position (torch.Tensor): Robot location
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
        self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                  self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)
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
        self._prepare_camera(cfg.camera)

        self.absolute_plant_locations: torch.Tensor = torch.tensor([])
        self.absolute_obstacle_locations: torch.Tensor = torch.tensor([])
        """
        Absolute locations of plants in each environment
        shape: (|environments| x |plants_per_env| x 3)
        (Attribute is instantiated in self._place_static_objects())
        """

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # for language server purposes only
        self.cfg: GO2DefaultCfg = self.cfg

        # Load low-level policy
        self.low_level_policy = load_low_level_policy(cfg, sim_device)
        # use `self.low_level_policy.act_inference(observations)`
        # or self.low_level_policy.actor(observations) to get an action.
        # See `ActorCritic` in rsl_rl/modules/actor_critic.py
        self.detected_objects = self._detect_objects()

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

    def _prepare_camera(self, camera):
        print("Preparing")
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.horizontal_fov = camera.horizontal_fov
        self.camera_props.width = camera.width
        self.camera_props.height = camera.height
        self.camera_props.enable_tensors = camera.enable_tensors
        self.camera_props.use_collision_geometry = True

    def compute_low_level_observations(self, high_level_actions):
        """
        Computes observations, including distances and angles to detected plants.
        Args:
            high_level_actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Base observation components combined with plant-related features
        self.low_level_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                high_level_actions,  # * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions
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
        # print(("plants_across_envs-", len(plants_across_envs), len(plants_across_envs[0]), len(plants_across_envs[0][0])))
        plant_probability = torch.tensor([plant["probability"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))
        plant_distances = torch.tensor([plant["distance"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))
        plant_angles = torch.tensor([plant["angle"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))

        # Distance sensors
        depth_information = - torch.tensor([self.gym.get_camera_image(
            self.sim, self.envs[i], self.cameras[i], gymapi.IMAGE_DEPTH
        ) for i in
            range(len(self.cameras))], device=self.device).squeeze(1)
        # print("depth_arrays", depth_information.shape)
        # print(depth_information[0].cpu().tolist())

        self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                  self.projected_gravity,
                                  plant_probability,
                                  torch.mul(plant_distances, plant_probability),
                                  torch.mul(plant_angles, plant_probability),
                                  depth_information,
                                  ), dim=-1)

    # dont rename, onPolicyRunner calls this
    def get_observations(self):
        # print(f"get_observations Self.obs.shape {self.obs_buf.shape}")
        # TODO: Add Distance measure to observation / preprocess it to a good range. Currently: -inf = infinitely far away and -0.1 is 10cm away
        # raise NotImplementedError
        # Here we get low level observations and need to transform them to high level observations
        # This will probably also need customization of the configuration
        return self.obs_buf

    # add custom rewards... here (use your robot_cfg for control)
    def _reward_sanity_check(self):
        # Tracking of angular velocity commands (yaw)
        # Provide slight reward for moving ahead
        ang_vel_error = torch.square(1.0 - self.base_ang_vel[:, 2])  # base_ang_vel
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)  # TODO: improve reward

    def _reward_plant_closeness(self):
        # Tracking of angular velocity commands (yaw)
        plants_across_envs = [obj["plants"] for obj in self.detected_objects]
        # print(
        #     ("plants_across_envs-", len(plants_across_envs), len(plants_across_envs[0]), len(plants_across_envs[0][0])))
        plant_probability = torch.tensor([plant["probability"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))
        plant_distances = torch.tensor([plant["distance"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))
        plant_angles = torch.tensor([plant["angle"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))
        combined_reward = torch.mul(torch.exp(-plant_distances.squeeze(1)), plant_probability.squeeze(1))
        combined_reward += 10 * torch.mul(torch.exp(-plant_distances.squeeze(1)*10.), plant_probability.squeeze(1))
        return combined_reward  # TODO: improve reward

    def _reward_obstacle_closeness(self):
        # Tracking of angular velocity commands (yaw)
        plants_across_envs = [obj["obstacles"] for obj in self.detected_objects]
        # print(
        #     ("plants_across_envs-", len(plants_across_envs), len(plants_across_envs[0]), len(plants_across_envs[0][0])))
        obstacle_probability = torch.tensor([plant["probability"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))
        obstacle_distances = torch.tensor([plant["distance"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))
        obstacle_angles = torch.tensor([plant["angle"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))
        return torch.exp(-obstacle_distances.squeeze(1)*10.0)  # TODO: improve reward

    def _reward_plant_ahead(self):
        # Tracking of angular velocity commands (yaw)
        plants_across_envs = [obj["plants"] for obj in self.detected_objects]
        # print(
        #     ("plants_across_envs-", len(plants_across_envs), len(plants_across_envs[0]), len(plants_across_envs[0][0])))
        plant_probability = torch.tensor([plant["probability"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))
        plant_distances = torch.tensor([plant["distance"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))
        plant_angles = torch.tensor([plant["angle"] for plants in plants_across_envs for plant in plants],
                                         device=self.device).view((len(plants_across_envs), len(plants_across_envs[0])))
        return torch.mul(torch.exp(-plant_angles.squeeze(1)*0.1), plant_probability.squeeze(1))  # TODO: improve reward

    def _detect_objects(self):
        """Detects objects in the environment and classifies them into obstacles and plants/targets.
        Additionally, computes angle and distance from the robot to each detected object.
        Only objects within the robot's field of view (120 degrees in both axes) are detected.

        Returns:
            dict: A dictionary containing detected objects for each environment, their classification, distances, and angles.
        """
        detected_objects = []
        fov_angle = torch.deg2rad(torch.tensor(120.0 / 2))  # Half of 120 degrees in radians

        for env_idx in range(self.absolute_plant_locations.shape[0]):
            robot_position = self.base_pos[env_idx]  # Get robot position for the current environment
            robot_orientation = self.rpy[env_idx, 2]  # Get robot yaw (orientation) for the current environment

            obstacles = []
            plants = []

            for plant_location in self.absolute_plant_locations[env_idx]:
                # Compute distance from robot to plant
                distance = torch.norm(plant_location - robot_position)

                # Compute angle from robot to plant
                relative_position = plant_location - robot_position
                angle = torch.atan2(relative_position[1], relative_position[0]) - robot_orientation
                angle = torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi  # Normalize angle to [-pi, pi]
                # TODO: use a better way of linking distance to a reduced prediction probability
                probability = torch.exp(-distance)

                # Check if the plant is within the robot's field of view (FOV)
                if torch.abs(angle) <= fov_angle:
                    plants.append({
                        "location": plant_location,
                        "probability": probability,
                        "distance": distance,
                        "angle": angle
                    })
                else:
                    plants.append({
                        "location": plant_location,
                        "probability": 0,
                        "distance": distance,
                        "angle": angle
                    })

            for obstacle_location in self.absolute_obstacle_locations[env_idx]:
                # Compute distance from robot to plant
                distance = torch.norm(obstacle_location - robot_position)
                # Compute angle from robot to plant
                relative_position = obstacle_location - robot_position
                angle = torch.atan2(relative_position[1], relative_position[0]) - robot_orientation
                angle = torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi  # Normalize angle to [-pi, pi]
                # TODO: use a better way of linking distance to a reduced prediction probability
                probability = torch.exp(-distance)

                # Check if the plant is within the robot's field of view (FOV)
                if torch.abs(angle) <= fov_angle:
                    obstacles.append({
                        "location": obstacle_location,
                        "probability": probability,
                        "distance": distance,
                        "angle": angle
                    })
                else:
                    obstacles.append({
                        "location": obstacle_location,
                        "probability": 0,
                        "distance": distance,
                        "angle": angle
                    })
            # print("plants...", len(plants))
            # print("sorted", sorted(plants, key=lambda p: p["probability"], reverse=True))
            detected_objects.append({
                "env_idx": env_idx,
                # No obstacles handled yet, placeholder for future extension
                "obstacles": sorted(obstacles, key=lambda p: p["probability"], reverse=True)[:1],
                "plants": sorted(plants, key=lambda p: p["probability"], reverse=True)[:1]
            })
        # In case the environment would not have plants, zero valued plants are predicted
        if self.absolute_plant_locations.shape[0] == 0:
            detected_objects = []
            obstacles = [{
                "location": (0, 0, 0),
                "probability": 0,
                "distance": 0,
                "angle": 0
            }]
            plants = [{
                "location": (0, 0, 0),
                "probability": 0,
                "distance": 0,
                "angle": 0
            }]
            detected_objects.append({
                "env_idx": env_idx,
                "obstacles": obstacles,  # No obstacles handled yet, placeholder for future extension
                "plants": plants
            })
        return detected_objects

    def step(self, high_level_actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            high_level_actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Here we get high level actions and need to translate them to low level actions
        # actions are always low_level (dim=12) and high_level_actions are high level (dim=5)

        self.compute_low_level_observations(high_level_actions)
        self.high_level_actions = high_level_actions

        # TODO: replace hardcoded commands with high_level_actions as input
        '''
        # Begin hardcoded high-level
        rotation_command = torch.clamp(2*torch.mul(self.obs_buf[:, 8], self.obs_buf[:, 6]) + 0.5, min=-1., max=1.)
        self.low_level_obs_buf[:, 9:12] = (torch.tensor([0, 0.0, 0.], dtype=torch.float).
                                 repeat(self.high_level_actions.shape[0],1).to(self.device))
        forward = (self.obs_buf[:, 6] > 0.25).int().float() * 1.5
        self.low_level_obs_buf[:, 11] = rotation_command
        self.low_level_obs_buf[:, 9] = forward
        '''
        # End hardcoded high-level

        actions = self.low_level_policy.act_inference(self.low_level_obs_buf)
        # for _ in range(self.secondary_decimation):  # TODO: use better approach to ameliorate the reward allocation problem
        obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras = super().step(actions)
        return self.obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras

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
