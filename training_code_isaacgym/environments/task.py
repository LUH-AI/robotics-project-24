from typing import Callable

import torch

from isaacgym import gymapi

from legged_gym.utils.task_registry import task_registry

from ..configs.robots import GO2DefaultCfg
from ..configs.scenes import BaseSceneCfg
from ..configs.algorithms import PPODefaultCfg
from .compatible_legged_robot import CompatibleLeggedRobot
from rsl_rl.modules import ActorCritic
from .util import load_low_level_policy
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

        # Load low-level policy
        self.low_level_policy = load_low_level_policy(cfg, sim_device)
        # use `self.low_level_policy.act(observations)` to get an action. See `ActorCritic` in rsl_rl/modules/actor_critic.py
	
        self.absolute_plant_locations: torch.Tensor
        """
        Absolute locations of plants in each environment
        shape: (|environments| x |plants_per_env| x 3)
        (Attribute is instantiated in self._place_static_objects())
        """

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


    def compute_observations(self):
        """Computes observations, including distances and angles to detected plants."""
        # Call object detection method
        detected_objects = self._detect_objects()
        
        # Collect plant-related features directly from detected objects
        plants = [plant for obj in detected_objects for plant in obj["plants"]]
    
        plant_distances = torch.tensor([plant["distance"] for plant in plants], device=self.device).unsqueeze(1)
        plant_angles = torch.tensor([plant["angle"] for plant in plants], device=self.device).unsqueeze(1)
    
        # Base observation components combined with plant-related features
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
                                    self.actions,
                                    plant_distances,
                                    plant_angles,
                                    ),dim=-1)
        print(f"Computed low level observations: {self.obs_buf.shape}")
    
        # Add noise if required
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

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
    
                # Check if the plant is within the robot's field of view (FOV)
                if torch.abs(angle) <= fov_angle:
                    plants.append({
                        "location": plant_location,
                        "distance": distance,
                        "angle": angle
                    })
                else:
                    # TODO think about best padding value for plants outside fov / (handling environments with different numbers of objects)
                    plants.append({
                        "location": plant_location,
                        "distance": -1,
                        "angle": -1
                    })

            detected_objects.append({
                "env_idx": env_idx,
                "obstacles": obstacles,  # No obstacles handled yet, placeholder for future extension
                "plants": plants
            })

        return detected_objects

    def step(self, high_level_actions):
        """
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
