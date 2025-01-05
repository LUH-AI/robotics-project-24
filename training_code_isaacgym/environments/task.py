from typing import Any, List

import torch
import cv2
import os

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
    robot_cfg: GO2DefaultCfg, scene_cfg: BaseSceneCfg, algorithm_cfg: PPODefaultCfg
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
    task_registry.register(name, CustomLeggedRobot, robot_cfg, algorithm_cfg)
    return name


class CustomLeggedRobot(CompatibleLeggedRobot):
    def __init__(
        self, cfg: GO2DefaultCfg, sim_params, physics_engine, sim_device, headless
    ):
        self._prepare_camera(cfg.camera)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # for language server purposes only
        self.cfg: GO2DefaultCfg = self.cfg
        # Parameters for video
        frame_width = cfg.camera.width  # Replace with your actual frame width
        frame_height = cfg.camera.height  # Replace with your actual frame height
        fps = 30  # Frames per second

        # Initialize video writer
        self.output_file = "testmovie.mp4"
        self.video_writer = cv2.VideoWriter(
            self.output_file,  # Output file name
            cv2.VideoWriter_fourcc(*'mp4v'),  # Codec
            fps,  # Frames per second
            (frame_width, frame_height)  # Frame size
        )
        self.frames = []

    def __del__(self):
        if self.video_writer:
            print("Destructor called, releasing video writer...")
            self.video_writer.release()
            print(f"Video saved as {self.output_file}")

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
        image = self.gym.get_camera_image(
            self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR
        )
        print(image)
        image = np.reshape(image, (self.cfg.camera.height, self.cfg.camera.width, 4))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        self.video_writer.write(image_bgr)

        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        self.cameras = []
        self.object_handles: List[List[Any]] = []  # TODO add type for handle
        self.object_assets: List[Any] = []  # TODO add type for gym asset
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
                1
            )
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            # ADD Camera Functionality
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
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            # add static objects/ actors to environment
            # function is only
            self._place_static_objects(i, env_handle)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

    # add custom rewards... here (use your robot_cfg for control)
