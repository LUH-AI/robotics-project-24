from typing import List, Any
import warnings
import os
from abc import ABC, abstractmethod

import numpy as np
import torch

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from .legged_robot import LeggedRobot

from . import utils


# DO NOT MAKE MODIFICATIONS IN THIS FILE!!!
# USE task.py!
# only if legged_gym framework cannot be used at all/ multiple functions... need to be adapted


class CompatibleLeggedRobot(LeggedRobot, ABC):
    """This class should not be called directly"""

    def _place_static_objects(
        self, env_idx: int, env_handle: Any, robot_position: torch.Tensor
    ):
        """Places static objects like walls into the provided environment
        It is called in the environment creation loop in super()._create_envs()

        Args:
            env_idx (int): Index of environment
            env_handle (Any): Environment handle
            robot_position (torch.Tensor): Robot location
        """
        if not len(self.cfg.scene.static_objects):
            return
        self.object_handles.append([])

        self.num_static_objects = len(self.cfg.scene.static_objects)

        # move all tensors to device
        for static_obj in self.cfg.scene.static_objects:
            static_obj.to(self.device)

        _plant_locations = []
        _obstacle_locations = []
        other_object_locations = []
        other_object_sizes = []
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
            i = 0
            object_location = utils.calculate_random_location(
                location_offset,
                static_obj.init_location,
                static_obj.max_random_loc_offset,
            )
            # does not detect collisions of non-random objects (e.g. walls)
            while i < 100 and static_obj.max_random_loc_offset.any():
                object_location = utils.calculate_random_location(
                    location_offset,
                    static_obj.init_location,
                    static_obj.max_random_loc_offset,
                )
                if utils.validate_location(
                    static_obj,
                    object_location,
                    robot_position,
                    other_object_locations,
                    other_object_sizes,
                ):
                    break
                i += 1
            if i == 100:
                warnings.warn(
                    f"Static object could not be placed randomly without collisions ({i} tries). This can cause problems"
                )

            other_object_locations.append(object_location)
            other_object_sizes.append(static_obj.size)
            if static_obj.type == "flower_pot":
                _plant_locations.append(object_location)
            if static_obj.type == "obstacle":
                _obstacle_locations.append(object_location)


            start_pose.p = gymapi.Vec3(*object_location)

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

        plant_locations = torch.stack(_plant_locations).unsqueeze(0)
        if len(self.absolute_plant_locations):
            self.absolute_plant_locations = torch.cat(
                (self.absolute_plant_locations, plant_locations)
            )
        else:
            self.absolute_plant_locations = plant_locations

        obstacle_locations = torch.stack(_obstacle_locations).unsqueeze(0)
        if len(self.absolute_obstacle_locations):
            self.absolute_obstacle_locations = torch.cat(
                (self.absolute_obstacle_locations, obstacle_locations)
            )
        else:
            self.absolute_obstacle_locations = obstacle_locations

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
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            # add static objects/ actors to environment
            # function is only
            self._place_static_objects(i, env_handle, robot_position=pos)

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

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        actor_ids = env_ids_int32 * (getattr(self, "num_static_objects", 0) + 1)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel

        _root_states = self.root_states.clone()
        # TODO randomize object positions on every reset and not just during initialization
        reset_indices = utils.get_reset_indices(env_ids, self.num_objects)

        self.root_states_complete[reset_indices] = self.root_states_initialization[
            reset_indices
        ]
        self.root_states_complete[:: self.num_objects] = _root_states

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states_complete),
            gymtorch.unwrap_tensor(reset_indices),
            len(reset_indices),
        )

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self.root_states_complete[:: self.num_objects] = self.root_states
        indices = torch.arange(
            0,
            self.num_envs * self.num_objects,
            self.num_objects,
            dtype=torch.int32,
            device=self.device,
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states_complete),
            gymtorch.unwrap_tensor(indices),
            len(indices),
        )

    # ----------------------------------------
    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states_complete = gymtorch.wrap_tensor(actor_root_state)
        self.root_states_initialization = self.root_states_complete.clone()
        # root_states_complete includes root_states of static objects which is not desired for the following logic
        self.num_objects = getattr(self, "num_static_objects", 0) + 1
        self.root_states = self.root_states_complete[:: self.num_objects]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[: self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )[
            :, : self.num_bodies
        ]  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        #self.obs is for low level
        self.obs_buf = torch.zeros(self.num_envs, self.cfg.low_level_policy.num_observations, device=self.device, dtype=torch.float) 
        self.torques = torch.zeros(
            self.num_envs,
            self.cfg.low_level_policy.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.cfg.low_level_policy.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.cfg.low_level_policy.num_actions, dtype=torch.float, device=self.device, requires_grad=False
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
        self.high_level_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )  # TODO change this
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
