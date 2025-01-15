import numpy as np

from isaacgym import gymapi
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from .go2_default import GO2DefaultCfg

from ..scenes import BaseSceneCfg


class GO2HighLevelPlantPolicyCfg(GO2DefaultCfg):
    name = "go2_default-high-level-policy_plant"

    class env(LeggedRobotCfg.env):
        num_observations = 50

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0

    # robot camera:
    class camera:
        horizontal_fov = 120
        width = 12
        height = 1
        enable_tensors = True
        vec_from_body_center = gymapi.Vec3(0.34, 0, 0.021) # Should be closest to reality: (0.34, 0, 0.021)m
        rot_of_camera = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), np.radians(0)
        )
