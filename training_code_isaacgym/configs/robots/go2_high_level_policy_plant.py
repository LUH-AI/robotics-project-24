import numpy as np

from isaacgym import gymapi
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from .go2_default import GO2DefaultCfg

from ..scenes import BaseSceneCfg
import os

# this is the GO2RoughCfg copied from unitree_rl_gym repo (do not change, create a new file)
class GO2HighLevelPlantPolicyCfg(LeggedRobotCfg):
    name = "go2_default-high-level-policy_plant"

    class low_level_policy:
        path = "./path/to/low_level_policy"  # [TODO: this is not properly set]
        num_observations = 48
        num_actions = 12
        model_path = os.path.join(os.path.dirname(__file__), "../models/model.pt")
    # Overwrite env from LeggedrobotCfg
    class env:
        num_envs = 32
        num_observations = 6  # [TODO: this is not properly set]
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 3  # [TODO: this is not properly set]
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        test = False


    class env(LeggedRobotCfg.env):
        num_observations = 50

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # only rewards that have a scale will be added (reward is named "_reward_{SCALE_NAME}")
            # [TODO: remove these reward scales, as their reward is not relevant for the high level policy anymore (left for now to h ave dummy values)]
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
