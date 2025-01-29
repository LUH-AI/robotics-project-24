import numpy as np
from pathlib import Path

from isaacgym import gymapi
from .go2_default import GO2DefaultCfg


class GO2HighLevelPlantPolicyCfg(GO2DefaultCfg):
    name = "go2_high-level-policy_plant"

    class asset(GO2DefaultCfg.asset):
        file = str(Path(__file__).parents[2] / "assets/robots" / "go2_with_watering" / "urdf/go2.urdf")

    class low_level_policy:
        path = Path(__file__).parents[1] / "models" / "model.pt"
        num_observations = 48
        num_actions = 12
        steps_per_high_level_action = 4

    class env(GO2DefaultCfg.env):
        num_envs = 64
        num_observations = 3 + 12  # [TODO: this is not properly set]
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 3  # [TODO: this is not properly set]
        episode_length_s = 8  # episode length in seconds

    class init_state(GO2DefaultCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]

    class rewards(GO2DefaultCfg.rewards):
        # Parameters for custom rewards HERE

        class scales():
            # only rewards that have a scale will be added (reward is named "_reward_{SCALE_NAME}")
            # sanity_check = 10.
            plant_closeness = 2.0
            plant_ahead = 1.0
            obstacle_closeness = 0.0  # TODO or -10 from upper values
            minimize_rotation = 0.

    # robot camera:
    class camera:
        horizontal_fov = 120
        width = 12
        height = 720
        enable_tensors = True
        vec_from_body_center = gymapi.Vec3(0.34, 0, 0.021)  # Should be closest to reality: (0.34, 0, 0.021)m
        rot_of_camera = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), np.radians(0)
        )
