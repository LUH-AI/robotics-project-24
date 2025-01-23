import numpy as np
from pathlib import Path

from isaacgym import gymapi
from .go2_default import GO2DefaultCfg


class GO2HighLevelPlantPolicyCfg(GO2DefaultCfg):
    name = "go2_high-level-policy_plant"

    class asset(GO2DefaultCfg.asset):
        file = str(Path(__file__).parents[2] / "assets/robots" / "go2_with_watering" / "urdf/go2.urdf")

    class low_level_policy:
        path = "./path/to/low_level_policy"  # [TODO: this is not properly set]
        num_observations = 48
        num_actions = 12
        model_path = os.path.join(os.path.dirname(__file__), "../models/model.pt")

    # Overwrite env from LeggedrobotCfg
    class env:
        num_envs = 32
        num_observations = 9 + 12 * 2  # [TODO: this is not properly set]
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 3  # [TODO: this is not properly set]
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 8  # episode length in seconds
        test = False


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.33]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_hip_joint": 0.1,  # [rad]
            "RL_hip_joint": 0.1,  # [rad]
            "FR_hip_joint": -0.1,  # [rad]
            "RR_hip_joint": -0.1,  # [rad]
            "FL_thigh_joint": 0.8,  # [rad]
            "RL_thigh_joint": 1.0,  # [rad]
            "FR_thigh_joint": 0.8,  # [rad]
            "RR_thigh_joint": 1.0,  # [rad]
            "FL_calf_joint": -1.5,  # [rad]
            "RL_calf_joint": -1.5,  # [rad]
            "FR_calf_joint": -1.5,  # [rad]
            "RR_calf_joint": -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {"joint": 20.0}  # [N*m/rad]
        damping = {"joint": 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "./training_code_isaacgym/resources/robots/go2/urdf/go2.urdf"
        # file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        # Custom rewards in order to walk to and reach the plant
        # sanity_check = 10.
        plant_closeness = 5.0
        plant_ahead = 5.0
        obstacle_closeness = 10.0

        class scales(LeggedRobotCfg.rewards.scales):
            # only rewards that have a scale will be added (reward is named "_reward_{SCALE_NAME}")
            # [TODO: remove these reward scales, as their reward is not relevant for the high level policy anymore
            #  TODO: (left for now to have dummy values)]
            torques = -0.0002
            dof_pos_limits = -10.0
            # sanity_check = 10.
            plant_closeness = 5.0
            plant_ahead = 5.0
            obstacle_closeness = -10.0

    # for language server purposes (the selected scene config is added automatically)
    class scene(BaseSceneCfg):
        pass

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
