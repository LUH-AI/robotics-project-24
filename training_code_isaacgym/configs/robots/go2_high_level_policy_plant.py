from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

from ..scenes import BaseSceneCfg


# this is the GO2RoughCfg copied from unitree_rl_gym repo (do not change, create a new file)
class GO2HighLevelPlantPolicyCfg(LeggedRobotCfg):
    name = "go2_high-level-plant-policy"

    class low_level_policy:
        path = "./path/to/low_level_policy"  # [TODO: this is not properly set]
        num_observations = 48
        num_actions = 12

    # Overwrite env from LeggedrobotCfg
    class env:
        num_envs = 32
        num_observations = 48  # [TODO: this is not properly set]
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12  # [TODO: this is not properly set]
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        test = False


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
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
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            # only rewards that have a scale will be added (reward is named "_reward_{SCALE_NAME}")
            # [TODO: remove these reward scales, as their reward is not relevant for the high level policy anymore (left for now to h ave dummy values)]
            torques = -0.0002
            dof_pos_limits = -10.0

    # for language server purposes (the selected scene config is added automatically)
    class scene(BaseSceneCfg):
        pass
