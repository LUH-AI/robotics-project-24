from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot

import configs
class CustomLeggedRobot(LeggedRobot):
    pass


# register all tasks derived from CustomLeggedRobot
task_registry.register( "go2_default", LeggedRobot, configs.GO2RoughCfg(), configs.GO2RoughCfgPPO())
