from legged_gym.utils import task_registry
from environments.legged_robot import LeggedRobot
from environments.toy_example import GO2ToyCfg, GO2ToyCfgPPO

task_registry.register("toy_example", LeggedRobot, GO2ToyCfg(), GO2ToyCfgPPO())
