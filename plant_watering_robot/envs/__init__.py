from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.go2.go2_v2_config import GO2RoughV2Cfg, GO2RoughCfgPPO

from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot
from .base.legged_robot_2 import LeggedRobot_v2
from .base.plant_watering_robot import PlantWateringRobot
from .base.plant_watering_robot_config import PlantWateringRobotCfg
from .base.plant_watering_robot_config import PlantWateringRobotCfgPPO
from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register_high_lvl_task( "plant_watering", PlantWateringRobot, PlantWateringRobotCfg(), GO2RoughCfg(), PlantWateringRobotCfgPPO())
