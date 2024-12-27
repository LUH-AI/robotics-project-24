from .robots.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .base.plant_watering_robot import PlantWateringRobot
from .base.plant_watering_robot_config import PlantWateringRobotCfg
from .base.plant_watering_robot_config import PlantWateringRobotCfgPPO
from plant_watering_robot.utils.task_registry import task_registry
task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register_high_lvl_task( "plant_watering", PlantWateringRobot, PlantWateringRobotCfg(), GO2RoughCfg(), PlantWateringRobotCfgPPO())