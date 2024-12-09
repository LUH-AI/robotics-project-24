from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot


class CustomLeggedRobot(LeggedRobot):
    # add custom rewards... here (use robot_cfg for control)
    pass


# register all tasks derived from CustomLeggedRobot
def register_task(robot_cfg, scene_cfg, algorithm_cfg) -> str:
    """Register task based on robot, scene and algorithms configurations
    All configs need a name attribute.

    Args:
        robot_cfg (_type_): Robot config
        scene_cfg (_type_): Scene config
        algorithm_cfg (_type_): Algorithm config

    Returns:
        str: Task name
    """
    name = f"{robot_cfg.name}_{scene_cfg.name}_{algorithm_cfg.name}"
    # add 
    robot_cfg.scene = scene_cfg
    algorithm_cfg.runner.experiment_name = name
    task_registry.register(name, CustomLeggedRobot, robot_cfg, algorithm_cfg)
    return name
    #task_registry.register( "go2_default", CustomLeggedRobot, configs.GO2RoughCfg(), configs.GO2RoughCfgPPO())
