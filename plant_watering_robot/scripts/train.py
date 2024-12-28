from plant_watering_robot.envs import *
from plant_watering_robot.envs.base.legged_robot import LeggedRobot
from plant_watering_robot.envs.robots.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from plant_watering_robot.envs.base.plant_watering_robot import PlantWateringRobot
from plant_watering_robot.envs.base.plant_watering_robot_config import PlantWateringRobotCfg, PlantWateringRobotCfgPPO


import os
import numpy as np
from datetime import datetime
import sys
import isaacgym

from plant_watering_robot.utils.helpers import get_args
from plant_watering_robot.utils.task_registry import *
import torch


task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register_high_lvl_task( "plant_watering", PlantWateringRobot, PlantWateringRobotCfg(), GO2RoughCfg(), PlantWateringRobotCfgPPO())

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
