import plant_watering_robot.envs
import os
import numpy as np
from datetime import datetime
import sys
import isaacgym

from envs import *

from utils.helpers import get_args
from utils.task_registry import *
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
