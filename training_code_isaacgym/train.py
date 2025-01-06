from typing import Tuple

from isaacgym import gymutil

from legged_gym.envs import *
from legged_gym.utils import task_registry

from .environments import task
from .configs import (
    robots as robot_configs,
    scenes as scene_configs,
    algorithms as alg_configs,
)


def get_args():
    custom_parameters = [
        # {"name": "--task", "type": str, "default": "go2", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {
            "name": "--robot",
            "type": str,
            "default": "go2_default",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--scene",
            "type": str,
            "default": "ground_plane",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--algorithm",
            "type": str,
            "default": "ppo_default",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--resume",
            "action": "store_true",
            "default": False,
            "help": "Resume training from a checkpoint",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--run_name",
            "type": str,
            "help": "Name of the run. Overrides config file if provided.",
        },
        {
            "name": "--load_run",
            "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Maximum number of training iterations. Overrides config file if provided.",
        },
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy", custom_parameters=custom_parameters
    )

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def get_configs(
    args,
) -> Tuple[
    robot_configs.GO2DefaultCfg, scene_configs.BaseSceneCfg, alg_configs.PPODefaultCfg
]:
    robots = {
        "go2_default": robot_configs.GO2DefaultCfg(),
    }
    scenes = {
        "ground_plane": scene_configs.BaseSceneCfg(),
        "empty_room_10x10": scene_configs.EmptyRoom10x10Cfg(),
        "empty_room_5x5": scene_configs.EmptyRoom5x5Cfg(),
        "plant_environment": scene_configs.PlantEnvironmentCfg(),
        "single_plant": scene_configs.SinglePlantCfg(),
    }
    algorithms = {
        "ppo_default": alg_configs.PPODefaultCfg(),
    }
    return robots[args.robot], scenes[args.scene], algorithms[args.algorithm]


def train(task_name, args):
    env, env_cfg = task_registry.make_env(name=task_name, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=task_name, args=args
    )
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    configs = get_configs(args)

    task_name = task.register_task(*configs)
    train(task_name, args)
