import pickle as pkl
import lcm
import sys
import torch
from pathlib import Path
import argparse

sys.path.append('walk-these-ways-go2')
from go2_gym_deploy.utils.deployment_runner import DeploymentRunner
from go2_gym_deploy.envs.lcm_agent import LCMAgent
from go2_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go2_gym_deploy.utils.command_profile import *

import pathlib

# TODO: can we un-hardcode this?
# lcm多播通信的标准格式
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(policy_path, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    parameter_path = Path(policy_path) / "parameters.pkl"
    checkpoint_path = Path(policy_path) / "checkpoints"

    with parameter_path.open() as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

    print('Config successfully loaded!')

    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    from go2_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)
    print('Agent successfully created!')

    policy = make_policy(checkpoint_path)
    print('Policy successfully loaded!')

    # load runner
    root = f"./deployment_logs"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

def make_policy(checkpoint_path):
    # try ------------------
    # body = torch.jit.load(logdir + '/checkpoints/body_latest.jit').to('cpu')
    body = torch.jit.load(checkpoint_path / 'checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(checkpoint_path / 'checkpoints/adaptation_module_latest.jit').to('cpu')

    def policy(obs, info):
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, default="example_policies/default")
    parser.add_argument("--experiment_name", type=str, default="example_experiment")
    parser.add_argument("--max_vel", type=float, default=2.5)
    parser.add_argument("--max_yaw_vel", type=float, default=5.0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # default:
    # max_vel=3.5, max_yaw_vel=5.0
    load_and_run_policy(args.policy_path, experiment_name=args.experiment_name, max_vel=args.max_vel, max_yaw_vel=args.max_yaw_vel)
