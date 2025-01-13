import os

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
import torch


class LowLevelModule:
    def __init__(self, low_lvl_model_path=None, num_low_lvl_observations=52, num_low_lvl_actions=12,
                 command_observation_index=9, device="cpu"):
        # Loads the separately trained low level policy in order to be used as a non-trained policy module
        self.device = device
        if low_lvl_model_path is None:  # not the best way to setup a default path, could/ should be changed later
            low_lvl_model_path = os.path.abspath(
                os.path.join(os.getcwd(), "..", "..", "policies", "low_lvl_model_v0.pt"))
        self.num_low_lvl_observations = num_low_lvl_observations
        self.num_low_lvl_actions = num_low_lvl_actions  # Should be initialized with self.env.num_actions
        self.command_observation_index = command_observation_index
        actor_critic_class_low = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic_low: ActorCritic = actor_critic_class_low(self.num_low_lvl_observations,
                                                               self.num_low_lvl_observations,
                                                               self.num_low_lvl_actions,
                                                               **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg_low: PPO = alg_class(actor_critic_low, device=self.device, **self.alg_cfg)
        loaded_dict_low = torch.load(low_lvl_model_path)
        self.alg_low.actor_critic.load_state_dict(loaded_dict_low['model_state_dict'])
        self.alg_low.actor_critic.eval()
        if self.device is not None:
            self.alg_low.actor_critic.to(self.device)

    def apply(self, observation, commands=None):
        observation_low_level = observation[:, :self.num_low_lvl_observations]
        if commands is not None:
            observation_low_level[:, self.command_observation_index:
                                     self.command_observation_index + commands.shape[1]] = commands
        return self.alg_low.actor_critic.act_inference(observation_low_level).detach()
