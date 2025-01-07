import torch
from rsl_rl.modules import ActorCritic
from .configs import GO2HighLevelPlantPolicyCfg


def load_low_level_policy(cfg: GO2HighLevelPlantPolicyCfg):
    module = ActorCritic(cfg.low_level_policy.num_observations, cfg.low_level_policy.num_observations, cfg.low_level_policy.num_actions)
    module.load_state_dict(torch.load(cfg.low_level_policy.path))
    return module
