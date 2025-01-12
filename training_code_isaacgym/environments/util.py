import torch
from rsl_rl.modules import ActorCritic
from ..configs.robots.go2_high_level_policy_plant import GO2HighLevelPlantPolicyCfg


def load_low_level_policy(cfg: GO2HighLevelPlantPolicyCfg, sim_device):
    module = ActorCritic(
        num_actor_obs=cfg.low_level_policy.num_observations,
        num_critic_obs=cfg.low_level_policy.num_observations,
        num_actions=cfg.low_level_policy.num_actions,
        actor_hidden_dims=[512, 256, 128], 
        critic_hidden_dims=[512, 256, 128],  
    )
    module = module.to(sim_device)
    checkpoint = torch.load(cfg.low_level_policy.model_path)

    model_state_dict = checkpoint.get('model_state_dict')
    if model_state_dict is None:
        raise ValueError("The checkpoint does not contain a 'model_state_dict' key.")

    try:
        module.load_state_dict(model_state_dict)
    except RuntimeError as e:
        print("\nError while loading state dictionary:")
        print(e)
        return None

    return module
