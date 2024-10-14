import gymnasium as gym
from training_code.new_go2_config_class import UnitreeGo2FlatAirtimeEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-HeinrichAirtime-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnitreeGo2FlatAirtimeEnvCfg,
    },
)