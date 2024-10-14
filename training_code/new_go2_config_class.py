from omni.isaac.lab.utils import configclass
from standard_go2_config_classes import UnitreeGo2RoughEnvCfg

@configclass
class UnitreeGo2FlatAirtimeEnvCfg(UnitreeGo2RoughEnvCfg):
    """This environment rewards airtime a lot!"""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 10.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None