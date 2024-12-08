from pathlib import Path

import gym

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveSceneCfg

from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.assets import AssetBaseCfg

from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.sensors import RayCasterCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.sensors import patterns

from omni.isaac.lab.envs import ManagerBasedEnvCfg
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG

# invariant to location from which script is executed
ASSET_DIRPATH = Path(__file__).parent.parent / "assets" / "robots" / "cartpole"

from . import UNITREE_GO2_CFG, GroundPlaneCfg

@configclass 
class Go2SceneCfg(InteractiveSceneCfg):
	# ground plane

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()) 
    # lights
    dome_light = AssetBaseCfg( 
    prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)) ) 

    # articulation 
    go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces  =  ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    """
    height_scanner  =  RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    """


    camera = CameraCfg(
		prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
		update_period=0.1,
		height=480,
		width=640,
		data_types=["rgb", "distance_to_image_plane"],
		spawn=sim_utils.PinholeCameraCfg(
			focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
		),
		offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
	)

@configclass
class Go2EnvCfg(ManagerBasedEnvCfg):

    # Scene settings
    scene = Go2SceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    #TODO from where and for what?
    #observations = ObservationsCfg()
    #actions = ActionsCfg()
    #events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def register() -> str:
    """Registers environment/ task

    Returns:
        str: id/name of environment
    """
    id = "Flat-Unitree-Go2-Camera-v0"
    gym.register(
        id=id,
        entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
        #disable_env_checker=True, unknown argument
        cfg={
            "env_cfg_entry_point": Go2EnvCfg,
            #TODO needed?
            #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
            #"skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        },
    )
    return id
