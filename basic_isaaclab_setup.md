# Basic Isaaclab setup

## Importing Assets
Import our scene using USD files. Design in Isaac sim and export the USD file.

## Scene configuration
**Interactive scene:**
 - https://isaac-sim.github.io/IsaacLab/main/source/tutorials/02_scene/create_scene.html

```
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from  omni.isaac.lab_assets.unitree  import  UNITREE_GO2_CFG

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
		
	 height_scanner  =  RayCasterCfg(
		prim_path="{ENV_REGEX_NS}/Robot/base",
		offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
		attach_yaw_only=True,
		pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
		debug_vis=False,
		mesh_prim_paths=["/World/ground"],
	 )
	 
	self.camera  =  CameraCfg(
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

)
```


## Design Task Environment 

 - There are two different workflows for designing environments
 - https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html

	-  **Manager-Based Base Environment**
		- ![enter image description here](https://isaac-sim.github.io/IsaacLab/main/_images/manager-based-light.svg)
		 - environment is decomposed into individual components (or managers) that handle different aspects of the environment (such as computing observations, applying actions, and applying randomization).
		 - https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html
		 - Example structure:
```
@configclass
class Go2EnvCfg(ManagerBasedEnvCfg):

    # Scene settings
    scene = Go2SceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz
```
-  **Direct**
	- defines a single class that implements the entire environment directly without the need for separate managers.
	- https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html

## Register Environment
 - https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html
```
gym.register(
	id="Isaac-Velocity-Rough-Unitree-Go2-Camera-v0",
	entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
	disable_env_checker=True,
	kwargs={
		"env_cfg_entry_point": Go2EnvCfg,
		"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
		"skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
	},
)
```
 - makes env accessible across different Learning algorithms, params for different algorithms can be set in the corresponding config file or config class. For example UnitreeGo2PPORunnerCfg

## Wrap Environment with Learning lib
- Isaaclab provides own wrappers to convert the environment into the expected interface by the learning library
- Available libs in Isaaclab: https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_frameworks.html
- Example of wrapping ev:
```
env  =  gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array"  if  args_cli.video else  None)
env  =  gym.wrappers.RecordVideo(env, **video_kwargs)
env  =  RslRlVecEnvWrapper(env)
```

## Training
- basic training scripts for different libraries can be found under IsaacLab\source\standalone\workflows
## Sim2Real

