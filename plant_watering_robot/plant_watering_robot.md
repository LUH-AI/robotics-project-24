

# Plant watering robot

- all necessary files are in plant_watering_robot
- the task is designed in plant_watering_robot/envs/base/plant_watering_robot and plant_watering_robot/envs/base/plant_watering_robot_config
- The plant watering robot gets two config files
	- **low_level_cfg**: LeggedRobotCfg for the low level actions
		- specifies the dof init positions
		- num_actions for low lvl policy
		- action_rate for low lvl policy
		- everything else for low lvl policy
	- **cfg**: PlantWateringRobotCfg for the high level task
		- right now is more or less just a copy of the low lvl config
		- should specify all params for plant watering task
	- ``class  PlantWateringRobot(BaseTask):
def  __init__(self, low_level_cfg: LeggedRobotCfg, cfg: PlantWateringRobotCfg, sim_params, physics_engine, sim_device, headless):``

## Structure
- learning algo calls step(self, actions) function in every iteration
- the outputted actions are applied to low_lvl_policy
	- right now those are 3 actions - lateral movement, vertical movement, turning
- low lvl policy outputs dof actions 
	- those are 12 actions - for 12 joints 
- call _compute_torques() and gym.set_dof_actuation_force_tensor() to apply robot movement to simulation
- call _post_physics_step_callback()
	- checks for termination
	- conputes rewards
	- computes high and low lvl observations
		- compute_observations() - high lvl
		- compute_low_level_observations() - low lvl

## TODOs
- implement high lvl commands into observation for high lvl policy in compute_observations()
- adjust compute_low_level_observations() so we can use the new low lvl policy from thor
- design high lvl task


