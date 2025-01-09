# Plant Watering Robot - Policy Training in Simulation

## Configurations
You can change/add configurations for the used RL-algorithm, the scene or the robot:
1. Go to configs/[algorithms, robots, scenes]
2.  **Create a new file** by e.g. copying the respective default config
3. Specify the name attribute of your new configuration
4. Import the configuration file/ class in the corresponding `__init__.py` file
5. Add the config name and your class to the respective dictionary in `train.py`

There is a predefined StaticObject class in configs/scenes/base.py that should be used in your scene configuration to place static objects in addition to the robot inside the scene.

## Assets
You can add custom assets in assets/ to change the robot or your scenes. Therefore, create a new folder and add all assets that you need for your scene/ robot.

## Robot Class
If you need to do more advanced changes than those that can be done by manipulating the configuration files (e.g. reward shaping), you can adapt the robot/environment classes in `environments/`
**Almost all changes should be done in `task.py` or eventually in `task_utils.py`**. Only adapt code in compatible_legged_robot.py if changes need many modifications in existing code or to prevent duplicated code in low-level and high-level policy classes.

### Low-Level Policy
The low-level policy should get high-level actions and the robot joint states as observations and should control the robot joints.
The low-level policy is implemented in `environments/task.py`, which contains `CustomLeggedRobot`. Add custom rewards in this class and make other needed modifications in there.
If you use additional parameters make them configurable in the configuration in a new file `configs/robots/go2_low_level_policy_plant.py`.

### High-Level Policy
The high-level policy should get sensory data as well like object detection/cameras from the environment and should control the low-level policy to perform actions. 
Most of the changes should be made in `environments/task.py`, which contains `HighLevelPlantPolicyLeggedRobot`. It contains important functions like `step`, `compute_observations` and `get_observations`, which are needed to properly interact with the low-level policy.
Also custom rewards can be added manually in that class.
If you use additional parameters make them configurable in the configuration in `configs/robots/go2_high_level_policy_plant.py`.

## PPO

There are classes in `configs/algorithms/ppo_plant.py` for both policies, which inherit the defaults from `configs/algorithms/ppo_default.py`.
