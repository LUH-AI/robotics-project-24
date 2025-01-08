# Plant Policy

The `train.sh` in the root directory provides an entrypoint to training the low-level and then the high-level policy. It contains this code:

```
python -m training_code_isaacgym.train --algorithm "ppo_move-policy_plant" --experiment "move_policy_plant" --robot_class "go2_default_class" --robot "go2_default"
python -m training_code_isaacgym.train --algorithm "ppo_high-level-policy_plant" --experiment "high_level_policy_plant" --robot_class "go2_high-level-policy_plant_class" --robot "go2_high-level-policy_plant"
```
The first line trains the low-level policy, while the second trains the high-level policy. Both lines call `train.py`, which does know about the code for the respective tasks. 

Most changes will probably have to be made in the Robot Class (next section).

## Robot Class

### High-Level Policy

Most of the changes should be made in `environments/task.py`, which contains `HighLevelPlantPolicyLeggedRobot`. It contains important functions like `step`, `compute_observations` and `get_observations`, which are needed to properly interact with the low-level policy.

### Low-Level Policy

**This is subject to change and may already be different.**: The low-level policy is implemented in `environments/task.py`, which contains `CustomLeggedRobot`.

## PPO

There are classes in `configs/algorithms/ppo_plant.py` for both policies, which inherit the defaults from `configs/algorithms/ppo_default.py`.

## Robot Config

There is a seperate configuration file for the robot in `configs/robots/go2_high_level_policy_plant.py`. It is probably not needed to change anything here.
