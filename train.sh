#!/usr/bin/env bash

#python -m training_code_isaacgym.train --algorithm "ppo_move-policy_plant" --run_name "move_policy_plant" --robot_class "go2_default_class" --robot "go2_low-level-policy"
python -m training_code_isaacgym.train --algorithm "ppo_high-level-policy_plant" --run_name "high_level_policy_plant" --robot_class "go2_high-level-policy_plant_class" --robot "go2_high-level-policy_plant" --scene single_plant
