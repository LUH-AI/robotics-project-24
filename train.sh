#!/usr/bin/env bash

python -m training_code_isaacgym.train --algorithm "ppo_move-policy_plant" --experiment "move_policy_plant" --robot_class "go2_default_class" --robot "go2_default"
python -m training_code_isaacgym.train --algorithm "ppo_high-level-policy_plant" --experiment "high_level_policy_plant" --robot_class "go2_high-level-policy_plant_class" --robot "go2_high-level-policy_plant"
