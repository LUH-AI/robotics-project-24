# Unitree RL GYM for the Go2
### Cluster specific installation instruction


This is a simple example of using Unitree Robots for reinforcement learning, including the Unitree Go2

## Installation
This first part of the installation should be done on the shell, since the Cluster Remote Desktop only allows limited access to the internet for installations.

1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
   ```
   conda create -n unitree_rl_env python=3.8 -y
   conda activate unitree_rl_env
   ```
2. Install pytorch 1.10 with cuda-11.3:
   ```
   pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
   pip install --upgrade torch torchvision
   pip install --upgrade tensorboard
   pip install setuptools
   ```
   Navigate a proper installation directory for the dependencies like the following and replace <username> with the actual username on the cluster:
   ```
   cd /bigwork/<username>/user
   ```
3. Install Isaac Gym

   - Download and install Isaac Gym Preview 4 from [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)
   ```
   wget https://developer.nvidia.com/isaac-gym-preview-4
   tar -xvzf isaac-gym-preview-4.tar.gz
   cd isaacgym/python && pip install -e .
   cd /..
   cd /..
   ```
   - Try running an example 
   ```
   python isaacgym/python/examples/1080_balls_of_solitude.py
   ```
   ```
   export LD_LIBRARY_PATH=/bigwork/<username>/.conda/envs/unitree_rl_env/lib:$LD_LIBRARY_PATH
   ```
4. 
   - For troubleshooting check docs isaacgym/docs/index.html
4. Install rsl_rl (PPO implementation)

   - Clone [https://github.com/leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)
   ```
   git clone https://github.com/leggedrobotics/rsl_rl
   cd rsl_rl
   pip install -e .
   cd /..
   ```

5. Install unitree_rl_gym
   ```
   git clone https://github.com/unitreerobotics/unitree_rl_gym
   cd unitree_rl_gym
   pip install -e .
   cd /..
   ```
## Graphical interface

![Isaac Gym Setup](../robotics-project-24/figures/instruction_1.png)
2. Log into the interactive sessions website of the university cluster:
https://login.cluster.uni-hannover.de/pun/sys/dashboard/batch_connect/sessions
2. Navigate to the "Interactive Apps" field and start a new "Cluster Remote Desktop" session.
3. Select the 
* number of hours (eg. 5h)
* memory per CPU node (32GB)
* a single RTX compatible GPU (rtxa6000:1)
* cluster partition (tnt as the only available)

## Training
   ```
   python legged_gym/scripts/train.py --task=go2
   ```

   * To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
   * To run headless (no rendering) add `--headless`.
   * **Important** : To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
   * The trained policy is saved in `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
   * The following command line arguments override the values set in the config files:
   * --task TASK: Task name.
   * --resume: Resume training from a checkpoint
   * --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
   * --run_name RUN_NAME: Name of the run.
   * --load_run LOAD_RUN: Name of the run to load when resume=True. If -1: will load the last run.
   * --checkpoint CHECKPOINT: Saved model checkpoint number. If -1: will load the last checkpoint.
   * --num_envs NUM_ENVS: Number of environments to create.
   * --seed SEED: Random seed.
   * --max_iterations MAX_ITERATIONS: Maximum number of training iterations.

## Inference
   ```
   python legged_gym/scripts/play.py --task=go2
   ```

   * By default, the loaded policy is the last model of the last run of the experiment folder.
   * Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

   ![Isaac Gym Setup](../robotics-project-24/figures/instruction_2.png)
   https://github.com/user-attachments/assets/98395d82-d3f6-4548-b6ee-8edfce70ac3e
