# Setup for IsaacGym and Unitree-RL-Gym
This is a introductory guide to setup the LUIS-Cluster of the Leibniz University Hanover to use IsaacGym and Unitree-RL-Gym (GO2 robot).  

## Cluster Access
* If you haven't requested cluster access yet, write an email to Aditya or Theresa with your student email.
* Connecting via ssh:
  * Make sure that you are connected to the university network (e.g. via VPN)
  * `ssh username@login.cluster.uni-hannover.de`
  * Insert your username and afterwards enter the password that was provided by Aditya or Theresa.
  * Note: For Windows you might have to install a ssh client.
  * Tutorial for creating a ssh key: https://askubuntu.com/questions/46930/how-can-i-set-up-password-less-ssh-login
  * Note: Setup your ssh config for faster access without entering your password
    * Afterwards you can login with `ssh luis_login`
    * Add the following stub to your ssh config file `~/.ssh/config`
```
Host luis_login
User username
HostName login.cluster.uni-hannover.de
IdentityFile ~/.ssh/your_ssh_key
IdentitiesOnly yes
```

### File management
* Only 10GB can be stored in your home directory ($HOME)
  * Therefore you should save files mostly in your bigwork directroy ($BIGWORK)
* Change conda environment and package saving directory
  * Open/Create the following file: `~/.condarc`
  * Insert the following snippet:
```
envs_dirs:
  - /bigwork/username/.conda/envs
pkgs_dirs:
  - /bigwork/username/.conda/pkgs
```
* Change location of pip cache
  * Open/Create the following file: `~/.config/pip/pip.conf`
  * Insert the following snippet:
```
[global]
cache-dir=/bigwork/username/.cache
```

### Github
* ssh for github access on cluster
  * Create a ssh key on the cluster (https://askubuntu.com/questions/46930/how-can-i-set-up-password-less-ssh-login)
  * Copy the public key from ~/.ssh/your-key.pub
  * Add it to github: settings / ssh and gpg keys / new ssh key
* Now you can clone the project repo and work with it as usual: `git clone git@github.com:LUH-AI/robotics-project-24.git`

### File transfer
* File transfer between local machine and cluster with rsync
  * `rsync source target`
  * `-r` option for transferring folders recursively
  * Use transfer node instead of login node for larger files
  * e.g. `rsync ./testfile username@transfer.cluster.uni-hannover.de:/bigwork/username/rl_project/`
  * If ssh config was setup, you can replace `username@transfer.cluster.uni-hannover.de` with the alias

### Remote VS-Code
1. Open VS-Code on your local machine
2. Install the ssh remote extension Remote - SSH
   1. Go to extensions
   2. Search for Remote-SSH
   3. Click on the first result and click on install
3. Press F1 and run the Remote-SSH: Open SSH Host... command
4. Enter your user and host/IP or your alias in the input box that appears and press enter: username@login.cluster.uni-hannover.de
5. If prompted, enter your password (not needed for key based authentication)
6. After you are connected, use File > Open Folder to open a folder on the host (e.g. /bigwork/username/robotics-project-24)
7. You may have to re-install the python extensions... on your remote VS-code
   * Go to extensions and click on Install in SSH: ...


* **For me running the Isaaclab installation with `./isaaclab.sh --install` lead to the following warning:**
  * Could not find Isaac Sim VSCode settings: ... . This will result in missing 'python.analysis.extraPaths' in the VSCode settings, which limits the functionality of the Python language server.
  * You can manually add the paths to your remote VS-code settings:
    1. Open settings
    2. Search for `python.analysis.extraPaths`
    3. Add the following paths (change paths as needed (e.g. username))
    * /bigwork/username/robotics-project-24/IsaacLab/source/extensions/omni.isaac.lab
    * /bigwork/username/robotics-project-24/IsaacLab/source/extensions/omni.isaac.lab_assets
    * /bigwork/username/robotics-project-24/IsaacLab/source/extensions/omni.isaac.lab_tasks

## Graphical interface

![Isaac Gym Setup](figures/instruction_1.png)
1. Log into the interactive sessions website of the university cluster:
https://login.cluster.uni-hannover.de/pun/sys/dashboard/batch_connect/sessions
2. Navigate to the "Interactive Apps" field and start a new "Cluster Remote Desktop" session.
3. Select
* number of hours (eg. 5h)
* number of cpu cores (e.g. 8)
* memory per CPU core (so that #cores * memory > 32GB)
* a single RTX compatible GPU (rtxa6000:1, rtx3090:1)
* cluster partition (tnt as the only available)

## Installation 
This first part of the installation should be done on the shell, since the Cluster Remote Desktop only allows limited access to the internet for installations.
The cluster could be either accessed with a shell or via a webbrowser: https://login.cluster.uni-hannover.de/pun/sys/shell/ssh/127.0.0.1 or via SSH:
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
   ```
   conda create -n unitree_rl_env python=3.8 -y
   conda activate unitree_rl_env
   ```
3. Navigate into the repo, which optimally should be in `$BIGWORK`.
   ```
   cd /bigwork/<username>/user
   mkdir isaacgym
   cd isaacgym
   ```
4. Install Isaac Gym

   - Download and install Isaac Gym Preview 4 from [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)
   ```
   wget https://developer.nvidia.com/isaac-gym-preview-4 --output-document /tmp/isaac-gym-preview-4.tar.gz
   tar -xvzf /tmp/isaac-gym-preview-4.tar.gz
   cd isaacgym/python && pip install -e .
   cd ../..
   ```
   - For troubleshooting check docs isaacgym/docs/index.html
5. Install rsl_rl v1.0.2 (PPO implementation)

   - Clone [https://github.com/leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)
   ```
   git clone https://github.com/leggedrobotics/rsl_rl
   cd rsl_rl
   git checkout v1.0.2
   pip install -e .
   cd ..
   ```
6. Install unitree_rl_gym
   ```
   git clone https://github.com/unitreerobotics/unitree_rl_gym
   cd unitree_rl_gym
   pip install -e .
   cd ..
   ```
7. Export all library paths. This step is sometimes also required after creating a new session if not updated permanently:
   ```
   export LD_LIBRARY_PATH=/bigwork/<username>/.conda/envs/<venv name>/lib:$LD_LIBRARY_PATH
   ```
