# Guide for Running IsaacLab on the LUIS-Cluster

### Cluster Access
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

### IsaacLab Installation
* Follow (mostly) the IsaacSim and IsaacLab installation instructions: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
  * You have to load the cmake module manually instead of installing it by running: `module load CMake`
  * If no conda is available, you have to load the corresponding module: `module load Miniforge3`
  * Note: Add the conda loading to your ~/.bashrc file for faster access
  * Note: The verification steps will not work because no gui is accessable

* IsaacLab Assets Download
  * Note: On the branch `feat/robot_model_simulation` in simulation/ is an example structure with assets, configs and a small demo script (only material assets for groundplane are missing)
  * The compute nodes do not have direct internet access. Therefore, assets like the unitreeGo2 config have to be downloaded in advance
  * Download the assets in advance with wget
  * Create a directory where you want to store your assets with mkdir and go to it with cd (e.g. `simulation/assets/{scenes | robots}`)
    * UnitreeGo2: wget http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd
    * Cartpole:
      * wget http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/IsaacLab/Robots/Classic/Cartpole/cartpole.usd
      * In "./Props" folder: wget http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/IsaacLab/Robots/Classic/Cartpole/Props/instanceable_meshes.usd
      * Additionally, base materials need to be downloaded but might only be downloadable from Launcher app (https://docs.omniverse.nvidia.com/launcher/latest/it-managed-launcher/content_install.html)
    * Note: vMaterials_2 can be downloaded here: https://developer.nvidia.com/vmaterials (if needed)
  * The code/configs may have to be updated for the new file locations

### WebRTC Client (Still not working with Cluster)
1. Start an interactive session with the following command. The following command uses a compute node with a single GPU for an hour (for me an NVIDIA RTX A6000 was provided which worked for the basic example. If no valid GPU type is selected, you can start a script with srun and specify the gpu type with --gres=gpu:gpu_type:number_of_gpus and the amount of RAM with --mem 10GB)
```bash
salloc --time=1:00:00 --partition tnt -G 1 --mem 32G
```
   * only tnt partition is supported for livestreaming: [Livestreaming is not supported when Isaac Sim is run on the A100 GPU](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/manual_livestream_clients.html)
   * you can verify the gpu type with the command `nvidia-smi`
1. Type `hostname` to get the name of the used compute node
2. Start your isaac applications with the LIVESTREAM environment variable (e.g. the dummy example from the verification step of the IsaacLab installation)
   * For Omniverse-Streaming-Client: 
```bash
LIVESTREAM=1 python source/standalone/tutorials/00_sim/create_empty.py
```
   * For WebRTC-Client: 
```bash
LIVESTREAM=2 python source/standalone/tutorials/00_sim/create_empty.py
```
   * **Note**: A connection can be established when the following information is logged: "[INFO]: Setup complete..." (This may take a while ~10-30 minutes, setting --mem 32G option seems to accelerate it)
   * Note: The warning "GLFW initialization failed." can be ignored
   * Note: The compute nodes do not have direct internet access. So, all needed data... has to be pre-downloaded
1. Establish a ssh connection via port-forwarding from your local machine (-f flag for execution in background)
   * IsaacSim uses port 8211. For other omniverse applications the ports can be found here: https://docs.omniverse.nvidia.com/extensions/latest/ext_livestream/webrtc.html
```bash
ssh -L 8211:compute_node_hostname:8211 username@login.cluster.uni-hannover.de
```
   * Additionally, the port 49100 has to be forwarded for authentication:
```bash
ssh -L 49100:compute_node_hostname:49100 username@login.cluster.uni-hannover.de
```
1. Open your browser (e.g. Chrome) and open the following URL: `http://localhost:8211/streaming/webrtc-client?server=localhost`
   * A black window with a play button should be visible now.

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
