# Deploying Policies on Heinrich
IsaacLab has a [suggested deployment process](https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/) which should largely be followed on Heinrich. Here's a short version with a few small changes. For details, please read the guide carefully:

1. Train a policy (recommended to be done using e.g. domain randomization)
2. Convert the trained policy to .onnx files and env.yaml to env_cfg.json
3. Check the JetPack version to make sure everything will run smoothly
4. Connect to Heinrich via ethernet and ssh onto him
5. Install the [Unitree SDK2]() on the Orion
6. Move everything to the correct file structures
7. Now run the deployment code here

Caution! This is part of the project and not actually implemented yet. It is also only a suggestion on how to proceed. The example code for spot is directly connected to IsaacLab and it is possible to set up the sender/receiver connection to Unitree's SDK2, but you can choose freely if you would like to pursue this direction or an alternative like [this example from Teddy Liao](https://github.com/Teddy-Liao/walk-these-ways-go2/tree/main) using LCM or even a quite involved setup like [in this project](https://github.com/ToruOwO/hato).