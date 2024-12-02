# Robotics Project 2024

This is the code repository for the 2024 RL robotics project at LUHAI. It currently contains starter code and basic documentation, but will be the home of the full project during the semester.

## Project Management

Project management should be done via the project attached to this repository. Please add all milestones and ToDos, use the team features and set completion targets for all items. This makes progress visible to us and this is how we'll judge your progress.

## Robot Documentation

Information on how to operate Heinrich can be found in the documentation page. These are the basic fact on Heinrich's usage with more advanced concepts like RL deployment not being explicitly stated since there are often multiple options on working with low level control, i.e. ROS. 
Still, there are some additional manuals linked for a more in-depth look and the code examples should give you insights on how to accomplish your specific goals.

## Installing Dependencies

You can install the dependencies for training, deployment or both via "make install-train", "make install-deploy" and "make install". Be aware that the last two require root access and either a x86_64 or a aarch64 architecture. IsaacLab is currently only supported on Linux and Windows.

## Training

We recommend IsaacLab for training policies in simulation. This is a powerful framework for robotic tasks and contains a model of Heinrich (Unitree GO2). You can use it to create interesting new tasks and use the built-in RL training frameworks to easily find policies.
IsaacLab has an [extensive documentation](https://isaac-sim.github.io/IsaacLab/index.html) in addition to several utility extensions [on GitHub](https://github.com/isaac-sim).

It is recommended you use the AI partition of the LUIS cluster for training. Please make prepare a list of everyone who needs cluster access during the project preparation phase.

## Deployment
There are multiple deployment options for ROS, but the simplest one might be using the [recommended IsaacLab workflow](https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/).

