# Training Heinrichs
For training, we recommend IsaacLab since it has a GO2 model built in can can easily be used to create new scenes, vary reward signals and add things like curricula or domain randomization. 
Since the documentation is quite good, there is not a lot of additional starter code. We simply extracted a few parts of the code base we thought were especially relevant at the beginning:

- The configurations of the standard locomotion tasks for GO2 as reference points on how to change them and add new ones
- An example of a new configuration and env registration of an environment using it
- The training script for one of the RL libraries included with IsaacLab, SKRL

You are not required to stick to this library or these configs at all, but you should understand how they work together before continuing. There is also a notebook with plotting examples you can use if you want to.