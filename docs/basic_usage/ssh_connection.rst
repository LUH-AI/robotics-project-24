Connecting With Heinrich
=========================

To do anything but the basic app commands, you'll need a network connection with Heinrich. This can be done via ethernet in combination with SSH.
A 15m ethernet cable is in the RL office, if you need more movement please talk to the RL team.

Plug the ethernet cable into the robot and your computer. Ideally at this point you'll be disconnected from all other ethernet connections.
You should see the robot. Now assign this connection the manual IPv4 adress **192.168.123.51** and network mask **255.255.255.0**.
At this point you should be able to run **make ping** and get a result back. Now run **make ssh** to connect to the robot's OS.
This is the primary way to deploy policies via ROS.

Alternatively, if you have access to an AI institute ethernet bus, take the Heinrich cable with the USB-C adapter and plug him in directly. 
Any computer in the AI lab net will be able to connect to him via **ping heinrich** and **make ai-ssh**. 
This currently doesn't support deployment, but makes updates and package installations easier.