import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
)

ChannelFactoryInitialize(0)

sport_client = SportClient()
sport_client.SetTimeout(10.0)
sport_client.Init()


starttime=time.time()

while time.time()-starttime < 1:
    sport_client.Move(0,0,1.0)
    print("MOVE")