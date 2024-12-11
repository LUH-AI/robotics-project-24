from typing import List, Literal, Tuple, Optional
import typing
from pathlib import Path

import torch
from isaacgym import gymapi


ObjectType = Literal["robot", "ground", "wall", "flower_pot", "ball", "obstacle"]
Location = Tuple[float, float, float]


class StaticObject():
    def __init__(self, name: str, type: ObjectType, asset_path: Path, init_location: Optional[Location] = None, max_random_loc_offset: Optional[Location] = None) -> None:
        """Create a static object

        Args:
            name (str): Name of object
            type (ObjectType): Type of static object (More options can be appended to type if needed)
            asset_path (Path): Path to urdf file
            init_location (Optional[Location]): new x, y, z coordinates for placement of object in environment
            max_random_loc_offset (Optional[Location]): maximal random absolute offset from x, y, z coordinates of init_location for environment/object initialization/reset
        """
        self.name: str = name
        self.type = type
        self.init_location = torch.tensor(init_location) if init_location != None else torch.zeros(3)
        self.max_random_loc_offset = torch.tensor(max_random_loc_offset) if max_random_loc_offset != None else torch.zeros(3)

        self.asset_path = asset_path
        self.asset_root = asset_path.parents[1]
        self.asset_file = f"{asset_path.parent.name}/{asset_path.name}"

        self.asset_options = gymapi.AssetOptions()
        self.asset_options.collapse_fixed_joints = True

        types = typing.get_args(ObjectType)
        self.segmentation_id: int = types.index(type)


class BaseSceneCfg():
    name: str = "ground_plane"
    static_objects: List[StaticObject] = []
    #initial_robot_position: Location = ...
    size: float = 3.
