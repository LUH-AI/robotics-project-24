from typing import List, Literal, Tuple, Optional
import typing
from pathlib import Path

import torch
from isaacgym import gymapi


ObjectType = Literal["robot", "ground", "wall", "flower_pot", "ball", "obstacle"]
Location = Tuple[float, float, float]


class StaticObject:
    def __init__(
        self,
        name: str,
        type: ObjectType,
        asset_path: Path,
        init_location: Location = (0.0, 0.0, 0.0),
        max_random_loc_offset: Location = (0.0, 0.0, 0.0),
        size: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Create a static object

        Args:
            name (str): Name of object
            type (ObjectType): Type of static object (More options can be appended to type if needed)
            asset_path (Path): Path to urdf file
            init_location (Optional[Location]): new x, y, z coordinates for placement of object in environment
            max_random_loc_offset (Optional[Location]): maximal random absolute offset from x, y, z coordinates of init_location for environment/object initialization/reset
        """
        self.name: str = name
        self.type: ObjectType = type
        self.init_location = torch.tensor(init_location)
        self.max_random_loc_offset = torch.tensor(max_random_loc_offset)

        self.size = torch.tensor(size)

        self.asset_path = asset_path
        self.asset_root = asset_path.parents[1]
        self.asset_file = f"{asset_path.parent.name}/{asset_path.name}"

        self.asset_options = gymapi.AssetOptions()
        self.asset_options.collapse_fixed_joints = True

        types = typing.get_args(ObjectType)
        self.segmentation_id: int = types.index(type)

    def to(self, device: str):
        """Moves all tensors in object to the provided device

        Args:
            device (str): Device of torch tensors
        """
        self.init_location = self.init_location.to(device)
        self.max_random_loc_offset = self.max_random_loc_offset.to(device)
        self.size = self.size.to(device)


class BaseSceneCfg:
    name: str = "ground_plane"
    static_objects: List[StaticObject] = []
    # initial_robot_position: Location = ...
    size: float = 3.0
    spacing: float = 1.0
