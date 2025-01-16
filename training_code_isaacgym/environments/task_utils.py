from typing import List
import torch


def calculate_random_location(
    location_offset: torch.Tensor,
    init_location: torch.Tensor,
    max_random_loc_offset: torch.Tensor,
) -> torch.Tensor:
    """Calculates a randomized location based on the specified init_location and the random offset.

    Args:
        location_offset (torch.Tensor): Offset for placement of the scene on groundplane
        init_location (torch.Tensor): Mean location with shape: (x,y,z)
        max_random_loc_offset (torch.Tensor): Maximum distance of calculated location to init_location per dimension with shape: (x,y,z)

    Returns:
        torch.Tensor: Randomized location
    """
    random_loc_offset = (
        max_random_loc_offset
        * (torch.rand(max_random_loc_offset.shape, device=init_location.device) - 0.5)
        * 2
    )
    return init_location + location_offset + random_loc_offset


def validate_location(
    object,
    location: torch.Tensor,
    robot_location: torch.Tensor,
    other_object_locations: List[torch.Tensor],
    other_object_sizes: List[torch.Tensor],
) -> bool:
    """Classifies whether location is allowed meaning that object does not collide with other objects or the robot.
    **Be aware:**
    It does not detect collisions with walls!
    The z axis is also not checked

    Args:
        object (StaticObject): Object that should be placed into scene
        location (torch.Tensor): Actual location of the object
        robot_location (torch.Tensor): Location of the robot in scene at initialisation
        other_object_locations (List[torch.Tensor]): Locations of objects that are already inserted in scene
        other_object_sizes (List[torch.Tensor]): Sizes of objects that are already inserted in scene

    Returns:
        bool: True if no collision found else False
    """
    if (torch.abs((location - robot_location)[:2]) < 0.5).any():  # TODO size of robot
        return False
    for other_location, other_size in zip(other_object_locations, other_object_sizes):
        if (
            other_location != None
            and (
                torch.abs((location - other_location)[:2])
                < ((other_size + object.size) / 2)[:2]
            ).any()
        ):
            return False
    return True

def get_reset_indices(env_ids: torch.Tensor, num_objects: int) -> torch.Tensor:
    """Creates a tensor with the indices for all objects within the specified environments.

    Args:
        env_ids (torch.Tensor): Environment IDs (0, 1, 2, 3, ...)
        num_objects (int): Number of objects per environment.

    Returns:
        torch.Tensor: Tensor with all indices for specified environments (torch.int32)
    """
    stubs = torch.arange(0, num_objects, device=env_ids.device).repeat((len(env_ids), 1))
    return (stubs + env_ids.unsqueeze(1) * num_objects).view(-1).to(dtype=torch.int32)