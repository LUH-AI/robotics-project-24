from typing import List, Tuple, Dict
import torch
from rsl_rl.modules import ActorCritic
from ..configs.robots.go2_high_level_policy_plant import GO2HighLevelPlantPolicyCfg




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
    if (torch.abs((location - robot_location)[:2]) < 2.5).any():  # TODO size of robot
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


def get_distance_and_angle(robot_location: torch.Tensor, robot_orientation: torch.Tensor, object_location: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates distance and angle of an object to the robot

    Args:
        robot_location (torch.Tensor): Absolute location of the robot
        robot_orientation (torch.Tensor): Orientation of the robot
        object_location (torch.Tensor): Absolute location of the object

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Distance, angle to robot
    """
    # Compute distance from robot to plant
    distance = torch.norm(object_location - robot_location)

    # Compute angle from robot to plant
    relative_position = object_location - robot_location
    angle = torch.atan2(relative_position[1], relative_position[0]) - robot_orientation
    angle = torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi  # Normalize angle to [-pi, pi]
    return distance, angle


def get_object_observation(location: torch.Tensor, distance: torch.Tensor, angle: torch.Tensor, probability: torch.Tensor, fov_angle: torch.Tensor) -> Dict[str, torch.Tensor]:
    # Check if the plant is within the robot's field of view (FOV)
    if torch.abs(angle) <= fov_angle:
        return {
            "location": location,
            "probability": probability,
            "distance": distance,
            "angle": angle,
        }
    else:
        return {
            "location": location,
            "probability": torch.tensor(0).to(location.device),
            "distance": distance,
            "angle": angle,
        }


def get_dummy_object_observation(device: str) -> Dict[str, torch.Tensor]:
    """Gets a dummy observation with 0 probability for correct observation shapes

    Args:
        device (str): Device for tensors

    Returns:
        Dict[str, torch.Tensor]: Dummy observation for error prevention
    """
    loc = torch.tensor([0,0]).to(device)
    prob = torch.tensor(0).to(device)
    dist = torch.tensor(0).to(device)
    angle = torch.tensor(0).to(device)
    return get_object_observation(loc, dist, angle, prob, 0)


def convert_object_property(objects: List[List[Dict[str, torch.Tensor]]], property: str, device: str) -> torch.Tensor:
    """Extracts object property from list of objects and creates a tensor from them

    Args:
        objects (List[List[Dict[str, torch.Tensor]]]): List of objects (for all environments)
        property (str): Property of object (location, distance, angle or probability)
        device (str): Device for tensors

    Returns:
        torch.Tensor: Converted object properties (for all environments)
    """
    return torch.tensor([object[property] for _objects in objects for object in _objects],
                                       device=device).view((len(objects), len(objects[0]))).squeeze(1)


def get_reset_indices(env_ids: torch.Tensor, num_objects: int) -> torch.Tensor:
    """Creates a tensor with the indices for all objects within the specified environments.

    Args:
        env_ids (torch.Tensor): Environment IDs (0, 1, 2, 3, ...)
        num_objects (int): Number of objects per environment.

    Returns:
        torch.Tensor: Tensor with all indices for specified environments (torch.int32)
    """
    stubs = torch.arange(0, num_objects, device=env_ids.device).repeat(
        (len(env_ids), 1)
    )
    return (stubs + env_ids.unsqueeze(1) * num_objects).view(-1).to(dtype=torch.int32)


def load_low_level_policy(cfg: GO2HighLevelPlantPolicyCfg, sim_device):
    module = ActorCritic(
        num_actor_obs=cfg.low_level_policy.num_observations,
        num_critic_obs=cfg.low_level_policy.num_observations,
        num_actions=cfg.low_level_policy.num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
    )
    module = module.to(sim_device)
    checkpoint = torch.load(cfg.low_level_policy.path)
    print("low level policy", module)

    model_state_dict = checkpoint.get('model_state_dict')
    if model_state_dict is None:
        raise ValueError("The checkpoint does not contain a 'model_state_dict' key.")

    try:
        module.load_state_dict(model_state_dict)
    except RuntimeError as e:
        print("\nError while loading state dictionary:")
        print(e)
        return None

    return module