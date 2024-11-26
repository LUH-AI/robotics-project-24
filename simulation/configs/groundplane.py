from collections.abc import Callable
from pathlib import Path

from omni.isaac.lab.sim.spawners import materials
from omni.isaac.lab.sim.spawners.spawner_cfg import SpawnerCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.sim.spawners.from_files import from_files

ASSET_DIRPATH = Path(__file__).parent.parent / "assets" / "scenes"

@configclass
class GroundPlaneCfg(SpawnerCfg):
    """Create a ground plane prim.

    This uses the USD for the standard grid-world ground plane from Isaac Sim by default.
    """

    func: Callable = from_files.spawn_ground_plane

    usd_path: str = str(ASSET_DIRPATH / "default_environment.usd")
    """Path to the USD file to spawn asset from. Defaults to the grid-world ground plane."""

    color: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    """The color of the ground plane. Defaults to (0.0, 0.0, 0.0).

    If None, then the color remains unchanged.
    """

    size: tuple[float, float] = (100.0, 100.0)
    """The size of the ground plane. Defaults to 100 m x 100 m."""

    physics_material: materials.RigidBodyMaterialCfg = materials.RigidBodyMaterialCfg()
    """Physics material properties. Defaults to the default rigid body material."""
