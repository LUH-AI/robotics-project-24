import random
from pathlib import Path
from . import BaseSceneCfg, StaticObject

asset_path = Path(__file__).parents[2] / "assets/scenes/plant_environment"
urdf_path = asset_path / "urdf" / "walls_10x10.urdf"
plant_path = asset_path / "urdf" / "plant1.urdf"
chair_path = asset_path / "urdf" / "chair1.urdf"

class PlantEnvironmentCfg(BaseSceneCfg):
    name = "plant_environment"
    size: float = 10.0

    static_objects = [
        StaticObject(
            "walls",
            "wall",
            urdf_path,
        ),
        StaticObject(
            "flower",
            "flower_pot",
            plant_path,
            init_location=(0., 0., 0.2),
            max_random_loc_offset=(4.8, 4.8, 0),
            size=(0.125, 0.125, 0.3),
        ),
        StaticObject(
            "flower",
            "flower_pot",
            plant_path,
            init_location=(0., 0., 0.2),
            max_random_loc_offset=(4.8, 4.8, 0),
            size=(0.125, 0.125, 0.3),
        ),
        StaticObject(
            "flower",
            "flower_pot",
            plant_path,
            init_location=(0., 0., 0.2),
            max_random_loc_offset=(4.8, 4.8, 0),
            size=(0.125, 0.125, 0.3),
        ),
        StaticObject(
            "chair",
            "obstacle",
            chair_path,
            init_location=(0., 0., 0.2),
            max_random_loc_offset=(4.8, 4.8, 0),
            size=(0.5, 0.5, 0.5),
        ),
        StaticObject(
            "chair",
            "obstacle",
            chair_path,
            init_location=(0., 0., 0.2),
            max_random_loc_offset=(4.8, 4.8, 0),
            size=(0.5, 0.5, 0.5),
        ),
        StaticObject(
            "chair",
            "obstacle",
            chair_path,
            init_location=(0., 0., 0.2),
            max_random_loc_offset=(4.8, 4.8, 0),
            size=(0.5, 0.5, 0.5),
        )
    ]
