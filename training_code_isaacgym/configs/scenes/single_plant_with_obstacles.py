from pathlib import Path

from . import BaseSceneCfg, StaticObject


asset_path = Path(__file__).parents[2] / "assets/scenes/plant_environment_with_obstacles"
wall_path = asset_path / "urdf" / "ceiled_walls_5x5x3.urdf"
plant_path = asset_path / "urdf" / "plant1.urdf"
chair_path = asset_path / "urdf" / "chair1.urdf"

class SinglePlantWithObstaclesCfg(BaseSceneCfg):
    name = "single_plant"
    static_objects = [
        StaticObject(
            "walls",
            "wall",
            wall_path,
        ),
        StaticObject(
            "flower",
            "flower_pot",
            plant_path,
            init_location=(0.0, 0.0, 0.2),
            max_random_loc_offset=(2., 2., 0),
            size=(0.125, 0.125, 0.3),
        ),
        StaticObject(
            "chair",
            "obstacle",
            chair_path,
            init_location=(0.0, 0.0, 0.2),
            max_random_loc_offset=(1.6, 1.6, 0),
            size=(0.5, 0.5, 0.5),
        ),
        StaticObject(
            "chair",
            "obstacle",
            chair_path,
            init_location=(0.0, 0.0, 0.2),
            max_random_loc_offset=(1.6, 1.6, 0),
            size=(0.5, 0.5, 0.5),
        ),
        StaticObject(
            "chair",
            "obstacle",
            chair_path,
            init_location=(0.0, 0.0, 0.2),
            max_random_loc_offset=(1.6, 1.6, 0),
            size=(0.5, 0.5, 0.5),
        ),
    ]
    size = 5.0
