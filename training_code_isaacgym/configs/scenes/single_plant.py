from pathlib import Path

from . import BaseSceneCfg, StaticObject


asset_path = Path(__file__).parents[2] / "assets/scenes/single_plant"
wall_path = asset_path / "urdf" / "ceiled_walls_5x5x3.urdf"
plant_path = asset_path / "urdf" / "plant1.urdf"


class SinglePlantCfg(BaseSceneCfg):
    name = "empty_room_5x5"
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
                max_random_loc_offset=(2.3, 2.3, 0),
                size=(0.125, 0.125, 0.3),
            ),
    ]
    size = 5.0
