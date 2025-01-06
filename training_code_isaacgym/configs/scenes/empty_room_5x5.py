from pathlib import Path

from . import BaseSceneCfg, StaticObject


asset_path = Path(__file__).parents[2] / "assets/scenes/empty_room_5x5"
urdf_path = asset_path / "urdf" / "walls_5x5.urdf"


class EmptyRoom5x5Cfg(BaseSceneCfg):
    name = "empty_room_5x5"
    static_objects = [
        StaticObject(
            "walls",
            "wall",
            urdf_path,
        )
    ]
    size: float = 5.0
