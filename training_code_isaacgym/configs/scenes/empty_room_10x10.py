from pathlib import Path

from . import BaseSceneCfg, StaticObject


asset_path = Path(__file__).parents[2] / "assets/scenes/empty_room_10x10"
urdf_path = asset_path / "urdf" / "walls_10x10.urdf"


class EmptyRoom10x10Cfg(BaseSceneCfg):
    name = "empty_room_10x10"
    static_objects = [
        StaticObject(
            "walls",
            "wall",
            urdf_path,
        )
    ]
    size: float = 10.0
