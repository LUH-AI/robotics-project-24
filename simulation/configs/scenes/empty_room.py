from pathlib import Path

from . import BaseSceneCfg, StaticObject


asset_path = Path("assets/scenes/empty_room")
urdf_path = asset_path / "urdf" / "walls.urdf"

class EmptyRoomCfg(BaseSceneCfg):
    name = "empty_room"
    static_objects = [
        StaticObject(
            "walls",
            "wall",
            urdf_path,
        )
    ]
    size: float = 11.
