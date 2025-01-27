from pathlib import Path

from .go2_default import GO2DefaultCfg


class GO2LowLevelPolicyCfg(GO2DefaultCfg):
    name = "go2_low-level-policy"

    class asset(GO2DefaultCfg.asset):
        file = str(Path(__file__).parents[2] / "assets/robots" / "go2_with_watering" / "urdf/go2.urdf")
