import random
from pathlib import Path
from . import BaseSceneCfg, StaticObject

asset_path = Path(__file__).parents[2] / "assets/scenes/plant_environment"
urdf_path = asset_path / "urdf" / "walls.urdf"
plant_path = asset_path / "urdf" / "plant1.urdf"
chair_path = asset_path / "urdf" / "chair1.urdf"

class PlantEnvironmentCfg(BaseSceneCfg):
    name = "empty_room"
    size: float = 11.0
    max_range = 3.5
    min_distance = 0.9

    def __init__(self, seed=None):
        """Initialize random positions for static objects."""
        super().__init__()
        if seed is not None:
            random.seed(seed)  # Set a unique seed for this instance

        self.positions = self._generate_random_positions(6)

        self.static_objects = [
            StaticObject(
                "walls",
                "wall",
                urdf_path,
            ),
            StaticObject(
                "flower",
                "flower_pot",
                plant_path,
                (self.positions[0][0], self.positions[0][1], 0.2),
            ),
            StaticObject(
                "flower",
                "flower_pot",
                plant_path,
                (self.positions[1][0], self.positions[1][1], 0.2),
            ),
            StaticObject(
                "flower",
                "flower_pot",
                plant_path,
                (self.positions[2][0], self.positions[2][1], 0.2),
            ),
            StaticObject(
                "chair",
                "obstacle",
                chair_path,
                (self.positions[3][0], self.positions[3][1], 0.2),
            ),
            StaticObject(
                "chair",
                "obstacle",
                chair_path,
                (self.positions[4][0], self.positions[4][1], 0.2),
            ),
            StaticObject(
                "chair",
                "obstacle",
                chair_path,
                (self.positions[5][0], self.positions[5][1], 0.2),
            )
        ]

    def _generate_random_positions(self, num_positions):
        """Generate random positions for objects."""
        positions = []
        while len(positions) < num_positions:
            x = (random.random() - 0.5) * 2.0 * self.max_range
            y = (random.random() - 0.5) * 2.0 * self.max_range
            # Ensure minimum distance between objects
            if all(((x - pos[0])**2 + (y - pos[1])**2)**0.5 >= self.min_distance for pos in positions) and x**2+y**2>=self.min_distance:
                positions.append((x, y))
        return positions
