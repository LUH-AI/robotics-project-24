from __future__ import annotations

import torch


from typing import TYPE_CHECKING
from dataclasses import MISSING


from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from omni.isaac.lab.assets.asset_base_cfg import AssetBaseCfg
from omni.isaac.lab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from omni.isaac.lab.scene.interactive_scene_cfg import InteractiveSceneCfg
from omni.isaac.lab.terrains.terrain_importer_cfg import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

class Go2MoveAction(ActionTerm):
    
    cfg: Go2MoveActionCfg
    
    def __init__(self, cfg: Go2MoveActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        #load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()
        
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        
        # prepare low level actions
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)
        
        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: self.low_level_actions
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions
        cfg.low_level_observations.velocity_commands.params = dict()
        
        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0
        
    @property
    def action_dim(self) -> int:
        return 3
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions
    
    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1
        
@configclass
class Go2MoveActionCfg(ActionTermCfg):
    
    class_type: type[ActionTerm] = Go2MoveAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""