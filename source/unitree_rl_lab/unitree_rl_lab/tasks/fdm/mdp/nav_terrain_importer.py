from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.log
from isaaclab.terrains import TerrainImporter as TerrainImporterBase
from isaaclab.terrains import TerrainImporterCfg as TerrainImporterCfgBase
from isaaclab.utils import configclass
from pxr import UsdGeom

if TYPE_CHECKING:
    from isaaclab.terrains import TerrainGeneratorCfg


class FDMNavTerrainImporter(TerrainImporterBase):
    """Local IsaacLab-2.3-compatible subset of nav-suite's NavTerrainImporter."""

    cfg: "FDMNavTerrainImporterCfg"

    def __init__(self, cfg: "FDMNavTerrainImporterCfg") -> None:
        super().__init__(cfg)

        if self.cfg.add_colliders:
            self._apply_physics_properties()

        if self.cfg.groundplane:
            ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=self.cfg.physics_material)
            ground_plane = ground_plane_cfg.func("/World/GroundPlane", ground_plane_cfg)
            ground_plane.visible = False

    def configure_env_origins(self, origins: np.ndarray | torch.Tensor | None = None):
        if origins is not None:
            if isinstance(origins, np.ndarray):
                origins = torch.from_numpy(origins)
            self.terrain_origins = origins.to(self.device, dtype=torch.float)
            self.env_origins = self._compute_env_origins_curriculum(self.cfg.num_envs, self.terrain_origins)
        elif self.cfg.usd_uniform_env_spacing is not None and self.cfg.terrain_type == "usd":
            prim = prim_utils.get_prim_at_path(self.cfg.prim_path + "/terrain")
            bbox = UsdGeom.Boundable(prim).ComputeWorldBound(0, UsdGeom.Tokens.default_).ComputeAlignedBox()
            spacing = self.cfg.usd_uniform_env_spacing
            grid_x, grid_y = torch.meshgrid(
                torch.arange(bbox.GetMin()[0] + spacing / 2, bbox.GetMax()[0] - spacing / 2, spacing),
                torch.arange(bbox.GetMin()[1] + spacing / 2, bbox.GetMax()[1] - spacing / 2, spacing),
                indexing="ij",
            )
            env_origins = torch.stack((grid_x.flatten(), grid_y.flatten(), torch.zeros_like(grid_x.flatten())), dim=1)
            repetitions = max(1, (self.cfg.num_envs // env_origins.shape[0]) + 1)
            self.terrain_origins = None
            self.env_origins = env_origins.repeat(repetitions, 1)[: self.cfg.num_envs].to(self.device)
        else:
            super().configure_env_origins(origins)

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        if not self.cfg.regular_spawning:
            return super()._compute_env_origins_curriculum(num_envs, origins)

        num_rows = origins.shape[0]
        if self.cfg.max_init_terrain_level is None:
            max_init_level = num_rows - 1
        else:
            max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)
        self.max_terrain_level = num_rows
        repeated_origins = origins[: max_init_level + 1, :].reshape(-1, 3)
        return repeated_origins.repeat(math.ceil(num_envs / repeated_origins.shape[0]), 1)[:num_envs]

    def _apply_physics_properties(self):
        mesh_prim = sim_utils.get_first_matching_child_prim(
            self.cfg.prim_path,
            lambda prim: prim.GetTypeName() == "Mesh",
        )
        if mesh_prim is None:
            omni.log.warn(f"Could not find any terrain meshes under {self.cfg.prim_path}.")
            return

        collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        sim_utils.define_collision_properties(self.cfg.prim_path, collider_cfg)

        self.cfg.physics_material.func(f"{self.cfg.prim_path}/physicsMaterial", self.cfg.physics_material)
        sim_utils.bind_physics_material(self.cfg.prim_path, f"{self.cfg.prim_path}/physicsMaterial")

        stage_utils.update_stage()
        omni.log.info("FDM USD terrain physics setup complete.")


@configclass
class FDMNavTerrainImporterCfg(TerrainImporterCfgBase):
    class_type: type = FDMNavTerrainImporter

    terrain_generator: "TerrainGeneratorCfg | None" = None

    usd_uniform_env_spacing: float | None = None
    """Grid-like environment spacing over a single USD terrain bbox."""

    regular_spawning: bool = False
    """Spawn regularly over all terrain origins when curriculum origins are provided."""

    groundplane: bool = True
    """Import a hidden ground plane, matching nav-suite's default."""

    add_colliders: bool = False
    """Apply collision properties and bind the terrain physics material after USD import."""
