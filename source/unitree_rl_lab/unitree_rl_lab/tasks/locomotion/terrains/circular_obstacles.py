from __future__ import annotations

from dataclasses import MISSING

import numpy as np

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def circular_obstacles_terrain(difficulty: float, cfg: "HfCircularObstaclesTerrainCfg") -> np.ndarray:
    """Generate a height-field terrain with discrete circular pillars."""

    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    obs_height = int(obs_height / cfg.vertical_scale)

    radius_min = max(1, int(cfg.obstacle_radius_range[0] / cfg.horizontal_scale))
    radius_max = max(radius_min, int(cfg.obstacle_radius_range[1] / cfg.horizontal_scale))
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))

    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2

    for _ in range(cfg.num_obstacles):
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        else:
            raise ValueError(
                f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'."
            )

        radius = np.random.randint(radius_min, radius_max + 1)

        center_x = None
        center_y = None
        for _ in range(cfg.max_sampling_attempts):
            candidate_x = np.random.randint(radius, max(radius + 1, width_pixels - radius))
            candidate_y = np.random.randint(radius, max(radius + 1, length_pixels - radius))

            overlaps_platform = (
                candidate_x + radius >= x1
                and candidate_x - radius <= x2
                and candidate_y + radius >= y1
                and candidate_y - radius <= y2
            )
            if not overlaps_platform:
                center_x = candidate_x
                center_y = candidate_y
                break

        if center_x is None or center_y is None:
            continue

        x_min = max(0, center_x - radius)
        x_max = min(width_pixels - 1, center_x + radius)
        y_min = max(0, center_y - radius)
        y_max = min(length_pixels - 1, center_y + radius)

        xx = np.arange(x_min, x_max + 1)[:, None]
        yy = np.arange(y_min, y_max + 1)[None, :]
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        local_patch = hf_raw[x_min : x_max + 1, y_min : y_max + 1]
        local_patch[mask] = height

    hf_raw[x1:x2, y1:y2] = 0
    return np.rint(hf_raw).astype(np.int16)


@configclass
class HfCircularObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a height-field terrain with discrete circular pillars."""

    function = circular_obstacles_terrain

    obstacle_height_mode: str = "choice"
    """The mode to use for the obstacle height. Supports: "choice", "fixed"."""

    obstacle_radius_range: tuple[float, float] = MISSING
    """The minimum and maximum obstacle radius in meters."""

    obstacle_height_range: tuple[float, float] = MISSING
    """The minimum and maximum obstacle height in meters."""

    num_obstacles: int = MISSING
    """The number of circular obstacles to generate."""

    platform_width: float = 1.0
    """Width of the obstacle-free square platform at the terrain center."""

    max_sampling_attempts: int = 50
    """Maximum attempts for placing each obstacle outside the spawn platform."""
