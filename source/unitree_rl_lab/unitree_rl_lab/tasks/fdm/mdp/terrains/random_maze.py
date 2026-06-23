# Copyright (c) 2025, The Nav-Suite Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import make_border, make_plane

if TYPE_CHECKING:
    from . import random_maze_cfg


def random_maze_terrain(
    difficulty: float, cfg: random_maze_cfg.RandomMazeTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a simple randomized wall maze with occasional stair blocks."""
    meshes: list[trimesh.Trimesh] = [make_plane(cfg.size, height=0.0, center_zero=False)]
    terrain_center = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    if cfg.border_width > 0.0:
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        border_center = [terrain_center[0], terrain_center[1], cfg.maze_height / 2.0]
        meshes += make_border(cfg.size, border_inner_size, cfg.maze_height, border_center)

    wall_thickness = max(0.08, 0.12 * cfg.resolution)
    x_cells = max(2, int((cfg.size[0] - 2 * cfg.border_width) / cfg.resolution))
    y_cells = max(2, int((cfg.size[1] - 2 * cfg.border_width) / cfg.resolution))
    x0 = cfg.border_width + cfg.resolution / 2.0
    y0 = cfg.border_width + cfg.resolution / 2.0
    wall_probability = 0.2 + 0.35 * difficulty

    for ix in range(x_cells):
        for iy in range(y_cells):
            if 0.35 * x_cells < ix < 0.65 * x_cells and 0.35 * y_cells < iy < 0.65 * y_cells:
                continue
            if np.random.random() > wall_probability:
                continue
            horizontal = np.random.random() > 0.5
            if horizontal:
                dims = (cfg.resolution, wall_thickness, cfg.maze_height)
            else:
                dims = (wall_thickness, cfg.resolution, cfg.maze_height)
            pos = (x0 + ix * cfg.resolution, y0 + iy * cfg.resolution, cfg.maze_height / 2.0)
            meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))

    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    step_width = cfg.step_width_range[0] + difficulty * (cfg.step_width_range[1] - cfg.step_width_range[0])
    for _ in range(cfg.num_stairs):
        center = np.array([
            np.random.uniform(cfg.border_width + cfg.resolution, cfg.size[0] - cfg.border_width - cfg.resolution),
            np.random.uniform(cfg.border_width + cfg.resolution, cfg.size[1] - cfg.border_width - cfg.resolution),
        ])
        yaw = np.random.choice([0.0, np.pi / 2.0])
        for k in range(3):
            dims = (step_width * (3 - k), cfg.resolution, step_height * (k + 1))
            pos = np.array([center[0] + k * step_width, center[1], dims[2] / 2.0])
            transform = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1], [center[0], center[1], 0])
            transform[:3, 3] = pos
            meshes.append(trimesh.creation.box(dims, transform))

    return meshes, np.array([terrain_center[0], terrain_center[1], 0.0])
