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
    from . import quad_pyramid_stairs_cfg


def quad_pyramid_stairs_terrain(
    difficulty: float, cfg: quad_pyramid_stairs_cfg.MeshQuadPyramidStairsCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate four pyramid-stair approaches around a central platform."""
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    terrain_center = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])
    inner_x = cfg.platform_width
    inner_y = cfg.platform_width
    available_x = max(cfg.size[0] - 2.0 * cfg.border_width - cfg.platform_width, 0.0)
    available_y = max(cfg.size[1] - 2.0 * cfg.border_width - cfg.platform_width, 0.0)
    num_steps = max(1, int(min(available_x, available_y) / (2.0 * cfg.step_width)))

    meshes: list[trimesh.Trimesh] = [make_plane(cfg.size, height=0.0, center_zero=False)]
    if cfg.border_width > 0.0:
        border_center = [terrain_center[0], terrain_center[1], step_height * num_steps / 2.0]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        meshes += make_border(cfg.size, border_inner_size, step_height * num_steps, border_center)

    for k in range(num_steps):
        height = (k + 1) * step_height
        z = height / 2.0
        length = max((num_steps - k) * cfg.step_width, cfg.step_width)
        width_x = inner_x + 2.0 * (num_steps - k) * cfg.step_width
        width_y = inner_y + 2.0 * (num_steps - k) * cfg.step_width
        offset = cfg.platform_width / 2.0 + k * cfg.step_width + length / 2.0

        dims_x = (length, width_y, height)
        for sign in (-1.0, 1.0):
            pos = (terrain_center[0] + sign * offset, terrain_center[1], z)
            meshes.append(trimesh.creation.box(dims_x, trimesh.transformations.translation_matrix(pos)))

        dims_y = (width_x, length, height)
        for sign in (-1.0, 1.0):
            pos = (terrain_center[0], terrain_center[1] + sign * offset, z)
            meshes.append(trimesh.creation.box(dims_y, trimesh.transformations.translation_matrix(pos)))

    platform_dims = (cfg.platform_width, cfg.platform_width, max(step_height * num_steps, 1e-3))
    platform_pos = (terrain_center[0], terrain_center[1], platform_dims[2] / 2.0)
    meshes.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_pos)))
    return meshes, np.array([terrain_center[0], terrain_center[1], platform_dims[2]])
