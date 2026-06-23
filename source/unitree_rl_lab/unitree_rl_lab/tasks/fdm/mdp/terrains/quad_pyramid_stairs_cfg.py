# Copyright (c) 2025, The Nav-Suite Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .quad_pyramid_stairs import quad_pyramid_stairs_terrain


@configclass
class MeshQuadPyramidStairsCfg(SubTerrainBaseCfg):
    """Configuration for a four-sided pyramid stair terrain."""

    function = quad_pyramid_stairs_terrain

    step_height_range: tuple[float, float] = (0.05, 0.25)
    """Minimum and maximum stair height in meters."""

    step_width: float = 0.3
    """Width of each stair step in meters."""

    platform_width: float = 3.0
    """Width of the central platform in meters."""

    border_width: float = 0.0
    """Width of the border around the terrain in meters."""

    holes: bool = False
    """Kept for compatibility with the nav-suite terrain config."""
