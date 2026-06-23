# Copyright (c) 2025, The Nav-Suite Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .random_maze import random_maze_terrain


@configclass
class RandomMazeTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a randomized wall maze terrain."""

    function = random_maze_terrain

    border_width: float = 0.0
    """Width of the border around the terrain in meters."""

    resolution: float = 1.25
    """Approximate cell size of the maze in meters."""

    maze_height: float = 3.0
    """Height of maze walls in meters."""

    step_height_range: tuple[float, float] = (0.15, 0.25)
    """Minimum and maximum stair block height in meters."""

    step_width_range: tuple[float, float] = (0.25, 0.35)
    """Minimum and maximum stair block width in meters."""

    num_stairs: int = 5
    """Number of small stair blocks sprinkled into the maze."""
