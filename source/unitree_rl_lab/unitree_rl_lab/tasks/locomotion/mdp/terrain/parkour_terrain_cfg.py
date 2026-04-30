"""Parkour terrain set copied from InstinctLab.

This module keeps the terrain recipe separate from the environment config so
locomotion tasks can import the generator directly.
"""

from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg

from . import (
    PerlinDiscreteObstaclesTerrainCfg,
    PerlinInvertedPyramidSlopedTerrainCfg,
    PerlinInvertedPyramidStairsTerrainCfg,
    PerlinMeshRandomMultiBoxTerrainCfg,
    PerlinPlaneTerrainCfg,
    PerlinPyramidStairsTerrainCfg,
    PerlinSquareGapTerrainCfg,
)


PARKOUR_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=3,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=1.0,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "perlin_rough": PerlinPlaneTerrainCfg(
            proportion=0.05,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                ),
            },
        ),
        "perlin_rough_stand": PerlinPlaneTerrainCfg(
            proportion=0.05,
            noise_scale=[0.0, 0.1],
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                ),
            },
        ),
        "square_gaps": PerlinSquareGapTerrainCfg(
            proportion=0.10,
            gap_distance_range=(0.1, 0.7),
            gap_depth=(0.4, 0.6),
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs": PerlinPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_high": PerlinPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.45),
            step_width=1.5,
            platform_width=4.0,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_inv": PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=2.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "pyramid_stairs_inv_high": PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.45),
            step_width=1.5,
            platform_width=4.0,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                    x_range=(3.7, 3.7),
                    y_range=(-0.0, 0.0),
                ),
            },
        ),
        "boxes": PerlinDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            num_obstacles=20,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.8, 1.5),
            obstacle_height_range=(0.05, 0.45),
            platform_width=1.5,
            border_width=0.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.05,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                ),
            },
        ),
        "mesh_boxes": PerlinMeshRandomMultiBoxTerrainCfg(
            proportion=0.10,
            box_height_mean=[0.1, 0.4],
            box_height_range=0.05,
            box_length_mean=0.4,
            box_length_range=0.1,
            box_width_mean=0.4,
            box_width_range=0.1,
            platform_width=1.5,
            generation_ratio=0.3,
            no_perlin_at_obstacle=True,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15],
                    max_height_diff=0.05,
                ),
            },
        ),
        "hf_pyramid_slope_inv": PerlinInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.7),
            platform_width=1.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=PerlinPlaneTerrainCfg(
                noise_scale=0.00,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                ),
            },
        ),
    },
)

# Keep the InstinctLab name for drop-in imports.
ROUGH_TERRAINS_CFG = PARKOUR_ROUGH_TERRAINS_CFG

