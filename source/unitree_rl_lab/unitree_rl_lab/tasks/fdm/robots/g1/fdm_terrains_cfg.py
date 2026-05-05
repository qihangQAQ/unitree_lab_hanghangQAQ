"""
FDM terrain configurations for G1.

Mixes traversable terrains (stairs, slopes, rough ground) with non-traversable
obstacles (pillars, walls) to train the Forward Dynamics Model to discriminate
walkable from unwalkable terrain.
"""

import isaaclab.terrains as terrain_gen

from unitree_rl_lab.tasks.fdm.mdp.terrains import (
    MeshPillarTerrainCfg,
    SingleObjectTerrainCfg,
    StairsRampEvalTerrainCfg,
    StairsRampTerrainCfg,
)
from unitree_rl_lab.tasks.fdm.mdp.terrains.single_object import cross_object_pattern

# ==============================================================================
# Simple terrain — single obstacle per tile, good for initial pipeline testing
# ==============================================================================
FDM_SIMPLE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    border_height=2.5,
    sub_terrains={
        # --- traversable ---
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(5e-3, 5e-2),
            noise_step=1e-2,
            border_width=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.2),
            step_width=0.3,
            platform_width=3.0,
        ),
        # --- non-traversable (single objects) ---
        "single_box": SingleObjectTerrainCfg(
            proportion=0.15,
            object_type="box",
            dim_range=[0.8, 1.5],
            height_range=[1.5, 2.5],
        ),
        "single_wall": SingleObjectTerrainCfg(
            proportion=0.15,
            object_type="wall",
            dim_range=[1.0, 2.5],
            height_range=[1.5, 2.5],
            wall_width=0.15,
        ),
        "single_cylinder": SingleObjectTerrainCfg(
            proportion=0.15,
            object_type="cylinder",
            dim_range=[0.4, 0.8],
            height_range=[1.5, 2.5],
        ),
    },
)

# ==============================================================================
# Full training terrain — pillar clusters + stairs/ramp/wall + traversable mix
# ==============================================================================
FDM_TRAIN_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    border_height=2.5,
    sub_terrains={
        # ==================================================================
        # Traversable (can walk through)
        # ==================================================================
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.12),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.08,
            noise_range=(1e-2, 6e-2),
            noise_step=1e-2,
            border_width=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.08,
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.07,
            amplitude_range=(0.0, 0.08),
            num_waves=3,
        ),

        # ==================================================================
        # Non-traversable (must go around)
        # ==================================================================

        # --- Single obstacles ---
        "single_box": SingleObjectTerrainCfg(
            proportion=0.06,
            object_type="box",
            dim_range=[0.8, 1.5],
            height_range=[1.5, 2.5],
        ),
        "single_wall": SingleObjectTerrainCfg(
            proportion=0.06,
            object_type="wall",
            dim_range=[1.5, 3.0],
            height_range=[1.5, 2.5],
            wall_width=0.15,
        ),
        # --- Cross-pattern obstacles (4 objects forming a barrier) ---
        "box_cross": SingleObjectTerrainCfg(
            proportion=0.05,
            object_type="box",
            dim_range=[0.5, 1.0],
            height_range=[1.5, 2.5],
            position_pattern=cross_object_pattern,
        ),
        "wall_cross": SingleObjectTerrainCfg(
            proportion=0.05,
            object_type="wall",
            dim_range=[1.0, 2.0],
            height_range=[1.5, 2.5],
            position_pattern=cross_object_pattern,
            wall_width=0.15,
        ),

        # --- Pillar clusters (random boxes + cylinders) ---
        "pillars_flat": MeshPillarTerrainCfg(
            proportion=0.1,
            box_objects=MeshPillarTerrainCfg.BoxCfg(
                width=(0.4, 1.0),
                length=(0.2, 0.5),
                max_yx_angle=(0, 10),
                height=(1.8, 2.5),
                num_objects=(3, 6),
            ),
            cylinder_cfg=MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5),
                max_yx_angle=(0, 5),
                height=(1.8, 2.5),
                num_objects=(3, 6),
            ),
        ),
        "pillars_rough": MeshPillarTerrainCfg(
            proportion=0.1,
            box_objects=MeshPillarTerrainCfg.BoxCfg(
                width=(0.4, 1.0),
                length=(0.2, 0.5),
                max_yx_angle=(0, 10),
                height=(1.8, 2.5),
                num_objects=(3, 6),
            ),
            cylinder_cfg=MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5),
                max_yx_angle=(0, 5),
                height=(1.8, 2.5),
                num_objects=(3, 6),
            ),
            rough_terrain=terrain_gen.HfRandomUniformTerrainCfg(
                noise_range=(0.02, 0.08),
                noise_step=0.02,
                border_width=0.25,
            ),
        ),

        # --- Stairs + ramp (with wall randomization) ---
        "stairs_ramp_mixed": StairsRampEvalTerrainCfg(
            proportion=0.08,
            modify_step_height=True,
            step_height_range=(0.1, 0.35),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            width_randomization=1.5,
            random_stairs_ramp_position_flipping=True,
            random_wall_probability=0.4,  # 40% chance wall instead of stairs/ramp
            max_height=1.0,
        ),
        "all_wall": StairsRampTerrainCfg(
            proportion=0.05,
            modify_step_height=True,
            step_height_range=(0.15, 0.4),
            step_width=0.3,
            platform_width=1.0,
            all_wall=True,  # always a wall — tests pure non-traversable
            max_height=1.5,
        ),
    },
)

# ==============================================================================
# Play / demo terrain — fewer tiles, visible obstacles for manual inspection
# ==============================================================================
FDM_PLAY_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=1.0,
    num_rows=2,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    border_height=2.5,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "single_box": SingleObjectTerrainCfg(
            proportion=0.25,
            object_type="box",
            dim_range=[1.0, 1.5],
            height_range=[1.5, 2.0],
        ),
        "pillars_flat": MeshPillarTerrainCfg(
            proportion=0.25,
            box_objects=MeshPillarTerrainCfg.BoxCfg(
                width=(0.5, 0.8),
                length=(0.3, 0.5),
                height=(1.5, 2.0),
                num_objects=(3, 5),
            ),
            cylinder_cfg=MeshPillarTerrainCfg.CylinderCfg(
                radius=(0.3, 0.5),
                height=(1.5, 2.0),
                num_objects=(3, 5),
            ),
        ),
        "stairs_wall": StairsRampEvalTerrainCfg(
            proportion=0.25,
            modify_step_height=True,
            step_height_range=(0.15, 0.3),
            step_width=0.3,
            platform_width=1.0,
            center_platform_width=1.0,
            random_wall_probability=0.5,
            max_height=1.0,
        ),
    },
)
