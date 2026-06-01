"""Test terrain config for keyboard play demo.

All terrain types have equal proportion so each gets its own tile.
Add or comment out entries to control which terrains appear.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import FlatPatchSamplingCfg

TEST_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # --------------------------------------------------
        #  常规地形（全部启用，等比例）
        # --------------------------------------------------
        # 1. 随机金字塔阶梯
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
        ),
        # 2. 离散障碍物
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            obstacle_height_range=(0.0, 0.2),
            obstacle_width_range=(1.0, 2.0),
            num_obstacles=40,
            obstacle_height_mode="fixed",
        ),
        # 3. 坑洼/波浪地面
        "random_rough": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            amplitude_range=(0.0, 0.05),
            num_waves=3,
        ),
        # 4. 平地
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        # 5. 不规则凹凸地面
        "random_uniform": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.0, 0.06),
            noise_step=0.02,
        ),
        # 6. 坡度地形
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.15, 0.32),
            platform_width=3.0,
        ),
        # --------------------------------------------------
        #  Obstacle 障碍地形
        # --------------------------------------------------
        # 7. 高离散障碍柱
        "high_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            num_obstacles=10,
            obstacle_height_mode="fixed",
            obstacle_height_range=(0.7, 1.5),
            obstacle_width_range=(0.3, 1.5),
            platform_width=0.5,
        ),
        # 8. 随机圆柱体（柱子/树干类障碍）
        "repeated_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=1.0,
            platform_width=1.5,
            abs_height_noise=(-0.3, 0.3),
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=15, height=0.6, radius=0.12, max_yx_angle=0.0, degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=25, height=1.2, radius=0.2, max_yx_angle=0.0, degrees=True
            ),
        ),
        # 9. 随机方盒（箱子/集装箱类障碍）
        "repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=1.0,
            platform_width=1.5,
            abs_height_noise=(-0.3, 0.3),
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=12, height=0.5, size=(0.3, 0.3), max_yx_angle=30.0, degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=20, height=1.0, size=(0.5, 0.5), max_yx_angle=45.0, degrees=True
            ),
        ),
        # 10. 随机锥体（碎石/路障锥类障碍）
        "repeated_pyramids": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            proportion=1.0,
            platform_width=1.5,
            abs_height_noise=(-0.3, 0.3),
            object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=15, height=0.4, radius=0.15, max_yx_angle=0.0, degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=25, height=0.8, radius=0.25, max_yx_angle=0.0, degrees=True
            ),
        ),
    },
)

# 为所有地形子类型添加平地采样：机器人只会出生在地形中的平地区域
for sub_cfg in TEST_TERRAINS_CFG.sub_terrains.values():
    sub_cfg.flat_patch_sampling = {
        "init_pos": FlatPatchSamplingCfg(
            num_patches=2,
            patch_radius=[0.01, 0.1, 0.5, 1.0],
            max_height_diff=0.5,
        )
    }
