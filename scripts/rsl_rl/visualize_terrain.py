#!/usr/bin/env python3

"""Visualize the rough terrain set used by velocity_perception_env_cfg.

This script launches an empty Isaac Sim scene and spawns a neatly organized 
10x8 grid of terrains. With curriculum=True, each column is a distinct 
terrain type, and each row is a progressively harder difficulty level.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize exactly 8 terrain types across 10 difficulty levels orderly.")
parser.add_argument("--debug_vis", action="store_true", help="Enable terrain debug visualization.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from unitree_rl_lab.tasks.locomotion.mdp.terrain import PARKOUR_ROUGH_TERRAINS_CFG

TERRAIN_TILE_NAMES = (
    "pyramid_stairs",
    "discrete_obstacles",
    "stairs",
    "random_rough",
    "flat",
    "random_uniform",
    "random_grid",
    "pyramid_slope",
)

PARKOUR_TILE_NAMES = (
    "perlin_rough",
    "perlin_rough_stand",
    "square_gaps",
    "pyramid_stairs",
    "pyramid_stairs_high",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv_high",
    "boxes",
    "mesh_boxes",
    "hf_pyramid_slope_inv",
)

ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,  # 10 个难度等级
    num_cols=len(TERRAIN_TILE_NAMES),  # 8 种地形类型
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,  # <--- 关键修改：开启这里，地形就会按照种类和难度整齐排列！
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            obstacle_height_range=(0.0, 0.2),
            obstacle_width_range=(1.0, 2.0),
            num_obstacles=40,
        ),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.0, 0.2),
            step_width=0.3,
            platform_width=3.0,
        ),
        "random_rough": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            amplitude_range=(0.0, 0.05),
            num_waves=3,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        "random_uniform": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.0, 0.06),
            noise_step=0.02,
        ),
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.49,
            grid_height_range=(0.0, 0.1),
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.15, 0.25),
            platform_width=3.0,
        ),
    },
)

def main():
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 使用 InstinctLab parkour 地形配置
    terrain_cfg_to_use = PARKOUR_ROUGH_TERRAINS_CFG
    terrain_tile_names = PARKOUR_TILE_NAMES
    # 切换地形 -- 切换回原本的 8 种地形：
    # terrain_cfg_to_use = ROUGH_TERRAINS_CFG
    # terrain_tile_names = TERRAIN_TILE_NAMES


    
    terrain_cfg_to_use.curriculum = True
    terrain_cfg_to_use.num_cols = len(terrain_tile_names)

    # 尺寸计算
    total_length = terrain_cfg_to_use.num_rows * 8.0  # 难度方向
    total_width = terrain_cfg_to_use.num_cols * 8.0   # 种类方向
    center_x = total_length / 2.0
    center_y = total_width / 2.0

    # 调整视角：放在地形阵列的侧上方，俯瞰这 num_cols排 x num_rows列 的方阵
    sim.set_camera_view(
        eye=(center_x - 50.0, center_y - 40.0, 50.0),
        target=(center_x, center_y, 0.0)
    )

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/SkyLight", light_cfg)

    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_cfg_to_use,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=args_cli.debug_vis,
    )
    TerrainImporter(terrain_cfg)

    sim.reset()
    print("[INFO] Terrain visualization is running. Press Ctrl+C or close Isaac Sim to stop.")
    print("[INFO] Curriculum is set to TRUE. You should now see an organized grid:")
    print(f"[INFO] One axis shows {len(terrain_tile_names)} terrain types, the other shows {terrain_cfg_to_use.num_rows} difficulty levels.")
    print(f"[INFO] Terrain types: {', '.join(terrain_tile_names)}")

    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()