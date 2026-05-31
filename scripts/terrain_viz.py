"""
地形可视化脚本 - 展示 IsaacLab v2.3.0 中所有障碍相关地形。

每个子地形占据网格中的一个 cell，所有地形同时加载，
可以在 Isaac Sim 中自由飞行查看。

Usage:
    # 在当前项目环境中运行
    python scripts/terrain_viz.py

    # 如果在 headless 服务器上
    python scripts/terrain_viz.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Obstacle terrain visualization.")
parser.add_argument("--num_rows", type=int, default=3, help="Number of terrain rows.")
parser.add_argument("--num_cols", type=int, default=4, help="Number of terrain columns.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


def build_obstacle_terrains() -> TerrainGeneratorCfg:
    """Build a terrain generator config with all obstacle-related sub-terrains."""

    sub_terrains = {
        # ============ Height Field 地形 ============

        # "1_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #     proportion=1.0,
        #     num_obstacles=15,
        #     obstacle_height_mode="fixed",
        #     obstacle_height_range=(0.5, 1.5),
        #     obstacle_width_range=(0.2, 0.8),
        #     platform_width=1.5,
        # ),

        # "2_stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #     proportion=1.0,
        #     stone_height_max=0.3,
        #     stone_width_range=(0.2, 0.5),
        #     stone_distance_range=(0.2, 0.5),
        #     holes_depth=-1.0,
        #     platform_width=1.5,
        # ),

        # ============ Mesh 重复物体地形（最适合避障） ============

        "3_repeated_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=1.0,
            platform_width=1.5,
            abs_height_noise=(-0.3, 0.3),
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=20, height=0.8, radius=0.12, max_yx_angle=0.0, degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=30, height=1.2, radius=0.2, max_yx_angle=0.0, degrees=True
            ),
        ),

        "4_repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=1.0,
            platform_width=1.5,
            abs_height_noise=(-0.3, 0.3),
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=15, height=0.6, size=(0.3, 0.3), max_yx_angle=30.0, degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=25, height=1.0, size=(0.5, 0.5), max_yx_angle=45.0, degrees=True
            ),
        ),

        "5_repeated_pyramids": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            proportion=1.0,
            platform_width=1.5,
            abs_height_noise=(-0.3, 0.3),
            object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=20, height=0.5, radius=0.15, max_yx_angle=0.0, degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=30, height=1.0, radius=0.25, max_yx_angle=0.0, degrees=True
            ),
        ),

        # ============ Mesh 坑/间隙地形 ============

        # "6_gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=1.0,
        #     platform_width=1.5,
        #     gap_width_range=(0.3, 0.8),
        # ),

        # "7_pit": terrain_gen.MeshPitTerrainCfg(
        #     proportion=1.0,
        #     platform_width=1.5,
        #     pit_depth_range=(0.3, 0.8),
        #     double_pit=False,
        # ),

        # "8_double_pit": terrain_gen.MeshPitTerrainCfg(
        #     proportion=1.0,
        #     platform_width=1.5,
        #     pit_depth_range=(0.3, 0.8),
        #     double_pit=True,
        # ),

        # # ============ Mesh 其他障碍结构 ============

        # "9_random_grid": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=1.0,
        #     platform_width=1.5,
        #     grid_width=0.5,
        #     grid_height_range=(0.05, 0.3),
        #     holes=False,
        # ),

        # "10_floating_ring": terrain_gen.MeshFloatingRingTerrainCfg(
        #     proportion=1.0,
        #     platform_width=1.5,
        #     ring_height_range=(0.4, 1.0),
        #     ring_width_range=(0.5, 1.0),
        #     ring_thickness=0.05,
        # ),

        # "11_star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=1.0,
        #     platform_width=1.5,
        #     num_bars=5,
        #     bar_width_range=(0.4, 0.8),
        #     bar_height_range=(0.1, 0.3),
        # ),

        # "12_rails": terrain_gen.MeshRailsTerrainCfg(
        #     proportion=1.0,
        #     platform_width=1.5,
        #     rail_thickness_range=(0.05, 0.1),
        #     rail_height_range=(0.1, 0.3),
        # ),
    }

    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=2.0,
        num_rows=args_cli.num_rows,
        num_cols=args_cli.num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains=sub_terrains,
    )


def design_scene() -> dict:
    """Design the scene with obstacle terrains."""
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Build terrain generator
    terrain_gen_cfg = build_obstacle_terrains()

    # Terrain importer
    terrain_importer_cfg = TerrainImporterCfg(
        num_envs=args_cli.num_rows * args_cli.num_cols,
        env_spacing=2.5,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=True,
    )
    terrain_importer = TerrainImporter(terrain_importer_cfg)

    return {"terrain": terrain_importer}


def run_simulator(sim: sim_utils.SimulationContext):
    """Run the simulation loop."""
    while simulation_app.is_running():
        sim.step()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set camera to overview position
    sim.set_camera_view(eye=[10.0, 10.0, 8.0], target=[0.0, 0.0, 0.0])

    design_scene()
    sim.reset()

    print("[INFO]: All obstacle terrains loaded. Fly around in Isaac Sim to inspect them.")
    print(f"[INFO]: Layout: {args_cli.num_rows} rows x {args_cli.num_cols} cols")
    print("[INFO]: Terrain types (left to right, top to bottom):")
    print("  1. HfDiscreteObstacles (fixed pillars)")
    print("  2. HfSteppingStones")
    print("  3. MeshRepeatedCylinders")
    print("  4. MeshRepeatedBoxes")
    print("  5. MeshRepeatedPyramids")
    print("  6. MeshGap")
    print("  7. MeshPit")
    print("  8. MeshDoublePit")
    print("  9. MeshRandomGrid")
    print("  10. MeshFloatingRing")
    print("  11. MeshStar")
    print("  12. MeshRails")

    run_simulator(sim)


if __name__ == "__main__":
    main()
    simulation_app.close()
