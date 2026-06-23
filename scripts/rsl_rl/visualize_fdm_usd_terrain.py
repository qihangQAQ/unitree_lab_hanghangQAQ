#!/usr/bin/env python3

"""Visualize the copied FDM training USD terrain without starting an RL environment."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USD_PATH = (
    REPO_ROOT
    / "source/unitree_rl_lab/unitree_rl_lab/tasks/fdm/data/Terrains/"
    "navigation_terrain_wall_usd_merge_large_single_object_maze.usd"
)

parser = argparse.ArgumentParser(description="Visualize a USD terrain file in Isaac Sim.")
parser.add_argument(
    "--usd_path",
    type=str,
    default=str(DEFAULT_USD_PATH),
    help="Path to the USD terrain file.",
)
parser.add_argument(
    "--camera_distance",
    type=float,
    default=90.0,
    help="Distance of the initial camera view from the terrain center.",
)
parser.add_argument(
    "--camera_height",
    type=float,
    default=65.0,
    help="Height of the initial camera view.",
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=10.0,
    help="Grid spacing used by IsaacLab to configure USD terrain environment origins.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg


def main():
    usd_path = Path(args_cli.usd_path).expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD terrain file not found: {usd_path}")

    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=(-args_cli.camera_distance, -args_cli.camera_distance, args_cli.camera_height),
        target=(0.0, 0.0, 0.0),
    )

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/SkyLight", light_cfg)

    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=str(usd_path),
        env_spacing=args_cli.env_spacing,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    TerrainImporter(terrain_cfg)

    sim.reset()
    print(f"[INFO] Visualizing FDM USD terrain: {usd_path}")
    print("[INFO] Press Ctrl+C or close Isaac Sim to stop.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
