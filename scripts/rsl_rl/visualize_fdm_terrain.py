#!/usr/bin/env python3

"""Visualise the FDM planner demo terrain without starting an RL environment.

Reuses TerrainGeneratorCfg directly via TerrainImporter — same pattern as
``visualize_terrain.py``.  By default this keeps the selected cfg unchanged,
matching the FDM planner test/demo terrain setup.  Use ``--random_layout`` to
force scattered terrain tiles.

Usage::

    python scripts/rsl_rl/visualize_fdm_terrain.py
    python scripts/rsl_rl/visualize_fdm_terrain.py --random_layout
    python scripts/rsl_rl/visualize_fdm_terrain.py --cfg_name FDM_TERRAINS_CFG
"""

from __future__ import annotations

import argparse
import copy
import importlib

from isaaclab.app import AppLauncher

# Default: FDM plan_test --mode test terrain (large planner eval obstacle field).
TERRAIN_CFG_MODULE = "unitree_rl_lab.tasks.fdm.terrain_cfg"
TERRAIN_CFG_NAME = "PLANNER_EVAL_CFG"

parser = argparse.ArgumentParser(description="Visualise an FDM TerrainGeneratorCfg.")
parser.add_argument(
    "--cfg_module", type=str, default=TERRAIN_CFG_MODULE, help="Python module with the terrain cfg."
)
parser.add_argument(
    "--cfg_name", type=str, default=TERRAIN_CFG_NAME, help="TerrainGeneratorCfg variable name."
)
parser.add_argument(
    "--random_layout",
    action="store_true",
    help="Scatter terrain tiles randomly (curriculum=False) instead of using the cfg layout.",
)
parser.add_argument("--debug_vis", action="store_true", help="Enable terrain debug wireframe.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


def _load_terrain_cfg():
    module = importlib.import_module(args_cli.cfg_module)
    if not hasattr(module, args_cli.cfg_name):
        available = sorted(
            name for name in dir(module) if name.endswith("TERRAINS_CFG") or name.endswith("TERRAIN_CFG")
        )
        raise AttributeError(
            f"{args_cli.cfg_module} has no {args_cli.cfg_name!r}. Available: {available}"
        )

    terrain_cfg = getattr(module, args_cli.cfg_name)
    if not hasattr(terrain_cfg, "sub_terrains"):
        raise TypeError(f"{args_cli.cfg_module}:{args_cli.cfg_name} is not a TerrainGeneratorCfg with sub_terrains.")

    return copy.deepcopy(terrain_cfg)


def main():
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    terrain_cfg = _load_terrain_cfg()
    terrain_names = tuple(terrain_cfg.sub_terrains.keys())

    if args_cli.random_layout:
        terrain_cfg.curriculum = False

    tile_x, tile_y = terrain_cfg.size
    total_x = terrain_cfg.num_rows * tile_x
    total_y = terrain_cfg.num_cols * tile_y
    sim.set_camera_view(
        eye=(total_x / 2 - 20, total_y / 2 - 15, 20),
        target=(total_x / 2, total_y / 2, 0),
    )

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/SkyLight", light_cfg)

    TerrainImporter(
        TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_cfg,
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=args_cli.debug_vis,
        )
    )

    sim.reset()
    mode = "random (scattered)" if args_cli.random_layout else "cfg default layout"
    print(f"[INFO] Visualizing {args_cli.cfg_module}:{args_cli.cfg_name}  --  {mode}")
    print(f"[INFO] Terrain types: {', '.join(terrain_names)}")
    print("[INFO] Press Ctrl+C or close Isaac Sim to stop.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
