#!/usr/bin/env python3

"""Visualize a task terrain generator without starting an RL environment."""

import argparse

from isaaclab.app import AppLauncher

# Edit these two defaults to inspect different task terrain configs.
#
# Examples:
#   velocity task:
#       TERRAIN_CFG_MODULE = "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
#       TERRAIN_CFG_NAME = "ROUGH_TERRAINS_CFG"
#   velocity_perception task:
#       TERRAIN_CFG_MODULE = "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_perception_env_cfg"
#       TERRAIN_CFG_NAME = "ROUGH_TERRAINS_CFG"
#   AME stage 1:
#       TERRAIN_CFG_MODULE = "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_ame_env_cfg"
#       TERRAIN_CFG_NAME = "ROUGH_TERRAINS_CFG"
#   AME stage 2:
#       TERRAIN_CFG_MODULE = "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_ame_env_cfg"
#       TERRAIN_CFG_NAME = "FINETUNE_ROUGH_TERRAINS_CFG"
TERRAIN_CFG_MODULE = "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_ame_env_cfg"
TERRAIN_CFG_NAME = "FINETUNE_ROUGH_TERRAINS_CFG"

parser = argparse.ArgumentParser(description="Visualize a TerrainGeneratorCfg from any task module.")
parser.add_argument(
    "--stage",
    type=int,
    choices=(1, 2),
    default=None,
    help="AME shortcut: 1=ROUGH_TERRAINS_CFG, 2=FINETUNE_ROUGH_TERRAINS_CFG. Overrides cfg_module/cfg_name.",
)
parser.add_argument("--cfg_module", type=str, default=TERRAIN_CFG_MODULE, help="Python module containing the terrain cfg.")
parser.add_argument("--cfg_name", type=str, default=TERRAIN_CFG_NAME, help="TerrainGeneratorCfg variable name to visualize.")
parser.add_argument("--debug_vis", action="store_true", help="Enable terrain debug visualization.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import importlib

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


def _selected_cfg_ref():
    cfg_module = args_cli.cfg_module
    cfg_name = args_cli.cfg_name
    label = f"{cfg_module}:{cfg_name}"

    if args_cli.stage is not None:
        cfg_module = "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_ame_env_cfg"
        cfg_name = "ROUGH_TERRAINS_CFG" if args_cli.stage == 1 else "FINETUNE_ROUGH_TERRAINS_CFG"
        label = f"AME stage {args_cli.stage} ({cfg_name})"

    return cfg_module, cfg_name, label


def _terrain_cfg_to_visualize():
    cfg_module, cfg_name, label = _selected_cfg_ref()
    module = importlib.import_module(cfg_module)
    if not hasattr(module, cfg_name):
        available = sorted(name for name in dir(module) if name.endswith("TERRAINS_CFG") or name.endswith("TERRAIN_CFG"))
        raise AttributeError(f"{cfg_module} has no terrain cfg named {cfg_name!r}. Available candidates: {available}")

    terrain_cfg = getattr(module, cfg_name)
    if not hasattr(terrain_cfg, "sub_terrains"):
        raise TypeError(f"{cfg_module}:{cfg_name} does not look like a TerrainGeneratorCfg with sub_terrains.")

    return copy.deepcopy(terrain_cfg), label, tuple(terrain_cfg.sub_terrains.keys())


def main():
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    terrain_cfg_to_use, stage_name, terrain_tile_names = _terrain_cfg_to_visualize()
    terrain_cfg_to_use.curriculum = True
    terrain_cfg_to_use.num_cols = len(terrain_tile_names)

    tile_size_x, tile_size_y = terrain_cfg_to_use.size
    total_length = terrain_cfg_to_use.num_rows * tile_size_x
    total_width = terrain_cfg_to_use.num_cols * tile_size_y
    center_x = total_length / 2.0
    center_y = total_width / 2.0

    sim.set_camera_view(
        eye=(center_x - 50.0, center_y - 40.0, 50.0),
        target=(center_x, center_y, 0.0),
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
    print(f"[INFO] Visualizing {stage_name} terrain.")
    print("[INFO] Curriculum is set to TRUE for an organized terrain-type x difficulty grid.")
    print(f"[INFO] Terrain types: {', '.join(terrain_tile_names)}")
    print("[INFO] Press Ctrl+C or close Isaac Sim to stop.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
