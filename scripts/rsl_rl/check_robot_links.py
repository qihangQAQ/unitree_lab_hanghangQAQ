# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

# 1. 配置命令行参数
parser = argparse.ArgumentParser(description="Export Robot Link Information to Console.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # 强制不启动 GUI 窗口

# 2. 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
import torch
import numpy as np
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation

# 导入 G1 配置
try:
    from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
    ROBOT_NAME = "Unitree-G1-29dof"
except ImportError:
    print("[ERROR] 无法找到 Unitree G1 配置。")
    simulation_app.close()
    exit()

def main():
    # 3. 配置仿真环境
    sim_cfg = SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 4. 配置机器人
    robot_cfg = ROBOT_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.spawn.fix_root_link = True
    robot = Articulation(robot_cfg)

    # 5. 重置并计算物理数据
    sim.reset()
    robot.update(0.0) 

    # ---------------------------------------------------------
    # 6. 获取数据
    # ---------------------------------------------------------
    body_names = robot.body_names
    pos_w = robot.data.body_pos_w[0]   # [num_bodies, 3]
    quat_w = robot.data.body_quat_w[0] # [num_bodies, 4] (w, x, y, z)
    
    # 修复 AttributeError: 使用 euler_xyz_from_quat
    # 该函数返回 (roll, pitch, yaw) 组成的 tuple
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_w)
    euler_w = torch.stack([roll, pitch, yaw], dim=-1) * 180.0 / np.pi

    # ---------------------------------------------------------
    # 7. 纯净输出
    # ---------------------------------------------------------
    print("\n" + "="*100)
    print(f"ROBOT: {ROBOT_NAME}")
    print(f"{'INDEX':<7} | {'LINK NAME':<35} | {'POSITION (X, Y, Z)':<25} | {'ORIENTATION (R, P, Y) deg'}")
    print("-" * 100)

    for i, name in enumerate(body_names):
        p = pos_w[i]
        e = euler_w[i]
        
        pos_str = f"{p[0]:>6.3f}, {p[1]:>6.3f}, {p[2]:>6.3f}"
        euler_str = f"{e[0]:>6.1f}, {e[1]:>6.1f}, {e[2]:>6.1f}"
        
        print(f"{i:<7} | {name:<35} | {pos_str:<25} | {euler_str}")
    
    print("="*100 + "\n")

    simulation_app.close()

if __name__ == "__main__":
    main()