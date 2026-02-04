# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

# 1. 配置命令行参数
parser = argparse.ArgumentParser(description="Launch Isaac Sim with toggleable floating base.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 必须在 simulation_app 启动后导入
# ---------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation
# [新增] 导入可视化标记工具
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# 导入 G1 配置
try:
    from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG

    print("[INFO] Successfully imported UNITREE_G1_29DOF_CFG.")
except ImportError:
    print("[ERROR] Could not import 'UNITREE_G1_29DOF_CFG'. Environment might be incomplete.")
    from isaaclab.assets import ArticulationCfg

    ROBOT_CFG = ArticulationCfg(spawn=None, init_state=None)  # Placeholder

# ================== 用户控制开关 ==================
# True  = 悬在空中 (MuJoCo 无 freejoint)
# False = 自由下落 (MuJoCo 有 freejoint)
FIX_BASE = True


# ==================================================

def main():
    """启动场景，根据 FIX_BASE 决定是否悬空，并可视化末端坐标系"""

    # 3. 配置仿真上下文 (注意 dt 设置)
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 4. 设置场景基础设施
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/SkyLight", cfg_light)

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # 5. 配置机器人
    robot_cfg = ROBOT_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"

    # --- 核心修正 ---
    if FIX_BASE:
        print("[MODE] Fixed Base")
        robot_cfg.init_state.pos = (0.0, 0.0, 1.5)
        if hasattr(robot_cfg.spawn, "fix_root_link"):
            robot_cfg.spawn.fix_root_link = True
        else:
            print("[WARNING] robot_cfg.spawn does not have fix_root_link param!")
    else:
        print("[MODE] Floating Base")
        robot_cfg.init_state.pos = (0.0, 0.0, 1.5)
        if hasattr(robot_cfg.spawn, "fix_root_link"):
            robot_cfg.spawn.fix_root_link = False

    # 创建机器人
    robot = Articulation(robot_cfg)

    # ================== [新增] 配置可视化坐标轴 ==================
    # 复制默认的坐标系配置 (红=X, 绿=Y, 蓝=Z)
    frame_marker_cfg = FRAME_MARKER_CFG.copy()

    # 【关键修复】必须明确指定生成的 prim 路径，否则会报错 Sdf.Path(MISSING)
    frame_marker_cfg.prim_path = "/World/Visuals/EE_Frame"

    # 设置缩放比例：(0.2, 0.2, 0.2) 表示箭头长 20cm
    frame_marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    # 创建标记器实例
    frame_marker = VisualizationMarkers(frame_marker_cfg)

    # 定义我们要看的手部 link 名字
    ee_name = "right_wrist_roll_link"
    ee_id = None
    # ==========================================================

    # 6. 重置仿真器
    sim.reset()
    print("[INFO]: Simulation is running... Press Ctrl+C to stop.")

    # 7. 查找末端索引 (需要在 reset 后查找，此时物理引擎已加载)
    try:
        # find_bodies 返回的是 (indices, names)，我们要取第一个匹配的 index
        ee_id = robot.find_bodies(ee_name)[0][0]
        print(f"[INFO] VISUALIZATION: Found '{ee_name}' at body index {ee_id}. Drawing axes...")
    except IndexError:
        print(f"[ERROR] VISUALIZATION: Could not find body named '{ee_name}'. Axes will not be drawn.")
        print(f"Available bodies: {robot.body_names}")

    # 8. 仿真主循环
    while simulation_app.is_running():
        # 写入默认数据
        robot.write_data_to_sim()

        # 物理步进
        sim.step()

        # 更新 buffer (sim.cfg.dt)
        robot.update(sim.cfg.dt)

        # ================== [新增] 绘制坐标轴 ==================
        if ee_id is not None:
            # 获取末端在世界坐标系下的位置 (pos) 和 姿态 (quat)
            # shape: (num_envs, num_bodies, 3/4) -> 取第0个环境，指定body
            ee_pos = robot.data.body_pos_w[:, ee_id, :]
            ee_quat = robot.data.body_quat_w[:, ee_id, :]

            # 让 VisualizationMarkers 在这个位置画出坐标轴
            frame_marker.visualize(ee_pos, ee_quat)
        # ======================================================


if __name__ == "__main__":
    main()
    simulation_app.close()