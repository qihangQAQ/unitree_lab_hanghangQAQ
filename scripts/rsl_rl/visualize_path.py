import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher

# 1. 配置参数
parser = argparse.ArgumentParser(description="Visualize NPY path in Isaac Sim.")
parser.add_argument("--file", type=str, required=True, help="Path to the .npy file.")
parser.add_argument("--index", type=int, default=0, help="Index of the batch to visualize (default: 0).")
parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for the points (optional).")

# === 新增：Z轴偏移量参数 ===
parser.add_argument("--z_offset", type=float, default=0.0, help="Height offset to lift the path (in meters).")
# ========================


# 添加 Isaac Lab 标准启动参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 导入 Isaac Lab 模块
# ---------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def load_trajectory_from_npy(filename, index):
    """从 npy 文件加载轨迹数据"""
    print(f"[INFO] Loading file: {filename}")
    try:
        data = np.load(filename, allow_pickle=True).item()

        # 优先读取预测轨迹 'traj_pred'，如果没有则读取真值 'traj'
        if 'traj_pred' in data and data['traj_pred'] is not None:
            traj_batch = data['traj_pred']
            print("[INFO] Found 'traj_pred' data.")
        elif 'traj' in data:
            traj_batch = data['traj']
            print("[INFO] Found 'traj' data.")
        else:
            raise ValueError("File does not contain 'traj' or 'traj_pred' keys.")

        # 确保索引在范围内
        if index >= len(traj_batch):
            print(f"[WARN] Index {index} out of bounds (size {len(traj_batch)}). Using index 0.")
            index = 0

        # 获取单条轨迹 (N, D)
        trajectory = traj_batch[index]
        return trajectory
    except Exception as e:
        print(f"[ERROR] Failed to load NPY file: {e}")
        return None


def main():
    # 3. 配置仿真环境 (空场景)
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 设置灯光和地面
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/SkyLight", cfg_light)
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # 4. 定义可视化标记 (Markers)
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visualizer/PathMarkers",
        markers={
            "path_points": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))  # 红色
            ),
        },
    )
    path_visualizer = VisualizationMarkers(marker_cfg)

    # 5. 加载数据
    traj_points_np = load_trajectory_from_npy(args_cli.file, args_cli.index)

    points_tensor = None
    if traj_points_np is not None:
        print(f"[INFO] Original shape: {traj_points_np.shape}")

        # ====== 修复核心：如果维度超过3，只取前3维 (x,y,z) ======
        if traj_points_np.shape[-1] > 3:
            print(f"[WARN] Data has {traj_points_np.shape[-1]} dimensions. Slicing first 3 as (x,y,z).")
            traj_points_np = traj_points_np[:, :3]
        # ====================================================

        # === 核心修改：加上 Z 轴偏移量 ===
        # 假设 Z 轴是第 2 列 (索引从0开始：0=x, 1=y, 2=z)
        print(f"[INFO] Applying z-offset of {args_cli.z_offset} meters.")
        traj_points_np[:, 2] += args_cli.z_offset
        # ==============================

        # 缩放处理
        traj_points_np = traj_points_np * args_cli.scale
        # 转换为 Tensor
        points_tensor = torch.from_numpy(traj_points_np).float().to(sim.device)
        print(f"[INFO] Visualizing trajectory with {len(points_tensor)} points.")

    # 重置仿真
    sim.reset()

    # 设置相机
    if points_tensor is not None:
        # 计算中心点：(N, 3) -> mean -> (3,)
        center = points_tensor.mean(dim=0).cpu().numpy()
        print(f"[INFO] Centering camera at: {center}")
        # 现在 center 是 (3,)，可以安全相加了
        sim.set_camera_view(eye=center + [2.0, 2.0, 2.0], target=center)

    print("[INFO] Simulation is running...")

    # 6. 主循环
    while simulation_app.is_running():
        if points_tensor is not None:
            path_visualizer.visualize(translations=points_tensor)
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()