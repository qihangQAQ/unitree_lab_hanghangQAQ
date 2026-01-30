import argparse
import numpy as np
import torch
import sys
import time

from isaaclab.app import AppLauncher

# ==========================================
# 1. 配置参数
# ==========================================
parser = argparse.ArgumentParser(description="Complete Sequential Visualization (Sorted & Colored)")
parser.add_argument("--file", type=str, required=True, help="Path to .npy file")
parser.add_argument("--index", type=int, default=0, help="Batch index")
parser.add_argument("--z_offset", type=float, default=0.0, help="Height offset (meters)")
parser.add_argument("--scale", type=float, default=1.0, help="Scale factor")
parser.add_argument("--speed", type=float, default=0.02, help="Time interval between points (seconds)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# =========================================================================
# 核心算法模块
# =========================================================================

def compute_ids_from_masks(pred_masks):
    """
    [算法1] 修复颜色杂乱
    从概率 (Logits) 中计算确定的 ID (Argmax)
    """
    # Sigmoid 激活
    prob_masks = 1 / (1 + np.exp(-pred_masks))
    # Argmax: 选概率最大的那个笔画作为 ID
    stroke_ids = np.argmax(prob_masks, axis=0)
    return stroke_ids


def clean_data_logic(traj, stroke_ids):
    """
    [算法2] 去除无效数据
    去除 -100 的 Padding 点，并对齐长度
    """
    # 维度展平
    if traj.ndim == 3:
        N, Lambda, Dim = traj.shape
        traj = traj.reshape(-1, Dim)
        if stroke_ids is not None and stroke_ids.ndim == 1:
            stroke_ids = np.repeat(stroke_ids, Lambda)

    # 长度对齐
    if stroke_ids is not None:
        min_len = min(len(traj), len(stroke_ids))
        traj = traj[:min_len]
        stroke_ids = stroke_ids[:min_len]

    # 核心：去除 -100
    fake_mask = np.all((traj[:, :3] == -100), axis=-1)
    valid_traj = traj[~fake_mask]
    valid_ids = None if stroke_ids is None else stroke_ids[~fake_mask]

    return valid_traj, valid_ids


def sort_points_greedy(points):
    """
    [算法3] 恢复时间顺序 (贪心/最近邻算法)
    原理：找一个端点开始，每次找最近的下一个点
    """
    if len(points) <= 1:
        return points

    pts = np.array(points)
    N = len(pts)
    visited = np.zeros(N, dtype=bool)
    ordered_indices = []

    # --- 启发式找起点 ---
    # 简单的逻辑：找离几何中心最远的点，通常是端点
    centroid = np.mean(pts, axis=0)
    dists_from_center = np.linalg.norm(pts - centroid, axis=1)
    start_idx = np.argmax(dists_from_center)

    current_idx = start_idx
    ordered_indices.append(current_idx)
    visited[current_idx] = True

    # --- 贪心搜索 ---
    for _ in range(N - 1):
        last_pt = pts[current_idx]

        # 计算到所有点的距离
        dists = np.linalg.norm(pts - last_pt, axis=1)
        # 已访问过的设为无限大
        dists[visited] = np.inf

        # 找最近的
        next_idx = np.argmin(dists)

        ordered_indices.append(next_idx)
        visited[next_idx] = True
        current_idx = next_idx

    return pts[ordered_indices]


# =========================================================================
# 主程序
# =========================================================================

def main():
    # 1. 场景设置
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    cfg_light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/SkyLight", cfg_light)
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # 2. 定义调色盘 (12色循环)
    palette = [
        (1.0, 0.0, 0.0),  # 红
        (0.0, 1.0, 0.0),  # 绿
        (0.0, 0.0, 1.0),  # 蓝
        (1.0, 1.0, 0.0),  # 黄
        (0.0, 1.0, 1.0),  # 青
        (1.0, 0.0, 1.0),  # 品红
        (1.0, 0.5, 0.0),  # 橙
        (0.5, 0.0, 0.5),  # 紫
        (0.5, 0.5, 0.5),  # 灰
        (0.0, 0.0, 0.0),  # 黑
        (0.5, 1.0, 0.5),  # 浅绿
        (0.5, 0.0, 0.0),  # 深红
    ]

    # 创建 markers 配置
    markers_dict = {}
    for i, color in enumerate(palette):
        markers_dict[f"marker_{i}"] = sim_utils.SphereCfg(
            radius=0.035,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        )

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visualizer/PathMarkers",
        markers=markers_dict,
    )
    path_visualizer = VisualizationMarkers(marker_cfg)

    # 3. 数据处理流水线
    print(f"[INFO] Processing file: {args_cli.file}")

    # 最终播放列表：元素为 (Tensor点集, 颜色ID)
    animation_sequence = []

    try:
        data = np.load(args_cli.file, allow_pickle=True).item()
        idx = args_cli.index

        # A. 获取原始数据
        traj = data.get('traj_pred')[idx] if 'traj_pred' in data else data.get('traj')[idx]

        # B. 获取并修复 ID
        if 'pred_stroke_masks' in data:
            print("[INFO] Computing clean IDs from masks...")
            stroke_ids = compute_ids_from_masks(data['pred_stroke_masks'][idx])
        elif 'stroke_ids_pred' in data:
            stroke_ids = data['stroke_ids_pred'][idx]
        else:
            stroke_ids = data['stroke_ids'][idx]

        # C. 清洗 (-100 padding)
        traj_clean, ids_clean = clean_data_logic(traj, stroke_ids)

        if len(traj_clean) == 0:
            print("[ERROR] No valid data found!")
            return

        # D. 坐标变换
        if traj_clean.shape[-1] > 3: traj_clean = traj_clean[:, :3]
        traj_clean[:, 2] += args_cli.z_offset
        traj_clean = traj_clean * args_cli.scale

        # E. 分组并排序 (关键步骤)
        unique_ids = np.unique(ids_clean)
        print(f"[INFO] Found {len(unique_ids)} unique paths. Sorting points...")

        for uid in unique_ids:
            # 1. 提取当前 Path 的所有点
            mask = (ids_clean == uid)
            points_group = traj_clean[mask]

            # 2. 对这一组点进行排序 (恢复时间顺序)
            sorted_group = sort_points_greedy(points_group)

            # 3. 存入序列 (转为 Tensor)
            pts_tensor = torch.from_numpy(sorted_group).float().to(sim.device)
            # 颜色 ID 取余数
            color_id = int(uid) % len(palette)

            animation_sequence.append({
                "points": pts_tensor,
                "color": color_id,
                "count": len(pts_tensor)
            })

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 动画循环
    sim.reset()

    # 设置相机看第一条路径的起点
    if len(animation_sequence) > 0:
        center = animation_sequence[0]["points"][0].cpu().numpy()
        sim.set_camera_view(eye=center + [2.0, 2.0, 2.0], target=center)

    print("[INFO] Playing Animation...")

    # 状态变量
    current_path_idx = 0  # 当前画第几条线
    current_point_idx = 0  # 当前线画到了第几个点
    timer = 0.0

    # 用于存储已经画完的所有线的点和颜色 (用于保持显示)
    static_points = []
    static_indices = []

    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        timer += dt

        # --- 更新逻辑 ---
        if timer >= args_cli.speed:
            timer = 0.0

            # 如果还有线没画完
            if current_path_idx < len(animation_sequence):
                path_data = animation_sequence[current_path_idx]

                # 当前线往前画一个点
                if current_point_idx < path_data["count"]:
                    current_point_idx += 1
                else:
                    # 当前线画完了，把它的数据存入"静态列表"
                    static_points.append(path_data["points"])
                    # 生成对应的颜色索引 Tensor
                    idx_tensor = torch.full((path_data["count"],), path_data["color"], device=sim.device,
                                            dtype=torch.long)
                    static_indices.append(idx_tensor)

                    # 切换到下一条线
                    current_path_idx += 1
                    current_point_idx = 0

        # --- 渲染逻辑 ---
        # 1. 收集所有已完成的线
        to_draw_points = list(static_points)
        to_draw_indices = list(static_indices)

        # 2. 加入当前正在画的线 (动态部分)
        if current_path_idx < len(animation_sequence):
            path_data = animation_sequence[current_path_idx]
            if current_point_idx > 0:
                # 只取前 current_point_idx 个点
                to_draw_points.append(path_data["points"][:current_point_idx])
                # 生成对应的颜色
                idx_tensor = torch.full((current_point_idx,), path_data["color"], device=sim.device, dtype=torch.long)
                to_draw_indices.append(idx_tensor)

        # 3. 拼接并绘制
        if len(to_draw_points) > 0:
            final_points = torch.cat(to_draw_points)
            final_indices = torch.cat(to_draw_indices)

            path_visualizer.visualize(
                translations=final_points,
                marker_indices=final_indices
            )

        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()