import argparse
import numpy as np
import torch
import sys

from isaaclab.app import AppLauncher

# ==========================================
# 配置参数
# ==========================================
parser = argparse.ArgumentParser(description="Segment Stitching & Coloring Visualization")
parser.add_argument("--file", type=str, required=True, help="Path to .npy file")
parser.add_argument("--index", type=int, default=0, help="Batch index")
parser.add_argument("--scale", type=float, default=1.0, help="Scale factor")
parser.add_argument("--z_offset", type=float, default=0.0, help="Z offset")
parser.add_argument("--speed", type=float, default=0.02, help="Time per point (seconds)")
parser.add_argument("--lambda_points", type=int, default=4, help="Points per segment")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# =========================================================================
# 核心算法：片段拼接 (Segment Stitching / TSP)
# =========================================================================
def sort_segments_greedy(segments):
    """
    输入: segments shape (N, 4, 3)
    功能: 像接龙一样，把乱序的片段连成一条线
    """
    N = len(segments)
    if N == 0: return np.empty((0, 3))
    if N == 1: return segments[0]

    starts = segments[:, 0, :]  # (N, 3)

    visited = np.zeros(N, dtype=bool)
    ordered_segments = []

    # 找起点
    center = np.mean(starts, axis=0)
    dists = np.linalg.norm(starts - center, axis=1)
    curr_idx = np.argmax(dists)

    visited[curr_idx] = True
    ordered_segments.append(segments[curr_idx])

    # 贪心搜索
    for _ in range(N - 1):
        curr_tail = segments[curr_idx][-1]
        dists_to_heads = np.linalg.norm(starts - curr_tail, axis=1)
        dists_to_heads[visited] = np.inf

        next_idx = np.argmin(dists_to_heads)
        ordered_segments.append(segments[next_idx])
        visited[next_idx] = True
        curr_idx = next_idx

    return np.vstack(ordered_segments)


# =========================================================================
# 主程序
# =========================================================================
def main():
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    cfg_light = sim_utils.DomeLightCfg(intensity=3000.0)
    cfg_light.func("/World/SkyLight", cfg_light)
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # ---------------------------------------------------------
    # 1. 定义多色调色盘 (关键修复)
    # ---------------------------------------------------------
    palette = [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0),
        (1.0, 0.5, 0.0), (0.5, 0.0, 0.5)
    ]

    markers_dict = {}
    for i in range(len(palette)):
        markers_dict[f"m_{i}"] = sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=palette[i])
        )

    marker_cfg = VisualizationMarkersCfg(prim_path="/World/Viz", markers=markers_dict)
    visualizer = VisualizationMarkers(marker_cfg)

    print(f"[INFO] Loading {args_cli.file}...")

    # ---------------------------------------------------------
    # 2. 数据处理 (拼接 + 分配颜色)
    # ---------------------------------------------------------
    try:
        data = np.load(args_cli.file, allow_pickle=True).item()
        idx = args_cli.index

        raw_traj = data.get('traj_pred')[idx] if 'traj_pred' in data else data.get('traj')[idx]

        if 'pred_stroke_masks' in data:
            prob = 1 / (1 + np.exp(-data['pred_stroke_masks'][idx]))
            stroke_ids = np.argmax(prob, axis=0)
        else:
            stroke_ids = data['stroke_ids'][idx]

        # Reshape (N, 4, 6)
        N_segs = raw_traj.shape[0]
        dim_flat = raw_traj.shape[1]
        lambda_pts = args_cli.lambda_points
        dim_per_pt = dim_flat // lambda_pts

        traj_segments = raw_traj.reshape(N_segs, lambda_pts, dim_per_pt)
        traj_xyz = traj_segments[..., :3] * args_cli.scale
        traj_xyz[..., 2] += args_cli.z_offset

        valid_mask = ~np.any(traj_xyz < -90, axis=(1, 2))
        traj_xyz = traj_xyz[valid_mask]
        stroke_ids = stroke_ids[valid_mask]

        unique_ids = sorted(np.unique(stroke_ids))
        final_paths = []

        print(f"[INFO] Processing {len(unique_ids)} paths...")

        for i, uid in enumerate(unique_ids):
            segments_of_path = traj_xyz[stroke_ids == uid]

            if len(segments_of_path) > 0:
                # 拼接
                sorted_points = sort_segments_greedy(segments_of_path)

                # 分配颜色 (循环使用调色盘)
                color_idx = i % len(palette)

                # 存起来：点 + 颜色
                final_paths.append({
                    "points": sorted_points,
                    "color_idx": color_idx
                })

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # 3. 播放
    # ---------------------------------------------------------
    sim.reset()
    if final_paths:
        start_pt = final_paths[0]["points"][0]
        sim.set_camera_view(eye=start_pt + np.array([1.5, 1.5, 1.5]), target=start_pt)

    path_idx = 0
    pt_idx = 0
    timer = 0.0

    history_pts = []
    history_ind = []

    print(f"\n>>> 开始播放 (共 {len(final_paths)} 条路径) <<<")

    while simulation_app.is_running():
        timer += sim.get_physics_dt()

        if timer >= args_cli.speed:
            timer = 0.0

            if path_idx < len(final_paths):
                current_path_data = final_paths[path_idx]
                points = current_path_data["points"]
                color_id = current_path_data["color_idx"]

                if pt_idx < len(points):
                    pt = torch.tensor(points[pt_idx], device=sim.device, dtype=torch.float)

                    # 存入点
                    history_pts.append(pt.unsqueeze(0))
                    # 存入对应的颜色索引
                    history_ind.append(torch.tensor([color_id], device=sim.device, dtype=torch.int32))

                    pt_idx += 1
                else:
                    path_idx += 1
                    pt_idx = 0
                    if path_idx < len(final_paths):
                        print(f"[INFO] 路径 {path_idx - 1} 结束，切换颜色 -> 路径 {path_idx}...")

        # 渲染
        if history_pts:
            pts_tensor = torch.cat(history_pts)
            ind_tensor = torch.cat(history_ind)
            visualizer.visualize(translations=pts_tensor, marker_indices=ind_tensor)

        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()