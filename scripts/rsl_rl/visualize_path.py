import argparse
import numpy as np
import torch
import sys

from isaaclab.app import AppLauncher

# 1. 配置参数
parser = argparse.ArgumentParser(description="Standalone Visualization V2 (Fix Colors)")
parser.add_argument("--file", type=str, required=True, help="Path to .npy file")
parser.add_argument("--index", type=int, default=0, help="Batch index")
parser.add_argument("--z_offset", type=float, default=0.0, help="Height offset (meters)")
parser.add_argument("--scale", type=float, default=1.0, help="Scale factor")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# =========================================================================
# 核心函数 1: 从 Mask 概率计算整齐的 ID (Ported from postprocessing.py)
# =========================================================================
def compute_ids_from_masks(pred_masks, conf_scores=None):
    """
    逻辑来源: utils/postprocessing.py -> process_pred_stroke_masks_to_stroke_ids
    功能: 将模型的概率输出 (Logits) 转换为确定的整数 ID
    """
    # pred_masks shape: [N_Strokes, N_Points] (单样本)

    # 1. Sigmoid 激活: 把 Logits 变成 0~1 的概率
    # 公式: 1 / (1 + exp(-x))
    prob_masks = 1 / (1 + np.exp(-pred_masks))

    # 2. 简单的过滤 (可选): 如果某笔画整体置信度太低，可以忽略
    # 这里为了简化，直接做 Argmax，通常效果已经足够好

    # 3. Argmax: 核心步骤！
    # 对每一列(每个点)，看哪一行的概率最大，就选哪一行作为 ID
    # axis=0 表示沿着“笔画数量”的维度找最大值
    stroke_ids = np.argmax(prob_masks, axis=0)

    return stroke_ids


# =========================================================================
# 核心函数 2: 数据清洗 (去除 -100 Padding)
# =========================================================================
def clean_data_logic(traj, stroke_ids):
    """
    逻辑来源: utils/pointcloud.py -> remove_padding_v2
    """
    # 1. 维度展平
    if traj.ndim == 3:
        N, Lambda, Dim = traj.shape
        traj = traj.reshape(-1, Dim)
        if stroke_ids is not None and stroke_ids.ndim == 1:
            stroke_ids = np.repeat(stroke_ids, Lambda)

    # 2. 长度对齐
    if stroke_ids is not None:
        min_len = min(len(traj), len(stroke_ids))
        traj = traj[:min_len]
        stroke_ids = stroke_ids[:min_len]

    # 3. 去除 -100 Padding (关键!)
    fake_mask = np.all((traj[:, :3] == -100), axis=-1)

    valid_traj = traj[~fake_mask]
    valid_ids = None if stroke_ids is None else stroke_ids[~fake_mask]

    return valid_traj, valid_ids


# =========================================================================

def main():
    # 3. 场景配置
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    cfg_light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/SkyLight", cfg_light)
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # 4. 定义调色盘 (12种颜色)
    palette = [
        (0.89, 0.10, 0.11), (0.22, 0.49, 0.72), (0.30, 0.69, 0.29), (0.60, 0.31, 0.64),
        (1.00, 0.50, 0.00), (1.00, 1.00, 0.20), (0.65, 0.34, 0.16), (0.97, 0.51, 0.75),
        (0.60, 0.60, 0.60), (0.00, 0.00, 0.00), (0.50, 1.00, 1.00), (0.50, 0.00, 0.00),
    ]
    num_colors = len(palette)

    markers_dict = {}
    for i, color in enumerate(palette):
        markers_dict[f"marker_{i}"] = sim_utils.SphereCfg(
            radius=0.035,  # 稍微调大一点点，让线条更连续
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
        )

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visualizer/PathMarkers",
        markers=markers_dict,
    )
    path_visualizer = VisualizationMarkers(marker_cfg)

    # 5. 加载数据
    print(f"[INFO] Loading file: {args_cli.file}")
    points_tensor = None
    indices_tensor = None

    try:
        data = np.load(args_cli.file, allow_pickle=True).item()
        idx = args_cli.index

        # A. 获取轨迹
        traj = data.get('traj_pred')[idx] if 'traj_pred' in data else data.get('traj')[idx]

        # B. 获取/计算 ID (核心修复部分)
        stroke_ids = None

        # 优先查找 'pred_stroke_masks' (预测掩膜)，这是画出整齐颜色的关键！
        if 'pred_stroke_masks' in data:
            print("[INFO] Found 'pred_stroke_masks'. Computing IDs via Argmax...")
            masks = data['pred_stroke_masks'][idx]  # Shape: [N_Strokes, N_Points]

            # === 调用计算函数 ===
            stroke_ids = compute_ids_from_masks(masks)
            # =================

        elif 'stroke_ids_pred' in data:
            print("[INFO] Using existing 'stroke_ids_pred'.")
            stroke_ids = data['stroke_ids_pred'][idx]
        elif 'stroke_ids' in data:
            print("[INFO] Using Ground Truth 'stroke_ids'.")
            stroke_ids = data['stroke_ids'][idx]

        print(f"[DEBUG] Traj Shape: {traj.shape}")
        if stroke_ids is not None:
            print(f"[DEBUG] IDs Shape: {stroke_ids.shape}")

        # C. 数据清洗 (-100 Padding)
        traj_clean, ids_clean = clean_data_logic(traj, stroke_ids)

        if len(traj_clean) == 0:
            print("[WARN] Valid trajectory is empty!")

        # D. 坐标变换
        if traj_clean.shape[-1] > 3:
            traj_clean = traj_clean[:, :3]
        traj_clean[:, 2] += args_cli.z_offset
        traj_clean = traj_clean * args_cli.scale

        # E. 转 Tensor
        points_tensor = torch.from_numpy(traj_clean).float().to(sim.device)

        if ids_clean is not None:
            # 取余数映射颜色
            ids_clean = ids_clean.astype(np.int64) % num_colors
            indices_tensor = torch.from_numpy(ids_clean).to(device=sim.device, dtype=torch.long)
        else:
            indices_tensor = torch.arange(len(points_tensor), device=sim.device) % num_colors

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 渲染循环
    sim.reset()
    if points_tensor is not None and len(points_tensor) > 0:
        center = points_tensor.mean(dim=0).cpu().numpy()
        sim.set_camera_view(eye=center + [3.0, 3.0, 3.0], target=center)

    print("[INFO] Visualizing...")
    while simulation_app.is_running():
        if points_tensor is not None:
            n = min(len(points_tensor), len(indices_tensor))
            if n > 0:
                path_visualizer.visualize(
                    translations=points_tensor[:n],
                    marker_indices=indices_tensor[:n]
                )
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()