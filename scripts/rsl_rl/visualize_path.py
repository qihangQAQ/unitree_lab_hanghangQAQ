import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# ==========================================
# 配置参数
# ==========================================
parser = argparse.ArgumentParser(description="Postprocessed Paths Visualization (No Greedy)")

parser.add_argument("--file", type=str, required=True, help="Path to .npy/.npz file")
parser.add_argument("--index", type=int, default=0, help="Batch index (which object to visualize)")

# 兼容：默认画 pred；需要画 GT 时显式 --use_gt
group = parser.add_mutually_exclusive_group()
group.add_argument("--use_gt", action="store_true", help="Visualize GT (default: visualize prediction)")
group.add_argument("--use_pred", action="store_true", help="(Deprecated) visualize prediction explicitly")

parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for xyz")
parser.add_argument("--z_offset", type=float, default=0.0, help="Z offset added to xyz")
parser.add_argument("--speed", type=float, default=0.02, help="Time per point (seconds)")

# 当你可视化 raw segments 时需要（你的数据是 lambda=4 -> 24=6*4）
parser.add_argument("--lambda_points", type=int, default=4, help="Points per segment (lambda) if using RAW segments")

parser.add_argument("--max_paths", type=int, default=-1, help="Max number of paths to render (-1 means all)")
parser.add_argument("--min_points_per_path", type=int, default=2, help="Filter out tiny paths")

parser.add_argument(
    "--drop_overlap",
    action="store_true",
    help="Drop overlapped waypoint when concatenating segments (recommended)"
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


# =========================================================================
# Helpers
# =========================================================================
def load_any(path: str):
    """Load .npy (dict) or .npz (keys) into a Python dict."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}

    obj = np.load(path, allow_pickle=True)
    # common pattern: 0-d object array containing dict
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        return obj.item()
    if isinstance(obj, dict):
        return obj
    return {"array": obj}


def stable_unique_order(arr_1d: np.ndarray):
    """Return unique values in order of first appearance."""
    order = []
    seen = set()
    for v in arr_1d.tolist():
        if v not in seen:
            seen.add(v)
            order.append(v)
    return order


def build_paths_from_pointwise_xyz(xyz: np.ndarray, stroke_ids: np.ndarray):
    """
    xyz: (T,3)
    stroke_ids: (T,)
    Return: list of paths, each path is (Ti,3), in the order of first appearance of stroke_id.
    """
    stroke_ids = stroke_ids.astype(np.int64)
    order = stable_unique_order(stroke_ids)

    paths = []
    for sid in order:
        pts = xyz[stroke_ids == sid]
        if pts.shape[0] >= args_cli.min_points_per_path:
            paths.append(pts)
    return paths


def _reshape_segments_to_xyz(traj_flat: np.ndarray, lambda_pts: int):
    """
    traj_flat: (K, 6*lambda) or (K, 24) where 24 = 6*4
    return: (K, lambda, 3) xyz only
    """
    if traj_flat.ndim != 2:
        raise ValueError(f"traj_flat must be 2D, got shape {traj_flat.shape}")
    K, dim_flat = traj_flat.shape
    dim_per_pt = dim_flat // lambda_pts
    if dim_per_pt * lambda_pts != dim_flat:
        raise ValueError(f"dim_flat {dim_flat} not divisible by lambda_points {lambda_pts}")
    seg = traj_flat.reshape(K, lambda_pts, dim_per_pt)
    xyz = seg[..., :3]
    return xyz


def build_paths_from_segmentwise(traj_flat: np.ndarray,
                                stroke_ids: np.ndarray,
                                lambda_pts: int,
                                drop_overlap: bool = True):
    """
    Fallback for RAW segments:
    traj_flat: (K, 6*lambda)
    stroke_ids: (K,)
    Return: list of paths (each (T,3)) by concatenating segments in their given order.
    NOTE: If traj_flat is RAW unordered segments, this will still look messy.
    """
    stroke_ids = stroke_ids.astype(np.int64)

    seg_xyz = _reshape_segments_to_xyz(traj_flat, lambda_pts)  # (K, lambda, 3)

    order = stable_unique_order(stroke_ids)
    paths = []

    for sid in order:
        idxs = np.nonzero(stroke_ids == sid)[0]
        if idxs.size == 0:
            continue

        pts = []
        last = None
        for j in idxs:
            seg_pts = seg_xyz[j]  # (lambda, 3)
            for k in range(seg_pts.shape[0]):
                p = seg_pts[k]
                if drop_overlap and last is not None:
                    if np.linalg.norm(p - last) < 1e-9:
                        continue
                pts.append(p)
                last = p

        if len(pts) >= args_cli.min_points_per_path:
            paths.append(np.stack(pts, axis=0))

    return paths


def build_paths_from_explicit_paths(paths_obj, scale: float, z_offset: float):
    """
    If you saved postprocess results directly as list of paths:
      paths = [ (T0,6) or (T0,3), (T1,6) or (T1,3), ... ]
    Convert to list[(T,3)] xyz.
    """
    out = []
    for p in paths_obj:
        arr = np.asarray(p)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        xyz = arr[:, :3].astype(np.float32)
        xyz *= scale
        xyz[:, 2] += z_offset
        if not np.any(xyz < -90):
            out.append(xyz)
    return out


def take_sample(maybe_list_or_array, idx: int):
    """
    For fields that can be:
      - list(len=B) each element variable-length ndarray
      - ndarray with batch dimension (B,...)
    """
    if isinstance(maybe_list_or_array, list):
        return maybe_list_or_array[idx]
    if isinstance(maybe_list_or_array, np.ndarray) and maybe_list_or_array.ndim >= 1:
        return maybe_list_or_array[idx]
    return maybe_list_or_array


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
    # 1. 调色盘
    # ---------------------------------------------------------
    palette = [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0),
        (1.0, 0.5, 0.0), (0.5, 0.0, 0.5),
        (0.7, 0.7, 0.7), (0.2, 0.6, 0.9)
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
    data = load_any(args_cli.file)
    idx = args_cli.index

    # 默认画 pred（除非显式 --use_gt）
    use_gt = args_cli.use_gt
    use_pred = (not use_gt)

    # ---------------------------------------------------------
    # 2. 从文件中构建 “按path顺序排列的点序列”
    # ---------------------------------------------------------
    final_paths_xyz = []

    # (A) Explicit paths already saved
    if "paths" in data:
        print("[INFO] Found explicit 'paths' in file. Visualizing directly (no greedy).")
        paths_obj = take_sample(data["paths"], idx)
        final_paths_xyz = build_paths_from_explicit_paths(paths_obj, args_cli.scale, args_cli.z_offset)

    else:
        if use_pred:
            # (B1) Your current postprocessed exporter output:
            # processed_traj_pred: list(len=B), each element ndarray (T,6)
            # processed_stroke_ids_pred: list(len=B), each element ndarray (T,)
            if "processed_traj_pred" in data and "processed_stroke_ids_pred" in data:
                print("[INFO] Using postprocessed prediction: processed_traj_pred + processed_stroke_ids_pred (no greedy).")
                P = np.asarray(take_sample(data["processed_traj_pred"], idx))          # (T,6)
                S = np.asarray(take_sample(data["processed_stroke_ids_pred"], idx))   # (T,)

                if P.ndim != 2 or P.shape[1] < 3 or S.ndim != 1 or S.shape[0] != P.shape[0]:
                    raise ValueError(
                        f"Unexpected postprocessed shapes: P={P.shape}, S={S.shape}. "
                        f"Expected P=(T,6) and S=(T,)."
                    )

                xyz = P[:, :3].astype(np.float32) * args_cli.scale
                xyz[:, 2] += args_cli.z_offset
                if np.any(xyz < -90):
                    print("[WARN] Found invalid points (< -90). They will still be included, but check your data scale.")
                final_paths_xyz = build_paths_from_pointwise_xyz(xyz, S)

            # (C) Fallback: raw segments (not recommended)
            elif "traj_pred_raw" in data:
                print("[WARN] No postprocessed fields found. Falling back to RAW segments (order may be wrong).")
                traj_flat = np.asarray(take_sample(data["traj_pred_raw"], idx))  # (K,24)
                # Try infer stroke ids from raw masks
                if "pred_stroke_masks_raw" in data:
                    logits = np.asarray(take_sample(data["pred_stroke_masks_raw"], idx))  # (N,K)
                    prob = 1.0 / (1.0 + np.exp(-logits))
                    stroke_ids = np.argmax(prob, axis=0).astype(np.int64)  # (K,)
                else:
                    raise ValueError("Cannot infer stroke ids: missing pred_stroke_masks_raw in file.")

                paths = build_paths_from_segmentwise(
                    traj_flat=traj_flat,
                    stroke_ids=stroke_ids,
                    lambda_pts=args_cli.lambda_points,
                    drop_overlap=args_cli.drop_overlap
                )

                # scale / z_offset
                final_paths_xyz = []
                for p in paths:
                    xyz = p.astype(np.float32) * args_cli.scale
                    xyz[:, 2] += args_cli.z_offset
                    if not np.any(xyz < -90):
                        final_paths_xyz.append(xyz)

            else:
                raise KeyError("No usable prediction keys found in file (need processed_traj_pred or traj_pred_raw).")

        else:
            # GT branch (only if you exported processed_traj_gt / processed_stroke_ids_gt)
            if "processed_traj_gt" in data and "processed_stroke_ids_gt" in data:
                print("[INFO] Using postprocessed GT: processed_traj_gt + processed_stroke_ids_gt (no greedy).")
                P = np.asarray(take_sample(data["processed_traj_gt"], idx))          # (T,6)
                S = np.asarray(take_sample(data["processed_stroke_ids_gt"], idx))   # (T,)

                xyz = P[:, :3].astype(np.float32) * args_cli.scale
                xyz[:, 2] += args_cli.z_offset
                final_paths_xyz = build_paths_from_pointwise_xyz(xyz, S)

            else:
                raise KeyError("GT requested (--use_gt) but processed_traj_gt/processed_stroke_ids_gt not found in file.")

    # Optional: limit count
    if args_cli.max_paths > 0:
        final_paths_xyz = final_paths_xyz[: args_cli.max_paths]

    print(f"[INFO] Final paths: {len(final_paths_xyz)}")
    if final_paths_xyz:
        print(f"[INFO] Path[0] points: {final_paths_xyz[0].shape}")

    # Package for playback
    final_paths = []
    for i, pts in enumerate(final_paths_xyz):
        final_paths.append({
            "points": pts,
            "color_idx": i % len(palette)
        })

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
                    history_pts.append(pt.unsqueeze(0))
                    history_ind.append(torch.tensor([color_id], device=sim.device, dtype=torch.int32))
                    pt_idx += 1
                else:
                    path_idx += 1
                    pt_idx = 0
                    if path_idx < len(final_paths):
                        print(f"[INFO] 路径 {path_idx - 1} 结束，切换颜色 -> 路径 {path_idx}...")

        if history_pts:
            pts_tensor = torch.cat(history_pts)
            ind_tensor = torch.cat(history_ind)
            visualizer.visualize(translations=pts_tensor, marker_indices=ind_tensor)

        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
