import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# 1) CLI：保持和 play.py 一致的启动方式
parser = argparse.ArgumentParser(description="Visualize a random surface spray path in Isaac Sim.")
AppLauncher.add_app_launcher_args(parser)

# 你可以手动调这些参数
parser.add_argument("--num_points", type=int, default=800, help="Number of points to visualize")
parser.add_argument("--dt_point", type=float, default=0.02, help="Seconds per point (visualization speed)")
parser.add_argument("--radius", type=float, default=1.2, help="Base radius of the surface (sphere patch)")
parser.add_argument("--patch_deg", type=float, default=70.0, help="Angular size of the patch in degrees")
parser.add_argument("--noise", type=float, default=0.0, help="Add small positional noise")
parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 means random)")

args_cli = parser.parse_args()

# 2) 启动 Isaac Sim（必须最先）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3) 之后才能 import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def snake_on_sphere_patch(num_points: int,
                          radius: float,
                          patch_deg: float,
                          rng: np.random.Generator):
    """
    Generate a snake-like raster path on a random sphere patch.
    Returns:
      xyz: (T,3) points on surface
      nrm: (T,3) outward normals
    """
    # Randomly choose patch center direction (unit vector)
    # sample a random unit vector
    v = rng.normal(size=3)
    v = v / (np.linalg.norm(v) + 1e-12)

    # Build an orthonormal basis around v: (e1, e2) tangent axes
    # pick something not parallel
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(v, a)
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(v, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)

    # Patch angular extent
    patch_rad = np.deg2rad(patch_deg)
    # We'll define local patch coords (u,w) in [-patch_rad/2, patch_rad/2]
    # Generate a snake/raster in (u,w)
    # Choose number of stripes ~ sqrt(num_points)
    stripes = int(np.sqrt(num_points / 4))  # heuristic
    stripes = max(6, stripes)
    pts_per_stripe = max(10, num_points // stripes)

    us = np.linspace(-patch_rad / 2, patch_rad / 2, pts_per_stripe, dtype=np.float64)
    ws = np.linspace(-patch_rad / 2, patch_rad / 2, stripes, dtype=np.float64)

    points = []
    normals = []

    # Map local (u,w) to a point on the sphere patch using exponential map approx:
    # direction = normalize( v + tan(u)*e1 + tan(w)*e2 )
    # For small angles, tan(theta) ~ theta, good enough for visualization/training generator.
    for i, w in enumerate(ws):
        u_seq = us if (i % 2 == 0) else us[::-1]  # snake direction
        for u in u_seq:
            d = v + np.tan(u) * e1 + np.tan(w) * e2
            d = d / (np.linalg.norm(d) + 1e-12)
            p = radius * d
            n = d  # sphere normal is radial
            points.append(p)
            normals.append(n)

    xyz = np.asarray(points, dtype=np.float32)
    nrm = np.asarray(normals, dtype=np.float32)

    # Trim or pad to num_points
    if xyz.shape[0] > num_points:
        xyz = xyz[:num_points]
        nrm = nrm[:num_points]
    return xyz, nrm


def main():
    # Random seed
    if args_cli.seed >= 0:
        rng = np.random.default_rng(args_cli.seed)
    else:
        rng = np.random.default_rng()

    # 4) 仿真上下文
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 5) 场景：灯光 + 地面
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/SkyLight", cfg_light)

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # 6) 可视化 marker（多颜色轮换）
    palette = [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0),
        (1.0, 0.5, 0.0), (0.5, 0.0, 0.5),
        (0.7, 0.7, 0.7), (0.2, 0.6, 0.9)
    ]
    markers = {}
    for i, c in enumerate(palette):
        markers[f"m_{i}"] = sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c)
        )

    marker_cfg = VisualizationMarkersCfg(prim_path="/World/PathViz", markers=markers)
    visualizer = VisualizationMarkers(marker_cfg)

    # 7) 生成一条随机曲面轨迹
    xyz, nrm = snake_on_sphere_patch(
        num_points=args_cli.num_points,
        radius=args_cli.radius,
        patch_deg=args_cli.patch_deg,
        rng=rng
    )

    if args_cli.noise > 0.0:
        xyz = xyz + rng.normal(scale=args_cli.noise, size=xyz.shape).astype(np.float32)

    # 8) reset + 相机
    sim.reset()
    center = xyz.mean(axis=0)
    sim.set_camera_view(
        eye=(center + np.array([2.0, 2.0, 2.0])).tolist(),
        target=center.tolist()
    )

    print("[INFO] Simulation running. Visualizing points one-by-one. Ctrl+C to stop.")
    print(f"[INFO] points: {xyz.shape[0]}, radius={args_cli.radius}, patch_deg={args_cli.patch_deg}")

    # 9) 主循环：一个点一个点蹦出来
    t_acc = 0.0
    idx = 0

    # 历史点缓存（每帧把所有历史点都画出来）
    hist_pts = []
    hist_ids = []

    # 给每条“stripe”换颜色：简单按 index 分段
    # 让你更直观地看出蛇形 raster
    color_stride = max(1, args_cli.num_points // 10)

    while simulation_app.is_running():
        dt = sim.get_physics_dt()
        t_acc += dt

        if t_acc >= args_cli.dt_point and idx < xyz.shape[0]:
            t_acc = 0.0

            p = xyz[idx]
            # 颜色轮换
            color_id = (idx // color_stride) % len(palette)

            hist_pts.append(torch.tensor(p, device=sim.device, dtype=torch.float).unsqueeze(0))
            hist_ids.append(torch.tensor([color_id], device=sim.device, dtype=torch.int32))

            idx += 1

        if hist_pts:
            pts_tensor = torch.cat(hist_pts, dim=0)
            ids_tensor = torch.cat(hist_ids, dim=0)
            visualizer.visualize(translations=pts_tensor, marker_indices=ids_tensor)

        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
