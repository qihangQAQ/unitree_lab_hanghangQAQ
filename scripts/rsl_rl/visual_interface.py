import argparse
from isaaclab.app import AppLauncher

# 1. 配置命令行参数 (保持和 play.py 一样的启动方式)
parser = argparse.ArgumentParser(description="Launch an empty Isaac Sim stage.")
# 添加 Isaac Lab 标准启动参数 (如 --headless, --livestream 等)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动 Isaac Sim (这步必须最先执行)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 必须在 simulation_app 启动后才能导入 Isaac Lab 的其他模块
# ---------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import AssetBaseCfg


def main():
    """启动一个空的 Isaac Sim 场景"""

    # 3. 配置仿真上下文
    # dt=0.01 表示仿真步长 10ms
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 4. 设置场景内容 (可选)
    # 如果完全不加下面这两行，场景将是一片漆黑且没有重力基准

    # A. 添加主光源 (Dome Light - 天空光)
    cfg_light = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.75, 0.75, 0.75)
    )
    cfg_light.func("/World/SkyLight", cfg_light)

    # B. 添加地面 (Ground Plane)
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # 5. 重置仿真器
    sim.reset()

    # 6. 设置相机视角 (让它看着原点)
    sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])

    print("[INFO]: Simulation is running... Press Ctrl+C to stop.")

    # 7. 仿真主循环
    while simulation_app.is_running():
        # 执行一步物理仿真
        sim.step()


if __name__ == "__main__":
    main()
    # 关闭应用
    simulation_app.close()