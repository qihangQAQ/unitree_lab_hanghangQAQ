import argparse
from isaaclab.app import AppLauncher

# 1. 配置命令行参数 (保持和 play.py 一样的启动方式)
parser = argparse.ArgumentParser(description="Launch an empty Isaac Sim stage with obstacles.")
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
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    """启动一个带有多种障碍物的 Isaac Sim 场景"""

    # 3. 配置仿真上下文
    # dt=0.01 表示仿真步长 10ms
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 4. 设置场景内容
    # A. 添加主光源 (Dome Light - 天空光)
    cfg_light = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.75, 0.75, 0.75)
    )
    cfg_light.func("/World/SkyLight", cfg_light)

    # B. 添加地面 (Ground Plane)
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # 5. 定义障碍物配置（从 position_env_cfg.py 复制）
    # 注意：去掉了 {ENV_REGEX_NS}，直接使用 /World 路径
    # [种类 1: 复杂镂空结构] - 工作台
    obstacle_table = RigidObjectCfg(
        prim_path="/World/Obstacle_PackingTable",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # [种类 2: 大面积垂直平面] - 柜子
    obstacle_cabinet = RigidObjectCfg(
        prim_path="/World/Obstacle_Cabinet",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.0, 1.8),  # (长, 宽, 高) 模拟一个 1.8米高的大衣柜
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)), # 类似木头的棕色
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.0)),
    )

    # [种类 3: 低矮直角几何] - 单一方块 (DexCube)
    obstacle_block = RigidObjectCfg(
        prim_path="/World/Obstacle_DexCube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 0.0)),
    )

    # [种类 4: 倾斜平面] - 纯正的圆锥 (完美还原原版论文要素)
    obstacle_cone = RigidObjectCfg(
        prim_path="/World/Obstacle_Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.4, height=0.9,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.0, 0.0, 0.0)),
    )

    # [种类 5: 垂直平滑曲面] - 胶囊体 (完美平替 行人 / 高大花瓶)
    obstacle_capsule = RigidObjectCfg(
        prim_path="/World/Obstacle_Capsule",
        spawn=sim_utils.CapsuleCfg(
            radius=0.3, height=1.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 0.0, 0.0)),
    )

    # [种类 6: 全向平滑曲面] - 大球体 (平替 矮胖型花瓶 / 健身球)
    obstacle_sphere = RigidObjectCfg(
        prim_path="/World/Obstacle_Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.45,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 0.0, 0.0)),
    )

    # 6. 重置仿真器
    sim.reset()

    # 7. 生成障碍物
    obstacles = [
        obstacle_table,
        obstacle_cabinet,
        obstacle_block,
        obstacle_cone,
        obstacle_capsule,
        obstacle_sphere,
    ]
    for obstacle in obstacles:
        # 提取我们在 init_state 中设定的位置和旋转
        pos = obstacle.init_state.pos
        rot = obstacle.init_state.rot
        
        # 显式传入 translation 和 orientation 参数
        obstacle.spawn.func(
            obstacle.prim_path, 
            obstacle.spawn,
            translation=pos,   # <-- 关键修复：指定位置
            orientation=rot    # <-- 关键修复：指定旋转
        )
        print(f"[INFO]: Spawned obstacle at {obstacle.prim_path} at position {pos}")

    # 8. 设置相机视角 (让它看着原点)
    sim.set_camera_view(eye=[15.0, 15.0, 15.0], target=[5.0, 0.0, 0.0])

    print("[INFO]: Simulation is running... Press Ctrl+C to stop.")
    print("[INFO]: Obstacles are placed along the x-axis from 0 to 10 meters.")

    # 9. 仿真主循环
    while simulation_app.is_running():
        # 执行一步物理仿真
        sim.step()


if __name__ == "__main__":
    main()
    # 关闭应用
    simulation_app.close()