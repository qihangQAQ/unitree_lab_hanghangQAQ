import argparse
from isaaclab.app import AppLauncher

# 1. 配置命令行参数
parser = argparse.ArgumentParser(description="Launch an Isaac Sim stage with local USD obstacles.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动 Isaac Sim (必须最先执行)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 必须在 simulation_app 启动后才能导入 Isaac Lab 的其他模块
# ---------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import RigidObjectCfg


def main():
    """启动一个纯本地加载的无网环境"""

    # 3. 配置仿真上下文
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 4. 设置场景内容：灯光与地面
    cfg_light = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.75, 0.75, 0.75)
    )
    cfg_light.func("/World/SkyLight", cfg_light)

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # ==========================================
    # 核心修改：指定你刚才生成 USD 文件的绝对路径
    # ==========================================
    LOCAL_USD_DIR = "/home/qihang/code_lab/unitree_rl_lab-main/source/unitree_rl_lab/unitree_rl_lab/assets/obstacles/usd"

# [1] 办公椅 (底部最低点在 -0.5m，所以 Z 轴抬高 0.5)
    obstacle_office_chair = RigidObjectCfg(
        prim_path="/World/Obstacle_OfficeChair",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_USD_DIR}/OfficeChair.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)), # <--- 修改了 Z
    )

    # [2] 餐椅 (底部最低点在 -0.5m，Z 轴抬高 0.5)
    obstacle_dining_chair = RigidObjectCfg(
        prim_path="/World/Obstacle_DiningChair",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_USD_DIR}/DiningChair.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.5)), # <--- 修改了 Z
    )

    # [3] 圆柱 (高 1.5，中心在原点，Z 轴抬高 0.75)
    obstacle_cylinder = RigidObjectCfg(
        prim_path="/World/Obstacle_Cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.35, height=1.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 0.75)), # <--- 修改了 Z
    )

    # [4] 行人替身 (胶囊体高 1.6，加上两端半球半径 0.3*2，总长 2.2，Z 轴抬高 1.1)
    obstacle_human_proxy = RigidObjectCfg(
        prim_path="/World/Obstacle_HumanProxy",
        spawn=sim_utils.CapsuleCfg(
            radius=0.3, height=1.6,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.3, 0.8)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.0, 0.0, 1.1)), # <--- 修改了 Z
    )

    # [5] 雪糕筒 (高 0.8，Z 轴抬高 0.4)
    obstacle_cone = RigidObjectCfg(
        prim_path="/World/Obstacle_Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.3, height=0.8,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.4, 0.1)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(8.0, 0.0, 0.4)), # <--- 修改了 Z
    )

    # 6. 重置仿真器
    sim.reset()

    # 7. 生成障碍物
    obstacles = [
        obstacle_office_chair,
        obstacle_dining_chair,
        obstacle_cylinder,
        obstacle_human_proxy,
        obstacle_cone
    ]
    
    for obstacle in obstacles:
        pos = obstacle.init_state.pos
        rot = obstacle.init_state.rot
        
        # 显式传入 translation 和 orientation 参数生成物体
        obstacle.spawn.func(
            obstacle.prim_path, 
            obstacle.spawn,
            translation=pos,
            orientation=rot
        )
        print(f"[INFO]: 成功生成障碍物 {obstacle.prim_path} 于坐标 {pos}")

    # 8. 设置相机视角
    sim.set_camera_view(eye=[10.0, 10.0, 8.0], target=[4.0, 0.0, 0.0])

    print("[INFO]: 正在运行本地 USD 环境，加载速度极快且无需网络连接！")

    # 9. 仿真主循环
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()