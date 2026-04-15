import os
import argparse
from isaaclab.app import AppLauncher

# 强制开启无头模式，不占用显存渲染 UI
parser = argparse.ArgumentParser(description="Batch convert URDF to USD using Isaac Lab native API")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True 

# 启动底层仿真引擎
app_launcher = AppLauncher(args_cli)
app = app_launcher.app

# 必须在 app_launcher 初始化之后导入核心模块
from isaaclab.sim.converters.urdf_converter import UrdfConverter, UrdfConverterCfg

def main():
    # 1. 设定路径
    base_dir = "/home/qihang/code_lab/unitree_rl_lab-main/source/unitree_rl_lab/unitree_rl_lab/assets/obstacles"
    urdf_dir = os.path.join(base_dir, "urdf")
    usd_dir = os.path.join(base_dir, "usd")

    if not os.path.exists(usd_dir):
        os.makedirs(usd_dir)

    # 2. 自动扫描 urdf 文件夹
    if not os.path.exists(urdf_dir):
        print(f"[错误]: 找不到 urdf 文件夹: {urdf_dir}")
        return
        
    files_to_convert = [f for f in os.listdir(urdf_dir) if f.endswith('.urdf')]
    
    if not files_to_convert:
        print("[INFO]: urdf 文件夹中没有找到任何 .urdf 文件。")
        return

    print(f"=== 开始批量转换，共扫描到 {len(files_to_convert)} 个 URDF 文件 ===")

    for filename in files_to_convert:
        input_urdf = os.path.join(urdf_dir, filename)
        output_usd_name = filename.replace(".urdf", ".usd")
        
        print(f"[转换中]: {filename} -> {output_usd_name} ...")
        
        # 3. 使用 Isaac Lab 官方配置类 (修复缺失参数)
        cfg = UrdfConverterCfg(
            asset_path=input_urdf,
            usd_dir=usd_dir,
            usd_file_name=output_usd_name,
            force_usd_conversion=True,
            fix_base=False  # 修复报错 1：告诉系统这把椅子不需要固定底座
        )
        
        # 修复报错 2：填补电机刚度检查。因为椅子没关节，随便填 0 绕过验证
        if hasattr(cfg, "joint_drive") and cfg.joint_drive is not None:
            cfg.joint_drive.gains.stiffness = 0.0
            cfg.joint_drive.gains.damping = 0.0  # 顺手填上阻尼，防患于未然
        
        # 4. 实例化对象，执行转换
        converter = UrdfConverter(cfg)
        
        print(f"[成功]: 已生成 {output_usd_name}")

    print("=== 任务全部完成 ===")
    app.close()

if __name__ == "__main__":
    main()