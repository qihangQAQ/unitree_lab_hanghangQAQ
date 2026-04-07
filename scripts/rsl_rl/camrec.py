# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to record Depth Camera images and RayCast labels using a trained RSL-RL agent.
Usage:
    python camrec.py --task G1_Locomotion_Pos --num_envs 9 --load_run <run_folder_name> --headless
"""

import argparse
from importlib.metadata import version
import os
import sys
import pickle
import shutil
import time
from datetime import datetime
import torch
import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args

# 1. 定义参数
parser = argparse.ArgumentParser(description="Record camera and ray data with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=9, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--save_interval", type=int, default=5, help="Save data every N steps.")
parser.add_argument("--max_steps", type=int, default=1000, help="Total steps to record.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras
args_cli.enable_cameras = True

# 2. 启动仿真器 (AppLauncher)
# 必须最先启动，否则后续 import 会报错
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# 3. 导入依赖库 (在仿真启动后)
import gymnasium as gym

# 导入 NP3O 扩展包（猴子补丁注入）
try:
    import unitree_rl_lab.rsl_rl_ext
    print("[INFO] NP3O 扩展包已导入，猴子补丁注入完成")
except ImportError as e:
    print(f"[WARNING] 无法导入 NP3O 扩展包: {e}")

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
import unitree_rl_lab.tasks  # 注册您的自定义任务

# =====================================================================
# 新增：导入深度图噪声模型
# 根据你提供的路径 /home/qihang/code_lab/unitree_rl_lab-main/sensor_noise_models/
# =====================================================================
import sys
import os
# 添加项目根目录到 Python 路径，确保 sensor_noise_models 可导入
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
try:
    from sensor_noise_models.depth_noise_model_cfg import DepthCameraNoiseCfg
    from sensor_noise_models.depth_noise_model import DepthCameraNoise
except ImportError:
    print("[ERROR] Cannot import depth noise model. Please ensure 'sensor_noise_models' is in your PYTHONPATH.")
    sys.exit(1)


def main():
    """Play with RSL-RL agent and record data."""

    # Parse Configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # ---------- A) 数据保存目录 ----------
    rec_root_path = os.path.join("logs", "rsl_rl", "collection_data")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rec_dir = os.path.join(rec_root_path, f"rec_cam_{timestamp}")
    os.makedirs(rec_dir, exist_ok=True)
    print(f"[INFO] Recording data to: {rec_dir}")

    # ---------- B) 模型查找目录 ----------
    ckpt_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    ckpt_root_path = os.path.abspath(ckpt_root_path)

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(ckpt_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Load Environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load Runner
    runner_class_name = getattr(agent_cfg, "class_name", "OnPolicyRunner")
    if runner_class_name == "NP3ORunner":
        try:
            # 首先尝试从 rsl_rl.runners 导入（猴子补丁注入后可用）
            from rsl_rl.runners import NP3ORunner
            runner = NP3ORunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        except (ImportError, NameError):
            # 如果猴子补丁失败，从扩展包直接导入
            from unitree_rl_lab.rsl_rl_ext import NP3ORunner as NP3ORunnerExt
            runner = NP3ORunnerExt(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Data Collection Init
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs, _ = obs

    labels = {}  
    total_steps = args_cli.max_steps
    save_interval = args_cli.save_interval
    unwrapped_env = env.unwrapped

    # =====================================================================
    # 新增：初始化噪声注入器
    # =====================================================================
    noise_cfg = DepthCameraNoiseCfg()
    noise_cfg.far_plane = 6.0  # 与你环境配置的最大探测距离保持一致
    depth_noise_injector = DepthCameraNoise(cfg=noise_cfg, device=env.unwrapped.device)
    print(f"[INFO] Depth Camera Noise Model initialized.")

    files_saved_count = 0

    print(f"\n{'=' * 60}")
    print(f"[START] Starting REAL-TIME recording loop...")
    print(f"{'=' * 60}\n")

    with torch.inference_mode():
        for i in range(total_steps):
            # 1. Inference & Step
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # 2. Sampling & Saving
            if i % save_interval == 0:
                try:
                    # A. 获取完美的深度数据
                    depth_data = unwrapped_env.scene.sensors["depth_camera"].data.output["distance_to_image_plane"]
                    
                    # 预处理：将无穷大(Inf)或无效值替换为最大探测距离，防止噪声模型报错
                    depth_data[depth_data == float('inf')] = noise_cfg.far_plane
                    depth_data[torch.isnan(depth_data)] = noise_cfg.far_plane

                    # 获取射线标签
                    pos_cmd_term = unwrapped_env.command_manager.get_term("position")
                    ray_data = pos_cmd_term.ray_obs

                    # =========================================================
                    # B. 注入真实噪声 (核心修改)
                    # =========================================================
                    # 确保维度是 (Batch, Channel, Height, Width) -> (num_envs, 1, 90, 160)
                    if depth_data.ndim == 3:
                        depth_data = depth_data.unsqueeze(1)
                    
                    # 调用 ETHZ 噪声模型
                    noisy_depth_data = depth_noise_injector(depth_data)

                    # 转 Numpy (去掉 Channel 维度，恢复成 N, H, W)
                    cam_data_np = noisy_depth_data.squeeze(1).cpu().numpy()
                    ray_label_np = ray_data.cpu().numpy()

                    # C. 实时写入硬盘
                    for robot_idx in range(env.num_envs):
                        save_name = f'robot_{robot_idx}_step{i}'

                        # 存标签到字典
                        labels[save_name] = ray_label_np[robot_idx]

                        # 存带有真实噪声的图片到硬盘
                        save_path = os.path.join(rec_dir, save_name + '.npy')
                        np.save(save_path, cam_data_np[robot_idx])

                        files_saved_count += 1

                    print(f"  >>> [SAVED] Step {i:04d}/{total_steps} | Saved {args_cli.num_envs} noisy images | Total Files: {files_saved_count}")

                except KeyError:
                    print(f"[ERROR] Step {i}: Sensors data missing.")
                    break
                except Exception as e:
                    print(f"[ERROR] Step {i}: Save failed: {e}")
                    break

    # 3. 循环结束，保存标签字典
    print(f"\n{'=' * 60}")
    print(f"[FINISH] Loop finished. Saving labels...")

    label_path = os.path.join(rec_dir, 'label.pkl')
    with open(label_path, 'wb') as f:
        pickle.dump(labels, f)

    print(f"[SUCCESS] All done.")
    print(f"{'=' * 60}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()