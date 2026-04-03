from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase

# 时间获取函数
def command_time_left(env: ManagerBasedRLEnv,command_name: str = "position") -> torch.Tensor:
    # 取出这个命令 term（比如 UniformPositionCommand）
    term = env.command_manager.get_term(command_name)
    # term.time_left: (num_envs,)
    time_left = term.time_left
    max_time = getattr(term.cfg, "max_goal_time_s", term.cfg.resampling_time_range[1])
    max_time = float(max_time) if max_time > 0.0 else 1.0
    return (time_left / max_time).unsqueeze(1)

def ray2d_distances(env: ManagerBasedRLEnv, command_name: str = "position") -> torch.Tensor:
    """机器人基座发出的 2D 射线距离观测。

    该函数从指定的 CommandTerm 中提取计算好的射线距离（ray_obs）。
    通常用于避障任务。
    """
    # 1. 获取包含射线逻辑的命令管理器 term
    term = env.command_manager.get_term(command_name)

    # 2. 获取我们在 UniformPositionCommand 中定义的 ray_obs (shape: [num_envs, 11])
    # 注意：这里需要确保你的 UniformPositionCommand 类里有 ray_obs 这个属性
    ray_obs = term.ray_obs

    # 3. 仿照 legged_robot_pos 的归一化/缩放逻辑（可选）
    # 如果你在配置里设置了 log2 缩放，可以在这里处理
    # return torch.log2(ray_obs)

    return ray_obs


def base_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回机器人基座的高度（Z坐标）。

    用于观测基座距离地面的高度。

    Returns:
        torch.Tensor: 基座高度，形状为 (num_envs, 1)
    """
    asset = env.scene["robot"]
    # root_pos_w: (num_envs, 3) [x, y, z]
    # 返回 z 坐标，保持二维形状 (num_envs, 1)
    return asset.data.root_pos_w[:, 2:3]


def feet_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回左右脚足底的接触力（每个脚3维力向量，共6维）。

    从接触传感器中提取指定身体（通常为左右脚踝）的 net_forces_w（世界坐标系下的力）。

    Args:
        env: 环境实例
        sensor_cfg: 传感器配置，指定传感器名称和身体名称

    Returns:
        torch.Tensor: 接触力，形状为 (num_envs, 6)，其中前3维为左脚，后3维为右脚。
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w 形状: (num_envs, num_bodies, 3)
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    # 假设 sensor_cfg.body_ids 按顺序包含左脚和右脚的身体ID
    # 展平最后两个维度: (num_envs, num_bodies * 3)
    batch_size = forces.shape[0]
    num_bodies = forces.shape[1]
    flattened = forces.reshape(batch_size, num_bodies * 3)
    return flattened

def height_scan_hpc(
    env: ManagerBasedEnv, 
    sensor_cfg: SceneEntityCfg, 
    offset: float = 0.78  # 默认值改为 G1 的合理站立高度
) -> torch.Tensor:
    """
    获取基于传感器坐标系的高度扫描图，并包含 NaN/Inf 工业级清洗。
    """
    # 提取传感器
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    
    # 1. 原始物理计算：地形相对高度 = 传感器Z - 击中点Z - 目标离地偏移量
    # 结果含义：0.0 代表完美平地，正数代表脚下有坑（击中点低），负数代表踩到台阶（击中点高）
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    # 2. 🔥 核心防护：清洗 NaN 和 Inf (防御物理引擎异常或射线射穿地图)
    # nan=0.0: 如果没测到，保守假设脚下是平地
    # posinf/neginf: 限制在一个物理上不可能达到的极限值，后续会被 config 中的 clip 截断
    heights = torch.nan_to_num(heights, nan=0.0, posinf=10.0, neginf=-10.0)

    return heights