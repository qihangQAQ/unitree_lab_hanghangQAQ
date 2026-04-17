from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


def goal_time_out(env, command_name: str = "position") -> torch.Tensor:
    """
    基于 CommandTerm.time_left 的超时终止条件。

    默认用 position 命令：
      - 找到 env.command_manager 里名字为 command_name 的 command term
      - 当它的 time_left <= 0 的时候，对应 env 触发 done
    """

    # return env.time_left <= 0.0
    cmd_term = env.command_manager.get_term(command_name)
    return cmd_term.time_left <= 0.0

def path_completed(env, command_name: str) -> torch.Tensor:
    """当手部追踪命令中的路径进度走完时，返回 True 以提前终止 Episode。"""
    return env.command_manager.get_term(command_name).is_completed

def illegal_contact_complex(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """
    当机器人的核心部位（躯干、髋部、上肢）发生碰撞时，触发回合终止。
    """
    # 获取接触传感器
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 提取历史接触力 shape: (num_envs, history_length, num_bodies, 3)
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    
    # 计算力向量的模长 shape: (num_envs, history_length, num_bodies)
    force_mag = torch.norm(net_forces, dim=-1)
    
    # 🔥 修复报错的地方：先把后两个维度展平为一维，再对这个展平后的维度求最大值
    # force_mag.view(force_mag.shape[0], -1) 会变成 (num_envs, history_length * num_bodies)
    max_force = torch.max(force_mag.view(force_mag.shape[0], -1), dim=1)[0]
    
    # 如果最大冲击力超过阈值，返回 True (终止)
    return max_force > threshold