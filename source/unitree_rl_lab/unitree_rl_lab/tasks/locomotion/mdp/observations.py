from __future__ import annotations

import torch
from typing import TYPE_CHECKING

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
