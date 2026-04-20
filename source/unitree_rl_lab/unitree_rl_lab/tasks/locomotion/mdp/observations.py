from __future__ import annotations

import torch
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sensor_noise_models.depth_noise_model import DepthCameraNoise
from sensor_noise_models.depth_noise_model_cfg import DepthCameraNoiseCfg

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


def _get_depth_noise_model(env: ManagerBasedRLEnv, far_plane: float) -> DepthCameraNoise:
    noise_model = getattr(env, "_depth_camera_noise_model", None)
    cached_far_plane = getattr(env, "_depth_camera_noise_far_plane", None)
    if noise_model is None or cached_far_plane != far_plane:
        noise_cfg = DepthCameraNoiseCfg()
        noise_cfg.far_plane = far_plane
        noise_model = DepthCameraNoise(cfg=noise_cfg, device=env.device)
        env._depth_camera_noise_model = noise_model
        env._depth_camera_noise_far_plane = far_plane
    return noise_model


def depth_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    far_plane: float = 8.0,
    flatten: bool = True,
    use_noise_model: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Return the raw depth image for policy input.

    The image is kept unencoded here. If ``flatten`` is True, the raw depth pixels are
    flattened so recurrent rsl_rl policies can reshape them internally before CNN encoding.
    """

    depth_sensor = env.scene.sensors[sensor_cfg.name]
    raw_depth = depth_sensor.data.output.get("distance_to_image_plane")

    if raw_depth is None:
        depth = torch.full((env.num_envs, 90, 160), far_plane, device=env.device)
    else:
        depth = raw_depth
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        depth = torch.nan_to_num(depth, nan=far_plane, posinf=far_plane, neginf=0.0)
        depth = torch.clamp(depth, 0.0, far_plane)

    if use_noise_model:
        noise_model = _get_depth_noise_model(env, far_plane)
        depth = noise_model(depth.unsqueeze(1)).squeeze(1)

    if normalize:
        depth = depth / far_plane

    if flatten:
        depth = depth.flatten(start_dim=1)

    return depth
