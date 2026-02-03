from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def position_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "reach_pos_target_soft",
) -> torch.Tensor:
    """位置命令的课程学习。

    - 使用 position 命令项；
    - 通过 reach_pos_target_soft 这条奖励的平均值判断是否升难度；
    - 难度通过增大 ranges.pos_1 的上界来体现（目标距离越来越远）。
    """
    # 1) 取出位置命令 term（CommandsCfg 里你叫 position）
    command_term = env.command_manager.get_term("position")
    ranges = command_term.cfg.ranges                # 里面有 pos_1, pos_2当前取值范围
    limit_ranges = command_term.cfg.limit_ranges    #命令限制范围

    # 2) 取出对应奖励项的 cfg + 本回合平均 reward
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    # episode_sums[term_name] 是 [num_envs]，这里对选中的 env_ids 求平均
    reward = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length_s
    )

    # 3) 每个 episode 结束时检查一次是否要升难度
    if env.common_step_counter % env.max_episode_length == 0:
        # 和原版一样：reward > 0.8 * weight 时升一级
        if reward > reward_term.weight * 0.8:
            # ====== X 方向：只增大 max_x，限制在 limit_ranges.pos_1 ======
            cur_min_x, cur_max_x = ranges.pos_1
            lim_min_x, lim_max_x = limit_ranges.pos_1

            delta_x = 0.5  # 每次最远点再远 0.5m
            new_max_x = min(cur_max_x + delta_x, lim_max_x)
            ranges.pos_1 = (cur_min_x, new_max_x)

            # ====== Y 方向：对称扩展到更大的 |y|，限制在 limit_ranges.pos_2 ======
            cur_min_y, cur_max_y = ranges.pos_2
            lim_min_y, lim_max_y = limit_ranges.pos_2

            delta_y = 0.2
            new_min_y = max(cur_min_y - delta_y, lim_min_y)
            new_max_y = min(cur_max_y + delta_y, lim_max_y)
            ranges.pos_2 = (new_min_y, new_max_y)


    # 4) 返回当前“难度级别”，这里用最大前向距离表示
    return torch.tensor(ranges.pos_1[1], device=env.device)

# pos的地形课程设计
def terrain_levels_pos(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """位置任务的地形课程：根据“当前离目标有多近/多远”调整地形难度。

    - 离目标很近（< sigma_tight）: move_up → 地形变难
    - 离目标很远（> sigma_soft）: move_down → 地形变简单
    """

    # 1. 拿到机器人和地形句柄
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # 2. 拿到 position 命令 term（注意这里是 term，不是 command 向量）
    cmd_term = env.command_manager.get_term("position")

    # 3. 当前 base 位置 & 目标位置（世界系 XY）
    base_xy = asset.data.root_pos_w[:, :2]              # (num_envs, 2)
    target_xy = cmd_term.position_targets[:, :2]        # (num_envs, 2)

    # 只对传进来的 env_ids 算距离
    distance = torch.norm(base_xy[env_ids] - target_xy[env_ids], dim=1)

    # 4. 从你的位置奖励超参数里拿 sigma（你现在放在 env.cfg.pos_reward_params 里）
    params = env.cfg.pos_reward_params
    sigma_soft = params.position_target_sigma_soft
    sigma_tight = params.position_target_sigma_tight

    # 5. 课程规则：
    #   - 足够靠近目标（< sigma_tight） → 升级地形
    #   - 离目标明显很远（> sigma_soft）→ 降级地形
    move_up = distance < sigma_tight
    move_down = distance > sigma_soft

    # 调用 IsaacLab 提供的 API 来更新 terrain_levels 和 env_origins
    terrain.update_env_origins(env_ids, move_up, move_down)

    # 返回这些 env 的平均地形等级（和原来的 velocity 版本保持一致）
    return torch.mean(terrain.terrain_levels[env_ids].float())


def hand_tracking_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "ee_pos_tracking",
) -> torch.Tensor:
    """
    喷漆任务的课程学习（只针对喷枪速度）：

    - 初始速度范围: (0.10, 0.10) m/s
    - 每次课程提升: +0.05 m/s (5 cm/s)
    - 最大速度上限: 0.40 m/s
    - 触发条件: 当前 episode 平均 reward > 0.8 * reward_weight
    """

    # 1. 拿到 hand_tracking 命令项
    command_term = env.command_manager.get_term("hand_tracking")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # 2. 当前 episode 的平均奖励（按时间归一）
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    mean_reward = (
        torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids])
        / env.max_episode_length_s
    )

    # 3. 每个 episode 结束时，检查是否升级课程
    if env.common_step_counter % env.max_episode_length == 0:
        if mean_reward > reward_term.weight * 0.8:

            # === 只升级喷枪速度 ===
            step_v = 0.05  # 5 cm/s
            cur_v_min, cur_v_max = ranges.velocity
            lim_v_min, lim_v_max = limit_ranges.velocity

            new_v_max = min(cur_v_max + step_v, lim_v_max)
            ranges.velocity = (cur_v_min, new_v_max)

    # 4. 返回当前课程“难度指标”（用于日志）
    return torch.tensor(ranges.velocity[1], device=env.device)