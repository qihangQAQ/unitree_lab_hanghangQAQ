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


def tracking_curriculum_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "hand_tracking",
) -> torch.Tensor:
    """Three-stage reward-gated sequential curriculum for the spray task.

    Inspired by ULC (Unified Loco-Manipulation Controller):
      Stage 1 "Locomotion":  base velocity + facing wall + fixed standing height.
      Stage 2 "Height":      add trajectory-driven height tracking (squat/stand).
      Stage 3 "Arm":         add arm tracking with progressive reward scaling.

    Each stage gates on running-average reward thresholds.  When all gates pass
    at an episode boundary the curriculum advances to the next stage.
    """

    # ------------------------------------------------------------------
    # 1. 一次性初始化 (first call only)
    # ------------------------------------------------------------------
    if not hasattr(env, "tracking_curriculum_stage"):
        env.tracking_curriculum_stage = 1           # 1-based: 1, 2, 3
        env.tracking_curriculum_last_update = -1
        env.tracking_curriculum_arm_scale = 0.0     # Stage-3 internal progress [0, 1]

    stage = env.tracking_curriculum_stage
    cmd_term = env.command_manager.get_term(command_name)

    # ------------------------------------------------------------------
    # 2. 每步都执行的参数应用 (确保 resume / checkpoint 安全)
    # ------------------------------------------------------------------
    if stage == 1:
        _apply_stage_1(env, cmd_term)
    elif stage == 2:
        _apply_stage_2(env, cmd_term)
    else:  # stage >= 3
        _apply_stage_3(env, cmd_term)

    # ------------------------------------------------------------------
    # 3. Episode 结束时检查 gate，决定是否升级
    # ------------------------------------------------------------------
    if (env.common_step_counter % env.max_episode_length == 0
            and env.common_step_counter != env.tracking_curriculum_last_update):
        env.tracking_curriculum_last_update = env.common_step_counter

        if stage == 1:
            if _check_gates(env, env_ids, {
                "track_base_vel": 0.60,
                "base_face_surface_normal": 0.60,
                "track_body_height": 0.70,
            }):
                env.tracking_curriculum_stage = 2
                # 移除固定高度，改为轨迹驱动
                if hasattr(cmd_term, 'curriculum_fixed_height'):
                    del cmd_term.curriculum_fixed_height
                print(f"[Curriculum] Stage 1 → 2 (Height tracking unlocked) "
                      f"at step {env.common_step_counter}")

        elif stage == 2:
            if _check_gates(env, env_ids, {
                "track_body_height": 0.65,
                "track_base_vel": 0.55,
                "base_face_surface_normal": 0.55,
            }):
                env.tracking_curriculum_stage = 3
                env.tracking_curriculum_arm_scale = 0.0
                print(f"[Curriculum] Stage 2 → 3 (Arm tracking unlocked) "
                      f"at step {env.common_step_counter}")

        elif stage == 3:
            # Stage 3 内部渐进：手臂 reward scale 从 0 → 1
            if env.tracking_curriculum_arm_scale < 1.0:
                if _check_gates(env, env_ids, {
                    "ee_pos_tracking_soft": 0.40,
                    "track_base_vel": 0.45,
                }):
                    env.tracking_curriculum_arm_scale = min(
                        1.0, env.tracking_curriculum_arm_scale + 0.1
                    )
                    print(f"[Curriculum] Stage 3 arm_scale → "
                          f"{env.tracking_curriculum_arm_scale:.1f} "
                          f"at step {env.common_step_counter}")

    return torch.tensor(float(stage), device=env.device)


# =========================================================================
# 辅助函数
# =========================================================================

def _set_weight(env: ManagerBasedRLEnv, name: str, weight: float):
    """Safe reward-weight setter (silently skip missing terms)."""
    try:
        env.reward_manager.get_term_cfg(name).weight = weight
    except KeyError:
        pass


def _check_gates(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    gates: dict[str, float],
) -> bool:
    """Return True when ALL gate rewards exceed their thresholds.

    Each gate value is a *threshold* on the **normalised** running-average
    reward (mean_episode_reward / |weight|).  A reward with weight=0 is
    treated as not-yet-active → gate fails.
    """
    for reward_name, threshold in gates.items():
        if reward_name not in env.reward_manager._episode_sums:
            print(f"  [Curriculum gate] {reward_name}=missing")
            return False
        reward_cfg = env.reward_manager.get_term_cfg(reward_name)
        abs_weight = abs(float(reward_cfg.weight))
        if abs_weight < 1e-9:
            print(f"  [Curriculum gate] {reward_name}=weight_zero")
            return False
        mean_reward = (
            torch.mean(env.reward_manager._episode_sums[reward_name][env_ids])
            / env.max_episode_length_s
        )
        normalized = mean_reward / abs_weight
        if normalized.item() < threshold:
            print(f"  [Curriculum gate] {reward_name}={normalized.item():.3f} "
                  f"(need {threshold:.2f})")
            return False
    return True


# ------------------------------------------------------------------
# Stage apply helpers
# ------------------------------------------------------------------

def _apply_stage_1(env: ManagerBasedRLEnv, cmd_term):
    """Stage 1: 纯侧向行走 — 面向墙壁、保持站立高度、手臂锁默认姿态."""

    # --- 命令参数：高度固定为站立高度 ---
    cmd_term.curriculum_fixed_height = 0.78

    # --- 底盘移动 (主导) ---
    _set_weight(env, "track_base_vel",             2.0)
    _set_weight(env, "base_face_surface_normal",   2.0)
    _set_weight(env, "track_body_height",          1.5)
    _set_weight(env, "com_support",                1.0)

    # --- 手臂：强力锁默认姿态 ---
    _set_weight(env, "joint_deviation_arms",      -2.0)
    _set_weight(env, "joint_deviation_waists",    -1.0)
    _set_weight(env, "joint_deviation_legs",      -0.5)

    # --- 手臂任务 reward (全部关闭) ---
    _set_weight(env, "ee_pos_tracking_soft",       0.0)
    _set_weight(env, "ee_pos_tracking_tight",      0.0)
    _set_weight(env, "ee_rot_tracking",            0.0)
    _set_weight(env, "ee_tangential_speed_tracking", 0.0)
    _set_weight(env, "action_smoothness",          0.0)

    # --- 安全 & 正则化 (保持不变) ---
    _set_weight(env, "feet_stumble",              -1.0)
    _set_weight(env, "feet_too_near",             -1.0)
    _set_weight(env, "feet_slide",               -1.0)
    _set_weight(env, "feet_air_time_variance",   -1.0)
    _set_weight(env, "alive",                      0.15)
    _set_weight(env, "base_linear_velocity",      -0.5)
    _set_weight(env, "base_angular_velocity",     -0.05)
    _set_weight(env, "joint_vel",                 -0.001)
    _set_weight(env, "joint_acc",                 -2.5e-7)
    _set_weight(env, "action_rate",               -0.05)
    _set_weight(env, "dof_pos_limits",            -5.0)
    _set_weight(env, "energy",                    -2e-5)
    _set_weight(env, "flat_orientation_l2",       -3.0)
    _set_weight(env, "undesired_contacts",        -1.0)


def _apply_stage_2(env: ManagerBasedRLEnv, cmd_term):
    """Stage 2: 高度适应 — 轨迹驱动高度、腰/腿协调下蹲站立."""

    # --- 命令参数：移除固定高度，使用轨迹反推 ---
    if hasattr(cmd_term, 'curriculum_fixed_height'):
        del cmd_term.curriculum_fixed_height

    # --- 底盘移动 ---
    _set_weight(env, "track_base_vel",             2.0)
    _set_weight(env, "base_face_surface_normal",   2.0)
    _set_weight(env, "track_body_height",          2.0)  # ← 提高权重
    _set_weight(env, "com_support",                1.0)

    # --- 手臂：仍然锁定 ---
    _set_weight(env, "joint_deviation_arms",      -2.0)
    # 腰部放松一点，允许为高度变化做俯仰
    _set_weight(env, "joint_deviation_waists",    -0.5)  # ← 放松
    # 腿部加强惩罚 (避免乱蹲，用腰来调高度)
    _set_weight(env, "joint_deviation_legs",      -1.5)  # ← 加强

    # --- 手臂任务 (仍关闭) ---
    _set_weight(env, "ee_pos_tracking_soft",       0.0)
    _set_weight(env, "ee_pos_tracking_tight",      0.0)
    _set_weight(env, "ee_rot_tracking",            0.0)
    _set_weight(env, "ee_tangential_speed_tracking", 0.0)
    _set_weight(env, "action_smoothness",          0.0)

    # --- 安全 & 正则化 ---
    _set_weight(env, "feet_stumble",              -1.0)
    _set_weight(env, "feet_too_near",             -1.0)
    _set_weight(env, "feet_slide",               -1.0)
    _set_weight(env, "feet_air_time_variance",   -1.0)
    _set_weight(env, "alive",                      0.15)
    _set_weight(env, "base_linear_velocity",      -0.5)
    _set_weight(env, "base_angular_velocity",     -0.05)
    _set_weight(env, "joint_vel",                 -0.001)
    _set_weight(env, "joint_acc",                 -2.5e-7)
    _set_weight(env, "action_rate",               -0.05)
    _set_weight(env, "dof_pos_limits",            -5.0)
    _set_weight(env, "energy",                    -2e-5)
    _set_weight(env, "flat_orientation_l2",       -3.0)
    _set_weight(env, "undesired_contacts",        -1.0)


def _apply_stage_3(env: ManagerBasedRLEnv, cmd_term):
    """Stage 3: 手臂喷涂 — 全部能力解锁，手臂 reward 渐进放大."""

    # --- 命令参数：轨迹驱动高度 ---
    if hasattr(cmd_term, 'curriculum_fixed_height'):
        del cmd_term.curriculum_fixed_height

    # 手臂渐进因子 [0, 1]
    arm_s = getattr(env, 'tracking_curriculum_arm_scale', 0.0)

    # --- 底盘移动 (逐步降权，让位给手臂) ---
    # arm_s=0 → w=2.0;  arm_s=1 → w=1.0
    w_base = 2.0 - arm_s * 1.0
    _set_weight(env, "track_base_vel",             w_base)
    _set_weight(env, "base_face_surface_normal",   w_base)
    _set_weight(env, "track_body_height",          2.0)
    _set_weight(env, "com_support",                1.0)

    # --- 手臂任务 (渐进打开) ---
    _set_weight(env, "ee_pos_tracking_soft",       0.5 + arm_s * 1.5)   # 0.5 → 2.0
    _set_weight(env, "ee_pos_tracking_tight",      0.0 + arm_s * 1.5)   # 0.0 → 1.5
    _set_weight(env, "ee_rot_tracking",            0.5 + arm_s * 1.5)   # 0.5 → 2.0
    _set_weight(env, "ee_tangential_speed_tracking", 0.0 + arm_s * 1.0) # 0.0 → 1.0
    _set_weight(env, "action_smoothness",         -0.005 - arm_s * 0.005) # -0.005 → -0.01

    # --- 手臂姿态惩罚：右臂解放，左臂保持 ---
    # arm_s=0 → -2.0;  arm_s=1 → -0.2 (只剩左臂的轻微约束)
    w_arm_dev = -2.0 + arm_s * 1.8
    _set_weight(env, "joint_deviation_arms",       w_arm_dev)
    _set_weight(env, "joint_deviation_waists",    -0.5)
    _set_weight(env, "joint_deviation_legs",      -1.5)

    # --- 安全 & 正则化 ---
    _set_weight(env, "feet_stumble",              -1.0)
    _set_weight(env, "feet_too_near",             -1.0)
    _set_weight(env, "feet_slide",               -1.0)
    _set_weight(env, "feet_air_time_variance",   -1.0)
    _set_weight(env, "alive",                      0.15)
    _set_weight(env, "base_linear_velocity",      -0.5)
    _set_weight(env, "base_angular_velocity",     -0.05)
    _set_weight(env, "joint_vel",                 -0.001)
    _set_weight(env, "joint_acc",                 -2.5e-7)
    _set_weight(env, "action_rate",               -0.05)
    _set_weight(env, "dof_pos_limits",            -5.0)
    _set_weight(env, "energy",                    -2e-5)
    _set_weight(env, "flat_orientation_l2",       -3.0)
    _set_weight(env, "undesired_contacts",        -1.0)


# ----------------  根据底盘追踪奖励的表现来放开手臂奖励的比例，而不是单纯调整命令范围。 ----------------
def arm_tracking_reward_curriculum(
        env,
        env_ids,
        trigger_reward_name: str = "base_xy_pos_tracking",
        step_size: float = 0.05,
        trigger_threshold: float = 0.80,   # 新增：固定绝对阈值
):
    if not hasattr(env, "arm_reward_scale"):
        env.arm_reward_scale = 0.0

    mean_reward = (
        torch.mean(env.reward_manager._episode_sums[trigger_reward_name][env_ids])
        / env.max_episode_length_s
    )

    if env.common_step_counter % env.max_episode_length == 0:
        print(
            f"[Curriculum Debug] step={env.common_step_counter}, "
            f"mean_reward={mean_reward.item():.4f}, "
            f"threshold={trigger_threshold:.4f}, "
            f"arm_reward_scale={env.arm_reward_scale:.2f}"
        )

        if mean_reward > trigger_threshold:
            env.arm_reward_scale = min(1.0, env.arm_reward_scale + step_size)
            print(f"[Curriculum] 当前手臂奖励放开比例: {env.arm_reward_scale * 100:.1f}%")

    return torch.tensor(env.arm_reward_scale, device=env.device)

def terrain_levels_new(
    env: "ManagerBasedRLEnv", 
    env_ids: Sequence[int], 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """复刻 humanoid_gym 的地形课程设计，通过相对期望距离来平滑升降级。"""
    
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    
    # 1. 计算实际行走的水平位移
    relative_dist = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - terrain.env_origins[env_ids, :2], dim=1
    )
    
    # 2. 升级逻辑 (Move Up)
    # 保持不变：走过地形方块一半距离（如 4.0m），证明有跨越当前障碍的能力
    move_up = relative_dist > (terrain.cfg.terrain_generator.size[0] * 0.5)
    
    # 3. 降级逻辑 (Move Down) - 【核心改进区】
    # 获取机器人当前的线速度指令向量 [v_x, v_y, w_z]
    command = env.command_manager.get_command("base_velocity")
    cmd_norm = torch.norm(command[env_ids, :2], dim=1) # 计算指令水平速度的大小
    
    # 计算理论期望距离：指令速度 * 整个Episode的时间
    # 例如：指令是 0.5m/s，20s 的回合理论上应该走 10m
    expected_dist = cmd_norm * env.max_episode_length_s
    
    # 新的降级判定：如果实际位移连“理论位移的 50%”都没达到，才判定为失败降级
    # 优势：如果指令速度为 0 (原地站立)，expected_dist 为 0，relative_dist 肯定 >= 0，不会被降级。
    move_down = (relative_dist < expected_dist * 0.5) * ~move_up
    
    # 4. 更新地形等级和环境原点
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    return torch.mean(terrain.terrain_levels[env_ids].float())
