from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""

# quat → yaw
def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """从 (w, x, y, z) 四元数提取 yaw，返回 shape (N,)"""
    # quat: (..., 4)
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    # 标准 ZYX 欧拉角里的 yaw 公式
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi

def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """q: (..., 4) world quaternion. Returns yaw in radians."""
    # Isaac 系里常见两种顺序：wxyz 或 xyzw
    # 这里做一个“尽量稳”的判断：哪个分量更像 w（接近±1）就用哪个当 w
    q0, q1, q2, q3 = q.unbind(-1)  # four components

    # 假设 wxyz: w=q0；假设 xyzw: w=q3
    use_wxyz = q0.abs().mean() > q3.abs().mean()

    w = torch.where(use_wxyz, q0, q3)
    x = torch.where(use_wxyz, q1, q0)
    y = torch.where(use_wxyz, q2, q1)
    z = torch.where(use_wxyz, q3, q2)

    # yaw (Z axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


# 位置命令奖励

# 新增0.command_duration_mask（掩码计算）
def _command_duration_mask(
    env: "ManagerBasedRLEnv",
    command_name: str,
    duration: float,
) -> torch.Tensor:
    """基于 position_command.command_time_left 做一个时间窗 mask。

    当 command_time_left <= duration 时置 1，否则 0，然后除以 duration 做简单归一化。
    """
    cmd_term = env.command_manager.get_term(command_name)
    # (num_envs,)
    time_left = cmd_term.time_left

    mask = time_left <= duration  # bool
    # 防止除 0
    duration_safe = max(float(duration), 1e-6)
    return mask.float() / duration_safe

# 新增1.reach_pos_target_soft（软位置追踪）
def reach_pos_target_soft(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """软距离奖励: 1 / (1 + (d / sigma_soft)^2)，再乘时间窗 mask。

    使用 cfg.rewards.position_target_sigma_soft 和 cfg.rewards.rew_duration。
    """

    asset: Articulation = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)
    # cfg = env.cfg.rewards
    params = env.cfg.pos_reward_params

    # 目标位置和当前 base 位置（世界系 XY）
    target_xy = cmd_term.position_targets[:, :2]
    base_xy = asset.data.root_pos_w[:, :2]
    distance = torch.norm(target_xy - base_xy, dim=1)

    sigma = params.position_target_sigma_soft
    base_rew = 1.0 / (1.0 + torch.square(distance / sigma))

    # return base_rew * _command_duration_mask(env, command_name, params.rew_duration)
    return base_rew

# 新增2.reach_pos_target_tight（硬位置追踪）
def reach_pos_target_tight(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """更严格的位置奖励：使用更小的 sigma_tight、时间窗更短。"""

    asset: Articulation = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)
    params = env.cfg.pos_reward_params


    target_xy = cmd_term.position_targets[:, :2]
    base_xy = asset.data.root_pos_w[:, :2]
    distance = torch.norm(target_xy - base_xy, dim=1)

    sigma_tight = params.position_target_sigma_tight
    base_rew = 1.0 / (1.0 + torch.square(distance / sigma_tight))

    return base_rew * _command_duration_mask(env, command_name, params.rew_duration / 2.0)
    # return base_rew

# 新增3.reach_heading_target（朝向追踪）
def reach_heading_target(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    """在接近目标位置后，根据 heading 误差给奖励。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)
    params = env.cfg.pos_reward_params

    target_xy = cmd_term.position_targets[:, :2]
    base_xy = asset.data.root_pos_w[:, :2]
    distance = torch.norm(target_xy - base_xy, dim=1)

    # 只有进入 soft 区域之后才看朝向
    near_goal = (distance < params.position_target_sigma_soft).float()

    # 机器人当前 yaw
    base_yaw = _quat_to_yaw(asset.data.root_quat_w)
    # 目标 yaw（在 PositionCommand 中维护）
    target_heading = cmd_term.heading_target  # (num_envs,)

    angle_difference = torch.abs(wrap_to_pi(base_yaw - target_heading))
    heading_rew = 1.0 / (1.0 + torch.square(angle_difference / params.heading_target_sigma))

    return heading_rew * near_goal * _command_duration_mask(env, command_name, params.rew_duration)


# 新增4.velo_dir（沿目标方向运动）
def velo_dir(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    """沿着目标方向移动的奖励。
        远离目标时：奖励“沿目标方向的前向速度”；接近目标时给常数 1。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)
    # cfg = env.cfg.rewards
    params = env.cfg.pos_reward_params

    target_xy = cmd_term.position_targets[:, :2]
    base_xy = asset.data.root_pos_w[:, :2]
    diff = target_xy - base_xy
    distance = torch.norm(diff, dim=1)

    # 目标方向单位向量
    dir_to_goal = diff / (torch.norm(diff, dim=1, keepdim=True) + 1e-6)

    # 当前前向方向（xy），由 yaw 得到
    base_yaw = _quat_to_yaw(asset.data.root_quat_w)
    forward_xy = torch.stack([torch.cos(base_yaw), torch.sin(base_yaw)], dim=1)

    # 正前向速度（body x）
    lin_vel_b = asset.data.root_lin_vel_b
    forward_speed = lin_vel_b[:, 0].clamp(min=0.0)

    # 与目标方向的夹角余弦
    cos_angle = (forward_xy * dir_to_goal).sum(dim=1)
    good_dir = cos_angle > -0.25  # 和参考项目同样的阈值

    sigma_tight = params.position_target_sigma_tight
    far = (distance > sigma_tight).float()
    near = (distance <= sigma_tight).float()

    return forward_speed * good_dir.float() * far / 4.5 + 1.0 * near


# 新增5.stand_still_pos（到点后的站姿）
def stand_still_pos(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """到点之后站姿偏离默认姿态的“偏差量”。
       在 cfg 中配置负的 weight，将这个偏差当成惩罚。
       """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)
    params = env.cfg.pos_reward_params

    target_xy = cmd_term.position_targets[:, :2]
    base_xy = asset.data.root_pos_w[:, :2]
    distance = torch.norm(target_xy - base_xy, dim=1)

    dof_pos = asset.data.joint_pos
    default_dof_pos = asset.data.default_joint_pos

    # stand_bias = torch.zeros_like(dof_pos)
    # # 和参考项目保持同样的偏置模式（微微弯膝、下压小腿）
    # stand_bias[:, 1::3] += 0.2
    # stand_bias[:, 2::3] -= 0.3

    # deviation = torch.sum(torch.abs(dof_pos - default_dof_pos - stand_bias), dim=1)
    deviation = torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1)

    mask = _command_duration_mask(env, command_name, params.rew_duration / 2.0)
    near_tight = (distance < params.position_target_sigma_tight).float()

    return deviation * mask * near_tight


# 新增6.nomove（远离目标时发呆惩罚）
def nomove(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """远离目标时，速度很小而且面朝反方向的惩罚（在 cfg 中配负权重）。

    对标 legged_robot_pos._reward_nomove。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)
    # cfg = env.cfg.rewards
    params = env.cfg.pos_reward_params

    target_xy = cmd_term.position_targets[:, :2]
    base_xy = asset.data.root_pos_w[:, :2]
    diff = target_xy - base_xy
    distance = torch.norm(diff, dim=1)

    # 线速度、角速度（body frame）
    lin_vel_b = asset.data.root_lin_vel_b
    ang_vel_b = asset.data.root_ang_vel_b

    static = torch.logical_and(
        torch.norm(lin_vel_b[:, :2], dim=-1) < 0.1,
        torch.abs(ang_vel_b[:, 2]) < 0.1,
    )

    # 目标方向单位向量
    dir_to_goal = diff / (torch.norm(diff, dim=1, keepdim=True) + 1e-6)

    # 当前前向方向（xy）
    base_yaw = _quat_to_yaw(asset.data.root_quat_w)
    forward_xy = torch.stack([torch.cos(base_yaw), torch.sin(base_yaw)], dim=1)

    cos_angle = (forward_xy * dir_to_goal).sum(dim=1)
    bad_dir = cos_angle < -0.25  # 朝向明显背对目标

    far = (distance > params.position_target_sigma_soft).float()

    return static.float() * bad_dir.float() * far


# 新增7.termination（接近目标时摔倒惩罚加倍）
def termination(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)
    params = env.cfg.pos_reward_params

    target_xy = cmd_term.position_targets[:, :2]
    base_xy = asset.data.root_pos_w[:, :2]
    distance = torch.norm(target_xy - base_xy, dim=1)

    terminated_not_timeout = env.reset_buf & (~env.reset_time_outs)

    mask = _command_duration_mask(env, command_name, params.rew_duration / 2.0)
    near_goal = (distance < params.position_target_sigma_tight).float()
    return terminated_not_timeout.float() * (1.0 + 4.0 * mask * near_goal)


# 新增reward(行进过程朝向奖励，防止机器人背对期望位置走过去)
def face_target_while_moving(
    env: "ManagerBasedRLEnv",
    command_name: str = "position",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_thresh: float = 0.4,     # 需要走路的阈值（米）
    stop_radius: float = 0.15,    # 靠近目标就不管它（米）
    sigma: float = 0.6,           # 朝向误差尺度
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)

    # 目标方向（世界系）
    target_xy = cmd_term.position_targets[:, :2]
    base_xy = asset.data.root_pos_w[:, :2]
    vec = target_xy - base_xy
    distance = torch.norm(vec, dim=1).clamp(min=1e-6)

    # 期望朝向 = 指向目标的方向
    desired_yaw = torch.atan2(vec[:, 1], vec[:, 0])

    # 当前 yaw（世界系）
    # yaw = asset.data.root_yaw_w  # 如果你没有这个字段，用 root_quat_w 转 yaw
    yaw = quat_to_yaw(asset.data.root_quat_w)

    yaw_err = wrap_to_pi(desired_yaw - yaw)

    # 只在“需要走路且还没到点”时启用
    move_mask = (distance > move_thresh).float()
    not_close = (distance > stop_radius).float()

    return torch.exp(-(yaw_err / sigma) ** 2) * move_mask * not_close

# 新增
def feet_gait_pos(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str = "position",
    move_thresh: float = 0.1,
) -> torch.Tensor:
    
    # 1) 读足底接触信息
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # (num_envs, num_legs) -> bool：True 表示该腿当前有接触
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    # 2) 根据 episode 内时间生成全局相位（0~1）
    #    t = episode_step * dt
    global_phase = (
        (env.episode_length_buf * env.step_dt) % period / period
    ).unsqueeze(1)  # (num_envs, 1)

    # 3) 为每条腿加 offset，得到单腿相位
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0  # [0, 1)
        phases.append(phase)
    # (num_envs, num_legs)
    leg_phase = torch.cat(phases, dim=-1)

    # 4) 期望 stance/swing 与真实接触的一致性奖励
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    num_legs = len(sensor_cfg.body_ids)
    for i in range(num_legs):
        # 相位 < threshold 认为应在 stance，相位 >= threshold 认为应在 swing
        is_stance = leg_phase[:, i] < threshold  # (num_envs,)
        # 当 is_stance 与 is_contact 相同（都 True 或都 False）时记为正确
        correct = ~(is_stance ^ is_contact[:, i])
        reward += correct.float()

    # 5) 基于位置命令的 gating：只有“需要行走”时才启用该奖励
    if command_name is not None:
        # CommandManager 中的位置命令，一般为 (dx_body, dy_body, heading_err)
        cmd = env.command_manager.get_command(command_name)
        # 只看平面位置误差（米）
        pos_err_xy = cmd[:, :2]
        cmd_norm = torch.norm(pos_err_xy, dim=1)  # (num_envs,)

        # 误差大于 move_thresh 才认为在“行走阶段”
        move_mask = cmd_norm > 0.15
        reward *= move_mask.float()

    return reward


"""
Other rewards.
"""

# 惩罚对称关节位置差异，鼓励对称运动
def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward

# 新增position--task-rewards
# 新增0.command_duration_mask（掩码计算）
def _command_duration_mask(
    env: "ManagerBasedRLEnv",
    command_name: str,
    duration: float,
) -> torch.Tensor:
    """基于 position_command.command_time_left 做一个时间窗 mask。

    当 command_time_left <= duration 时置 1，否则 0，然后除以 duration 做简单归一化。
    """
    cmd_term = env.command_manager.get_term(command_name)
    # (num_envs,)
    time_left = cmd_term.time_left

    mask = time_left <= duration  # bool
    # 防止除 0
    duration_safe = max(float(duration), 1e-6)
    return mask.float() / duration_safe

# 新增1.reach_pos_target_soft（软位置追踪）
def reach_pos_target_soft(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """软距离奖励: 1 / (1 + (d / sigma_soft)^2)，再乘时间窗 mask。

    使用 cfg.rewards.position_target_sigma_soft 和 cfg.rewards.rew_duration。
    """

    asset: Articulation = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)
    # cfg = env.cfg.rewards
    params = env.cfg.pos_reward_params

    # 目标位置和当前 base 位置（世界系 XY）
    target_xy = cmd_term.position_targets[:, :2]
    base_xy = asset.data.root_pos_w[:, :2]
    distance = torch.norm(target_xy - base_xy, dim=1)

    sigma = params.position_target_sigma_soft
    base_rew = 1.0 / (1.0 + torch.square(distance / sigma))

    # return base_rew * _command_duration_mask(env, command_name, params.rew_duration)
    return base_rew

def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def foot_clearance_reward_position(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str = "position",
    stop_radius: float = 0.10,      # 10cm 内关闭
    fade_band: float = 0.05,        # 软过渡带宽（可设 0 变成硬阈值）
) -> torch.Tensor:
    """Reward the feet for clearing a specified height off the ground.
    Disabled when close to the position target (e.g. within 10 cm).
    """
    asset = env.scene[asset_cfg.name]

    # 原 clearance shaping（你原封不动）
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    per_foot = foot_z_target_error * foot_velocity_tanh
    base_reward = torch.exp(-torch.sum(per_foot, dim=1) / std)  # (num_envs,)

    # --- 距离 gate：到目标 10cm 以内关闭 ---
    cmd_term = env.command_manager.get_term(command_name)
    target_xy = cmd_term.position_targets[:, :2]
    base_xy = env.scene["robot"].data.root_pos_w[:, :2]
    distance = torch.norm(target_xy - base_xy, dim=1)  # (num_envs,)

    if fade_band is None or fade_band <= 0.0:
        # 硬阈值：<= stop_radius 直接 0
        gate = (distance > stop_radius).float()
    else:
        # 软过渡：distance <= stop_radius -> 0
        # distance >= stop_radius + fade_band -> 1
        gate = torch.clamp((distance - stop_radius) / fade_band, 0.0, 1.0)

    return base_reward * gate

def obstacle_collision(
        env: ManagerBasedRLEnv,
        threshold: float = 1.0,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        command_name: str = "position",
) -> torch.Tensor:
    """针对障碍物的碰撞惩罚，仿照 legged_robot_pos 逻辑实现。

    该奖励检测机器人非脚部部位的接触力，并在靠近目标点时显著放大惩罚，
    迫使机器人为了获得高额的任务奖励而必须学会避障。
    """
    # 1. 获取接触传感器数据
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 2. 检测指定部位（通常是大腿、躯干等）的碰撞力是否超过阈值
    # shape: (num_envs, history, body_ids, 3) -> 取历史记录中的最大力
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 计算力向量的模，并在历史维度和身体部位维度上寻找是否存在超过 threshold 的力
    collision_detected = torch.max(
        torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1
    )[0] > threshold

    # 转换为 0/1 浮点数，并求和（统计当前有多少个设定的部位发生了碰撞）
    reward = torch.sum(collision_detected, dim=1).float()

    # 3. 增强逻辑：如果距离目标点很近（进入了 tight 区域）发生碰撞，惩罚翻倍
    # 参考了你 rewards.py 中 _reward_termination 的逻辑
    cmd_term = env.command_manager.get_term(command_name)
    params = env.cfg.pos_reward_params

    distance = torch.norm(cmd_term.position_targets[:, :2] - env.scene["robot"].data.root_pos_w[:, :2], dim=1)
    near_goal = (distance < params.position_target_sigma_tight).float()

    # 靠近目标时碰撞惩罚增加（例如 5 倍），防止机器人在最后一刻为了冲向目标而撞击障碍物
    return reward * (1.0 + 4.0 * near_goal)

def ee_reach_pos_target_soft(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """
    末端位置追踪奖励（软约束）。
    
    逻辑：
    HandTrackingCommand 的 command[:, 0:3] 已经是 Body坐标系下的位置误差。
    我们直接最小化这个误差的模长。
    """
    # 获取命令张量: (N, 7) -> [pos_err(3), rot_err(3), speed(1)]
    command = env.command_manager.get_command(command_name)
    
    # 提取位置误差向量
    pos_error_vec = command[:, 0:3]
    
    # 计算欧几里得距离 L2 Norm
    distance = torch.norm(pos_error_vec, dim=-1)
    
    # 软约束核函数: 1 / (1 + (d/std)^2)
    return 1.0 / (1.0 + torch.square(distance / std))


def reach_rot_target(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """
    末端姿态追踪奖励。
    
    逻辑：
    HandTrackingCommand 的 command[:, 3:6] 是 Body坐标系下的旋转误差(Axis-Angle)。
    """
    command = env.command_manager.get_command(command_name)
    
    # 提取旋转误差向量
    rot_error_vec = command[:, 3:6]
    
    # 模长即为角度误差 (弧度)
    angle_error = torch.norm(rot_error_vec, dim=-1)
    
    # 软约束核函数
    return 1.0 / (1.0 + torch.square(angle_error / std))


def ee_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    ee_body_name: str,
    std: float
) -> torch.Tensor:
    # 1) command: 可能是 (N, 7) / (N, 7, 1) / (N, 7, K)
    command = env.command_manager.get_command(command_name)

    # 取第 7 维速度，并压成 (N,)
    desired_speed = command[:, 6]
    # 如果还有多余维度（如 (N,1) / (N,K)），统一 squeeze/选取
    if desired_speed.ndim > 1:
        desired_speed = desired_speed.squeeze(-1)
    # 兜底：保证最终是一维
    desired_speed = desired_speed.view(-1)

    # 2) actual ee speed
    asset: Articulation = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(ee_body_name)[0]
    ee_lin_vel = asset.data.body_lin_vel_w[:, body_idx, :]   # (N, 3)

    current_speed = torch.norm(ee_lin_vel, dim=-1)          # (N,)
    if current_speed.ndim > 1:
        current_speed = current_speed.squeeze(-1)
    current_speed = current_speed.view(-1)

    # 3) error & reward (N,)
    speed_error = (current_speed - desired_speed).abs()      # (N,)
    rew = 1.0 / (1.0 + (speed_error / std) ** 2)             # (N,)

    # 4) hard guarantee
    return rew.view(env.num_envs)
