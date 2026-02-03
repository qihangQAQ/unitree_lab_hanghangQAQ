# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ================= Helper Functions =================
def quat_to_rot_vector(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为旋转向量 (Axis-Angle form)。
    """
    q = torch.nn.functional.normalize(q, dim=-1)
    w, v = q[:, 0], q[:, 1:]
    w = torch.clamp(w, -1.0, 1.0)

    angle = 2.0 * torch.acos(w)
    sin_half_angle = torch.sqrt(torch.clamp(1 - w * w, min=0.0))

    scale = torch.where(sin_half_angle > 1e-6, angle / sin_half_angle, 2.0)
    return v * scale.unsqueeze(-1)


# def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
#     """
#     从旋转矩阵计算四元数 (N, 3, 3) -> (N, 4) [w, x, y, z]
#     """
#     m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
#     m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
#     m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]
#
#     trace = m00 + m11 + m22
#
#     def safe_sqrt(x):
#         return torch.sqrt(torch.clamp(x, min=0.0))
#
#     s = safe_sqrt(trace + 1.0) * 2
#     w = 0.25 * s
#     x = (m21 - m12) / s
#     y = (m02 - m20) / s
#     z = (m10 - m01) / s
#
#     q = torch.stack([w, x, y, z], dim=1)
#     return torch.nn.functional.normalize(q, dim=-1)

def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为四元数 (w, x, y, z)。
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m = matrix.view(-1, 3, 3)

    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    trace = m00 + m11 + m22

    # =========== 【修正这里】 ===========
    # 之前写 zeros_like(m[:, :4, 0]) 会得到 (N, 3)，因为 m 只有 3 行
    # 必须显式指定形状为 (N, 4)
    q = torch.zeros((m.shape[0], 4), dtype=matrix.dtype, device=matrix.device)
    # ==================================

    # Case 1: Trace > 0
    trace_positive = trace > 0
    if torch.any(trace_positive):
        s = torch.sqrt(trace[trace_positive] + 1.0) * 2
        q[trace_positive, 0] = 0.25 * s
        q[trace_positive, 1] = (m21[trace_positive] - m12[trace_positive]) / s
        q[trace_positive, 2] = (m02[trace_positive] - m20[trace_positive]) / s
        q[trace_positive, 3] = (m10[trace_positive] - m01[trace_positive]) / s

    # Case 2: Trace <= 0
    trace_negative = ~trace_positive
    if torch.any(trace_negative):
        m_neg = m[trace_negative]

        diag = torch.stack([m00, m11, m22], dim=1)
        max_diag_idx = torch.argmax(diag, dim=1)

        # m00 max
        idx_0 = (max_diag_idx == 0) & trace_negative
        if torch.any(idx_0):
            s = torch.sqrt(1.0 + m00[idx_0] - m11[idx_0] - m22[idx_0]) * 2
            q[idx_0, 0] = (m21[idx_0] - m12[idx_0]) / s
            q[idx_0, 1] = 0.25 * s
            q[idx_0, 2] = (m01[idx_0] + m10[idx_0]) / s
            q[idx_0, 3] = (m02[idx_0] + m20[idx_0]) / s

        # m11 max
        idx_1 = (max_diag_idx == 1) & trace_negative
        if torch.any(idx_1):
            s = torch.sqrt(1.0 + m11[idx_1] - m00[idx_1] - m22[idx_1]) * 2
            q[idx_1, 0] = (m02[idx_1] - m20[idx_1]) / s
            q[idx_1, 1] = (m01[idx_1] + m10[idx_1]) / s
            q[idx_1, 2] = 0.25 * s
            q[idx_1, 3] = (m12[idx_1] + m21[idx_1]) / s

        # m22 max
        idx_2 = (max_diag_idx == 2) & trace_negative
        if torch.any(idx_2):
            s = torch.sqrt(1.0 + m22[idx_2] - m00[idx_2] - m11[idx_2]) * 2
            q[idx_2, 0] = (m10[idx_2] - m01[idx_2]) / s
            q[idx_2, 1] = (m02[idx_2] + m20[idx_2]) / s
            q[idx_2, 2] = (m12[idx_2] + m21[idx_2]) / s
            q[idx_2, 3] = 0.25 * s

    q = torch.nn.functional.normalize(q, dim=-1)
    return q.view(*batch_dim, 4)


def random_walk_surface_generation(
    num_envs: int, 
    max_points: int, 
    spacing: torch.Tensor, 
    noise_scale: float, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成随机游走的表面路径点和法向量。
    初始法向为 (0, 0, 1)，后续添加噪声。
    """
    # 初始化 Buffer
    # points: (N, max_points, 3)
    # normals: (N, max_points, 3)
    points = torch.zeros(num_envs, max_points, 3, device=device)
    normals = torch.zeros(num_envs, max_points, 3, device=device)

    # 起始点 (这里假设在机器人前方的一个工作空间内，例如 x=0.5, z=0.4)
    # 你可以根据实际场景调整起始位置
    start_pos = torch.tensor([0.5, 0.0, 0.4], device=device).expand(num_envs, 3)
    points[:, 0, :] = start_pos
    
    # 起始法向 (0, 0, 1)
    start_normal = torch.tensor([0.0, 0.0, 1.0], device=device).expand(num_envs, 3)
    normals[:, 0, :] = start_normal

    # 循环生成后续点
    curr_pos = start_pos.clone()
    curr_normal = start_normal.clone()
    
    # 辅助向量 (用于计算切向)
    # 既然法向接近 Z 轴，我们可以在 XY 平面随机游走
    
    for i in range(1, max_points):
        # 1. 更新法向 (添加微小噪声)
        # 噪声在 -scale 到 +scale 之间
        noise = (torch.rand(num_envs, 3, device=device) - 0.5) * 2 * noise_scale
        curr_normal = torch.nn.functional.normalize(curr_normal + noise, dim=1)
        normals[:, i, :] = curr_normal

        # 2. 计算位移方向 (在法向的垂直平面内随机)
        # 生成一个随机向量
        rand_vec = torch.randn(num_envs, 3, device=device)
        # 施密特正交化: v_perp = v - (v . n) * n
        dot = torch.sum(rand_vec * curr_normal, dim=1, keepdim=True)
        tangent = rand_vec - dot * curr_normal
        tangent = torch.nn.functional.normalize(tangent, dim=1)

        # 3. 更新位置
        # pos_new = pos_old + tangent * spacing
        # 注意 spacing 是 (N,)
        step_vec = tangent * spacing.unsqueeze(1)
        curr_pos = curr_pos + step_vec
        points[:, i, :] = curr_pos
        
    return points, normals

# ================= 核心类 =================

class HandTrackingCommand(CommandTerm):
    """
    6D 喷涂路径追踪命令。
    支持 Config 中的 Default 值和 Range 值切换。
    """
    cfg: HandTrackingCommandCfg

    def __init__(self, cfg: HandTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ee_link_idx = cfg.ee_link_idx
        
        # 预估最大点数 (足够大即可)
        self.max_traj_points = 300
        
        # Buffers
        self.target_traj_buffer = torch.zeros(self.num_envs, self.max_traj_points, 7, device=self.device)
        
        # 存储采样后的参数
        self.env_speeds = torch.zeros(self.num_envs, device=self.device)
        self.env_spacings = torch.zeros(self.num_envs, device=self.device)
        self.env_num_points = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 计时器
        self.time_elapsed = torch.zeros(self.num_envs, device=self.device)

        # Output Command
        self.command_b = torch.zeros(self.num_envs, 7, device=self.device)
        
        # 可视化
        self.target_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/target",
                markers={
                    "sphere": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
                }
            )
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """
        根据 Config 决定使用固定值还是随机范围值。
        """
        num_resets = len(env_ids)
        if num_resets == 0: return

        # --------------------------------------------------------
        # 参数采样逻辑：检查 cfg.ranges 是否存在，以及是否启用 Curriculum
        # 这里为了简单，我们假设：如果 ranges 定义了，就从 ranges 采；
        # 如果你想严格控制只在 Curriculum 开启时才用 range，可以加标志位判断。
        # --------------------------------------------------------

        # 1. 速度采样 (Velocity)
        r_vel = self.cfg.ranges.velocity
        if r_vel is not None:
            # 在 (min, max) 范围内随机
            self.env_speeds[env_ids] = torch.empty(num_resets, device=self.device).uniform_(*r_vel)
        else:
            # 使用固定默认值
            self.env_speeds[env_ids] = self.cfg.default_velocity

        # 2. 间距采样 (Spacing)
        r_space = self.cfg.ranges.spacing
        # if r_space is not None:
        #     self.env_spacings[env_ids] = torch.empty(num_resets, device=self.device).uniform_(*r_space)
        # else:
        self.env_spacings[env_ids] = self.cfg.default_spacing

        # 3. 喷涂距离采样 (Standoff)
        r_stand = self.cfg.ranges.standoff
        # if r_stand is not None:
        #     standoffs = torch.empty(num_resets, device=self.device).uniform_(*r_stand)
        # else:
        standoffs = torch.full((num_resets,), self.cfg.default_standoff, device=self.device)

        # 4. 路径长度采样 (Length)
        # r_len = self.cfg.ranges.path_length
        # if r_len is not None:
        #     lengths = torch.empty(num_resets, device=self.device).uniform_(*r_len)
        # else:
        lengths = torch.full((num_resets,), self.cfg.default_path_length, device=self.device)

        # 重置时间
        self.time_elapsed[env_ids] = 0.0

        # --------------------------------------------------------
        # 路径生成逻辑 (Random Walk) - 保持原样
        # --------------------------------------------------------
        
        # 计算需要的点数
        num_points_needed = torch.ceil(lengths / self.env_spacings[env_ids]).long() + 1
        num_points_needed = torch.clamp(num_points_needed, max=self.max_traj_points)
        self.env_num_points[env_ids] = num_points_needed
        
        # 生成表面点 (这里直接用采样好的 spacing)
        surface_points, surface_normals = random_walk_surface_generation(
            num_resets, self.max_traj_points, self.env_spacings[env_ids], noise_scale=0.05, device=self.device
        )
        
        # 计算 Target Pose
        target_positions = surface_points + surface_normals * standoffs.unsqueeze(1).unsqueeze(2)
        
        # # 计算 Target Orientation (Rx 对齐 -Normal)
        # target_rx = -surface_normals
        # global_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(num_resets, self.max_traj_points, 3)
        # target_ry = torch.cross(global_up, target_rx, dim=-1)
        # target_ry = torch.nn.functional.normalize(target_ry, dim=-1)
        # target_rz = torch.cross(target_rx, target_ry, dim=-1)

        # ==================== 修复开始 ====================
        # 计算 Target Orientation (Rx 对齐 -Normal)
        target_rx = -surface_normals

        # 定义世界坐标系的 Up 轴 (0,0,1)
        global_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 1, 3).expand(num_resets,
                                                                                           self.max_traj_points, 3)

        # 1. 计算 Ry = Up x Rx
        target_ry = torch.cross(global_up, target_rx, dim=-1)

        # 2. 【关键修复】处理死锁 (Rx 与 Up 平行的情况)
        # 计算 Ry 的模长，如果太小说明平行
        ry_norm = torch.norm(target_ry, dim=-1, keepdim=True)
        is_parallel = ry_norm < 1e-4

        # 如果平行，改用 X 轴 (1,0,0) 作为临时 Up 轴来计算 Ry
        # 这样能保证总能算出一个垂直向量
        if torch.any(is_parallel):
            global_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device).view(1, 1, 3).expand_as(global_up)
            alt_ry = torch.cross(global_forward, target_rx, dim=-1)
            # 使用 where 语句进行替换
            target_ry = torch.where(is_parallel, alt_ry, target_ry)

        # 3. 归一化 (加上 eps 防止除零)
        target_ry = torch.nn.functional.normalize(target_ry, dim=-1, eps=1e-6)

        # 4. 计算 Rz
        target_rz = torch.cross(target_rx, target_ry, dim=-1)
        # ==================== 修复结束 ====================

        rot_mat = torch.stack([target_rx, target_ry, target_rz], dim=-1)

        flat_quat = matrix_to_quat(rot_mat.view(-1, 3, 3))

        target_quats = flat_quat.view(num_resets, self.max_traj_points, 4)
        
        # 写入 Buffer
        self.target_traj_buffer[env_ids, :, :3] = target_positions
        self.target_traj_buffer[env_ids, :, 3:] = target_quats

    @property
    def command(self) -> torch.Tensor:
        """
        向外界暴露生成的命令张量。
        CommandManager 会调用这个属性来获取数据。
        """
        return self.command_b


    def _update_command(self):
        """
        基于时间的插值更新 (逻辑保持不变，确保丝滑)
        """
        dt = self.cfg.step_dt
        self.time_elapsed += dt
        
        curr_dist = self.time_elapsed * self.env_speeds
        
        # 使用采样好的 spacing 进行索引计算
        float_idx = curr_dist / self.env_spacings
        
        idx_low = float_idx.long()
        idx_high = idx_low + 1
        
        max_idx = self.env_num_points - 1
        idx_low = torch.min(idx_low, max_idx)
        idx_high = torch.min(idx_high, max_idx)
        
        alpha = float_idx - idx_low.float()
        alpha = torch.clamp(alpha, 0.0, 1.0).unsqueeze(-1)
        
        # Gather logic
        idx_low_expanded = idx_low.view(-1, 1, 1).expand(-1, 1, 7)
        idx_high_expanded = idx_high.view(-1, 1, 1).expand(-1, 1, 7)
        
        p_low = torch.gather(self.target_traj_buffer, 1, idx_low_expanded).squeeze(1)
        p_high = torch.gather(self.target_traj_buffer, 1, idx_high_expanded).squeeze(1)
        
        # Lerp Position
        target_pos_w = (1 - alpha) * p_low[:, :3] + alpha * p_high[:, :3]
        
        # Nlerp Rotation
        rot_interp = (1 - alpha) * p_low[:, 3:] + alpha * p_high[:, 3:]
        target_rot_w = torch.nn.functional.normalize(rot_interp, dim=-1)
        
        # Error Calculation
        ee_state = self.robot.data.body_state_w[:, self.ee_link_idx]
        pos_err_w = target_pos_w - ee_state[:, :3]
        
        quat_err = math_utils.quat_mul(target_rot_w, math_utils.quat_inv(ee_state[:, 3:7]))
        quat_err = torch.where(quat_err[:, 0:1] < 0.0, -quat_err, quat_err)
        rot_err_w = quat_to_rot_vector(quat_err)
        
        # To Base Frame
        root_quat_w = self.robot.data.root_quat_w
        self.command_b[:, 0:3] = math_utils.quat_apply_inverse(root_quat_w, pos_err_w)
        self.command_b[:, 3:6] = math_utils.quat_apply_inverse(root_quat_w, rot_err_w)
        self.command_b[:, 6] = self.env_speeds # 传递当前采样的速度
        
        if self.cfg.debug_vis:
            self.target_marker.visualize(target_pos_w, target_rot_w)

    def _update_metrics(self):
        """
        计算并更新用于日志记录的指标 (metrics)。
        父类 CommandTerm 强制要求实现此方法。
        """
        # 1. 计算位置误差 (command_b 的前3维是 body frame 下的位置误差)
        #    我们要看的是这个误差的模长
        pos_error = torch.norm(self.command_b[:, :3], dim=1)

        # 2. 计算姿态误差 (command_b 的 3:6 维是旋转误差向量)
        #    它的模长近似等于角度误差(弧度)
        rot_error = torch.norm(self.command_b[:, 3:6], dim=1)

        # 3. 将这些数据写入 self.metrics 字典
        #    Isaac Lab 会自动提取这些值并算平均值记录到日志里
        self.metrics["pos_error_l2"] = pos_error
        self.metrics["rot_error_rad"] = rot_error

        # 还可以记录当前任务要求的速度，看看课程学习是否生效
        self.metrics["target_speed"] = self.env_speeds


@configclass
class HandTrackingCommandCfg(CommandTermCfg):
    class_type = HandTrackingCommand
    asset_name: str = MISSING
    ee_link_idx: int = MISSING
    step_dt: float = 0.02

    # 默认值 (当 ranges=None 时使用)
    default_velocity: float = 0.10
    default_spacing: float = 0.01
    default_standoff: float = 0.05
    default_path_length: float = 1.2

    @configclass
    class Ranges:
        velocity: tuple[float, float] = (0.10, 0.40)
        spacing: tuple[float, float] = (0.01, 0.02)
        standoff: tuple[float, float] = (0.05, 0.15)
        path_length: tuple[float, float] = (1.2, 1.5)

    # 【关键修改】同时定义 ranges (当前) 和 limit_ranges (上限)
    ranges: Ranges = Ranges()
    limit_ranges: Ranges = Ranges() # 新增这个，用于存储课程上限

    debug_vis: bool = True