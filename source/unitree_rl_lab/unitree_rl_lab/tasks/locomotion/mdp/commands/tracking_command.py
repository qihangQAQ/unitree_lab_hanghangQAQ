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


# ================= 辅助函数 =================
def quat_to_rot_vector(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为旋转向量 (Axis-Angle 形式)。
    """
    q = torch.nn.functional.normalize(q, dim=-1)
    w, v = q[:, 0], q[:, 1:]
    w = torch.clamp(w, -1.0, 1.0)

    angle = 2.0 * torch.acos(w)
    sin_half_angle = torch.sqrt(torch.clamp(1 - w * w, min=0.0))

    scale = torch.where(sin_half_angle > 1e-6, angle / sin_half_angle, 2.0)
    return v * scale.unsqueeze(-1)


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
    q = torch.zeros((m.shape[0], 4), dtype=matrix.dtype, device=matrix.device)

    # 情况 1: Trace > 0
    trace_positive = trace > 0
    if torch.any(trace_positive):
        s = torch.sqrt(trace[trace_positive] + 1.0) * 2
        q[trace_positive, 0] = 0.25 * s
        q[trace_positive, 1] = (m21[trace_positive] - m12[trace_positive]) / s
        q[trace_positive, 2] = (m02[trace_positive] - m20[trace_positive]) / s
        q[trace_positive, 3] = (m10[trace_positive] - m01[trace_positive]) / s

    # 情况 2: Trace <= 0
    trace_negative = ~trace_positive
    if torch.any(trace_negative):
        m_neg = m[trace_negative]

        diag = torch.stack([m00, m11, m22], dim=1)
        max_diag_idx = torch.argmax(diag, dim=1)

        # m00 最大
        idx_0 = (max_diag_idx == 0) & trace_negative
        if torch.any(idx_0):
            s = torch.sqrt(1.0 + m00[idx_0] - m11[idx_0] - m22[idx_0]) * 2
            q[idx_0, 0] = (m21[idx_0] - m12[idx_0]) / s
            q[idx_0, 1] = 0.25 * s
            q[idx_0, 2] = (m01[idx_0] + m10[idx_0]) / s
            q[idx_0, 3] = (m02[idx_0] + m20[idx_0]) / s

        # m11 最大
        idx_1 = (max_diag_idx == 1) & trace_negative
        if torch.any(idx_1):
            s = torch.sqrt(1.0 + m11[idx_1] - m00[idx_1] - m22[idx_1]) * 2
            q[idx_1, 0] = (m02[idx_1] - m20[idx_1]) / s
            q[idx_1, 1] = (m01[idx_1] + m10[idx_1]) / s
            q[idx_1, 2] = 0.25 * s
            q[idx_1, 3] = (m12[idx_1] + m21[idx_1]) / s

        # m22 最大
        idx_2 = (max_diag_idx == 2) & trace_negative
        if torch.any(idx_2):
            s = torch.sqrt(1.0 + m22[idx_2] - m00[idx_2] - m11[idx_2]) * 2
            q[idx_2, 0] = (m10[idx_2] - m01[idx_2]) / s
            q[idx_2, 1] = (m02[idx_2] + m20[idx_2]) / s
            q[idx_2, 2] = (m12[idx_2] + m21[idx_2]) / s
            q[idx_2, 3] = 0.25 * s

    q = torch.nn.functional.normalize(q, dim=-1)
    return q.view(*batch_dim, 4)


# ================= 核心类 =================

class HandTrackingCommand(CommandTerm):
    """
    在线动态生成的 6D 喷涂路径追踪命令。
    不再预先生成整条轨迹，而是每隔 N 个 step 动态生成下一个目标点。
    """
    cfg: HandTrackingCommandCfg

    def __init__(self, cfg: HandTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ee_link_idx = cfg.ee_link_idx

        # === 核心状态 Buffer ===
        # 当前目标点的位置 (N, 3)
        self.curr_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # 当前目标点的法向 (N, 3) - 用于姿态对齐
        self.curr_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        # 上一次的切向方向 (N, 3) - 用于防止回转
        self.last_tangent_w = torch.zeros(self.num_envs, 3, device=self.device)

        # 采样到的速度 (N,)
        self.env_speeds = torch.zeros(self.num_envs, device=self.device)

        # 计数器 (N,) 用于判断何时更新
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 输出命令 (Body Frame)
        # [0:3] 位置误差, [3:6] 姿态误差, [6] 期望速度
        self.command_b = torch.zeros(self.num_envs, 7, device=self.device)

        # 配置参数：每隔多少个 step 更新一次命令
        self.update_period_steps = 2

        # 可视化设置
        self.target_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/target",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.015,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                    ),
                    "arrow": sim_utils.ConeCfg(
                        radius=0.01,  # 圆锥底部半径
                        height=0.05,  # 圆锥高度（箭头长度）
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
                    ),
                }
            )
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """
        Reset 时的初始化逻辑：设定起点和初始法向。
        """
        num_resets = len(env_ids)
        if num_resets == 0: return

        # 1. 速度采样
        r_vel = self.cfg.ranges.velocity
        if r_vel is not None:
            self.env_speeds[env_ids] = torch.empty(num_resets, device=self.device).uniform_(*r_vel)
        else:
            self.env_speeds[env_ids] = self.cfg.default_velocity

        # 2. 初始化位置: (0.5, 0.0, 0.5) 假设在胸前
        # 初始高度 0.5m 也是在 [0.35, 1.8] 范围内的
        start_pos = torch.tensor([0.5, 0.0, 0.5], device=self.device).expand(num_resets, 3)
        self.curr_pos_w[env_ids] = start_pos

        # 3. 初始化法向: (0.0, -1.0, 0.0) -> 指向右侧 (-Y)
        start_normal = torch.tensor([0.0, -1.0, 0.0], device=self.device).expand(num_resets, 3)
        self.curr_normal_w[env_ids] = start_normal

        # 4. 初始化“上一次切向”
        rand_vec = torch.randn(num_resets, 3, device=self.device)
        dot = torch.sum(rand_vec * start_normal, dim=1, keepdim=True)
        tangent = rand_vec - dot * start_normal
        self.last_tangent_w[env_ids] = torch.nn.functional.normalize(tangent, dim=1)

        # 5. 重置计数器
        self.step_counter[env_ids] = 0

        # 6. 立即计算一次 Command
        self._compute_and_store_command(env_ids)

    def _update_command(self):
        """
        在每个 Simulation Step 调用。
        """
        # 全局计数器增加
        self.step_counter += 1

        # 找出本帧需要更新目标点的环境
        update_mask = (self.step_counter % self.update_period_steps == 0)
        env_ids_to_update = update_mask.nonzero(as_tuple=False).flatten()

        if len(env_ids_to_update) > 0:
            self._generate_next_step(env_ids_to_update)

        # 无论是否更新目标点，每帧都需要刷新 Body Frame 下的 Error
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._compute_and_store_command(all_env_ids)

        # Debug 可视化
        if self.cfg.debug_vis:
            arrow_quat = self._compute_arrow_quat(all_env_ids)
            self.target_marker.visualize(self.curr_pos_w, arrow_quat)

    def _generate_next_step(self, env_ids: torch.Tensor):
        """
        核心生成逻辑：基于当前点，生成下一个点。
        """
        num_updates = len(env_ids)
        dt = self.cfg.step_dt

        # 1. 计算步长: distance = steps * dt * v
        step_dist = self.update_period_steps * dt * self.env_speeds[env_ids]

        curr_n = self.curr_normal_w[env_ids]
        curr_p = self.curr_pos_w[env_ids]
        last_t = self.last_tangent_w[env_ids]

        # 2. 更新法向 (Normal) - 加微小噪声
        noise_scale = 0.05
        noise = (torch.rand(num_updates, 3, device=self.device) - 0.5) * 2 * noise_scale
        new_n = torch.nn.functional.normalize(curr_n + noise, dim=1)

        # 3. 更新位置 (Position) - 防回转逻辑
        rand_vec = torch.randn(num_updates, 3, device=self.device)

        # 投影到切平面
        dot_n = torch.sum(rand_vec * new_n, dim=1, keepdim=True)
        tangent = rand_vec - dot_n * new_n
        tangent = torch.nn.functional.normalize(tangent, dim=1)

        # 防回转检测
        direction_check = torch.sum(tangent * last_t, dim=1, keepdim=True)
        flip_factor = torch.sign(direction_check + 1e-6)
        tangent = tangent * flip_factor

        # 4. 应用更新
        new_p = curr_p + tangent * step_dist.unsqueeze(1)

        # 【新增】高度限制约束
        # 强制将 Z 轴坐标 Clamp 在配置的范围内
        z_min, z_max = self.cfg.height_limits
        new_p[:, 2] = torch.clamp(new_p[:, 2], z_min, z_max)

        # 5. 回写 Buffer
        self.curr_pos_w[env_ids] = new_p
        self.curr_normal_w[env_ids] = new_n
        self.last_tangent_w[env_ids] = tangent

    def _compute_target_quat(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        计算目标的旋转四元数。
        Target X = -Normal (X轴指向墙面)。
        """
        normals = self.curr_normal_w[env_ids]
        num = len(env_ids)

        # 1. 设定 Target X = -Normal
        target_rx = -normals

        # 2. 构建其余轴
        global_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 3).expand(num, 3)

        # Y = Up x X
        target_ry = torch.cross(global_up, target_rx, dim=-1)

        # 处理死锁
        ry_norm = torch.norm(target_ry, dim=-1, keepdim=True)
        is_parallel = (ry_norm < 1e-4).squeeze(-1)

        if torch.any(is_parallel):
            global_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device).view(1, 3).expand(num, 3)
            alt_ry = torch.cross(global_forward[is_parallel], target_rx[is_parallel], dim=-1)
            target_ry[is_parallel] = alt_ry

        target_ry = torch.nn.functional.normalize(target_ry, dim=-1)

        # Z = X x Y
        target_rz = torch.cross(target_rx, target_ry, dim=-1)

        # 构建旋转矩阵 [X, Y, Z]
        rot_mat = torch.stack([target_rx, target_ry, target_rz], dim=-1)

        return matrix_to_quat(rot_mat)

    def _compute_arrow_quat(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        【可视化专用】 Cone 指向 +Z。需对准 Normal。
        Target Z = Normal.
        """
        normals = self.curr_normal_w[env_ids]
        num = len(env_ids)

        target_rz = normals
        global_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 3).expand(num, 3)

        # X = Up x Z
        target_rx = torch.cross(global_up, target_rz, dim=-1)

        rx_norm = torch.norm(target_rx, dim=-1, keepdim=True)
        is_parallel = (rx_norm < 1e-4).squeeze(-1)

        if torch.any(is_parallel):
            global_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device).view(1, 3).expand(num, 3)
            alt_rx = torch.cross(global_forward[is_parallel], target_rz[is_parallel], dim=-1)
            target_rx[is_parallel] = alt_rx

        target_rx = torch.nn.functional.normalize(target_rx, dim=-1)
        target_ry = torch.cross(target_rz, target_rx, dim=-1)
        rot_mat = torch.stack([target_rx, target_ry, target_rz], dim=-1)
        return matrix_to_quat(rot_mat)

    def _compute_and_store_command(self, env_ids: torch.Tensor):
        """
        计算误差并转到 Body Frame，存入 command_b
        """
        target_quat_w = self._compute_target_quat(env_ids)
        target_pos_w = self.curr_pos_w[env_ids]

        ee_state = self.robot.data.body_state_w[env_ids, self.ee_link_idx]
        ee_pos_w = ee_state[:, :3]
        ee_quat_w = ee_state[:, 3:7]

        # 1. 位置误差 (World)
        pos_err_w = target_pos_w - ee_pos_w

        # 2. 姿态误差
        quat_err = math_utils.quat_mul(target_quat_w, math_utils.quat_inv(ee_quat_w))
        quat_err = torch.where(quat_err[:, 0:1] < 0.0, -quat_err, quat_err)
        rot_err_w = quat_to_rot_vector(quat_err)

        # 3. 转到 Base Frame
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        pos_err_b = math_utils.quat_apply_inverse(root_quat_w, pos_err_w)
        rot_err_b = math_utils.quat_apply_inverse(root_quat_w, rot_err_w)

        # 4. 写入 Buffer
        self.command_b[env_ids, 0:3] = pos_err_b
        self.command_b[env_ids, 3:6] = rot_err_b
        self.command_b[env_ids, 6] = self.env_speeds[env_ids]

    @property
    def command(self) -> torch.Tensor:
        return self.command_b

    def _update_metrics(self):
        pos_error = torch.norm(self.command_b[:, :3], dim=1)
        rot_error = torch.norm(self.command_b[:, 3:6], dim=1)
        self.metrics["pos_error_l2"] = pos_error
        self.metrics["rot_error_rad"] = rot_error


@configclass
class HandTrackingCommandCfg(CommandTermCfg):
    class_type = HandTrackingCommand
    asset_name: str = MISSING
    ee_link_idx: int = MISSING
    step_dt: float = 0.02

    # 默认值
    default_velocity: float = 0.10
    default_spacing: float = 0.01
    default_standoff: float = 0.05
    default_path_length: float = 1.2

    # 【新增】高度限制
    height_limits: tuple[float, float] = (0.35, 1.8)

    @configclass
    class Ranges:
        velocity: tuple[float, float] = (0.10, 0.40)
        spacing: tuple[float, float] = (0.01, 0.02)
        standoff: tuple[float, float] = (0.05, 0.15)
        path_length: tuple[float, float] = (1.2, 1.5)

    ranges: Ranges = Ranges()
    limit_ranges: Ranges = Ranges()

    debug_vis: bool = True