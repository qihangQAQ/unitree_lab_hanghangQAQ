# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# =========================
# 辅助数学函数
# =========================

def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=eps)

def quat_to_rot_vector(q: torch.Tensor) -> torch.Tensor:
    q = safe_normalize(q, dim=-1)
    w = torch.clamp(q[:, 0], -1.0, 1.0)
    xyz = q[:, 1:]
    angle = 2.0 * torch.acos(w)
    sin_half = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    scale = torch.where(sin_half > 1e-6, angle / sin_half, 2.0 * torch.ones_like(sin_half))
    return xyz * scale.unsqueeze(-1)

def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {matrix.shape}")
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    q = torch.zeros((m.shape[0], 4), device=m.device, dtype=m.dtype)
    trace = m00 + m11 + m22

    cond0 = trace > 0.0
    cond1 = (~cond0) & (m00 >= m11) & (m00 >= m22)
    cond2 = (~cond0) & (~cond1) & (m11 > m22)
    cond3 = (~cond0) & (~cond1) & (~cond2)

    if torch.any(cond0):
        s = torch.sqrt(torch.clamp(trace[cond0] + 1.0, min=1e-8)) * 2.0
        q[cond0, 0] = 0.25 * s
        q[cond0, 1] = (m21[cond0] - m12[cond0]) / s
        q[cond0, 2] = (m02[cond0] - m20[cond0]) / s
        q[cond0, 3] = (m10[cond0] - m01[cond0]) / s

    if torch.any(cond1):
        s = torch.sqrt(torch.clamp(1.0 + m00[cond1] - m11[cond1] - m22[cond1], min=1e-8)) * 2.0
        q[cond1, 0] = (m21[cond1] - m12[cond1]) / s
        q[cond1, 1] = 0.25 * s
        q[cond1, 2] = (m01[cond1] + m10[cond1]) / s
        q[cond1, 3] = (m02[cond1] + m20[cond1]) / s

    if torch.any(cond2):
        s = torch.sqrt(torch.clamp(1.0 + m11[cond2] - m00[cond2] - m22[cond2], min=1e-8)) * 2.0
        q[cond2, 0] = (m02[cond2] - m20[cond2]) / s
        q[cond2, 1] = (m01[cond2] + m10[cond2]) / s
        q[cond2, 2] = 0.25 * s
        q[cond2, 3] = (m12[cond2] + m21[cond2]) / s

    if torch.any(cond3):
        s = torch.sqrt(torch.clamp(1.0 + m22[cond3] - m00[cond3] - m11[cond3], min=1e-8)) * 2.0
        q[cond3, 0] = (m10[cond3] - m01[cond3]) / s
        q[cond3, 1] = (m02[cond3] + m20[cond3]) / s
        q[cond3, 2] = (m12[cond3] + m21[cond3]) / s
        q[cond3, 3] = 0.25 * s

    return safe_normalize(q, dim=-1).reshape(*batch_shape, 4)

# =========================
# 核心命令类
# =========================

class HandTrackingCommand(CommandTerm):
    cfg: HandTrackingCommandCfg

    def __init__(self, cfg: HandTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ee_link_idx = cfg.ee_link_idx

        self.lookahead_offsets = torch.tensor(
            [i * cfg.lookahead_spacing for i in range(cfg.num_lookahead_points)],
            device=self.device, dtype=torch.float32,
        )

        max_len = max(cfg.ranges.path_length[1], cfg.limit_ranges.path_length[1])
        self.max_num_samples = int(math.ceil(max_len / cfg.point_spacing)) + 2

        self.path_points_w = torch.zeros(self.num_envs, self.max_num_samples, 3, device=self.device)
        self.path_normals_w = torch.zeros(self.num_envs, self.max_num_samples, 3, device=self.device)
        self.path_arc_lengths = torch.zeros(self.num_envs, self.max_num_samples, device=self.device)
        self.path_num_samples = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.path_total_lengths = torch.zeros(self.num_envs, device=self.device)
        
        self.path_completed_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.env_speeds = torch.zeros(self.num_envs, device=self.device)
        self.env_spray_dists = torch.zeros(self.num_envs, device=self.device)
        self.current_arc_length = torch.zeros(self.num_envs, device=self.device)

        self.command_b = torch.zeros(self.num_envs, 28, device=self.device)
        self.current_tangent_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_forward_ref_w = torch.zeros(self.num_envs, 2, device=self.device)

        self.path_ee_targets = torch.zeros(self.num_envs, self.max_num_samples, 3, device=self.device)

        self.current_surf_n_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.target_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/lookahead_points",
                markers={"sphere": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))},
            )
        )
        self.path_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/full_path_line",
                markers={"sphere": sim_utils.SphereCfg(radius=0.004, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))},
            )
        )

    # =========================
    # 张量化路径生成 (拼接模式)
    # =========================

    def _generate_composite_paths(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        if n == 0:
            return

        device = self.device
        root_pos = self.robot.data.root_pos_w[env_ids]
        
        # 1. 采样长度，并计算这批环境所需的实际采样点数
        lengths = torch.empty(n, device=device).uniform_(*self.cfg.ranges.path_length)
        num_samples_each = torch.ceil(lengths / self.cfg.point_spacing).long() + 1
        num_samples_each = torch.clamp(num_samples_each, min=2, max=self.max_num_samples)
        
        # 2. 生成起点 p0
        p0 = root_pos.clone()
        p0[:, 0] += self.cfg.start_x_forward
        p0[:, 1] += torch.empty(n, device=device).uniform_(*self.cfg.start_y_offset_range)
        p0[:, 2] += torch.empty(n, device=device).uniform_(*self.cfg.start_z_offset_range)

        # 3. 准备分块信息
        max_samples = self.max_num_samples
        num_chunks = self.cfg.num_chunks
        chunk_size = (max_samples // num_chunks) + 1  # 确保能盖住 max_samples

        heading = torch.sign(torch.randn(n, 1, 1, device=device))
        
        rand_vals = torch.rand(n, num_chunks, 1, device=device)
        chunk_types = torch.zeros(n, num_chunks, 1, dtype=torch.long, device=device)
        chunk_types[rand_vals < 0.30] = 0
        chunk_types[(rand_vals >= 0.30) & (rand_vals < 0.50)] = 1
        chunk_types[(rand_vals >= 0.50) & (rand_vals < 0.75)] = 2
        chunk_types[rand_vals >= 0.75] = 3

        A = torch.empty(n, num_chunks, 1, device=device).uniform_(*self.cfg.amplitude_range)
        omega = torch.empty(n, num_chunks, 1, device=device).uniform_(*self.cfg.frequency_range)

        # 扩充到所有点 (切片截取到 max_samples)
        types_pt = chunk_types.expand(-1, -1, chunk_size).reshape(n, -1)[:, :max_samples]
        A_pt = A.expand(-1, -1, chunk_size).reshape(n, -1)[:, :max_samples]
        omega_pt = omega.expand(-1, -1, chunk_size).reshape(n, -1)[:, :max_samples]

        ds = self.cfg.point_spacing
        s_local = torch.arange(chunk_size, device=device).float() * ds
        s_pt = s_local.unsqueeze(0).unsqueeze(0).expand(n, num_chunks, chunk_size).reshape(n, -1)[:, :max_samples]

        # 4. 计算速度向量
        vy_0 = torch.ones_like(s_pt)
        vz_0 = torch.zeros_like(s_pt)

        vy_1 = torch.ones_like(s_pt)
        vz_1 = A_pt * omega_pt * torch.cos(omega_pt * s_pt)

        # 大圆环（允许逆行）
        vy_2 = 0.15 - A_pt * omega_pt * torch.sin(omega_pt * s_pt)
        vz_2 = A_pt * omega_pt * torch.cos(omega_pt * s_pt)

        k = 4.0
        vy_3 = torch.ones_like(s_pt) * 0.5
        vz_3 = A_pt * k * omega_pt * torch.cos(omega_pt * s_pt) / torch.cosh(torch.sin(omega_pt * s_pt) * k)**2

        vy = torch.where(types_pt == 0, vy_0, torch.where(types_pt == 1, vy_1, torch.where(types_pt == 2, vy_2, vy_3)))
        vz = torch.where(types_pt == 0, vz_0, torch.where(types_pt == 1, vz_1, torch.where(types_pt == 2, vz_2, vz_3)))

        vy = vy * heading.squeeze(-1)

        v_norm = torch.sqrt(vy**2 + vz**2 + 1e-8)
        vy = (vy / v_norm) * ds
        vz = (vz / v_norm) * ds

        # 5. 速度积分与限幅
        Y = p0[:, 1:2] + torch.cumsum(vy, dim=1)
        Z_raw = p0[:, 2:3] + torch.cumsum(vz, dim=1)
        Z = torch.clamp(Z_raw, min=self.cfg.workspace_z[0], max=self.cfg.workspace_z[1])

        # X轴平滑噪声
        x_noise = torch.randn(n, max_samples, device=device) * self.cfg.x_noise_scale
        X = p0[:, 0:1] + torch.cumsum(x_noise, dim=1)

        curve = torch.stack([X, Y, Z], dim=-1)

        # 6. 计算真实弧长 (使用实际的线段长度累加，最为精准)
        deltas = curve[:, 1:, :] - curve[:, :-1, :]
        seg_lens = torch.norm(deltas, dim=-1)
        arc = torch.zeros(n, max_samples, device=device)
        arc[:, 1:] = torch.cumsum(seg_lens, dim=1)

        batch = torch.arange(n, device=device)
        total_lengths = arc[batch, num_samples_each - 1]

        # 7. 生成带漂移的法向量
        n_base = torch.tensor([-1.0, 0.0, 0.0], device=device).view(1, 1, 3).expand(n, max_samples, 3)
        n_step_noise = torch.randn(n, max_samples, 3, device=device) * self.cfg.normal_noise_scale
        n_step_noise[:, :, 0] = 0.0
        n_drift = torch.cumsum(n_step_noise, dim=1)
        normals = safe_normalize(n_base + n_drift, dim=-1)

        # 8. 写入 Buffer
        self.path_points_w[env_ids] = curve
        self.path_normals_w[env_ids] = normals
        self.path_arc_lengths[env_ids] = arc
        self.path_num_samples[env_ids] = num_samples_each
        self.path_total_lengths[env_ids] = total_lengths

        total_offset = self.cfg.gun_length + self.env_spray_dists[env_ids]
        self.path_ee_targets[env_ids] = curve + normals * total_offset.view(-1, 1, 1)

        if self.cfg.debug_vis and env_ids.numel() > 0:
            first_env = env_ids[:1]
            # 只提取第一个环境的有效点，避免画面太乱
            valid_pts = num_samples_each[0].item()
            self.path_markers.visualize(self.path_ee_targets[first_env, :valid_pts, :].reshape(-1, 3))

    def _make_target_quat_from_normals(self, normals: torch.Tensor) -> torch.Tensor:
        target_rx = safe_normalize(-normals, dim=-1)
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(target_rx)
        target_ry = safe_normalize(torch.cross(world_up, target_rx, dim=-1), dim=-1)
        target_rz = safe_normalize(torch.cross(target_rx, target_ry, dim=-1), dim=-1)
        target_ry = safe_normalize(torch.cross(target_rz, target_rx, dim=-1), dim=-1)
        rot_mat = torch.stack([target_rx, target_ry, target_rz], dim=-1)
        return matrix_to_quat(rot_mat)

    # =========================
    # 更新与重置
    # =========================

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0: return

        self.env_speeds[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.velocity)
        self.env_spray_dists[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.spray_distance)
        
        self.current_arc_length[env_ids] = 0.0
        self.path_completed_buf[env_ids] = False

        forward_w = math_utils.quat_apply(self.robot.data.root_quat_w[env_ids], torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1))
        self.base_forward_ref_w[env_ids] = forward_w[:, :2] / (torch.norm(forward_w[:, :2], dim=-1, keepdim=True) + 1e-8)

        self._generate_composite_paths(env_ids)
        self._compute_and_store_command(env_ids)

    def _update_command(self):
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        self.current_arc_length += self.env_speeds * self.cfg.step_dt
        self.current_arc_length = torch.minimum(self.current_arc_length, self.path_total_lengths)
        
        self.path_completed_buf = self.current_arc_length >= (self.path_total_lengths - 1e-3)

        self._compute_and_store_command(env_ids)

    def _get_interpolated_target(self, env_ids: torch.Tensor, target_s: torch.Tensor):
        arc = self.path_arc_lengths[env_ids]
        max_s = self.path_total_lengths[env_ids]
        target_s = torch.clamp(target_s, torch.zeros_like(target_s), max_s - 1e-6)

        mask = arc >= target_s.unsqueeze(1)
        idx1 = torch.clamp(torch.argmax(mask.to(torch.int64), dim=1), min=1)
        idx0 = idx1 - 1
        batch = torch.arange(env_ids.shape[0], device=self.device)

        s0, s1 = arc[batch, idx0], arc[batch, idx1]
        p0, p1 = self.path_points_w[env_ids, idx0], self.path_points_w[env_ids, idx1]
        n0, n1 = self.path_normals_w[env_ids, idx0], self.path_normals_w[env_ids, idx1]

        alpha = ((target_s - s0) / torch.clamp(s1 - s0, min=1e-6)).unsqueeze(1)
        
        interp_p = p0 + alpha * (p1 - p0)
        interp_n = safe_normalize(n0 + alpha * (n1 - n0), dim=-1)

        return interp_p, interp_n

    def _compute_and_store_command(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0: return

        curr_s = self.current_arc_length[env_ids]
        ee_state = self.robot.data.body_state_w[env_ids, self.ee_link_idx]
        ee_pos_w, ee_quat_w = ee_state[:, :3], ee_state[:, 3:7]
        root_quat_w = self.robot.data.root_quat_w[env_ids]

        command_chunks = []
        vis_points = []

        for offset in self.lookahead_offsets:
            target_s = curr_s + offset
            surf_p, surf_n = self._get_interpolated_target(env_ids, target_s)

            total_offset = self.cfg.gun_length + self.env_spray_dists[env_ids]
            target_p_ee = surf_p + surf_n * total_offset.unsqueeze(1)
            target_q_ee = self._make_target_quat_from_normals(surf_n)

            pos_err_w = target_p_ee - ee_pos_w
            pos_err_b = math_utils.quat_apply_inverse(root_quat_w, pos_err_w)

            quat_err = math_utils.quat_mul(target_q_ee, math_utils.quat_inv(ee_quat_w))
            quat_err = torch.where(quat_err[:, 0:1] < 0.0, -quat_err, quat_err)
            rot_err_b = math_utils.quat_apply_inverse(root_quat_w, quat_to_rot_vector(quat_err))

            command_chunks.extend([pos_err_b, rot_err_b])
            vis_points.append(target_p_ee)

        self.command_b[env_ids, :24] = torch.cat(command_chunks, dim=1)
        self.command_b[env_ids, 24] = self.env_speeds[env_ids]

        p_curr, n_curr = self._get_interpolated_target(env_ids, curr_s)
        p_next, _ = self._get_interpolated_target(env_ids, curr_s + 0.05)
        tangent_w = safe_normalize(p_next - p_curr, dim=-1)
        tangent_b = math_utils.quat_apply_inverse(root_quat_w, tangent_w)
        self.current_tangent_b[env_ids] = tangent_b

        # 基座期望速度命令 (base frame)
        speeds = self.env_speeds[env_ids]
        self.command_b[env_ids, 25] = tangent_b[:, 0] * speeds  # v_x_b
        self.command_b[env_ids, 26] = tangent_b[:, 1] * speeds  # v_y_b

        # 身体期望高度命令 (暂时固定站立高度，后续可改为动态)
        self.command_b[env_ids, 27] = 0.78

        self.current_surf_n_w[env_ids] = n_curr

        if self.cfg.debug_vis and len(vis_points) > 0:
            self.target_markers.visualize(torch.cat(vis_points, dim=0))

    @property
    def command(self) -> torch.Tensor:
        return self.command_b

    @property
    def is_completed(self) -> torch.Tensor:
        return self.path_completed_buf

    def _update_metrics(self):
        self.metrics["tracking_pos_error"] = torch.norm(self.command_b[:, 0:3], dim=-1)
        self.metrics["tracking_rot_error"] = torch.norm(self.command_b[:, 3:6], dim=-1)


# =========================
# 配置文件
# =========================

@configclass
class HandTrackingCommandCfg(CommandTermCfg):
    class_type = HandTrackingCommand

    asset_name: str = MISSING
    ee_link_idx: int = MISSING
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    step_dt: float = 0.02       # 环境的物理控制周期，0.02 表示 50Hz 控制频率
    gun_length: float = 0.15    # 喷枪的物理长度 (米)
    point_spacing: float = 0.01 # 轨迹点的离散采样间距。0.01 表示每隔 1cm 存一个坐标点

    num_lookahead_points: int = 4       # 观测空间(Observations)中的前瞻点数量
    lookahead_spacing: float = 0.05     # 前瞻点之间的弧长间隔。0.05 表示未来 0cm(当前), 5cm, 10cm, 15cm 处的四个目标

    # ==========================
    # 拼接轨迹专有配置 (新增)
    # ==========================
    num_chunks: int = 5                                  # 每条路径被分成的形态段数
    amplitude_range: tuple[float, float] = (0.15, 0.40)  # 波浪/圆环的振幅范围
    frequency_range: tuple[float, float] = (8.0, 15.0)   # 频率范围
    x_noise_scale: float = 0.0012                        # 墙面不平整度 (X 轴随机游走噪声)。
    normal_noise_scale: float = 0.00                     # 墙面法向不平整度。0.02 意味着法向量（决定喷枪姿态）会有轻微的扭曲摇摆

    start_x_forward: float = 0.50                       # 起点控制：第一点固定在机器人 root 坐标系正前方 50cm 处
    start_y_offset_range: tuple[float, float] = (-0.1, 0.1)# 起点在左右 (Y 轴) 方向上的随机偏移范围，增加初始位置的多样性 (-10cm 到 10cm)
    start_z_offset_range: tuple[float, float] = (0.2, 0.4)# 起点高度相对于机器人 root 高度的偏移范围 (往上偏 20cm 到 40cm，大概是胸前位置)
    workspace_z: tuple[float, float] = (0.88, 1.10)# 绝对安全工作空间 (Z 轴高度)。生成的轨迹在任何情况下都会被强制截断在这个高度范围内，

    debug_vis: bool = True

    @configclass
    class Ranges:
        velocity: tuple[float, float] = (0.10, 0.10)# 期望的喷涂移动速度 (10cm/s)，起步阶段固定速度，
        spray_distance: tuple[float, float] = (0.05, 0.10)# 喷枪头距离墙面的期望空隙距离 (5cm 到 10cm 随机)
        path_length: tuple[float, float] = (1.0, 3.0)# 整条轨迹的长度范围 (1m 到 3m)。前期走得短，更容易拿高分，建立信心

    @configclass
    class LimitRanges:
        velocity: tuple[float, float] = (0.10, 0.30)# 随着课程解锁，最终机器人要能处理高达 40cm/s 的高速喷涂任务
        spray_distance: tuple[float, float] = (0.05, 0.15)# 喷涂距离允许在 5cm 到 15cm 之间随机变化
        path_length: tuple[float, float] = (5.0, 8.0)# 最终机器人需要能一口气走完 5m 到 8m 长的超长复合墙面

    ranges: Ranges = Ranges()
    limit_ranges: LimitRanges = LimitRanges()