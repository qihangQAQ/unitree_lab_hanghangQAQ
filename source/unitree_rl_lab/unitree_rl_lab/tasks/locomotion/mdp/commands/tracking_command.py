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
# 辅助函数
# =========================

def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=eps)


def quat_to_rot_vector(q: torch.Tensor) -> torch.Tensor:
    """将四元数转换为旋转向量 (axis-angle / rotvec). 输入四元数格式: (w, x, y, z)"""
    q = safe_normalize(q, dim=-1)
    w = torch.clamp(q[:, 0], -1.0, 1.0)
    xyz = q[:, 1:]

    angle = 2.0 * torch.acos(w)
    sin_half = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

    scale = torch.where(sin_half > 1e-6, angle / sin_half, 2.0 * torch.ones_like(sin_half))
    return xyz * scale.unsqueeze(-1)


def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 -> 四元数 (w, x, y, z)，支持 batch。"""
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

    q = safe_normalize(q, dim=-1)
    return q.reshape(*batch_shape, 4)


# =========================
# 核心类
# =========================

class HandTrackingCommand(CommandTerm):
    """
    简化喷涂路径跟踪命令生成器（轻量加速版）。

    设计目标：
    1. 每个 episode 生成一条 Bézier 曲线
    2. 起点在机器人胸前约 0.5m
    3. 整体主运动在 y-z 平面，x 只有小扰动
    4. 高度 z 限制在 [0.5, 2.2]
    5. 法向从 (-1, 0, 0) 开始，后续在前一点基础上小幅扰动
    6. 命令 = 4 个参考点的位置误差 + 姿态误差 + 当前期望速度
    """

    cfg: HandTrackingCommandCfg

    def __init__(self, cfg: HandTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ee_link_idx = cfg.ee_link_idx

        # lookahead
        self.lookahead_offsets = torch.tensor(
            [i * cfg.lookahead_spacing for i in range(cfg.num_lookahead_points)],
            device=self.device,
            dtype=torch.float32,
        )

        # 依据当前范围和课程上限，决定内部最大缓存长度
        max_len = max(cfg.ranges.path_length[1], cfg.limit_ranges.path_length[1])
        self.max_num_samples = int(math.ceil(max_len / cfg.point_spacing)) + 2

        # path buffers
        self.path_points_w = torch.zeros(self.num_envs, self.max_num_samples, 3, device=self.device)
        self.path_normals_w = torch.zeros(self.num_envs, self.max_num_samples, 3, device=self.device)
        self.path_arc_lengths = torch.zeros(self.num_envs, self.max_num_samples, device=self.device)
        self.path_num_samples = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.path_total_lengths = torch.zeros(self.num_envs, device=self.device)
        self.path_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # per-episode sampled parameters
        self.env_speeds = torch.zeros(self.num_envs, device=self.device)
        self.env_spray_dists = torch.zeros(self.num_envs, device=self.device)

        # path progress (arc-length parameter s)
        self.current_arc_length = torch.zeros(self.num_envs, device=self.device)

        # command: 4 * (3 pos + 3 rot) + 1 speed = 25
        self.command_b = torch.zeros(self.num_envs, 25, device=self.device)

        # debug buffers
        self.path_ee_targets = torch.zeros(self.num_envs, self.max_num_samples, 3, device=self.device)

        # 给 reward 用的参考前向
        self.base_forward_ref_w = torch.zeros(self.num_envs, 2, device=self.device)

        # 给 reward 用的当前路径切线（Base 系）
        self.current_tangent_b = torch.zeros(self.num_envs, 3, device=self.device)

        # visualization
        self.target_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/lookahead_points",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.015,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
        )

        self.path_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/full_path_line",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.004,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
        )

    # =========================
    # 内部几何辅助
    # =========================

    def _sample_random_walk_normals(self, num_envs: int, num_samples: int) -> torch.Tensor:
        """
        法向从 (-1, 0, 0) 开始，后续每个点在前一个点基础上做微小扰动。
        只在 y/z 上扰动，保持整体大致朝 -x。
        """
        normals = torch.zeros(num_envs, num_samples, 3, device=self.device)
        normals[:, 0, 0] = -1.0

        if num_samples <= 1:
            return normals

        for i in range(1, num_samples):
            step_noise = torch.zeros(num_envs, 3, device=self.device)
            step_noise[:, 1] = torch.empty(num_envs, device=self.device).uniform_(
                -self.cfg.normal_noise_yz, self.cfg.normal_noise_yz
            )
            step_noise[:, 2] = torch.empty(num_envs, device=self.device).uniform_(
                -self.cfg.normal_noise_yz, self.cfg.normal_noise_yz
            )
            normals[:, i] = safe_normalize(normals[:, i - 1] + step_noise, dim=-1)

        return normals

    def _make_target_quat_from_normals(self, normals: torch.Tensor) -> torch.Tensor:
        """
        根据 surface normal 生成末端期望姿态。
        约定：喷枪工具坐标系 x 轴是喷射方向，因此 x 轴对准 -normal。
        """
        num = normals.shape[0]
        target_rx = safe_normalize(-normals, dim=-1)

        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(num, 3)
        target_ry = torch.cross(world_up, target_rx, dim=-1)

        # 若 target_rx 与 world_up 平行，则改用 world_y 作为参考
        bad = torch.norm(target_ry, dim=-1) < 1e-6
        if torch.any(bad):
            alt_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(num, 3)
            target_ry[bad] = torch.cross(alt_axis[bad], target_rx[bad], dim=-1)

        target_ry = safe_normalize(target_ry, dim=-1)
        target_rz = safe_normalize(torch.cross(target_rx, target_ry, dim=-1), dim=-1)
        target_ry = safe_normalize(torch.cross(target_rz, target_rx, dim=-1), dim=-1)

        rot_mat = torch.stack([target_rx, target_ry, target_rz], dim=-1)
        return matrix_to_quat(rot_mat)

    # =========================
    # 路径生成（轻量版）
    # =========================

    def _generate_bezier_curves(self, env_ids: torch.Tensor):
        """
        轻量版 Bézier 路径生成：
        - 不做 rejection sampling
        - 不做复杂合法性检查
        - 起点在胸前
        - z 高度直接 clamp 到 [0.5, 2.2]
        - 法向用 random walk
        """
        n = env_ids.numel()
        if n == 0:
            return

        device = self.device
        root_pos = self.robot.data.root_pos_w[env_ids]  # (N, 3)

        # 1) 采样路径长度
        path_lengths = torch.empty(n, device=device).uniform_(*self.cfg.ranges.path_length)

        # 2) 根据“平均 1cm 采样密度”决定每条轨迹的有效点数
        num_samples_each = torch.ceil(path_lengths / self.cfg.point_spacing).long() + 1
        num_samples_each = torch.clamp(num_samples_each, min=2, max=self.max_num_samples)

        # 3) 起点：机器人胸前约 0.5m
        p0 = root_pos.clone()
        p0[:, 0] = root_pos[:, 0] + self.cfg.start_x_forward
        p0[:, 1] = root_pos[:, 1] + torch.empty(n, device=device).uniform_(*self.cfg.start_y_offset_range)
        p0[:, 2] = root_pos[:, 2] + torch.empty(n, device=device).uniform_(*self.cfg.start_z_offset_range)

        # 4) 主方向：大致在 y-z 平面，x 只有小扰动
        theta = torch.empty(n, device=device).uniform_(*self.cfg.yz_heading_angle_range)
        x_drift = torch.empty(n, device=device).uniform_(
            -self.cfg.max_x_drift_ratio, self.cfg.max_x_drift_ratio
        )

        tangent = torch.stack([x_drift, torch.cos(theta), torch.sin(theta)], dim=1)
        tangent = safe_normalize(tangent, dim=-1)

        # 5) 弯曲方向，仍主要在 y-z 平面内
        bend_dir = torch.stack(
            [
                torch.zeros(n, device=device),
                -tangent[:, 2],
                tangent[:, 1],
            ],
            dim=1,
        )
        bend_dir = safe_normalize(bend_dir, dim=-1)

        bend_scale = path_lengths * self.cfg.curvature_ratio
        b1 = torch.empty(n, device=device).uniform_(-1.0, 1.0) * bend_scale
        b2 = torch.empty(n, device=device).uniform_(-1.0, 1.0) * bend_scale

        p1 = p0 + tangent * (path_lengths / 3.0).unsqueeze(1) + bend_dir * b1.unsqueeze(1)
        p2 = p0 + tangent * (2.0 * path_lengths / 3.0).unsqueeze(1) + bend_dir * b2.unsqueeze(1)
        p3 = p0 + tangent * path_lengths.unsqueeze(1)

        # 6) 使用统一 max_num_samples 做张量生成。
        #    对于每条 path，前 m 个点按各自长度均匀采样，后面自动保持在终点。
        idx = torch.arange(self.max_num_samples, device=device).view(1, -1).float()
        denom = torch.clamp((num_samples_each - 1).view(-1, 1).float(), min=1.0)
        t = torch.clamp(idx / denom, max=1.0).unsqueeze(-1)  # (N, M, 1)
        u = 1.0 - t

        curve = (
            (u ** 3) * p0.unsqueeze(1)
            + 3.0 * (u ** 2) * t * p1.unsqueeze(1)
            + 3.0 * u * (t ** 2) * p2.unsqueeze(1)
            + (t ** 3) * p3.unsqueeze(1)
        )  # (N, M, 3)

        # 7) 高度限制：直接 clamp 到 [0.5, 2.2]
        curve[:, :, 2] = torch.clamp(curve[:, :, 2], min=self.cfg.workspace_z[0], max=self.cfg.workspace_z[1])

        # 8) 法向：random walk
        normals = self._sample_random_walk_normals(n, self.max_num_samples)

        # 把每条路径超出有效点数之后的法向固定为最后一个有效法向
        valid_mask = (
            torch.arange(self.max_num_samples, device=device).view(1, -1)
            < num_samples_each.view(-1, 1)
        ).unsqueeze(-1)  # (N, M, 1)

        last_idx = (num_samples_each - 1).view(-1, 1, 1).expand(-1, 1, 3)
        last_normals = normals.gather(1, last_idx).expand(-1, self.max_num_samples, -1)
        normals = torch.where(valid_mask, normals, last_normals)

        # 9) 计算弧长
        deltas = curve[:, 1:, :] - curve[:, :-1, :]
        seg_lens = torch.norm(deltas, dim=-1)

        arc = torch.zeros(n, self.max_num_samples, device=device)
        arc[:, 1:] = torch.cumsum(seg_lens, dim=1)

        # 各条 path 的真实总长度取各自最后一个有效点对应的弧长
        batch = torch.arange(n, device=device)
        total_lengths = arc[batch, num_samples_each - 1]

        # 10) 写入 buffer
        self.path_points_w[env_ids] = curve
        self.path_normals_w[env_ids] = normals
        self.path_arc_lengths[env_ids] = arc
        self.path_num_samples[env_ids] = num_samples_each
        self.path_total_lengths[env_ids] = total_lengths
        self.path_completed[env_ids] = False

        total_offset = self.cfg.gun_length + self.env_spray_dists[env_ids]
        self.path_ee_targets[env_ids] = curve + normals * total_offset.view(-1, 1, 1)

        # debug 可视化：只画第一个 env 的整条轨迹，避免太重
        if self.cfg.debug_vis and env_ids.numel() > 0:
            first_env = env_ids[:1]
            self.path_markers.visualize(self.path_ee_targets[first_env].reshape(-1, 3))

    # =========================
    # CommandTerm 生命周期函数
    # =========================

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()

        # 每个 episode 采样速度、喷涂距离
        self.env_speeds[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.velocity)
        self.env_spray_dists[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.spray_distance)

        # reset path progress
        self.current_arc_length[env_ids] = 0.0
        self.path_completed[env_ids] = False

        # 记录 reset 时 base 的世界系前向（奖励里可能会用）
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(n, 1)
        forward_w = math_utils.quat_apply(self.robot.data.root_quat_w[env_ids], forward_local)
        self.base_forward_ref_w[env_ids] = safe_normalize(forward_w[:, :2], dim=-1)

        # 生成新 path
        self._generate_bezier_curves(env_ids)

        # 不 teleport base，保持默认 reset 姿态
        self._compute_and_store_command(env_ids)

    def _update_command(self):
        """每个控制周期推进弧长进度：s_{t+1} = s_t + v_des * dt"""
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        self.current_arc_length += self.env_speeds * self.cfg.step_dt
        self.current_arc_length = torch.minimum(self.current_arc_length, self.path_total_lengths)

        self.path_completed = self.current_arc_length >= (self.path_total_lengths - self.cfg.path_finish_epsilon)

        self._compute_and_store_command(env_ids)

    def _get_interpolated_target(self, env_ids: torch.Tensor, target_s: torch.Tensor):
        """
        按弧长在预计算曲线上插值，返回:
        - surface point
        - surface normal
        """
        arc = self.path_arc_lengths[env_ids]  # (N, M)
        max_s = self.path_total_lengths[env_ids]

        # clamp 到合法范围
        target_s = torch.clamp(target_s, min=0.0)
        target_s = torch.minimum(target_s, torch.clamp(max_s - 1e-6, min=0.0))

        # 找到第一个 >= target_s 的索引
        mask = arc >= target_s.unsqueeze(1)
        idx1 = torch.argmax(mask.to(torch.int64), dim=1)
        idx1 = torch.clamp(idx1, min=1)
        idx0 = idx1 - 1

        batch = torch.arange(env_ids.shape[0], device=self.device)

        s0 = arc[batch, idx0]
        s1 = arc[batch, idx1]

        p0 = self.path_points_w[env_ids, idx0]
        p1 = self.path_points_w[env_ids, idx1]

        n0 = self.path_normals_w[env_ids, idx0]
        n1 = self.path_normals_w[env_ids, idx1]

        denom = torch.clamp(s1 - s0, min=1e-6)
        alpha = ((target_s - s0) / denom).unsqueeze(1)

        interp_p = p0 + alpha * (p1 - p0)
        interp_n = safe_normalize(n0 + alpha * (n1 - n0), dim=-1)

        return interp_p, interp_n

    def _compute_and_store_command(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        curr_s = self.current_arc_length[env_ids]

        ee_state = self.robot.data.body_state_w[env_ids, self.ee_link_idx]
        ee_pos_w = ee_state[:, :3]
        ee_quat_w = ee_state[:, 3:7]
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

            quat_err = math_utils.quat_mul(target_q_ee, math_utils.quat_inv(ee_quat_w))
            quat_err = torch.where(quat_err[:, 0:1] < 0.0, -quat_err, quat_err)
            rot_err_w = quat_to_rot_vector(quat_err)

            pos_err_b = math_utils.quat_apply_inverse(root_quat_w, pos_err_w)
            rot_err_b = math_utils.quat_apply_inverse(root_quat_w, rot_err_w)

            command_chunks.append(pos_err_b)
            command_chunks.append(rot_err_b)
            vis_points.append(target_p_ee)

        all_errors = torch.cat(command_chunks, dim=1)  # (N, 24)
        self.command_b[env_ids, :24] = all_errors
        self.command_b[env_ids, 24] = self.env_speeds[env_ids]

        # debug 可视化：只画第一个 env 的 4 个 lookahead 点，减轻开销
        if self.cfg.debug_vis and len(vis_points) > 0:
            vis_points = torch.stack(vis_points, dim=1)  # (N, 4, 3)
            self.target_markers.visualize(vis_points[0])

        # 计算 Base 系下的当前路径切线（reward 里会用）
        p_curr, _ = self._get_interpolated_target(env_ids, curr_s)
        p_next, _ = self._get_interpolated_target(env_ids, curr_s + self.cfg.lookahead_spacing)

        tangent_w = safe_normalize(p_next - p_curr, dim=-1)
        self.current_tangent_b[env_ids] = math_utils.quat_apply_inverse(root_quat_w, tangent_w)

    @property
    def command(self) -> torch.Tensor:
        return self.command_b

    @property
    def is_completed(self) -> torch.Tensor:
        return self.path_completed

    @property
    def is_path_completed(self) -> torch.Tensor:
        return self.path_completed

    @property
    def path_progress(self) -> torch.Tensor:
        return self.current_arc_length / torch.clamp(self.path_total_lengths, min=1e-6)

    def _update_metrics(self):
        pos_error = torch.norm(self.command_b[:, 0:3], dim=-1)
        rot_error = torch.norm(self.command_b[:, 3:6], dim=-1)

        self.metrics["tracking_pos_error"] = pos_error
        self.metrics["tracking_rot_error"] = rot_error
        self.metrics["path_progress"] = self.path_progress.float()
        self.metrics["path_completed"] = self.path_completed.float()


# =========================
# 配置
# =========================

@configclass
class HandTrackingCommandCfg(CommandTermCfg):
    class_type = HandTrackingCommand

    asset_name: str = MISSING
    ee_link_idx: int = MISSING

    # 控制周期
    step_dt: float = 0.02

    # 喷枪长度
    gun_length: float = 0.15

    # 路径离散间距：按这个值决定采样密度（平均接近 1cm）
    point_spacing: float = 0.01

    # 4 个参考点，间隔 5cm
    num_lookahead_points: int = 4
    lookahead_spacing: float = 0.05

    # 法向扰动幅度（random walk，每一步只在 y/z 上加微小扰动）
    normal_noise_yz: float = 0.01

    # 起点设置：机器人胸前约 0.5m
    start_x_forward: float = 0.50
    start_y_offset_range: tuple[float, float] = (-0.08, 0.08)
    start_z_offset_range: tuple[float, float] = (0.35, 0.55)  # 相对 root z

    # path 主要在 y-z 平面，x 只允许小扰动
    max_x_drift_ratio: float = 0.03
    curvature_ratio: float = 0.06
    yz_heading_angle_range: tuple[float, float] = (-1.10, 1.10)

    # 高度范围（只对 z 做硬限制）
    workspace_z: tuple[float, float] = (0.50, 2.20)

    # path 完成阈值
    path_finish_epsilon: float = 1e-3

    # 下面这些字段保留，是为了尽量兼容你原来的 cfg，不再参与复杂筛选
    workspace_x_rel: tuple[float, float] = (0.35, 0.75)
    workspace_y_rel: tuple[float, float] = (-1.10, 1.10)
    ee_reach_radius: tuple[float, float] = (0.35, 1.30)
    length_tolerance_ratio: tuple[float, float] = (0.85, 1.15)
    max_resample_attempts: int = 1

    @configclass
    class Ranges:
        # 第一阶段：先用 1~3m
        velocity: tuple[float, float] = (0.10, 0.10)
        spray_distance: tuple[float, float] = (0.05, 0.10)
        path_length: tuple[float, float] = (1.0, 3.0)

    @configclass
    class LimitRanges:
        # 后续课程设计可逐步拉到这里
        velocity: tuple[float, float] = (0.10, 0.40)
        spray_distance: tuple[float, float] = (0.05, 0.12)
        path_length: tuple[float, float] = (5.0, 8.0)

    ranges: Ranges = Ranges()
    limit_ranges: LimitRanges = LimitRanges()

    # 训练时强烈建议关掉
    debug_vis: bool = False