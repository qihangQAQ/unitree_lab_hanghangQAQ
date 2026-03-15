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
    简化喷涂路径跟踪命令生成器。

    设计目标：
    1. 每个 episode 生成一条受约束的 Bézier 曲线
    2. path 主运动在 y-z 平面，x 只允许小扰动
    3. 起点在机器人胸前约 0.5m 处
    4. 全部采样点满足 workspace 限制
    5. 法向近似固定为 (-1, 0, 0)，允许平滑小扰动
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

        # 为了支持 1cm 间距 + 课程长度上限，内部固定最大 buffer 长度
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

        # debug path for ee target
        self.path_ee_targets = torch.zeros(self.num_envs, self.max_num_samples, 3, device=self.device)

        # [补丁] 提供给 base_heading_hold 奖励用的基准朝向
        self.base_forward_ref_w = torch.zeros(self.num_envs, 2, device=self.device)

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

    def _sample_smooth_normals(self, num_samples: int) -> torch.Tensor:
        """生成一条 path 上的法向：整体接近 (-1, 0, 0)，但有平滑小扰动。"""
        base = torch.tensor([-1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)

        # 起点噪声 / 终点噪声，做平滑插值，避免每点独立白噪声抖动
        n0 = base.clone()
        n1 = base.clone()

        n0[1] += torch.empty(1, device=self.device).uniform_(
            -self.cfg.normal_noise_yz, self.cfg.normal_noise_yz
        )[0]
        n0[2] += torch.empty(1, device=self.device).uniform_(
            -self.cfg.normal_noise_yz, self.cfg.normal_noise_yz
        )[0]

        n1[1] += torch.empty(1, device=self.device).uniform_(
            -self.cfg.normal_noise_yz, self.cfg.normal_noise_yz
        )[0]
        n1[2] += torch.empty(1, device=self.device).uniform_(
            -self.cfg.normal_noise_yz, self.cfg.normal_noise_yz
        )[0]

        alpha = torch.linspace(0.0, 1.0, num_samples, device=self.device).unsqueeze(1)
        normals = (1.0 - alpha) * n0.unsqueeze(0) + alpha * n1.unsqueeze(0)
        normals = safe_normalize(normals, dim=-1)
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
    # 路径生成
    # =========================

    def _is_curve_valid(
        self,
        env_id: int,
        curve: torch.Tensor,
        normals: torch.Tensor,
        actual_length: float,
        desired_length: float,
    ) -> bool:
        """检查整条曲线是否合法。"""
        root_pos = self.robot.data.root_pos_w[env_id]

        # 1. 长度接近期望长度
        if not (self.cfg.length_tolerance_ratio[0] * desired_length <= actual_length <=
                self.cfg.length_tolerance_ratio[1] * desired_length):
            return False

        # 2. 高度范围
        z_ok = torch.all((curve[:, 2] >= self.cfg.workspace_z[0]) & (curve[:, 2] <= self.cfg.workspace_z[1]))
        if not bool(z_ok):
            return False

        # 3. x 相对 root 只允许小扰动
        rel_x = curve[:, 0] - root_pos[0]
        x_ok = torch.all((rel_x >= self.cfg.workspace_x_rel[0]) & (rel_x <= self.cfg.workspace_x_rel[1]))
        if not bool(x_ok):
            return False

        # 4. y 相对 root 限制
        rel_y = curve[:, 1] - root_pos[1]
        y_ok = torch.all((rel_y >= self.cfg.workspace_y_rel[0]) & (rel_y <= self.cfg.workspace_y_rel[1]))
        if not bool(y_ok):
            return False

        # 5. 末端参考轨迹在可达半径内（粗筛）
        total_offset = self.cfg.gun_length + self.env_spray_dists[env_id]
        ee_curve = curve + normals * total_offset
        ee_rel = ee_curve - root_pos.unsqueeze(0)
        ee_dist = torch.norm(ee_rel, dim=-1)
        reach_ok = torch.all(
            (ee_dist >= self.cfg.ee_reach_radius[0]) & (ee_dist <= self.cfg.ee_reach_radius[1])
        )
        if not bool(reach_ok):
            return False

        return True

    def _generate_single_bezier_curve(self, env_id: int):
        """为单个 env 生成一条满足约束的 Bézier 曲线。"""
        root_pos = self.robot.data.root_pos_w[env_id]
        device = self.device

        for _ in range(self.cfg.max_resample_attempts):
            desired_length = float(
                torch.empty(1, device=device).uniform_(*self.cfg.ranges.path_length).item()
            )
            num_samples = int(math.ceil(desired_length / self.cfg.point_spacing)) + 1
            num_samples = max(2, min(num_samples, self.max_num_samples))

            # 起点：机器人胸前约 0.5m，左右和高度只做小范围扰动
            p0 = root_pos.clone()
            p0[0] = root_pos[0] + self.cfg.start_x_forward
            p0[1] = root_pos[1] + torch.empty(1, device=device).uniform_(*self.cfg.start_y_offset_range).item()
            p0[2] = root_pos[2] + torch.empty(1, device=device).uniform_(*self.cfg.start_z_offset_range).item()

            # 主方向几乎在 y-z 平面，x 只有小漂移
            theta = torch.empty(1, device=device).uniform_(
                self.cfg.yz_heading_angle_range[0], self.cfg.yz_heading_angle_range[1]
            )[0]
            x_drift = torch.empty(1, device=device).uniform_(
                -self.cfg.max_x_drift_ratio, self.cfg.max_x_drift_ratio
            )[0]

            tangent = torch.tensor(
                [x_drift.item(), math.cos(theta.item()), math.sin(theta.item())],
                device=device,
                dtype=torch.float32,
            )
            tangent = safe_normalize(tangent, dim=-1)

            # y-z 平面内的“弯曲方向”
            bend_dir = torch.tensor([0.0, -tangent[2].item(), tangent[1].item()], device=device)
            bend_dir = safe_normalize(bend_dir, dim=-1)

            bend_scale = desired_length * self.cfg.curvature_ratio
            b1 = torch.empty(1, device=device).uniform_(-bend_scale, bend_scale)[0]
            b2 = torch.empty(1, device=device).uniform_(-bend_scale, bend_scale)[0]

            p1 = p0 + tangent * (desired_length / 3.0) + bend_dir * b1
            p2 = p0 + tangent * (2.0 * desired_length / 3.0) + bend_dir * b2
            p3 = p0 + tangent * desired_length

            t = torch.linspace(0.0, 1.0, num_samples, device=device).view(-1, 1)
            u = 1.0 - t

            curve = (
                (u ** 3) * p0.unsqueeze(0)
                + 3.0 * (u ** 2) * t * p1.unsqueeze(0)
                + 3.0 * u * (t ** 2) * p2.unsqueeze(0)
                + (t ** 3) * p3.unsqueeze(0)
            )

            deltas = curve[1:] - curve[:-1]
            seg_lens = torch.norm(deltas, dim=-1)
            actual_length = float(seg_lens.sum().item())

            arc = torch.zeros(num_samples, device=device)
            arc[1:] = torch.cumsum(seg_lens, dim=0)

            normals = self._sample_smooth_normals(num_samples)

            if self._is_curve_valid(env_id, curve, normals, actual_length, desired_length):
                return curve, normals, arc, actual_length, num_samples

        # fallback：生成一条非常保守的直线/弱弯曲轨迹
        desired_length = float(sum(self.cfg.ranges.path_length) * 0.5)
        num_samples = int(math.ceil(desired_length / self.cfg.point_spacing)) + 1
        num_samples = max(2, min(num_samples, self.max_num_samples))

        p0 = root_pos.clone()
        p0[0] = root_pos[0] + self.cfg.start_x_forward
        p0[1] = root_pos[1]
        p0[2] = root_pos[2] + sum(self.cfg.start_z_offset_range) * 0.5

        tangent = torch.tensor([0.0, 0.0, 1.0], device=device)
        p1 = p0 + torch.tensor([0.0, 0.0, desired_length / 3.0], device=device)
        p2 = p0 + torch.tensor([0.0, 0.0, 2.0 * desired_length / 3.0], device=device)
        p3 = p0 + tangent * desired_length

        t = torch.linspace(0.0, 1.0, num_samples, device=device).view(-1, 1)
        u = 1.0 - t
        curve = (
            (u ** 3) * p0.unsqueeze(0)
            + 3.0 * (u ** 2) * t * p1.unsqueeze(0)
            + 3.0 * u * (t ** 2) * p2.unsqueeze(0)
            + (t ** 3) * p3.unsqueeze(0)
        )

        deltas = curve[1:] - curve[:-1]
        seg_lens = torch.norm(deltas, dim=-1)
        arc = torch.zeros(num_samples, device=device)
        arc[1:] = torch.cumsum(seg_lens, dim=0)
        actual_length = float(seg_lens.sum().item())
        normals = self._sample_smooth_normals(num_samples)

        return curve, normals, arc, actual_length, num_samples

    def _generate_bezier_curves(self, env_ids: torch.Tensor):
        """为一批 env 生成路径，并写入 buffer。"""
        for env_id in env_ids.tolist():
            curve, normals, arc, total_length, num_samples = self._generate_single_bezier_curve(env_id)

            # 先清空
            self.path_points_w[env_id].zero_()
            self.path_normals_w[env_id].zero_()
            self.path_arc_lengths[env_id].zero_()
            self.path_ee_targets[env_id].zero_()

            # 写入有效区间
            self.path_points_w[env_id, :num_samples] = curve
            self.path_normals_w[env_id, :num_samples] = normals
            self.path_arc_lengths[env_id, :num_samples] = arc

            # 尾部用最后一个点/法向/弧长填充，方便后续张量操作
            if num_samples < self.max_num_samples:
                self.path_points_w[env_id, num_samples:] = curve[-1]
                self.path_normals_w[env_id, num_samples:] = normals[-1]
                self.path_arc_lengths[env_id, num_samples:] = arc[-1]

            self.path_num_samples[env_id] = num_samples
            self.path_total_lengths[env_id] = total_length
            self.path_completed[env_id] = False

            total_offset = self.cfg.gun_length + self.env_spray_dists[env_id]
            ee_curve = curve + normals * total_offset
            self.path_ee_targets[env_id, :num_samples] = ee_curve
            if num_samples < self.max_num_samples:
                self.path_ee_targets[env_id, num_samples:] = ee_curve[-1]

        if self.cfg.debug_vis:
            self.path_markers.visualize(self.path_ee_targets.reshape(-1, 3))

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

        # [补丁] 记录重置时的 base 前向，默认环境初始化是面向 +X 的
        forward_w = math_utils.quat_apply(self.robot.data.root_quat_w[env_ids],
                                          torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1))
        self.base_forward_ref_w[env_ids] = forward_w[:, :2] / (
                    torch.norm(forward_w[:, :2], dim=-1, keepdim=True) + 1e-8)

        # 生成新 path
        self._generate_bezier_curves(env_ids)

        # 不再 teleport base，保持默认 reset 姿态
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
        arc = self.path_arc_lengths[env_ids]  # (N, max_num_samples)
        max_s = self.path_total_lengths[env_ids]

        # clamp 到合法范围
        target_s = torch.clamp(target_s, min=0.0)
        target_s = torch.minimum(target_s, torch.clamp(max_s - 1e-6, min=0.0))

        # 找到第一个 >= target_s 的索引
        mask = arc >= target_s.unsqueeze(1)
        idx1 = torch.argmax(mask.to(torch.int64), dim=1)

        # 保证 idx1 至少为 1，避免 idx0 = idx1 = 0
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

        if self.cfg.debug_vis:
            vis_points = torch.cat(vis_points, dim=0)  # 4N x 3
            self.target_markers.visualize(vis_points)

        # 计算 Base 系下的当前路径切线 (Rewards 中需要用到)
        p_curr, _ = self._get_interpolated_target(env_ids, curr_s)
        p_next, _ = self._get_interpolated_target(env_ids, curr_s + 0.05)

        # 世界系切线方向
        tangent_w = safe_normalize(p_next - p_curr, dim=-1)
        # 转到 Base 系下并存储
        self.current_tangent_b[env_ids] = math_utils.quat_apply_inverse(root_quat_w, tangent_w)

        if self.cfg.debug_vis:
            vis_points = torch.cat(vis_points, dim=0)  # 4N x 3
            self.target_markers.visualize(vis_points)

    @property
    def command(self) -> torch.Tensor:
        return self.command_b

    @property
    def is_completed(self) -> torch.Tensor:
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

    # 路径离散间距：1cm
    point_spacing: float = 0.01

    # 4 个参考点，间隔 5cm
    num_lookahead_points: int = 4
    lookahead_spacing: float = 0.05

    # 法向噪声（只加在 y/z）
    normal_noise_yz: float = 0.02

    # 起点设置：机器人胸前约 0.5m
    start_x_forward: float = 0.50
    start_y_offset_range: tuple[float, float] = (-0.08, 0.08)
    start_z_offset_range: tuple[float, float] = (0.35, 0.55)  # 相对 root z

    # path 主要在 y-z 平面，x 只允许小扰动
    max_x_drift_ratio: float = 0.05   # tangent 的 x 分量很小
    curvature_ratio: float = 0.08     # 轻微弯曲
    yz_heading_angle_range: tuple[float, float] = (-1.25, 1.25)

    # workspace 限制
    workspace_x_rel: tuple[float, float] = (0.35, 0.75)   # 相对 root x
    workspace_y_rel: tuple[float, float] = (-1.10, 1.10)  # 相对 root y
    workspace_z: tuple[float, float] = (0.50, 2.20)       # 世界 z 高度范围

    # 粗略可达半径约束（针对 ee target）
    ee_reach_radius: tuple[float, float] = (0.35, 1.30)

    # Bézier 实际长度允许与目标长度有一点偏差
    length_tolerance_ratio: tuple[float, float] = (0.85, 1.15)

    # 认为 path 走完的阈值
    path_finish_epsilon: float = 1e-3

    # rejection sampling 次数
    max_resample_attempts: int = 64

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

    debug_vis: bool = True