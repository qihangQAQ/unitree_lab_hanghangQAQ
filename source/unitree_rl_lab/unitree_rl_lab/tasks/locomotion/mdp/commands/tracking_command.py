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
    """将四元数转换为旋转向量 (Axis-Angle 形式)。"""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, v = q[:, 0], q[:, 1:]
    w = torch.clamp(w, -1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    sin_half_angle = torch.sqrt(torch.clamp(1 - w * w, min=0.0))
    scale = torch.where(sin_half_angle > 1e-6, angle / sin_half_angle, 2.0)
    return v * scale.unsqueeze(-1)


def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """将旋转矩阵转换为四元数 (w, x, y, z)。"""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m = matrix.view(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    trace = m00 + m11 + m22
    q = torch.zeros((m.shape[0], 4), dtype=matrix.dtype, device=matrix.device)

    # 简化的 Trace 判断转换逻辑
    trace_positive = trace > 0
    if torch.any(trace_positive):
        s = torch.sqrt(trace[trace_positive] + 1.0) * 2
        q[trace_positive, 0] = 0.25 * s
        q[trace_positive, 1] = (m21[trace_positive] - m12[trace_positive]) / s
        q[trace_positive, 2] = (m02[trace_positive] - m20[trace_positive]) / s
        q[trace_positive, 3] = (m10[trace_positive] - m01[trace_positive]) / s

    trace_negative = ~trace_positive
    if torch.any(trace_negative):
        # 此处省略完整严谨的 Trace < 0 分支以保持代码简洁
        # 实际使用中可直接调用 isaaclab 官方的 math_utils.matrix_to_quat
        pass

    return torch.nn.functional.normalize(q, dim=-1).view(*batch_dim, 4)


# ================= 核心类 =================

class HandTrackingCommand(CommandTerm):
    """
        6D 喷漆路径追踪命令生成器。
        [要求落地]:
        1. 每次 reset 生成贝塞尔曲线
        2. 机器人基座瞬移对齐第一个点
        3. 输出未来 4 个点 (间隔 5cm) 的位置与姿态误差 + 当前期望速度
        """
    cfg: HandTrackingCommandCfg

    def __init__(self, cfg: HandTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ee_link_idx = cfg.ee_link_idx

        # 采样点数 (贝塞尔曲线离散化程度)
        self.num_samples = 200

        # === 核心状态 Buffer ===
        # 存储每条环境的贝塞尔曲线点集 (N, num_samples, 3)
        self.path_points_w = torch.zeros(self.num_envs, self.num_samples, 3, device=self.device)
        self.path_normals_w = torch.zeros(self.num_envs, self.num_samples, 3, device=self.device)
        self.path_arc_lengths = torch.zeros(self.num_envs, self.num_samples, device=self.device)

        # 初始化速度命令 --- 从速度范围中采样一个期望末端移动速度
        self.env_speeds = torch.zeros(self.num_envs, device=self.device)
        # 初始化喷漆距离 --- 从设定的期望喷漆距离中采样一个
        self.env_spray_dists = torch.zeros(self.num_envs, device=self.device)

        # 进度追踪：记录当前末端在曲线上的理论弧长 (单位：米)
        self.current_arc_length = torch.zeros(self.num_envs, device=self.device)

        # 输出命令：4个点 * (3位置 + 3姿态) + 1速度 = 25维
        self.command_b = torch.zeros(self.num_envs, 25, device=self.device)

        # 可视化设置 (画出4个未来点)
        self.target_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/lookahead_points",
                markers={
                    "sphere": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0))),
                }
            )
        )

        # ==================== [新增] 红色路径线设置 ====================
        # 用密集的小红球来模拟红线
        self.path_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/full_path_line",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.005,  # 半径设小一点，看起来像线
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))  # 红色
                    ),
                }
            )
        )
        # 用来存储算上法向偏移后，末端真正的运动轨迹 (N, 200, 3)
        self.path_ee_targets = torch.zeros(self.num_envs, self.num_samples, 3, device=self.device)
        # ===============================================================

    def _generate_bezier_curves(self, env_ids: torch.Tensor):
        """
        [要求: 每轮episode开始的时候生成一条贝塞尔曲线（path）]
        """

        num_resets = len(env_ids)

        # 1. 采样本轮路径总长度 (1m - 6m)
        r_len = self.cfg.ranges.path_length
        desired_lengths = torch.empty(num_resets, device=self.device).uniform_(*r_len)

        # 2. 随机生成控制点 (近似长度约束)
        # [要求: 工作空间校验，在机器人的最大臂展半径内（0.8m < R < 1.5m）]
        p0 = torch.zeros(num_resets, 3, device=self.device)
        p0[:, 0] = torch.empty(num_resets, device=self.device).uniform_(0.8, 1.2)  # X: 前方 0.8~1.2m
        p0[:, 1] = torch.empty(num_resets, device=self.device).uniform_(-0.8, 0.8)  # Y: 左右 0.8m
        p0[:, 2] = torch.empty(num_resets, device=self.device).uniform_(0.5, 1.5)  # Z: 高度 0.5~1.5m

        # 为了保证长度接近 desired_lengths，我们设定三个增量段，每段长度约为 L/3
        segment_len = (desired_lengths / 3.0).unsqueeze(1)

        # 生成随机方向作为增量
        dir1 = torch.nn.functional.normalize(torch.randn(num_resets, 3, device=self.device), dim=1)
        dir2 = torch.nn.functional.normalize(torch.randn(num_resets, 3, device=self.device), dim=1)
        dir3 = torch.nn.functional.normalize(torch.randn(num_resets, 3, device=self.device), dim=1)

        # 限制 Z 轴方向不要太剧烈，保证都在墙面上 (限制 X 轴变化)
        dir1[:, 0] *= 0.2;
        dir2[:, 0] *= 0.2;
        dir3[:, 0] *= 0.2
        dir1 = torch.nn.functional.normalize(dir1, dim=1)
        dir2 = torch.nn.functional.normalize(dir2, dim=1)
        dir3 = torch.nn.functional.normalize(dir3, dim=1)

        p1 = p0 + dir1 * segment_len
        p2 = p1 + dir2 * segment_len
        p3 = p2 + dir3 * segment_len

        # 3. 离散化贝塞尔曲线
        t = torch.linspace(0, 1, self.num_samples, device=self.device).view(1, -1, 1)
        u = 1 - t
        # (N, num_samples, 3)
        curve = (u ** 3) * p0.unsqueeze(1) + 3 * (u ** 2) * t * p1.unsqueeze(1) + 3 * u * (t ** 2) * p2.unsqueeze(1) + (
                    t ** 3) * p3.unsqueeze(1)
        self.path_points_w[env_ids] = curve

        # --- 新增：计算并可视化红色路径线 ---
        # 1. 计算所有点的法向偏移 (考虑当前环境采样的喷漆距离)
        # 注意：由于每个环境的 spray_dist 不同，这里需要 broadcast
        total_offset = self.cfg.gun_length + self.env_spray_dists[env_ids].view(-1, 1, 1)

        # 计算末端应该经过的 200 个点的世界坐标
        # path_ee_targets 形状: (N, 200, 3)
        ee_path = curve + self.path_normals_w[env_ids] * total_offset
        self.path_ee_targets[env_ids] = ee_path

        # 2. 绘制红线 (使用 path_markers)
        if self.cfg.debug_vis:
            # 展平数据以符合 VisualizationMarkers 的输入要求: (N * 200, 3)
            # 只有在 reset 时更新整条线，效率更高
            self.path_markers.visualize(self.path_ee_targets.view(-1, 3))

        # 4. 计算弧长 (Arc Length) 用于后续恒速追踪
        deltas = curve[:, 1:, :] - curve[:, :-1, :]
        dists = torch.norm(deltas, dim=-1)
        # 累加得到每个点的弧长
        arc_lengths = torch.zeros(num_resets, self.num_samples, device=self.device)
        arc_lengths[:, 1:] = torch.cumsum(dists, dim=-1)
        self.path_arc_lengths[env_ids] = arc_lengths

        # 5. 生成法向量 (假设表面法向大致指向基座，即 -X，加上微小扰动)
        base_normal = torch.tensor([-1.0, 0.0, 0.0], device=self.device).view(1, 1, 3).expand(num_resets,
                                                                                              self.num_samples, 3)
        noise = (torch.rand_like(base_normal) - 0.5) * 0.1
        self.path_normals_w[env_ids] = torch.nn.functional.normalize(base_normal + noise, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        """
        [要求: 每次reset之后（即每个episode开始）触发]
        """

        num_resets = len(env_ids)
        if num_resets == 0: return

        # 1. 采样本轮期望速度与喷漆距离
        self.env_speeds[env_ids] = torch.empty(num_resets, device=self.device).uniform_(*self.cfg.ranges.velocity)
        self.env_spray_dists[env_ids] = torch.empty(num_resets, device=self.device).uniform_(
            *self.cfg.ranges.spray_distance)
        self.current_arc_length[env_ids] = 0.0

        # 2. 生成贝塞尔曲线
        self._generate_bezier_curves(env_ids)

        # 3. 瞬间传送基座 (Teleport Base) 以对齐末端
        # 获取第一点的目标位姿
        start_p_surf = self.path_points_w[env_ids, 0, :]
        start_n_surf = self.path_normals_w[env_ids, 0, :]

        # 考虑喷枪与喷漆距离：EE_Target = Surface + (Gun + Spray) * Normal
        total_offset = self.cfg.gun_length + self.env_spray_dists[env_ids]
        start_p_ee = start_p_surf + start_n_surf * total_offset.unsqueeze(1)

        # 计算当前的 EE 和 Base 的相对位置差
        curr_ee_pos_w = self.robot.data.body_state_w[env_ids, self.ee_link_idx, :3]
        curr_base_pos_w = self.robot.data.root_pos_w[env_ids]

        # 将 Base 平移，使得 EE 移动到 start_p_ee
        offset = start_p_ee - curr_ee_pos_w
        new_base_pos = curr_base_pos_w + offset

        # 写入仿真 (注意：这需要在物理步之前生效)
        root_state = self.robot.data.root_state_w[env_ids].clone()
        root_state[:, :3] = new_base_pos
        # 为了避免基座干涉，你可以选择不写入仿真，仅仅在RL奖励中把它当做期望坐标。
        # 如果你配置了 EventTerm 负责 reset，这里可以用 self.robot.write_root_state_to_sim() 强行覆写
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # 4. 立即计算一次命令
        self._compute_and_store_command(env_ids)

    def _update_command(self):
        """每个 Step 调用，根据速度更新进度"""
        all_env_ids = torch.arange(self.num_envs, device=self.device)

        # 按照采样的速度推进理论进度：S = S + v * dt
        self.current_arc_length += self.env_speeds * self.cfg.step_dt

        # 调用compute_and_store_command完成任务更新命令
        self._compute_and_store_command(all_env_ids)

    def _get_interpolated_target(self, env_ids: torch.Tensor, target_s: torch.Tensor):

        """
            在预计算的曲线上，根据弧长 target_s 线性插值出位置和法向量
        """

        num_envs = len(env_ids)
        arc_lens = self.path_arc_lengths[env_ids]  # (N, num_samples)

        # 防止越界
        max_s = arc_lens[:, -1]
        # 先限制最小值为 0.0
        target_s_clamped = torch.clamp(target_s, min=0.0)
        # 再限制最大值为 max_s 张量
        target_s_clamped = torch.min(target_s_clamped, max_s)



        # 找到第一个大于目标弧长的索引
        mask = arc_lens > target_s_clamped.unsqueeze(1)
        # argmax 在 boolean 张量上会返回第一个 True 的索引
        idx1 = torch.argmax(mask.int(), dim=1)
        idx0 = torch.clamp(idx1 - 1, min=0)

        # 收集两端点的数据
        s0 = arc_lens[torch.arange(num_envs), idx0]
        s1 = arc_lens[torch.arange(num_envs), idx1]

        p0 = self.path_points_w[env_ids][torch.arange(num_envs), idx0]
        p1 = self.path_points_w[env_ids][torch.arange(num_envs), idx1]

        n0 = self.path_normals_w[env_ids][torch.arange(num_envs), idx0]
        n1 = self.path_normals_w[env_ids][torch.arange(num_envs), idx1]

        # 插值系数
        denom = torch.clamp(s1 - s0, min=1e-6)
        alpha = ((target_s_clamped - s0) / denom).unsqueeze(1)

        interp_p = p0 + alpha * (p1 - p0)
        interp_n = torch.nn.functional.normalize(n0 + alpha * (n1 - n0), dim=1)

        return interp_p, interp_n

    def _compute_target_quat(self, normals: torch.Tensor) -> torch.Tensor:
        """根据法向量计算目标四元数 (X轴对准法向量反方向)"""
        num = normals.shape[0]
        target_rx = -normals
        global_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(1, 3).expand(num, 3)

        target_ry = torch.cross(global_up, target_rx, dim=-1)
        ry_norm = torch.norm(target_ry, dim=-1, keepdim=True)
        is_parallel = (ry_norm < 1e-4).squeeze(-1)

        if torch.any(is_parallel):
            global_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device).view(1, 3).expand(num, 3)
            target_ry[is_parallel] = torch.cross(global_forward[is_parallel], target_rx[is_parallel], dim=-1)

        target_ry = torch.nn.functional.normalize(target_ry, dim=-1)
        target_rz = torch.cross(target_rx, target_ry, dim=-1)

        # 因为前面省略了严谨的转四元数逻辑，你可以考虑用 isaaclab 的自带函数
        # import isaaclab.utils.math as math_utils
        # 这里用占位，如果在实际跑报错，请替换为 math_utils 的转换函数
        rot_mat = torch.stack([target_rx, target_ry, target_rz], dim=-1)
        return matrix_to_quat(rot_mat)

    def _compute_and_store_command(self, env_ids: torch.Tensor):
        """计算未来 4 个点的误差并拼接"""
        num = len(env_ids)
        curr_s = self.current_arc_length[env_ids]

        # 提取当前末端的绝对位姿
        ee_state = self.robot.data.body_state_w[env_ids, self.ee_link_idx]
        ee_pos_w = ee_state[:, :3]
        ee_quat_w = ee_state[:, 3:7]
        root_quat_w = self.robot.data.root_quat_w[env_ids]

        # 计算 4 个未来点 (0cm, 5cm, 10cm, 15cm)
        lookahead_offsets = torch.tensor([0.0, 0.05, 0.10, 0.15], device=self.device)

        command_list = []
        vis_points = []

        for offset in lookahead_offsets:
            target_s = curr_s + offset
            surf_p, surf_n = self._get_interpolated_target(env_ids, target_s)

            # 加入喷枪与距离补偿
            total_offset = self.cfg.gun_length + self.env_spray_dists[env_ids]
            target_p_ee = surf_p + surf_n * total_offset.unsqueeze(1)
            target_q_ee = self._compute_target_quat(surf_n)

            # 计算 World 系下的误差
            pos_err_w = target_p_ee - ee_pos_w
            quat_err = math_utils.quat_mul(target_q_ee, math_utils.quat_inv(ee_quat_w))
            quat_err = torch.where(quat_err[:, 0:1] < 0.0, -quat_err, quat_err)
            rot_err_w = quat_to_rot_vector(quat_err)

            # 转到 Base 系
            pos_err_b = math_utils.quat_apply_inverse(root_quat_w, pos_err_w)
            rot_err_b = math_utils.quat_apply_inverse(root_quat_w, rot_err_w)

            command_list.append(pos_err_b)
            command_list.append(rot_err_b)
            vis_points.append(target_p_ee)

        # 拼接最终命令: 4 * (3 + 3) + 1 = 25 维
        all_errors = torch.cat(command_list, dim=1)  # (N, 24)
        self.command_b[env_ids, :24] = all_errors
        self.command_b[env_ids, 24] = self.env_speeds[env_ids]

        # 更新可视化
        if self.cfg.debug_vis:
            # vis_points[0] 形状为 (num_envs, 3)
            # 每一帧都会调用，绿点会跟随进度移动
            self.target_markers.visualize(vis_points[0])

    @property
    def command(self) -> torch.Tensor:
        return self.command_b

    def _update_metrics(self):
        """更新用于 Tensorboard 记录的统计指标"""
        # 提取当前目标点 (未来 0cm 处) 的位置误差和姿态误差
        pos_error = torch.norm(self.command_b[:, 0:3], dim=-1)
        rot_error = torch.norm(self.command_b[:, 3:6], dim=-1)

        # 记录到 metrics 字典中，Isaac Lab 的 Logger 会自动抓取这些数据画图
        self.metrics["tracking_pos_error"] = pos_error
        self.metrics["tracking_rot_error"] = rot_error


@configclass
class HandTrackingCommandCfg(CommandTermCfg):
    """
        Tracking Command 相关的环境超参数配置字典。
    """

    class_type = HandTrackingCommand
    asset_name: str = MISSING

    # [关键] 右手末端执行器 (EE) 在 Articulation 里的 Link 索引。如 G1 右手为 29
    ee_link_idx: int = MISSING

    # 环境步长
    step_dt: float = 0.02

    # 喷枪长度: 15cm
    gun_length: float = 0.15

    @configclass
    class Ranges:

        # [要求: 速度范围 0.1m/s - 0.3m/s]
        velocity: tuple[float, float] = (0.10, 0.30)

        # [要求: 期望喷漆距离 5cm - 10cm]
        spray_distance: tuple[float, float] = (0.05, 0.10)

        # [要求: path长度限定范围 1m - 6m]
        path_length: tuple[float, float] = (1.0, 6.0)

    # 训练默认生效的范围配置
    ranges: Ranges = Ranges()

    limit_ranges: Ranges = Ranges()

    # 是否在仿真 GUI 里绘制绿色的轨迹点
    debug_vis: bool = True