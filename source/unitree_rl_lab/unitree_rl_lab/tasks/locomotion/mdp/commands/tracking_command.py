# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import glob
import torch
import numpy as np
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


def compute_target_quat_from_normal(normals: torch.Tensor, device: str) -> torch.Tensor:
    """
    根据法向量计算目标四元数 (X轴对准法向量的反方向)。
    假设: 工具坐标系 X 轴为喷射方向，同时这也是机器人基座的正前方面向。
    """
    num = normals.shape[0]
    target_rx = -normals  # 朝向平面内部
    global_up = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3).expand(num, 3)

    target_ry = torch.cross(global_up, target_rx, dim=-1)
    ry_norm = torch.norm(target_ry, dim=-1, keepdim=True)
    is_parallel = (ry_norm < 1e-4).squeeze(-1)

    # 如果法线刚好完全朝上或朝下
    if torch.any(is_parallel):
        global_forward = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3).expand(num, 3)
        target_ry[is_parallel] = torch.cross(global_forward[is_parallel], target_rx[is_parallel], dim=-1)

    target_ry = torch.nn.functional.normalize(target_ry, dim=-1)
    target_rz = torch.cross(target_rx, target_ry, dim=-1)

    rot_mat = torch.stack([target_rx, target_ry, target_rz], dim=-1)
    return matrix_to_quat(rot_mat)


def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """将旋转矩阵转换为四元数 (w, x, y, z)"""
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    
    w = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    x = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0))
    y = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0))
    z = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0))
    
    # 依靠 copysign 恢复正确的正负号，避免万向节死锁和除零错误
    x = torch.copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    y = torch.copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    z = torch.copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    
    return torch.stack([w, x, y, z], dim=-1)

    

# ================= 核心类 =================

class HandTrackingCommand(CommandTerm):
    """
    基于真实数据集的 6D 喷漆路径追踪命令生成器 (无 IK 瞬移版)。
    """
    cfg: HandTrackingCommandCfg

    def __init__(self, cfg: HandTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.ee_link_idx = cfg.ee_link_idx
        
        # 1. 加载本地打磨好的 v1 数据集到 GPU 显存
        self._load_dataset_to_vram()

        # 2. 每个 Env 当前分配的路径缓冲区 (支持最长 10 米)
        self.max_pts = 1000 
        self.env_path_points = torch.zeros(self.num_envs, self.max_pts, 3, device=self.device)
        self.env_path_normals = torch.zeros(self.num_envs, self.max_pts, 3, device=self.device)
        self.env_path_max_s = torch.zeros(self.num_envs, device=self.device)

        # 3. 环境状态变量
        self.env_speeds = torch.zeros(self.num_envs, device=self.device)
        self.env_spray_dists = torch.zeros(self.num_envs, device=self.device)
        self.current_arc_length = torch.zeros(self.num_envs, device=self.device)
        self.is_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 输出命令：4个参考点 * (3位置 + 3姿态误差) + 1期望速度 = 25维
        self.command_b = torch.zeros(self.num_envs, 25, device=self.device)

        # 4. 可视化
        if self.cfg.debug_vis:
            self.target_markers = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/lookahead_points",
                    markers={
                        "sphere": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))),
                    }
                )
            )

        #  ---- 新增 ----
        # 当前路径切线方向（base frame），用于 reward
        self.current_tangent_b = torch.zeros(self.num_envs, 3, device=self.device)

        # reset 时记录的 base 前向（world xy），用于抑制“侧身擦黑板”
        self.base_forward_ref_w = torch.zeros(self.num_envs, 2, device=self.device)

    def _load_dataset_to_vram(self):
        """一次性将所有 NPY 文件加载进显存"""
        print(f"[Command] 正在将数据集加载至显存: {self.cfg.dataset_dir}")
        files = glob.glob(os.path.join(self.cfg.dataset_dir, "*.npy"))
        assert len(files) > 0, "未找到数据集文件，请检查 dataset_dir 配置！"

        all_paths = []
        for f in files:
            data = np.load(f, allow_pickle=True).item()
            traj_key = "refined_traj_pred" if "refined_traj_pred" in data else "processed_traj_pred"
            if traj_key not in data: continue
            
            for batch_trajs in data[traj_key]:
                if len(batch_trajs) < 2: continue
                all_paths.append(batch_trajs)

        self.db_num_paths = len(all_paths)
        max_len = max([p.shape[0] for p in all_paths])
        
        self.db_points = torch.zeros(self.db_num_paths, max_len, 3, device=self.device)
        self.db_normals = torch.zeros(self.db_num_paths, max_len, 3, device=self.device)
        self.db_lengths = torch.zeros(self.db_num_paths, device=self.device)

        for i, p in enumerate(all_paths):
            length = p.shape[0]
            self.db_points[i, :length] = torch.tensor(p[:, 0:3], device=self.device)
            self.db_normals[i, :length] = torch.tensor(p[:, 3:6], device=self.device)
            self.db_lengths[i] = length * self.cfg.ds

        print(f"[Command] 成功入库 {self.db_num_paths} 条路径, 最大长度 {max_len} 点。")

    def _resample_command(self, env_ids: Sequence[int]):
        """
        每次 Episode Reset 触发：抽取轨迹 -> 归零 -> 旋转(法向对准机器人) -> 平移到机器人正前方
        """
        num_resets = len(env_ids)
        if num_resets == 0: return
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # 1. 随机抽取基础 Path
        sampled_idxs = torch.randint(0, self.db_num_paths, (num_resets,), device=self.device)
        raw_lens = self.db_lengths[sampled_idxs]

        # 2. 截断逻辑 (超长则随机截取 3~4m)
        for i, env_idx in enumerate(env_ids_tensor):
            db_idx = sampled_idxs[i]
            path_len = raw_lens[i].item()
            
            if path_len <= 5.0:
                start_idx = 0
                end_idx = int(path_len / self.cfg.ds)
                final_len = path_len
            else:
                target_len = np.random.uniform(3.0, 4.0)
                target_pts = int(target_len / self.cfg.ds)
                max_start_idx = int((path_len - target_len) / self.cfg.ds)
                start_idx = np.random.randint(0, max_start_idx + 1)
                end_idx = start_idx + target_pts
                final_len = target_len

            num_pts = end_idx - start_idx
            self.env_path_points[env_idx, :num_pts] = self.db_points[db_idx, start_idx:end_idx]
            self.env_path_normals[env_idx, :num_pts] = self.db_normals[db_idx, start_idx:end_idx]
            self.env_path_max_s[env_idx] = final_len

        # ================= 核心重构：把路径稳稳生成在机器人面前 =================
        # 1. 获取轨迹在数据集中的绝对起点和初始法向
        p0_orig = self.env_path_points[env_ids_tensor, 0, :].clone()
        n0_orig = self.env_path_normals[env_ids_tensor, 0, :].clone()

        # 2. 计算让墙面严格朝向机器人 (-X方向, 即 [-1, 0, 0]) 所需的水平旋转角 (Yaw)
        yaw_orig = torch.atan2(n0_orig[:, 1], n0_orig[:, 0])
        yaw_target = torch.full_like(yaw_orig, torch.pi) # 目标角度是 pi
        yaw_diff = yaw_target - yaw_orig

        # 构建只绕 Z 轴水平旋转的四元数
        rot_quat = torch.zeros(num_resets, 4, device=self.device)
        rot_quat[:, 0] = torch.cos(yaw_diff / 2.0)
        rot_quat[:, 3] = torch.sin(yaw_diff / 2.0)
        rot_quat_expanded = rot_quat.unsqueeze(1).expand(-1, self.max_pts, -1)

        # 3. 几何变换三步曲：归零 -> 旋转 -> 平移
        
        # [步骤 A]: 归零 (抽出轨迹的绝对坐标，消除投石机甩飞 Bug)
        self.env_path_points[env_ids_tensor] -= p0_orig.unsqueeze(1)
        
        # [步骤 B]: 旋转 (让整条路径跟着转，保证墙面的法向永远正对机器人)
        self.env_path_points[env_ids_tensor] = math_utils.quat_apply(rot_quat_expanded, self.env_path_points[env_ids_tensor])
        self.env_path_normals[env_ids_tensor] = math_utils.quat_apply(rot_quat_expanded, self.env_path_normals[env_ids_tensor])

        # [步骤 C]: 平移到机器人正前方
        root_pos_w = self.robot.data.root_pos_w[env_ids_tensor].clone()
        target_p0 = torch.zeros_like(p0_orig)
        
        # X 轴放到机器人正前方 (例如 default_ee_local_pos 的 0.4m)
        target_p0[:, 0] = root_pos_w[:, 0] + self.cfg.default_ee_local_pos[0]
        # Y 轴对齐机器人，并加上偏置 (例如 default_ee_local_pos 的 -0.2m)
        target_p0[:, 1] = root_pos_w[:, 1] + self.cfg.default_ee_local_pos[1]
        # 【灵魂操作】：Z 轴坚决使用真实数据集的绝对高度，这样机器人才会学下蹲或踮脚！
        target_p0[:, 2] = p0_orig[:, 2]

        # 最终写入平移
        self.env_path_points[env_ids_tensor] += target_p0.unsqueeze(1)
        # =========================================================================

        # 4. 采样喷漆速度和距离
        self.env_speeds[env_ids_tensor] = torch.empty(num_resets, device=self.device).uniform_(*self.cfg.ranges.velocity)
        self.env_spray_dists[env_ids_tensor] = torch.empty(num_resets, device=self.device).uniform_(*self.cfg.ranges.spray_distance)
        self.current_arc_length[env_ids_tensor] = 0.0
        self.is_completed[env_ids_tensor] = False

        # 5. 记录墙面参考前向 (用于抑制机器人侧身走)
        # 因为我们已经把法向强行旋转到了 -X，且你的 cfg 保证了机器人永远面朝 +X (yaw=0)
        # 所以机器人的理想前向就死死绑定在世界坐标系的 +X [1.0, 0.0]
        self.base_forward_ref_w[env_ids_tensor] = torch.tensor([1.0, 0.0], device=self.device).repeat(num_resets, 1)

        # 6. 计算初始命令
        self._compute_and_store_command(env_ids_tensor)

    def _update_command(self):
        """Step 推进：s = s + v * dt"""
        self.current_arc_length += self.env_speeds * self.cfg.step_dt
        self.is_completed = self.current_arc_length >= self.env_path_max_s

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._compute_and_store_command(all_env_ids)

    def _get_interpolated_target(self, env_ids: torch.Tensor, target_s: torch.Tensor):
        """根据弧长目标获取插值后的点和法向"""
        float_idx = target_s / self.cfg.ds
        max_idx_val = (self.env_path_max_s[env_ids] / self.cfg.ds) - 1.001
        
        float_idx_clamped = torch.clamp(float_idx, min=0.0)
        float_idx_clamped = torch.min(float_idx_clamped, max_idx_val)

        idx0 = torch.floor(float_idx_clamped).long()
        idx1 = idx0 + 1
        alpha = (float_idx_clamped - idx0).unsqueeze(1)

        p0 = self.env_path_points[env_ids, idx0, :]
        p1 = self.env_path_points[env_ids, idx1, :]
        n0 = self.env_path_normals[env_ids, idx0, :]
        n1 = self.env_path_normals[env_ids, idx1, :]

        interp_p = p0 + alpha * (p1 - p0)
        interp_n = torch.nn.functional.normalize(n0 + alpha * (n1 - n0), dim=1)

        return interp_p, interp_n

    def _compute_and_store_command(self, env_ids: torch.Tensor):
        """构建 25 维观测空间，并额外保存当前路径切线方向（base frame）"""

        curr_s = self.current_arc_length[env_ids]

        ee_state = self.robot.data.body_state_w[env_ids, self.ee_link_idx]
        ee_pos_w = ee_state[:, :3]
        ee_quat_w = ee_state[:, 3:7]
        root_quat_w = self.robot.data.root_quat_w[env_ids]

        lookahead_offsets = torch.tensor([0.0, 0.05, 0.10, 0.15], device=self.device)

        command_list = []
        vis_points = []

        # ===== 新增：专门保存第0个点和第1个点，用来构造当前路径切线 =====
        target_p_ee_0 = None
        target_p_ee_1 = None

        for k, offset in enumerate(lookahead_offsets):
            target_s = curr_s + offset
            surf_p, surf_n = self._get_interpolated_target(env_ids, target_s)

            # 喷枪长度 + 喷涂距离补偿
            total_offset = self.cfg.gun_length + self.env_spray_dists[env_ids]
            target_p_ee = surf_p + surf_n * total_offset.unsqueeze(1)
            target_q_ee = compute_target_quat_from_normal(surf_n, self.device)

            # ===== 新增：保存第0个和第1个参考点 =====
            if k == 0:
                target_p_ee_0 = target_p_ee
            elif k == 1:
                target_p_ee_1 = target_p_ee

            # -------------------------
            # World 系误差
            # -------------------------
            pos_err_w = target_p_ee - ee_pos_w

            quat_err = math_utils.quat_mul(
                target_q_ee, math_utils.quat_inv(ee_quat_w)
            )
            quat_err = torch.where(quat_err[:, 0:1] < 0.0, -quat_err, quat_err)
            rot_err_w = quat_to_rot_vector(quat_err)

            # -------------------------
            # 转到 Base 系（提高平移 / 旋转不变性）
            # -------------------------
            pos_err_b = math_utils.quat_apply_inverse(root_quat_w, pos_err_w)
            rot_err_b = math_utils.quat_apply_inverse(root_quat_w, rot_err_w)

            command_list.append(pos_err_b)
            command_list.append(rot_err_b)

            vis_points.append(target_p_ee)

        # 拼 24 维误差 + 1 维期望速度
        all_errors = torch.cat(command_list, dim=1)
        self.command_b[env_ids, :24] = all_errors
        self.command_b[env_ids, 24] = self.env_speeds[env_ids]

        # ===== 新增：计算当前路径切线方向（base frame）=====
        # 用 0cm 点 -> 5cm 点 的方向近似当前局部路径切线
        if target_p_ee_0 is not None and target_p_ee_1 is not None:
            tangent_w = target_p_ee_1 - target_p_ee_0                      # (N, 3)
            tangent_b = math_utils.quat_apply_inverse(root_quat_w, tangent_w)
            tangent_b = tangent_b / (torch.norm(tangent_b, dim=-1, keepdim=True) + 1e-8)
            self.current_tangent_b[env_ids] = tangent_b

        if self.cfg.debug_vis:
            self.target_markers.visualize(torch.cat(vis_points, dim=0))

    @property
    def command(self) -> torch.Tensor:
        return self.command_b

    def _update_metrics(self):
        """记录第一点 (0cm处) 的误差供 WandB 监控"""
        pos_error = torch.norm(self.command_b[:, 0:3], dim=-1)
        rot_error = torch.norm(self.command_b[:, 3:6], dim=-1)
        self.metrics["tracking_pos_error"] = pos_error
        self.metrics["tracking_rot_error"] = rot_error


@configclass
class HandTrackingCommandCfg(CommandTermCfg):
    class_type = HandTrackingCommand
    asset_name: str = MISSING
    ee_link_idx: int = MISSING
    
    # 数据集路径
    dataset_dir: str = "/tmp/spray_painting_paths_v1"
    
    # 路径离散间距
    ds: float = 0.01
    step_dt: float = 0.02
    gun_length: float = 0.15

    # 【重要新增】：待机姿态下，右手末端相对于基座的本地坐标系便宜量 (X前后, Y左右, Z上下)
    # 你可能需要根据 G1 的真实待机位姿微调这三个数字
    default_ee_local_pos: tuple[float, float, float] = (0.4, -0.2, 0.2)

    @configclass
    class Ranges:
        velocity: tuple[float, float] = (0.10, 0.40)
        spray_distance: tuple[float, float] = (0.05, 0.10)
        path_length: tuple[float, float] = (1.0, 6.0)

    ranges: Ranges = Ranges()
    limit_ranges: Ranges = Ranges()
    debug_vis: bool = True