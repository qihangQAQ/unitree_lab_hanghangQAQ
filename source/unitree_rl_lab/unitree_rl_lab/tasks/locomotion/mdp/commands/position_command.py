from __future__ import annotations


import math
from dataclasses import MISSING
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers,VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
# from isaaclab.envs import ManagerBasedEnv
import isaaclab.sim as sim_utils
# from .commands_cfg import UniformPositionCommandCfg
import os


# ================= 配置区 =================
# 指向你训练好的模型路径
MODEL_PATH = "/home/qihang/code_lab/unitree_rl_lab-main/logs/rsl_rl/ray_prediction/20260110-211721_resnet18_N20007/epoch50.pt"
# 训练时设定的最大距离（用于归一化输入）
MAX_DEPTH_RANGE = 6.0
# =========================================

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class UniformPositionCommand(CommandTerm):
    
    cfg : UniformPositionCommandCfg

    def __init__(self, cfg: UniformPositionCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # 期望位置设定
        self.position_targets = torch.zeros(self.num_envs, 3, device=self.device)
        # 初始位置设定
        self.env_origins = self.robot.data.root_pos_w.clone()

        # 命令设定：command: x_pos_error, y_pos_error,heading_error 
        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        

        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
 
        # 创建可视化球体
        self.goal_pos_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/goal_position",
                markers = {
                    "sphere":sim_utils.SphereCfg(
                        radius=0.08,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.2, 1.0)   # 粉色
                        ),
                    )
                }
            )
        )
        # 创建可视化圆锥体
        self.goal_heading_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/goal_heading",
                markers={
                    "cone": sim_utils.ConeCfg(
                        radius=0.05,    # 底半径
                        height=0.18,    # 圆锥长度（尖尖到圆底）
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.7, 0.2)  # 你可以改颜色
                        ),
                    )
                },
            )
        )
        # =======================修改：统一使用模型参数=====================
        self.ray_max_dist = MAX_DEPTH_RANGE  # 修改为配置的常量
        # ================================================================

        # --- 2D 射线参数设置 ---
        self.ray_max_dist = 6.0
        self.ray_num = 11
        # 生成 -45 到 45.1 度的 11 条射线
        self.ray_thetas = torch.linspace(-torch.pi / 4, torch.pi / 4 + 0.0017, self.ray_num, device=self.device)
        self.ray_obs = torch.ones(self.num_envs, self.ray_num, device=self.device) * self.ray_max_dist

        # 传感器安装偏置 (机器人的基座)
        self.ray_x0 = 0.1  # 基座前方 10cm
        self.ray_y0 = 0.0

        # 射线可视化 Marker (每条线上 8 个小球)
        self.ray_visualizer = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Command/ray_dots",
                markers={
                    "dot": sim_utils.SphereCfg(
                        radius=0.03,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))  # 青色
                    )
                }
            )
        )

        # =======================新增：加载神经网络模型=====================
        print(f"[RayPredictor] Loading ray prediction model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model path does not exist: {MODEL_PATH}")

        try:
            # 加载 TorchScript 模型并设为评估模式
            self.ray_model = torch.jit.load(MODEL_PATH, map_location=self.device)
            self.ray_model.eval()
            print("[RayPredictor] Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        # ================================================================
    
    # def _resample_command(self, env_ids: Sequence[int]):
    #     num = len(env_ids)
    #     if num == 0:
    #         return
    #     # === 1.随机位置生成 ====
    #
    #     pos1 = torch.empty(num, 1, device=self.device).uniform_(*self.cfg.ranges.pos_1)
    #     pos2 = torch.empty(num, 1, device=self.device).uniform_(*self.cfg.ranges.pos_2)
    #
    #     # 随机偏置位置生成
    #     if self.cfg.ranges.heading is not None:
    #         heading_offset = torch.empty(num, 1, device=self.device).uniform_(
    #             *self.cfg.ranges.heading
    #         )
    #     else:
    #         # 不用额外偏置就设为 0
    #         heading_offset = torch.zeros(num, 1, device=self.device)
    #
    #     # === 2.计算目标位置 position_targets (世界系) ===
    #
    #     # #这里写错了，这里用的是第一次的初始位置
    #     # 而不是每次reset之后，机器人的初始位置
    #     # env_origins = self.env_origins[env_ids]
    #     # 矫正
    #     env_origins = self.robot.data.root_pos_w[env_ids].clone()
    #
    #     # 计算期望位置相对初始的偏移
    #     self.position_targets[env_ids, 0:1] = env_origins[:, 0:1] + pos1
    #     self.position_targets[env_ids, 1:2] = env_origins[:, 1:2] + pos2
    #     self.position_targets[env_ids, 2] = env_origins[:, 2] + 0.0
    #
    #     # === 3) 计算 heading_target （世界系 yaw） ===
    #
    #     root_pos_w = self.robot.data.root_pos_w[env_ids, :3]
    #     pos_diff = self.position_targets[env_ids] - root_pos_w
    #
    #     # 指向目标点的“基础朝向”
    #     base_heading_to_target = torch.atan2(pos_diff[:, 1:2], pos_diff[:, 0:1])  # (num, 1)
    #
    #     # 最终目标 heading = 指向目标的朝向 + 一个随机偏置
    #     heading_target = base_heading_to_target + heading_offset
    #
    #     # wrap 到 [-pi, pi]
    #     self.heading_target[env_ids] = math_utils.wrap_to_pi(heading_target[:, 0])
    #
    #     # === 2. 画球 ===
    #     self._debug_vis_callback()
    #
    #     # === 3. 画圆锥箭头（期望朝向） ===
    #     self._debug_vis_heading_callback()
    #
    #     # === 4) 设定位置命令 ===
    #     # 写到update_command中
    #
    #     # # 世界系下的当前位置 / 姿态
    #     # root_pos_w = self.robot.data.root_pos_w[env_ids, :3]    # (n, 3)
    #     # root_quat_w = self.robot.data.root_quat_w[env_ids]      # (n, 4)
    #     # current_heading = self.robot.data.heading_w[env_ids]    # (n,)
    #
    #     # # 位置误差（世界系）
    #     # pos_diff_w = self.position_targets[env_ids] - root_pos_w  # (n, 3)
    #
    #     # # 转到 base frame
    #     # pos_diff_b = math_utils.quat_rotate_inverse(root_quat_w, pos_diff_w)  # (n, 3)
    #
    #     # # 写入 x, y 误差
    #     # self.pos_command_b[env_ids, 0:2] = pos_diff_b[:, 0:2]
    #
    #     # # heading 误差
    #     # heading_err = math_utils.wrap_to_pi(self.heading_target[env_ids] - current_heading)
    #     # self.pos_command_b[env_ids, 2] = heading_err

    def _resample_command(self, env_ids: Sequence[int]):
        num = len(env_ids)
        if num == 0:
            return

        # 1. 采样基础位置和朝向 (回归到 1-7m 的稳定范围)
        pos1 = torch.empty(num, 1, device=self.device).uniform_(*self.cfg.ranges.pos_1)
        pos2 = torch.empty(num, 1, device=self.device).uniform_(*self.cfg.ranges.pos_2)

        # 2. 锁定起点(当前机器人位置)
        root_pos_w = self.robot.data.root_pos_w[env_ids].clone()
        env_origins = root_pos_w.clone()

        # 3. 计算目标位置并更新 Buffer
        target_pos_w = torch.zeros(num, 3, device=self.device)
        target_pos_w[:, 0] = env_origins[:, 0] + pos1.squeeze()
        target_pos_w[:, 1] = env_origins[:, 1] + pos2.squeeze()
        target_pos_w[:, 2] = env_origins[:, 2]
        self.position_targets[env_ids] = target_pos_w

        # 更新朝向目标
        pos_diff = self.position_targets[env_ids] - root_pos_w[:, :3]
        base_heading = torch.atan2(pos_diff[:, 1], pos_diff[:, 0])
        self.heading_target[env_ids] = math_utils.wrap_to_pi(base_heading)

        # 4. 障碍物高级生成逻辑 (含 2m 安全区检测)
        obs_names = ["obstacle_box_0", "obstacle_cylinder_0","obstacle_cylinder_1"
                     "obstacle_cylinder_2","obstacle_sphere_0","obstacle_cone_0"]

        # 预估一个通用的障碍物半径 (用于计算偏移量)，取平均值约 0.35m
        AVG_OBS_RADIUS = 0.35

        for name in obs_names:
            if name not in self._env.scene.keys(): continue
            obj = self._env.scene[name]

            # 生成概率随机数 [0, 1)
            prob = torch.rand(num, device=self.device)

            # --- 初始化位置容器 ---
            final_pos_2d = torch.zeros(num, 2, device=self.device)

            # =========================================================
            # Case 1: 20% 概率 - 障碍物完全不在路径上 (Off Path)
            # =========================================================
            mask_off = prob < 0.2
            if mask_off.any():
                # 全场随机撒点：x[-2, 22], y[-6, 6]
                n_off = mask_off.sum()
                final_pos_2d[mask_off, 0] = (torch.rand(n_off, device=self.device) * 24.0) - 2.0
                final_pos_2d[mask_off, 1] = (torch.rand(n_off, device=self.device) * 12.0) - 6.0

            # =========================================================
            # 计算路径基础信息 (供 Case 2 和 Case 3 使用)
            # =========================================================
            mask_on_path = ~mask_off  # 剩下的 80% 都是和路径有关的
            if mask_on_path.any():
                # 采样路径上的进度 t (0.2 ~ 0.8)，避免太靠近起点或终点
                n_on = mask_on_path.sum()
                t = torch.rand(n_on, 1, device=self.device) * 0.6 + 0.2

                # 路径向量
                start_p = env_origins[mask_on_path, :2]
                end_p = target_pos_w[mask_on_path, :2]
                path_vec = end_p - start_p

                # 路径上的基础点
                base_pos = start_p + path_vec * t

                # 计算路径的单位法向量 (用于施加横向偏移)
                # path_vec: (dx, dy) -> normal: (-dy, dx)
                path_len = torch.norm(path_vec, dim=1, keepdim=True) + 1e-6
                normal_vec = torch.cat([-path_vec[:, 1:2], path_vec[:, 0:1]], dim=1) / path_len

                # =========================================================
                # Case 2: 20% 概率 (0.2 <= p < 0.4) - 全部出现在路径上 (Full Block)
                # =========================================================
                # 逻辑：横向偏移很小，障碍物中心几乎就在连线上
                mask_full = (prob >= 0.2) & (prob < 0.4)
                # 注意：mask_full 是全局的，需要映射到 mask_on_path 的子集
                # 这里为了代码简洁，我们直接操作 final_pos_2d，利用 mask_full 索引

                if mask_full.any():
                    # 重新获取对应的 base_pos 和 normal (需要切片对应)
                    # 这种切片稍微麻烦，为了方便，我们可以直接对所有 mask_on_path 计算偏移，然后按 mask 赋值
                    pass

                    # =========================================================
                # Case 3: 60% 概率 (0.4 <= p < 1.0) - 一部分出现在路径上 (Partial Block)
                # =========================================================
                # 逻辑：横向偏移 ≈ 障碍物半径。这样障碍物中心偏离路径，但边缘挡住路径。
                # 随机向左偏或向右偏

                # --- 统一计算偏移量 ---
                # 1. 基础偏移：如果是 Full Block，偏移为 0；如果是 Partial，偏移为 Radius
                #    这里我们用 torch.where

                # 在 mask_on_path 的范围内区分 Full 和 Partial
                # 提取 mask_on_path 对应的 prob
                probs_on = prob[mask_on_path]
                is_full = probs_on < 0.4  # 在 on_path (p>=0.2) 的前提下，p<0.4 是 Full

                offset_mag = torch.where(
                    is_full,
                    torch.zeros_like(probs_on),  # Full: 偏移 0
                    torch.tensor(AVG_OBS_RADIUS + 0.1, device=self.device)  # Partial: 偏移 半径+一点余量
                )

                # 2. 加上随机扰动 (Jitter)
                # Full: 允许 +/- 0.1m 的微小误差
                # Partial: 允许 +/- 0.1m 的浮动，让“遮挡多少”有变化
                jitter = (torch.rand_like(probs_on) - 0.5) * 0.2
                offset_mag += jitter

                # 3. 随机左右方向 (Sign)
                sign = torch.sign(torch.rand_like(probs_on) - 0.5)

                # 4. 计算最终偏移向量
                offset_vec = normal_vec * (offset_mag * sign).unsqueeze(1)

                # 5. 赋值
                final_pos_2d[mask_on_path] = base_pos + offset_vec

            # =========================================================
            # 安全性检查 (Safety Check)
            # =========================================================
            # 如果生成的点离机器人太近 (<2m)，强制扔到远处的随机位置
            dist_to_robot = torch.norm(final_pos_2d - env_origins[:, :2], dim=1)
            is_unsafe = dist_to_robot < 2.0

            if is_unsafe.any():
                n_unsafe = is_unsafe.sum()
                # 重新随机撒点
                safe_random_x = (torch.rand(n_unsafe, device=self.device) * 24.0) - 2.0
                safe_random_y = (torch.rand(n_unsafe, device=self.device) * 12.0) - 6.0
                final_pos_2d[is_unsafe, 0] = safe_random_x
                final_pos_2d[is_unsafe, 1] = safe_random_y

            # =========================================================
            # 写入物理状态
            # =========================================================
            root_state = obj.data.default_root_state[env_ids].clone()

            # 写入 XY
            root_state[:, 0] = final_pos_2d[:, 0]
            root_state[:, 1] = final_pos_2d[:, 1]

            # 写入 Z (高度)
            # 根据物体形状调整中心高度
            if "Box" in name:
                z_h = 0.25  # 假设Box高0.5
            elif "Cylinder" in name:
                z_h = 0.4  # 假设Cylinder高0.8左右
            elif "Sphere" in name:
                z_h = 0.35  # 半径0.35
            elif "Cone" in name:
                z_h = 0.4  # 高0.8
            else:
                z_h = 0.5

            # 如果被扔到地下(例如初始化未ready时)，这里保持在地面上
            root_state[:, 2] = env_origins[:, 2] + z_h

            obj.write_root_state_to_sim(root_state, env_ids=env_ids)

        # 记得更新 _update_command 里的 obs_names 列表，和上面保持一致！
        self._debug_vis_callback()
        self._debug_vis_heading_callback()



    def _update_command(self):
        """每个 step 调用：根据当前状态实时计算 [x_err, y_err, heading_err]."""
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        root_pos_w = self.robot.data.root_pos_w[env_ids, :3]
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        current_heading = self.robot.data.heading_w[env_ids]

        pos_diff_w = self.position_targets[env_ids] - root_pos_w
        # pos_diff_b = math_utils.quat_rotate_inverse(root_quat_w, pos_diff_w)
        pos_diff_b = math_utils.quat_apply_inverse(root_quat_w, pos_diff_w)

        self.pos_command_b[env_ids, 0:2] = pos_diff_b[:, 0:2]

        heading_err = math_utils.wrap_to_pi(self.heading_target[env_ids] - current_heading)
        self.pos_command_b[env_ids, 2] = heading_err

        # 如果有 standing_env，可以在这里置 0
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.pos_command_b[standing_env_ids, :] = 0.0

        # # --- 手搓射线逻辑 ---
        # # 1. 初始化距离
        # self.ray_obs[:] = self.ray_max_dist
        #
        # # 2. 遍历场景中的障碍物 (Box 和 Cylinder)
        # obs_names = ["obstacle_box_0", "obstacle_cylinder_0","obstacle_cylinder_1",
        #              "obstacle_cylinder_2","obstacle_sphere_0","obstacle_cone_0"]
        # for name in obs_names:
        #     if name not in self._env.scene.keys(): continue
        #     obj = self._env.scene[name]
        #
        #     # 获取物体相对于机器人的位置 (Local Frame)
        #     obj_pos_w = obj.data.root_pos_w[:, :3]
        #     obj_rel_pos_b = math_utils.quat_apply_inverse(self.robot.data.root_quat_w,
        #                                                   obj_pos_w - self.robot.data.root_pos_w[:, :3])
        #
        #     # 调用数学查询 (简化处理：Box 也当做半径 0.4 的圆柱处理，或根据需求微调半径)
        #     radius = 0.4 if "box" in name else 0.25
        #     dist = self.circle_ray_query(self.ray_x0, self.ray_y0, self.ray_thetas, obj_rel_pos_b[:, :2], radius,
        #                                  self.ray_max_dist)
        #
        #     # 取最近距离
        #     self.ray_obs = torch.minimum(self.ray_obs, dist)

        # =======================新增：使用模型预测代替手搓射线=====================
        # 1. 获取深度相机数据 (假设cfg中名字叫 depth_camera)
        # 如果当前仿真步还没有数据，直接返回，保持 ray_obs 为初始值
        if "depth_camera" not in self._env.scene.sensors:
            # 如果你改了cfg里的名字，这里会报错，请确保 position_env_cfg.py 里有 depth_camera
            return

        depth_sensor = self._env.scene.sensors["depth_camera"]
        # 数据通常是 (num_envs, Height, Width) 或 (num_envs, H, W, 1)
        if "distance_to_image_plane" not in depth_sensor.data.output:
            return

        raw_depth = depth_sensor.data.output["distance_to_image_plane"]

        # 2. 数据预处理 (必须严格匹配 train_ray_final.py 中的逻辑)
        # 步骤A: 处理 Inf (将天空/无限远 设为 MAX)
        # 训练脚本中 CamrecFolderDataset 使用 np.nan_to_num(..., posinf=MAX)
        depth_proc = torch.nan_to_num(raw_depth, posinf=MAX_DEPTH_RANGE, neginf=0.0, nan=0.0)

        # 步骤B: 归一化到 [0, 1] (depth = clip(d, 0, MAX) / MAX)
        depth_proc = torch.clamp(depth_proc, 0.0, MAX_DEPTH_RANGE) / MAX_DEPTH_RANGE

        # 步骤C: 维度调整，模型输入需要 (N, 1, H, W)
        if depth_proc.ndim == 3:  # (N, H, W)
            depth_proc = depth_proc.unsqueeze(1)
        elif depth_proc.ndim == 4:  # (N, H, W, 1) -> (N, 1, H, W)
            depth_proc = depth_proc.permute(0, 3, 1, 2)

        # 3. 模型推理
        with torch.no_grad():
            # 模型输出是原始米数 (训练脚本中 label y 没有除以 6.0，所以输出不需要反归一化)
            pred_rays = self.ray_model(depth_proc)

        # 4. 更新观测 buffer
        # 截断一下结果，保证在合理范围内
        self.ray_obs[:] = torch.clamp(pred_rays, 0.0, MAX_DEPTH_RANGE)

        # 5. 调用原来的可视化，这样你就能看到模型预测出的“点”在哪里了
        self._debug_vis_rays_callback()


        # 调用可视化
        self._debug_vis_rays_callback()



    def _update_metrics(self):
        """根据当前状态更新指标（比如位置误差和朝向误差）。"""
        # 世界系下当前位置 / 姿态
        root_pos_w = self.robot.data.root_pos_w[:, :3]  # (num_envs, 3)
        current_heading = self.robot.data.heading_w  # (num_envs,)

        # 位置误差（世界系）
        pos_diff_w = self.position_targets - root_pos_w  # (num_envs, 3)
        pos_err_xy = torch.norm(pos_diff_w[:, :2], dim=-1)  # (num_envs,)

        # 朝向误差
        heading_err = math_utils.wrap_to_pi(self.heading_target - current_heading)  # (num_envs,)
        heading_err_abs = torch.abs(heading_err)

        # 可以归一化一下，比如除以 max_goal_time_s，对标 velocity 里的“平均误差/时间”
        max_command_time = self.cfg.max_goal_time_s
        max_command_step = max_command_time / self._env.step_dt

        # 如果你希望 metrics 是“平均误差”，就累加再除以步数
        self.metrics["pos_err_xy"] = self.metrics.get("pos_err_xy", torch.zeros_like(pos_err_xy))
        self.metrics["heading_err"] = self.metrics.get("heading_err", torch.zeros_like(heading_err_abs))

        self.metrics["pos_err_xy"] += pos_err_xy / max_command_step
        self.metrics["heading_err"] += heading_err_abs / max_command_step

        
    # 对外暴露函数
    # (num_envs, 3) = [x_err, y_err, heading_err]."""
    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.pos_command_b
    
    # 画 球 函数
    def _debug_vis_callback(self, event=None):
        """在期望位置 self.position_targets 处画一个粉色球。"""
        # 机器人还没初始化就先退出，防止访问无效数据
        if not self.robot.is_initialized:
            return

        # 画目标位置球
        # 目标位置（世界坐标）：(num_envs, 3)
        pos_sphere = self.position_targets.clone()

        # 球的旋转：用单位四元数 [1, 0, 0, 0]，表示“无旋转”
        quat = torch.zeros(self.num_envs, 4, device=self.device)
        quat[:, 0] = 1.0

        # scale：这里用 (1,1,1)，因为已经在 marker 配置里定了球半径
        scales = torch.ones(self.num_envs, 3, device=self.device)

        # 画球
        self.goal_pos_marker.visualize(pos_sphere, quat, scales)

    # 画 箭头 函数
    def _debug_vis_heading_callback(self):
        """在目标球上方画一个圆锥，圆锥尖尖指向 heading_target（世界系 yaw）。"""
        if not self.robot.is_initialized:
            return

        # 圆锥位置：目标位置上方一点
        pos = self.position_targets.clone()
        pos[:, 2] += 0.20  # 你想离球更近/更远就调这个

        # 尺寸
        scales = torch.ones(self.num_envs, 3, device=self.device)

        # 目标朝向（世界系 yaw）
        yaw = self.heading_target  # shape: (num_envs,)

        # 1) q_yaw: 绕 Z 轴旋转 yaw
        quat_yaw = torch.zeros(self.num_envs, 4, device=self.device)
        quat_yaw[:, 0] = torch.cos(0.5 * yaw)
        quat_yaw[:, 3] = torch.sin(0.5 * yaw)

        # 2) 把“圆锥默认轴”转到水平面
        # 多数 primitive 的 Cone 默认是沿 +Z 轴“尖朝上”，我们希望它沿 +X 水平指向；
        # 所以再绕 Y 轴转 -90°（把 +Z 旋到 +X）
        pitch = torch.tensor(-0.5 * torch.pi, device=self.device)   # Tensor，不是 float

        quat_pitch = torch.zeros(self.num_envs, 4, device=self.device)
        quat_pitch[:, 0] = torch.cos(0.5 * pitch)
        quat_pitch[:, 2] = torch.sin(0.5 * pitch)  # 绕 Y

        # 3) 合成旋转：q = q_yaw * q_pitch（先把圆锥放平，再按 yaw 转向）
        w1, x1, y1, z1 = quat_yaw[:, 0], quat_yaw[:, 1], quat_yaw[:, 2], quat_yaw[:, 3]
        w2, x2, y2, z2 = quat_pitch[:, 0], quat_pitch[:, 1], quat_pitch[:, 2], quat_pitch[:, 3]
        quat = torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=1,
        )

        self.goal_heading_marker.visualize(pos, quat, scales)
        
        

    # 覆写compute 
    def compute(self, dt: float):
        """覆盖基类 CommandTerm.compute.

        - 不再使用 CommandTerm.time_left 做命令重采样
        - 一局内 goal 不变，只根据当前状态实时更新 [x_err, y_err, heading_err]
        """
        # 调用_update_metrics方法更新当前状态指标
        self._update_metrics()

        # 减时间
        self.time_left -= dt

        # 更新命令（位置误差）
        self._update_command()

    def circle_ray_query(self,ray_x0, ray_y0, ray_thetas, obj_rel_pos, radius, max_dist):
        """
        数学手搓：计算从 (ray_x0, ray_y0) 发出的射线与圆心在 obj_rel_pos、半径为 radius 的圆的交点距离
        """
        # 转换为相对于射线的坐标
        dx = obj_rel_pos[:, 0:1] - ray_x0
        dy = obj_rel_pos[:, 1:2] - ray_y0

        # 计算射线方向向量
        cos_t = torch.cos(ray_thetas)
        sin_t = torch.sin(ray_thetas)

        # 计算投影距离
        dist_projection = dx * cos_t + dy * sin_t

        # 计算垂直距离平方
        dist_perpendicular_sq = dx ** 2 + dy ** 2 - dist_projection ** 2

        # 检查是否相交
        radius_sq = radius ** 2
        intersect = (dist_perpendicular_sq <= radius_sq) & (dist_projection > 0)

        # 计算交点距离
        offset = torch.sqrt(torch.clamp(radius_sq - dist_perpendicular_sq, min=0))
        dist = dist_projection - offset

        # 裁剪范围
        return torch.where(intersect & (dist < max_dist), dist, torch.tensor(max_dist, device=dist.device))

    def _debug_vis_rays_callback(self):
        """在每条射线上沿距离画 8 个点"""
        if not self.robot.is_initialized: return

        # 每条射线画 8 个点，一共 num_envs * 11 * 8 个 Marker
        num_dots_per_ray = 8
        total_dots = self.num_envs * self.ray_num * num_dots_per_ray

        # 计算所有点在机器人坐标系下的位置
        # 生成线性的比例 [0.1, 0.2, ..., 1.0]
        dot_scales = torch.linspace(0.1, 1.0, num_dots_per_ray, device=self.device)

        # 计算点在 Body 系下的 XY
        # (num_envs, ray_num, 8)
        current_ray_dist = self.ray_obs.unsqueeze(-1) * dot_scales  # 距离按比例分布

        # 计算 X, Y (Body Frame)
        dot_x_b = self.ray_x0 + current_ray_dist * torch.cos(self.ray_thetas).unsqueeze(-1)
        dot_y_b = self.ray_y0 + current_ray_dist * torch.sin(self.ray_thetas).unsqueeze(-1)
        dot_z_b = torch.zeros_like(dot_x_b) + 0.1  # 离地 10cm

        dots_b = torch.stack([dot_x_b, dot_y_b, dot_z_b], dim=-1)  # (N, 11, 8, 3)

        # 转换到世界坐标系
        # 铺平为 (Total, 3) 方便计算
        dots_b_flat = dots_b.view(-1, 3)
        # 重复 root 状态以匹配点数
        root_pos_w = self.robot.data.root_pos_w.repeat_interleave(self.ray_num * num_dots_per_ray, dim=0)
        root_quat_w = self.robot.data.root_quat_w.repeat_interleave(self.ray_num * num_dots_per_ray, dim=0)

        dots_w = math_utils.quat_apply(root_quat_w, dots_b_flat) + root_pos_w[:, :3]

        # 可视化
        self.ray_visualizer.visualize(dots_w)


@configclass
class UniformPositionCommandCfg(CommandTermCfg):
    
    class_type = UniformPositionCommand
    
    asset_name: str = MISSING
    
    heading_command: bool = False

    heading_control_stiffness: float = 1.0
    
    rel_standing_envs: float = 0.0
    
    # rel_heading_envs: float = 1.0

    # 每个目标的最大时间（比如 5s 内要走到）
    max_goal_time_s: float = MISSING

    # 随机减去多少（例如 [max - 2, max]）
    randomize_goal_time_minus_s: float = MISSING

    @configclass
    class Ranges:
        # 你的 pos_1, pos_2, heading 等范围
        pos_1: tuple[float, float] = MISSING
        pos_2: tuple[float, float] = MISSING
        heading: tuple[float, float] | None = None
        use_polar: bool = False

    ranges: Ranges = MISSING

    limit_ranges: UniformPositionCommandCfg.Ranges = MISSING
