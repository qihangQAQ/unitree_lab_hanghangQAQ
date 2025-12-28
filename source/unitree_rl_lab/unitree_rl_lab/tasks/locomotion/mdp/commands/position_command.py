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
    
    def _resample_command(self, env_ids: Sequence[int]):
        num = len(env_ids)
        if num == 0:
            return
        # === 1.随机位置生成 ====

        pos1 = torch.empty(num, 1, device=self.device).uniform_(*self.cfg.ranges.pos_1)
        pos2 = torch.empty(num, 1, device=self.device).uniform_(*self.cfg.ranges.pos_2)
        
        # 随机偏置位置生成
        if self.cfg.ranges.heading is not None:
            heading_offset = torch.empty(num, 1, device=self.device).uniform_(
                *self.cfg.ranges.heading
            )
        else:
            # 不用额外偏置就设为 0
            heading_offset = torch.zeros(num, 1, device=self.device)
        
        # === 2.计算目标位置 position_targets (世界系) ===

        # #这里写错了，这里用的是第一次的初始位置
        # 而不是每次reset之后，机器人的初始位置
        # env_origins = self.env_origins[env_ids]
        # 矫正
        env_origins = self.robot.data.root_pos_w[env_ids].clone()

        # 计算期望位置相对初始的偏移
        self.position_targets[env_ids, 0:1] = env_origins[:, 0:1] + pos1
        self.position_targets[env_ids, 1:2] = env_origins[:, 1:2] + pos2
        self.position_targets[env_ids, 2] = env_origins[:, 2] + 0.0

        # === 3) 计算 heading_target （世界系 yaw） ===

        root_pos_w = self.robot.data.root_pos_w[env_ids, :3]
        pos_diff = self.position_targets[env_ids] - root_pos_w
        
        # 指向目标点的“基础朝向”
        base_heading_to_target = torch.atan2(pos_diff[:, 1:2], pos_diff[:, 0:1])  # (num, 1)

        # 最终目标 heading = 指向目标的朝向 + 一个随机偏置
        heading_target = base_heading_to_target + heading_offset

        # wrap 到 [-pi, pi]
        self.heading_target[env_ids] = math_utils.wrap_to_pi(heading_target[:, 0])

        # === 2. 画球 ===
        self._debug_vis_callback()

        # === 3. 画圆锥箭头（期望朝向） ===
        self._debug_vis_heading_callback()

        # === 4) 设定位置命令 ===
        # 写到update_command中

        # # 世界系下的当前位置 / 姿态
        # root_pos_w = self.robot.data.root_pos_w[env_ids, :3]    # (n, 3)
        # root_quat_w = self.robot.data.root_quat_w[env_ids]      # (n, 4)
        # current_heading = self.robot.data.heading_w[env_ids]    # (n,)

        # # 位置误差（世界系）
        # pos_diff_w = self.position_targets[env_ids] - root_pos_w  # (n, 3)

        # # 转到 base frame
        # pos_diff_b = math_utils.quat_rotate_inverse(root_quat_w, pos_diff_w)  # (n, 3)

        # # 写入 x, y 误差
        # self.pos_command_b[env_ids, 0:2] = pos_diff_b[:, 0:2]

        # # heading 误差
        # heading_err = math_utils.wrap_to_pi(self.heading_target[env_ids] - current_heading)
        # self.pos_command_b[env_ids, 2] = heading_err

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
