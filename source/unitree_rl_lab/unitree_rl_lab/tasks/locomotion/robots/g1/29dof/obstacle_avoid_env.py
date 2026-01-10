import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import gymnasium as gym
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg  # 引入需要的配置类

if TYPE_CHECKING:
    from .position_env_cfg import RobotEnvCfg


class LeggedRobotPosNp3oEnv(ManagerBasedRLEnv):
    """位置命令专用环境类，继承自ManagerBasedRLEnv，添加环境级计时器"""

    cfg: 'RobotEnvCfg'  # 类型注解

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        """初始化位置命令环境"""
        # 调用父类初始化
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # ==================== 新增：缓存非脚部刚体的索引 ====================
        # 获取接触传感器实例
        contact_sensor = self.scene.sensors["contact_forces"]

        # 获取所有被该传感器监听的刚体名称列表
        all_body_names = contact_sensor.body_names

        # 筛选出不包含 "ankle" (或 foot/toe，取决于你的URDF命名) 的刚体索引
        self.non_foot_body_indices = [
            i for i, name in enumerate(all_body_names)
            if "ankle" not in name and "foot" not in name
        ]

        # 转为 Tensor 方便后续在 step 中进行切片索引
        self.non_foot_body_indices = torch.tensor(
            self.non_foot_body_indices, device=self.device, dtype=torch.long
        )
        print(f"[N-P3O] Collision check will monitor these bodies: "
              f"{[all_body_names[i] for i in self.non_foot_body_indices]}")
        # =================================================================

    def step(self, action: torch.Tensor):
        """单步仿真 + 终止 + 重置 + 命令更新 + Cost计算。"""

        # 1) 处理动作
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        # 2) 物理仿真（decimation 次）（共七步）
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            # 更新模拟步数计数器
            self._sim_step_counter += 1
            # 调用 action_manager 应用动作到缓冲区
            self.action_manager.apply_action()
            # 动作写入仿真
            self.scene.write_data_to_sim()
            # 调用sim执行一步物理模拟
            self.sim.step(render=False)
            # 调用recorder_manager记录物理步进后状态
            self.recorder_manager.record_post_physics_decimation_step()
            # 按 render_interval 渲染
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # 更新机器人状态信息
            self.scene.update(dt=self.physics_dt)

        # 3) 后处理：计数器 + 终止 + 奖励
        # ---------------------- post_physics_step ---------------------
        self.episode_length_buf += 1  # 每个 env 的 episode 步数
        self.common_step_counter += 1  # 全局步数

        # 更新命令
        self.command_manager.compute(dt=self.step_dt)
        # step 间隔事件（例如外力、干扰等）
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 计算终止
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # 计算奖励
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # ===================== 新增：计算约束成本 (Costs) =========================
        # 这里集中计算所有的 Cost，并汇总到一个 Tensor 中
        # 假设我们有两个约束：关节限位 (Joint Limits) 和 碰撞 (Collision)

        # 1. 计算各项具体的 Cost
        joint_limit_cost = self._compute_joint_pos_cost()
        collision_cost = self._compute_obstacle_collision_cost()

        # 2. 将它们存入 extras，供 Runner 读取
        # 注意：extras["costs"] 的 key 必须是固定的，Runner 会找这个 key
        # 我们将各项 Cost 叠加，或者分别存储。
        # N-P3O 通常处理标量 Cost Sum，也可以处理向量 Cost。
        # 这里简单起见，我们把它们加起来作为总 Cost 信号。
        # 如果你想分别看，可以存到 extras["log"] 里。

        # total_cost = joint_limit_cost + collision_cost
        costs = torch.stack([joint_limit_cost, collision_cost], dim=-1)

        # 存入 extras，Runner 的 process_env_step 会读取它
        self.extras["cost"] = costs

        # 可选：记录详细信息用于 WandB 展示
        self.extras["log"] = self.extras.get("log", {})
        self.extras["log"]["Cost/joint_limits"] = joint_limit_cost.mean()
        self.extras["log"]["Cost/collision"] = collision_cost.mean()
        # self.extras["log"]["Cost/total"] = total_cost.mean()
        self.extras["log"]["Cost/0_joint_limits_mean"] = costs[:, 0].mean()
        self.extras["log"]["Cost/1_collision_mean"] = costs[:, 1].mean()
        # =========================================================================

        #  重置需要 reset 的 env
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # reset 前记录
            self.recorder_manager.record_pre_reset(reset_env_ids)

            # 调用环境自己的 _reset_idx
            self._reset_idx(reset_env_ids)

            # # 重置后重新渲染
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # 调用recorder_manager记录重置后状态
            self.recorder_manager.record_post_reset(reset_env_ids)

        # 如果开了 recorder，需要先算一次 obs 给 recorder 用
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # 计算观测
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    # ===================== 新增：Cost 计算辅助函数 =========================
    def _compute_joint_pos_cost(self, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """
        计算关节限位 Cost。
        逻辑：Sum(ReLU(lower - q) + ReLU(q - upper))
        """
        asset: Articulation = self.scene[asset_cfg.name]

        # 计算超出下限的部分 (lower - q) > 0
        out_of_lower = (asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0] - asset.data.joint_pos[
            :, asset_cfg.joint_ids]).clip(min=0.0)

        # 计算超出上限的部分 (q - upper) > 0
        out_of_upper = (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[
            :, asset_cfg.joint_ids, 1]).clip(min=0.0)

        # 对所有关节求和
        return torch.sum(out_of_lower + out_of_upper, dim=1)

    def _compute_obstacle_collision_cost(self, threshold: float = 1.0) -> torch.Tensor:
        """
        计算障碍物碰撞 Cost。
        逻辑：只要非脚部刚体受力 > threshold，Cost = 1.0，否则 0.0。
        """
        contact_sensor: ContactSensor = self.scene.sensors["contact_forces"]

        # 获取历史受力: (num_envs, history_len, num_bodies, 3)
        net_contact_forces = contact_sensor.data.net_forces_w_history

        # 1. 取力的模长 -> (num_envs, history, num_bodies)
        forces_norm = torch.norm(net_contact_forces, dim=-1)

        # 2. 筛选非脚部刚体 -> (num_envs, history, num_non_foot_bodies)
        # 注意：这里用到了 __init__ 里缓存的 self.non_foot_body_indices
        non_foot_forces = forces_norm[:, :, self.non_foot_body_indices]

        # 3. 取历史最大值 & 刚体最大值 -> (num_envs,)
        # 只要任何一个非脚部部位在任何历史时刻受力超标，就视为碰撞
        max_force_val = torch.max(torch.max(non_foot_forces, dim=1)[0], dim=1)[0]

        # 4. 判定是否碰撞 (大于阈值则为 1.0)
        collision_detected = (max_force_val > threshold).float()

        return collision_detected
    # =====================================================================