import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import gymnasium as gym
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .position_env_cfg import RobotEnvCfg


class LeggedRobotPosEnv(ManagerBasedRLEnv):
    """位置命令专用环境类，继承自ManagerBasedRLEnv，添加环境级计时器"""

    cfg: 'RobotEnvCfg'  # 类型注解

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        """初始化位置命令环境"""
        # 调用父类初始化
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def step(self, action: torch.Tensor):
        """单步仿真 + 终止 + 重置 + 命令更新。

        逻辑几乎完全照抄 ManagerBasedRLEnv.step，
        唯一的区别是最后一行：调用 self.command_manager.compute_position(...)。

        仿照ABS的逻辑写了一个新的step,这个step的逻辑和managerbasedrlenv中的不太一样
        """
        
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
        # 相当于post_physics_step_callback
        self.command_manager.compute(dt=self.step_dt)
        # step 间隔事件（例如外力、干扰等）
        # 推力干扰
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 计算终止
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # 计算奖励
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        #  重置需要 reset 的 env
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # reset 前记录
            self.recorder_manager.record_pre_reset(reset_env_ids)

            # 调用环境自己的 _reset_idx（这里会触发 command_manager.reset，
            # 从而重采样 position 命令）
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

        # return 
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )