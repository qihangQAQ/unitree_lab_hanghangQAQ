import torch
from typing import TYPE_CHECKING
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from .tracking_env_cfg import HandTrackingEnvCfg

class HandTrackingEnv(ManagerBasedRLEnv):
    """
    机械臂末端追踪专用环境类。
    继承自 ManagerBasedRLEnv，自定义了 step 逻辑以精确控制命令更新和奖励计算的时序。
    """

    cfg: 'HandTrackingEnvCfg'

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        """初始化环境，设置碰撞检测缓存等"""
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        
        # ==================== 1. 缓存碰撞检测的 Body Index ====================
        # 这一步是为了在计算 Reward 时快速索引，判断是否发生了非预期的碰撞
        # 假设你的 Scene 配置里有一个名为 "contact_forces" 的 ContactSensor
        if "contact_forces" in self.scene.sensors:
            contact_sensor: ContactSensor = self.scene.sensors["contact_forces"]
            all_body_names = contact_sensor.body_names
            
            # 对于路径追踪任务 (Tracking)，通常机械臂的任何部位都不应该碰到环境(地面/障碍物)
            # 如果你的任务允许末端 (End-Effector) 碰东西（比如绘画笔触），
            # 你可以在这里过滤掉末端 link。
            # 这里我写了一个通用的过滤逻辑：
            
            # 示例：假设我们要监控除了 "base" (基座) 以外的所有刚体碰撞
            # (因为固定基座可能会和地面有常驻接触，取决于建模方式)
            self.collision_body_indices = [
                i for i, name in enumerate(all_body_names) 
                if "base" not in name and "fixed" not in name
            ]
            
            # 转为 Tensor 提速
            self.collision_body_indices = torch.tensor(
                self.collision_body_indices, device=self.device, dtype=torch.long
            )
            
            print(f"[HandTrackingEnv] Monitoring collision for bodies: "
                  f"{[all_body_names[i] for i in self.collision_body_indices.tolist()]}")
        else:
            print("[HandTrackingEnv] Warning: No 'contact_forces' sensor found in scene.")
        # =================================================================

    def step(self, action: torch.Tensor):
        """
        单步仿真循环。
        逻辑顺序：Action处理 -> 物理仿真(Decimation) -> 命令更新 -> 终止判断 -> 奖励计算 -> 重置 -> 观测。
        """
        
        # 1) 处理动作 (Process Actions)
        # 将神经网络输出的 action 转换为通过 Actuator 处理的底层指令
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        # 2) 物理仿真循环 (Physics Stepping)
        # 这里的 cfg.decimation 决定了推理一次对应多少次物理步进
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            
            # 应用动作
            self.action_manager.apply_action()
            
            # 写入数据到 PhysX
            self.scene.write_data_to_sim()
            
            # 执行一步物理模拟
            self.sim.step(render=False)
            
            # 记录数据 (如果是录制模式)
            self.recorder_manager.record_post_physics_decimation_step()
            
            # 渲染 (如果开启 GUI 或相机传感器)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            
            # 更新 Scene 中的 Buffer (读取物理引擎返回的状态)
            self.scene.update(dt=self.physics_dt)

        # 3) 后处理 (Post-Processing)
        # ---------------------- post_physics_step ---------------------
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # ==================== 关键点：更新命令 ====================
        # 这里会调用 TrackingCommand._update_command() 
        # 以及 TrackingCommand._compute_command()
        # 从而更新 buffer 中的当前目标点，并计算 6D 误差
        self.command_manager.compute(dt=self.step_dt)
        
        # 处理定时事件 (Events)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # ==================== 计算终止条件 (Terminations) ====================
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # ==================== 计算奖励 (Rewards) ====================
        # 注意：Reward 计算依赖于 command_manager 刚刚更新过的误差
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # ==================== 处理重置 (Resets) ====================
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)

            # 调用 _reset_idx。
            # 这会触发 command_manager.reset -> TrackingCommand._resample_command
            # 为这些环境生成新的随机贝塞尔曲线
            self._reset_idx(reset_env_ids)

            # 重置后若需要重新渲染 (避免视觉闪烁)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        # ==================== 计算观测 (Observations) ====================
        # 如果有 Recorder 先算一次
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # 计算给 Policy 用的 Observation (update_history=True 处理时序堆叠)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )