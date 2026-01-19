import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation

# 复用你已有的 env
from .position_env import LeggedRobotPosEnv  # 按你项目实际路径改


class LeggedRobotPosNp3oEnv(LeggedRobotPosEnv):
    """N-P3O 版本位置命令环境：在 step 中额外计算 costs（关节限位、碰撞）。"""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # ==================== 新增：N-P3O cost 配置缓存 ====================
        # 新增：cost 的数量（两个约束：关节限位、碰撞）
        self.num_costs = 2

        # 新增：碰撞阈值（仿照 reward 里的 obstacle_collision threshold）
        # 这里先硬编码；后续你可以把它放进 cfg 里（例如 cfg.np3o.cost_collision_threshold）
        self.cost_collision_threshold = 1.0
        # =================================================================

    # ==================== 新增：计算关节限位 cost ====================
    # 新增：joint limit cost -- 仿照 mdp.joint_pos_limits 的 out_of_limits 计算
    def _compute_cost_joint_limits(self) -> torch.Tensor:
        """计算关节限位 cost：越超过 soft limit 越大，按关节维度求和。"""
        robot: Articulation = self.scene["robot"]

        # soft_joint_pos_limits: (num_envs, num_joints, 2)
        lower = robot.data.soft_joint_pos_limits[:, :, 0]
        upper = robot.data.soft_joint_pos_limits[:, :, 1]
        q = robot.data.joint_pos  # (num_envs, num_joints)

        out_of_limits = -(q - lower).clamp(max=0.0)  # q < lower 的超限量
        out_of_limits += (q - upper).clamp(min=0.0)  # q > upper 的超限量

        # shape: (num_envs,)
        return torch.sum(out_of_limits, dim=1)
    # ===========================================================

    # ==================== 新增：计算碰撞 cost ====================
    # 新增：collision cost -- 仿照 reward 的 obstacle_collision 逻辑（接触力阈值 + 非脚部 body）
    def _compute_cost_collision(self) -> torch.Tensor:
        """计算碰撞 cost：检测非脚部刚体的接触力是否超过阈值，靠近目标 tight 区域可放大。"""
        contact_sensor: ContactSensor = self.scene.sensors["contact_forces"]

        # net_forces_w_history: (num_envs, history, body, 3)
        forces_hist = contact_sensor.data.net_forces_w_history

        # 只取非脚部 body（你在父类 __init__ 里已经缓存了 non_foot_body_indices）
        forces_hist = forces_hist[:, :, self.non_foot_body_indices, :]

        # 每个 body 是否发生碰撞：历史维度取 max
        # norm -> (num_envs, history, bodies)
        force_norm = torch.norm(forces_hist, dim=-1)
        # max over history -> (num_envs, bodies)
        max_over_hist = torch.max(force_norm, dim=1)[0]
        collision_detected = max_over_hist > self.cost_collision_threshold

        # 统计发生碰撞的 body 数（和你 reward 注释里一致）
        cost = torch.sum(collision_detected, dim=1).float()  # (num_envs,)

        # -------- 仿照你 reward 注释里的 near_goal 放大逻辑 --------
        # 新增：靠近目标 tight 区域时，碰撞 cost 放大（防止最后冲刺撞障碍）
        cmd_term = self.command_manager.get_term("position")
        params = self.cfg.pos_reward_params

        robot: Articulation = self.scene["robot"]
        dist = torch.norm(cmd_term.position_targets[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
        near_goal = (dist < params.position_target_sigma_tight).float()

        # 放大倍率：1 + 4 * near_goal（与你 reward 注释一致）
        cost = cost * (1.0 + 4.0 * near_goal)
        # ------------------------------------------------------

        return cost
    # ===========================================================

    def step(self, action: torch.Tensor):
        # 直接复用你现有 step 的主体流程（父类已经把 reset/obs 都处理好了）
        obs, rew, terminated, time_outs, extras = super().step(action)

        # ==================== 新增：计算 N-P3O costs ====================
        # 新增：计算两个 cost（关节限位、碰撞）
        cost_limits = self._compute_cost_joint_limits()
        cost_collision = self._compute_cost_collision()

        # 新增：以 (num_envs, 2) 形式写入 extras，供 rollout_storage / algo 取用
        # 顺序约定：0=关节限位，1=碰撞
        costs = torch.stack([cost_limits, cost_collision], dim=-1)

        # extras 可能在父类里被用作 dict，这里做健壮处理
        if extras is None:
            extras = {}
        if not isinstance(extras, dict):
            # 尽量不破坏原逻辑；如果 extras 不是 dict，就强行替换成 dict
            extras = {}

        extras["costs"] = costs
        extras["cost_limits"] = cost_limits
        extras["cost_collision"] = cost_collision
        # ===========================================================

        return obs, rew, terminated, time_outs, extras
